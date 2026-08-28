#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Profile a PyPTO hand-written CCE extern with a real args-dump replay.

The regular incore profiler consumes PTOAS ``.cpp + .pto`` pairs. An
``@pl.jit.extern`` kernel has neither a PTO launch signature nor a sibling
``.pto``: its entry is ``kernel_entry(__gm__ int64_t *args)`` and the tensor
descriptors, scalar tail, SPMD context, and control tensors are built by the
runtime.

This tool bridges that gap:

1. load external-kernel metadata from ``kernel_config.py``;
2. select one dumped task and reconstruct its real positional ``args[]``;
3. restore level-2 tensor payloads, including data-dependent tiling metadata;
4. compile a compiler-owned mixed chevron shell and separate cube/vector bodies;
5. give each physical lane a copied ``args[]`` row with matching SPMD context;
6. let the chevron launch and op-simulator install the hard-MIX FFTS context.

The default one-block replay is an instruction/pipeline view of logical block
0. Use ``--replay-blocks`` equal to the production ``pl.spmd`` width for a
whole-grid replay when cross-block timing must be preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from incore_profile import (
    StepError,
    collect_artifacts,
    detect_degenerate_trace,
    discover_cann_set_env,
    find_export_src,
    private_dir,
    run_cmd,
    source_env,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (parent / ".git").exists()),
    SCRIPT_DIR.parent,
)
KARGS_SLOTS = 50
SUCCESS_TEXT = "Profiling running finished. All task success."
REPLAY_KERNEL_NAME = "replay_entry"

DTYPE_RAW = {
    "FLOAT32": 0,
    "FLOAT16": 1,
    "INT32": 2,
    "INT16": 3,
    "INT8": 4,
    "UINT8": 5,
    "BFLOAT16": 6,
    "INT64": 7,
    "UINT64": 8,
    "UINT16": 9,
    "UINT32": 10,
}
DTYPE_SIZE = {
    "FLOAT32": 4,
    "FLOAT16": 2,
    "INT32": 4,
    "INT16": 2,
    "INT8": 1,
    "UINT8": 1,
    "BFLOAT16": 2,
    "INT64": 8,
    "UINT64": 8,
    "UINT16": 2,
    "UINT32": 4,
}

TARGETS = {
    "a2a3": {
        "soc_version": "dav_2201",
        "aicore_arch": "dav-c220",
        "aic_arch": "dav-c220-cube",
        "aiv_arch": "dav-c220-vec",
        "cce_version": 220,
        "npu_arch": "PTO_NPU_ARCH_A2A3",
        "runtime_arch": "a2a3",
    },
    "a5": {
        "soc_version": "dav_3510",
        "aicore_arch": "dav-c310",
        "aic_arch": "dav-c310-cube",
        "aiv_arch": "dav-c310-vec",
        "cce_version": 310,
        "npu_arch": "PTO_NPU_ARCH_A5",
        "runtime_arch": "a5",
    },
}


@dataclass
class TensorReplay:
    slot: int
    role: str
    dtype: str
    shape: list[int]
    strides: list[int]
    start_offset: int
    buffer_bytes: int
    payload_file: Path | None


@dataclass
class TaskReplay:
    task_id: str
    func_ids: list[int]
    tensors: list[TensorReplay]
    scalars: list[tuple[int, int]]
    warnings: list[str]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_kernel_config(build_dir: Path) -> list[dict[str, Any]]:
    config_path = build_dir / "kernel_config.py"
    if not config_path.is_file():
        raise StepError(f"kernel_config.py not found: {config_path}")
    spec = importlib.util.spec_from_file_location("incore_extern_kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise StepError(f"cannot import kernel config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernels = list(getattr(module, "KERNELS", ()))
    if not kernels:
        raise StepError(f"KERNELS is empty in {config_path}")
    return kernels


def _load_manifest(dump_dir: Path) -> tuple[dict[str, Any], Path | None]:
    manifest_path = dump_dir / "args_dump.json"
    if not manifest_path.is_file():
        raise StepError(f"args dump manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    bin_name = data.get("bin_file")
    bin_path = dump_dir / bin_name if bin_name else None
    if bin_path is not None and not bin_path.is_file():
        raise StepError(f"args payload named by manifest is missing: {bin_path}")
    return data, bin_path


def _task_groups(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in manifest.get("args", ()):
        groups.setdefault(str(record["task_id"]), []).append(record)
    return groups


def _task_membership(records: list[dict[str, Any]]) -> list[int]:
    return [int(value) for value in (records[0].get("func_id") or ())]


def _resolve_requested_ids(
    kernels: list[dict[str, Any]],
    func_ids: str | None,
    funcs: list[str],
) -> set[int] | None:
    if func_ids and funcs:
        raise StepError("use either --func-id or --func, not both")
    if func_ids:
        try:
            return {int(value) for value in func_ids.split(",")}
        except ValueError as exc:
            raise StepError("--func-id must be a comma-separated integer list") from exc
    if not funcs:
        return None
    by_name = {str(kernel["name"]): int(kernel["func_id"]) for kernel in kernels}
    missing = [name for name in funcs if name not in by_name]
    if missing:
        raise StepError(f"unknown --func name(s): {missing}; available: {sorted(by_name)}")
    return {by_name[name] for name in funcs}


def _choose_task(
    manifest: dict[str, Any],
    kernels: list[dict[str, Any]],
    requested_ids: set[int] | None,
    requested_task_id: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    groups = _task_groups(manifest)
    if requested_task_id:
        if requested_task_id not in groups:
            raise StepError(f"--task-id {requested_task_id} not found; available: {sorted(groups)}")
        records = groups[requested_task_id]
        membership = set(_task_membership(records))
        if requested_ids is not None and membership != requested_ids:
            raise StepError(
                f"task {requested_task_id} has func ids {sorted(membership)}, "
                f"not requested {sorted(requested_ids)}"
            )
        return requested_task_id, records

    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for task_id, records in groups.items():
        membership = set(_task_membership(records))
        if requested_ids is None or membership == requested_ids:
            candidates.append((task_id, records))
    if not candidates:
        memberships = sorted({tuple(_task_membership(records)) for records in groups.values()})
        raise StepError(f"no dumped task matches requested funcs; dump memberships: {memberships}")
    if requested_ids is not None:
        return sorted(candidates, key=lambda item: item[0])[0]

    by_id = {int(kernel["func_id"]): kernel for kernel in kernels}

    def auto_score(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int, str]:
        task_id, records = item
        ids = set(_task_membership(records))
        members = [by_id[value] for value in ids if value in by_id]
        external = sum(bool(member.get("external")) for member in members)
        core_types = {member.get("core_type") for member in members}
        mixed = int({"aic", "aiv"} <= core_types)
        return mixed, external, task_id

    best = max(candidates, key=auto_score)
    best_ids = set(_task_membership(best[1]))
    best_members = [by_id[value] for value in best_ids if value in by_id]
    if not best_members or not all(member.get("external") for member in best_members):
        raise StepError("could not auto-select an all-external dumped task; pass --func or --func-id")
    return best


def _row_major(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return strides


def _extent_elem(shape: list[int], strides: list[int]) -> int:
    return 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides))


def _is_contiguous(shape: list[int], strides: list[int], start_offset: int) -> bool:
    return start_offset == 0 and strides == _row_major(shape)


def _read_payload(bin_path: Path, record: dict[str, Any]) -> bytes:
    size = int(record.get("bin_size", 0))
    with bin_path.open("rb") as stream:
        stream.seek(int(record["bin_offset"]))
        payload = stream.read(size)
    if len(payload) != size:
        raise StepError(
            f"short args payload for slot {record['arg_index']}: expected {size}, got {len(payload)}"
        )
    return payload


def _select_before_payload(
    bin_path: Path,
    slot: int,
    variants: list[dict[str, Any]],
) -> tuple[dict[str, Any], bytes, str | None]:
    before = [record for record in variants if record["stage"] == "before_dispatch"]
    if not before:
        raise StepError(f"slot {slot} has no before_dispatch payload")
    groups: dict[bytes, dict[str, Any]] = {}
    for index, record in enumerate(before):
        if record.get("truncated") or record.get("overwritten"):
            continue
        if int(record.get("bin_size", 0)) == 0:
            continue
        payload = _read_payload(bin_path, record)
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        group = groups.setdefault(
            digest,
            {
                "count": 0,
                "first": index,
                "nonzero": any(payload),
                "record": record,
                "payload": payload,
            },
        )
        group["count"] += 1
    if not groups:
        raise StepError(
            f"slot {slot} has no complete payload; rerun the dump with a smaller workload"
        )

    nonzero = [group for group in groups.values() if group["nonzero"]]
    candidates = nonzero or list(groups.values())
    selected = max(candidates, key=lambda group: (group["count"], -group["first"]))
    warning = None
    if len(groups) > 1:
        warning = (
            f"slot {slot} has {len(before)} repeated pre-dispatch snapshots with "
            f"{len(groups)} payload variants; selected the modal "
            f"{'non-zero ' if nonzero else ''}variant ({selected['count']} copies)"
        )
    return selected["record"], selected["payload"], warning


def _scatter_payload(
    payload: bytes,
    shape: list[int],
    strides: list[int],
    start_offset: int,
    element_bytes: int,
) -> bytes:
    numel = math.prod(shape)
    expected = numel * element_bytes
    if len(payload) != expected:
        raise StepError(f"logical payload size mismatch: expected {expected}, got {len(payload)}")
    footprint = start_offset + _extent_elem(shape, strides)
    if _is_contiguous(shape, strides, start_offset):
        return payload
    physical = bytearray(footprint * element_bytes)
    logical_strides = _row_major(shape)
    for linear in range(numel):
        remaining = linear
        physical_index = start_offset
        for dim, logical_stride, physical_stride in zip(shape, logical_strides, strides):
            coord = remaining // logical_stride
            remaining %= logical_stride
            if coord >= dim:
                raise AssertionError("logical coordinate overflow")
            physical_index += coord * physical_stride
        src = linear * element_bytes
        dst = physical_index * element_bytes
        physical[dst : dst + element_bytes] = payload[src : src + element_bytes]
    return bytes(physical)


def _prepare_task_replay(
    task_id: str,
    records: list[dict[str, Any]],
    bin_path: Path | None,
    inputs_dir: Path,
    zero_args: set[int],
) -> TaskReplay:
    func_ids = _task_membership(records)
    by_arg: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_arg.setdefault(int(record["arg_index"]), []).append(record)

    tensors: list[TensorReplay] = []
    scalars: list[tuple[int, int]] = []
    warnings: list[str] = []
    for slot, variants in sorted(by_arg.items()):
        scalar = next((record for record in variants if record.get("kind") == "scalar"), None)
        if scalar is not None:
            scalars.append((slot, int(scalar["value"])))
            continue

        before = next((record for record in variants if record["stage"] == "before_dispatch"), None)
        record = before or variants[0]
        dtype = str(record["dtype"]).upper()
        if dtype not in DTYPE_SIZE:
            raise StepError(f"unsupported dtype {dtype} in args slot {slot}")
        shape = [int(value) for value in record["shape"]]
        strides = [int(value) for value in (record.get("strides") or _row_major(shape))]
        start_offset = int(record.get("start_offset", 0))
        buffer_bytes = (start_offset + _extent_elem(shape, strides)) * DTYPE_SIZE[dtype]
        role = str(record["role"])
        payload_file: Path | None = None

        force_zero = slot in zero_args or role == "output"
        if force_zero:
            reason = "--zero-arg" if slot in zero_args else "output"
            warnings.append(f"slot {slot} ({role}) zero-initialized by {reason}")
        else:
            if bin_path is None:
                raise StepError(
                    f"slot {slot} has no payload; extern replay needs a level-2 args dump "
                    "(level 3 metadata-only is insufficient)"
                )
            record, logical, selection_warning = _select_before_payload(
                bin_path, slot, variants
            )
            if selection_warning:
                warnings.append(selection_warning)
            dtype = str(record["dtype"]).upper()
            shape = [int(value) for value in record["shape"]]
            strides = [int(value) for value in (record.get("strides") or _row_major(shape))]
            start_offset = int(record.get("start_offset", 0))
            physical = _scatter_payload(logical, shape, strides, start_offset, DTYPE_SIZE[dtype])
            payload_file = inputs_dir / f"arg{slot}.bin"
            payload_file.write_bytes(physical)
            buffer_bytes = len(physical)

        tensors.append(
            TensorReplay(
                slot=slot,
                role=role,
                dtype=dtype,
                shape=shape,
                strides=strides,
                start_offset=start_offset,
                buffer_bytes=buffer_bytes,
                payload_file=payload_file,
            )
        )
    return TaskReplay(
        task_id=task_id,
        func_ids=func_ids,
        tensors=tensors,
        scalars=scalars,
        warnings=warnings,
    )


def _common_kernel_label(members: list[dict[str, Any]]) -> str:
    names = [re.sub(r"_(?:aic|aiv)$", "", str(member["name"])) for member in members]
    return names[0] if len(set(names)) == 1 else "_".join(dict.fromkeys(names))


def _source_include(output_root: Path, source: Path) -> str:
    try:
        relative = source.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return os.path.relpath(source.resolve(), output_root.resolve())
    return os.path.relpath(REPO_ROOT / relative, output_root)


def _emit_replay_kernel(
    output_root: Path,
    members: list[dict[str, Any]],
    replay_blocks: int,
) -> str:
    aic = [member for member in members if member["core_type"] == "aic"]
    aiv = [member for member in members if member["core_type"] == "aiv"]
    if len(aic) > 1:
        raise StepError("extern replay supports at most one AIC member")
    if not aic and not aiv:
        raise StepError("selected task has no AIC/AIV members")
    sources = {Path(member["source"]).resolve() for member in members}
    if len(sources) != 1:
        raise StepError(
            "extern replay currently requires cooperative mixed members to share one source; "
            f"got: {sorted(str(source) for source in sources)}"
        )
    return f"""\
#include <cstdint>
#include <cce_aicore_intrinsics.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

static constexpr int32_t kReplayArgsSlots = {KARGS_SLOTS};

#if defined(__DAV_CUBE__)
extern "C" AICORE __attribute__((weak, noinline)) void
replay_arch_entry_aic(__gm__ int64_t *args) {{
    (void)args;
}}
#endif

#if defined(__DAV_VEC__)
extern "C" AICORE __attribute__((weak, noinline)) void
replay_arch_entry_aiv(__gm__ int64_t *args) {{
    (void)args;
}}
#endif

extern "C" __global__ AICORE void
replay_entry(__gm__ int64_t *args) {{
#if defined(__DAV_CUBE__)
    const int32_t row = static_cast<int32_t>(get_block_idx()) * 3;
    replay_arch_entry_aic(args + row * kReplayArgsSlots);
#elif defined(__DAV_VEC__)
    const int32_t row = static_cast<int32_t>(get_block_idx()) * 3
        + 1 + static_cast<int32_t>(get_subblockid());
    replay_arch_entry_aiv(args + row * kReplayArgsSlots);
#else
    (void)args;
#endif
}}
"""


def _emit_replay_launch(replay_blocks: int) -> str:
    return f"""\
#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void replay_entry(__gm__ int64_t *args);

extern "C" void launch_replay(void *args, void *stream) {{
    replay_entry<<<{replay_blocks}, nullptr, stream>>>((__gm__ int64_t *)args);
}}
"""


def _emit_replay_body(output_root: Path, members: list[dict[str, Any]]) -> str:
    source = Path(members[0]["source"]).resolve()
    include = _source_include(output_root, source)
    return f"""\
#if defined(REPLAY_BUILD_AIC) && defined(__DAV_CUBE__)
#define kernel_entry replay_arch_entry_aic
#include "{include}"
#undef kernel_entry
#endif

#if defined(REPLAY_BUILD_AIV) && defined(__DAV_VEC__)
#define kernel_entry replay_arch_entry_aiv
#include "{include}"
#undef kernel_entry
#endif
"""


def _emit_fatbin_injector() -> str:
    return """\
#!/usr/bin/env python3
\"\"\"Replace a CCE host object's embedded device ELF while keeping its stub.\"\"\"

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path


def section_offsets(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data, 0)
    section_header_offset = header[6]
    section_header_size = header[11]
    section_count = header[12]
    string_table_index = header[13]
    sections = [
        struct.unpack_from(
            "<IIQQQQIIQQ", data, section_header_offset + index * section_header_size
        )
        for index in range(section_count)
    ]
    string_section = sections[string_table_index]
    strings = data[string_section[4] : string_section[4] + string_section[5]]

    def read_name(offset: int) -> str:
        end = strings.find(b"\\0", offset)
        return strings[offset:end].decode()

    return {read_name(section[0]): section[4] for section in sections}


def main() -> int:
    host_object, executable_elf, relocatable_elf, output_object = map(Path, sys.argv[1:])
    objcopy = shutil.which("objcopy")
    if objcopy is None:
        raise RuntimeError("objcopy was not found")

    output_object.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(host_object, output_object)
    subprocess.run(
        [
            objcopy,
            "--update-section",
            f".aicore_binary={executable_elf}",
            "--update-section",
            f"__aicore_rel_binary={relocatable_elf}",
            output_object,
        ],
        check=True,
    )
    offsets = section_offsets(output_object)
    with output_object.open("r+b") as stream:
        for section, payload in (
            (".aicoreBinRec", executable_elf),
            ("__aicore_rel_rec", relocatable_elf),
        ):
            stream.seek(offsets[section] + 16)
            stream.write(struct.pack("<Q", payload.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _c_array(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def _emit_replay_host(task: TaskReplay, replay_blocks: int) -> str:
    allocations: list[str] = []
    descriptors: list[str] = []
    arg_rows: list[str] = []
    frees: list[str] = []
    for tensor_index, tensor in enumerate(task.tensors):
        slot = tensor.slot
        allocations.append(
            f"""\
    void *d_t{tensor_index} = nullptr;
    constexpr size_t t{tensor_index}Bytes = {tensor.buffer_bytes}ULL;
    ACL_CHECK(aclrtMalloc(&d_t{tensor_index}, t{tensor_index}Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
"""
        )
        if tensor.payload_file is None:
            allocations.append(
                f"    ACL_CHECK(aclrtMemset(d_t{tensor_index}, t{tensor_index}Bytes, 0, t{tensor_index}Bytes));\n"
            )
        else:
            allocations.append(
                f"""\
    std::vector<uint8_t> h_t{tensor_index}(t{tensor_index}Bytes);
    read_file("inputs/arg{slot}.bin", h_t{tensor_index});
    ACL_CHECK(aclrtMemcpy(
        d_t{tensor_index}, t{tensor_index}Bytes, h_t{tensor_index}.data(),
        t{tensor_index}Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
"""
            )
        descriptors.append(
            f"""\
    {{
        const uint32_t shape[] = {{{_c_array(tensor.shape)}}};
        const uint32_t strides[] = {{{_c_array(tensor.strides)}}};
        make_desc(
            h_tensors.data() + {tensor_index} * 128,
            reinterpret_cast<uint64_t>(d_t{tensor_index}), t{tensor_index}Bytes,
            {tensor.start_offset}ULL, shape, strides, {len(tensor.shape)},
            {DTYPE_RAW[tensor.dtype]}, {int(_is_contiguous(tensor.shape, tensor.strides, tensor.start_offset))});
    }}
"""
        )
        arg_rows.append(
            f"    h_common_args[{slot}] = static_cast<int64_t>("
            f"reinterpret_cast<uintptr_t>(d_tensors) + {tensor_index} * 128ULL);"
        )
        frees.append(f"    aclrtFree(d_t{tensor_index});")
    for slot, value in task.scalars:
        arg_rows.append(f"    h_common_args[{slot}] = static_cast<int64_t>({value}LL);")

    return f"""\
#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "intrinsic.h"

#define ACL_CHECK(expr)                                                        \\
    do {{                                                                       \\
        aclError error = (expr);                                               \\
        if (error != ACL_SUCCESS) {{                                           \\
            std::fprintf(stderr, "ACL error %d at %s:%d\\n", error, __FILE__, __LINE__); \\
            return 1;                                                          \\
        }}                                                                     \\
    }} while (0)

extern "C" void launch_replay(void *args, void *stream);

static void read_file(const char *path, std::vector<uint8_t> &data) {{
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()))) {{
        std::fprintf(stderr, "failed to read %s (%zu bytes)\\n", path, data.size());
        std::exit(2);
    }}
}}

// 128-byte Tensor descriptor offsets are pinned by tensor.h static assertions:
// buffer.addr@0, buffer.size@8, start_offset@24, ndims@36, dtype@40,
// is_contiguous@42, shapes@44, and strides@72.
static void make_desc(
    void *dst, uint64_t device_addr, uint64_t buffer_bytes, uint64_t start_offset,
    const uint32_t *shape, const uint32_t *strides, uint32_t ndims,
    uint8_t dtype, uint8_t is_contiguous) {{
    uint8_t bytes[128] = {{0}};
    *reinterpret_cast<uint64_t *>(bytes + 0) = device_addr;
    *reinterpret_cast<uint64_t *>(bytes + 8) = buffer_bytes;
    *reinterpret_cast<uint64_t *>(bytes + 24) = start_offset;
    *reinterpret_cast<uint32_t *>(bytes + 36) = ndims;
    bytes[40] = dtype;
    bytes[42] = is_contiguous;
    for (uint32_t index = 0; index < ndims; ++index) {{
        *reinterpret_cast<uint32_t *>(bytes + 44 + index * 4) = shape[index];
        *reinterpret_cast<uint32_t *>(bytes + 72 + index * 4) = strides[index];
    }}
    std::memcpy(dst, bytes, sizeof(bytes));
}}

int main() {{
    int32_t device_id = 0;
    if (const char *device = std::getenv("ACL_DEVICE_ID")) {{
        device_id = std::atoi(device);
    }}
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

{''.join(allocations)}
    std::vector<uint8_t> h_tensors({len(task.tensors)}ULL * 128, 0);
{''.join(descriptors)}
    void *d_tensors = nullptr;
    ACL_CHECK(aclrtMalloc(
        &d_tensors, h_tensors.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        d_tensors, h_tensors.size(), h_tensors.data(), h_tensors.size(),
        ACL_MEMCPY_HOST_TO_DEVICE));

    constexpr int32_t kBlocks = {replay_blocks};
    constexpr int32_t kRowsPerBlock = 3;
    constexpr int32_t kParticipants = kBlocks * kRowsPerBlock;
    std::vector<int64_t> h_common_args({KARGS_SLOTS}, 0);
{os.linesep.join(arg_rows)}
    std::vector<LocalContext> h_local(kParticipants);
    std::vector<GlobalContext> h_global(kParticipants);
    for (int32_t block = 0; block < kBlocks; ++block) {{
        for (int32_t lane = 0; lane < kRowsPerBlock; ++lane) {{
            const int32_t row = block * kRowsPerBlock + lane;
            h_local[row].block_idx = block;
            h_local[row].block_num = kBlocks;
            h_global[row].sub_block_id = lane == 2 ? 1 : 0;
        }}
    }}
    void *d_local = nullptr;
    void *d_global = nullptr;
    ACL_CHECK(aclrtMalloc(
        &d_local, h_local.size() * sizeof(LocalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(
        &d_global, h_global.size() * sizeof(GlobalContext), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        d_local, h_local.size() * sizeof(LocalContext), h_local.data(),
        h_local.size() * sizeof(LocalContext), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(
        d_global, h_global.size() * sizeof(GlobalContext), h_global.data(),
        h_global.size() * sizeof(GlobalContext), ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> h_args(
        static_cast<size_t>(kParticipants) * {KARGS_SLOTS}, 0);
    for (int32_t row = 0; row < kParticipants; ++row) {{
        int64_t *row_args = h_args.data() + static_cast<size_t>(row) * {KARGS_SLOTS};
        std::memcpy(
            row_args, h_common_args.data(), {KARGS_SLOTS} * sizeof(int64_t));
        row_args[SPMD_LOCAL_CONTEXT_INDEX] = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(d_local)
            + static_cast<size_t>(row) * sizeof(LocalContext));
        row_args[SPMD_GLOBAL_CONTEXT_INDEX] = static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(d_global)
            + static_cast<size_t>(row) * sizeof(GlobalContext));
    }}
    void *d_args = nullptr;
    ACL_CHECK(aclrtMalloc(
        &d_args, h_args.size() * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(
        d_args, h_args.size() * sizeof(int64_t), h_args.data(),
        h_args.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    std::printf(
        "[extern-replay] task={task.task_id} blocks={replay_blocks} "
        "args=%p arg0=0x%llx local=0x%llx global=0x%llx\\n",
        d_args,
        static_cast<unsigned long long>(h_args[0]),
        static_cast<unsigned long long>(h_args[SPMD_LOCAL_CONTEXT_INDEX]),
        static_cast<unsigned long long>(h_args[SPMD_GLOBAL_CONTEXT_INDEX]));
    launch_replay(d_args, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

{os.linesep.join(frees)}
    aclrtFree(d_tensors);
    aclrtFree(d_local);
    aclrtFree(d_global);
    aclrtFree(d_args);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}}
"""


def _cmake_path(path: str | Path, ascend_home: Path) -> str:
    resolved = Path(path).resolve()
    try:
        suffix = resolved.relative_to(ascend_home.resolve())
        return "${ASCEND_HOME_PATH}/" + suffix.as_posix()
    except ValueError:
        pass
    try:
        suffix = resolved.relative_to(REPO_ROOT)
        return "${PYPTO_LIB_ROOT}/" + suffix.as_posix()
    except ValueError:
        return resolved.as_posix()


def _emit_cmake(
    target: dict[str, Any],
    include_dirs: list[str],
    ascend_home: Path,
    debug_line: bool,
) -> str:
    external_includes = "\n    ".join(f"-I{_cmake_path(path, ascend_home)}" for path in include_dirs)
    link_options = "-Wl,-z,relro -Wl,-z,now" if debug_line else "-s -Wl,-z,relro -Wl,-z,now"
    debug_flag = "-g" if debug_line else ""
    device_strip_flag = "" if debug_line else "-x"
    runtime_arch = target["runtime_arch"]
    return f"""\
cmake_minimum_required(VERSION 3.16)
set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)
project(pypto_extern_cce_incore_replay)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set")
endif()
set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
set(PTO_ISA_ROOT $ENV{{PTO_ISA_ROOT}} CACHE PATH "PTO ISA root")
set(SIMPLER_ROOT $ENV{{SIMPLER_ROOT}} CACHE PATH "simpler runtime asset root")
set(PYPTO_LIB_ROOT $ENV{{PYPTO_LIB_ROOT}} CACHE PATH "pypto-lib root")
set(SOC_VERSION {target["soc_version"]} CACHE STRING "camodel SoC")

add_compile_options(
    -D_FORTIFY_SOURCE=2 -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong -fPIC)
add_link_options({link_options})

set(REPLAY_FAT_HOST_OPTIONS
    -xcce -fenable-matrix --cce-aicore-enable-tl -fPIC
    -Xhost-start -Xhost-end
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false)
set(REPLAY_DEVICE_OPTIONS
    -x cce -Wall
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false)
set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++ "SHELL:-include stdint.h" "SHELL:-include stddef.h")

set(REPLAY_INCLUDE_FLAGS
    -I${{PTO_ISA_ROOT}}/include
    -I${{PTO_ISA_ROOT}}/include/pto
    -I${{SIMPLER_ROOT}}/src/{runtime_arch}/runtime/tensormap_and_ringbuffer/runtime
    -I${{SIMPLER_ROOT}}/src/{runtime_arch}/runtime/tensormap_and_ringbuffer/common
    -I${{SIMPLER_ROOT}}/src/common/task_interface
    -I${{SIMPLER_ROOT}}/src/{runtime_arch}/platform/include
    -I${{SIMPLER_ROOT}}/simpler_setup/incore
    -I${{ASCEND_HOME_PATH}}/pkg_inc
    -I${{ASCEND_HOME_PATH}}/pkg_inc/profiling
    -I${{ASCEND_HOME_PATH}}/pkg_inc/runtime
    -I${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
    -I${{ASCEND_HOME_PATH}}/include
    {external_includes})

set(FAT_HOST_OBJ ${{CMAKE_CURRENT_BINARY_DIR}}/replay_fat_host.o)
set(REPLAY_LAUNCH_OBJ ${{CMAKE_CURRENT_BINARY_DIR}}/replay_launch.o)
set(WRAPPER_REL_ELF ${{CMAKE_CURRENT_BINARY_DIR}}/replay_wrapper_rel.elf)
set(AIC_BODY_OBJ ${{CMAKE_CURRENT_BINARY_DIR}}/replay_aic_body.o)
set(AIC_BODY_MIX_OBJ ${{CMAKE_CURRENT_BINARY_DIR}}/replay_aic_body_mix.o)
set(AIV_BODY_OBJ ${{CMAKE_CURRENT_BINARY_DIR}}/replay_aiv_body.o)
set(AIV_BODY_MIX_OBJ ${{CMAKE_CURRENT_BINARY_DIR}}/replay_aiv_body_mix.o)
set(MIX_REL_ELF ${{CMAKE_CURRENT_BINARY_DIR}}/replay_mix_rel.elf)
set(MIX_EXEC_ELF ${{CMAKE_CURRENT_BINARY_DIR}}/replay_mix_exec.elf)
set(PATCHED_HOST_OBJ ${{CMAKE_CURRENT_BINARY_DIR}}/replay_fat_host_patched.o)
set(BISHENG_LD ${{ASCEND_HOME_PATH}}/bin/ld.lld)
set(BISHENG_OBJCOPY ${{ASCEND_HOME_PATH}}/bin/llvm-objcopy)

add_custom_command(
    OUTPUT ${{FAT_HOST_OBJ}}
    COMMAND ${{CMAKE_CXX_COMPILER}} ${{REPLAY_FAT_HOST_OPTIONS}}
            --cce-aicore-arch={target["aicore_arch"]}
            -O3 -DREGISTER_BASE -std=c++17 {debug_flag}
            ${{REPLAY_INCLUDE_FLAGS}} -o ${{FAT_HOST_OBJ}} -c
            ${{CMAKE_CURRENT_SOURCE_DIR}}/replay_kernel.cpp
    DEPENDS replay_kernel.cpp
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_command(
    OUTPUT ${{REPLAY_LAUNCH_OBJ}}
    COMMAND ${{CMAKE_CXX_COMPILER}} ${{REPLAY_FAT_HOST_OPTIONS}}
            --cce-aicore-arch={target["aicore_arch"]}
            -O3 -DREGISTER_BASE -std=c++17 {debug_flag}
            ${{REPLAY_INCLUDE_FLAGS}} -o ${{REPLAY_LAUNCH_OBJ}} -c
            ${{CMAKE_CURRENT_SOURCE_DIR}}/replay_launch.cpp
    DEPENDS replay_launch.cpp
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_command(
    OUTPUT ${{WRAPPER_REL_ELF}}
    COMMAND ${{BISHENG_OBJCOPY}}
            --dump-section __aicore_rel_binary=${{WRAPPER_REL_ELF}}
            ${{FAT_HOST_OBJ}}
    DEPENDS ${{FAT_HOST_OBJ}}
    VERBATIM)
add_custom_command(
    OUTPUT ${{AIC_BODY_OBJ}}
    COMMAND ${{CMAKE_CXX_COMPILER}} ${{REPLAY_DEVICE_OPTIONS}}
            --cce-aicore-arch={target["aic_arch"]}
            --cce-aicore-only -O3 -DMEMORY_BASE
            -DREPLAY_BUILD_AIC -std=c++17 {debug_flag}
            ${{REPLAY_INCLUDE_FLAGS}} -o ${{AIC_BODY_OBJ}} -c
            ${{CMAKE_CURRENT_SOURCE_DIR}}/replay_body.cpp
    DEPENDS replay_body.cpp
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_command(
    OUTPUT ${{AIV_BODY_OBJ}}
    COMMAND ${{CMAKE_CXX_COMPILER}} ${{REPLAY_DEVICE_OPTIONS}}
            --cce-aicore-arch={target["aiv_arch"]}
            --cce-aicore-only -O3 -DMEMORY_BASE
            -DREPLAY_BUILD_AIV -std=c++17 {debug_flag}
            ${{REPLAY_INCLUDE_FLAGS}} -o ${{AIV_BODY_OBJ}} -c
            ${{CMAKE_CURRENT_SOURCE_DIR}}/replay_body.cpp
    DEPENDS replay_body.cpp
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_command(
    OUTPUT ${{AIC_BODY_MIX_OBJ}}
    COMMAND ${{BISHENG_OBJCOPY}}
            --redefine-sym replay_arch_entry_aic=replay_arch_entry_aic.cube
            ${{AIC_BODY_OBJ}} ${{AIC_BODY_MIX_OBJ}}
    DEPENDS ${{AIC_BODY_OBJ}}
    VERBATIM)
add_custom_command(
    OUTPUT ${{AIV_BODY_MIX_OBJ}}
    COMMAND ${{BISHENG_OBJCOPY}}
            --redefine-sym replay_arch_entry_aiv=replay_arch_entry_aiv.vector
            ${{AIV_BODY_OBJ}} ${{AIV_BODY_MIX_OBJ}}
    DEPENDS ${{AIV_BODY_OBJ}}
    VERBATIM)
add_custom_command(
    OUTPUT ${{MIX_REL_ELF}}
    COMMAND ${{BISHENG_LD}} -m aicorelinux -r -Ttext=0 -q
            {device_strip_flag}
            ${{WRAPPER_REL_ELF}} ${{AIC_BODY_MIX_OBJ}} ${{AIV_BODY_MIX_OBJ}}
            -static -o ${{MIX_REL_ELF}}
    DEPENDS ${{WRAPPER_REL_ELF}} ${{AIC_BODY_MIX_OBJ}} ${{AIV_BODY_MIX_OBJ}}
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_command(
    OUTPUT ${{MIX_EXEC_ELF}}
    COMMAND ${{BISHENG_LD}} -m aicorelinux -Ttext=0 -q
            ${{MIX_REL_ELF}} -static -o ${{MIX_EXEC_ELF}}
    DEPENDS ${{MIX_REL_ELF}}
    VERBATIM)
add_custom_command(
    OUTPUT ${{PATCHED_HOST_OBJ}}
    COMMAND ${{Python3_EXECUTABLE}}
            ${{CMAKE_CURRENT_SOURCE_DIR}}/inject_aicore_binary.py
            ${{FAT_HOST_OBJ}} ${{MIX_EXEC_ELF}} ${{MIX_REL_ELF}}
            ${{PATCHED_HOST_OBJ}}
    DEPENDS ${{FAT_HOST_OBJ}} ${{MIX_EXEC_ELF}} ${{MIX_REL_ELF}}
            inject_aicore_binary.py
    VERBATIM)
add_library(replay_kernel SHARED ${{PATCHED_HOST_OBJ}} ${{REPLAY_LAUNCH_OBJ}})
set_target_properties(replay_kernel PROPERTIES LINKER_LANGUAGE CXX)
set_source_files_properties(
    ${{PATCHED_HOST_OBJ}} ${{REPLAY_LAUNCH_OBJ}}
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_include_directories(replay_kernel PRIVATE
    ${{ASCEND_HOME_PATH}}/pkg_inc
    ${{ASCEND_HOME_PATH}}/pkg_inc/profiling
    ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime)
target_link_options(replay_kernel PRIVATE --cce-fatobj-link)

add_executable(replay_host replay_host.cpp)
target_compile_options(replay_host PRIVATE ${{CMAKE_CPP_COMPILE_OPTIONS}})
target_include_directories(replay_host PRIVATE
    ${{SIMPLER_ROOT}}/src/{runtime_arch}/runtime/tensormap_and_ringbuffer/runtime
    ${{SIMPLER_ROOT}}/src/{runtime_arch}/runtime/tensormap_and_ringbuffer/common
    ${{SIMPLER_ROOT}}/src/common/task_interface
    ${{ASCEND_HOME_PATH}}/include
    ${{ASCEND_HOME_PATH}}/pkg_inc
    ${{ASCEND_HOME_PATH}}/pkg_inc/profiling
    ${{ASCEND_HOME_PATH}}/pkg_inc/runtime
    ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime)
target_link_directories(replay_host PUBLIC
    ${{ASCEND_HOME_PATH}}/lib64
    ${{ASCEND_HOME_PATH}}/aarch64-linux/simulator/${{SOC_VERSION}}/lib)
target_link_libraries(replay_host PRIVATE
    replay_kernel runtime_camodel stdc++ ascendcl m tiling_api platform c_sec dl nnopbase)
"""


def _resolve_simpler_root() -> Path:
    from simpler_setup.environment import PROJECT_ROOT

    root = Path(PROJECT_ROOT)
    if not (root / "src/common/task_interface").is_dir():
        raise StepError(f"simpler runtime assets are incomplete: {root}")
    return root


def _resolve_pto_isa_root(value: str | None) -> Path:
    candidates = [
        Path(value).expanduser() if value else None,
        Path(os.environ["PTO_ISA_ROOT"]).expanduser() if os.environ.get("PTO_ISA_ROOT") else None,
        Path.home() / "pto-isa",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "include").is_dir():
            return candidate.resolve()
    raise StepError("PTO ISA headers not found; pass --pto-isa-root")


def _build_workspace(
    output_root: Path,
    task: TaskReplay,
    members: list[dict[str, Any]],
    target: dict[str, Any],
    ascend_home: Path,
    simpler_root: Path,
    pto_isa_root: Path,
    replay_blocks: int,
    debug_line: bool,
    env: dict[str, str],
) -> tuple[Path, dict[str, str]]:
    include_dirs = sorted(
        {
            str(path)
            for member in members
            for path in (member.get("extra_include_dirs") or ())
        }
    )
    (output_root / "replay_kernel.cpp").write_text(
        _emit_replay_kernel(output_root, members, replay_blocks), encoding="utf-8"
    )
    (output_root / "replay_launch.cpp").write_text(
        _emit_replay_launch(replay_blocks), encoding="utf-8"
    )
    (output_root / "replay_body.cpp").write_text(
        _emit_replay_body(output_root, members), encoding="utf-8"
    )
    (output_root / "inject_aicore_binary.py").write_text(
        _emit_fatbin_injector(), encoding="utf-8"
    )
    (output_root / "replay_host.cpp").write_text(
        _emit_replay_host(task, replay_blocks), encoding="utf-8"
    )
    (output_root / "CMakeLists.txt").write_text(
        _emit_cmake(target, include_dirs, ascend_home, debug_line), encoding="utf-8"
    )

    build_dir = output_root / "build"
    private_dir(build_dir)
    build_env = env.copy()
    build_env.update(
        {
            "PTO_ISA_ROOT": str(pto_isa_root),
            "SIMPLER_ROOT": str(simpler_root),
            "PYPTO_LIB_ROOT": str(REPO_ROOT),
        }
    )
    run_cmd(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(output_root),
            "-B",
            str(build_dir),
            f"-DSOC_VERSION={target['soc_version']}",
        ],
        cwd=output_root,
        env=build_env,
        log_path=output_root / "cmake.log",
        timeout=300,
    )
    run_cmd(
        ["cmake", "--build", str(build_dir), "--target", "replay_host"],
        cwd=output_root,
        env=build_env,
        log_path=output_root / "build.log",
        timeout=900,
    )
    app = build_dir / "replay_host"
    kernel_library = build_dir / "libreplay_kernel.so"
    if not app.is_file() or not kernel_library.is_file():
        raise StepError("replay build succeeded without replay_host/libreplay_kernel.so")
    symbols = subprocess.run(
        ["nm", "-D", str(kernel_library)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    if " launch_replay" not in symbols:
        raise StepError("replay kernel library is missing the chevron launch stub")
    return app, build_env


def _make_sim_env(
    env: dict[str, str],
    build_dir: Path,
    ascend_home: Path,
    soc_version: str,
    device: str,
) -> dict[str, str]:
    sim_env = env.copy()
    paths = [
        build_dir,
        ascend_home / "aarch64-linux" / "simulator" / soc_version / "lib",
        ascend_home / "lib64",
        ascend_home / "aarch64-linux" / "devlib",
        ascend_home / "devlib",
    ]
    old = sim_env.get("LD_LIBRARY_PATH")
    if old:
        paths.append(Path(old))
    sim_env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in paths)
    sim_env["ACL_DEVICE_ID"] = device
    return sim_env


def _collect(
    output_root: Path,
    app: Path,
    env: dict[str, str],
    target: dict[str, Any],
    timeout: int,
) -> Path:
    collect_root = output_root / "collect"
    private_dir(collect_root)
    collect_out = collect_root / "out"
    command = [
        "msprof",
        "op",
        "simulator",
        f"--application={app}",
        f"--kernel-name={REPLAY_KERNEL_NAME}",
        "--launch-count=1",
        f"--soc-version={target['soc_version']}",
        f"--timeout={timeout}",
        f"--output={collect_out}",
    ]
    completed = run_cmd(
        command,
        cwd=output_root,
        env=env,
        log_path=collect_root / "collect.log",
        timeout=timeout + 120,
        check=False,
    )
    if completed.returncode != 0 or SUCCESS_TEXT not in completed.stdout:
        tail = completed.stdout[-1200:].replace("\n", " ")
        raise StepError(f"msprof collect failed rc={completed.returncode}: {tail}")

    artifacts = collect_artifacts(collect_out)
    if artifacts["trace_json"] and artifacts["visualize_data_bin"]:
        return Path(str(artifacts["visualize_data_bin"])).parent
    export_src = find_export_src(collect_out)
    if export_src is None:
        raise StepError("msprof produced neither final trace nor dump/tmp_dump export source")
    export_out = output_root / "export"
    private_dir(export_out)
    run_cmd(
        ["msprof", "op", "simulator", f"--export={export_src}", f"--output={export_out}"],
        cwd=output_root,
        env=env,
        log_path=export_out / "export.log",
        timeout=timeout + 120,
    )
    artifacts = collect_artifacts(export_out)
    if not artifacts["trace_json"] or not artifacts["visualize_data_bin"]:
        raise StepError("msprof export completed without trace.json/visualize_data.bin")
    return Path(str(artifacts["visualize_data_bin"])).parent


def _clean_trace(
    raw_root: Path,
    output_root: Path,
    kernel_label: str,
    env: dict[str, str],
) -> tuple[Path, Path]:
    visualize_data = raw_root / "visualize_data.bin"
    if not visualize_data.is_file():
        raise StepError(f"visualize_data.bin was not generated under {raw_root}")
    run_cmd(
        [
            sys.executable,
            "-m",
            "pypto.tools.clean_sim_trace",
            str(visualize_data),
            "-o",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        log_path=output_root / "clean_trace.log",
        timeout=300,
    )
    trace = output_root / "trace.clean.json"
    named_trace = output_root / f"{kernel_label}.clean.json"
    if not trace.is_file():
        raise StepError(f"clean trace was not generated: {trace}")
    trace.rename(named_trace)
    metrics = output_root / "instr_metrics.json"
    if not metrics.is_file():
        raise StepError(f"instruction metrics were not generated: {metrics}")
    return named_trace, metrics


def _write_summary(
    output_root: Path,
    build_dir: Path,
    dump_dir: Path,
    task: TaskReplay,
    members: list[dict[str, Any]],
    replay_blocks: int,
    zero_args: set[int],
    raw_root: Path | None,
    warning: str | None,
    metrics_path: Path | None,
) -> None:
    lines = [
        "PyPTO external CCE in-core replay",
        f"build_dir={build_dir}",
        f"args_dump={dump_dir}",
        f"task_id={task.task_id}",
        f"func_ids={task.func_ids}",
        f"members={[member['name'] for member in members]}",
        f"replay_blocks={replay_blocks}",
        f"zero_args={sorted(zero_args)}",
        f"raw_trace_root={raw_root or 'not collected'}",
        f"trace_warning={warning or 'none'}",
        "",
        "Argument workload:",
    ]
    for tensor in task.tensors:
        init = str(tensor.payload_file.relative_to(output_root)) if tensor.payload_file else "zero"
        lines.append(
            f"slot {tensor.slot}: tensor role={tensor.role} dtype={tensor.dtype} "
            f"shape={tensor.shape} strides={tensor.strides} start_offset={tensor.start_offset} "
            f"buffer_bytes={tensor.buffer_bytes} init={init}"
        )
    for slot, value in task.scalars:
        lines.append(f"slot {slot}: scalar value={value}")
    if task.warnings:
        lines += ["", "Replay initialization notes:", *task.warnings]
    if metrics_path is not None and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        lines += ["", "Profiled pipe cycles:"]
        for core, instructions in metrics.get("instructions", {}).items():
            pipe_cycles: dict[str, int] = {}
            for instruction in instructions:
                pipe = str(instruction["pipe"])
                pipe_cycles[pipe] = pipe_cycles.get(pipe, 0) + int(instruction["cycles"])
            rendered = ", ".join(
                f"{pipe}={cycles}" for pipe, cycles in sorted(pipe_cycles.items())
            )
            lines.append(f"{core}: {rendered}")
    lines += [
        "",
        "Fidelity notes:",
        "- Runtime args[] tensor descriptors and level-2 before-dispatch payloads are replayed.",
        "- Each physical AIC/AIV lane receives a copied args[] row with matching SPMD context.",
        "- A compiler-generated mixed chevron shell calls the separately compiled "
        "production cube/vector bodies.",
        "- The op-simulator owns the FFTS control region used by mixed-core barriers.",
    ]
    if replay_blocks == 1:
        lines += [
            "- One-block mode profiles logical block 0 only.",
            "- Cross-block barriers shrink to one cluster; global combine work may be redistributed "
            "onto its two AIV lanes. Use the production block count for whole-grid timing.",
        ]
    else:
        lines.append("- Multi-block mode preserves the selected physical replay grid.")
    (output_root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a hand-written PyPTO CCE extern from a level-2 args dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--build-dir", required=True, help="PyPTO build_output directory")
    parser.add_argument(
        "--args-dump",
        help="args_dump directory; defaults to <build-dir>/dfx_outputs/args_dump",
    )
    parser.add_argument("--func-id", help="selected task func-id set, e.g. 1,2")
    parser.add_argument(
        "--func",
        action="append",
        default=[],
        help="selected kernel_config name; repeat for mixed members",
    )
    parser.add_argument("--task-id", help="specific dumped task id")
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="list dumped task memberships and exit",
    )
    parser.add_argument("--target", choices=sorted(TARGETS), default="a2a3")
    parser.add_argument(
        "--replay-blocks",
        type=int,
        default=1,
        help="physical block count; 1 profiles representative block 0",
    )
    parser.add_argument(
        "--zero-arg",
        type=int,
        action="append",
        default=[],
        help="zero-initialize a tensor slot instead of replaying payload; repeatable",
    )
    parser.add_argument("--output-root", help="deliverable directory under build_output by default")
    parser.add_argument("--name", help="output folder name override")
    parser.add_argument("--cann-set-env", help="CANN set_env.sh")
    parser.add_argument("--pto-isa-root", help="PTO ISA checkout")
    parser.add_argument("--device", default=None, help="ACL device for simulator init")
    parser.add_argument("--debug-line", action="store_true")
    parser.add_argument("--no-collect", action="store_true", help="stop after replay smoke build")
    parser.add_argument("--msprof-timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.replay_blocks <= 0:
        parser.error("--replay-blocks must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    build_dir = Path(args.build_dir).expanduser().resolve()
    if not build_dir.is_dir():
        raise StepError(f"build dir not found: {build_dir}")
    dump_dir = (
        Path(args.args_dump).expanduser().resolve()
        if args.args_dump
        else build_dir / "dfx_outputs" / "args_dump"
    )
    kernels = _load_kernel_config(build_dir)
    manifest, bin_path = _load_manifest(dump_dir)

    if args.list_tasks:
        by_id = {int(kernel["func_id"]): kernel for kernel in kernels}
        for task_id, records in sorted(_task_groups(manifest).items()):
            ids = _task_membership(records)
            names = [by_id.get(func_id, {}).get("name", "?") for func_id in ids]
            print(f"{task_id}\tfunc_ids={ids}\tnames={names}")
        return 0

    requested_ids = _resolve_requested_ids(kernels, args.func_id, args.func)
    task_id, records = _choose_task(manifest, kernels, requested_ids, args.task_id)
    func_ids = _task_membership(records)
    by_id = {int(kernel["func_id"]): kernel for kernel in kernels}
    missing = [func_id for func_id in set(func_ids) if func_id not in by_id]
    if missing:
        raise StepError(f"dumped func ids missing from kernel_config.py: {missing}")
    members = [by_id[func_id] for func_id in func_ids]
    if not all(member.get("external") for member in members):
        raise StepError("selected task includes a non-external kernel; use incore_profile.py for PTOAS kernels")

    kernel_label = _common_kernel_label(members)
    default_name = f"incore_{kernel_label}_{build_dir.name.removeprefix('_jit_')}_{_timestamp()}"
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = REPO_ROOT / "build_output" / (args.name or default_name)
    if output_root.exists():
        raise StepError(f"output root already exists: {output_root}")
    private_dir(output_root)
    inputs_dir = output_root / "inputs"
    private_dir(inputs_dir)

    zero_args = set(args.zero_arg)
    task = _prepare_task_replay(task_id, records, bin_path, inputs_dir, zero_args)
    target = TARGETS[args.target]
    cann_set_env = (
        Path(args.cann_set_env).expanduser().resolve()
        if args.cann_set_env
        else discover_cann_set_env()
    )
    if cann_set_env is None or not cann_set_env.is_file():
        raise StepError("CANN set_env.sh not found; pass --cann-set-env")
    env = source_env(cann_set_env, os.environ.copy())
    ascend_home = Path(env.get("ASCEND_HOME_PATH", "")).resolve()
    if not ascend_home.is_dir():
        raise StepError("ASCEND_HOME_PATH is invalid after sourcing CANN")
    if not shutil.which("msprof", path=env.get("PATH")):
        raise StepError("msprof not found after sourcing CANN")
    simpler_root = _resolve_simpler_root()
    pto_isa_root = _resolve_pto_isa_root(args.pto_isa_root)
    device = str(args.device or os.environ.get("TASK_DEVICE") or os.environ.get("ACL_DEVICE_ID") or "0")

    app, build_env = _build_workspace(
        output_root,
        task,
        members,
        target,
        ascend_home,
        simpler_root,
        pto_isa_root,
        args.replay_blocks,
        args.debug_line,
        env,
    )
    print(f"[extern-profile] replay build OK: {app}")
    if args.no_collect:
        _write_summary(
            output_root,
            build_dir,
            dump_dir,
            task,
            members,
            args.replay_blocks,
            zero_args,
            None,
            None,
            None,
        )
        print(f"[extern-profile] --no-collect: {output_root}")
        return 0

    sim_env = _make_sim_env(
        build_env,
        app.parent,
        ascend_home,
        str(target["soc_version"]),
        device,
    )
    raw_root = _collect(output_root, app, sim_env, target, args.msprof_timeout)
    warning = detect_degenerate_trace(raw_root)
    trace, metrics = _clean_trace(raw_root, output_root, kernel_label, sim_env)
    _write_summary(
        output_root,
        build_dir,
        dump_dir,
        task,
        members,
        args.replay_blocks,
        zero_args,
        raw_root,
        warning,
        metrics,
    )
    print(f"[extern-profile] trace: {trace}")
    print(f"[extern-profile] metrics: {metrics}")
    if warning:
        print(f"[extern-profile] WARNING: {warning}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StepError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
