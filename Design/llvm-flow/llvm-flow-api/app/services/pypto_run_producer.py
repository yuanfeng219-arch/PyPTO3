"""Immutable Review Run manifests and asynchronous producer state.

The only enabled producer imports the canonical, already-generated PyPTO
artifact set. It never invokes arbitrary commands and never accepts a caller
supplied filesystem path. A future controlled Linux/Ascend runner can adopt
the same request, state, and manifest contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.services.pypto_review import collect_review_run

ProducerKind = Literal["auto", "artifact-import", "pypto-jit"]

_RUN_ID = re.compile(r"^pcr_[0-9a-f]{24}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_root(storage_root: Path | None = None) -> Path:
    if storage_root is not None:
        return storage_root.resolve()
    return (
        Path(__file__).resolve().parents[2]
        / "media/pypto-control-flow-review-runs"
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode()


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(_json_bytes(payload))


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _run_directory(run_id: str, storage_root: Path | None = None) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid Review Run id")
    return _storage_root(storage_root) / run_id


def review_run_capabilities() -> dict[str, Any]:
    """Describe producer availability without mutating toolchains."""
    return {
        "schemaVersion": "pypto.control-flow-producer-capabilities.v1",
        "producers": [
            {
                "kind": "artifact-import",
                "available": True,
                "compilerInvoked": False,
                "description": (
                    "Validates the canonical existing artifact set and writes "
                    "an immutable evidence manifest."
                ),
            },
            {
                "kind": "pypto-jit",
                "available": False,
                "compilerInvoked": True,
                "reasonCode": "controlled-runner-not-configured",
                "description": (
                    "Requires a controlled Linux/Ascend runner with PyPTO, "
                    "compile_debug_mode=1, and an isolated output directory."
                ),
            },
        ],
    }


def start_review_run(
    input_t: int,
    producer: ProducerKind = "auto",
    storage_root: Path | None = None,
) -> dict[str, Any]:
    """Persist an immutable request and a queued lifecycle state."""
    run_id = f"pcr_{uuid.uuid4().hex[:24]}"
    run_dir = _run_directory(run_id, storage_root)
    submitted_at = _now()
    request = {
        "schemaVersion": "pypto.control-flow-review-request.v1",
        "id": run_id,
        "submittedAt": submitted_at,
        "requestedScenario": {"inputT": input_t},
        "requestedProducer": producer,
    }
    state = {
        "schemaVersion": "pypto.control-flow-review-state.v1",
        "id": run_id,
        "status": "queued",
        "producer": producer,
        "requestedScenario": {"inputT": input_t},
        "submittedAt": submitted_at,
        "updatedAt": submitted_at,
        "message": "Review Run queued for evidence production.",
    }
    _write_immutable(run_dir / "request.json", request)
    _write_state(run_dir / "state.json", state)
    return state


def get_review_run(
    run_id: str, storage_root: Path | None = None
) -> dict[str, Any]:
    state_path = _run_directory(run_id, storage_root) / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Review Run not found: {run_id}")
    return json.loads(state_path.read_text())


def _inventory(review: dict[str, Any]) -> dict[str, Any]:
    artifact_root = Path(review["sourceSnapshot"]["path"]).resolve().parent
    materialization = review["frontendMaterialization"]
    paths = [Path(review["sourceSnapshot"]["path"]).resolve()]
    paths.extend(
        [
            artifact_root
            / "Pass_00_LoopUnroll"
            / "Before_000_LoopUnroll_PROGRAM_ENTRY.json",
            artifact_root
            / "Pass_00_LoopUnroll"
            / "After_000_LoopUnroll_PROGRAM_ENTRY.json",
        ]
    )
    materialization_root = Path(
        materialization["artifactDirectory"]
    ).resolve()
    paths.extend(
        materialization_root / item["artifact"]
        for item in materialization["functions"]
    )
    files = []
    for path in paths:
        if not path.is_relative_to(artifact_root):
            raise ValueError("Artifact inventory escaped the canonical root")
        files.append(
            {
                "path": str(path.relative_to(artifact_root)),
                "sha256": _sha256(path),
                "sizeBytes": path.stat().st_size,
            }
        )
    files.sort(key=lambda item: item["path"])
    digest = hashlib.sha256(_json_bytes({"files": files})).hexdigest()
    return {
        "root": str(artifact_root),
        "bindingMode": "read-only-reference",
        "scope": "control-flow-minimum-evidence-set",
        "fileCount": len(files),
        "digest": digest,
        "files": files,
    }


def _manifest(
    run_id: str, request: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    completed_at = _now()
    return {
        "schemaVersion": "pypto.control-flow-review-manifest.v1",
        "id": run_id,
        "status": "complete",
        "submittedAt": request["submittedAt"],
        "completedAt": completed_at,
        "producer": {
            "kind": "artifact-import",
            "compilerInvoked": False,
            "evidenceOrigin": "pre-existing-canonical-artifacts",
        },
        "source": {
            "path": review["sourceSnapshot"]["path"],
            "sha256": review["sourceSnapshot"]["sha256"],
            "verified": review["sourceSnapshot"]["verified"],
        },
        "inputShape": {
            "t": request["requestedScenario"]["inputT"],
            "evidenceStatus": review["requestedScenario"]["evidenceStatus"],
        },
        "compileOptions": {
            "status": "partial",
            "observedEvidence": {
                "debug_options": {"compile_debug_mode": 1}
            },
            "note": (
                "Per-pass dumps prove debug capture was enabled; the full "
                "original JIT option set was not recorded."
            ),
        },
        "revisions": {
            "compiler": {"value": None, "status": "unverified"},
            "framework": {"value": None, "status": "unverified"},
        },
        "artifacts": _inventory(review),
        "reviewResult": review,
    }


def produce_review_run(
    run_id: str, storage_root: Path | None = None
) -> dict[str, Any]:
    """Execute the selected producer and persist terminal state."""
    run_dir = _run_directory(run_id, storage_root)
    request_path = run_dir / "request.json"
    if not request_path.is_file():
        raise FileNotFoundError(f"Review Run request not found: {run_id}")
    request = json.loads(request_path.read_text())
    input_t = request["requestedScenario"]["inputT"]
    requested_producer = request["requestedProducer"]
    selected_producer = requested_producer

    running = get_review_run(run_id, storage_root)
    running.update(
        {
            "status": "running",
            "producer": selected_producer,
            "updatedAt": _now(),
            "message": f"Producer {selected_producer} is running.",
        }
    )
    _write_state(run_dir / "state.json", running)

    try:
        review = collect_review_run(input_t)
        if requested_producer == "auto":
            selected_producer = (
                "artifact-import"
                if review["status"] == "complete"
                else "pypto-jit"
            )
            running.update(
                {
                    "producer": selected_producer,
                    "updatedAt": _now(),
                    "message": f"Producer {selected_producer} is running.",
                }
            )
            _write_state(run_dir / "state.json", running)
        if selected_producer == "pypto-jit":
            blocked = {
                **running,
                "status": "blocked",
                "updatedAt": _now(),
                "reasonCode": "controlled-runner-not-configured",
                "message": (
                    "No controlled Linux/Ascend PyPTO JIT runner is "
                    "configured; the request remains Draft."
                ),
                "result": review,
            }
            _write_state(run_dir / "state.json", blocked)
            return blocked
        if selected_producer != "artifact-import":
            raise ValueError(f"Unsupported producer: {selected_producer}")
        if review["status"] != "complete":
            blocked = {
                **running,
                "status": "blocked",
                "updatedAt": _now(),
                "reasonCode": "artifact-scenario-mismatch",
                "message": (
                    "The canonical artifacts do not belong to the requested "
                    "shape; the request remains Draft."
                ),
                "result": review,
            }
            _write_state(run_dir / "state.json", blocked)
            return blocked

        manifest = _manifest(run_id, request, review)
        manifest_path = run_dir / "manifest.json"
        _write_immutable(manifest_path, manifest)
        manifest_hash = _sha256(manifest_path)
        complete = {
            **running,
            "status": "complete",
            "updatedAt": manifest["completedAt"],
            "message": (
                "Canonical artifacts verified and immutable manifest written."
            ),
            "result": review,
            "manifest": {
                "schemaVersion": manifest["schemaVersion"],
                "path": str(manifest_path),
                "sha256": manifest_hash,
                "artifactDigest": manifest["artifacts"]["digest"],
                "artifactFileCount": manifest["artifacts"]["fileCount"],
                "compilerRevisionStatus": "unverified",
                "frameworkRevisionStatus": "unverified",
            },
        }
        _write_state(run_dir / "state.json", complete)
        return complete
    except Exception as error:
        failed = {
            **running,
            "status": "failed",
            "updatedAt": _now(),
            "reasonCode": "producer-failed",
            "message": str(error),
        }
        _write_state(run_dir / "state.json", failed)
        return failed
