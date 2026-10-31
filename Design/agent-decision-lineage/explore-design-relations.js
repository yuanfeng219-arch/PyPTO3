/* Audited cross-stage design evidence; not execution edges or measurements. */
window.EXPLORE_DESIGN_RELATIONS = [
  {
    "id": "kb-vf",
    "sourceId": "d-kb",
    "targetId": "d-vf",
    "relationType": "informs",
    "reason": "逐行 reduce→broadcast 模式支持 Vector/VF；官方 Softmax VF 样例提供独立实现参考。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§4.2 / §5.1"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§1 Vector 选择"
      },
      {
        "sourceId": "KB_SELECTION.json",
        "location": "optional_patterns: patterns/vec-row-reduce-broadcast.md"
      },
      {
        "sourceId": "DESIGN_BINDINGS.json",
        "location": "patterns/vec-row-reduce-broadcast.md / R01 / R03"
      },
      {
        "sourceId": "KB_USAGE.json",
        "location": "invariants[reference=patterns/vec-row-reduce-broadcast.md]"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": {
      "reference": "patterns/vec-row-reduce-broadcast.md",
      "requirementIds": [
        "R01",
        "R03"
      ]
    }
  },
  {
    "id": "kb-tile",
    "sourceId": "d-kb",
    "targetId": "d-tiling",
    "relationType": "informs",
    "reason": "行归约模式以完整行为单位；VF 不写 DN 归约 Tile，因此 tile-op 的 TR%8 规则不约束 TILE_ROWS=3。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§4.2 / §5.1"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§0 归约轴容量结论 / §2"
      },
      {
        "sourceId": "KB_SELECTION.json",
        "location": "optional_patterns: patterns/vec-row-reduce-broadcast.md"
      },
      {
        "sourceId": "DESIGN_BINDINGS.json",
        "location": "patterns/vec-row-reduce-broadcast.md / R01 / R03"
      },
      {
        "sourceId": "KB_USAGE.json",
        "location": "invariants[reference=patterns/vec-row-reduce-broadcast.md]"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": {
      "reference": "patterns/vec-row-reduce-broadcast.md",
      "requirementIds": [
        "R01",
        "R03"
      ]
    }
  },
  {
    "id": "kb-stream",
    "sourceId": "d-kb",
    "targetId": "d-tiling",
    "relationType": "evaluated_not_triggered",
    "reason": "跨 Tile streaming recurrence 已评估；四槽 192KB≤248KB，整行 N≤8192 可容纳，因此未触发。不是编译失败或性能淘汰。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§4.2 / §5.1"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§0 归约轴容量结论"
      },
      {
        "sourceId": "KB_SELECTION.json",
        "location": "optional_patterns: patterns/online-softmax-tail.md"
      },
      {
        "sourceId": "DESIGN_BINDINGS.json",
        "location": "patterns/online-softmax-tail.md / R04"
      },
      {
        "sourceId": "KB_USAGE.json",
        "location": "invariants[reference=patterns/online-softmax-tail.md]"
      }
    ],
    "applicabilityStatus": "not_triggered",
    "bindingPath": {
      "reference": "patterns/online-softmax-tail.md",
      "requirementIds": [
        "R04"
      ]
    }
  },
  {
    "id": "kb-buffer",
    "sourceId": "d-kb",
    "targetId": "d-buffer",
    "relationType": "informs",
    "reason": "buffer-reuse-lifetime 规定 Slot 轮转与生命周期；输入、输出各双槽，每行块各 advance 一次。双缓冲不证明实际重叠。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§4.2 / §5.1"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§2–§3 / §6"
      },
      {
        "sourceId": "KB_SELECTION.json",
        "location": "optional_patterns: patterns/buffer-reuse-lifetime.md"
      },
      {
        "sourceId": "DESIGN_BINDINGS.json",
        "location": "patterns/buffer-reuse-lifetime.md / R01 / R02 / R03"
      },
      {
        "sourceId": "KB_USAGE.json",
        "location": "invariants[reference=patterns/buffer-reuse-lifetime.md]"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": {
      "reference": "patterns/buffer-reuse-lifetime.md",
      "requirementIds": [
        "R01",
        "R02",
        "R03"
      ]
    }
  },
  {
    "id": "kb-precision",
    "sourceId": "d-kb",
    "targetId": "d-precision",
    "relationType": "informs",
    "reason": "已选知识要求高精度累加；本次 max 保持精确 fp16，exp/sum/div 用 fp32，输出再缩窄。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§4.2 / §5.1"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§1 Vector 选择"
      },
      {
        "sourceId": "KB_SELECTION.json",
        "location": "optional_patterns: patterns/online-softmax-tail.md"
      },
      {
        "sourceId": "DESIGN_BINDINGS.json",
        "location": "patterns/online-softmax-tail.md / R02"
      },
      {
        "sourceId": "KB_USAGE.json",
        "location": "invariants[reference=patterns/online-softmax-tail.md]"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": {
      "reference": "patterns/online-softmax-tail.md",
      "requirementIds": [
        "R02"
      ]
    }
  },
  {
    "id": "api-vf",
    "sourceId": "d-api",
    "targetId": "d-vf",
    "relationType": "informs",
    "reason": "Explore 核实 VF API 覆盖 Softmax 运算链，官方三趟 VF 样例为实现路径提供依据。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§3.3 / §4.2–4.3"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§1 API 映射"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": null
  },
  {
    "id": "api-precision",
    "sourceId": "d-api",
    "targetId": "d-precision",
    "relationType": "constrains",
    "reason": "FP16 reduce_sum 按 FP16 累加，长轴有精度风险；采用 FP32 sum，并按 exp_sub/astype 的 ZERO/ONE 半元素约束覆盖完整数据。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§3.3 / §4.2–4.3"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§1 Vector 选择"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": null
  },
  {
    "id": "api-tile",
    "sourceId": "d-api",
    "targetId": "d-tiling",
    "relationType": "constrains",
    "reason": "MAX_N 需满足 fp16 寄存器 128 元素粒度与数据 Tile 对齐；尾寄存器用 Mask。它约束列维对齐，不单独推导 TILE_ROWS=3。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§3.3 / §4.2–4.3"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§2 / §7"
      },
      {
        "sourceId": "KB_SELECTION.json",
        "location": "optional_patterns: patterns/vec-row-reduce-broadcast.md"
      },
      {
        "sourceId": "DESIGN_BINDINGS.json",
        "location": "patterns/vec-row-reduce-broadcast.md / R02 / R03"
      },
      {
        "sourceId": "KB_USAGE.json",
        "location": "invariants[reference=patterns/vec-row-reduce-broadcast.md]"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": {
      "reference": "patterns/vec-row-reduce-broadcast.md",
      "requirementIds": [
        "R02",
        "R03"
      ]
    }
  },
  {
    "id": "hw-tile",
    "sourceId": "d-constraint",
    "targetId": "d-tiling",
    "relationType": "constrains",
    "reason": "248KB UB 约束 MAX_N×TILE_ROWS×2 字节×4槽；3×8192×2×4=192KB，覆盖完整行。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§5.2 / §6.1 / §7"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§2–§3"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": null
  },
  {
    "id": "hw-buffer",
    "sourceId": "d-constraint",
    "targetId": "d-buffer",
    "relationType": "constrains",
    "reason": "四个 48KB Slot 共 192KB，小于 248KB；容量预算支持输入与输出各双槽。A5/950 最初为默认假设。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§5.2 / §6.1 / §7"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§3"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": null
  },
  {
    "id": "hw-schedule",
    "sourceId": "d-constraint",
    "targetId": "d-schedule",
    "relationType": "constrains",
    "reason": "仅 Vector Kernel 的 block_dim 由运行时 vector_core_num 与 num_tiles 的较小值限定；行块跨核 striding。没有硬编码实际核数。",
    "evidenceRefs": [
      {
        "sourceId": "EXPLORE_REPORT.md",
        "location": "§5.2 / §6.1 / §7"
      },
      {
        "sourceId": "DESIGN.md",
        "location": "§4–§5"
      }
    ],
    "applicabilityStatus": "applies",
    "bindingPath": null
  }
];
