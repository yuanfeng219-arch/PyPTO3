/* Measured DFX runtime & compiler evidence extracted from Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617. */
(function (global) {
  'use strict';
  global.PTO_CSA_RUNTIME_DATA = Object.freeze({
  "meta": {
    "artifactDir": "_jit_l3_decode_csa_20260903_010617",
    "platform": "a2a3",
    "chipSwimlaneLevel": 4,
    "numCores": 72,
    "aicCores": 24,
    "aivCores": 48,
    "totalTasks": 86,
    "totalTensors": 138,
    "totalEdges": 338,
    "totalKernels": 64,
    "ranks": {
      "0": {
        "spanUs": 4879.82,
        "workerSlices": 4081,
        "hbViolations": 13,
        "coldWallUs": 45193.08,
        "warmWallUs": 5132.8,
        "warmOrchUs": 202.38,
        "warmSchedUs": 4891.8
      },
      "1": {
        "spanUs": 3774.04,
        "workerSlices": 4081,
        "hbViolations": 12,
        "coldWallUs": 3613.34,
        "warmWallUs": 4017.34,
        "warmOrchUs": 152.32,
        "warmSchedUs": 3778.02
      }
    },
    "qkPvCompilerEvidence": {
      "passFile": "25_after_SplitVectorKernel.py",
      "dualAivDispatch": true,
      "c2vSlotBufferBytes": 32768,
      "v2cSlotBufferBytes": 32768,
      "slotSizeBytes": 16384,
      "slotNum": 2,
      "aiv1ValidShape": [
        0,
        0
      ],
      "qkKTiles": 4,
      "qkKStep": 128,
      "pvKTiles": 4,
      "pvKStep": 32
    }
  },
  "stages": [
    {
      "id": "input/q",
      "label": "Q Proj",
      "full": "Pre-Norm · AllGather · Q Projection",
      "short": "HC Pre-Norm、CP Token AllGather 与本地 Q Projection（含 RoPE）",
      "dataOwnership": "Rank 0 / Rank 1 各自持有本地 T 行 x_normed_t，直接驱动 Q Projection（不等待 AllGather 完成）；同时通过 cp_token_allgather 交换本地行拼出 [2T, 4096] 整组流供 KV 与双压缩器读取。",
      "source": [
        [
          "decode_csa.py",
          122,
          232
        ],
        [
          "qkv_proj_rope.py",
          1,
          140
        ],
        [
          "decode_cp_token_allgather.py",
          1,
          120
        ]
      ],
      "kernels": [
        {
          "name": "hc_pre_rms",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 56.86,
              "endUs": 76.92,
              "spanUs": 20.06,
              "minDurUs": 13.18,
              "medDurUs": 17.02,
              "maxDurUs": 19.96,
              "medKernelUs": 13.52,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 22.56,
              "endUs": 47.26,
              "spanUs": 24.7,
              "minDurUs": 16.66,
              "medDurUs": 20.62,
              "maxDurUs": 24.4,
              "medKernelUs": 17.18,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "hc_pre_linear",
          "coreType": "AIC",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 56.26,
              "endUs": 159.88,
              "spanUs": 103.62,
              "minDurUs": 5.86,
              "medDurUs": 6.72,
              "maxDurUs": 18.86,
              "medKernelUs": 6.16,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 64,
              "startUs": 26.78,
              "endUs": 125.84,
              "spanUs": 99.06,
              "minDurUs": 5.84,
              "medDurUs": 8.04,
              "maxDurUs": 20.64,
              "medKernelUs": 6.46,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "hc_pre_linear_reduce",
          "coreType": "AIV",
          "blockNum": 16,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 16,
              "startUs": 157.32,
              "endUs": 168.06,
              "spanUs": 10.74,
              "minDurUs": 6.42,
              "medDurUs": 7.02,
              "maxDurUs": 10.52,
              "medKernelUs": 2.1,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            },
            "1": {
              "slices": 16,
              "startUs": 123.22,
              "endUs": 132.02,
              "spanUs": 8.8,
              "minDurUs": 3.48,
              "medDurUs": 8.06,
              "maxDurUs": 8.74,
              "medKernelUs": 2.1,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            }
          }
        },
        {
          "name": "split_pre_post",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 172.86,
              "endUs": 208.04,
              "spanUs": 35.18,
              "minDurUs": 2.78,
              "medDurUs": 3.94,
              "maxDurUs": 15.48,
              "medKernelUs": 3.16,
              "coresUsed": 23,
              "aicUsed": 0,
              "aivUsed": 23
            },
            "1": {
              "slices": 32,
              "startUs": 141.3,
              "endUs": 166.38,
              "spanUs": 25.08,
              "minDurUs": 2.64,
              "medDurUs": 3.98,
              "maxDurUs": 6.06,
              "medKernelUs": 3.12,
              "coresUsed": 20,
              "aicUsed": 0,
              "aivUsed": 20
            }
          }
        },
        {
          "name": "comb_sinkhorn",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 172.08,
              "endUs": 206.88,
              "spanUs": 34.8,
              "minDurUs": 14.76,
              "medDurUs": 19.3,
              "maxDurUs": 25.3,
              "medKernelUs": 14.98,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 132.94,
              "endUs": 167.58,
              "spanUs": 34.64,
              "minDurUs": 14.78,
              "medDurUs": 16.98,
              "maxDurUs": 20.82,
              "medKernelUs": 15.62,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "mix_x",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 210.16,
              "endUs": 229.7,
              "spanUs": 19.54,
              "minDurUs": 8.84,
              "medDurUs": 10.28,
              "maxDurUs": 14.5,
              "medKernelUs": 9.22,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 166.52,
              "endUs": 186.78,
              "spanUs": 20.26,
              "minDurUs": 8.62,
              "medDurUs": 11.28,
              "maxDurUs": 16.54,
              "medKernelUs": 9.92,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "csa_rope_interleave",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 73.24,
              "endUs": 336.02,
              "spanUs": 262.78,
              "minDurUs": 262.78,
              "medDurUs": 262.78,
              "maxDurUs": 262.78,
              "medKernelUs": 261.62,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 50.3,
              "endUs": 313.9,
              "spanUs": 263.6,
              "minDurUs": 263.6,
              "medDurUs": 263.6,
              "maxDurUs": 263.6,
              "medKernelUs": 263.02,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "rms_norm",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 224.42,
              "endUs": 255.28,
              "spanUs": 30.86,
              "minDurUs": 18.24,
              "medDurUs": 25.32,
              "maxDurUs": 30.62,
              "medKernelUs": 15.18,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 188.16,
              "endUs": 214.08,
              "spanUs": 25.92,
              "minDurUs": 19.88,
              "medDurUs": 23.42,
              "maxDurUs": 25.28,
              "medKernelUs": 16.36,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "cp_token_allgather_push",
          "coreType": "AIV",
          "blockNum": 16,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 16,
              "startUs": 245.6,
              "endUs": 288.16,
              "spanUs": 42.56,
              "minDurUs": 30.12,
              "medDurUs": 36.76,
              "maxDurUs": 41.82,
              "medKernelUs": 20.24,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            },
            "1": {
              "slices": 16,
              "startUs": 210.66,
              "endUs": 246.48,
              "spanUs": 35.82,
              "minDurUs": 20.3,
              "medDurUs": 32.94,
              "maxDurUs": 34.2,
              "medKernelUs": 19.24,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            }
          }
        },
        {
          "name": "cp_token_allgather_payload_wait",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 268.1,
              "endUs": 1348.62,
              "spanUs": 1080.52,
              "minDurUs": 1080.52,
              "medDurUs": 1080.52,
              "maxDurUs": 1080.52,
              "medKernelUs": 1056.06,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 231.54,
              "endUs": 252.56,
              "spanUs": 21.02,
              "minDurUs": 21.02,
              "medDurUs": 21.02,
              "maxDurUs": 21.02,
              "medKernelUs": 0.88,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "cp_token_allgather_readback",
          "coreType": "AIV",
          "blockNum": 16,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 16,
              "startUs": 1355.6,
              "endUs": 1364.44,
              "spanUs": 8.84,
              "minDurUs": 4.8,
              "medDurUs": 7.28,
              "maxDurUs": 8.78,
              "medKernelUs": 6.94,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            },
            "1": {
              "slices": 16,
              "startUs": 257.34,
              "endUs": 320.58,
              "spanUs": 63.24,
              "minDurUs": 5.86,
              "medDurUs": 6.44,
              "maxDurUs": 8.06,
              "medKernelUs": 5.74,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            }
          }
        },
        {
          "name": "cp_token_allgather_readback_wait",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 1371.56,
              "endUs": 1418.58,
              "spanUs": 47.02,
              "minDurUs": 47.02,
              "medDurUs": 47.02,
              "maxDurUs": 47.02,
              "medKernelUs": 46.66,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 337.3,
              "endUs": 338.7,
              "spanUs": 1.4,
              "minDurUs": 1.4,
              "medDurUs": 1.4,
              "maxDurUs": 1.4,
              "medKernelUs": 1.0,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "cp_token_allgather_retire",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 1421.46,
              "endUs": 1423.18,
              "spanUs": 1.72,
              "minDurUs": 1.72,
              "medDurUs": 1.72,
              "maxDurUs": 1.72,
              "medKernelUs": 1.3,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 347.92,
              "endUs": 349.58,
              "spanUs": 1.66,
              "minDurUs": 1.66,
              "medDurUs": 1.66,
              "maxDurUs": 1.66,
              "medKernelUs": 1.34,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "q_rope_prepare",
          "coreType": "AIV",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 88.1,
              "endUs": 120.82,
              "spanUs": 32.72,
              "minDurUs": 2.4,
              "medDurUs": 4.8,
              "maxDurUs": 7.0,
              "medKernelUs": 4.04,
              "coresUsed": 47,
              "aicUsed": 0,
              "aivUsed": 47
            },
            "1": {
              "slices": 64,
              "startUs": 50.3,
              "endUs": 94.34,
              "spanUs": 44.04,
              "minDurUs": 2.1,
              "medDurUs": 4.78,
              "maxDurUs": 7.58,
              "medKernelUs": 4.3,
              "coresUsed": 46,
              "aicUsed": 0,
              "aivUsed": 46
            }
          }
        },
        {
          "name": "q_rope_prepare_0",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 123.44,
              "endUs": 139.44,
              "spanUs": 16.0,
              "minDurUs": 2.74,
              "medDurUs": 4.04,
              "maxDurUs": 5.98,
              "medKernelUs": 3.66,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 101.36,
              "endUs": 127.8,
              "spanUs": 26.44,
              "minDurUs": 2.96,
              "medDurUs": 5.8,
              "maxDurUs": 8.78,
              "medKernelUs": 5.42,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "qr_proj_seed",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 123.44,
              "endUs": 149.72,
              "spanUs": 26.28,
              "minDurUs": 26.28,
              "medDurUs": 26.28,
              "maxDurUs": 26.28,
              "medKernelUs": 25.9,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 125.24,
              "endUs": 151.38,
              "spanUs": 26.14,
              "minDurUs": 26.14,
              "medDurUs": 26.14,
              "maxDurUs": 26.14,
              "medKernelUs": 25.82,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "qr_proj_matmul",
          "coreType": "AIC",
          "blockNum": 16,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 16,
              "startUs": 268.08,
              "endUs": 354.98,
              "spanUs": 86.9,
              "minDurUs": 79.2,
              "medDurUs": 83.38,
              "maxDurUs": 86.34,
              "medKernelUs": 83.0,
              "coresUsed": 16,
              "aicUsed": 16,
              "aivUsed": 0
            },
            "1": {
              "slices": 16,
              "startUs": 226.76,
              "endUs": 316.28,
              "spanUs": 89.52,
              "minDurUs": 79.66,
              "medDurUs": 81.08,
              "maxDurUs": 86.76,
              "medKernelUs": 80.68,
              "coresUsed": 16,
              "aicUsed": 16,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "qr_rms_norm_quant",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 273.56,
              "endUs": 384.88,
              "spanUs": 111.32,
              "minDurUs": 39.4,
              "medDurUs": 102.48,
              "maxDurUs": 109.18,
              "medKernelUs": 11.64,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 234.1,
              "endUs": 330.28,
              "spanUs": 96.18,
              "minDurUs": 77.96,
              "medDurUs": 87.98,
              "maxDurUs": 96.02,
              "medKernelUs": 9.34,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "qproj_matmul",
          "coreType": "AIC",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 408.56,
              "endUs": 533.2,
              "spanUs": 124.64,
              "minDurUs": 28.3,
              "medDurUs": 34.42,
              "maxDurUs": 45.5,
              "medKernelUs": 33.68,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 64,
              "startUs": 375.02,
              "endUs": 633.5,
              "spanUs": 258.48,
              "minDurUs": 26.68,
              "medDurUs": 30.34,
              "maxDurUs": 45.68,
              "medKernelUs": 30.04,
              "coresUsed": 23,
              "aicUsed": 23,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "qproj_dequant_rms_nope_rope",
          "coreType": "AIV",
          "blockNum": 16,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 16,
              "startUs": 541.58,
              "endUs": 692.1,
              "spanUs": 150.52,
              "minDurUs": 145.84,
              "medDurUs": 148.32,
              "maxDurUs": 150.52,
              "medKernelUs": 147.5,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            },
            "1": {
              "slices": 16,
              "startUs": 644.86,
              "endUs": 793.9,
              "spanUs": 149.04,
              "minDurUs": 144.56,
              "medDurUs": 147.96,
              "maxDurUs": 149.04,
              "medKernelUs": 147.68,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            }
          }
        }
      ],
      "internalEdges": [
        {
          "from": "cp_token_allgather_payload_wait",
          "to": "cp_token_allgather_readback",
          "source": "explicit"
        },
        {
          "from": "cp_token_allgather_push",
          "to": "cp_token_allgather_payload_wait",
          "source": "explicit"
        },
        {
          "from": "cp_token_allgather_push",
          "to": "cp_token_allgather_payload_wait",
          "source": "tensormap"
        },
        {
          "from": "cp_token_allgather_push",
          "to": "cp_token_allgather_readback",
          "source": "tensormap"
        },
        {
          "from": "cp_token_allgather_push",
          "to": "cp_token_allgather_readback",
          "source": "explicit"
        },
        {
          "from": "cp_token_allgather_readback",
          "to": "cp_token_allgather_readback_wait",
          "source": "tensormap"
        },
        {
          "from": "cp_token_allgather_readback",
          "to": "cp_token_allgather_readback_wait",
          "source": "explicit"
        },
        {
          "from": "cp_token_allgather_readback",
          "to": "cp_token_allgather_retire",
          "source": "explicit"
        },
        {
          "from": "cp_token_allgather_readback",
          "to": "cp_token_allgather_retire",
          "source": "tensormap"
        },
        {
          "from": "cp_token_allgather_readback_wait",
          "to": "cp_token_allgather_retire",
          "source": "explicit"
        },
        {
          "from": "hc_pre_linear",
          "to": "hc_pre_linear_reduce",
          "source": "tensormap"
        },
        {
          "from": "hc_pre_linear_reduce",
          "to": "comb_sinkhorn",
          "source": "tensormap"
        },
        {
          "from": "hc_pre_linear_reduce",
          "to": "split_pre_post",
          "source": "tensormap"
        },
        {
          "from": "hc_pre_rms",
          "to": "comb_sinkhorn",
          "source": "tensormap"
        },
        {
          "from": "hc_pre_rms",
          "to": "split_pre_post",
          "source": "tensormap"
        },
        {
          "from": "mix_x",
          "to": "rms_norm",
          "source": "tensormap"
        },
        {
          "from": "q_rope_prepare_0",
          "to": "qproj_dequant_rms_nope_rope",
          "source": "tensormap"
        },
        {
          "from": "qproj_matmul",
          "to": "qproj_dequant_rms_nope_rope",
          "source": "tensormap"
        },
        {
          "from": "qr_proj_matmul",
          "to": "qr_rms_norm_quant",
          "source": "tensormap"
        },
        {
          "from": "qr_proj_seed",
          "to": "qr_proj_matmul",
          "source": "tensormap"
        },
        {
          "from": "qr_rms_norm_quant",
          "to": "qproj_matmul",
          "source": "tensormap"
        },
        {
          "from": "rms_norm",
          "to": "cp_token_allgather_push",
          "source": "tensormap"
        },
        {
          "from": "rms_norm",
          "to": "qr_proj_matmul",
          "source": "tensormap"
        },
        {
          "from": "split_pre_post",
          "to": "mix_x",
          "source": "tensormap"
        }
      ],
      "perfHints": [
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_csa.py",
          "line": 245,
          "detail": "tile.load has innermost dim = 64B (tile bfloat16[32], target_memory=Vec); moves 256B as 4 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_csa.py",
          "line": 250,
          "detail": "tile.load has innermost dim = 64B (tile bfloat16[32], target_memory=Vec); moves 256B as 4 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_csa.py",
          "line": 259,
          "detail": "tile.load has innermost dim = 64B (tile bfloat16[32], target_memory=Vec); moves 256B as 4 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_csa.py",
          "line": 264,
          "detail": "tile.load has innermost dim = 64B (tile bfloat16[32], target_memory=Vec); moves 256B as 4 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "hc_pre.py",
          "line": 92,
          "detail": "tile.store has innermost dim = 4B (tile fp32[1], target_memory=Vec); moves 32B as 8 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = 5"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "hc_pre.py",
          "line": 112,
          "detail": "tile.store has innermost dim = 128B (tile fp32[32], target_memory=Acc); moves 2048B as 16 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "hc_pre.py",
          "line": 118,
          "detail": "tile.load has innermost dim = 128B (tile fp32[32], target_memory=Vec); moves 2048B as 16 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "hc_pre.py",
          "line": 121,
          "detail": "tile.load has innermost dim = 128B (tile fp32[32], target_memory=Vec); moves 2048B as 16 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        }
      ],
      "ranks": {
        "0": {
          "startUs": 56.26,
          "endUs": 1423.18,
          "spanUs": 1366.92,
          "slices": 501,
          "hbViolations": 9
        },
        "1": {
          "startUs": 22.56,
          "endUs": 793.9,
          "spanUs": 771.34,
          "slices": 501,
          "hbViolations": 8
        }
      }
    },
    {
      "id": "input/kv",
      "label": "KV Proj",
      "full": "KV Projection · Original KV Cache Writeback",
      "short": "读取整组 2T 流计算 KV、RoPE 与原始滑窗 KV Cache 写回",
      "dataOwnership": "读取 AllGather 产出的整组副本 x_normed_full [2T, 4096]（顺序 [R0 rows | R1 rows]），经 kv_proj_matmul 与 kv_rms_norm_rope 后写回 Original KV Cache [N_ori, 32, 1, 512]。",
      "source": [
        [
          "decode_csa.py",
          233,
          278
        ],
        [
          "qkv_proj_rope.py",
          141,
          240
        ]
      ],
      "kernels": [
        {
          "name": "kv_proj_seed",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 123.44,
              "endUs": 149.46,
              "spanUs": 26.02,
              "minDurUs": 26.02,
              "medDurUs": 26.02,
              "maxDurUs": 26.02,
              "medKernelUs": 25.46,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 65.1,
              "endUs": 94.48,
              "spanUs": 29.38,
              "minDurUs": 29.38,
              "medDurUs": 29.38,
              "maxDurUs": 29.38,
              "medKernelUs": 29.0,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "kv_proj_matmul",
          "coreType": "AIC",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 1431.28,
              "endUs": 1532.48,
              "spanUs": 101.2,
              "minDurUs": 35.62,
              "medDurUs": 44.82,
              "maxDurUs": 62.46,
              "medKernelUs": 44.48,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 32,
              "startUs": 379.38,
              "endUs": 676.8,
              "spanUs": 297.42,
              "minDurUs": 37.62,
              "medDurUs": 44.3,
              "maxDurUs": 68.7,
              "medKernelUs": 43.88,
              "coresUsed": 18,
              "aicUsed": 18,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "kv_rms_norm_rope",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 1544.84,
              "endUs": 1558.96,
              "spanUs": 14.12,
              "minDurUs": 6.1,
              "medDurUs": 7.98,
              "maxDurUs": 9.12,
              "medKernelUs": 7.56,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 688.84,
              "endUs": 702.84,
              "spanUs": 14.0,
              "minDurUs": 7.06,
              "medDurUs": 8.4,
              "maxDurUs": 9.38,
              "medKernelUs": 7.92,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "csa_cache_writeback",
          "coreType": "AIV",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 1578.56,
              "endUs": 1607.66,
              "spanUs": 29.1,
              "minDurUs": 3.14,
              "medDurUs": 4.66,
              "maxDurUs": 7.86,
              "medKernelUs": 4.3,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 64,
              "startUs": 726.3,
              "endUs": 769.46,
              "spanUs": 43.16,
              "minDurUs": 2.98,
              "medDurUs": 3.82,
              "maxDurUs": 5.88,
              "medKernelUs": 3.4,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "kv_touch",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 1624.98,
              "endUs": 1626.28,
              "spanUs": 1.3,
              "minDurUs": 1.3,
              "medDurUs": 1.3,
              "maxDurUs": 1.3,
              "medKernelUs": 0.88,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 786.48,
              "endUs": 787.82,
              "spanUs": 1.34,
              "minDurUs": 1.34,
              "medDurUs": 1.34,
              "maxDurUs": 1.34,
              "medKernelUs": 0.98,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        }
      ],
      "internalEdges": [
        {
          "from": "csa_cache_writeback",
          "to": "kv_touch",
          "source": "tensormap"
        },
        {
          "from": "kv_proj_matmul",
          "to": "kv_rms_norm_rope",
          "source": "tensormap"
        },
        {
          "from": "kv_proj_seed",
          "to": "kv_proj_matmul",
          "source": "tensormap"
        },
        {
          "from": "kv_rms_norm_rope",
          "to": "csa_cache_writeback",
          "source": "tensormap"
        }
      ],
      "perfHints": [
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "qkv_proj_rope.py",
          "line": 158,
          "detail": "tile.load has innermost dim = 128B (tile bfloat16[64], target_memory=Vec); moves 1024B as 8 x 128B rows; recommended >= 512B for backend a2a3 (L2 cach"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "qkv_proj_rope.py",
          "line": 159,
          "detail": "tile.load has innermost dim = 128B (tile bfloat16[64], target_memory=Vec); moves 1024B as 8 x 128B rows; recommended >= 512B for backend a2a3 (L2 cach"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "qkv_proj_rope.py",
          "line": 165,
          "detail": "tile.store has innermost dim = 256B (tile fp32[64], target_memory=Vec); moves 2048B as 8 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "qkv_proj_rope.py",
          "line": 166,
          "detail": "tile.store has innermost dim = 256B (tile fp32[64], target_memory=Vec); moves 2048B as 8 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        }
      ],
      "ranks": {
        "0": {
          "startUs": 123.44,
          "endUs": 1626.28,
          "spanUs": 1502.84,
          "slices": 130,
          "hbViolations": 9
        },
        "1": {
          "startUs": 65.1,
          "endUs": 787.82,
          "spanUs": 722.72,
          "slices": 130,
          "hbViolations": 8
        }
      }
    },
    {
      "id": "cache/main",
      "label": "Compressor",
      "full": "Main & Indexer KV Compressors (Ratio-4)",
      "short": "双路 Ratio-4 压缩状态更新与 Compressed / Indexer KV Cache 写入",
      "dataOwnership": "两条独立压缩支路各自读取 x_normed_full [2T, 4096] 并维护请求级滑动压缩状态：Main 支路写回 Compressed KV Cache（BF16），Indexer 支路经 Hadamard 量化写回 Indexer KV Cache（INT8 + FP32 scale）。",
      "source": [
        [
          "decode_csa.py",
          279,
          345
        ],
        [
          "decode_compressor_ratio4.py",
          1,
          160
        ],
        [
          "decode_indexer_compressor.py",
          1,
          160
        ]
      ],
      "kernels": [
        {
          "name": "kv_score_proj",
          "coreType": "AIC",
          "blockNum": 512,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 512,
              "startUs": 1430.6,
              "endUs": 2194.16,
              "spanUs": 763.56,
              "minDurUs": 9.2,
              "medDurUs": 10.32,
              "maxDurUs": 30.66,
              "medKernelUs": 9.94,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 512,
              "startUs": 386.46,
              "endUs": 1537.52,
              "spanUs": 1151.06,
              "minDurUs": 9.24,
              "medDurUs": 10.02,
              "maxDurUs": 36.76,
              "medKernelUs": 9.64,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "scatter_softmax_pool",
          "coreType": "AIV",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 2303.56,
              "endUs": 2379.54,
              "spanUs": 75.98,
              "minDurUs": 29.38,
              "medDurUs": 33.98,
              "maxDurUs": 36.54,
              "medKernelUs": 33.6,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 64,
              "startUs": 1557.7,
              "endUs": 1636.06,
              "spanUs": 78.36,
              "minDurUs": 29.2,
              "medDurUs": 34.44,
              "maxDurUs": 37.98,
              "medKernelUs": 34.1,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        },
        {
          "name": "compress_state_commit",
          "coreType": "AIV",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 2419.24,
              "endUs": 2465.86,
              "spanUs": 46.62,
              "minDurUs": 5.9,
              "medDurUs": 7.46,
              "maxDurUs": 9.16,
              "medKernelUs": 7.08,
              "coresUsed": 47,
              "aicUsed": 0,
              "aivUsed": 47
            },
            "1": {
              "slices": 64,
              "startUs": 1646.1,
              "endUs": 1746.06,
              "spanUs": 99.96,
              "minDurUs": 5.34,
              "medDurUs": 6.34,
              "maxDurUs": 7.84,
              "medKernelUs": 5.98,
              "coresUsed": 45,
              "aicUsed": 0,
              "aivUsed": 45
            }
          }
        },
        {
          "name": "rmsnorm_rope_cache_write",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 2418.56,
              "endUs": 2449.02,
              "spanUs": 30.46,
              "minDurUs": 9.42,
              "medDurUs": 11.36,
              "maxDurUs": 12.88,
              "medKernelUs": 11.0,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 1644.16,
              "endUs": 1698.1,
              "spanUs": 53.94,
              "minDurUs": 8.28,
              "medDurUs": 10.68,
              "maxDurUs": 12.9,
              "medKernelUs": 10.34,
              "coresUsed": 24,
              "aicUsed": 0,
              "aivUsed": 24
            }
          }
        },
        {
          "name": "kv_score_proj_0",
          "coreType": "AIC",
          "blockNum": 256,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 256,
              "startUs": 1428.08,
              "endUs": 1823.72,
              "spanUs": 395.64,
              "minDurUs": 5.66,
              "medDurUs": 6.24,
              "maxDurUs": 18.04,
              "medKernelUs": 5.9,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 256,
              "startUs": 360.36,
              "endUs": 1014.68,
              "spanUs": 654.32,
              "minDurUs": 5.56,
              "medDurUs": 6.38,
              "maxDurUs": 20.78,
              "medKernelUs": 6.02,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "scatter_softmax_pool_0",
          "coreType": "AIV",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 1839.6,
              "endUs": 1875.28,
              "spanUs": 35.68,
              "minDurUs": 9.4,
              "medDurUs": 13.48,
              "maxDurUs": 17.34,
              "medKernelUs": 13.16,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 64,
              "startUs": 1102.38,
              "endUs": 1170.76,
              "spanUs": 68.38,
              "minDurUs": 9.92,
              "medDurUs": 12.92,
              "maxDurUs": 20.48,
              "medKernelUs": 12.56,
              "coresUsed": 46,
              "aicUsed": 0,
              "aivUsed": 46
            }
          }
        },
        {
          "name": "compress_state_commit_0",
          "coreType": "AIV",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 1902.32,
              "endUs": 1949.38,
              "spanUs": 47.06,
              "minDurUs": 5.78,
              "medDurUs": 7.36,
              "maxDurUs": 9.6,
              "medKernelUs": 6.98,
              "coresUsed": 43,
              "aicUsed": 0,
              "aivUsed": 43
            },
            "1": {
              "slices": 64,
              "startUs": 1205.32,
              "endUs": 1239.8,
              "spanUs": 34.48,
              "minDurUs": 5.1,
              "medDurUs": 6.9,
              "maxDurUs": 10.48,
              "medKernelUs": 6.48,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        },
        {
          "name": "rmsnorm_rope",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 1896.8,
              "endUs": 1908.72,
              "spanUs": 11.92,
              "minDurUs": 4.22,
              "medDurUs": 8.5,
              "maxDurUs": 11.86,
              "medKernelUs": 8.0,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 1187.28,
              "endUs": 1203.24,
              "spanUs": 15.96,
              "minDurUs": 5.14,
              "medDurUs": 6.7,
              "maxDurUs": 8.26,
              "medKernelUs": 6.24,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        },
        {
          "name": "kv_hadamard",
          "coreType": "AIC",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 1956.84,
              "endUs": 1989.28,
              "spanUs": 32.44,
              "minDurUs": 1.94,
              "medDurUs": 2.82,
              "maxDurUs": 3.82,
              "medKernelUs": 2.5,
              "coresUsed": 21,
              "aicUsed": 21,
              "aivUsed": 0
            },
            "1": {
              "slices": 32,
              "startUs": 1309.7,
              "endUs": 1411.02,
              "spanUs": 101.32,
              "minDurUs": 2.0,
              "medDurUs": 3.0,
              "maxDurUs": 4.2,
              "medKernelUs": 2.66,
              "coresUsed": 23,
              "aicUsed": 23,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "kv_and_cache_write",
          "coreType": "AIV",
          "blockNum": 32,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 32,
              "startUs": 2003.18,
              "endUs": 2013.26,
              "spanUs": 10.08,
              "minDurUs": 4.3,
              "medDurUs": 7.46,
              "maxDurUs": 10.06,
              "medKernelUs": 7.08,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            },
            "1": {
              "slices": 32,
              "startUs": 1546.0,
              "endUs": 1560.5,
              "spanUs": 14.5,
              "minDurUs": 3.46,
              "medDurUs": 4.62,
              "maxDurUs": 6.34,
              "medKernelUs": 4.1,
              "coresUsed": 32,
              "aicUsed": 0,
              "aivUsed": 32
            }
          }
        }
      ],
      "internalEdges": [
        {
          "from": "kv_hadamard",
          "to": "kv_and_cache_write",
          "source": "tensormap"
        },
        {
          "from": "kv_hadamard",
          "to": "kv_and_cache_write",
          "source": "explicit"
        },
        {
          "from": "kv_score_proj",
          "to": "compress_state_commit",
          "source": "tensormap"
        },
        {
          "from": "kv_score_proj",
          "to": "scatter_softmax_pool",
          "source": "tensormap"
        },
        {
          "from": "kv_score_proj",
          "to": "scatter_softmax_pool",
          "source": "explicit"
        },
        {
          "from": "kv_score_proj_0",
          "to": "compress_state_commit_0",
          "source": "tensormap"
        },
        {
          "from": "kv_score_proj_0",
          "to": "scatter_softmax_pool_0",
          "source": "explicit"
        },
        {
          "from": "kv_score_proj_0",
          "to": "scatter_softmax_pool_0",
          "source": "tensormap"
        },
        {
          "from": "rmsnorm_rope",
          "to": "kv_hadamard",
          "source": "tensormap"
        },
        {
          "from": "rmsnorm_rope",
          "to": "kv_hadamard",
          "source": "explicit"
        },
        {
          "from": "scatter_softmax_pool",
          "to": "compress_state_commit",
          "source": "explicit"
        },
        {
          "from": "scatter_softmax_pool",
          "to": "rmsnorm_rope_cache_write",
          "source": "tensormap"
        },
        {
          "from": "scatter_softmax_pool",
          "to": "rmsnorm_rope_cache_write",
          "source": "explicit"
        },
        {
          "from": "scatter_softmax_pool_0",
          "to": "compress_state_commit_0",
          "source": "explicit"
        },
        {
          "from": "scatter_softmax_pool_0",
          "to": "rmsnorm_rope",
          "source": "explicit"
        },
        {
          "from": "scatter_softmax_pool_0",
          "to": "rmsnorm_rope",
          "source": "tensormap"
        }
      ],
      "perfHints": [
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_compressor_ratio4.py",
          "line": 135,
          "detail": "tile.store has innermost dim = 256B (tile fp32[64], target_memory=Acc); moves 4096B as 16 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_compressor_ratio4.py",
          "line": 136,
          "detail": "tile.store has innermost dim = 256B (tile fp32[64], target_memory=Acc); moves 4096B as 16 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_compressor_ratio4.py",
          "line": 156,
          "detail": "tile.load has innermost dim = 256B (tile fp32[64], target_memory=Vec); moves 256B as 1 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache lin"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_compressor_ratio4.py",
          "line": 167,
          "detail": "tile.load has innermost dim = 256B (tile fp32[64], target_memory=Vec); moves 256B as 1 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache lin"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_indexer_compressor.py",
          "line": 138,
          "detail": "tile.store has innermost dim = 128B (tile fp32[32], target_memory=Acc); moves 2048B as 16 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_indexer_compressor.py",
          "line": 139,
          "detail": "tile.store has innermost dim = 128B (tile fp32[32], target_memory=Acc); moves 2048B as 16 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_indexer_compressor.py",
          "line": 159,
          "detail": "tile.load has innermost dim = 256B (tile fp32[64], target_memory=Vec); moves 256B as 1 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache lin"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_indexer_compressor.py",
          "line": 170,
          "detail": "tile.load has innermost dim = 256B (tile fp32[64], target_memory=Vec); moves 256B as 1 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache lin"
        }
      ],
      "ranks": {
        "0": {
          "startUs": 1428.08,
          "endUs": 2465.86,
          "spanUs": 1037.78,
          "slices": 1152,
          "hbViolations": 0
        },
        "1": {
          "startUs": 360.36,
          "endUs": 1746.06,
          "spanUs": 1385.7,
          "slices": 1152,
          "hbViolations": 0
        }
      }
    },
    {
      "id": "cache/indexer",
      "label": "Indexer",
      "full": "Indexer Query · Score Leaf Wave · Top-K Selection",
      "short": "Indexer Query Hadamard 量化、1:2 混核相关性打分与 Top-K 筛选",
      "dataOwnership": "仅读取本地 T 行构造 INT8 Indexer Query，与 Indexer KV Cache 通过 1:2 混核 indexer_score_leaf_wave（24 AIC + 48 AIV）打分，再经两级 Top-K 规约输出 idx_topk [T, 512]。",
      "source": [
        [
          "decode_csa.py",
          346,
          376
        ],
        [
          "decode_indexer.py",
          1,
          200
        ]
      ],
      "kernels": [
        {
          "name": "idx_qr_proj_matmul",
          "coreType": "AIC",
          "blockNum": 128,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 128,
              "startUs": 396.46,
              "endUs": 586.54,
              "spanUs": 190.08,
              "minDurUs": 9.76,
              "medDurUs": 10.2,
              "maxDurUs": 20.06,
              "medKernelUs": 9.76,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 128,
              "startUs": 352.58,
              "endUs": 758.88,
              "spanUs": 406.3,
              "minDurUs": 9.72,
              "medDurUs": 10.22,
              "maxDurUs": 25.32,
              "medKernelUs": 9.84,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "idx_qr_proj_dequant",
          "coreType": "AIV",
          "blockNum": 8,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 8,
              "startUs": 580.1,
              "endUs": 620.6,
              "spanUs": 40.5,
              "minDurUs": 38.04,
              "medDurUs": 39.82,
              "maxDurUs": 40.44,
              "medKernelUs": 31.82,
              "coresUsed": 8,
              "aicUsed": 0,
              "aivUsed": 8
            },
            "1": {
              "slices": 8,
              "startUs": 779.68,
              "endUs": 814.76,
              "spanUs": 35.08,
              "minDurUs": 33.68,
              "medDurUs": 34.74,
              "maxDurUs": 35.04,
              "medKernelUs": 34.3,
              "coresUsed": 8,
              "aicUsed": 0,
              "aivUsed": 8
            }
          }
        },
        {
          "name": "qr_rope_swap_idx",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 135.68,
              "endUs": 138.54,
              "spanUs": 2.86,
              "minDurUs": 2.86,
              "medDurUs": 2.86,
              "maxDurUs": 2.86,
              "medKernelUs": 2.3,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 125.24,
              "endUs": 128.14,
              "spanUs": 2.9,
              "minDurUs": 2.9,
              "medDurUs": 2.9,
              "maxDurUs": 2.9,
              "medKernelUs": 2.44,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "qr_rope",
          "coreType": "AIV",
          "blockNum": 512,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 512,
              "startUs": 629.5,
              "endUs": 881.62,
              "spanUs": 252.12,
              "minDurUs": 2.9,
              "medDurUs": 4.02,
              "maxDurUs": 9.6,
              "medKernelUs": 3.66,
              "coresUsed": 47,
              "aicUsed": 0,
              "aivUsed": 47
            },
            "1": {
              "slices": 512,
              "startUs": 828.02,
              "endUs": 1106.3,
              "spanUs": 278.28,
              "minDurUs": 2.88,
              "medDurUs": 4.28,
              "maxDurUs": 9.26,
              "medKernelUs": 3.9,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        },
        {
          "name": "qr_hadamard_matmul",
          "coreType": "AIC",
          "blockNum": 256,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 256,
              "startUs": 882.66,
              "endUs": 1035.2,
              "spanUs": 152.54,
              "minDurUs": 1.78,
              "medDurUs": 2.52,
              "maxDurUs": 19.7,
              "medKernelUs": 2.14,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 256,
              "startUs": 1148.64,
              "endUs": 1401.64,
              "spanUs": 253.0,
              "minDurUs": 1.68,
              "medDurUs": 2.66,
              "maxDurUs": 13.88,
              "medKernelUs": 2.3,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "qr_hadamard_quant",
          "coreType": "AIV",
          "blockNum": 256,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 256,
              "startUs": 1038.48,
              "endUs": 1354.44,
              "spanUs": 315.96,
              "minDurUs": 5.46,
              "medDurUs": 7.02,
              "maxDurUs": 20.08,
              "medKernelUs": 6.58,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 256,
              "startUs": 1406.36,
              "endUs": 1569.56,
              "spanUs": 163.2,
              "minDurUs": 5.54,
              "medDurUs": 7.92,
              "maxDurUs": 18.6,
              "medKernelUs": 7.52,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        },
        {
          "name": "weights_proj",
          "coreType": "AIC",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 347.22,
              "endUs": 398.66,
              "spanUs": 51.44,
              "minDurUs": 2.9,
              "medDurUs": 9.58,
              "maxDurUs": 23.16,
              "medKernelUs": 9.24,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 64,
              "startUs": 324.16,
              "endUs": 455.28,
              "spanUs": 131.12,
              "minDurUs": 2.96,
              "medDurUs": 15.3,
              "maxDurUs": 25.6,
              "medKernelUs": 14.92,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "weights_proj_reduce",
          "coreType": "AIV",
          "blockNum": 16,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 16,
              "startUs": 411.38,
              "endUs": 416.34,
              "spanUs": 4.96,
              "minDurUs": 2.72,
              "medDurUs": 3.72,
              "maxDurUs": 4.9,
              "medKernelUs": 3.24,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            },
            "1": {
              "slices": 16,
              "startUs": 462.86,
              "endUs": 467.42,
              "spanUs": 4.56,
              "minDurUs": 2.18,
              "medDurUs": 3.22,
              "maxDurUs": 4.5,
              "medKernelUs": 2.82,
              "coresUsed": 16,
              "aicUsed": 0,
              "aivUsed": 16
            }
          }
        },
        {
          "name": "indexer_score_leaf_wave_aic",
          "coreType": "MIX",
          "blockNum": 24,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 24,
              "startUs": 2036.92,
              "endUs": 2183.94,
              "spanUs": 147.02,
              "minDurUs": 20.78,
              "medDurUs": 110.9,
              "maxDurUs": 125.36,
              "medKernelUs": 110.54,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 24,
              "startUs": 1589.26,
              "endUs": 1738.82,
              "spanUs": 149.56,
              "minDurUs": 18.82,
              "medDurUs": 67.5,
              "maxDurUs": 127.56,
              "medKernelUs": 67.16,
              "coresUsed": 22,
              "aicUsed": 22,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "indexer_score_leaf_wave_aiv",
          "coreType": "MIX",
          "blockNum": 24,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 48,
              "startUs": 2036.92,
              "endUs": 2184.62,
              "spanUs": 147.7,
              "minDurUs": 20.96,
              "medDurUs": 111.36,
              "maxDurUs": 125.72,
              "medKernelUs": 110.82,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 48,
              "startUs": 1594.2,
              "endUs": 1739.9,
              "spanUs": 145.7,
              "minDurUs": 18.66,
              "medDurUs": 63.74,
              "maxDurUs": 110.98,
              "medKernelUs": 63.38,
              "coresUsed": 44,
              "aicUsed": 0,
              "aivUsed": 44
            }
          }
        },
        {
          "name": "indexer_topk_group_wave",
          "coreType": "AIV",
          "blockNum": 48,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 48,
              "startUs": 2192.72,
              "endUs": 2325.46,
              "spanUs": 132.74,
              "minDurUs": 110.54,
              "medDurUs": 112.74,
              "maxDurUs": 132.74,
              "medKernelUs": 112.36,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 48,
              "startUs": 1746.62,
              "endUs": 1890.76,
              "spanUs": 144.14,
              "minDurUs": 109.74,
              "medDurUs": 111.52,
              "maxDurUs": 130.58,
              "medKernelUs": 111.06,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        },
        {
          "name": "indexer_topk_query_merge",
          "coreType": "AIV",
          "blockNum": 256,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 256,
              "startUs": 2340.2,
              "endUs": 2501.6,
              "spanUs": 161.4,
              "minDurUs": 1.2,
              "medDurUs": 3.56,
              "maxDurUs": 12.26,
              "medKernelUs": 3.16,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 256,
              "startUs": 1898.0,
              "endUs": 2004.32,
              "spanUs": 106.32,
              "minDurUs": 1.5,
              "medDurUs": 4.26,
              "maxDurUs": 12.68,
              "medKernelUs": 3.88,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        }
      ],
      "internalEdges": [
        {
          "from": "idx_qr_proj_dequant",
          "to": "qr_rope",
          "source": "tensormap"
        },
        {
          "from": "idx_qr_proj_matmul",
          "to": "idx_qr_proj_dequant",
          "source": "tensormap"
        },
        {
          "from": "indexer_score_leaf_wave_aic",
          "to": "indexer_topk_group_wave",
          "source": "explicit"
        },
        {
          "from": "indexer_score_leaf_wave_aic",
          "to": "indexer_topk_group_wave",
          "source": "tensormap"
        },
        {
          "from": "indexer_score_leaf_wave_aiv",
          "to": "indexer_topk_group_wave",
          "source": "explicit"
        },
        {
          "from": "indexer_score_leaf_wave_aiv",
          "to": "indexer_topk_group_wave",
          "source": "tensormap"
        },
        {
          "from": "indexer_topk_group_wave",
          "to": "indexer_topk_query_merge",
          "source": "explicit"
        },
        {
          "from": "indexer_topk_group_wave",
          "to": "indexer_topk_query_merge",
          "source": "tensormap"
        },
        {
          "from": "qr_hadamard_matmul",
          "to": "qr_hadamard_quant",
          "source": "tensormap"
        },
        {
          "from": "qr_hadamard_quant",
          "to": "indexer_score_leaf_wave_aic",
          "source": "explicit"
        },
        {
          "from": "qr_hadamard_quant",
          "to": "indexer_score_leaf_wave_aic",
          "source": "tensormap"
        },
        {
          "from": "qr_hadamard_quant",
          "to": "indexer_score_leaf_wave_aiv",
          "source": "tensormap"
        },
        {
          "from": "qr_hadamard_quant",
          "to": "indexer_score_leaf_wave_aiv",
          "source": "explicit"
        },
        {
          "from": "qr_rope",
          "to": "qr_hadamard_matmul",
          "source": "tensormap"
        },
        {
          "from": "qr_rope_swap_idx",
          "to": "qr_rope",
          "source": "tensormap"
        },
        {
          "from": "weights_proj",
          "to": "weights_proj_reduce",
          "source": "tensormap"
        },
        {
          "from": "weights_proj_reduce",
          "to": "indexer_score_leaf_wave_aic",
          "source": "explicit"
        },
        {
          "from": "weights_proj_reduce",
          "to": "indexer_score_leaf_wave_aic",
          "source": "tensormap"
        },
        {
          "from": "weights_proj_reduce",
          "to": "indexer_score_leaf_wave_aiv",
          "source": "tensormap"
        },
        {
          "from": "weights_proj_reduce",
          "to": "indexer_score_leaf_wave_aiv",
          "source": "explicit"
        }
      ],
      "perfHints": [
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_indexer.py",
          "line": 519,
          "detail": "tile.load has innermost dim = 128B (tile int8[128], target_memory=Mat); moves 8192B as 64 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_indexer.py",
          "line": 523,
          "detail": "tile.load has innermost dim = 4B (tile fp32[1], target_memory=Vec); moves 256B as 64 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_indexer.py",
          "line": 530,
          "detail": "tile.load has innermost dim = 256B (tile fp32[64], target_memory=Vec); moves 256B as 1 x 256B rows; recommended >= 512B for backend a2a3 (L2 cache lin"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_indexer.py",
          "line": 552,
          "detail": "tile.load has innermost dim = 128B (tile int8[128], target_memory=Mat); moves 4096B as 32 x 128B rows; recommended >= 512B for backend a2a3 (L2 cache "
        }
      ],
      "ranks": {
        "0": {
          "startUs": 135.68,
          "endUs": 2501.6,
          "spanUs": 2365.92,
          "slices": 1617,
          "hbViolations": 9
        },
        "1": {
          "startUs": 125.24,
          "endUs": 2004.32,
          "spanUs": 1879.08,
          "slices": 1617,
          "hbViolations": 7
        }
      }
    },
    {
      "id": "sparse/attention/plan",
      "label": "qk_plan",
      "full": "Candidate Planning · valid_block_mask & qk_order",
      "short": "单核重排 valid-first 工作队列、生成 sparse_bias 与逆 RoPE 系数",
      "dataOwnership": "由单 Block AIV 核 csa_slots_build_valid_qk_plan 消费 idx_topk 与 position_ids，生成 valid_block_mask [T, 5]、sparse_bias [T, 640] 与重排队列 qk_order [1280]；rope_cs 独立生成逆 RoPE 系数。",
      "source": [
        [
          "decode_sparse_attn_csa.py",
          127,
          200
        ],
        [
          "decode_sparse_attn_csa.py",
          315,
          349
        ]
      ],
      "kernels": [
        {
          "name": "csa_slots_build_valid_qk_plan",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 2510.62,
              "endUs": 2625.34,
              "spanUs": 114.72,
              "minDurUs": 114.72,
              "medDurUs": 114.72,
              "maxDurUs": 114.72,
              "medKernelUs": 114.3,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 2011.76,
              "endUs": 2125.72,
              "spanUs": 113.96,
              "minDurUs": 113.96,
              "medDurUs": 113.96,
              "maxDurUs": 113.96,
              "medKernelUs": 113.62,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "rope_cs",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 147.64,
              "endUs": 196.78,
              "spanUs": 49.14,
              "minDurUs": 49.14,
              "medDurUs": 49.14,
              "maxDurUs": 49.14,
              "medKernelUs": 48.82,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 125.24,
              "endUs": 174.12,
              "spanUs": 48.88,
              "minDurUs": 48.88,
              "medDurUs": 48.88,
              "maxDurUs": 48.88,
              "medKernelUs": 48.46,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        }
      ],
      "internalEdges": [],
      "perfHints": [
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_sparse_attn_csa.py",
          "line": 148,
          "detail": "tile.load has innermost dim = 4B (tile int32[1], target_memory=Vec); moves 32B as 8 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = 5"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_sparse_attn_csa.py",
          "line": 302,
          "detail": "tile.store has innermost dim = 4B (tile fp32[1], target_memory=Vec); moves 64B as 16 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_sparse_attn_csa.py",
          "line": 303,
          "detail": "tile.store has innermost dim = 4B (tile fp32[1], target_memory=Vec); moves 64B as 16 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_sparse_attn_csa.py",
          "line": 311,
          "detail": "tile.store has innermost dim = 4B (tile fp32[1], target_memory=Vec); moves 64B as 16 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = "
        }
      ],
      "ranks": {
        "0": {
          "startUs": 147.64,
          "endUs": 2625.34,
          "spanUs": 2477.7,
          "slices": 2,
          "hbViolations": 9
        },
        "1": {
          "startUs": 125.24,
          "endUs": 2125.72,
          "spanUs": 2000.48,
          "slices": 2,
          "hbViolations": 7
        }
      }
    },
    {
      "id": "sparse/attention/qk-pv",
      "label": "qk_pv",
      "full": "Sparse Attention · qk_pv (1:2 MIX SPMD)",
      "short": "24 组 1 AIC + 2 AIV 混核执行 KV Gather、双 Head Batch QK/Softmax/PV 与 Partial 输出",
      "dataOwnership": "单次 pl.spmd(24) 派发 24 AIC + 48 AIV，按 qk_order 领取 (token, sparseBlock) Work Item；AIC/AIV 通过双槽 16KB C2V/V2C Pipe 协作（AIV1 为 dual_aiv_dispatch 零有效行伴随核），写出 sparse_blk_mi/li/oi。",
      "source": [
        [
          "decode_sparse_attn_csa.py",
          201,
          314
        ],
        [
          "decode_csa.py",
          377,
          394
        ]
      ],
      "kernels": [
        {
          "name": "qk_pv_aic",
          "coreType": "MIX",
          "blockNum": 24,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 24,
              "startUs": 2634.28,
              "endUs": 3470.34,
              "spanUs": 836.06,
              "minDurUs": 377.56,
              "medDurUs": 783.22,
              "maxDurUs": 830.26,
              "medKernelUs": 782.86,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 24,
              "startUs": 2134.96,
              "endUs": 3030.98,
              "spanUs": 896.02,
              "minDurUs": 362.74,
              "medDurUs": 835.44,
              "maxDurUs": 896.0,
              "medKernelUs": 835.02,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "qk_pv_aiv",
          "coreType": "MIX",
          "blockNum": 24,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 48,
              "startUs": 2634.28,
              "endUs": 3520.66,
              "spanUs": 886.38,
              "minDurUs": 375.98,
              "medDurUs": 782.34,
              "maxDurUs": 881.22,
              "medKernelUs": 781.74,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 48,
              "startUs": 2134.94,
              "endUs": 3083.46,
              "spanUs": 948.52,
              "minDurUs": 360.38,
              "medDurUs": 834.44,
              "maxDurUs": 948.46,
              "medKernelUs": 834.08,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        }
      ],
      "internalEdges": [],
      "perfHints": [
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_sparse_attn_csa.py",
          "line": 148,
          "detail": "tile.load has innermost dim = 4B (tile int32[1], target_memory=Vec); moves 32B as 8 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = 5"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_sparse_attn_csa.py",
          "line": 302,
          "detail": "tile.store has innermost dim = 4B (tile fp32[1], target_memory=Vec); moves 64B as 16 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_sparse_attn_csa.py",
          "line": 303,
          "detail": "tile.store has innermost dim = 4B (tile fp32[1], target_memory=Vec); moves 64B as 16 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = "
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_sparse_attn_csa.py",
          "line": 311,
          "detail": "tile.store has innermost dim = 4B (tile fp32[1], target_memory=Vec); moves 64B as 16 x 4B rows; recommended >= 512B for backend a2a3 (L2 cache line = "
        }
      ],
      "ranks": {
        "0": {
          "startUs": 2634.28,
          "endUs": 3520.66,
          "spanUs": 886.38,
          "slices": 72,
          "hbViolations": 4
        },
        "1": {
          "startUs": 2134.94,
          "endUs": 3083.46,
          "spanUs": 948.52,
          "slices": 72,
          "hbViolations": 4
        }
      }
    },
    {
      "id": "redistribute/publish",
      "label": "Merge",
      "full": "Merge · Inverse RoPE · Pack & O-Group All-to-All",
      "short": "合并 5 个 Sparse Block 的 Partial m/l/o、执行逆 RoPE 并按 O-Group 跨卡重分布",
      "dataOwnership": "48 Block AIV 核 csa_merge_pack_publish 合并同一 Query 的 5 个 Sparse Block 并完成 Sink 归一化与逆 RoPE，按目标 O-Group（Rank 0: Groups 0–3，Rank 1: Groups 4–7）写远端窗口并触发 o_group_a2a 同步读回。",
      "source": [
        [
          "decode_csa.py",
          395,
          488
        ]
      ],
      "kernels": [
        {
          "name": "csa_merge_pack_publish",
          "coreType": "AIV",
          "blockNum": 48,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 48,
              "startUs": 3018.5,
              "endUs": 3729.16,
              "spanUs": 710.66,
              "minDurUs": 148.24,
              "medDurUs": 429.76,
              "maxDurUs": 660.42,
              "medKernelUs": 161.98,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 48,
              "startUs": 2496.5,
              "endUs": 3293.42,
              "spanUs": 796.92,
              "minDurUs": 150.26,
              "medDurUs": 539.52,
              "maxDurUs": 756.12,
              "medKernelUs": 165.52,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        },
        {
          "name": "o_group_a2a_wait",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 3732.34,
              "endUs": 4395.34,
              "spanUs": 663.0,
              "minDurUs": 663.0,
              "medDurUs": 663.0,
              "maxDurUs": 663.0,
              "medKernelUs": 662.54,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 3296.92,
              "endUs": 3298.26,
              "spanUs": 1.34,
              "minDurUs": 1.34,
              "medDurUs": 1.34,
              "maxDurUs": 1.34,
              "medKernelUs": 0.84,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "o_group_a2a_gather",
          "coreType": "AIV",
          "blockNum": 48,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 48,
              "startUs": 4401.66,
              "endUs": 4437.38,
              "spanUs": 35.72,
              "minDurUs": 21.8,
              "medDurUs": 26.74,
              "maxDurUs": 29.88,
              "medKernelUs": 26.02,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 48,
              "startUs": 3304.56,
              "endUs": 3336.64,
              "spanUs": 32.08,
              "minDurUs": 25.5,
              "medDurUs": 29.04,
              "maxDurUs": 31.66,
              "medKernelUs": 28.48,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        },
        {
          "name": "o_group_a2a_complete",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 4446.86,
              "endUs": 4454.56,
              "spanUs": 7.7,
              "minDurUs": 7.7,
              "medDurUs": 7.7,
              "maxDurUs": 7.7,
              "medKernelUs": 7.28,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 3350.08,
              "endUs": 3352.94,
              "spanUs": 2.86,
              "minDurUs": 2.86,
              "medDurUs": 2.86,
              "maxDurUs": 2.86,
              "medKernelUs": 2.48,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        }
      ],
      "internalEdges": [
        {
          "from": "csa_merge_pack_publish",
          "to": "o_group_a2a_complete",
          "source": "tensormap"
        },
        {
          "from": "csa_merge_pack_publish",
          "to": "o_group_a2a_gather",
          "source": "tensormap"
        },
        {
          "from": "csa_merge_pack_publish",
          "to": "o_group_a2a_wait",
          "source": "explicit"
        },
        {
          "from": "csa_merge_pack_publish",
          "to": "o_group_a2a_wait",
          "source": "tensormap"
        },
        {
          "from": "o_group_a2a_gather",
          "to": "o_group_a2a_complete",
          "source": "tensormap"
        },
        {
          "from": "o_group_a2a_gather",
          "to": "o_group_a2a_complete",
          "source": "explicit"
        },
        {
          "from": "o_group_a2a_wait",
          "to": "o_group_a2a_gather",
          "source": "explicit"
        }
      ],
      "perfHints": [
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_csa.py",
          "line": 245,
          "detail": "tile.load has innermost dim = 64B (tile bfloat16[32], target_memory=Vec); moves 256B as 4 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_csa.py",
          "line": 250,
          "detail": "tile.load has innermost dim = 64B (tile bfloat16[32], target_memory=Vec); moves 256B as 4 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_csa.py",
          "line": 259,
          "detail": "tile.load has innermost dim = 64B (tile bfloat16[32], target_memory=Vec); moves 256B as 4 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_csa.py",
          "line": 264,
          "detail": "tile.load has innermost dim = 64B (tile bfloat16[32], target_memory=Vec); moves 256B as 4 x 64B rows; recommended >= 512B for backend a2a3 (L2 cache l"
        }
      ],
      "ranks": {
        "0": {
          "startUs": 3018.5,
          "endUs": 4454.56,
          "spanUs": 1436.06,
          "slices": 98,
          "hbViolations": 0
        },
        "1": {
          "startUs": 2496.5,
          "endUs": 3352.94,
          "spanUs": 856.44,
          "slices": 98,
          "hbViolations": 0
        }
      }
    },
    {
      "id": "output/oproj",
      "label": "O Proj",
      "full": "Grouped O Projection · ReduceScatter · HC Post",
      "short": "8 组两段 LoRA O-Proj（INT8 量化）、跨卡 ReduceScatter 累加与残差写回",
      "dataOwnership": "每个 Rank 计算各自 4 个 O-Group 的两段投影（tp_o_a → tp_o_a_quant → tp_o_b → tp_o_b_dequant，共 8 组重复调用），通过 tp_o_b_publish / tp_o_rs_* 将部分和发回 Token Owner 做 FP32 累加，最后 hc_post 写回残差流。",
      "source": [
        [
          "decode_csa.py",
          489,
          580
        ],
        [
          "decode_o_proj.py",
          1,
          180
        ],
        [
          "hc_post.py",
          1,
          100
        ]
      ],
      "kernels": [
        {
          "name": "tp_o_a",
          "coreType": "AIC",
          "blockNum": 16,
          "taskCount": 8,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 128,
              "startUs": 4460.14,
              "endUs": 4578.3,
              "spanUs": 118.16,
              "minDurUs": 13.9,
              "medDurUs": 15.7,
              "maxDurUs": 37.76,
              "medKernelUs": 15.02,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 128,
              "startUs": 3357.96,
              "endUs": 3491.96,
              "spanUs": 134.0,
              "minDurUs": 14.14,
              "medDurUs": 15.52,
              "maxDurUs": 39.66,
              "medKernelUs": 15.12,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "tp_o_a_quant",
          "coreType": "AIV",
          "blockNum": 6,
          "taskCount": 8,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 48,
              "startUs": 4579.44,
              "endUs": 4606.24,
              "spanUs": 26.8,
              "minDurUs": 8.66,
              "medDurUs": 10.06,
              "maxDurUs": 12.6,
              "medKernelUs": 9.52,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 48,
              "startUs": 3470.84,
              "endUs": 3522.3,
              "spanUs": 51.46,
              "minDurUs": 7.98,
              "medDurUs": 10.62,
              "maxDurUs": 15.06,
              "medKernelUs": 10.12,
              "coresUsed": 40,
              "aicUsed": 0,
              "aivUsed": 40
            }
          }
        },
        {
          "name": "tp_o_b",
          "coreType": "AIC",
          "blockNum": 16,
          "taskCount": 8,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 128,
              "startUs": 4602.22,
              "endUs": 4703.66,
              "spanUs": 101.44,
              "minDurUs": 11.1,
              "medDurUs": 13.06,
              "maxDurUs": 25.04,
              "medKernelUs": 12.66,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            },
            "1": {
              "slices": 128,
              "startUs": 3492.98,
              "endUs": 3590.82,
              "spanUs": 97.84,
              "minDurUs": 10.74,
              "medDurUs": 12.9,
              "maxDurUs": 22.92,
              "medKernelUs": 12.5,
              "coresUsed": 24,
              "aicUsed": 24,
              "aivUsed": 0
            }
          }
        },
        {
          "name": "tp_o_b_dequant",
          "coreType": "AIV",
          "blockNum": 12,
          "taskCount": 2,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 24,
              "startUs": 4688.34,
              "endUs": 4745.3,
              "spanUs": 56.96,
              "minDurUs": 28.26,
              "medDurUs": 31.52,
              "maxDurUs": 33.6,
              "medKernelUs": 31.06,
              "coresUsed": 24,
              "aicUsed": 0,
              "aivUsed": 24
            },
            "1": {
              "slices": 24,
              "startUs": 3591.64,
              "endUs": 3634.32,
              "spanUs": 42.68,
              "minDurUs": 27.84,
              "medDurUs": 31.28,
              "maxDurUs": 32.06,
              "medKernelUs": 30.86,
              "coresUsed": 24,
              "aicUsed": 0,
              "aivUsed": 24
            }
          }
        },
        {
          "name": "tp_o_b_publish",
          "coreType": "AIV",
          "blockNum": 24,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 24,
              "startUs": 4751.78,
              "endUs": 4774.84,
              "spanUs": 23.06,
              "minDurUs": 8.0,
              "medDurUs": 16.32,
              "maxDurUs": 19.4,
              "medKernelUs": 15.92,
              "coresUsed": 24,
              "aicUsed": 0,
              "aivUsed": 24
            },
            "1": {
              "slices": 24,
              "startUs": 3641.24,
              "endUs": 3665.64,
              "spanUs": 24.4,
              "minDurUs": 9.96,
              "medDurUs": 15.5,
              "maxDurUs": 21.68,
              "medKernelUs": 15.16,
              "coresUsed": 24,
              "aicUsed": 0,
              "aivUsed": 24
            }
          }
        },
        {
          "name": "tp_o_rs_wait",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 4779.52,
              "endUs": 4780.8,
              "spanUs": 1.28,
              "minDurUs": 1.28,
              "medDurUs": 1.28,
              "maxDurUs": 1.28,
              "medKernelUs": 0.92,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 3668.46,
              "endUs": 3672.62,
              "spanUs": 4.16,
              "minDurUs": 4.16,
              "medDurUs": 4.16,
              "maxDurUs": 4.16,
              "medKernelUs": 3.72,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "tp_o_rs_reduce",
          "coreType": "AIV",
          "blockNum": 48,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 48,
              "startUs": 4786.58,
              "endUs": 4799.54,
              "spanUs": 12.96,
              "minDurUs": 4.92,
              "medDurUs": 6.9,
              "maxDurUs": 9.04,
              "medKernelUs": 6.56,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 48,
              "startUs": 3679.26,
              "endUs": 3691.86,
              "spanUs": 12.6,
              "minDurUs": 4.94,
              "medDurUs": 7.3,
              "maxDurUs": 10.06,
              "medKernelUs": 6.84,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        },
        {
          "name": "tp_o_rs_complete",
          "coreType": "AIV",
          "blockNum": 1,
          "taskCount": 1,
          "isComm": true,
          "ranks": {
            "0": {
              "slices": 1,
              "startUs": 4813.9,
              "endUs": 4817.76,
              "spanUs": 3.86,
              "minDurUs": 3.86,
              "medDurUs": 3.86,
              "maxDurUs": 3.86,
              "medKernelUs": 3.5,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            },
            "1": {
              "slices": 1,
              "startUs": 3704.2,
              "endUs": 3714.96,
              "spanUs": 10.76,
              "minDurUs": 10.76,
              "medDurUs": 10.76,
              "maxDurUs": 10.76,
              "medKernelUs": 10.46,
              "coresUsed": 1,
              "aicUsed": 0,
              "aivUsed": 1
            }
          }
        },
        {
          "name": "hc_post",
          "coreType": "AIV",
          "blockNum": 64,
          "taskCount": 1,
          "isComm": false,
          "ranks": {
            "0": {
              "slices": 64,
              "startUs": 4824.38,
              "endUs": 4879.82,
              "spanUs": 55.44,
              "minDurUs": 18.18,
              "medDurUs": 28.34,
              "maxDurUs": 35.82,
              "medKernelUs": 27.96,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            },
            "1": {
              "slices": 64,
              "startUs": 3721.0,
              "endUs": 3774.04,
              "spanUs": 53.04,
              "minDurUs": 17.68,
              "medDurUs": 31.8,
              "maxDurUs": 35.1,
              "medKernelUs": 31.46,
              "coresUsed": 48,
              "aicUsed": 0,
              "aivUsed": 48
            }
          }
        }
      ],
      "internalEdges": [
        {
          "from": "tp_o_a",
          "to": "tp_o_a_quant",
          "source": "tensormap"
        },
        {
          "from": "tp_o_a_quant",
          "to": "tp_o_b",
          "source": "tensormap"
        },
        {
          "from": "tp_o_b",
          "to": "tp_o_b_dequant",
          "source": "tensormap"
        },
        {
          "from": "tp_o_b_dequant",
          "to": "tp_o_b_publish",
          "source": "tensormap"
        },
        {
          "from": "tp_o_b_publish",
          "to": "tp_o_rs_complete",
          "source": "tensormap"
        },
        {
          "from": "tp_o_b_publish",
          "to": "tp_o_rs_reduce",
          "source": "tensormap"
        },
        {
          "from": "tp_o_b_publish",
          "to": "tp_o_rs_wait",
          "source": "tensormap"
        },
        {
          "from": "tp_o_b_publish",
          "to": "tp_o_rs_wait",
          "source": "explicit"
        },
        {
          "from": "tp_o_rs_complete",
          "to": "hc_post",
          "source": "tensormap"
        },
        {
          "from": "tp_o_rs_reduce",
          "to": "tp_o_rs_complete",
          "source": "explicit"
        },
        {
          "from": "tp_o_rs_reduce",
          "to": "tp_o_rs_complete",
          "source": "tensormap"
        },
        {
          "from": "tp_o_rs_wait",
          "to": "tp_o_rs_reduce",
          "source": "explicit"
        }
      ],
      "perfHints": [
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_o_proj.py",
          "line": 536,
          "detail": "tile.store has innermost dim = 32B (tile fp32[8], target_memory=Vec); recommended >= 512B for backend a2a3 (L2 cache line = 512B). Consider increasing"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_o_proj.py",
          "line": 563,
          "detail": "tile.load has innermost dim = 256B (tile int8[256], target_memory=Mat); moves 32768B as 128 x 256B rows; recommended >= 512B for backend a2a3 (L2 cach"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_o_proj.py",
          "line": 564,
          "detail": "tile.load has innermost dim = 256B (tile int8[256], target_memory=Mat); moves 65536B as 256 x 256B rows; recommended >= 512B for backend a2a3 (L2 cach"
        },
        {
          "code": "PH001",
          "rule": "TileInnermostDimGranularity",
          "file": "decode_o_proj.py",
          "line": 568,
          "detail": "tile.load has innermost dim = 256B (tile int8[256], target_memory=Mat); moves 32768B as 128 x 256B rows; recommended >= 512B for backend a2a3 (L2 cach"
        }
      ],
      "ranks": {
        "0": {
          "startUs": 4460.14,
          "endUs": 4879.82,
          "spanUs": 419.68,
          "slices": 466,
          "hbViolations": 0
        },
        "1": {
          "startUs": 3357.96,
          "endUs": 3774.04,
          "spanUs": 416.08,
          "slices": 466,
          "hbViolations": 0
        }
      }
    }
  ]
});
})(window);
