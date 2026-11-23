/* Measured 72-core (24 Core Groups) runtime trace for qk_pv & downstream csa_merge_pack_publish from _jit_l3_decode_csa_20260903_010617. */
(function (global) {
  "use strict";
  global.PTO_CSA_QK_RUNTIME_TRACE = Object.freeze({
  "artifact": "_jit_l3_decode_csa_20260903_010617",
  "pairingRule": "Group g = AIC[g] + AIV0[24 + 2*g] + AIV1[24 + 2*g + 1]",
  "compilerEvidence": {
    "passFile": "25_after_SplitVectorKernel.py",
    "dualAivDispatch": true,
    "aiv0Role": "subblock_idx == 0 · 32/32 Heads Softmax + Partial / Neutral Write",
    "aiv1Role": "subblock_idx == 1 · valid_shape = [0, 0] 仅 C2V/V2C 管道握手",
    "pipeSpec": "C2V / V2C 32 KB 双槽（2 × 16 KB）"
  },
  "ranks": {
    "0": {
      "rank": 0,
      "sourceFile": "merged_swimlane_20260903_010746.json",
      "taskId": "8589934645",
      "mergeTaskId": "8589934647",
      "windowOriginUs": 2634.28,
      "qkEndUs": 3520.66,
      "qkSpanUs": 886.38,
      "mergeEndUs": 3729.16,
      "totalSpanUs": 1094.88,
      "summary": {
        "aicMinUs": 377.56,
        "aicMedUs": 645.75,
        "aicMeanUs": 626.49,
        "aicMaxUs": 830.26,
        "aicMaxMinRatio": 2.2,
        "aicMaxMeanRatio": 1.33,
        "aiv0MinUs": 478.96,
        "aiv0MaxUs": 881.22,
        "aiv1MinUs": 375.98,
        "aiv1MaxUs": 828.82,
        "aivAllMedUs": 678.58,
        "aiv0TailMeanUs": 66.3,
        "aiv1DeltaMeanUs": -1.48,
        "qkAivSetupPct": 0.06,
        "mergeSetupTotalPct": 57.8,
        "mergeSetupFastPct": 73.2,
        "mergeSetupSlowPct": 20.5,
        "fastGroupCount": 12,
        "slowGroupCount": 12,
        "slowestGroup": 5,
        "fastestGroup": 8
      },
      "groups": [
        {
          "group": 0,
          "tier": "fast",
          "aic": {
            "core": 0,
            "startUs": 2642.4,
            "endUs": 3052.2,
            "durUs": 409.8,
            "setupUs": 0.42,
            "kernelUs": 409.38
          },
          "aiv0": {
            "core": 24,
            "startUs": 2642.36,
            "endUs": 3140.86,
            "durUs": 498.5,
            "setupUs": 0.6,
            "kernelUs": 497.9
          },
          "aiv1": {
            "core": 25,
            "startUs": 2642.36,
            "endUs": 3050.56,
            "durUs": 408.2,
            "setupUs": 0.4,
            "kernelUs": 407.8
          },
          "merge0": {
            "core": 24,
            "startUs": 3141.1,
            "endUs": 3681.78,
            "durUs": 540.68,
            "setupUs": 381.66,
            "kernelUs": 159.02
          },
          "merge1": {
            "core": 25,
            "startUs": 3050.78,
            "endUs": 3680.52,
            "durUs": 629.74,
            "setupUs": 471.8,
            "kernelUs": 157.94
          },
          "aiv0TailUs": 88.7,
          "aiv1DeltaUs": -1.6
        },
        {
          "group": 1,
          "tier": "fast",
          "aic": {
            "core": 1,
            "startUs": 2634.3,
            "endUs": 3085.92,
            "durUs": 451.62,
            "setupUs": 0.38,
            "kernelUs": 451.24
          },
          "aiv0": {
            "core": 26,
            "startUs": 2634.28,
            "endUs": 3167.78,
            "durUs": 533.5,
            "setupUs": 0.5,
            "kernelUs": 533.0
          },
          "aiv1": {
            "core": 27,
            "startUs": 2634.28,
            "endUs": 3083.9,
            "durUs": 449.62,
            "setupUs": 0.34,
            "kernelUs": 449.28
          },
          "merge0": {
            "core": 26,
            "startUs": 3167.98,
            "endUs": 3683.32,
            "durUs": 515.34,
            "setupUs": 354.88,
            "kernelUs": 160.46
          },
          "merge1": {
            "core": 27,
            "startUs": 3084.16,
            "endUs": 3679.58,
            "durUs": 595.42,
            "setupUs": 438.46,
            "kernelUs": 156.96
          },
          "aiv0TailUs": 81.88,
          "aiv1DeltaUs": -2.0
        },
        {
          "group": 2,
          "tier": "fast",
          "aic": {
            "core": 2,
            "startUs": 2642.32,
            "endUs": 3048.52,
            "durUs": 406.2,
            "setupUs": 0.4,
            "kernelUs": 405.8
          },
          "aiv0": {
            "core": 28,
            "startUs": 2642.3,
            "endUs": 3138.06,
            "durUs": 495.76,
            "setupUs": 0.3,
            "kernelUs": 495.46
          },
          "aiv1": {
            "core": 29,
            "startUs": 2642.32,
            "endUs": 3047.62,
            "durUs": 405.3,
            "setupUs": 0.56,
            "kernelUs": 404.74
          },
          "merge0": {
            "core": 28,
            "startUs": 3138.3,
            "endUs": 3634.4,
            "durUs": 496.1,
            "setupUs": 384.52,
            "kernelUs": 111.58
          },
          "merge1": {
            "core": 29,
            "startUs": 3047.86,
            "endUs": 3639.48,
            "durUs": 591.62,
            "setupUs": 474.8,
            "kernelUs": 116.82
          },
          "aiv0TailUs": 89.56,
          "aiv1DeltaUs": -0.9
        },
        {
          "group": 3,
          "tier": "slow",
          "aic": {
            "core": 3,
            "startUs": 2642.38,
            "endUs": 3463.68,
            "durUs": 821.3,
            "setupUs": 0.36,
            "kernelUs": 820.94
          },
          "aiv0": {
            "core": 30,
            "startUs": 2642.36,
            "endUs": 3510.04,
            "durUs": 867.68,
            "setupUs": 0.36,
            "kernelUs": 867.32
          },
          "aiv1": {
            "core": 31,
            "startUs": 2642.36,
            "endUs": 3462.56,
            "durUs": 820.2,
            "setupUs": 0.3,
            "kernelUs": 819.9
          },
          "merge0": {
            "core": 30,
            "startUs": 3510.36,
            "endUs": 3709.16,
            "durUs": 198.8,
            "setupUs": 12.5,
            "kernelUs": 186.3
          },
          "merge1": {
            "core": 31,
            "startUs": 3462.8,
            "endUs": 3718.14,
            "durUs": 255.34,
            "setupUs": 59.92,
            "kernelUs": 195.42
          },
          "aiv0TailUs": 46.38,
          "aiv1DeltaUs": -1.1
        },
        {
          "group": 4,
          "tier": "slow",
          "aic": {
            "core": 4,
            "startUs": 2634.28,
            "endUs": 3435.94,
            "durUs": 801.66,
            "setupUs": 0.34,
            "kernelUs": 801.32
          },
          "aiv0": {
            "core": 32,
            "startUs": 2634.3,
            "endUs": 3484.38,
            "durUs": 850.08,
            "setupUs": 0.42,
            "kernelUs": 849.66
          },
          "aiv1": {
            "core": 33,
            "startUs": 2634.32,
            "endUs": 3434.48,
            "durUs": 800.16,
            "setupUs": 0.58,
            "kernelUs": 799.58
          },
          "merge0": {
            "core": 32,
            "startUs": 3484.64,
            "endUs": 3726.0,
            "durUs": 241.36,
            "setupUs": 38.3,
            "kernelUs": 203.06
          },
          "merge1": {
            "core": 33,
            "startUs": 3434.76,
            "endUs": 3721.18,
            "durUs": 286.42,
            "setupUs": 88.14,
            "kernelUs": 198.28
          },
          "aiv0TailUs": 48.42,
          "aiv1DeltaUs": -1.5
        },
        {
          "group": 5,
          "tier": "slow",
          "aic": {
            "core": 5,
            "startUs": 2642.3,
            "endUs": 3470.34,
            "durUs": 828.04,
            "setupUs": 0.48,
            "kernelUs": 827.56
          },
          "aiv0": {
            "core": 34,
            "startUs": 2642.32,
            "endUs": 3520.66,
            "durUs": 878.34,
            "setupUs": 0.54,
            "kernelUs": 877.8
          },
          "aiv1": {
            "core": 35,
            "startUs": 2642.32,
            "endUs": 3468.98,
            "durUs": 826.66,
            "setupUs": 0.36,
            "kernelUs": 826.3
          },
          "merge0": {
            "core": 34,
            "startUs": 3521.02,
            "endUs": 3669.26,
            "durUs": 148.24,
            "setupUs": 1.9,
            "kernelUs": 146.34
          },
          "merge1": {
            "core": 35,
            "startUs": 3469.24,
            "endUs": 3656.32,
            "durUs": 187.08,
            "setupUs": 53.44,
            "kernelUs": 133.64
          },
          "aiv0TailUs": 50.3,
          "aiv1DeltaUs": -1.38
        },
        {
          "group": 6,
          "tier": "fast",
          "aic": {
            "core": 6,
            "startUs": 2642.36,
            "endUs": 3027.7,
            "durUs": 385.34,
            "setupUs": 0.6,
            "kernelUs": 384.74
          },
          "aiv0": {
            "core": 36,
            "startUs": 2642.38,
            "endUs": 3125.34,
            "durUs": 482.96,
            "setupUs": 0.5,
            "kernelUs": 482.46
          },
          "aiv1": {
            "core": 37,
            "startUs": 2642.38,
            "endUs": 3025.3,
            "durUs": 382.92,
            "setupUs": 0.34,
            "kernelUs": 382.58
          },
          "merge0": {
            "core": 36,
            "startUs": 3125.56,
            "endUs": 3691.64,
            "durUs": 566.08,
            "setupUs": 397.32,
            "kernelUs": 168.76
          },
          "merge1": {
            "core": 37,
            "startUs": 3025.54,
            "endUs": 3685.96,
            "durUs": 660.42,
            "setupUs": 497.14,
            "kernelUs": 163.28
          },
          "aiv0TailUs": 97.62,
          "aiv1DeltaUs": -2.42
        },
        {
          "group": 7,
          "tier": "fast",
          "aic": {
            "core": 7,
            "startUs": 2634.32,
            "endUs": 3090.96,
            "durUs": 456.64,
            "setupUs": 0.4,
            "kernelUs": 456.24
          },
          "aiv0": {
            "core": 38,
            "startUs": 2634.32,
            "endUs": 3173.18,
            "durUs": 538.86,
            "setupUs": 0.32,
            "kernelUs": 538.54
          },
          "aiv1": {
            "core": 39,
            "startUs": 2634.3,
            "endUs": 3088.9,
            "durUs": 454.6,
            "setupUs": 0.4,
            "kernelUs": 454.2
          },
          "merge0": {
            "core": 38,
            "startUs": 3173.46,
            "endUs": 3688.04,
            "durUs": 514.58,
            "setupUs": 349.4,
            "kernelUs": 165.18
          },
          "merge1": {
            "core": 39,
            "startUs": 3089.18,
            "endUs": 3684.68,
            "durUs": 595.5,
            "setupUs": 433.52,
            "kernelUs": 161.98
          },
          "aiv0TailUs": 82.22,
          "aiv1DeltaUs": -2.04
        },
        {
          "group": 8,
          "tier": "fast",
          "aic": {
            "core": 8,
            "startUs": 2642.32,
            "endUs": 3019.88,
            "durUs": 377.56,
            "setupUs": 0.32,
            "kernelUs": 377.24
          },
          "aiv0": {
            "core": 40,
            "startUs": 2642.3,
            "endUs": 3121.26,
            "durUs": 478.96,
            "setupUs": 0.36,
            "kernelUs": 478.6
          },
          "aiv1": {
            "core": 41,
            "startUs": 2642.3,
            "endUs": 3018.28,
            "durUs": 375.98,
            "setupUs": 0.4,
            "kernelUs": 375.58
          },
          "merge0": {
            "core": 40,
            "startUs": 3121.78,
            "endUs": 3634.1,
            "durUs": 512.32,
            "setupUs": 401.12,
            "kernelUs": 111.2
          },
          "merge1": {
            "core": 41,
            "startUs": 3018.5,
            "endUs": 3634.6,
            "durUs": 616.1,
            "setupUs": 504.44,
            "kernelUs": 111.66
          },
          "aiv0TailUs": 101.4,
          "aiv1DeltaUs": -1.58
        },
        {
          "group": 9,
          "tier": "slow",
          "aic": {
            "core": 9,
            "startUs": 2642.38,
            "endUs": 3455.06,
            "durUs": 812.68,
            "setupUs": 0.32,
            "kernelUs": 812.36
          },
          "aiv0": {
            "core": 42,
            "startUs": 2642.38,
            "endUs": 3503.76,
            "durUs": 861.38,
            "setupUs": 0.34,
            "kernelUs": 861.04
          },
          "aiv1": {
            "core": 43,
            "startUs": 2642.4,
            "endUs": 3454.1,
            "durUs": 811.7,
            "setupUs": 0.36,
            "kernelUs": 811.34
          },
          "merge0": {
            "core": 42,
            "startUs": 3504.02,
            "endUs": 3712.58,
            "durUs": 208.56,
            "setupUs": 18.9,
            "kernelUs": 189.66
          },
          "merge1": {
            "core": 43,
            "startUs": 3454.34,
            "endUs": 3727.84,
            "durUs": 273.5,
            "setupUs": 68.36,
            "kernelUs": 205.14
          },
          "aiv0TailUs": 48.7,
          "aiv1DeltaUs": -0.98
        },
        {
          "group": 10,
          "tier": "slow",
          "aic": {
            "core": 10,
            "startUs": 2634.32,
            "endUs": 3460.94,
            "durUs": 826.62,
            "setupUs": 0.38,
            "kernelUs": 826.24
          },
          "aiv0": {
            "core": 44,
            "startUs": 2634.32,
            "endUs": 3510.74,
            "durUs": 876.42,
            "setupUs": 0.32,
            "kernelUs": 876.1
          },
          "aiv1": {
            "core": 45,
            "startUs": 2634.36,
            "endUs": 3460.04,
            "durUs": 825.68,
            "setupUs": 0.34,
            "kernelUs": 825.34
          },
          "merge0": {
            "core": 44,
            "startUs": 3511.02,
            "endUs": 3722.3,
            "durUs": 211.28,
            "setupUs": 11.88,
            "kernelUs": 199.4
          },
          "merge1": {
            "core": 45,
            "startUs": 3460.32,
            "endUs": 3721.52,
            "durUs": 261.2,
            "setupUs": 62.4,
            "kernelUs": 198.8
          },
          "aiv0TailUs": 49.8,
          "aiv1DeltaUs": -0.94
        },
        {
          "group": 11,
          "tier": "slow",
          "aic": {
            "core": 11,
            "startUs": 2642.32,
            "endUs": 3451.24,
            "durUs": 808.92,
            "setupUs": 0.3,
            "kernelUs": 808.62
          },
          "aiv0": {
            "core": 46,
            "startUs": 2642.32,
            "endUs": 3501.3,
            "durUs": 858.98,
            "setupUs": 0.3,
            "kernelUs": 858.68
          },
          "aiv1": {
            "core": 47,
            "startUs": 2642.32,
            "endUs": 3450.3,
            "durUs": 807.98,
            "setupUs": 0.4,
            "kernelUs": 807.58
          },
          "merge0": {
            "core": 46,
            "startUs": 3501.66,
            "endUs": 3669.92,
            "durUs": 168.26,
            "setupUs": 21.24,
            "kernelUs": 147.02
          },
          "merge1": {
            "core": 47,
            "startUs": 3450.58,
            "endUs": 3659.8,
            "durUs": 209.22,
            "setupUs": 72.16,
            "kernelUs": 137.06
          },
          "aiv0TailUs": 50.06,
          "aiv1DeltaUs": -0.94
        },
        {
          "group": 12,
          "tier": "fast",
          "aic": {
            "core": 12,
            "startUs": 2642.4,
            "endUs": 3084.66,
            "durUs": 442.26,
            "setupUs": 0.3,
            "kernelUs": 441.96
          },
          "aiv0": {
            "core": 48,
            "startUs": 2642.44,
            "endUs": 3162.8,
            "durUs": 520.36,
            "setupUs": 0.5,
            "kernelUs": 519.86
          },
          "aiv1": {
            "core": 49,
            "startUs": 2642.42,
            "endUs": 3082.58,
            "durUs": 440.16,
            "setupUs": 0.48,
            "kernelUs": 439.68
          },
          "merge0": {
            "core": 48,
            "startUs": 3163.0,
            "endUs": 3692.9,
            "durUs": 529.9,
            "setupUs": 360.0,
            "kernelUs": 169.9
          },
          "merge1": {
            "core": 49,
            "startUs": 3082.82,
            "endUs": 3682.98,
            "durUs": 600.16,
            "setupUs": 440.1,
            "kernelUs": 160.06
          },
          "aiv0TailUs": 78.1,
          "aiv1DeltaUs": -2.1
        },
        {
          "group": 13,
          "tier": "fast",
          "aic": {
            "core": 13,
            "startUs": 2634.36,
            "endUs": 3142.64,
            "durUs": 508.28,
            "setupUs": 0.4,
            "kernelUs": 507.88
          },
          "aiv0": {
            "core": 50,
            "startUs": 2634.34,
            "endUs": 3209.16,
            "durUs": 574.82,
            "setupUs": 0.4,
            "kernelUs": 574.42
          },
          "aiv1": {
            "core": 51,
            "startUs": 2634.34,
            "endUs": 3140.94,
            "durUs": 506.6,
            "setupUs": 0.74,
            "kernelUs": 505.86
          },
          "merge0": {
            "core": 50,
            "startUs": 3209.42,
            "endUs": 3686.84,
            "durUs": 477.42,
            "setupUs": 313.32,
            "kernelUs": 164.1
          },
          "merge1": {
            "core": 51,
            "startUs": 3141.18,
            "endUs": 3685.82,
            "durUs": 544.64,
            "setupUs": 381.76,
            "kernelUs": 162.88
          },
          "aiv0TailUs": 66.54,
          "aiv1DeltaUs": -1.68
        },
        {
          "group": 14,
          "tier": "fast",
          "aic": {
            "core": 14,
            "startUs": 2642.32,
            "endUs": 3135.4,
            "durUs": 493.08,
            "setupUs": 0.26,
            "kernelUs": 492.82
          },
          "aiv0": {
            "core": 52,
            "startUs": 2642.32,
            "endUs": 3205.76,
            "durUs": 563.44,
            "setupUs": 0.26,
            "kernelUs": 563.18
          },
          "aiv1": {
            "core": 53,
            "startUs": 2642.32,
            "endUs": 3134.3,
            "durUs": 491.98,
            "setupUs": 0.42,
            "kernelUs": 491.56
          },
          "merge0": {
            "core": 52,
            "startUs": 3206.0,
            "endUs": 3635.76,
            "durUs": 429.76,
            "setupUs": 316.98,
            "kernelUs": 112.78
          },
          "merge1": {
            "core": 53,
            "startUs": 3134.56,
            "endUs": 3642.28,
            "durUs": 507.72,
            "setupUs": 388.2,
            "kernelUs": 119.52
          },
          "aiv0TailUs": 70.36,
          "aiv1DeltaUs": -1.1
        },
        {
          "group": 15,
          "tier": "slow",
          "aic": {
            "core": 15,
            "startUs": 2642.44,
            "endUs": 3426.92,
            "durUs": 784.48,
            "setupUs": 0.42,
            "kernelUs": 784.06
          },
          "aiv0": {
            "core": 54,
            "startUs": 2642.44,
            "endUs": 3476.28,
            "durUs": 833.84,
            "setupUs": 0.44,
            "kernelUs": 833.4
          },
          "aiv1": {
            "core": 55,
            "startUs": 2642.44,
            "endUs": 3425.88,
            "durUs": 783.44,
            "setupUs": 0.36,
            "kernelUs": 783.08
          },
          "merge0": {
            "core": 54,
            "startUs": 3476.52,
            "endUs": 3716.74,
            "durUs": 240.22,
            "setupUs": 46.42,
            "kernelUs": 193.8
          },
          "merge1": {
            "core": 55,
            "startUs": 3426.1,
            "endUs": 3729.16,
            "durUs": 303.06,
            "setupUs": 96.66,
            "kernelUs": 206.4
          },
          "aiv0TailUs": 49.36,
          "aiv1DeltaUs": -1.04
        },
        {
          "group": 16,
          "tier": "slow",
          "aic": {
            "core": 16,
            "startUs": 2634.36,
            "endUs": 3417.58,
            "durUs": 783.22,
            "setupUs": 0.36,
            "kernelUs": 782.86
          },
          "aiv0": {
            "core": 56,
            "startUs": 2634.38,
            "endUs": 3468.78,
            "durUs": 834.4,
            "setupUs": 0.34,
            "kernelUs": 834.06
          },
          "aiv1": {
            "core": 57,
            "startUs": 2634.4,
            "endUs": 3416.74,
            "durUs": 782.34,
            "setupUs": 0.6,
            "kernelUs": 781.74
          },
          "merge0": {
            "core": 56,
            "startUs": 3469.02,
            "endUs": 3727.0,
            "durUs": 257.98,
            "setupUs": 53.92,
            "kernelUs": 204.06
          },
          "merge1": {
            "core": 57,
            "startUs": 3417.0,
            "endUs": 3724.3,
            "durUs": 307.3,
            "setupUs": 105.98,
            "kernelUs": 201.32
          },
          "aiv0TailUs": 51.18,
          "aiv1DeltaUs": -0.88
        },
        {
          "group": 17,
          "tier": "slow",
          "aic": {
            "core": 17,
            "startUs": 2642.34,
            "endUs": 3449.76,
            "durUs": 807.42,
            "setupUs": 0.32,
            "kernelUs": 807.1
          },
          "aiv0": {
            "core": 58,
            "startUs": 2642.34,
            "endUs": 3499.98,
            "durUs": 857.64,
            "setupUs": 0.54,
            "kernelUs": 857.1
          },
          "aiv1": {
            "core": 59,
            "startUs": 2642.34,
            "endUs": 3448.54,
            "durUs": 806.2,
            "setupUs": 0.36,
            "kernelUs": 805.84
          },
          "merge0": {
            "core": 58,
            "startUs": 3500.2,
            "endUs": 3671.4,
            "durUs": 171.2,
            "setupUs": 22.8,
            "kernelUs": 148.4
          },
          "merge1": {
            "core": 59,
            "startUs": 3448.82,
            "endUs": 3650.38,
            "durUs": 201.56,
            "setupUs": 73.98,
            "kernelUs": 127.58
          },
          "aiv0TailUs": 50.22,
          "aiv1DeltaUs": -1.22
        },
        {
          "group": 18,
          "tier": "fast",
          "aic": {
            "core": 18,
            "startUs": 2642.46,
            "endUs": 3120.2,
            "durUs": 477.74,
            "setupUs": 0.44,
            "kernelUs": 477.3
          },
          "aiv0": {
            "core": 60,
            "startUs": 2642.48,
            "endUs": 3193.44,
            "durUs": 550.96,
            "setupUs": 0.44,
            "kernelUs": 550.52
          },
          "aiv1": {
            "core": 61,
            "startUs": 2642.48,
            "endUs": 3118.62,
            "durUs": 476.14,
            "setupUs": 0.34,
            "kernelUs": 475.8
          },
          "merge0": {
            "core": 60,
            "startUs": 3193.76,
            "endUs": 3692.22,
            "durUs": 498.46,
            "setupUs": 329.22,
            "kernelUs": 169.24
          },
          "merge1": {
            "core": 61,
            "startUs": 3118.84,
            "endUs": 3684.0,
            "durUs": 565.16,
            "setupUs": 403.94,
            "kernelUs": 161.22
          },
          "aiv0TailUs": 73.22,
          "aiv1DeltaUs": -1.6
        },
        {
          "group": 19,
          "tier": "fast",
          "aic": {
            "core": 19,
            "startUs": 2634.42,
            "endUs": 3057.76,
            "durUs": 423.34,
            "setupUs": 0.34,
            "kernelUs": 423.0
          },
          "aiv0": {
            "core": 62,
            "startUs": 2634.4,
            "endUs": 3146.84,
            "durUs": 512.44,
            "setupUs": 0.34,
            "kernelUs": 512.1
          },
          "aiv1": {
            "core": 63,
            "startUs": 2634.4,
            "endUs": 3055.02,
            "durUs": 420.62,
            "setupUs": 0.34,
            "kernelUs": 420.28
          },
          "merge0": {
            "core": 62,
            "startUs": 3147.08,
            "endUs": 3683.12,
            "durUs": 536.04,
            "setupUs": 375.9,
            "kernelUs": 160.14
          },
          "merge1": {
            "core": 63,
            "startUs": 3055.26,
            "endUs": 3682.7,
            "durUs": 627.44,
            "setupUs": 467.58,
            "kernelUs": 159.86
          },
          "aiv0TailUs": 89.1,
          "aiv1DeltaUs": -2.72
        },
        {
          "group": 20,
          "tier": "fast",
          "aic": {
            "core": 20,
            "startUs": 2642.34,
            "endUs": 3097.96,
            "durUs": 455.62,
            "setupUs": 0.42,
            "kernelUs": 455.2
          },
          "aiv0": {
            "core": 64,
            "startUs": 2642.34,
            "endUs": 3177.2,
            "durUs": 534.86,
            "setupUs": 0.34,
            "kernelUs": 534.52
          },
          "aiv1": {
            "core": 65,
            "startUs": 2642.36,
            "endUs": 3096.34,
            "durUs": 453.98,
            "setupUs": 0.48,
            "kernelUs": 453.5
          },
          "merge0": {
            "core": 64,
            "startUs": 3177.64,
            "endUs": 3638.38,
            "durUs": 460.74,
            "setupUs": 345.52,
            "kernelUs": 115.22
          },
          "merge1": {
            "core": 65,
            "startUs": 3096.56,
            "endUs": 3645.44,
            "durUs": 548.88,
            "setupUs": 426.64,
            "kernelUs": 122.24
          },
          "aiv0TailUs": 79.24,
          "aiv1DeltaUs": -1.64
        },
        {
          "group": 21,
          "tier": "slow",
          "aic": {
            "core": 21,
            "startUs": 2642.48,
            "endUs": 3463.9,
            "durUs": 821.42,
            "setupUs": 0.38,
            "kernelUs": 821.04
          },
          "aiv0": {
            "core": 66,
            "startUs": 2642.52,
            "endUs": 3510.4,
            "durUs": 867.88,
            "setupUs": 0.58,
            "kernelUs": 867.3
          },
          "aiv1": {
            "core": 67,
            "startUs": 2642.54,
            "endUs": 3462.4,
            "durUs": 819.86,
            "setupUs": 0.36,
            "kernelUs": 819.5
          },
          "merge0": {
            "core": 66,
            "startUs": 3510.62,
            "endUs": 3719.76,
            "durUs": 209.14,
            "setupUs": 12.54,
            "kernelUs": 196.6
          },
          "merge1": {
            "core": 67,
            "startUs": 3462.62,
            "endUs": 3724.94,
            "durUs": 262.32,
            "setupUs": 60.86,
            "kernelUs": 201.46
          },
          "aiv0TailUs": 46.46,
          "aiv1DeltaUs": -1.56
        },
        {
          "group": 22,
          "tier": "slow",
          "aic": {
            "core": 22,
            "startUs": 2634.4,
            "endUs": 3464.66,
            "durUs": 830.26,
            "setupUs": 0.36,
            "kernelUs": 829.9
          },
          "aiv0": {
            "core": 68,
            "startUs": 2634.42,
            "endUs": 3515.64,
            "durUs": 881.22,
            "setupUs": 0.36,
            "kernelUs": 880.86
          },
          "aiv1": {
            "core": 69,
            "startUs": 2634.42,
            "endUs": 3463.24,
            "durUs": 828.82,
            "setupUs": 0.38,
            "kernelUs": 828.44
          },
          "merge0": {
            "core": 68,
            "startUs": 3515.9,
            "endUs": 3728.52,
            "durUs": 212.62,
            "setupUs": 7.62,
            "kernelUs": 205.0
          },
          "merge1": {
            "core": 69,
            "startUs": 3463.5,
            "endUs": 3727.68,
            "durUs": 264.18,
            "setupUs": 59.88,
            "kernelUs": 204.3
          },
          "aiv0TailUs": 50.96,
          "aiv1DeltaUs": -1.44
        },
        {
          "group": 23,
          "tier": "slow",
          "aic": {
            "core": 23,
            "startUs": 2642.48,
            "endUs": 3464.74,
            "durUs": 822.26,
            "setupUs": 0.48,
            "kernelUs": 821.78
          },
          "aiv0": {
            "core": 70,
            "startUs": 2642.5,
            "endUs": 3515.1,
            "durUs": 872.6,
            "setupUs": 0.36,
            "kernelUs": 872.24
          },
          "aiv1": {
            "core": 71,
            "startUs": 2642.54,
            "endUs": 3463.6,
            "durUs": 821.06,
            "setupUs": 0.36,
            "kernelUs": 820.7
          },
          "merge0": {
            "core": 70,
            "startUs": 3515.32,
            "endUs": 3673.36,
            "durUs": 158.04,
            "setupUs": 7.88,
            "kernelUs": 150.16
          },
          "merge1": {
            "core": 71,
            "startUs": 3464.08,
            "endUs": 3663.86,
            "durUs": 199.78,
            "setupUs": 59.16,
            "kernelUs": 140.62
          },
          "aiv0TailUs": 50.34,
          "aiv1DeltaUs": -1.2
        }
      ]
    },
    "1": {
      "rank": 1,
      "sourceFile": "merged_swimlane_20260903_010747.json",
      "taskId": "8589934645",
      "mergeTaskId": "8589934647",
      "windowOriginUs": 2134.94,
      "qkEndUs": 3083.46,
      "qkSpanUs": 948.52,
      "mergeEndUs": 3293.42,
      "totalSpanUs": 1158.48,
      "summary": {
        "aicMinUs": 362.74,
        "aicMedUs": 644.01,
        "aicMeanUs": 632.9,
        "aicMaxUs": 896.0,
        "aicMaxMinRatio": 2.47,
        "aicMaxMeanRatio": 1.42,
        "aiv0MinUs": 452.32,
        "aiv0MaxUs": 948.46,
        "aiv1MinUs": 360.38,
        "aiv1MaxUs": 894.6,
        "aivAllMedUs": 686.72,
        "aiv0TailMeanUs": 72.3,
        "aiv1DeltaMeanUs": -1.46,
        "qkAivSetupPct": 0.06,
        "mergeSetupTotalPct": 63.1,
        "mergeSetupFastPct": 77.5,
        "mergeSetupSlowPct": 21.0,
        "fastGroupCount": 12,
        "slowGroupCount": 12,
        "slowestGroup": 13,
        "fastestGroup": 16
      },
      "groups": [
        {
          "group": 0,
          "tier": "slow",
          "aic": {
            "core": 0,
            "startUs": 2145.38,
            "endUs": 3002.84,
            "durUs": 857.46,
            "setupUs": 0.38,
            "kernelUs": 857.08
          },
          "aiv0": {
            "core": 24,
            "startUs": 2145.38,
            "endUs": 3058.26,
            "durUs": 912.88,
            "setupUs": 0.28,
            "kernelUs": 912.6
          },
          "aiv1": {
            "core": 25,
            "startUs": 2145.4,
            "endUs": 3000.84,
            "durUs": 855.44,
            "setupUs": 0.3,
            "kernelUs": 855.14
          },
          "merge0": {
            "core": 24,
            "startUs": 3058.5,
            "endUs": 3227.7,
            "durUs": 169.2,
            "setupUs": 26.82,
            "kernelUs": 142.38
          },
          "merge1": {
            "core": 25,
            "startUs": 3001.06,
            "endUs": 3233.28,
            "durUs": 232.22,
            "setupUs": 84.46,
            "kernelUs": 147.76
          },
          "aiv0TailUs": 55.42,
          "aiv1DeltaUs": -2.02
        },
        {
          "group": 1,
          "tier": "slow",
          "aic": {
            "core": 1,
            "startUs": 2134.96,
            "endUs": 2976.52,
            "durUs": 841.56,
            "setupUs": 0.4,
            "kernelUs": 841.16
          },
          "aiv0": {
            "core": 26,
            "startUs": 2134.94,
            "endUs": 3043.88,
            "durUs": 908.94,
            "setupUs": 0.34,
            "kernelUs": 908.6
          },
          "aiv1": {
            "core": 27,
            "startUs": 2134.94,
            "endUs": 2974.74,
            "durUs": 839.8,
            "setupUs": 0.52,
            "kernelUs": 839.28
          },
          "merge0": {
            "core": 26,
            "startUs": 3044.1,
            "endUs": 3290.06,
            "durUs": 245.96,
            "setupUs": 41.42,
            "kernelUs": 204.54
          },
          "merge1": {
            "core": 27,
            "startUs": 2974.98,
            "endUs": 3285.26,
            "durUs": 310.28,
            "setupUs": 110.52,
            "kernelUs": 199.76
          },
          "aiv0TailUs": 67.38,
          "aiv1DeltaUs": -1.76
        },
        {
          "group": 2,
          "tier": "slow",
          "aic": {
            "core": 2,
            "startUs": 2136.02,
            "endUs": 2971.46,
            "durUs": 835.44,
            "setupUs": 0.42,
            "kernelUs": 835.02
          },
          "aiv0": {
            "core": 28,
            "startUs": 2136.0,
            "endUs": 3044.42,
            "durUs": 908.42,
            "setupUs": 0.24,
            "kernelUs": 908.18
          },
          "aiv1": {
            "core": 29,
            "startUs": 2136.0,
            "endUs": 2970.44,
            "durUs": 834.44,
            "setupUs": 0.36,
            "kernelUs": 834.08
          },
          "merge0": {
            "core": 28,
            "startUs": 3044.62,
            "endUs": 3260.12,
            "durUs": 215.5,
            "setupUs": 40.68,
            "kernelUs": 174.82
          },
          "merge1": {
            "core": 29,
            "startUs": 2970.68,
            "endUs": 3279.52,
            "durUs": 308.84,
            "setupUs": 114.66,
            "kernelUs": 194.18
          },
          "aiv0TailUs": 72.98,
          "aiv1DeltaUs": -1.0
        },
        {
          "group": 3,
          "tier": "fast",
          "aic": {
            "core": 3,
            "startUs": 2145.36,
            "endUs": 2570.36,
            "durUs": 425.0,
            "setupUs": 0.24,
            "kernelUs": 424.76
          },
          "aiv0": {
            "core": 30,
            "startUs": 2145.36,
            "endUs": 2654.46,
            "durUs": 509.1,
            "setupUs": 0.36,
            "kernelUs": 508.74
          },
          "aiv1": {
            "core": 31,
            "startUs": 2145.36,
            "endUs": 2568.8,
            "durUs": 423.44,
            "setupUs": 0.26,
            "kernelUs": 423.18
          },
          "merge0": {
            "core": 30,
            "startUs": 2654.78,
            "endUs": 3197.1,
            "durUs": 542.32,
            "setupUs": 430.76,
            "kernelUs": 111.56
          },
          "merge1": {
            "core": 31,
            "startUs": 2569.18,
            "endUs": 3198.78,
            "durUs": 629.6,
            "setupUs": 516.14,
            "kernelUs": 113.46
          },
          "aiv0TailUs": 84.1,
          "aiv1DeltaUs": -1.56
        },
        {
          "group": 4,
          "tier": "fast",
          "aic": {
            "core": 4,
            "startUs": 2134.96,
            "endUs": 2520.06,
            "durUs": 385.1,
            "setupUs": 0.38,
            "kernelUs": 384.72
          },
          "aiv0": {
            "core": 32,
            "startUs": 2134.94,
            "endUs": 2612.06,
            "durUs": 477.12,
            "setupUs": 0.34,
            "kernelUs": 476.78
          },
          "aiv1": {
            "core": 33,
            "startUs": 2134.94,
            "endUs": 2519.32,
            "durUs": 384.38,
            "setupUs": 0.52,
            "kernelUs": 383.86
          },
          "merge0": {
            "core": 32,
            "startUs": 2612.34,
            "endUs": 3251.36,
            "durUs": 639.02,
            "setupUs": 473.18,
            "kernelUs": 165.84
          },
          "merge1": {
            "core": 33,
            "startUs": 2519.54,
            "endUs": 3249.04,
            "durUs": 729.5,
            "setupUs": 566.02,
            "kernelUs": 163.48
          },
          "aiv0TailUs": 92.02,
          "aiv1DeltaUs": -0.72
        },
        {
          "group": 5,
          "tier": "fast",
          "aic": {
            "core": 5,
            "startUs": 2136.02,
            "endUs": 2514.5,
            "durUs": 378.48,
            "setupUs": 0.4,
            "kernelUs": 378.08
          },
          "aiv0": {
            "core": 34,
            "startUs": 2136.0,
            "endUs": 2611.6,
            "durUs": 475.6,
            "setupUs": 0.56,
            "kernelUs": 475.04
          },
          "aiv1": {
            "core": 35,
            "startUs": 2136.0,
            "endUs": 2512.82,
            "durUs": 376.82,
            "setupUs": 0.38,
            "kernelUs": 376.44
          },
          "merge0": {
            "core": 34,
            "startUs": 2611.86,
            "endUs": 3253.42,
            "durUs": 641.56,
            "setupUs": 473.68,
            "kernelUs": 167.88
          },
          "merge1": {
            "core": 35,
            "startUs": 2513.06,
            "endUs": 3250.26,
            "durUs": 737.2,
            "setupUs": 572.68,
            "kernelUs": 164.52
          },
          "aiv0TailUs": 97.12,
          "aiv1DeltaUs": -1.66
        },
        {
          "group": 6,
          "tier": "slow",
          "aic": {
            "core": 6,
            "startUs": 2145.36,
            "endUs": 3018.58,
            "durUs": 873.22,
            "setupUs": 0.42,
            "kernelUs": 872.8
          },
          "aiv0": {
            "core": 36,
            "startUs": 2145.4,
            "endUs": 3070.4,
            "durUs": 925.0,
            "setupUs": 0.28,
            "kernelUs": 924.72
          },
          "aiv1": {
            "core": 37,
            "startUs": 2145.42,
            "endUs": 3017.8,
            "durUs": 872.38,
            "setupUs": 0.4,
            "kernelUs": 871.98
          },
          "merge0": {
            "core": 36,
            "startUs": 3070.76,
            "endUs": 3221.02,
            "durUs": 150.26,
            "setupUs": 15.02,
            "kernelUs": 135.24
          },
          "merge1": {
            "core": 37,
            "startUs": 3018.3,
            "endUs": 3232.16,
            "durUs": 213.86,
            "setupUs": 67.48,
            "kernelUs": 146.38
          },
          "aiv0TailUs": 51.78,
          "aiv1DeltaUs": -0.84
        },
        {
          "group": 7,
          "tier": "slow",
          "aic": {
            "core": 7,
            "startUs": 2134.98,
            "endUs": 3030.72,
            "durUs": 895.74,
            "setupUs": 0.4,
            "kernelUs": 895.34
          },
          "aiv0": {
            "core": 38,
            "startUs": 2134.94,
            "endUs": 3080.64,
            "durUs": 945.7,
            "setupUs": 0.38,
            "kernelUs": 945.32
          },
          "aiv1": {
            "core": 39,
            "startUs": 2134.94,
            "endUs": 3029.54,
            "durUs": 894.6,
            "setupUs": 0.36,
            "kernelUs": 894.24
          },
          "merge0": {
            "core": 38,
            "startUs": 3081.18,
            "endUs": 3277.9,
            "durUs": 196.72,
            "setupUs": 4.4,
            "kernelUs": 192.32
          },
          "merge1": {
            "core": 39,
            "startUs": 3029.86,
            "endUs": 3271.46,
            "durUs": 241.6,
            "setupUs": 55.72,
            "kernelUs": 185.88
          },
          "aiv0TailUs": 49.96,
          "aiv1DeltaUs": -1.14
        },
        {
          "group": 8,
          "tier": "slow",
          "aic": {
            "core": 8,
            "startUs": 2136.04,
            "endUs": 3012.36,
            "durUs": 876.32,
            "setupUs": 0.38,
            "kernelUs": 875.94
          },
          "aiv0": {
            "core": 40,
            "startUs": 2136.0,
            "endUs": 3063.64,
            "durUs": 927.64,
            "setupUs": 0.38,
            "kernelUs": 927.26
          },
          "aiv1": {
            "core": 41,
            "startUs": 2136.02,
            "endUs": 3011.6,
            "durUs": 875.58,
            "setupUs": 0.4,
            "kernelUs": 875.18
          },
          "merge0": {
            "core": 40,
            "startUs": 3063.88,
            "endUs": 3275.32,
            "durUs": 211.44,
            "setupUs": 21.9,
            "kernelUs": 189.54
          },
          "merge1": {
            "core": 41,
            "startUs": 3011.84,
            "endUs": 3287.36,
            "durUs": 275.52,
            "setupUs": 73.98,
            "kernelUs": 201.54
          },
          "aiv0TailUs": 51.32,
          "aiv1DeltaUs": -0.74
        },
        {
          "group": 9,
          "tier": "fast",
          "aic": {
            "core": 9,
            "startUs": 2145.38,
            "endUs": 2512.06,
            "durUs": 366.68,
            "setupUs": 0.32,
            "kernelUs": 366.36
          },
          "aiv0": {
            "core": 42,
            "startUs": 2145.4,
            "endUs": 2602.4,
            "durUs": 457.0,
            "setupUs": 0.54,
            "kernelUs": 456.46
          },
          "aiv1": {
            "core": 43,
            "startUs": 2145.4,
            "endUs": 2510.1,
            "durUs": 364.7,
            "setupUs": 0.58,
            "kernelUs": 364.12
          },
          "merge0": {
            "core": 42,
            "startUs": 2602.62,
            "endUs": 3199.46,
            "durUs": 596.84,
            "setupUs": 483.12,
            "kernelUs": 113.72
          },
          "merge1": {
            "core": 43,
            "startUs": 2510.32,
            "endUs": 3196.02,
            "durUs": 685.7,
            "setupUs": 575.68,
            "kernelUs": 110.02
          },
          "aiv0TailUs": 90.32,
          "aiv1DeltaUs": -1.98
        },
        {
          "group": 10,
          "tier": "fast",
          "aic": {
            "core": 10,
            "startUs": 2134.96,
            "endUs": 2516.68,
            "durUs": 381.72,
            "setupUs": 0.38,
            "kernelUs": 381.34
          },
          "aiv0": {
            "core": 44,
            "startUs": 2134.94,
            "endUs": 2611.8,
            "durUs": 476.86,
            "setupUs": 0.42,
            "kernelUs": 476.44
          },
          "aiv1": {
            "core": 45,
            "startUs": 2134.96,
            "endUs": 2515.62,
            "durUs": 380.66,
            "setupUs": 0.28,
            "kernelUs": 380.38
          },
          "merge0": {
            "core": 44,
            "startUs": 2612.02,
            "endUs": 3248.38,
            "durUs": 636.36,
            "setupUs": 473.94,
            "kernelUs": 162.42
          },
          "merge1": {
            "core": 45,
            "startUs": 2515.88,
            "endUs": 3247.92,
            "durUs": 732.04,
            "setupUs": 569.84,
            "kernelUs": 162.2
          },
          "aiv0TailUs": 95.14,
          "aiv1DeltaUs": -1.06
        },
        {
          "group": 11,
          "tier": "fast",
          "aic": {
            "core": 11,
            "startUs": 2136.02,
            "endUs": 2509.14,
            "durUs": 373.12,
            "setupUs": 0.4,
            "kernelUs": 372.72
          },
          "aiv0": {
            "core": 46,
            "startUs": 2136.0,
            "endUs": 2606.64,
            "durUs": 470.64,
            "setupUs": 0.36,
            "kernelUs": 470.28
          },
          "aiv1": {
            "core": 47,
            "startUs": 2136.0,
            "endUs": 2508.78,
            "durUs": 372.78,
            "setupUs": 0.36,
            "kernelUs": 372.42
          },
          "merge0": {
            "core": 46,
            "startUs": 2606.92,
            "endUs": 3249.42,
            "durUs": 642.5,
            "setupUs": 478.98,
            "kernelUs": 163.52
          },
          "merge1": {
            "core": 47,
            "startUs": 2509.0,
            "endUs": 3253.22,
            "durUs": 744.22,
            "setupUs": 576.74,
            "kernelUs": 167.48
          },
          "aiv0TailUs": 97.52,
          "aiv1DeltaUs": -0.34
        },
        {
          "group": 12,
          "tier": "slow",
          "aic": {
            "core": 12,
            "startUs": 2145.38,
            "endUs": 3013.1,
            "durUs": 867.72,
            "setupUs": 0.2,
            "kernelUs": 867.52
          },
          "aiv0": {
            "core": 48,
            "startUs": 2145.44,
            "endUs": 3067.6,
            "durUs": 922.16,
            "setupUs": 0.32,
            "kernelUs": 921.84
          },
          "aiv1": {
            "core": 49,
            "startUs": 2145.44,
            "endUs": 3012.52,
            "durUs": 867.08,
            "setupUs": 0.34,
            "kernelUs": 866.74
          },
          "merge0": {
            "core": 48,
            "startUs": 3067.84,
            "endUs": 3233.98,
            "durUs": 166.14,
            "setupUs": 18.12,
            "kernelUs": 148.02
          },
          "merge1": {
            "core": 49,
            "startUs": 3012.82,
            "endUs": 3231.9,
            "durUs": 219.08,
            "setupUs": 73.16,
            "kernelUs": 145.92
          },
          "aiv0TailUs": 54.44,
          "aiv1DeltaUs": -0.64
        },
        {
          "group": 13,
          "tier": "slow",
          "aic": {
            "core": 13,
            "startUs": 2134.98,
            "endUs": 3030.98,
            "durUs": 896.0,
            "setupUs": 0.34,
            "kernelUs": 895.66
          },
          "aiv0": {
            "core": 50,
            "startUs": 2135.0,
            "endUs": 3083.46,
            "durUs": 948.46,
            "setupUs": 0.38,
            "kernelUs": 948.08
          },
          "aiv1": {
            "core": 51,
            "startUs": 2135.02,
            "endUs": 3029.6,
            "durUs": 894.58,
            "setupUs": 0.56,
            "kernelUs": 894.02
          },
          "merge0": {
            "core": 50,
            "startUs": 3083.76,
            "endUs": 3293.42,
            "durUs": 209.66,
            "setupUs": 2.44,
            "kernelUs": 207.22
          },
          "merge1": {
            "core": 51,
            "startUs": 3029.88,
            "endUs": 3291.04,
            "durUs": 261.16,
            "setupUs": 56.34,
            "kernelUs": 204.82
          },
          "aiv0TailUs": 52.46,
          "aiv1DeltaUs": -1.42
        },
        {
          "group": 14,
          "tier": "slow",
          "aic": {
            "core": 14,
            "startUs": 2136.04,
            "endUs": 3019.18,
            "durUs": 883.14,
            "setupUs": 0.3,
            "kernelUs": 882.84
          },
          "aiv0": {
            "core": 52,
            "startUs": 2136.06,
            "endUs": 3068.5,
            "durUs": 932.44,
            "setupUs": 0.28,
            "kernelUs": 932.16
          },
          "aiv1": {
            "core": 53,
            "startUs": 2136.08,
            "endUs": 3017.84,
            "durUs": 881.76,
            "setupUs": 0.34,
            "kernelUs": 881.42
          },
          "merge0": {
            "core": 52,
            "startUs": 3068.92,
            "endUs": 3273.4,
            "durUs": 204.48,
            "setupUs": 17.06,
            "kernelUs": 187.42
          },
          "merge1": {
            "core": 53,
            "startUs": 3018.2,
            "endUs": 3284.18,
            "durUs": 265.98,
            "setupUs": 67.94,
            "kernelUs": 198.04
          },
          "aiv0TailUs": 49.3,
          "aiv1DeltaUs": -1.38
        },
        {
          "group": 15,
          "tier": "fast",
          "aic": {
            "core": 15,
            "startUs": 2145.52,
            "endUs": 2512.22,
            "durUs": 366.7,
            "setupUs": 0.38,
            "kernelUs": 366.32
          },
          "aiv0": {
            "core": 54,
            "startUs": 2145.5,
            "endUs": 2599.76,
            "durUs": 454.26,
            "setupUs": 0.3,
            "kernelUs": 453.96
          },
          "aiv1": {
            "core": 55,
            "startUs": 2145.5,
            "endUs": 2509.8,
            "durUs": 364.3,
            "setupUs": 0.3,
            "kernelUs": 364.0
          },
          "merge0": {
            "core": 54,
            "startUs": 2600.0,
            "endUs": 3197.86,
            "durUs": 597.86,
            "setupUs": 486.0,
            "kernelUs": 111.86
          },
          "merge1": {
            "core": 55,
            "startUs": 2510.1,
            "endUs": 3200.54,
            "durUs": 690.44,
            "setupUs": 576.1,
            "kernelUs": 114.34
          },
          "aiv0TailUs": 87.56,
          "aiv1DeltaUs": -2.4
        },
        {
          "group": 16,
          "tier": "fast",
          "aic": {
            "core": 16,
            "startUs": 2134.98,
            "endUs": 2497.72,
            "durUs": 362.74,
            "setupUs": 0.58,
            "kernelUs": 362.16
          },
          "aiv0": {
            "core": 56,
            "startUs": 2135.0,
            "endUs": 2587.32,
            "durUs": 452.32,
            "setupUs": 0.56,
            "kernelUs": 451.76
          },
          "aiv1": {
            "core": 57,
            "startUs": 2135.0,
            "endUs": 2496.24,
            "durUs": 361.24,
            "setupUs": 0.38,
            "kernelUs": 360.86
          },
          "merge0": {
            "core": 56,
            "startUs": 2587.7,
            "endUs": 3248.66,
            "durUs": 660.96,
            "setupUs": 498.32,
            "kernelUs": 162.64
          },
          "merge1": {
            "core": 57,
            "startUs": 2496.5,
            "endUs": 3251.02,
            "durUs": 754.52,
            "setupUs": 589.72,
            "kernelUs": 164.8
          },
          "aiv0TailUs": 89.58,
          "aiv1DeltaUs": -1.5
        },
        {
          "group": 17,
          "tier": "fast",
          "aic": {
            "core": 17,
            "startUs": 2136.04,
            "endUs": 2499.24,
            "durUs": 363.2,
            "setupUs": 1.84,
            "kernelUs": 361.36
          },
          "aiv0": {
            "core": 58,
            "startUs": 2136.08,
            "endUs": 2595.56,
            "durUs": 459.48,
            "setupUs": 0.32,
            "kernelUs": 459.16
          },
          "aiv1": {
            "core": 59,
            "startUs": 2136.08,
            "endUs": 2496.46,
            "durUs": 360.38,
            "setupUs": 0.5,
            "kernelUs": 359.88
          },
          "merge0": {
            "core": 58,
            "startUs": 2595.82,
            "endUs": 3253.58,
            "durUs": 657.76,
            "setupUs": 490.34,
            "kernelUs": 167.42
          },
          "merge1": {
            "core": 59,
            "startUs": 2496.68,
            "endUs": 3252.8,
            "durUs": 756.12,
            "setupUs": 589.66,
            "kernelUs": 166.46
          },
          "aiv0TailUs": 96.28,
          "aiv1DeltaUs": -2.82
        },
        {
          "group": 18,
          "tier": "slow",
          "aic": {
            "core": 18,
            "startUs": 2145.52,
            "endUs": 3019.38,
            "durUs": 873.86,
            "setupUs": 0.58,
            "kernelUs": 873.28
          },
          "aiv0": {
            "core": 60,
            "startUs": 2145.5,
            "endUs": 3071.12,
            "durUs": 925.62,
            "setupUs": 0.28,
            "kernelUs": 925.34
          },
          "aiv1": {
            "core": 61,
            "startUs": 2145.52,
            "endUs": 3018.28,
            "durUs": 872.76,
            "setupUs": 0.32,
            "kernelUs": 872.44
          },
          "merge0": {
            "core": 60,
            "startUs": 3071.36,
            "endUs": 3232.84,
            "durUs": 161.48,
            "setupUs": 15.0,
            "kernelUs": 146.48
          },
          "merge1": {
            "core": 61,
            "startUs": 3018.54,
            "endUs": 3229.96,
            "durUs": 211.42,
            "setupUs": 67.72,
            "kernelUs": 143.7
          },
          "aiv0TailUs": 51.76,
          "aiv1DeltaUs": -1.1
        },
        {
          "group": 19,
          "tier": "slow",
          "aic": {
            "core": 19,
            "startUs": 2135.0,
            "endUs": 3022.44,
            "durUs": 887.44,
            "setupUs": 0.38,
            "kernelUs": 887.06
          },
          "aiv0": {
            "core": 62,
            "startUs": 2135.04,
            "endUs": 3074.62,
            "durUs": 939.58,
            "setupUs": 0.38,
            "kernelUs": 939.2
          },
          "aiv1": {
            "core": 63,
            "startUs": 2135.04,
            "endUs": 3020.8,
            "durUs": 885.76,
            "setupUs": 0.4,
            "kernelUs": 885.36
          },
          "merge0": {
            "core": 62,
            "startUs": 3075.0,
            "endUs": 3281.12,
            "durUs": 206.12,
            "setupUs": 11.44,
            "kernelUs": 194.68
          },
          "merge1": {
            "core": 63,
            "startUs": 3021.06,
            "endUs": 3288.98,
            "durUs": 267.92,
            "setupUs": 65.2,
            "kernelUs": 202.72
          },
          "aiv0TailUs": 52.14,
          "aiv1DeltaUs": -1.68
        },
        {
          "group": 20,
          "tier": "slow",
          "aic": {
            "core": 20,
            "startUs": 2136.06,
            "endUs": 3018.2,
            "durUs": 882.14,
            "setupUs": 0.48,
            "kernelUs": 881.66
          },
          "aiv0": {
            "core": 64,
            "startUs": 2136.1,
            "endUs": 3067.94,
            "durUs": 931.84,
            "setupUs": 0.38,
            "kernelUs": 931.46
          },
          "aiv1": {
            "core": 65,
            "startUs": 2136.1,
            "endUs": 3016.36,
            "durUs": 880.26,
            "setupUs": 0.34,
            "kernelUs": 879.92
          },
          "merge0": {
            "core": 64,
            "startUs": 3068.26,
            "endUs": 3289.36,
            "durUs": 221.1,
            "setupUs": 18.38,
            "kernelUs": 202.72
          },
          "merge1": {
            "core": 65,
            "startUs": 3016.72,
            "endUs": 3283.38,
            "durUs": 266.66,
            "setupUs": 69.88,
            "kernelUs": 196.78
          },
          "aiv0TailUs": 49.7,
          "aiv1DeltaUs": -1.88
        },
        {
          "group": 21,
          "tier": "fast",
          "aic": {
            "core": 21,
            "startUs": 2145.56,
            "endUs": 2581.38,
            "durUs": 435.82,
            "setupUs": 0.38,
            "kernelUs": 435.44
          },
          "aiv0": {
            "core": 66,
            "startUs": 2145.54,
            "endUs": 2663.9,
            "durUs": 518.36,
            "setupUs": 0.52,
            "kernelUs": 517.84
          },
          "aiv1": {
            "core": 67,
            "startUs": 2145.54,
            "endUs": 2580.08,
            "durUs": 434.54,
            "setupUs": 0.34,
            "kernelUs": 434.2
          },
          "merge0": {
            "core": 66,
            "startUs": 2664.12,
            "endUs": 3203.64,
            "durUs": 539.52,
            "setupUs": 424.78,
            "kernelUs": 114.74
          },
          "merge1": {
            "core": 67,
            "startUs": 2580.32,
            "endUs": 3203.26,
            "durUs": 622.94,
            "setupUs": 508.8,
            "kernelUs": 114.14
          },
          "aiv0TailUs": 82.54,
          "aiv1DeltaUs": -1.28
        },
        {
          "group": 22,
          "tier": "fast",
          "aic": {
            "core": 22,
            "startUs": 2135.0,
            "endUs": 2563.38,
            "durUs": 428.38,
            "setupUs": 0.46,
            "kernelUs": 427.92
          },
          "aiv0": {
            "core": 68,
            "startUs": 2135.02,
            "endUs": 2640.82,
            "durUs": 505.8,
            "setupUs": 0.4,
            "kernelUs": 505.4
          },
          "aiv1": {
            "core": 69,
            "startUs": 2135.02,
            "endUs": 2561.06,
            "durUs": 426.04,
            "setupUs": 0.32,
            "kernelUs": 425.72
          },
          "merge0": {
            "core": 68,
            "startUs": 2641.28,
            "endUs": 3253.96,
            "durUs": 612.68,
            "setupUs": 447.16,
            "kernelUs": 165.52
          },
          "merge1": {
            "core": 69,
            "startUs": 2561.4,
            "endUs": 3253.04,
            "durUs": 691.64,
            "setupUs": 525.4,
            "kernelUs": 166.24
          },
          "aiv0TailUs": 77.42,
          "aiv1DeltaUs": -2.34
        },
        {
          "group": 23,
          "tier": "fast",
          "aic": {
            "core": 23,
            "startUs": 2136.06,
            "endUs": 2588.64,
            "durUs": 452.58,
            "setupUs": 0.3,
            "kernelUs": 452.28
          },
          "aiv0": {
            "core": 70,
            "startUs": 2136.08,
            "endUs": 2675.08,
            "durUs": 539.0,
            "setupUs": 0.38,
            "kernelUs": 538.62
          },
          "aiv1": {
            "core": 71,
            "startUs": 2136.1,
            "endUs": 2586.88,
            "durUs": 450.78,
            "setupUs": 0.38,
            "kernelUs": 450.4
          },
          "merge0": {
            "core": 70,
            "startUs": 2675.34,
            "endUs": 3249.88,
            "durUs": 574.54,
            "setupUs": 411.26,
            "kernelUs": 163.28
          },
          "merge1": {
            "core": 71,
            "startUs": 2587.14,
            "endUs": 3253.74,
            "durUs": 666.6,
            "setupUs": 499.44,
            "kernelUs": 167.16
          },
          "aiv0TailUs": 86.42,
          "aiv1DeltaUs": -1.8
        }
      ]
    }
  }
});
})(window);
