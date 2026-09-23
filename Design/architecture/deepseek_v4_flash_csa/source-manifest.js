window.PtoCsaSourceManifest = Object.freeze({
  "../../../Data/DeepSeek-V4-Flash-Official/inference-config.json": {
    "hash": "6cc6f816ca73a8d38750194e330398e4f6955b4b45f674f7d29c96da14ccb733",
    "lines": 35,
    "symbols": []
  },
  "../../../Data/DeepSeek-V4-Flash-Official/model.py": {
    "hash": "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71",
    "lines": 828,
    "symbols": [
      {
        "line": 25,
        "name": "set_dtype"
      },
      {
        "line": 35,
        "name": "ModelArgs"
      },
      {
        "line": 83,
        "name": "ParallelEmbedding"
      },
      {
        "line": 86,
        "name": "ParallelEmbedding.__init__"
      },
      {
        "line": 96,
        "name": "ParallelEmbedding.forward"
      },
      {
        "line": 108,
        "name": "linear"
      },
      {
        "line": 123,
        "name": "Linear"
      },
      {
        "line": 126,
        "name": "Linear.__init__"
      },
      {
        "line": 151,
        "name": "Linear.forward"
      },
      {
        "line": 155,
        "name": "ColumnParallelLinear"
      },
      {
        "line": 157,
        "name": "ColumnParallelLinear.__init__"
      },
      {
        "line": 162,
        "name": "ColumnParallelLinear.forward"
      },
      {
        "line": 166,
        "name": "RowParallelLinear"
      },
      {
        "line": 168,
        "name": "RowParallelLinear.__init__"
      },
      {
        "line": 173,
        "name": "RowParallelLinear.forward"
      },
      {
        "line": 183,
        "name": "RMSNorm"
      },
      {
        "line": 184,
        "name": "RMSNorm.__init__"
      },
      {
        "line": 191,
        "name": "RMSNorm.forward"
      },
      {
        "line": 200,
        "name": "precompute_freqs_cis"
      },
      {
        "line": 205,
        "name": "find_correction_dim"
      },
      {
        "line": 208,
        "name": "find_correction_range"
      },
      {
        "line": 213,
        "name": "linear_ramp_factor"
      },
      {
        "line": 232,
        "name": "apply_rotary_emb"
      },
      {
        "line": 247,
        "name": "rotate_activation"
      },
      {
        "line": 255,
        "name": "get_window_topk_idxs"
      },
      {
        "line": 269,
        "name": "get_compress_topk_idxs"
      },
      {
        "line": 279,
        "name": "Compressor"
      },
      {
        "line": 283,
        "name": "Compressor.__init__"
      },
      {
        "line": 307,
        "name": "Compressor.overlap_transform"
      },
      {
        "line": 316,
        "name": "Compressor.forward"
      },
      {
        "line": 380,
        "name": "Indexer"
      },
      {
        "line": 384,
        "name": "Indexer.__init__"
      },
      {
        "line": 402,
        "name": "Indexer.forward"
      },
      {
        "line": 436,
        "name": "Attention"
      },
      {
        "line": 439,
        "name": "Attention.__init__"
      },
      {
        "line": 484,
        "name": "Attention.forward"
      },
      {
        "line": 546,
        "name": "Gate"
      },
      {
        "line": 550,
        "name": "Gate.__init__"
      },
      {
        "line": 564,
        "name": "Gate.forward"
      },
      {
        "line": 587,
        "name": "Expert"
      },
      {
        "line": 589,
        "name": "Expert.__init__"
      },
      {
        "line": 596,
        "name": "Expert.forward"
      },
      {
        "line": 609,
        "name": "MoE"
      },
      {
        "line": 612,
        "name": "MoE.__init__"
      },
      {
        "line": 629,
        "name": "MoE.forward"
      },
      {
        "line": 647,
        "name": "Block"
      },
      {
        "line": 652,
        "name": "Block.__init__"
      },
      {
        "line": 673,
        "name": "Block.hc_pre"
      },
      {
        "line": 683,
        "name": "Block.hc_post"
      },
      {
        "line": 688,
        "name": "Block.forward"
      },
      {
        "line": 703,
        "name": "ParallelHead"
      },
      {
        "line": 705,
        "name": "ParallelHead.__init__"
      },
      {
        "line": 715,
        "name": "ParallelHead.get_logits"
      },
      {
        "line": 718,
        "name": "ParallelHead.forward"
      },
      {
        "line": 728,
        "name": "ParallelHead.hc_head"
      },
      {
        "line": 738,
        "name": "MTPBlock"
      },
      {
        "line": 740,
        "name": "MTPBlock.__init__"
      },
      {
        "line": 757,
        "name": "MTPBlock.forward"
      },
      {
        "line": 769,
        "name": "Transformer"
      },
      {
        "line": 772,
        "name": "Transformer.__init__"
      },
      {
        "line": 802,
        "name": "Transformer.forward"
      }
    ]
  },
  "../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/next_levels/decode_csa_test/kernel_config.py": {
    "hash": "a0272539d2f45a5d7489e87b36a0b938672f9563522fa43dfd2b173fcc45e3fe",
    "lines": 88,
    "symbols": []
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_compressor_ratio4.py": {
    "hash": "2df2d83a2c6c29b6bba0f60c0bcf96f8d97af421c73d7211a2dfdc07b545bb54",
    "lines": 609,
    "symbols": [
      {
        "line": 75,
        "name": "compressor_ratio4"
      },
      {
        "line": 293,
        "name": "compressor_test"
      },
      {
        "line": 340,
        "name": "golden_compressor"
      },
      {
        "line": 417,
        "name": "rmsnorm"
      },
      {
        "line": 441,
        "name": "build_tensor_specs"
      },
      {
        "line": 453,
        "name": "default_starts"
      },
      {
        "line": 506,
        "name": "init_x"
      },
      {
        "line": 508,
        "name": "init_compress_state"
      },
      {
        "line": 512,
        "name": "init_compress_state_block_table"
      },
      {
        "line": 516,
        "name": "init_wkv"
      },
      {
        "line": 518,
        "name": "init_wgate"
      },
      {
        "line": 520,
        "name": "init_ape"
      },
      {
        "line": 522,
        "name": "init_norm_w"
      },
      {
        "line": 524,
        "name": "init_cos"
      },
      {
        "line": 526,
        "name": "init_sin"
      },
      {
        "line": 528,
        "name": "init_cmp_kv_cache"
      },
      {
        "line": 530,
        "name": "init_position_ids"
      },
      {
        "line": 532,
        "name": "init_state_slot_mapping"
      },
      {
        "line": 534,
        "name": "init_cmp_slot_mapping"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_cp_token_allgather.py": {
    "hash": "e0bb0158bec7b5da7f702fa1df2b4e4bcc98f298268293fbcc0e3ccdd1258eb0",
    "lines": 332,
    "symbols": [
      {
        "line": 23,
        "name": "_parse_tp_argv"
      },
      {
        "line": 79,
        "name": "decode_cp_token_allgather_step"
      },
      {
        "line": 188,
        "name": "decode_cp_token_allgather_fixture"
      },
      {
        "line": 208,
        "name": "l3_decode_cp_token_allgather_fixture"
      },
      {
        "line": 230,
        "name": "materialize_spec"
      },
      {
        "line": 244,
        "name": "cp_stack"
      },
      {
        "line": 249,
        "name": "cp_split"
      },
      {
        "line": 254,
        "name": "build_tensor_specs"
      },
      {
        "line": 264,
        "name": "init_hidden_local"
      },
      {
        "line": 279,
        "name": "golden_decode_cp_token_allgather"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py": {
    "hash": "5376b4dbcda62b90f31c734ed533cecd4442f770cd7994f08350f08dfca69a0f",
    "lines": 2080,
    "symbols": [
      {
        "line": 24,
        "name": "_parse_tp_argv"
      },
      {
        "line": 152,
        "name": "decode_csa"
      },
      {
        "line": 503,
        "name": "decode_csa_test"
      },
      {
        "line": 624,
        "name": "l3_decode_csa"
      },
      {
        "line": 755,
        "name": "decode_csa_tp1"
      },
      {
        "line": 936,
        "name": "decode_csa_tp1_test"
      },
      {
        "line": 1046,
        "name": "golden_decode_csa_tp1"
      },
      {
        "line": 1085,
        "name": "interleave_rope"
      },
      {
        "line": 1231,
        "name": "build_tensor_specs"
      },
      {
        "line": 1297,
        "name": "ring_slots"
      },
      {
        "line": 1312,
        "name": "round_half_away_from_zero"
      },
      {
        "line": 1315,
        "name": "quant_w_per_output_channel"
      },
      {
        "line": 1324,
        "name": "quant_w_per_row"
      },
      {
        "line": 1333,
        "name": "init_x_hc"
      },
      {
        "line": 1337,
        "name": "init_hc_attn_fn"
      },
      {
        "line": 1340,
        "name": "init_hc_attn_scale"
      },
      {
        "line": 1343,
        "name": "init_hc_attn_base"
      },
      {
        "line": 1353,
        "name": "init_attn_norm_w"
      },
      {
        "line": 1356,
        "name": "init_wq_a"
      },
      {
        "line": 1359,
        "name": "init_wq_b"
      },
      {
        "line": 1362,
        "name": "init_wkv"
      },
      {
        "line": 1365,
        "name": "init_gamma_cq"
      },
      {
        "line": 1368,
        "name": "init_gamma_ckv"
      },
      {
        "line": 1371,
        "name": "init_normalized_cache"
      },
      {
        "line": 1379,
        "name": "init_cmp_wkv"
      },
      {
        "line": 1382,
        "name": "init_cmp_wgate"
      },
      {
        "line": 1385,
        "name": "init_cmp_ape"
      },
      {
        "line": 1388,
        "name": "init_cmp_norm_w"
      },
      {
        "line": 1391,
        "name": "init_compress_state"
      },
      {
        "line": 1398,
        "name": "init_compress_state_block_table"
      },
      {
        "line": 1401,
        "name": "init_weights_proj"
      },
      {
        "line": 1404,
        "name": "init_hadamard_idx"
      },
      {
        "line": 1410,
        "name": "init_inner_wkv"
      },
      {
        "line": 1413,
        "name": "init_inner_wgate"
      },
      {
        "line": 1416,
        "name": "init_inner_ape"
      },
      {
        "line": 1419,
        "name": "init_inner_norm_w"
      },
      {
        "line": 1422,
        "name": "init_inner_compress_state"
      },
      {
        "line": 1429,
        "name": "init_inner_compress_state_block_table"
      },
      {
        "line": 1432,
        "name": "init_kv_cache"
      },
      {
        "line": 1435,
        "name": "init_window_block_table"
      },
      {
        "line": 1438,
        "name": "init_cmp_kv"
      },
      {
        "line": 1441,
        "name": "init_cmp_block_table"
      },
      {
        "line": 1444,
        "name": "init_idx_kv_cache"
      },
      {
        "line": 1449,
        "name": "init_idx_block_table"
      },
      {
        "line": 1452,
        "name": "init_attn_sink"
      },
      {
        "line": 1455,
        "name": "init_position_ids"
      },
      {
        "line": 1458,
        "name": "init_kv_seq_lens"
      },
      {
        "line": 1461,
        "name": "init_ori_slot_mapping"
      },
      {
        "line": 1468,
        "name": "init_window_swa_metadata"
      },
      {
        "line": 1476,
        "name": "init_window_swa_indices"
      },
      {
        "line": 1479,
        "name": "init_window_swa_lens"
      },
      {
        "line": 1482,
        "name": "init_cmp_slot_mapping"
      },
      {
        "line": 1490,
        "name": "init_idx_slot_mapping"
      },
      {
        "line": 1498,
        "name": "init_state_slot_mapping"
      },
      {
        "line": 1501,
        "name": "init_inner_state_slot_mapping"
      },
      {
        "line": 1504,
        "name": "init_wo_a"
      },
      {
        "line": 1507,
        "name": "init_wo_b"
      },
      {
        "line": 1626,
        "name": "build_distributed_tensor_specs"
      },
      {
        "line": 1752,
        "name": "golden_decode_csa"
      },
      {
        "line": 1831,
        "name": "_csa_x_out_compare"
      },
      {
        "line": 1843,
        "name": "compare"
      },
      {
        "line": 1891,
        "name": "build_full_compare"
      },
      {
        "line": 1897,
        "name": "pool_mapping"
      },
      {
        "line": 1901,
        "name": "mapped"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_indexer.py": {
    "hash": "e89a4d9ce42ec08c8a8abdc472a51273361f5aaafa1ef5e39ab398765d45a3c5",
    "lines": 1337,
    "symbols": [
      {
        "line": 126,
        "name": "merge2_top512_pairs"
      },
      {
        "line": 148,
        "name": "merge_topk_level_pairs"
      },
      {
        "line": 176,
        "name": "indexer_topk_leaf"
      },
      {
        "line": 228,
        "name": "indexer_topk_group_wave"
      },
      {
        "line": 316,
        "name": "indexer_topk_query_merge"
      },
      {
        "line": 429,
        "name": "indexer_score_topk_forest"
      },
      {
        "line": 626,
        "name": "indexer"
      },
      {
        "line": 816,
        "name": "indexer_test"
      },
      {
        "line": 895,
        "name": "gen_shared_weight"
      },
      {
        "line": 912,
        "name": "sim_fp8"
      },
      {
        "line": 928,
        "name": "golden_indexer"
      },
      {
        "line": 1073,
        "name": "build_tensor_specs"
      },
      {
        "line": 1139,
        "name": "interleave_rope"
      },
      {
        "line": 1174,
        "name": "init_x"
      },
      {
        "line": 1176,
        "name": "init_qr"
      },
      {
        "line": 1180,
        "name": "init_weights_proj"
      },
      {
        "line": 1182,
        "name": "init_cos"
      },
      {
        "line": 1184,
        "name": "init_sin"
      },
      {
        "line": 1186,
        "name": "init_cmp_cos"
      },
      {
        "line": 1188,
        "name": "init_cmp_sin"
      },
      {
        "line": 1190,
        "name": "init_hadamard"
      },
      {
        "line": 1192,
        "name": "init_inner_compress_state"
      },
      {
        "line": 1198,
        "name": "init_inner_compress_state_block_table"
      },
      {
        "line": 1200,
        "name": "init_inner_wkv"
      },
      {
        "line": 1202,
        "name": "init_inner_wgate"
      },
      {
        "line": 1204,
        "name": "init_inner_ape"
      },
      {
        "line": 1206,
        "name": "init_inner_norm_w"
      },
      {
        "line": 1208,
        "name": "init_idx_block_table"
      },
      {
        "line": 1210,
        "name": "init_position_ids"
      },
      {
        "line": 1212,
        "name": "init_kv_seq_lens"
      },
      {
        "line": 1214,
        "name": "init_inner_state_slot_mapping"
      },
      {
        "line": 1216,
        "name": "init_idx_slot_mapping"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_indexer_compressor.py": {
    "hash": "863ee4ddd7ec45df1ca5fa6d69b4b3efe25809e5046aaf02aff07e51dd1a3bea",
    "lines": 690,
    "symbols": [
      {
        "line": 74,
        "name": "indexer_compressor"
      },
      {
        "line": 339,
        "name": "compressor_test"
      },
      {
        "line": 390,
        "name": "golden_compressor"
      },
      {
        "line": 474,
        "name": "rmsnorm"
      },
      {
        "line": 510,
        "name": "build_tensor_specs"
      },
      {
        "line": 522,
        "name": "default_starts"
      },
      {
        "line": 575,
        "name": "init_x"
      },
      {
        "line": 577,
        "name": "init_compress_state"
      },
      {
        "line": 581,
        "name": "init_compress_state_block_table"
      },
      {
        "line": 585,
        "name": "init_wkv"
      },
      {
        "line": 587,
        "name": "init_wgate"
      },
      {
        "line": 589,
        "name": "init_ape"
      },
      {
        "line": 591,
        "name": "init_norm_w"
      },
      {
        "line": 593,
        "name": "init_cos"
      },
      {
        "line": 595,
        "name": "init_sin"
      },
      {
        "line": 597,
        "name": "init_hadamard"
      },
      {
        "line": 599,
        "name": "init_idx_kv_cache"
      },
      {
        "line": 601,
        "name": "init_idx_kv_scale"
      },
      {
        "line": 603,
        "name": "init_position_ids"
      },
      {
        "line": 605,
        "name": "init_inner_state_slot_mapping"
      },
      {
        "line": 607,
        "name": "init_idx_slot_mapping"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_o_proj.py": {
    "hash": "d7a4d8b66a58989e9b278fdf44a71576828d0508ea8b8fb413a6e23548d255aa",
    "lines": 776,
    "symbols": [
      {
        "line": 23,
        "name": "_parse_tp_argv"
      },
      {
        "line": 138,
        "name": "decode_o_proj_tp1"
      },
      {
        "line": 269,
        "name": "o_group_a2a"
      },
      {
        "line": 339,
        "name": "l2_o_group_a2a"
      },
      {
        "line": 391,
        "name": "l3_o_group_a2a"
      },
      {
        "line": 410,
        "name": "build_o_group_a2a_specs"
      },
      {
        "line": 419,
        "name": "init_attention_grouped"
      },
      {
        "line": 439,
        "name": "golden_o_group_a2a"
      },
      {
        "line": 457,
        "name": "o_proj_reduce_scatter"
      },
      {
        "line": 701,
        "name": "golden_decode_o_proj_tp1"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py": {
    "hash": "7e4afa5eefa4eebabe421cda6da6c32c49f5e200c1dc74dcc63b4c3b6aa8800b",
    "lines": 820,
    "symbols": [
      {
        "line": 105,
        "name": "sparse_attn_csa"
      },
      {
        "line": 361,
        "name": "sparse_attn_csa_tp1"
      },
      {
        "line": 446,
        "name": "sparse_attn_csa_test"
      },
      {
        "line": 480,
        "name": "golden_sparse_attn"
      },
      {
        "line": 588,
        "name": "build_tensor_specs"
      },
      {
        "line": 640,
        "name": "init_q"
      },
      {
        "line": 647,
        "name": "init_ori_kv"
      },
      {
        "line": 659,
        "name": "init_window_swa_indices"
      },
      {
        "line": 670,
        "name": "init_cmp_kv"
      },
      {
        "line": 674,
        "name": "init_attn_sink"
      },
      {
        "line": 678,
        "name": "init_window_block_table"
      },
      {
        "line": 682,
        "name": "init_cmp_block_table"
      },
      {
        "line": 686,
        "name": "init_cmp_sparse_indices"
      },
      {
        "line": 712,
        "name": "init_idx_topk"
      },
      {
        "line": 716,
        "name": "init_position_ids"
      },
      {
        "line": 719,
        "name": "init_cos"
      },
      {
        "line": 723,
        "name": "init_sin"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/hc_post.py": {
    "hash": "b70836ef546dd426a6615d022e299619fd844df103b5497ab5751c505079397b",
    "lines": 240,
    "symbols": [
      {
        "line": 38,
        "name": "hc_post"
      },
      {
        "line": 73,
        "name": "hc_post_prefill"
      },
      {
        "line": 128,
        "name": "hc_post_test"
      },
      {
        "line": 145,
        "name": "golden_hc_post"
      },
      {
        "line": 166,
        "name": "golden_hc_post_prefill"
      },
      {
        "line": 173,
        "name": "build_tensor_specs"
      },
      {
        "line": 179,
        "name": "init_x"
      },
      {
        "line": 181,
        "name": "init_residual"
      },
      {
        "line": 183,
        "name": "init_post"
      },
      {
        "line": 185,
        "name": "init_comb"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/hc_pre.py": {
    "hash": "aeaca8bea2b163b34502adc63d907a83d492661f37a065f792b8769279d4be54",
    "lines": 468,
    "symbols": [
      {
        "line": 48,
        "name": "hc_pre"
      },
      {
        "line": 291,
        "name": "hc_pre_test"
      },
      {
        "line": 309,
        "name": "_golden_a2a3_cube_linear"
      },
      {
        "line": 336,
        "name": "golden_hc_pre"
      },
      {
        "line": 384,
        "name": "build_tensor_specs"
      },
      {
        "line": 391,
        "name": "init_x"
      },
      {
        "line": 393,
        "name": "init_hc_fn"
      },
      {
        "line": 395,
        "name": "init_hc_scale"
      },
      {
        "line": 397,
        "name": "init_hc_base"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/qkv_proj_rope.py": {
    "hash": "25ad8def0e60cab8cef1d0244ac7ebd8899c224a8f2eb35f7f379b014e442a45",
    "lines": 1177,
    "symbols": [
      {
        "line": 77,
        "name": "materialize_rope_rows_dynamic"
      },
      {
        "line": 98,
        "name": "materialize_rope_rows"
      },
      {
        "line": 120,
        "name": "rope_prepare"
      },
      {
        "line": 244,
        "name": "q_proj_rope"
      },
      {
        "line": 556,
        "name": "kv_proj_rope"
      },
      {
        "line": 763,
        "name": "qkv_proj_rope"
      },
      {
        "line": 812,
        "name": "qkv_proj_rope_test"
      },
      {
        "line": 862,
        "name": "q_kv_split_test"
      },
      {
        "line": 918,
        "name": "golden_q_kv_split"
      },
      {
        "line": 938,
        "name": "build_split_tensor_specs"
      },
      {
        "line": 978,
        "name": "golden_qkv_proj_rope"
      },
      {
        "line": 993,
        "name": "rms_norm"
      },
      {
        "line": 997,
        "name": "matmul_bf16_input_fp32"
      },
      {
        "line": 1002,
        "name": "apply_rope"
      },
      {
        "line": 1044,
        "name": "build_tensor_specs"
      },
      {
        "line": 1050,
        "name": "quant_w_per_output_channel"
      },
      {
        "line": 1059,
        "name": "init_x"
      },
      {
        "line": 1062,
        "name": "init_wq_a"
      },
      {
        "line": 1065,
        "name": "init_wq_b"
      },
      {
        "line": 1068,
        "name": "init_wkv"
      },
      {
        "line": 1071,
        "name": "init_cos"
      },
      {
        "line": 1074,
        "name": "init_sin"
      },
      {
        "line": 1077,
        "name": "init_gamma_cq"
      },
      {
        "line": 1080,
        "name": "init_gamma_ckv"
      }
    ]
  },
  "../../../Data/DeepseekV4/deepseek_v4_flash_dspark/rmsnorm.py": {
    "hash": "826d5a439b180e78ad8ff94d800f19370df7fe12aa6cb0670e3332792a5c9864",
    "lines": 234,
    "symbols": [
      {
        "line": 34,
        "name": "_rms_norm_full_tile"
      },
      {
        "line": 67,
        "name": "_rms_norm_tail_tile"
      },
      {
        "line": 119,
        "name": "rms_norm"
      },
      {
        "line": 141,
        "name": "rms_norm_test"
      },
      {
        "line": 153,
        "name": "golden_rms_norm"
      },
      {
        "line": 162,
        "name": "golden_rms_norm_test"
      },
      {
        "line": 166,
        "name": "build_tensor_specs"
      },
      {
        "line": 172,
        "name": "init_x"
      },
      {
        "line": 175,
        "name": "init_norm_w"
      }
    ]
  }
});
