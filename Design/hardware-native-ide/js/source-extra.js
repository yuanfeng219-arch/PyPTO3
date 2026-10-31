// 计算主链路：摘自 Design/Toolkit Studio/decode_layer.py，保留原始行号。
window.HN_EXTRA_SOURCE = [
  {
    "no": 411,
    "t": "        with pl.at("
  },
  {
    "no": 412,
    "t": "            level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 413,
    "t": "            name_hint=\"rms_recip\","
  },
  {
    "no": 414,
    "t": "            deps=[prev_out_tids[i] for i in range(DOWN_ON)],"
  },
  {
    "no": 415,
    "t": "        ) as rms_tid:"
  },
  {
    "no": 416,
    "t": "            partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)"
  },
  {
    "no": 417,
    "t": "            for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):"
  },
  {
    "no": 418,
    "t": "                k0 = kb * RMSNORM_K_CHUNK"
  },
  {
    "no": 419,
    "t": "                x_chunk = hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK]  # FP32 already (was cast from BF16)"
  },
  {
    "no": 420,
    "t": "                partial_sq = pl.add("
  },
  {
    "no": 421,
    "t": "                    partial_sq,"
  },
  {
    "no": 422,
    "t": "                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]),"
  },
  {
    "no": 423,
    "t": "                )"
  },
  {
    "no": 424,
    "t": "            variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])"
  },
  {
    "no": 425,
    "t": "            inv_rms = pl.recip(pl.sqrt(variance))"
  },
  {
    "no": 426,
    "t": "            inv_rms_states = pl.assemble(inv_rms_states, inv_rms, [0, 0])"
  },
  {
    "no": 429,
    "t": "        with pl.at(level=pl.Level.CORE_GROUP, name_hint=\"q_seed\") as q_seed_tid:  # no explicit dep: runtime q_proj WAR hazard orders it after prev qk_norm"
  },
  {
    "no": 430,
    "t": "            for snb in pl.pipeline(Q_ON, stage=2):"
  },
  {
    "no": 431,
    "t": "                q_proj = pl.assemble("
  },
  {
    "no": 432,
    "t": "                    q_proj, pl.full([BATCH, QKV_N_TILE], dtype=pl.FP32, value=0.0), [0, snb * QKV_N_TILE]"
  },
  {
    "no": 433,
    "t": "                )"
  },
  {
    "no": 434,
    "t": "        for q_nt in pl.parallel(Q_ON):"
  },
  {
    "no": 435,
    "t": "            q_n_region = q_nt * QKV_N_TILE"
  },
  {
    "no": 436,
    "t": "            for q_ks in pl.range(QKV_OK):"
  },
  {
    "no": 437,
    "t": "                q_k_base = q_ks * QKV_K_SLICE"
  },
  {
    "no": 438,
    "t": "                with pl.at("
  },
  {
    "no": 439,
    "t": "                    level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 440,
    "t": "                    name_hint=\"q_proj\","
  },
  {
    "no": 441,
    "t": "                    deps=[prev_normed_tids[q_ks], q_seed_tid],  # 1:1 with normed slab (QKV_K_SLICE=DOWN_TN)"
  },
  {
    "no": 442,
    "t": "                ) as q_tid:"
  },
  {
    "no": 443,
    "t": "                    for n_sub in pl.range(N_SUB):"
  },
  {
    "no": 444,
    "t": "                        n0 = q_n_region + n_sub * TN"
  },
  {
    "no": 445,
    "t": "                        q_acc = pl.matmul("
  },
  {
    "no": 446,
    "t": "                            normed_in[:, q_k_base : q_k_base + TK],"
  },
  {
    "no": 447,
    "t": "                            wq[layer_hidden_base + q_k_base : layer_hidden_base + q_k_base + TK, n0 : n0 + TN],"
  },
  {
    "no": 448,
    "t": "                            out_dtype=pl.FP32,"
  },
  {
    "no": 449,
    "t": "                        )"
  },
  {
    "no": 450,
    "t": "                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):"
  },
  {
    "no": 451,
    "t": "                            kk = q_k_base + kc * TK"
  },
  {
    "no": 452,
    "t": "                            q_acc = pl.matmul_acc("
  },
  {
    "no": 453,
    "t": "                                q_acc, normed_in[:, kk : kk + TK], wq[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]"
  },
  {
    "no": 454,
    "t": "                            )"
  },
  {
    "no": 455,
    "t": "                        q_proj = pl.assemble(q_proj, q_acc, [0, n0], atomic=pl.AtomicType.Add)"
  },
  {
    "no": 456,
    "t": "                q_tile_tids[q_nt * QKV_OK + q_ks] = q_tid"
  },
  {
    "no": 459,
    "t": "        with pl.at(level=pl.Level.CORE_GROUP, name_hint=\"k_seed\", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as k_seed_tid:"
  },
  {
    "no": 460,
    "t": "            k_proj = pl.assemble(k_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])"
  },
  {
    "no": 461,
    "t": "        for k_nt in pl.parallel(KV_ON):"
  },
  {
    "no": 462,
    "t": "            k_n_region = k_nt * QKV_N_TILE"
  },
  {
    "no": 463,
    "t": "            for k_ks in pl.range(QKV_OK):"
  },
  {
    "no": 464,
    "t": "                k_k_base = k_ks * QKV_K_SLICE"
  },
  {
    "no": 465,
    "t": "                with pl.at("
  },
  {
    "no": 466,
    "t": "                    level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 467,
    "t": "                    name_hint=\"k_proj\","
  },
  {
    "no": 468,
    "t": "                    deps=[prev_normed_tids[0], prev_normed_tids[1], prev_normed_tids[2], prev_normed_tids[3], prev_normed_tids[4], k_seed_tid],"
  },
  {
    "no": 469,
    "t": "                ) as k_tid:"
  },
  {
    "no": 470,
    "t": "                    for n_sub in pl.range(N_SUB):"
  },
  {
    "no": 471,
    "t": "                        n0 = k_n_region + n_sub * TN"
  },
  {
    "no": 472,
    "t": "                        k_acc = pl.matmul("
  },
  {
    "no": 473,
    "t": "                            normed_in[:, k_k_base : k_k_base + TK],"
  },
  {
    "no": 474,
    "t": "                            wk[layer_hidden_base + k_k_base : layer_hidden_base + k_k_base + TK, n0 : n0 + TN],"
  },
  {
    "no": 475,
    "t": "                            out_dtype=pl.FP32,"
  },
  {
    "no": 476,
    "t": "                        )"
  },
  {
    "no": 477,
    "t": "                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):"
  },
  {
    "no": 478,
    "t": "                            kk = k_k_base + kc * TK"
  },
  {
    "no": 479,
    "t": "                            k_acc = pl.matmul_acc("
  },
  {
    "no": 480,
    "t": "                                k_acc, normed_in[:, kk : kk + TK], wk[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]"
  },
  {
    "no": 481,
    "t": "                            )"
  },
  {
    "no": 482,
    "t": "                        k_proj = pl.assemble(k_proj, k_acc, [0, n0], atomic=pl.AtomicType.Add)"
  },
  {
    "no": 483,
    "t": "                k_tile_tids[k_nt * QKV_OK + k_ks] = k_tid"
  },
  {
    "no": 484,
    "t": ""
  },
  {
    "no": 485,
    "t": "        # ── Scope 1: V projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──"
  },
  {
    "no": 486,
    "t": "        with pl.at(level=pl.Level.CORE_GROUP, name_hint=\"v_seed\", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as v_seed_tid:"
  },
  {
    "no": 487,
    "t": "            v_proj = pl.assemble(v_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])"
  },
  {
    "no": 488,
    "t": "        for v_nt in pl.parallel(KV_ON):"
  },
  {
    "no": 489,
    "t": "            v_n_region = v_nt * QKV_N_TILE"
  },
  {
    "no": 490,
    "t": "            for v_ks in pl.range(QKV_OK):"
  },
  {
    "no": 491,
    "t": "                v_k_base = v_ks * QKV_K_SLICE"
  },
  {
    "no": 492,
    "t": "                with pl.at("
  },
  {
    "no": 493,
    "t": "                    level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 494,
    "t": "                    name_hint=\"v_proj\","
  },
  {
    "no": 495,
    "t": "                    deps=[prev_normed_tids[0], prev_normed_tids[1], prev_normed_tids[2], prev_normed_tids[3], prev_normed_tids[4], v_seed_tid],"
  },
  {
    "no": 496,
    "t": "                ) as v_tid:"
  },
  {
    "no": 497,
    "t": "                    for n_sub in pl.range(N_SUB):"
  },
  {
    "no": 498,
    "t": "                        n0 = v_n_region + n_sub * TN"
  },
  {
    "no": 499,
    "t": "                        v_acc = pl.matmul("
  },
  {
    "no": 500,
    "t": "                            normed_in[:, v_k_base : v_k_base + TK],"
  },
  {
    "no": 501,
    "t": "                            wv[layer_hidden_base + v_k_base : layer_hidden_base + v_k_base + TK, n0 : n0 + TN],"
  },
  {
    "no": 502,
    "t": "                            out_dtype=pl.FP32,"
  },
  {
    "no": 503,
    "t": "                        )"
  },
  {
    "no": 504,
    "t": "                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):"
  },
  {
    "no": 505,
    "t": "                            kk = v_k_base + kc * TK"
  },
  {
    "no": 506,
    "t": "                            v_acc = pl.matmul_acc("
  },
  {
    "no": 507,
    "t": "                                v_acc, normed_in[:, kk : kk + TK], wv[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]"
  },
  {
    "no": 508,
    "t": "                            )"
  },
  {
    "no": 509,
    "t": "                        v_proj = pl.assemble(v_proj, v_acc, [0, n0], atomic=pl.AtomicType.Add)"
  },
  {
    "no": 510,
    "t": "                v_tile_tids[v_nt * QKV_OK + v_ks] = v_tid"
  },
  {
    "no": 615,
    "t": "        with pl.spmd("
  },
  {
    "no": 616,
    "t": "            ROPE_CORES,"
  },
  {
    "no": 617,
    "t": "            name_hint=\"rope_qkv\","
  },
  {
    "no": 618,
    "t": "            deps=["
  },
  {
    "no": 619,
    "t": "                qk_tids[0], qk_tids[1], qk_tids[2], qk_tids[3],"
  },
  {
    "no": 620,
    "t": "                qk_tids[4], qk_tids[5], qk_tids[6], qk_tids[7],"
  },
  {
    "no": 621,
    "t": "                rms_tid,"
  },
  {
    "no": 622,
    "t": "                v_tile_tids[0], v_tile_tids[1], v_tile_tids[2], v_tile_tids[3],"
  },
  {
    "no": 623,
    "t": "                v_tile_tids[4], v_tile_tids[5], v_tile_tids[6], v_tile_tids[7],"
  },
  {
    "no": 624,
    "t": "                v_tile_tids[8], v_tile_tids[9],"
  },
  {
    "no": 625,
    "t": "            ],"
  },
  {
    "no": 626,
    "t": "        ) as rope_tid:"
  },
  {
    "no": 627,
    "t": "            rope_core = pl.get_block_idx()"
  },
  {
    "no": 628,
    "t": "            for it in pl.pipeline(ROPE_GROUP_ITEMS_PER_CORE, stage=2):"
  },
  {
    "no": 629,
    "t": "                g_idx = rope_core * ROPE_ITEMS_PER_CORE + it"
  },
  {
    "no": 630,
    "t": "                ki = g_idx // BATCH"
  },
  {
    "no": 631,
    "t": "                b = g_idx % BATCH"
  },
  {
    "no": 632,
    "t": "                ctx_len = pl.read(seq_lens, [b])"
  },
  {
    "no": 633,
    "t": "                inv_rms_b = pl.read(inv_rms_states, [b, 0])"
  },
  {
    "no": 634,
    "t": "                pos = ctx_len - 1  # absolute position -> RoPE cos/sin row (NOT the cache row)"
  },
  {
    "no": 635,
    "t": "                # Paged write target for this row's current token: slot_mapping[b]"
  },
  {
    "no": 636,
    "t": "                # decomposes into (physical page, in-page offset). Same scheme prefill"
  },
  {
    "no": 637,
    "t": "                # uses; one physical page == one SEQ_TILE/BLOCK_SIZE band of the pool."
  },
  {
    "no": 638,
    "t": "                wr_slot = pl.cast(pl.tensor.read(slot_mapping, [b]), pl.INDEX)"
  },
  {
    "no": 639,
    "t": "                wr_slot_block = wr_slot // BLOCK_SIZE"
  },
  {
    "no": 640,
    "t": "                wr_slot_offset = wr_slot - wr_slot_block * BLOCK_SIZE"
  },
  {
    "no": 641,
    "t": "                cos_lo = rope_cos[pos : pos + 1, 0:HALF_DIM]"
  },
  {
    "no": 642,
    "t": "                cos_hi = rope_cos[pos : pos + 1, HALF_DIM:HEAD_DIM]"
  },
  {
    "no": 643,
    "t": "                sin_lo = rope_sin[pos : pos + 1, 0:HALF_DIM]"
  },
  {
    "no": 644,
    "t": "                sin_hi = rope_sin[pos : pos + 1, HALF_DIM:HEAD_DIM]"
  },
  {
    "no": 645,
    "t": ""
  },
  {
    "no": 646,
    "t": "                kv_col = ki * HEAD_DIM"
  },
  {
    "no": 647,
    "t": "                # K carries qk_norm gamma (qk_gamma); fold the deferred per-head qk_inv"
  },
  {
    "no": 648,
    "t": "                # scalar here. inv_rms cancels inside qk_norm, so no inv_rms factor on K."
  },
  {
    "no": 649,
    "t": "                k_inv_b = pl.read(k_inv_states, [ki * BATCH + b, 0])"
  },
  {
    "no": 650,
    "t": "                k_full = pl.mul(k_proj_norm[b : b + 1, kv_col : kv_col + HEAD_DIM], k_inv_b)"
  },
  {
    "no": 651,
    "t": "                k_lo = k_full[:, 0:HALF_DIM]"
  },
  {
    "no": 652,
    "t": "                k_hi = k_full[:, HALF_DIM:HEAD_DIM]"
  },
  {
    "no": 653,
    "t": "                rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))"
  },
  {
    "no": 654,
    "t": "                rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))"
  },
  {
    "no": 655,
    "t": "                cache_row = layer_cache_base + (wr_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + wr_slot_offset"
  },
  {
    "no": 656,
    "t": "                k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_row, 0])"
  },
  {
    "no": 657,
    "t": "                k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_row, HALF_DIM])"
  },
  {
    "no": 658,
    "t": "                v_row_bf16 = pl.cast("
  },
  {
    "no": 659,
    "t": "                    pl.mul(v_proj[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM], inv_rms_b),"
  },
  {
    "no": 660,
    "t": "                    target_type=pl.BF16,"
  },
  {
    "no": 661,
    "t": "                )"
  },
  {
    "no": 662,
    "t": "                v_cache = pl.assemble(v_cache, v_row_bf16, [cache_row, 0])"
  },
  {
    "no": 663,
    "t": ""
  },
  {
    "no": 664,
    "t": "                q_base = ki * Q_PER_KV"
  },
  {
    "no": 665,
    "t": "                q_pad_row0 = b * NUM_KV_HEADS * Q_HEAD_PAD + ki * Q_HEAD_PAD"
  },
  {
    "no": 666,
    "t": "                q_inv_base = ki * BATCH * Q_PER_KV + b * Q_PER_KV"
  },
  {
    "no": 667,
    "t": "                for qj in pl.range(Q_PER_KV):"
  },
  {
    "no": 668,
    "t": "                    q_inv_bj = pl.read(q_inv_states, [q_inv_base + qj, 0])"
  },
  {
    "no": 669,
    "t": "                    q_head = pl.mul("
  },
  {
    "no": 670,
    "t": "                        q_proj_norm["
  },
  {
    "no": 671,
    "t": "                            b : b + 1, (q_base + qj) * HEAD_DIM : (q_base + qj + 1) * HEAD_DIM"
  },
  {
    "no": 672,
    "t": "                        ],"
  },
  {
    "no": 673,
    "t": "                        q_inv_bj,"
  },
  {
    "no": 674,
    "t": "                    )"
  },
  {
    "no": 675,
    "t": "                    q_lo = q_head[:, 0:HALF_DIM]"
  },
  {
    "no": 676,
    "t": "                    q_hi = q_head[:, HALF_DIM:HEAD_DIM]"
  },
  {
    "no": 677,
    "t": "                    q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))"
  },
  {
    "no": 678,
    "t": "                    q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))"
  },
  {
    "no": 679,
    "t": "                    all_q_padded = pl.assemble("
  },
  {
    "no": 680,
    "t": "                        all_q_padded, pl.cast(q_rot_lo, target_type=pl.BF16), [q_pad_row0 + qj, 0]"
  },
  {
    "no": 681,
    "t": "                    )"
  },
  {
    "no": 682,
    "t": "                    all_q_padded = pl.assemble("
  },
  {
    "no": 683,
    "t": "                        all_q_padded, pl.cast(q_rot_hi, target_type=pl.BF16), [q_pad_row0 + qj, HALF_DIM]"
  },
  {
    "no": 684,
    "t": "                    )"
  },
  {
    "no": 685,
    "t": "                q_pad_zero = pl.cast("
  },
  {
    "no": 686,
    "t": "                    pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),"
  },
  {
    "no": 687,
    "t": "                    target_type=pl.BF16,"
  },
  {
    "no": 688,
    "t": "                )"
  },
  {
    "no": 689,
    "t": "                all_q_padded = pl.assemble(all_q_padded, q_pad_zero, [q_pad_row0 + Q_HEAD_BATCH, 0])"
  },
  {
    "no": 690,
    "t": "        rope_grp_tids[0] = rope_tid"
  },
  {
    "no": 728,
    "t": "        with pl.spmd("
  },
  {
    "no": 729,
    "t": "            NUM_CORES,"
  },
  {
    "no": 730,
    "t": "            name_hint=\"fa_fused\","
  },
  {
    "no": 731,
    "t": "            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],"
  },
  {
    "no": 732,
    "t": "            deps=[work_tid, rope_grp_tids[0]],  # single rope grid now"
  },
  {
    "no": 733,
    "t": "        ) as fa_tid:"
  },
  {
    "no": 734,
    "t": "            fa_core = pl.get_block_idx()"
  },
  {
    "no": 735,
    "t": "            # Read the device-computed block count ON-DEVICE (inside the kernel) so"
  },
  {
    "no": 736,
    "t": "            # `fa_total` enters fa_fused as an INPUT and the normal task auto-dep"
  },
  {
    "no": 737,
    "t": "            # orders it after the `fa_work_build` producer. Reading it at"
  },
  {
    "no": 738,
    "t": "            # orchestration scope instead lowers to a host get_tensor_data that does"
  },
  {
    "no": 739,
    "t": "            # NOT wait for the producer (stale ~0 read → empty dispatch; see"
  },
  {
    "no": 740,
    "t": "            # KNOWN_ISSUES). This mirrors the existing on-device fa_work_table read."
  },
  {
    "no": 741,
    "t": "            fa_total_blocks = pl.cast(pl.read(fa_total, [0, 0]), target_type=pl.INDEX)"
  },
  {
    "no": 742,
    "t": "            # Grid-stride over the dense real-block list: core fa_core owns table"
  },
  {
    "no": 743,
    "t": "            # entries fa_core, fa_core+NUM_CORES, … < fa_total_blocks."
  },
  {
    "no": 744,
    "t": "            for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):"
  },
  {
    "no": 745,
    "t": "                fa_enc = pl.cast(pl.read(fa_work_table, [fa_w, 0]), target_type=pl.INDEX)"
  },
  {
    "no": 746,
    "t": "                fa_b = fa_enc // MAX_CTX_BLOCKS"
  },
  {
    "no": 747,
    "t": "                fa_p = fa_enc % MAX_CTX_BLOCKS"
  },
  {
    "no": 748,
    "t": "                fa_hg = 0  # HEAD_GROUPS == 1 (GP_SIZE == NUM_KV_HEADS)"
  },
  {
    "no": 749,
    "t": "                fa_ctx_len = pl.read(seq_lens, [fa_b])"
  },
  {
    "no": 750,
    "t": "                # Table holds only real blocks → exactly one block per entry, at fa_p."
  },
  {
    "no": 751,
    "t": "                sb = fa_p  # logical KV block index (no inner loop — old p_blocks was 1)"
  },
  {
    "no": 752,
    "t": "                s0 = sb * SEQ_TILE"
  },
  {
    "no": 753,
    "t": "                valid_len = pl.min(SEQ_TILE, fa_ctx_len - s0)"
  },
  {
    "no": 758,
    "t": "                fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)"
  },
  {
    "no": 764,
    "t": "                for gp in pl.pipeline(GP_SIZE, stage={{FA_STAGE}}):"
  },
  {
    "no": 765,
    "t": "                    gi = fa_hg * GP_SIZE + gp"
  },
  {
    "no": 766,
    "t": "                    kvh = gi  # Q_GROUPS=1"
  },
  {
    "no": 767,
    "t": "                    q_pad_row_g = fa_b * NUM_KV_HEADS * Q_HEAD_PAD + gi * Q_HEAD_PAD"
  },
  {
    "no": 768,
    "t": "                    q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]"
  },
  {
    "no": 769,
    "t": "                    g_base = (fa_b * NUM_KV_HEADS + gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD"
  },
  {
    "no": 770,
    "t": "                    cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE"
  },
  {
    "no": 771,
    "t": ""
  },
  {
    "no": 772,
    "t": "                    # QK matmul (cube) -> C2V boundary move (first vec consumer)."
  },
  {
    "no": 773,
    "t": "                    k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]"
  },
  {
    "no": 774,
    "t": "                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)"
  },
  {
    "no": 775,
    "t": "                    scores_scaled = pl.mul(raw_scores, ATTN_SCALE)"
  },
  {
    "no": 781,
    "t": "                    scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_BATCH, valid_len)"
  },
  {
    "no": 782,
    "t": "                    scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)"
  },
  {
    "no": 783,
    "t": "                    cur_mi = pl.row_max(scores)"
  },
  {
    "no": 784,
    "t": "                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))"
  },
  {
    "no": 785,
    "t": "                    cur_li = pl.row_sum(exp_scores)"
  },
  {
    "no": 786,
    "t": "                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)"
  },
  {
    "no": 787,
    "t": ""
  },
  {
    "no": 788,
    "t": "                    # SV matmul (cube) reads exp directly -> V2C boundary move."
  },
  {
    "no": 789,
    "t": "                    v_tile = v_cache[cache_row : cache_row + SEQ_TILE, :]"
  },
  {
    "no": 790,
    "t": "                    oi_tmp = pl.matmul(exp_scores_bf16, v_tile, out_dtype=pl.FP32)"
  },
  {
    "no": 791,
    "t": "                    # The SV-matmul output spans the full padded 16 rows but only"
  },
  {
    "no": 792,
    "t": "                    # the first Q_HEAD_BATCH are real — mark 5 valid (tile stays 16)"
  },
  {
    "no": 793,
    "t": "                    # so the (dominant) all_oi_tmp GM write stores 5 rows instead of 16."
  },
  {
    "no": 794,
    "t": "                    oi_tmp = pl.set_validshape(oi_tmp, Q_HEAD_BATCH, HEAD_DIM)"
  },
  {
    "no": 795,
    "t": ""
  },
  {
    "no": 796,
    "t": "                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [g_base + sb * Q_HEAD_PAD, 0])"
  },
  {
    "no": 797,
    "t": "                    all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + sb * Q_HEAD_PAD, 0])"
  },
  {
    "no": 798,
    "t": "                    all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + sb * Q_HEAD_PAD, 0])"
  },
  {
    "no": 810,
    "t": "        with pl.spmd("
  },
  {
    "no": 811,
    "t": "            NUM_CORES * 2, name_hint=\"online_softmax\", deps=[fa_tid]"
  },
  {
    "no": 812,
    "t": "        ) as attn_done_tid:"
  },
  {
    "no": 813,
    "t": "            os_core = pl.get_block_idx()"
  },
  {
    "no": 814,
    "t": "            # Grid-stride over online_softmax work items (one per (b, kvh) lane)."
  },
  {
    "no": 815,
    "t": "            for os_spmd_idx in pl.range(os_core, OS_WORK, NUM_CORES * 2):"
  },
  {
    "no": 816,
    "t": "                os_b = os_spmd_idx // NUM_KV_HEADS"
  },
  {
    "no": 817,
    "t": "                os_gi = os_spmd_idx % NUM_KV_HEADS"
  },
  {
    "no": 818,
    "t": "                os_ctx_len = pl.read(seq_lens, [os_b])"
  },
  {
    "no": 819,
    "t": "                os_ctx_blocks = (os_ctx_len + SEQ_TILE - 1) // SEQ_TILE"
  },
  {
    "no": 820,
    "t": "                os_kvh = os_gi  # Q_GROUPS=1"
  },
  {
    "no": 821,
    "t": "                os_q_base = os_kvh * Q_PER_KV"
  },
  {
    "no": 822,
    "t": "                os_g_base = (os_b * NUM_KV_HEADS + os_gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD"
  },
  {
    "no": 823,
    "t": ""
  },
  {
    "no": 824,
    "t": "                # Loop-carried accumulators: full Q_HEAD_PAD (16) rows. They can't be"
  },
  {
    "no": 825,
    "t": "                # valid-shaped because the recurrence (add / maximum) drops valid_shape,"
  },
  {
    "no": 826,
    "t": "                # which would break loop-carry type consistency. Only the per-block GM"
  },
  {
    "no": 827,
    "t": "                # reads below are trimmed to the 5 real rows via pl.slice valid_shape."
  },
  {
    "no": 828,
    "t": "                oi = all_oi_tmp[os_g_base : os_g_base + Q_HEAD_PAD, :]"
  },
  {
    "no": 829,
    "t": "                mi = all_cur_mi[os_g_base : os_g_base + Q_HEAD_PAD, :]"
  },
  {
    "no": 830,
    "t": "                li = all_cur_li[os_g_base : os_g_base + Q_HEAD_PAD, :]"
  },
  {
    "no": 831,
    "t": "                for sb in pl.pipeline(1, os_ctx_blocks, stage=2):"
  },
  {
    "no": 832,
    "t": "                    rec = os_g_base + sb * Q_HEAD_PAD"
  },
  {
    "no": 833,
    "t": "                    # Partial load: tile stays Q_HEAD_PAD (16) rows, GM transfer = the 5"
  },
  {
    "no": 834,
    "t": "                    # real rows (valid_shape on the slice → tile.load valid_shapes)."
  },
  {
    "no": 835,
    "t": "                    oi_tmp_valid = pl.slice("
  },
  {
    "no": 836,
    "t": "                        all_oi_tmp, [Q_HEAD_PAD, HEAD_DIM], [rec, 0], valid_shape=[Q_HEAD_BATCH, HEAD_DIM]"
  },
  {
    "no": 837,
    "t": "                    )"
  },
  {
    "no": 838,
    "t": "                    online_cur_mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [rec, 0], valid_shape=[Q_HEAD_BATCH, 1])"
  },
  {
    "no": 839,
    "t": "                    online_cur_li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [rec, 0], valid_shape=[Q_HEAD_BATCH, 1])"
  },
  {
    "no": 840,
    "t": "                    mi_new = pl.maximum(mi, online_cur_mi)"
  },
  {
    "no": 841,
    "t": "                    alpha = pl.exp(pl.sub(mi, mi_new))"
  },
  {
    "no": 842,
    "t": "                    beta = pl.exp(pl.sub(online_cur_mi, mi_new))"
  },
  {
    "no": 843,
    "t": "                    li = pl.add(pl.mul(alpha, li), pl.mul(beta, online_cur_li))"
  },
  {
    "no": 844,
    "t": "                    oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp_valid, beta))"
  },
  {
    "no": 845,
    "t": "                    mi = mi_new"
  },
  {
    "no": 846,
    "t": ""
  },
  {
    "no": 847,
    "t": "                ctx = pl.row_expand_div(oi, li)"
  },
  {
    "no": 848,
    "t": "                ctx_valid = ctx[0:Q_HEAD_BATCH, :]"
  },
  {
    "no": 849,
    "t": "                ctx_flat_bf16 = pl.cast("
  },
  {
    "no": 850,
    "t": "                    pl.reshape(ctx_valid, [1, Q_HEAD_BATCH * HEAD_DIM]), target_type=pl.BF16"
  },
  {
    "no": 851,
    "t": "                )"
  },
  {
    "no": 852,
    "t": "                attn_out = pl.assemble(attn_out, ctx_flat_bf16, [os_b, os_q_base * HEAD_DIM])"
  },
  {
    "no": 881,
    "t": "        for n_out_proj in pl.parallel(N_SPLITS_OUT):"
  },
  {
    "no": 882,
    "t": "            n_op = n_out_proj * OUT_TN"
  },
  {
    "no": 883,
    "t": "            for k_split_out in pl.range(K_SPLITS_OUT):"
  },
  {
    "no": 884,
    "t": "                k_op = k_split_out * OUT_TK"
  },
  {
    "no": 885,
    "t": "                with pl.at("
  },
  {
    "no": 886,
    "t": "                    level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 887,
    "t": "                    name_hint=\"out_proj\","
  },
  {
    "no": 888,
    "t": "                    deps=[out_seed_tid, attn_done_tid],"
  },
  {
    "no": 889,
    "t": "                ) as out_tid:"
  },
  {
    "no": 890,
    "t": "                    out_a0 = attn_out[:, k_op : k_op + OUT_INNER_TK]"
  },
  {
    "no": 891,
    "t": "                    out_w0 = wo[layer_hidden_base + k_op : layer_hidden_base + k_op + OUT_INNER_TK, n_op : n_op + OUT_TN]"
  },
  {
    "no": 892,
    "t": "                    out_c_acc = pl.matmul(out_a0, out_w0, out_dtype=pl.FP32)"
  },
  {
    "no": 893,
    "t": "                    for out_lk in pl.pipeline(1, OUT_N_SUB_K, stage=2):"
  },
  {
    "no": 894,
    "t": "                        out_ks_off = out_lk * OUT_INNER_TK"
  },
  {
    "no": 895,
    "t": "                        out_a_k = attn_out[:, k_op + out_ks_off : k_op + out_ks_off + OUT_INNER_TK]"
  },
  {
    "no": 896,
    "t": "                        out_w_k = wo["
  },
  {
    "no": 897,
    "t": "                            layer_hidden_base + k_op + out_ks_off : layer_hidden_base + k_op + out_ks_off + OUT_INNER_TK,"
  },
  {
    "no": 898,
    "t": "                            n_op : n_op + OUT_TN,"
  },
  {
    "no": 899,
    "t": "                        ]"
  },
  {
    "no": 900,
    "t": "                        out_c_acc = pl.matmul_acc(out_c_acc, out_a_k, out_w_k)"
  },
  {
    "no": 901,
    "t": "                    attn_proj_fp32 = pl.assemble("
  },
  {
    "no": 902,
    "t": "                        attn_proj_fp32, out_c_acc, [0, n_op], atomic=pl.AtomicType.Add"
  },
  {
    "no": 903,
    "t": "                    )"
  },
  {
    "no": 904,
    "t": "                out_tids[n_out_proj * K_SPLITS_OUT + k_split_out] = out_tid"
  },
  {
    "no": 906,
    "t": "        # Tiled residual + BF16 cast."
  },
  {
    "no": 907,
    "t": "        for k_slice in pl.unroll(K_SPLITS_MLP):"
  },
  {
    "no": 908,
    "t": "            k_base = k_slice * MLP_K_SLICE"
  },
  {
    "no": 909,
    "t": "            n_split_base = k_slice * N_PER_CAST_K"
  },
  {
    "no": 910,
    "t": "            with pl.at("
  },
  {
    "no": 911,
    "t": "                level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 912,
    "t": "                name_hint=\"residual_rms_cast\","
  },
  {
    "no": 913,
    "t": "                deps=["
  },
  {
    "no": 914,
    "t": "                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 0],"
  },
  {
    "no": 915,
    "t": "                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 1],"
  },
  {
    "no": 916,
    "t": "                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 2],"
  },
  {
    "no": 917,
    "t": "                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 3],"
  },
  {
    "no": 918,
    "t": "                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 4],"
  },
  {
    "no": 919,
    "t": "                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 0],"
  },
  {
    "no": 920,
    "t": "                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 1],"
  },
  {
    "no": 921,
    "t": "                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 2],"
  },
  {
    "no": 922,
    "t": "                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 3],"
  },
  {
    "no": 923,
    "t": "                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 4],"
  },
  {
    "no": 924,
    "t": "                ],"
  },
  {
    "no": 925,
    "t": "            ) as cast_tid_k:"
  },
  {
    "no": 926,
    "t": "                for kb in pl.pipeline(MLP_K_SLICE // K_CHUNK, stage=2):"
  },
  {
    "no": 927,
    "t": "                    k0 = k_base + kb * K_CHUNK"
  },
  {
    "no": 928,
    "t": "                    attn_chunk = attn_proj_fp32[:, k0 : k0 + K_CHUNK]"
  },
  {
    "no": 929,
    "t": "                    hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]  # FP32 already"
  },
  {
    "no": 930,
    "t": "                    resid_fp32 = pl.add(attn_chunk, hidden_chunk)"
  },
  {
    "no": 931,
    "t": "                    # Raw residual h1 — added back after down_proj (must NOT be gamma-scaled)."
  },
  {
    "no": 932,
    "t": "                    post_norm_partial = pl.assemble(post_norm_partial, resid_fp32, [0, k0])  # FP32 (no BF16 cast)"
  },
  {
    "no": 933,
    "t": "                    # Explicit post-RMS gamma: gate/up input = h1 * post_gamma. gamma is"
  },
  {
    "no": 934,
    "t": "                    # per-K (the matmul contraction dim) so it canNOT defer past the matmul"
  },
  {
    "no": 935,
    "t": "                    # like inv_rms does — it scales the input here (with raw w_gate/w_up)."
  },
  {
    "no": 936,
    "t": "                    post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, k0])"
  },
  {
    "no": 937,
    "t": "                    mlp_norm_in = pl.assemble("
  },
  {
    "no": 938,
    "t": "                        mlp_norm_in,"
  },
  {
    "no": 939,
    "t": "                        pl.cast(pl.col_expand_mul(resid_fp32, post_gamma), target_type=pl.BF16),"
  },
  {
    "no": 940,
    "t": "                        [0, k0],"
  },
  {
    "no": 941,
    "t": "                    )"
  },
  {
    "no": 942,
    "t": "            cast_tids[k_slice] = cast_tid_k"
  },
  {
    "no": 943,
    "t": ""
  },
  {
    "no": 944,
    "t": "        # RMS reduction reads all of attn_proj_fp32."
  },
  {
    "no": 945,
    "t": "        with pl.at(level=pl.Level.CORE_GROUP, name_hint=\"post_rms_reduce\", deps=[out_tids]) as reduce_tid:"
  },
  {
    "no": 946,
    "t": "            sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)"
  },
  {
    "no": 947,
    "t": "            for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=2):"
  },
  {
    "no": 948,
    "t": "                k0 = kb * K_CHUNK"
  },
  {
    "no": 949,
    "t": "                attn_chunk = attn_proj_fp32[:, k0 : k0 + K_CHUNK]"
  },
  {
    "no": 950,
    "t": "                hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]  # FP32 already"
  },
  {
    "no": 951,
    "t": "                resid_chunk = pl.add(attn_chunk, hidden_chunk)"
  },
  {
    "no": 952,
    "t": "                sq_sum = pl.add("
  },
  {
    "no": 953,
    "t": "                    sq_sum,"
  },
  {
    "no": 954,
    "t": "                    pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH]),"
  },
  {
    "no": 955,
    "t": "                )"
  },
  {
    "no": 956,
    "t": "            post_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))"
  },
  {
    "no": 957,
    "t": "            post_inv_rms_col = pl.reshape(post_inv_rms, [BATCH, 1])"
  },
  {
    "no": 958,
    "t": "            inv_rms_tile = pl.assemble(inv_rms_tile, post_inv_rms_col, [0, 0])"
  },
  {
    "no": 960,
    "t": "        # Split-K gate + up interleaved."
  },
  {
    "no": 961,
    "t": "        for n_out in pl.parallel(MLP_ON):"
  },
  {
    "no": 962,
    "t": "            n0 = n_out * MLP_TN"
  },
  {
    "no": 963,
    "t": "            for k_split in pl.range(K_SPLITS_MLP):"
  },
  {
    "no": 964,
    "t": "                k0 = k_split * MLP_K_SLICE"
  },
  {
    "no": 965,
    "t": "                with pl.at("
  },
  {
    "no": 966,
    "t": "                    level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 967,
    "t": "                    name_hint=\"gate_proj\","
  },
  {
    "no": 968,
    "t": "                    deps=[cast_tids[k_split], gate_seed_tid],"
  },
  {
    "no": 969,
    "t": "                ) as gate_tid:"
  },
  {
    "no": 970,
    "t": "                    a0 = mlp_norm_in[:, k0 : k0 + MLP_INNER_TK]"
  },
  {
    "no": 971,
    "t": "                    w0 = w_gate[layer_hidden_base + k0 : layer_hidden_base + k0 + MLP_INNER_TK, n0 : n0 + MLP_TN]"
  },
  {
    "no": 972,
    "t": "                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)"
  },
  {
    "no": 973,
    "t": "                    for lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):"
  },
  {
    "no": 974,
    "t": "                        ks_off = lk * MLP_INNER_TK"
  },
  {
    "no": 975,
    "t": "                        a_k = mlp_norm_in[:, k0 + ks_off : k0 + ks_off + MLP_INNER_TK]"
  },
  {
    "no": 976,
    "t": "                        w_k = w_gate[layer_hidden_base + k0 + ks_off : layer_hidden_base + k0 + ks_off + MLP_INNER_TK, n0 : n0 + MLP_TN]"
  },
  {
    "no": 977,
    "t": "                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)"
  },
  {
    "no": 978,
    "t": "                    gate_acc_all = pl.assemble(gate_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)"
  },
  {
    "no": 979,
    "t": "                gate_tids[n_out * K_SPLITS_MLP + k_split] = gate_tid"
  },
  {
    "no": 980,
    "t": ""
  },
  {
    "no": 981,
    "t": "                with pl.at("
  },
  {
    "no": 982,
    "t": "                    level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 983,
    "t": "                    name_hint=\"up_proj\","
  },
  {
    "no": 984,
    "t": "                    deps=[cast_tids[k_split], up_seed_tid],"
  },
  {
    "no": 985,
    "t": "                ) as up_tid:"
  },
  {
    "no": 986,
    "t": "                    a0 = mlp_norm_in[:, k0 : k0 + MLP_INNER_TK]"
  },
  {
    "no": 987,
    "t": "                    w0 = w_up[layer_hidden_base + k0 : layer_hidden_base + k0 + MLP_INNER_TK, n0 : n0 + MLP_TN]"
  },
  {
    "no": 988,
    "t": "                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)"
  },
  {
    "no": 989,
    "t": "                    for lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):"
  },
  {
    "no": 990,
    "t": "                        ks_off = lk * MLP_INNER_TK"
  },
  {
    "no": 991,
    "t": "                        a_k = mlp_norm_in[:, k0 + ks_off : k0 + ks_off + MLP_INNER_TK]"
  },
  {
    "no": 992,
    "t": "                        w_k = w_up[layer_hidden_base + k0 + ks_off : layer_hidden_base + k0 + ks_off + MLP_INNER_TK, n0 : n0 + MLP_TN]"
  },
  {
    "no": 993,
    "t": "                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)"
  },
  {
    "no": 994,
    "t": "                    up_acc_all = pl.assemble(up_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)"
  },
  {
    "no": 995,
    "t": "                up_tids[n_out * K_SPLITS_MLP + k_split] = up_tid"
  },
  {
    "no": 998,
    "t": "        for n_out in pl.parallel(MLP_ON):"
  },
  {
    "no": 999,
    "t": "            n0 = n_out * MLP_TN"
  },
  {
    "no": 1000,
    "t": "            with pl.at("
  },
  {
    "no": 1001,
    "t": "                level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 1002,
    "t": "                name_hint=\"silu\","
  },
  {
    "no": 1003,
    "t": "                deps=["
  },
  {
    "no": 1004,
    "t": "                    reduce_tid,"
  },
  {
    "no": 1005,
    "t": "                    gate_tids[n_out * K_SPLITS_MLP + 0],"
  },
  {
    "no": 1006,
    "t": "                    gate_tids[n_out * K_SPLITS_MLP + 1],"
  },
  {
    "no": 1007,
    "t": "                    gate_tids[n_out * K_SPLITS_MLP + 2],"
  },
  {
    "no": 1008,
    "t": "                    gate_tids[n_out * K_SPLITS_MLP + 3],"
  },
  {
    "no": 1009,
    "t": "                    gate_tids[n_out * K_SPLITS_MLP + 4],"
  },
  {
    "no": 1010,
    "t": "                    up_tids[n_out * K_SPLITS_MLP + 0],"
  },
  {
    "no": 1011,
    "t": "                    up_tids[n_out * K_SPLITS_MLP + 1],"
  },
  {
    "no": 1012,
    "t": "                    up_tids[n_out * K_SPLITS_MLP + 2],"
  },
  {
    "no": 1013,
    "t": "                    up_tids[n_out * K_SPLITS_MLP + 3],"
  },
  {
    "no": 1014,
    "t": "                    up_tids[n_out * K_SPLITS_MLP + 4],"
  },
  {
    "no": 1015,
    "t": "                ],"
  },
  {
    "no": 1016,
    "t": "            ) as silu_tid:"
  },
  {
    "no": 1017,
    "t": "                inv_rms_chunk = inv_rms_tile[:, 0:1]"
  },
  {
    "no": 1018,
    "t": "                for sub in pl.pipeline(SILU_INNER_CHUNKS, stage=2):"
  },
  {
    "no": 1019,
    "t": "                    silu_off = n0 + sub * MLP_OUT_CHUNK"
  },
  {
    "no": 1020,
    "t": "                    gate_chunk = gate_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]"
  },
  {
    "no": 1021,
    "t": "                    up_chunk = up_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]"
  },
  {
    "no": 1022,
    "t": "                    scaled_gate = pl.row_expand_mul(gate_chunk, inv_rms_chunk)"
  },
  {
    "no": 1023,
    "t": "                    scaled_up = pl.row_expand_mul(up_chunk, inv_rms_chunk)"
  },
  {
    "no": 1024,
    "t": "                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(scaled_gate)), 1.0))"
  },
  {
    "no": 1025,
    "t": "                    mlp_chunk = pl.mul(pl.mul(scaled_gate, sigmoid), scaled_up)"
  },
  {
    "no": 1026,
    "t": "                    mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, silu_off])"
  },
  {
    "no": 1027,
    "t": "            silu_tids[n_out] = silu_tid"
  },
  {
    "no": 1028,
    "t": ""
  },
  {
    "no": 1029,
    "t": "        for n_out in pl.parallel(DOWN_ON):"
  },
  {
    "no": 1030,
    "t": "            n0 = n_out * DOWN_TN"
  },
  {
    "no": 1031,
    "t": "            for k_split in pl.range(K_SPLITS):"
  },
  {
    "no": 1032,
    "t": "                k0 = k_split * DOWN_TN"
  },
  {
    "no": 1033,
    "t": "                with pl.at("
  },
  {
    "no": 1034,
    "t": "                    level=pl.Level.CORE_GROUP,"
  },
  {
    "no": 1035,
    "t": "                    name_hint=\"down_proj\","
  },
  {
    "no": 1036,
    "t": "                    deps=[seed_tid, silu_tids[k_split]],"
  },
  {
    "no": 1037,
    "t": "                ) as down_tid:"
  },
  {
    "no": 1038,
    "t": "                    a0 = mlp_tile[:, k0 : k0 + DOWN_TK]"
  },
  {
    "no": 1039,
    "t": "                    w0 = w_down[layer_inter_base + k0 : layer_inter_base + k0 + DOWN_TK, n0 : n0 + DOWN_TN]"
  },
  {
    "no": 1040,
    "t": "                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)"
  },
  {
    "no": 1041,
    "t": "                    for lk in pl.pipeline(1, N_SUB_K, stage=2):"
  },
  {
    "no": 1042,
    "t": "                        ks_off = lk * DOWN_TK"
  },
  {
    "no": 1043,
    "t": "                        a_k = mlp_tile[:, k0 + ks_off : k0 + ks_off + DOWN_TK]"
  },
  {
    "no": 1044,
    "t": "                        w_k = w_down[layer_inter_base + k0 + ks_off : layer_inter_base + k0 + ks_off + DOWN_TK, n0 : n0 + DOWN_TN]"
  },
  {
    "no": 1045,
    "t": "                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)"
  },
  {
    "no": 1046,
    "t": "                    down_acc_all = pl.assemble(down_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)"
  },
  {
    "no": 1047,
    "t": "                down_tids[n_out * K_SPLITS + k_split] = down_tid"
  },
  {
    "no": 1079,
    "t": "    with pl.spmd("
  },
  {
    "no": 1080,
    "t": "        DOWN_ON,"
  },
  {
    "no": 1081,
    "t": "        name_hint=\"dcr_xgamma\","
  },
  {
    "no": 1082,
    "t": "        deps=[down_tids[i] for i in range(DOWN_ON * K_SPLITS)],"
  },
  {
    "no": 1083,
    "t": "    ) as dcr_tid:"
  },
  {
    "no": 1084,
    "t": "        n_out = pl.tile.get_block_idx()"
  },
  {
    "no": 1085,
    "t": "        n0 = n_out * DOWN_TN"
  },
  {
    "no": 1086,
    "t": "        # OUTPUT 1: layer residual (down_acc + post_norm, both FP32) -> `out` (cur)."
  },
  {
    "no": 1087,
    "t": "        out_chunk = pl.add("
  },
  {
    "no": 1088,
    "t": "            down_acc_all[:, n0 : n0 + DOWN_TN], post_norm_partial[:, n0 : n0 + DOWN_TN]"
  },
  {
    "no": 1089,
    "t": "        )"
  },
  {
    "no": 1090,
    "t": "        out = pl.assemble(out, out_chunk, [0, n0])"
  },
  {
    "no": 1091,
    "t": "        # OUTPUT 2: NEXT layer's x*gamma from the same in-register FP32 chunk (no GM"
  },
  {
    "no": 1092,
    "t": "        # re-read of `out`). gamma row clamped via next_gamma_idx (last layer unused)."
  },
  {
    "no": 1093,
    "t": "        gamma_next = pl.slice(input_rms_weight, [1, DOWN_TN], [next_gamma_idx, n0])"
  },
  {
    "no": 1094,
    "t": "        xg = pl.col_expand_mul(out_chunk, gamma_next)"
  },
  {
    "no": 1095,
    "t": "        normed_out = pl.assemble(normed_out, pl.cast(xg, target_type=pl.BF16), [0, n0])"
  },
  {
    "no": 1101,
    "t": "    for _slab in pl.unroll(DOWN_ON):"
  },
  {
    "no": 1102,
    "t": "        prev_out_tids[_slab] = dcr_tid"
  },
  {
    "no": 1103,
    "t": "        prev_normed_tids[_slab] = dcr_tid"
  },
  {
    "no": 1104,
    "t": "    return out"
  }
];
