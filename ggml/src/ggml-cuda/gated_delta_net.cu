#include "gated_delta_net.cuh"

template <int S_v, bool KDA>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

// Chunked prefill (non-KDA), single block per (h, seq), CS=32 across both S_v values
// for SMEM headroom on SM86/SM89. Algorithm follows build_delta_net_chunking in
// src/models/delta-net-base.cpp: precompute kb/kq/attn/k_cd per chunk, then run an
// inter-chunk inner loop over tokens using warp-cooperative dot products. MMA
// substitution lands in a follow-up commit; this commit establishes the algorithm.
constexpr int GDN_CHUNKED_THRESHOLD = 192; // TODO: tune via PP-{96..256} sweep
constexpr int GDN_CHUNKED_CS        = 32;

static bool gdn_chunked_eligible(int S_v, int64_t n_tokens, bool kda, int cc) {
    if (kda)                              return false; // KDA chunked is PR3
    if (n_tokens < GDN_CHUNKED_THRESHOLD) return false;
    if (S_v != 64 && S_v != 128)          return false;
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return false;                                       // PR1 chunked is NV-only
#else
    if (cc < GGML_CUDA_CC_AMPERE)         return false; // f32.tf32 MMA needs SM80+
    return true;
#endif
}

template <int S_v, int CS>
static constexpr size_t gdn_chunked_smem_bytes() {
    // sm_q + sm_k + sm_v (each CS*S_v) + 4 vectors of length CS (beta/g_cs/g_exp/g_diff)
    // + sm_kb + sm_attn + sm_kq (each CS*CS) + sm_k_cd (S_v*CS)
    return (3 * CS * S_v + 4 * CS + 3 * CS * CS + S_v * CS) * sizeof(float);
}

template <int S_v, int CS, bool KDA>
__launch_bounds__(128, 1)
__global__ void gated_delta_net_chunked_cuda(const float * q,
                                             const float * k,
                                             const float * v,
                                             const float * g,
                                             const float * beta,
                                             const float * curr_state,
                                             float *       dst,
                                             int64_t       H,
                                             int64_t       n_tokens,
                                             int64_t       n_seqs,
                                             int64_t       sq1,
                                             int64_t       sq2,
                                             int64_t       sq3,
                                             int64_t       sv1,
                                             int64_t       sv2,
                                             int64_t       sv3,
                                             int64_t       sb1,
                                             int64_t       sb2,
                                             int64_t       sb3,
                                             const uint3   neqk1_magic,
                                             const uint3   rq3_magic,
                                             float         scale) {
    static_assert(!KDA, "PR1 chunked path is non-KDA only");
    static_assert(S_v == 64 || S_v == 128, "PR1 supports S_v in {64, 128}");
    static_assert(CS == 32, "commit 3 locks CS=32; larger CS revisited with MMA");

    constexpr int warp_size     = 32;
    constexpr int num_warps     = 4;
    constexpr int tot_threads   = warp_size * num_warps;
    constexpr int cols_per_warp = S_v / num_warps;
    constexpr int rows_per_lane = S_v / warp_size;
    static_assert(cols_per_warp * num_warps == S_v, "S_v must be a multiple of num_warps");
    static_assert(rows_per_lane * warp_size == S_v, "S_v must be a multiple of warp_size");

    const int lane     = threadIdx.x;
    const int warp_id  = threadIdx.y;
    const int tid      = warp_id * warp_size + lane;
    const int h_idx    = blockIdx.x;
    const int sequence = blockIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    const int64_t state_offset     = (sequence * H + h_idx) * S_v * S_v;
    float *       attn_data_base   = dst + (sequence * n_tokens * H + h_idx) * S_v;
    float *       state_out        = dst + attn_score_elems + state_offset;
    const float * state_in         = curr_state + state_offset;

    // SMEM layout
    extern __shared__ float smem[];
    float * sm_q      = smem;                          // CS * S_v   (chunk q, pre-scaled)
    float * sm_k      = sm_q      + CS * S_v;          // CS * S_v
    float * sm_v      = sm_k      + CS * S_v;          // CS * S_v   (v -> v_chunk -> v_new)
    float * sm_beta   = sm_v      + CS * S_v;          // CS
    float * sm_g_cs   = sm_beta   + CS;                // CS (cumsum of g, clamped)
    float * sm_g_exp  = sm_g_cs   + CS;                // CS (= exp(g_cs))
    float * sm_g_diff = sm_g_exp  + CS;                // CS (= exp(g_cs[CS-1] - g_cs))
    float * sm_kb     = sm_g_diff + CS;                // CS * CS  (G1 output)
    float * sm_attn   = sm_kb     + CS * CS;           // CS * CS  (WY solve output, + I)
    float * sm_kq     = sm_attn   + CS * CS;           // CS * CS  (G2 output, tril)
    float * sm_k_cd   = sm_kq     + CS * CS;           // S_v * CS (G5 output)

    // Register state: same sharding as commit 2
    float s_shard[cols_per_warp][rows_per_lane];
    #pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
        const int col = warp_id * cols_per_warp + c;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            s_shard[c][r] = state_in[col * S_v + r * warp_size + lane];
        }
    }

    const int n_chunks = ((int) n_tokens + CS - 1) / CS;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int t_start   = chunk * CS;
        const int chunk_len = ((int) n_tokens - t_start < CS) ? ((int) n_tokens - t_start) : CS;

        // Phase 1: load chunk q (pre-scaled), k, v, beta, g
        for (int idx = tid; idx < CS * S_v; idx += tot_threads) {
            const int t   = idx / S_v;
            const int i   = idx % S_v;
            const int t_g = t_start + t;
            if (t < chunk_len) {
                sm_q[idx] = q[iq3 * sq3 + t_g * sq2 + iq1 * sq1 + i] * scale;
                sm_k[idx] = k[iq3 * sq3 + t_g * sq2 + iq1 * sq1 + i];
                sm_v[idx] = v[sequence * sv3 + t_g * sv2 + h_idx * sv1 + i];
            } else {
                sm_q[idx] = 0.0f;
                sm_k[idx] = 0.0f;
                sm_v[idx] = 0.0f;
            }
        }
        if (tid < CS) {
            const int t_g = t_start + tid;
            if (tid < chunk_len) {
                const int64_t gb_off = sequence * sb3 + t_g * sb2 + h_idx * sb1;
                sm_beta[tid] = beta[gb_off];
                sm_g_cs[tid] = g[gb_off];
            } else {
                sm_beta[tid] = 0.0f;
                sm_g_cs[tid] = 0.0f;
            }
        }
        __syncthreads();

        // Phase 2: g_cs = clamp(cumsum(g), 50); g_exp = exp(g_cs); g_diff = exp(g_cs[last] - g_cs)
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < CS; ++t) {
                acc += sm_g_cs[t];
                if (acc > 50.0f) acc = 50.0f;
                sm_g_cs[t]  = acc;
                sm_g_exp[t] = expf(acc);
            }
            const float g_last_log = sm_g_cs[CS - 1];
            for (int t = 0; t < CS; ++t) {
                sm_g_diff[t] = expf(fminf(g_last_log - sm_g_cs[t], 50.0f));
            }
        }
        __syncthreads();

        // Phase 3: G1   kb[i, j] = decay[i, j] * sum_d (k[i, d] * beta[i]) * k[j, d]   for j <= i
        //   decay[i, j] = exp(g_cs[i] - g_cs[j]) so older keys (j < i) attenuate; this matches
        //   delta-net-base.cpp where the ggml layout puts the query position on the row axis.
        for (int idx = tid; idx < CS * CS; idx += tot_threads) {
            const int i = idx / CS;
            const int j = idx % CS;
            float val = 0.0f;
            if (j <= i) {
                const float decay = expf(fminf(sm_g_cs[i] - sm_g_cs[j], 50.0f));
                float dot = 0.0f;
                for (int d = 0; d < S_v; ++d) {
                    dot += sm_k[i * S_v + d] * sm_k[j * S_v + d];
                }
                val = dot * sm_beta[i] * decay;
            }
            sm_kb[idx] = val;
        }
        __syncthreads();

        // Phase 4: G2   kq[i, j] = decay[i, j] * sum_d q[i, d] * k[j, d]   for j <= i (tril)
        for (int idx = tid; idx < CS * CS; idx += tot_threads) {
            const int i = idx / CS;
            const int j = idx % CS;
            float val = 0.0f;
            if (j <= i) {
                const float decay = expf(fminf(sm_g_cs[i] - sm_g_cs[j], 50.0f));
                float dot = 0.0f;
                for (int d = 0; d < S_v; ++d) {
                    dot += sm_q[i * S_v + d] * sm_k[j * S_v + d];
                }
                val = dot * decay;
            }
            sm_kq[idx] = val;
        }
        __syncthreads();

        // Phase 5: G3   WY solve.  attn = solve_tri(I + tril(kb,-1), -tril(kb,-1)) + I
        //   - lhs[r, k]  = kb[r, k] for k < r,  1 if k == r,  0 otherwise (diag handled implicitly)
        //   - rhs[r, c]  = -kb[r, c] for c < r,  0 otherwise
        //   - X[r, c]   := rhs[r, c] - sum_{k<r} kb[r, k] * X[k, c]   (diag of lhs is 1)
        //   - attn       = X + I
        for (int idx = tid; idx < CS * CS; idx += tot_threads) {
            const int i = idx / CS;
            const int j = idx % CS;
            sm_attn[idx] = (j < i) ? -sm_kb[idx] : 0.0f;
        }
        __syncthreads();

        // Sequential forward substitution. CS=32 columns map to one warp; row-by-row update.
        if (warp_id == 0) {
            for (int r = 1; r < CS; ++r) {
                const int c = lane; // CS == warp_size == 32
                float x = sm_attn[r * CS + c];
                for (int kk = 0; kk < r; ++kk) {
                    x -= sm_kb[r * CS + kk] * sm_attn[kk * CS + c];
                }
                sm_attn[r * CS + c] = x;
                __syncwarp();
            }
        }
        __syncthreads();
        if (tid < CS) {
            sm_attn[tid * CS + tid] += 1.0f;  // attn = X + I
        }
        __syncthreads();

        // Phase 6: G4   v_chunk[t, d] = sum_j attn[t, j] * v_b[j, d],  v_b[j, d] = v[j, d] * beta[j]
        // In-place rewrite of sm_v (two-pass via registers to avoid intra-block races).
        constexpr int per_thread_v = (CS * S_v + tot_threads - 1) / tot_threads;
        float v_chunk_reg[per_thread_v];
        #pragma unroll
        for (int p = 0; p < per_thread_v; ++p) {
            const int idx = tid + p * tot_threads;
            float acc = 0.0f;
            if (idx < CS * S_v) {
                const int t = idx / S_v;
                const int d = idx % S_v;
                for (int j = 0; j < CS; ++j) {
                    acc += sm_attn[t * CS + j] * sm_v[j * S_v + d] * sm_beta[j];
                }
            }
            v_chunk_reg[p] = acc;
        }
        __syncthreads();
        #pragma unroll
        for (int p = 0; p < per_thread_v; ++p) {
            const int idx = tid + p * tot_threads;
            if (idx < CS * S_v) {
                sm_v[idx] = v_chunk_reg[p];
            }
        }
        __syncthreads();

        // Phase 7: G5   k_cd[d, i] = sum_j (k[j, d] * beta[j] * g_exp[j]) * attn[j, i]
        for (int idx = tid; idx < S_v * CS; idx += tot_threads) {
            const int d = idx / CS;
            const int i = idx % CS;
            float acc = 0.0f;
            for (int j = 0; j < CS; ++j) {
                acc += sm_k[j * S_v + d] * sm_beta[j] * sm_g_exp[j] * sm_attn[j * CS + i];
            }
            sm_k_cd[idx] = acc;
        }
        __syncthreads();

        // Phase 8: inter-chunk inner loop.  For each t in chunk:
        //   v_prime[col]    = sum_d k_cd[d, t] * S[d, col]
        //   v_new[t, col]   = v_chunk[t, col] - v_prime[col]                (overwrites sm_v[t, col])
        //   attn_inter[col] = g_exp[t] * sum_d q[t, d] * S[d, col]          (with pre-scaled q)
        //   v_attn[col]     = sum_{j<=t} kq[t, j] * v_new[j, col]
        //   output[t, col]  = attn_inter[col] + v_attn[col]
        for (int t = 0; t < chunk_len; ++t) {
            const float   g_exp_t     = sm_g_exp[t];
            float *       attn_data_t = attn_data_base + (t_start + t) * S_v * H;
            float         attn_inter_reg[cols_per_warp];

            #pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                const int col = warp_id * cols_per_warp + c;

                float vp = 0.0f;
                float ai = 0.0f;
                #pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int d = r * warp_size + lane;
                    vp += sm_k_cd[d * CS + t] * s_shard[c][r];
                    ai += sm_q[t * S_v + d]   * s_shard[c][r];
                }
                const float vp_col = warp_reduce_sum<warp_size>(vp);
                attn_inter_reg[c]  = warp_reduce_sum<warp_size>(ai) * g_exp_t;

                if (lane == 0) {
                    sm_v[t * S_v + col] -= vp_col;  // sm_v[t, col] was v_chunk -> now v_new
                }
            }
            __syncthreads();  // ensure v_new[t, *] visible across warps for v_attn read

            #pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                const int col = warp_id * cols_per_warp + c;

                float va = 0.0f;
                for (int j = lane; j <= t; j += warp_size) {
                    va += sm_kq[t * CS + j] * sm_v[j * S_v + col];
                }
                const float v_attn_col = warp_reduce_sum<warp_size>(va);

                if (lane == 0) {
                    attn_data_t[col] = attn_inter_reg[c] + v_attn_col;
                }
            }
        }
        __syncthreads();

        // Phase 9: G9   S[d, col] = g_last * S[d, col] + sum_t (k[t, d] * g_diff[t]) * v_new[t, col]
        const float g_last = sm_g_exp[CS - 1];
        #pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
            const int col = warp_id * cols_per_warp + c;
            #pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int d   = r * warp_size + lane;
                float     kgv = 0.0f;
                for (int t = 0; t < chunk_len; ++t) {
                    kgv += sm_k[t * S_v + d] * sm_g_diff[t] * sm_v[t * S_v + col];
                }
                s_shard[c][r] = g_last * s_shard[c][r] + kgv;
            }
        }
        __syncthreads();
    }

    // Write final state back to gmem (transposed layout, matches sequential)
    #pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
        const int col = warp_id * cols_per_warp + c;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            state_out[col * S_v + r * warp_size + lane] = s_shard[c][r];
        }
    }
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (gdn_chunked_eligible((int) S_v, n_tokens, KDA, cc)) {
        const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
        dim3 ck_grid(H, n_seqs, 1);
        dim3 ck_block(warp_size, num_warps, 1);
        constexpr int CS = GDN_CHUNKED_CS;
        if constexpr (!KDA) {
            if (S_v == 64) {
                constexpr size_t smem = gdn_chunked_smem_bytes<64, CS>();
                if (smem <= smpbo) {
                    CUDA_SET_SHARED_MEMORY_LIMIT((gated_delta_net_chunked_cuda<64, CS, false>), smem);
                    gated_delta_net_chunked_cuda<64, CS, false><<<ck_grid, ck_block, smem, stream>>>(
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    return;
                }
            }
            if (S_v == 128) {
                constexpr size_t smem = gdn_chunked_smem_bytes<128, CS>();
                if (smem <= smpbo) {
                    CUDA_SET_SHARED_MEMORY_LIMIT((gated_delta_net_chunked_cuda<128, CS, false>), smem);
                    gated_delta_net_chunked_cuda<128, CS, false><<<ck_grid, ck_block, smem, stream>>>(
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    return;
                }
            }
        }
        // smem > smpbo on this device — fall through to sequential
    }

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            gated_delta_net_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            gated_delta_net_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }
}
