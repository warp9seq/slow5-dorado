#include "openfish.h"

#include <stdio.h>

int koi_matmul_hopper(void *stream, void *a_bfr, void *b_bfr, void *c_bfr, int M, int N, int K) {
    fprintf(stderr, "openfish: koi_matmul_hopper not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_swiglu_hopper(void *stream, void *a_bfr, void *b_bfr, void *c_bfr, int M, int N, int K) {
    fprintf(stderr, "openfish: koi_swiglu_hopper not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_rmsnorm_hopper(void *stream, void *in_bfr, void *res_bfr, void *weights_bfr,
                       void *out_bfr_fp16, void *out_bfr_fp8_i8, void *out_scale_float,
                       int M, int C_, float alpha, int ampere_layout_in) {
    fprintf(stderr, "openfish: koi_rmsnorm_hopper not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_hopper_tc_is_available(enum KoiTypeId type_id) {
    fprintf(stderr, "openfish: koi_hopper_tc_is_available not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_volta_linear(void *stream, void *in_a, void *in_b, void *in_c, void *in_bias,
                     int float_accum, int use_bias, int M, int K) {
    fprintf(stderr, "openfish: koi_volta_linear not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_volta_mm_swiglu(void *stream, void *in_a, void *in_b, void *in_c, int float_accum, int M) {
    fprintf(stderr, "openfish: koi_volta_mm_swiglu not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_volta_qkv_rotary(void *stream, void *in_a, void *in_b, void *in_c, void *in_sincos_ptr,
                          int float_accum, int batch_size, int T) {
    fprintf(stderr, "openfish: koi_volta_qkv_rotary not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_volta_rmsnorm_residual(void *stream, void *in_residual, void *in_input, void *weights_ptr,
                               void *in_c, float alpha, int M) {
    fprintf(stderr, "openfish: koi_volta_rmsnorm_residual not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_volta_attn(void *stream, void *in_qkv, void *in_out, int batch_size, int T) {
    fprintf(stderr, "openfish: koi_volta_attn not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_volta_tc_is_available(enum KoiTypeId type_id) {
    fprintf(stderr, "openfish: koi_volta_tc_is_available not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_linear(void *stream, KoiTensor *a, KoiTensor *b, KoiTensor *bias, KoiTensor *out,
               int *ctr) {
    fprintf(stderr, "openfish: koi_linear not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_mm_swiglu(void *stream, KoiTensor *a, KoiTensor *b, KoiTensor *out, int *ctr,
                  int use_f32_accum) {
    fprintf(stderr, "openfish: koi_mm_swiglu not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_rmsnorm_residual(void *stream, KoiTensor *in, KoiTensor *residual, float alpha,
                         KoiTensor *weights, KoiTensor *out_f16, KoiTensor *out_f8,
                         KoiTensor *out_i8) {
    fprintf(stderr, "openfish: koi_rmsnorm_residual not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_qkv_rotary(void *stream, float theta, KoiTensor *in, KoiTensor *qkv_weights,
                   KoiTensor *sincos, KoiTensor *out_qkv, int *ctr) {
    fprintf(stderr, "openfish: koi_qkv_rotary not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_masked_attention(void *stream, int win_upper, int win_lower, KoiTensor *qkv,
                         KoiTensor *out) {
    fprintf(stderr, "openfish: koi_masked_attention not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int koi_tc_is_available(enum KoiTypeId type_id) {
    fprintf(stderr, "openfish: koi_tc_is_available not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_masked_attention_f16(void *stream, int N, int T, int H, int D, int win_upper,
                              int win_lower, void *q, void *k, void *v, void *out) {
    fprintf(stderr, "openfish: host_masked_attention_f16 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_fused_residual_rmsnorm_f16(void *stream, int layer_size, int rows, void *in,
                                    void *residual, void *alpha, void *weights, void *out) {
    fprintf(stderr, "openfish: host_fused_residual_rmsnorm_f16 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_rotary_embed_transpose_f16(void *stream, int N, int T, int H, int D, float theta,
                                    void *in, void *out) {
    fprintf(stderr, "openfish: host_rotary_embed_transpose_f16 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_linear_swiglu_f16(void *stream, int M, int N, int K, void *in, void *weights, void *out) {
    fprintf(stderr, "openfish: host_linear_swiglu_f16 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_linear(void *stream, enum KoiTypeId type_id_in, enum KoiActivation activation,
                enum KoiTypeId type_id_out, int size0, int size1, int C_in, int C_out,
                int in_stride0, int in_stride1, int out_stride0, int out_stride1, void *in,
                void *weights, void *out, void *weight_scale, void *bias) {
    fprintf(stderr, "openfish: host_linear not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_cublas_gemm_f16(void *cublas_h, int m, int n, int k, int flags, void *A, void *B,
                         void *C) {
    fprintf(stderr, "openfish: host_cublas_gemm_f16 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_small_lstm(void *stream, int num_chunks, int chunk_size, int layer_size, int direction,
                    void *outvW, void *lW, void *b, void *quantization_scale, void *out) {
    fprintf(stderr, "openfish: host_small_lstm not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

void host_lstm_step_f16(void *stream, int batch_size, int layer_size, void *bias, void *gate_buf,
                        void *state_buf, void *out) {
    fprintf(stderr, "openfish: host_lstm_step_f16 not yet implemented\n");
}

void host_cutlass_lstm(void *stream, enum KoiTypeId type, int layer_idx, int batch_size,
                       int layer_size, int chunk_size, int direction, int inout_stride, void *inout,
                       void *weights, void *bias, void *scale, void *lstm_state,
                       void *workspace_4KiB, int interleave, int brf_state) {
    fprintf(stderr, "openfish: host_cutlass_lstm not yet implemented\n");
}

int host_convert(void *stream, void *in, int in_stride0, int in_stride1, int in_stride2,
                 enum KoiTypeId in_type, void *out, int out_stride0, int out_stride1,
                 int out_stride2, enum KoiTypeId out_type, int size0, int size1, int size2) {
    fprintf(stderr, "openfish: host_convert not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_bias_activation_f16_inplace(void *stream, int TN, int C, int stride, void *in_out,
                                     void *bias, enum KoiActivation activation) {
    fprintf(stderr, "openfish: host_bias_activation_f16_inplace not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_bias_activation_interleave_convert(void *stream, int T, int N, int C, void *in,
                                            int in_interleave, enum KoiTypeId in_type_id, void *out,
                                            int out_interleave, enum KoiTypeId out_type_id,
                                            void *bias, enum KoiActivation activation) {
    fprintf(stderr, "openfish: host_bias_activation_interleave_convert not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_window_ntwc_f16(void *stream, int N, int T_in, int C, int W, int conv_stride,
                         int N_out_stride, int T_out_stride, void *in_buf, void *out_buf) {
    fprintf(stderr, "openfish: host_window_ntwc_f16 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_convolution_f16(void *stream, int N, int C_in, int C_out, int T_in, int window, int stride,
                         int padding, int out_N_stride, void *in_buf, void *out_buf, void *weights,
                         void *bias, enum KoiActivation activation) {
    fprintf(stderr, "openfish: host_convolution_f16 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_back_guide_step(void *stream, void *chunks, void *chunk_results, int num_chunks,
                         void *post, float post_clamp, int post_stride, void *aux_buffer,
                         void *path, void *moves, void *weights, void *sequence, void *q_string,
                         float qscale, float qshift, int beam_width, float beam_cut,
                         float fixed_stay_score) {
    fprintf(stderr, "openfish: host_back_guide_step not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_beam_search_step(void *stream, void *chunks, void *chunk_results, int num_chunks,
                          void *post, float post_clamp, int post_stride, void *aux_buffer,
                          void *path, void *moves, void *weights, void *sequence, void *q_string,
                          float qscale, float qshift, int beam_width, float beam_cut,
                          float fixed_stay_score) {
    fprintf(stderr, "openfish: host_beam_search_step not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_compute_posts_step(void *stream, void *chunks, void *chunk_results, int num_chunks,
                            void *post, float post_clamp, int post_stride, void *aux_buffer,
                            void *path, void *moves, void *weights, void *sequence, void *q_string,
                            float qscale, float qshift, int beam_width, float beam_cut,
                            float fixed_stay_score) {
    fprintf(stderr, "openfish: host_compute_posts_step not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int host_run_decode(void *stream, void *chunks, void *chunk_results, int num_chunks, void *post,
                    float post_clamp, int post_stride, void *aux_buffer, void *path, void *moves,
                    void *weights, void *sequence, void *q_string, float qscale, float qshift,
                    int beam_width, float beam_cut, float fixed_stay_score, int move_pad) {
    fprintf(stderr, "openfish: host_run_decode not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int fwd_bwd_logspace_host(int T, int N, int L, void *alpha, void *beta_T, void *beta_stay,
                          void *beta_move, void *stay_scores, void *move_scores) {
    fprintf(stderr, "openfish: fwd_bwd_logspace_host not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int fwd_bwd_logspace_loop_host(int T, int N, int L, void *alpha, void *beta_T, void *beta_stay,
                                void *beta_move, void *stay_scores, void *move_scores) {
    fprintf(stderr, "openfish: fwd_bwd_logspace_loop_host not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int logZ_fwd_host_log_NZ5(int T, int N, int C, int k, float *logZ, float *Ms_grad, float *Ms,
                           float *v0, float *vT, int *idx) {
    fprintf(stderr, "openfish: logZ_fwd_host_log_NZ5 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int logZ_fwd_host_log_NZ3(int T, int N, int C, int k, float *logZ, float *Ms_grad, float *Ms,
                           float *v0, float *vT, int *idx) {
    fprintf(stderr, "openfish: logZ_fwd_host_log_NZ3 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int logZ_fwd_host_max(int T, int N, int C, int k, float *logZ, float *Ms_grad, float *Ms,
                      float *v0, float *vT, int *idx) {
    fprintf(stderr, "openfish: logZ_fwd_host_max not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int bwd_scores_host_sparse_log_NZ5(float *betas, float *Ms, float *vT, int *idx_T, int T, int N,
                                    int C) {
    fprintf(stderr, "openfish: bwd_scores_host_sparse_log_NZ5 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int bwd_scores_host_sparse_log_NZ3(float *betas, float *Ms, float *vT, int *idx_T, int T, int N,
                                    int C) {
    fprintf(stderr, "openfish: bwd_scores_host_sparse_log_NZ3 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int bwd_scores_host_sparse_max(float *betas, float *Ms, float *vT, int *idx_T, int T, int N,
                                int C) {
    fprintf(stderr, "openfish: bwd_scores_host_sparse_max not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int fwd_scores_host_sparse_log_NZ5(float *alphas, float *Ms, float *v0, int *idx, int T, int N,
                                    int C) {
    fprintf(stderr, "openfish: fwd_scores_host_sparse_log_NZ5 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int fwd_scores_host_sparse_log_NZ3(float *alphas, float *Ms, float *v0, int *idx, int T, int N,
                                    int C) {
    fprintf(stderr, "openfish: fwd_scores_host_sparse_log_NZ3 not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}

int fwd_scores_host_sparse_max(float *alphas, float *Ms, float *v0, int *idx, int T, int N,
                                int C) {
    fprintf(stderr, "openfish: fwd_scores_host_sparse_max not yet implemented\n");
    return KOI_NOT_SUPPORTED;
}
