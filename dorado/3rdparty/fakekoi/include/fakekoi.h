#pragma once

enum KoiTypeId { KOI_NONE = 0, KOI_I8, KOI_E4M3, KOI_F16, KOI_F32 };
enum KoiActivation { KOI_IDENTITY = 0, KOI_SWISH, KOI_SWISH_CLAMP, KOI_TANH, KOI_TANH_X5 };

enum KoiResult {
    KOI_SUCCESS = 0,
    KOI_NOT_SUPPORTED,  // Operation with given args is not supported on current device/platform
    KOI_INVALID_VALUE,  // Operation with given args is not supported at all
    KOI_INTERNAL_ERROR  // Any other error
};

// Special tags describing the dimensions of a tensor passed to Koi. Usually the tag will be
// a single char such as 'N' (batch dimension), 'T' (time dimension), etc., but some dimensions
// don't have an obvious single-letter descriptor.
enum KoiDimTag {
    KOI_TENSOR_MAX_DIMS = 8,  // Just a constant, not a tag

    KOI_DIM_MAT_Q_K_V = 0,
    KOI_DIM_GATE_IFGO,
    KOI_DIM_MAT_W_U
};

typedef struct KoiTensorT {
    void *data_ptr;
    int ndims;
    enum KoiTypeId type_id;
    struct Dim {
        int tag;
        long long size;
        long long stride;
    } dims[KOI_TENSOR_MAX_DIMS];
    void *scale_data;
    int scale_tag;
    enum KoiTypeId scale_type_id;
} KoiTensor;

int koi_matmul_hopper(void *stream, void *a_bfr, void *b_bfr, void *c_bfr, int M, int N, int K);

int koi_swiglu_hopper(void *stream, void *a_bfr, void *b_bfr, void *c_bfr, int M, int N, int K);

int koi_rmsnorm_hopper(
    void *stream,
    void *in_bfr,
    void *res_bfr,
    void *weights_bfr,
    void *out_bfr_fp16,
    void *out_bfr_fp8_i8,
    void *out_scale_float,
    int M,
    int C_,
    float alpha,
    int ampere_layout_in);

int koi_hopper_tc_is_available(enum KoiTypeId type_id);

int koi_volta_linear(void *stream,
                     void *in_a,
                     void *in_b,
                     void *in_c,
                     void *in_bias,
                     int float_accum,
                     int use_bias,
                     int M,
                     int K);

int koi_volta_mm_swiglu(void *stream, void *in_a, void *in_b, void *in_c, int float_accum, int M);

int koi_volta_qkv_rotary(void *stream,
                         void *in_a,
                         void *in_b,
                         void *in_c,
                         void *in_sincos_ptr,
                         int float_accum,
                         int batch_size,
                         int T);

int koi_volta_rmsnorm_residual(void *stream,
                               void *in_residual,
                               void *in_input,
                               void *weights_ptr,
                               void *in_c,
                               float alpha,
                               int M);

int koi_volta_attn(void *stream, void *in_qkv, void *in_out, int batch_size, int T);

// ! Adding an argument although not used, if not, it does not link successfully.
int koi_volta_tc_is_available(enum KoiTypeId type_id);

int koi_linear(void *stream, KoiTensor *a, KoiTensor *b, KoiTensor *bias, KoiTensor *out, int *ctr);

int koi_mm_swiglu(void *stream,
                  KoiTensor *a,
                  KoiTensor *b,
                  KoiTensor *out,
                  int *ctr,
                  int use_f32_accum);

int koi_rmsnorm_residual(void *stream,
                         KoiTensor *in,
                         KoiTensor *residual,
                         float alpha,
                         KoiTensor *weights,
                         KoiTensor *out_f16,
                         KoiTensor *out_f8,
                         KoiTensor *out_i8);

int koi_qkv_rotary(void *stream,
                   float theta,
                   KoiTensor *in,
                   KoiTensor *qkv_weights,
                   KoiTensor *sincos,
                   KoiTensor *out_qkv,
                   int *ctr);

int koi_masked_attention(void *stream,
                         int win_upper,
                         int win_lower,
                         KoiTensor *qkv,
                         KoiTensor *out);

int koi_tc_is_available(enum KoiTypeId type_id);

// TODO: Update all `host_*` functions below to use KoiTensor inputs, combining similar fns

int host_masked_attention_f16(void *stream,
                              int N,
                              int T,
                              int H,
                              int D,
                              int win_upper,
                              int win_lower,
                              void *q,
                              void *k,
                              void *v,
                              void *out);

int host_fused_residual_rmsnorm_f16(void *stream,
                                    int layer_size,
                                    int rows,
                                    void *in,
                                    void *residual,
                                    void *alpha,
                                    void *weights,
                                    void *out);

// `in` is [N, T, q|k|v, H, D], `out` is [q|k|v, N, H, T, D]
int host_rotary_embed_transpose_f16(void *stream,
                                    int N,
                                    int T,
                                    int H,
                                    int D,
                                    float theta,
                                    void *in,
                                    void *out);
// `in` is [M, K], `weights` is [N, K], `out` is [M, N]
int host_linear_swiglu_f16(void *stream, int M, int N, int K, void *in, void *weights, void *out);
/// host_linear
///
/// Perform
/// - matrix multiplication (either [int8 * int8 -> int32] or [half * half -> half]),
/// - followed by conversion to float,
/// - pointwise application of scale vector (stored in half format)
/// - pointwise application of bias vector (stored in half format)
/// - application of an activation function,
/// - and conversion to output type (cast to half if output type is KOI_F16, or mapping the float
///   range [-1.0, 1.0] to [-127, 127] if output type is KOI_I8)
///
/// ```
/// auto mm = convert<Accum_T, float>(matmul<In_T, Accum_T>(in, weights)) / weight_scale;
/// out = convert<float, out_t>(activation(mm + bias))
/// ```
///
/// \param stream cudaStream_t
/// \param type_id_in Dtype of both `in` and `weights`. Either KOI_I8 or KOI_F16.
/// \param activation Activation function to be applied after scale and bias
/// \param type_id_out Dtype of `out`
/// \param size0 `in.size(0)`
/// \param size1 `in.size(1)`
/// \param C_in layer input size (`in.size(2)`)
/// \param C_out layer output size (`out.size(2)`)
/// \param in_stride0 `in.stride(0)`
/// \param in_stride1 `in.stride(1)`
/// \param in `.data_ptr()` of tensor of size [`size0`, `size1`, `C_in`], strides
///         [`in_stride0`, `in_stride1`, 1], dtype matches `type_id_in`
/// \param weights `.data_ptr()` of tensor of size [`C_out`, `C_in`], contiguous, (i.e. strides are
///         [`C_in`, 1]), dtype matches `type_id_in`
/// \param out `data_ptr()` of tensor of size [`size0`, `size1`, 'C_out`], contiguous, (i.e. strides
///         are [`size1 * C_out`, `C_out`, 1]), dtype matches `type_id_out`
/// \param weight_scale `.data_ptr()` of tensor of size [`C_out`], stride [1], dtype float16.
///         May be nullptr (equivalent to vector of all ones)
/// \param bias `.data_ptr() of tensor of size [`C_out`], stride [1], dtype float16.
///         May be nullptr (equivalent to vector of all zeroes)
/// \return KOI_SUCCESS on success, other KoiResult values otherwise
int host_linear(void *stream,
                enum KoiTypeId type_id_in,
                enum KoiActivation activation,
                enum KoiTypeId type_id_out,
                int size0,
                int size1,
                int C_in,
                int C_out,
                int in_stride0,
                int in_stride1,
                int out_stride0,
                int out_stride1,
                void *in,
                void *weights,
                void *out,
                void *weight_scale,
                void *bias);

int host_cublas_gemm_f16(void *cublas_h, int m, int n, int k, int flags, void *A, void *B, void *C);

int host_small_lstm(void *stream,
                    int num_chunks,
                    int chunk_size,
                    int layer_size,
                    int direction,
                    void *outvW,
                    void *lW,
                    void *b,
                    void *quantization_scale,
                    void *out);

void host_lstm_step_f16(void *stream,
                        int batch_size,
                        int layer_size,
                        void *bias,
                        void *gate_buf,
                        void *state_buf,
                        void *out);

void host_cutlass_lstm(void *stream,
                       enum KoiTypeId type,
                       int layer_idx,
                       int batch_size,
                       int layer_size,
                       int chunk_size,
                       int direction,
                       int inout_stride,
                       void *inout,
                       void *weights,
                       void *bias,
                       void *scale,
                       void *lstm_state,
                       void *workspace_4KiB,
                       int interleave,
                       int brf_state);

int host_convert(void *stream,
                 void *in,
                 int in_stride0,
                 int in_stride1,
                 int in_stride2,
                 enum KoiTypeId in_type,
                 void *out,
                 int out_stride0,
                 int out_stride1,
                 int out_stride2,
                 enum KoiTypeId out_type,
                 int size0,
                 int size1,
                 int size2);

int host_bias_activation_f16_inplace(void *stream,
                                     int TN,
                                     int C,
                                     int stride,
                                     void *in_out,
                                     void *bias,
                                     enum KoiActivation activation);

int host_bias_activation_interleave_convert(void *stream,
                                            int T,
                                            int N,
                                            int C,
                                            void *in,
                                            int in_interleave,
                                            enum KoiTypeId in_type_id,
                                            void *out,
                                            int out_interleave,
                                            enum KoiTypeId out_type_id,
                                            void *bias,
                                            enum KoiActivation activation);

int host_window_ntwc_f16(void *stream,
                         int N,
                         int T_in,
                         int C,
                         int W,
                         int conv_stride,
                         int N_out_stride,
                         int T_out_stride,
                         void *in_buf,
                         void *out_buf);

int host_convolution_f16(void *stream,
                         int N,
                         int C_in,
                         int C_out,
                         int T_in,
                         int window,
                         int stride,
                         int padding,
                         int out_N_stride,
                         void *in_buf,
                         void *out_buf,
                         void *weights,
                         void *bias,
                         enum KoiActivation activation);

int host_back_guide_step(void *stream,
                         void *chunks,
                         void *chunk_results,
                         int num_chunks,
                         void *post,
                         float post_clamp,
                         int post_stride,
                         void *aux_buffer,
                         void *path,
                         void *moves,
                         void *weights,
                         void *sequence,
                         void *q_string,
                         float qscale,
                         float qshift,
                         int beam_width,
                         float beam_cut,
                         float fixed_stay_score);

int host_beam_search_step(void *stream,
                          void *chunks,
                          void *chunk_results,
                          int num_chunks,
                          void *post,
                          float post_clamp,
                          int post_stride,
                          void *aux_buffer,
                          void *path,
                          void *moves,
                          void *weights,
                          void *sequence,
                          void *q_string,
                          float qscale,
                          float qshift,
                          int beam_width,
                          float beam_cut,
                          float fixed_stay_score);

int host_compute_posts_step(void *stream,
                            void *chunks,
                            void *chunk_results,
                            int num_chunks,
                            void *post,
                            float post_clamp,
                            int post_stride,
                            void *aux_buffer,
                            void *path,
                            void *moves,
                            void *weights,
                            void *sequence,
                            void *q_string,
                            float qscale,
                            float qshift,
                            int beam_width,
                            float beam_cut,
                            float fixed_stay_score);

int host_run_decode(void *stream,
                    void *chunks,
                    void *chunk_results,
                    int num_chunks,
                    void *post,
                    float post_clamp,
                    int post_stride,
                    void *aux_buffer,
                    void *path,
                    void *moves,
                    void *weights,
                    void *sequence,
                    void *q_string,
                    float qscale,
                    float qshift,
                    int beam_width,
                    float beam_cut,
                    float fixed_stay_score,
                    int move_pad);

int fwd_bwd_logspace_host(int T,
                          int N,
                          int L,
                          void *alpha,
                          void *beta_T,
                          void *beta_stay,
                          void *beta_move,
                          void *stay_scores,
                          void *move_scores);

int fwd_bwd_logspace_loop_host(int T,
                               int N,
                               int L,
                               void *alpha,
                               void *beta_T,
                               void *beta_stay,
                               void *beta_move,
                               void *stay_scores,
                               void *move_scores);

int logZ_fwd_host_log_NZ5(int T,
                          int N,
                          int C,
                          int k,
                          float *logZ,
                          float *Ms_grad,
                          float *Ms,
                          float *v0,
                          float *vT,
                          int *idx);

int logZ_fwd_host_log_NZ3(int T,
                          int N,
                          int C,
                          int k,
                          float *logZ,
                          float *Ms_grad,
                          float *Ms,
                          float *v0,
                          float *vT,
                          int *idx);

int logZ_fwd_host_max(int T,
                      int N,
                      int C,
                      int k,
                      float *logZ,
                      float *Ms_grad,
                      float *Ms,
                      float *v0,
                      float *vT,
                      int *idx);

int bwd_scores_host_sparse_log_NZ5(float *betas,
                                   float *Ms,
                                   float *vT,
                                   int *idx_T,
                                   int T,
                                   int N,
                                   int C);

int bwd_scores_host_sparse_log_NZ3(float *betas,
                                   float *Ms,
                                   float *vT,
                                   int *idx_T,
                                   int T,
                                   int N,
                                   int C);

int bwd_scores_host_sparse_max(float *betas, float *Ms, float *vT, int *idx_T, int T, int N, int C);

int fwd_scores_host_sparse_log_NZ5(float *alphas,
                                   float *Ms,
                                   float *v0,
                                   int *idx,
                                   int T,
                                   int N,
                                   int C);

int fwd_scores_host_sparse_log_NZ3(float *alphas,
                                   float *Ms,
                                   float *v0,
                                   int *idx,
                                   int T,
                                   int N,
                                   int C);

int fwd_scores_host_sparse_max(float *alphas, float *Ms, float *v0, int *idx, int T, int N, int C);
