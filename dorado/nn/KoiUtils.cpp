#include "nn/KoiUtils.h"

#if !DORADO_ROCM_BUILD
#include <ATen/cuda/CUDAContext.h>
#endif

namespace dorado::nn {

//todo hm: these may need to be addressed in openfish
// TODO: These should really be part of Koi
bool koi_can_use_cutlass() {
#if DORADO_ROCM_BUILD
    return false;
#else
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    return (prop->major >= 8);
#endif
}
bool koi_can_use_quantised_lstm() {
#if DORADO_ROCM_BUILD
    return false;
#else
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    // DP4A is supported on Pascal and later, except for TX2 (sm_62).
    return (prop->major > 6) || (prop->major == 6 && prop->minor != 2);
#endif
}

}  // namespace dorado::nn
