#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kernel_roi_align.h"

#include <stdio.h>

#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "roi_align.h"

using namespace tensorflow;

template <typename T>
__global__ void
RoiAlignKernel2D(int size, const T *input, const T *pSpatial_scale,
                 int channels, int height, int width, int pooled_height,
                 int pooled_width, const int32_t *pSampling_ratio,
                 const bool *pAligned, const T *rois, T *output) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    roi_align_2D<T>(index, size, input, pSpatial_scale, channels,
                            height, width, pooled_height, pooled_width,
                            pSampling_ratio, pAligned, rois, output);
  }
}

template <typename T>
__global__ void
RoiAlignKernel3D(int size, const T *input, const T *pSpatial_scale,
                 int channels, int depth, int height, int width,
                 int pooled_depth, int pooled_height, int pooled_width,
                 const int32_t *pSampling_ratio, const bool *pAligned,
                 const T *rois, T *output) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    roi_align_3D(index, size, input, pSpatial_scale, channels, depth,
                         height, width, pooled_depth, pooled_height,
                         pooled_width, pSampling_ratio, pAligned, rois, output);
  }
}

template <typename T>
void RoiAlignFunctor<Eigen::GpuDevice, T>::operator()(
    const Eigen::GpuDevice &d, int size, const T *in, const T *spatial_scale,
    int channels, std::vector<int> input_spatial_dims,
    std::vector<int> output_spatial_dims, const int32_t *sampling_ratio,
    const bool *aligned, const T *rois, T *out) {

  int thread_per_block = 16;
  int block_count = (size / thread_per_block) + 1;

  if (input_spatial_dims.size() == 2) {
    RoiAlignKernel2D<T><<<block_count, thread_per_block, 0, d.stream()>>>(
        size, in, spatial_scale, channels, input_spatial_dims[0],
        input_spatial_dims[1], output_spatial_dims[0], output_spatial_dims[1],
        sampling_ratio, aligned, rois, out);
  } else {
    RoiAlignKernel3D<T><<<block_count, thread_per_block, 0, d.stream()>>>(
        size, in, spatial_scale, channels, input_spatial_dims[0],
        input_spatial_dims[1], input_spatial_dims[2], output_spatial_dims[0],
        output_spatial_dims[1], output_spatial_dims[2], sampling_ratio, aligned,
        rois, out);
  }
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct RoiAlignFunctor<Eigen::GpuDevice, float>;

#endif // GOOGLE_CUDA
