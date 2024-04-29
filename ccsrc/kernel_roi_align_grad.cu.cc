#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kernel_roi_align_grad.h"

#include <stdio.h>

#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "roi_align.h"

using namespace tensorflow;

template <typename T> __global__ void ClearGradient(int size, T *grad_input) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    grad_input[index] = 0.0;
  }
}

// Define the CUDA kernel.
template <typename T>
__global__ void RoiAlignGradCudaKernel2D(
    int size, const T *grad_output, const T *pSpatial_scale, int channels,
    int height, int width, int pooled_height, int pooled_width,
    const int32_t *pSampling_ratio, const bool *pAligned, const T *rois,
    T *grad_input, int n_stride, int h_stride, int w_stride, int c_stride) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    roi_align_gradient_2D<T>(
        index, size, grad_output, pSpatial_scale, channels, height, width,
        pooled_height, pooled_width, pSampling_ratio, pAligned, rois,
        grad_input, n_stride, h_stride, w_stride, c_stride);
  }
}

template <typename T>
__global__ void RoiAlignGradCudaKernel3D(
    int size, const T *grad_output, const T *pSpatial_scale, int channels,
    int depth, int height, int width, int pooled_depth, int pooled_height,
    int pooled_width, const int32_t *pSampling_ratio, const bool *pAligned,
    const T *rois, T *grad_input, int n_stride, int d_stride, int h_stride,
    int w_stride, int c_stride) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    roi_align_gradient_3D(index, size, grad_output, pSpatial_scale,
                                 channels, depth, height, width, pooled_depth,
                                 pooled_height, pooled_width, pSampling_ratio,
                                 pAligned, rois, grad_input, n_stride, d_stride,
                                 h_stride, w_stride, c_stride);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void RoiAlignGradFunctor<Eigen::GpuDevice, T>::operator()(
    const Eigen::GpuDevice &d, int size, const T *grad_output, const T *spatial_scale,
    int channels, std::vector<int> grad_input_spatial_dims,
    std::vector<int> pooled_output_spatial_dims, const int32_t *sampling_ratio,
    const bool *aligned, const T *rois, T *grad_input,
    const int size_grad_input, std::vector<int> grad_output_strides) {
  // Launch the cuda kernel.
  //
  // See core/util/gpu_kernel_helper.h for example of computing
  // block count and thread_per_block count.

  ClearGradient<T>
      <<<(size_grad_input / 16) + 1, 16>>>(size_grad_input, grad_input);
  cudaPeekAtLastError();
  cudaDeviceSynchronize();

  int thread_per_block = 16;
  int block_count = (size / thread_per_block) + 1;

  if (pooled_output_spatial_dims.size() == 2) {
    RoiAlignGradCudaKernel2D<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(
            size, grad_output, spatial_scale, channels,
            grad_input_spatial_dims[0], grad_input_spatial_dims[1],
            pooled_output_spatial_dims[0], pooled_output_spatial_dims[1],
            sampling_ratio, aligned, rois, grad_input, grad_output_strides[0],
            grad_output_strides[1], grad_output_strides[2],
            grad_output_strides[3]);

  } else {
    RoiAlignGradCudaKernel3D<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(
            size, grad_output, spatial_scale, channels,
            grad_input_spatial_dims[0], grad_input_spatial_dims[1],
            grad_input_spatial_dims[2], pooled_output_spatial_dims[0],
            pooled_output_spatial_dims[1], pooled_output_spatial_dims[2],
            sampling_ratio, aligned, rois, grad_input, grad_output_strides[0],
            grad_output_strides[1], grad_output_strides[2],
            grad_output_strides[3], grad_output_strides[4]);
  }
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct RoiAlignGradFunctor<Eigen::GpuDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

#endif // GOOGLE_CUDA
