#ifndef KERNEL_GRAD_CUDA_H_
#define KERNEL_GRAD_CUDA_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T> struct RoiAlignGradFunctor {
  void operator()(const Device &d, int size, const T *in, T *out);
};

template <typename T> struct RoiAlignGradFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice &d, int size,
                  const T *grad_output, const T *spatial_scale, int channels,
                  std::vector<int> grad_input_spatial_dims,
                  std::vector<int> pooled_output_spatial_dims,
                  const int32_t *sampling_ratio, const bool *aligned,
                  const T *rois, T *grad_input, const int grad_input_size,
                  std::vector<int> grad_output_strides);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.

template <typename T> struct RoiAlignGradFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice &d, int size, const T *grad_output,
                  const T *spatial_scale, int channels,
                  std::vector<int> grad_input_spatial_dims,
                  std::vector<int> pooled_output_spatial_dims,
                  const int32_t *sampling_ratio, const bool *aligned,
                  const T *rois, T *grad_input, const int grad_input_size,
                  std::vector<int> grad_output_strides);
};
#endif

#endif
