#ifndef KERNEL_CUDA_H_
#define KERNEL_CUDA_H_

// We need to define and include the following otherwise the ThreadPoolDevice
// will be only partially defined thus device.parallelFor will not compile.
// https://stackoverflow.com/questions/43786754/using-eigen-in-custom-tensorflow-op
#define EIGEN_USE_THREADS
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>
//#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T> struct RoiAlignFunctor {
  void operator()(const Device &d, int size, const T *in, T *out);
};

template <typename T> struct RoiAlignFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice &d, int size, const T *in,
                  const T *spatial_scale, int channels,
                  std::vector<int> input_spatial_dims,
                  std::vector<int> output_spatial_dims,
                  const int32_t *sampling_ratio, const bool *aligned,
                  const T *rois, T *out);
};

#if GOOGLE_CUDA

template <typename T> struct RoiAlignFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice &d, int size, const T *in,
                  const T *spatial_scale, int channels,
                  std::vector<int> input_spatial_dims,
                  std::vector<int> output_spatial_dims,
                  const int32_t *sampling_ratio, const bool *aligned,
                  const T *rois, T *out);
};

#endif

#endif
