#include "kernel_roi_align.h"

#include <math.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

#include "roi_align.h"

using namespace tensorflow;

REGISTER_OP("RoiAlign")
    .Attr("T: numbertype")
    .Input("input: T")
    .Input("boxes: T")
    .Input("spatial_scale: T")
    .Input("sampling_ratio: int32")
    .Input("aligned: bool")
    .Input("pooled_dims: int32")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

template <typename T>
void RoiAlignFunctor<Eigen::ThreadPoolDevice, T>::operator()(
    const Eigen::ThreadPoolDevice &d, int size, const T *in,
    const T *spatial_scale, int channels, std::vector<int> input_spatial_dims,
    std::vector<int> output_spatial_dims, const int32_t *sampling_ratio,
    const bool *aligned, const T *rois, T *out) {

  auto fun = [&size, &in, &spatial_scale, &channels, &input_spatial_dims,
              &output_spatial_dims, &sampling_ratio, &aligned, &rois,
              &out](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      if (input_spatial_dims.size() == 2) {
        roi_align_2D<T>(i, size, in, spatial_scale, channels,
                                input_spatial_dims[0], input_spatial_dims[1],
                                output_spatial_dims[0], output_spatial_dims[1],
                                sampling_ratio, aligned, rois, out);
      } else {
        roi_align_3D<T>(i, size, in, spatial_scale, channels,
                                input_spatial_dims[0], input_spatial_dims[1],
                                input_spatial_dims[2], output_spatial_dims[0],
                                output_spatial_dims[1], output_spatial_dims[2],
                                sampling_ratio, aligned, rois, out);
      }
    }
  };

  const Eigen::TensorOpCost cost_result(100, 100, 10000);
  d.parallelFor(size, cost_result, fun);
}

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class RoiAlignOp : public OpKernel {
private:
public:
  explicit RoiAlignOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    const Tensor &boxes_tensor = context->input(1);
    const Tensor &spatial_scale_tensor = context->input(2);
    const Tensor &sampling_ratio_tensor = context->input(3);
    const Tensor &aligned_tensor = context->input(4);
    const Tensor &pooled_dims_tensor = context->input(5);

    std::vector<int> output_dims;
    output_dims.push_back(boxes_tensor.shape().dim_size(0));
    for (int i = 0; i < pooled_dims_tensor.NumElements(); i++) {
      output_dims.push_back(int(pooled_dims_tensor.flat<int32_t>().data()[i]));
    }
    output_dims.push_back(
        input_tensor.shape().dim_size(input_tensor.dims() - 1));

    int channels = output_dims.back();
    std::vector<int> output_spatial_dims(output_dims.begin() + 1,
                                         output_dims.end() - 1);
    std::vector<int> input_spatial_dims;

    for (int i = 1; i < input_tensor.shape().dims() - 1; i++) {
      input_spatial_dims.push_back(int(input_tensor.shape().dim_size(i)));
    }

    TensorShape input_shape = input_tensor.shape();
    TensorShape output_shape;
    std::vector<int64_t> output_dims_int_64t;
    for (int i = 0; i < output_dims.size(); i++) {
      output_dims_int_64t.push_back(int64_t(output_dims[i]));
    }

    TensorShape::BuildTensorShape(output_dims_int_64t, &output_shape);

    // Create an output tensor
    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    RoiAlignFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(output_tensor->NumElements()),
        input_tensor.flat<T>().data(), spatial_scale_tensor.flat<T>().data(),
        channels, input_spatial_dims, output_spatial_dims,
        sampling_ratio_tensor.flat<int32_t>().data(),
        aligned_tensor.flat<bool>().data(), boxes_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RoiAlign").Device(DEVICE_CPU).TypeConstraint<T>("T"),              \
      RoiAlignOp<Eigen::ThreadPoolDevice, T>);
REGISTER_CPU(float);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                        \
  /* Declare explicit instantiations in kernel_example.cu.cc. */               \
  extern template class RoiAlignFunctor<Eigen::GpuDevice, T>;                  \
  REGISTER_KERNEL_BUILDER(Name("RoiAlign")                                     \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .HostMemory("pooled_dims"),                      \
                          RoiAlignOp<Eigen::GpuDevice, T>);
REGISTER_GPU(float);
#endif // GOOGLE_CUDA
