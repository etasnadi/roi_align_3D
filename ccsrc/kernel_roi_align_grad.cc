#include "kernel_roi_align_grad.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "roi_align.h"

using namespace tensorflow;

REGISTER_OP("RoiAlignGrad")
    .Attr("T: numbertype")
    .Input("grad_output: T")
    .Input("boxes: T")
    .Input("spatial_scale: T")
    .Input("sampling_ratio: int32")
    .Input("aligned: bool")
    .Input("forward_image_input: T")
    .Input("pooled_dims: int32")
    .Output("grad_input: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    });

template <typename T>
void RoiAlignGradFunctor<Eigen::ThreadPoolDevice, T>::operator()(
    const Eigen::ThreadPoolDevice &d,
    int size,             // grad_output numel
    const T *grad_output, // grad_output
    const T *spatial_scale, int channels,
    std::vector<int> grad_input_spatial_dims,
    std::vector<int> pooled_output_spatial_dims, const int32_t *sampling_ratio,
    const bool *aligned, const T *rois, T *grad_input,
    const int size_grad_input, // grad_input numel
    std::vector<int> grad_output_strides) {

  // TODO: convert to parallel processing (depends on atomic update on cpu in
  // the thread implementation code)

  // First celar the output
  // TODO: use Tensorflow's call
  for (int i = 0; i < size_grad_input; i++) {
    grad_input[i] = 0.0;
  }

  for (int i = 0; i < size; i++) {
    if (grad_input_spatial_dims.size() == 2) {
      roi_align_gradient_2D<T>(
          i, size, grad_output, spatial_scale, channels,
          grad_input_spatial_dims[0], grad_input_spatial_dims[1],
          pooled_output_spatial_dims[0], pooled_output_spatial_dims[1],
          sampling_ratio, aligned, rois, grad_input, grad_output_strides[0],
          grad_output_strides[1], grad_output_strides[2],
          grad_output_strides[3]);
    } else {
      roi_align_gradient_3D<T>(
          i, size, grad_output, spatial_scale, channels,
          grad_input_spatial_dims[0], grad_input_spatial_dims[1],
          grad_input_spatial_dims[2], pooled_output_spatial_dims[0],
          pooled_output_spatial_dims[1], pooled_output_spatial_dims[2],
          sampling_ratio, aligned, rois, grad_input, grad_output_strides[0],
          grad_output_strides[1], grad_output_strides[2],
          grad_output_strides[3], grad_output_strides[4]);
    }
  }
}

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class RoiAlignGradOp : public OpKernel {
private:
public:
  explicit RoiAlignGradOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    const Tensor &grad_output_tensor = context->input(0);
    const Tensor &boxes_tensor = context->input(1);
    const Tensor &spatial_scale_tensor = context->input(2);
    const Tensor &sampling_ratio_tensor = context->input(3);
    const Tensor &aligned_tensor = context->input(4);
    const Tensor &forward_image_input = context->input(5);
    const Tensor &pooled_dims_tensor = context->input(6);

    std::vector<int> pooled_output_spatial_dims;
    for (int i = 0; i < pooled_dims_tensor.NumElements(); i++) {
      pooled_output_spatial_dims.push_back(
          int(pooled_dims_tensor.flat<int32_t>().data()[i]));
    }

    TensorShape grad_output_shape = grad_output_tensor.shape();
    // The output of backward (grad_input) shape is inferred from the input
    // shape of the forward
    TensorShape grad_input_shape = forward_image_input.shape();
    std::vector<int> grad_input_spatial_dims;
    for (int i = 1; i < grad_input_shape.dims() - 1; i++) {
      grad_input_spatial_dims.push_back(int(grad_input_shape.dim_size(i)));
    }
    int channels = grad_input_shape.dim_size(grad_input_shape.dims() - 1);

    // Create an output tensor
    Tensor *grad_input_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_input_shape,
                                                     &grad_input_tensor));

    // Do the computation.
    OP_REQUIRES(context,
                grad_output_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    // Determine the strides (prefix sum)

    // (N, H, W, C)

    // Stride of C is 1
    // Stride of W is C
    // Stride of H is W*C
    // Stride of N is H*W*C

    // int n_stride =
    // grad_output_shape.dim_size(1)*grad_output_shape.dim_size(2)*grad_output_shape.dim_size(3);
    // int h_stride =
    // grad_output_shape.dim_size(2)*grad_output_shape.dim_size(3); int w_stride
    // = grad_output_shape.dim_size(3); int c_stride = 1;

    int grad_output_ndims = grad_output_shape.dims();
    std::vector<int> grad_output_strides(grad_output_ndims, 0);
    grad_output_strides[grad_output_ndims - 1] = 1;
    for (int i = grad_output_ndims - 2; i >= 0; i--) {
      grad_output_strides[i] =
          grad_output_strides[i + 1] * grad_output_shape.dim_size(i + 1);
    }

    RoiAlignGradFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(grad_output_tensor.NumElements()),
        grad_output_tensor.flat<T>().data(),
        spatial_scale_tensor.flat<T>().data(), channels,
        grad_input_spatial_dims, pooled_output_spatial_dims,
        sampling_ratio_tensor.flat<int32_t>().data(),
        aligned_tensor.flat<bool>().data(), boxes_tensor.flat<T>().data(),
        grad_input_tensor->flat<T>().data(), grad_input_tensor->NumElements(),
        grad_output_strides);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RoiAlignGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),          \
      RoiAlignGradOp<Eigen::ThreadPoolDevice, T>);
REGISTER_CPU(float);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                        \
  /* Declare explicit instantiations in kernel_example.cu.cc. */               \
  extern template class RoiAlignGradFunctor<Eigen::GpuDevice, T>;              \
  REGISTER_KERNEL_BUILDER(Name("RoiAlignGrad")                                 \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<T>("T")                          \
                              .HostMemory("pooled_dims"),                      \
                          RoiAlignGradOp<Eigen::GpuDevice, T>);
REGISTER_GPU(float);
#endif // GOOGLE_CUDA
