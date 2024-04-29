#include "roi_align.h"

// 2D

// Forward

#ifdef __CUDA_ARCH__
static inline __device__ float gpuAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}
#endif

template <typename T>
#ifdef __CUDA_ARCH__
__host__
    __device__
#endif
        T
        bilinear_interpolate_2D(const T *input, int height, int width,
                                int channels, T y, T x, int c,
                                int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // do bilinear interpolation
  T v1 = input[channels * (y_low * width + x_low) + c];
  T v2 = input[channels * (y_low * width + x_high) + c];
  T v3 = input[channels * (y_high * width + x_low) + c];
  T v4 = input[channels * (y_high * width + x_high) + c];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    roi_align_2D(int index, int size, const T *input, const T *pSpatial_scale,
                 int channels, int height, int width, int pooled_height,
                 int pooled_width, const int32_t *pSampling_ratio,
                 const bool *pAligned, const T *rois, T *output) {

  const T spatial_scale = pSpatial_scale[0];
  const int sampling_ratio = int(pSampling_ratio[0]);
  const bool aligned = pAligned[0];

  /*
  Original torchvision code: assumes NxCxHxW
  // (n, c, ph, pw) is an element in the pooled output
  int pw = index % pooled_width;
  int ph = (index / pooled_width) % pooled_height;
  int c = (index / pooled_width / pooled_height) % channels;
  int n = index / pooled_width / pooled_height / channels;
  */

  // TF code: assumes NxHxWxC

  int c = index % channels;
  int pw = (index / channels) % pooled_width;
  int ph = (index / channels / pooled_width) % pooled_height;
  int n = index / channels / pooled_width / pooled_height;

  const T *offset_rois = rois + n * 5;
  int roi_batch_ind = offset_rois[0];

  // Do not using rounding; this implementation detail is critical
  T offset = aligned ? (T)0.5 : (T)0.0;
  T roi_start_w = offset_rois[1] * spatial_scale - offset;
  T roi_start_h = offset_rois[2] * spatial_scale - offset;
  T roi_end_w = offset_rois[3] * spatial_scale - offset;
  T roi_end_h = offset_rois[4] * spatial_scale - offset;

  T roi_width = roi_end_w - roi_start_w;
  T roi_height = roi_end_h - roi_start_h;
  if (!aligned) {
    // Force malformed ROIs to be 1x1
    roi_width = std::max(roi_width, (T)1.);
    roi_height = std::max(roi_height, (T)1.);
  }

  T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
  T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

  /*
  const T* offset_input =
      input + (roi_batch_ind * channels + c) * height * width;
  */

  // n, c, h, w (Torch)
  // n, h, w, c (TF)

  const T *offset_input = input + (roi_batch_ind * height * width * channels);

  // We use roi_bin_grid to sample the grid and mimic integral
  int roi_bin_grid_h = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_height / pooled_height); // e.g., = 2
  int roi_bin_grid_w =
      (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

  // We do average (integral) pooling inside a bin
  // When the grid is empty, output zeros.
  const T count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

  T output_val = 0.;
  for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
  {
    const T y = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h /
                    static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
    for (int ix = 0; ix < roi_bin_grid_w; ix++) {
      const T x = roi_start_w + pw * bin_size_w +
                  static_cast<T>(ix + .5f) * bin_size_w /
                      static_cast<T>(roi_bin_grid_w);

      T val = bilinear_interpolate_2D<T>(offset_input, height, width, channels,
                                         y, x, c, index);
      output_val += val;
    }
  }
  output_val /= count;

  output[index] = output_val;
}

// Backward

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    bilinear_interpolate_gradient_2D(int height, int width, T y, T x, T &w1,
                                     T &w2, T &w3, T &w4, int &x_low,
                                     int &x_high, int &y_low, int &y_high,
                                     int index) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

// Define the CUDA kernel.
template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    roi_align_gradient_2D(int index, int size, const T *grad_output,
                          const T *pSpatial_scale, int channels, int height,
                          int width, int pooled_height, int pooled_width,
                          const int32_t *pSampling_ratio, const bool *pAligned,
                          const T *rois, T *grad_input, int n_stride,
                          int h_stride, int w_stride, int c_stride) {

  const T spatial_scale = pSpatial_scale[0];
  int sampling_ratio = int(pSampling_ratio[0]);
  bool aligned = pAligned[0];

  // (n, c, ph, pw) is an element in the pooled output
  /*
  int pw = index % pooled_width;
  int ph = (index / pooled_width) % pooled_height;
  int c = (index / pooled_width / pooled_height) % channels;
  int n = index / pooled_width / pooled_height / channels;
  */
  int c = index % channels;
  int pw = (index / channels) % pooled_width;
  int ph = (index / channels / pooled_width) % pooled_height;
  int n = index / channels / pooled_width / pooled_height;

  const T *offset_rois = rois + n * 5;
  int roi_batch_ind = offset_rois[0];

  // Do not using rounding; this implementation detail is critical
  T offset = aligned ? (T)0.5 : (T)0.0;
  T roi_start_w = offset_rois[1] * spatial_scale - offset;
  T roi_start_h = offset_rois[2] * spatial_scale - offset;
  T roi_end_w = offset_rois[3] * spatial_scale - offset;
  T roi_end_h = offset_rois[4] * spatial_scale - offset;

  T roi_width = roi_end_w - roi_start_w;
  T roi_height = roi_end_h - roi_start_h;
  if (!aligned) {
    // Force malformed ROIs to be 1x1
    roi_width = std::max(roi_width, (T)1.);
    roi_height = std::max(roi_height, (T)1.);
  }

  T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
  T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

  T *offset_grad_input =
      grad_input + (roi_batch_ind * height * width * channels);
  /*
  T* offset_grad_input =
      grad_input + ((roi_batch_ind * channels + c) * height * width);
  */

  // We need to index the gradient using the tensor strides to access the
  // correct values.

  // int output_offset = n * n_stride + c * c_stride;
  int output_offset = n * n_stride;

  const T *offset_grad_output = grad_output + output_offset;

  /*
  const T grad_output_this_bin =
      offset_grad_output[ph * h_stride + pw * w_stride];
  */
  const T grad_output_this_bin =
      offset_grad_output[ph * h_stride + pw * w_stride + c * c_stride];

  // We use roi_bin_grid to sample the grid and mimic integral
  int roi_bin_grid_h = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_height / pooled_height); // e.g., = 2
  int roi_bin_grid_w =
      (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

  // We do average (integral) pooling inside a bin
  const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

  for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
  {
    const T y = roi_start_h + ph * bin_size_h +
                static_cast<T>(iy + .5f) * bin_size_h /
                    static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
    for (int ix = 0; ix < roi_bin_grid_w; ix++) {
      const T x = roi_start_w + pw * bin_size_w +
                  static_cast<T>(ix + .5f) * bin_size_w /
                      static_cast<T>(roi_bin_grid_w);

      T w1, w2, w3, w4;
      int x_low, x_high, y_low, y_high;

      bilinear_interpolate_gradient_2D(height, width, y, x, w1, w2, w3, w4,
                                       x_low, x_high, y_low, y_high, index);

      T g1 = grad_output_this_bin * w1 / count;
      T g2 = grad_output_this_bin * w2 / count;
      T g3 = grad_output_this_bin * w3 / count;
      T g4 = grad_output_this_bin * w4 / count;

      if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
        float *addr_g1 =
            offset_grad_input + (y_low * width + x_low) * channels + c;
        float *addr_g2 =
            offset_grad_input + (y_low * width + x_high) * channels + c;
        float *addr_g3 =
            offset_grad_input + (y_high * width + x_low) * channels + c;
        float *addr_g4 =
            offset_grad_input + (y_high * width + x_high) * channels + c;
#ifdef __CUDA_ARCH__
        gpuAtomicAdd(addr_g1, static_cast<T>(g1));
        gpuAtomicAdd(addr_g2, static_cast<T>(g2));
        gpuAtomicAdd(addr_g3, static_cast<T>(g3));
        gpuAtomicAdd(addr_g4, static_cast<T>(g4));
#else
        // TODO: make update atomic on CPU
        *addr_g1 += static_cast<T>(g1);
        *addr_g2 += static_cast<T>(g2);
        *addr_g3 += static_cast<T>(g3);
        *addr_g4 += static_cast<T>(g4);
#endif
      } // if
    }   // ix
  }     // iy
}

// 3D

// Forward

template <typename T>
#ifdef __CUDA_ARCH__
__host__
    __device__
#endif
        T
        bilinear_interpolate_3D(const T *input, int depth, int height,
                                int width, int channels, T z, T y, T x, int c,
                                int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 ||
      z > depth) {
    // empty
    return 0;
  }

  if (z <= 0)
    z = 0;
  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  int z_low = (int)z;
  int y_low = (int)y;
  int x_low = (int)x;
  int z_high;
  int y_high;
  int x_high;

  if (z_low >= height - 1) {
    z_high = z_low = height - 1;
    z = (T)z_low;
  } else {
    z_high = z_low + 1;
  }

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T lz = z - z_low;
  T ly = y - y_low;
  T lx = x - x_low;
  T hz = 1. - lz, hy = 1. - ly, hx = 1. - lx;

  // do trilinear interpolation
  T v1 = input[channels * (z_low * height * width + y_low * width + x_low) + c];
  T v2 =
      input[channels * (z_low * height * width + y_low * width + x_high) + c];
  T v3 =
      input[channels * (z_low * height * width + y_high * width + x_low) + c];
  T v4 =
      input[channels * (z_low * height * width + y_high * width + x_high) + c];
  T v5 =
      input[channels * (z_high * height * width + y_low * width + x_low) + c];
  T v6 =
      input[channels * (z_high * height * width + y_low * width + x_high) + c];
  T v7 =
      input[channels * (z_high * height * width + y_high * width + x_low) + c];
  T v8 =
      input[channels * (z_high * height * width + y_high * width + x_high) + c];

  T w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
  T w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 +
           w8 * v8);

  return val;
}

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    roi_align_3D(int index, int size, const T *input, const T *pSpatial_scale,
                 int channels, int depth, int height, int width,
                 int pooled_depth, int pooled_height, int pooled_width,
                 const int32_t *pSampling_ratio, const bool *pAligned,
                 const T *rois, T *output) {

  const T spatial_scale = pSpatial_scale[0];
  const int sampling_ratio = int(pSampling_ratio[0]);
  const bool aligned = pAligned[0];

  /*
  Original torchvision code: assumes NxCxHxW
  // (n, c, ph, pw) is an element in the pooled output
  int pw = index % pooled_width;
  int ph = (index / pooled_width) % pooled_height;
  int c = (index / pooled_width / pooled_height) % channels;
  int n = index / pooled_width / pooled_height / channels;
  */

  // TF code: assumes NxDxHxWxC

  int c = index % channels;
  int pw = (index / channels) % pooled_width;
  int ph = (index / channels / pooled_width) % pooled_height;
  int pd = (index / channels / pooled_width / pooled_height) % pooled_depth;
  int n = index / channels / pooled_width / pooled_height / pooled_depth;

  // Roi: [b, x1, y1, z1, x2, y2, z2]
  const T *offset_rois = rois + n * 7;
  int roi_batch_ind = offset_rois[0];

  // Do not using rounding; this implementation detail is critical
  T offset = aligned ? (T)0.5 : (T)0.0;
  T roi_start_w = offset_rois[1] * spatial_scale - offset;
  T roi_start_h = offset_rois[2] * spatial_scale - offset;
  T roi_start_d = offset_rois[3] * spatial_scale - offset;
  T roi_end_w = offset_rois[4] * spatial_scale - offset;
  T roi_end_h = offset_rois[5] * spatial_scale - offset;
  T roi_end_d = offset_rois[6] * spatial_scale - offset;

  T roi_width = roi_end_w - roi_start_w;
  T roi_height = roi_end_h - roi_start_h;
  T roi_depth = roi_end_d - roi_start_d;
  if (!aligned) {
    // Force malformed ROIs to be 1x1
    roi_width = std::max(roi_width, (T)1.);
    roi_height = std::max(roi_height, (T)1.);
    roi_depth = std::max(roi_depth, (T)1.);
  }

  T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);
  T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
  T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

  /*
  const T* offset_input =
      input + (roi_batch_ind * channels + c) * height * width;
  */

  // n, c, h, w (Torch)
  // n, h, w, c (TF)

  const T *offset_input =
      input + (roi_batch_ind * depth * height * width * channels);

  // We use roi_bin_grid to sample the grid and mimic integral
  int roi_bin_grid_d = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_depth / pooled_depth); // e.g., = 2
  int roi_bin_grid_h = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_height / pooled_height); // e.g., = 2
  int roi_bin_grid_w =
      (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

  // We do average (integral) pooling inside a bin
  // When the grid is empty, output zeros.
  const T count =
      std::max(roi_bin_grid_d * roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

  T output_val = 0.;
  for (int iz = 0; iz < roi_bin_grid_d; iz++) {
    const T z = roi_start_d + pd * bin_size_d +
                static_cast<T>(iz + .5f) * bin_size_d /
                    static_cast<T>(roi_bin_grid_d); // e.g., 0.5, 1.5

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate_3D(offset_input, depth, height, width,
                                        channels, z, y, x, c, index);
        output_val += val;
      }
    }
  }
  output_val /= count;

  output[index] = output_val;
}

// Backward

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    bilinear_interpolate_gradient_3D(int depth, int height, int width, T z, T y,
                                     T x, T &w1, T &w2, T &w3, T &w4, T &w5,
                                     T &w6, T &w7, T &w8, int &x_low,
                                     int &x_high, int &y_low, int &y_high,
                                     int &z_low, int &z_high, int index) {
  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > depth || y < -1.0 || y > height || x < -1.0 ||
      x > width) {
    // empty
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
    x_low = x_high = y_low = y_high = z_low = z_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;
  if (z <= 0)
    z = 0;

  z_low = (int)z;
  y_low = (int)y;
  x_low = (int)x;

  if (z_low >= depth - 1) {
    z_high = z_low = depth - 1;
    z = (T)z_low;
  } else {
    z_high = z_low + 1;
  }

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T lz = z - z_low;
  T ly = y - y_low;
  T lx = x - x_low;
  T hz = 1. - lz, hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
  w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;
}

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    roi_align_gradient_3D(int index, int size, const T *grad_output,
                          const T *pSpatial_scale, int channels, int depth,
                          int height, int width, int pooled_depth,
                          int pooled_height, int pooled_width,
                          const int32_t *pSampling_ratio, const bool *pAligned,
                          const T *rois, T *grad_input, int n_stride,
                          int d_stride, int h_stride, int w_stride,
                          int c_stride) {

  const T spatial_scale = pSpatial_scale[0];
  int sampling_ratio = int(pSampling_ratio[0]);
  bool aligned = pAligned[0];

  // (n, c, ph, pw) is an element in the pooled output
  /*
  int pw = index % pooled_width;
  int ph = (index / pooled_width) % pooled_height;
  int c = (index / pooled_width / pooled_height) % channels;
  int n = index / pooled_width / pooled_height / channels;
  */
  int c = index % channels;
  int pw = (index / channels) % pooled_width;
  int ph = (index / channels / pooled_width) % pooled_height;
  int pd = (index / channels / pooled_width / pooled_height) % pooled_depth;
  int n = index / channels / pooled_width / pooled_height / pooled_depth;

  const T *offset_rois = rois + n * 7;
  int roi_batch_ind = offset_rois[0];

  // Do not using rounding; this implementation detail is critical
  T offset = aligned ? (T)0.5 : (T)0.0;
  T roi_start_w = offset_rois[1] * spatial_scale - offset;
  T roi_start_h = offset_rois[2] * spatial_scale - offset;
  T roi_start_d = offset_rois[3] * spatial_scale - offset;
  T roi_end_w = offset_rois[4] * spatial_scale - offset;
  T roi_end_h = offset_rois[5] * spatial_scale - offset;
  T roi_end_d = offset_rois[6] * spatial_scale - offset;

  T roi_width = roi_end_w - roi_start_w;
  T roi_height = roi_end_h - roi_start_h;
  T roi_depth = roi_end_d - roi_start_d;

  if (!aligned) {
    // Force malformed ROIs to be 1x1
    roi_width = std::max(roi_width, (T)1.);
    roi_height = std::max(roi_height, (T)1.);
    roi_depth = std::max(roi_depth, (T)1.);
  }

  T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);
  T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
  T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

  T *offset_grad_input =
      grad_input + (roi_batch_ind * depth * height * width * channels);
  /*
  T* offset_grad_input =
      grad_input + ((roi_batch_ind * channels + c) * height * width);
  */

  // We need to index the gradient using the tensor strides to access the
  // correct values.

  // int output_offset = n * n_stride + c * c_stride;
  int output_offset = n * n_stride;

  const T *offset_grad_output = grad_output + output_offset;

  /*
  const T grad_output_this_bin =
      offset_grad_output[ph * h_stride + pw * w_stride];
  */
  const T grad_output_this_bin =
      offset_grad_output[pd * d_stride + ph * h_stride + pw * w_stride +
                         c * c_stride];

  // We use roi_bin_grid to sample the grid and mimic integral
  int roi_bin_grid_d = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_depth / pooled_depth); // e.g., = 2
  int roi_bin_grid_h = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_height / pooled_height); // e.g., = 2
  int roi_bin_grid_w =
      (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

  // We do average (integral) pooling inside a bin
  const T count = roi_bin_grid_d * roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

  for (int iz = 0; iz < roi_bin_grid_h; iz++) // e.g., iy = 0, 1
  {
    const T z = roi_start_d + pd * bin_size_d +
                static_cast<T>(iz + .5f) * bin_size_d /
                    static_cast<T>(roi_bin_grid_d); // e.g., 0.5, 1.5
    for (int iy = 0; iy < roi_bin_grid_h; iy++)     // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4, w5, w6, w7, w8;
        int x_low, x_high, y_low, y_high, z_low, z_high;

        bilinear_interpolate_gradient_3D(depth, height, width, z, y, x, w1, w2,
                                         w3, w4, w5, w6, w7, w8, x_low, x_high,
                                         y_low, y_high, z_low, z_high, index);

        T g1 = grad_output_this_bin * w1 / count;
        T g2 = grad_output_this_bin * w2 / count;
        T g3 = grad_output_this_bin * w3 / count;
        T g4 = grad_output_this_bin * w4 / count;
        T g5 = grad_output_this_bin * w5 / count;
        T g6 = grad_output_this_bin * w6 / count;
        T g7 = grad_output_this_bin * w7 / count;
        T g8 = grad_output_this_bin * w8 / count;

        float *addr_g1 =
            offset_grad_input +
            (z_low * width * height + y_low * width + x_low) * channels + c;
        float *addr_g2 =
            offset_grad_input +
            (z_low * width * height + y_low * width + x_high) * channels + c;
        float *addr_g3 =
            offset_grad_input +
            (z_low * width * height + y_high * width + x_low) * channels + c;
        float *addr_g4 =
            offset_grad_input +
            (z_low * width * height + y_high * width + x_high) * channels + c;
        float *addr_g5 =
            offset_grad_input +
            (z_high * width * height + y_low * width + x_low) * channels + c;
        float *addr_g6 =
            offset_grad_input +
            (z_high * width * height + y_low * width + x_high) * channels + c;
        float *addr_g7 =
            offset_grad_input +
            (z_high * width * height + y_high * width + x_low) * channels + c;
        float *addr_g8 =
            offset_grad_input +
            (z_high * width * height + y_high * width + x_high) * channels + c;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 &&
            z_low >= 0 && z_high >= 0) {
#ifdef __CUDA_ARCH__
          // Z_low
          gpuAtomicAdd(addr_g1, static_cast<T>(g1));
          gpuAtomicAdd(addr_g2, static_cast<T>(g2));
          gpuAtomicAdd(addr_g3, static_cast<T>(g3));
          gpuAtomicAdd(addr_g4, static_cast<T>(g4));

          // Z_high
          gpuAtomicAdd(addr_g5, static_cast<T>(g5));
          gpuAtomicAdd(addr_g6, static_cast<T>(g6));
          gpuAtomicAdd(addr_g7, static_cast<T>(g7));
          gpuAtomicAdd(addr_g8, static_cast<T>(g8));
#else
          *addr_g1 += static_cast<T>(g1);
          *addr_g2 += static_cast<T>(g2);
          *addr_g3 += static_cast<T>(g3);
          *addr_g4 += static_cast<T>(g4);

          // Z_high
          *addr_g5 += static_cast<T>(g5);
          *addr_g6 += static_cast<T>(g6);
          *addr_g7 += static_cast<T>(g7);
          *addr_g8 += static_cast<T>(g8);
#endif
        } // if
      }   // ix
    }     // iy
  }       // iz
}

// Instantiate all function templates for float
template float
bilinear_interpolate_2D<float>(const float *input, int height, int width,
                               int channels, float y, float x, int c,
                               int index /* index for debug only*/);

template void roi_align_2D<float>(int index, int size, const float *input,
                                  const float *pSpatial_scale, int channels,
                                  int height, int width, int pooled_height,
                                  int pooled_width,
                                  const int32_t *pSampling_ratio,
                                  const bool *pAligned, const float *rois,
                                  float *output);

template void bilinear_interpolate_gradient_2D<float>(
    int height, int width, float y, float x, float &w1, float &w2, float &w3,
    float &w4, int &x_low, int &x_high, int &y_low, int &y_high, int index);

template void roi_align_gradient_2D<float>(
    int index, int size, const float *grad_output, const float *pSpatial_scale,
    int channels, int height, int width, int pooled_height, int pooled_width,
    const int32_t *pSampling_ratio, const bool *pAligned, const float *rois,
    float *grad_input, int n_stride, int h_stride, int w_stride, int c_stride);

template float bilinear_interpolate_3D<float>(
    const float *input, int depth, int height, int width, int channels, float z,
    float y, float x, int c, int index /* index for debug only*/);

template void
roi_align_3D<float>(int index, int size, const float *input,
                    const float *pSpatial_scale, int channels, int depth,
                    int height, int width, int pooled_depth, int pooled_height,
                    int pooled_width, const int32_t *pSampling_ratio,
                    const bool *pAligned, const float *rois, float *output);

template void bilinear_interpolate_gradient_3D<float>(
    int depth, int height, int width, float z, float y, float x, float &w1,
    float &w2, float &w3, float &w4, float &w5, float &w6, float &w7, float &w8,
    int &x_low, int &x_high, int &y_low, int &y_high, int &z_low, int &z_high,
    int index);

template void roi_align_gradient_3D<float>(
    int index, int size, const float *grad_output, const float *pSpatial_scale,
    int channels, int depth, int height, int width, int pooled_depth,
    int pooled_height, int pooled_width, const int32_t *pSampling_ratio,
    const bool *pAligned, const float *rois, float *grad_input, int n_stride,
    int d_stride, int h_stride, int w_stride, int c_stride);