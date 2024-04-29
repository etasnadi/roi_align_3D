#ifndef ROI_ALIGN_H_
#define ROI_ALIGN_H_

 #include <stdint.h>
 #include <math.h>

template <typename T>
#ifdef __CUDA_ARCH__
__host__
    __device__
#endif
        T
        bilinear_interpolate_2D(const T *input, int height, int width,
                                int channels, T y, T x, int c,
                                int index /* index for debug only*/);

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    roi_align_2D(int index, int size, const T *input, const T *pSpatial_scale,
                 int channels, int height, int width, int pooled_height,
                 int pooled_width, const int32_t *pSampling_ratio,
                 const bool *pAligned, const T *rois, T *output);

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    bilinear_interpolate_gradient_2D(int height, int width, T y, T x, T &w1,
                                     T &w2, T &w3, T &w4, int &x_low,
                                     int &x_high, int &y_low, int &y_high,
                                     int index);

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
                          int h_stride, int w_stride, int c_stride);

template <typename T>
#ifdef __CUDA_ARCH__
__host__
    __device__
#endif
        T
        bilinear_interpolate_3D(const T *input, int depth, int height,
                                int width, int channels, T z, T y, T x, int c,
                                int index /* index for debug only*/);

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    roi_align_3D(int index, int size, const T *input, const T *pSpatial_scale,
                 int channels, int depth, int height, int width,
                 int pooled_depth, int pooled_height, int pooled_width,
                 const int32_t *pSampling_ratio, const bool *pAligned,
                 const T *rois, T *output);

template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
    void
    bilinear_interpolate_gradient_3D(int depth, int height, int width, T z, T y,
                                     T x, T &w1, T &w2, T &w3, T &w4, T &w5,
                                     T &w6, T &w7, T &w8, int &x_low,
                                     int &x_high, int &y_low, int &y_high,
                                     int &z_low, int &z_high, int index);

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
                          int c_stride);

#endif