import os
import sys

import tensorflow as tf
from tensorflow.python.framework import ops

class RoiAlign:

	def __init__(self, fast_impl=True):
		if fast_impl:
			env_name = 'ROI_ALIGN_LIB'
			if not env_name in os.environ:
				sys.exit('Please set the %s environment variable to point to '
			 'the op implementation binary or set fast_impl=False in RoiAlign to use the '
			 'slow plain TF implementation.' % env_name)
			roi_align_op_path = os.environ[env_name]
			self.roi_align_lib = tf.load_op_library(roi_align_op_path)
			self.fun = self.roi_align_fast

			# Gradient of roi_align_fast
			@ops.RegisterGradient("RoiAlign")
			def roi_align_fast_grad(op, grad):
				forward_image_input = op.inputs[0]
				boxes = op.inputs[1]
				spatial_scale = op.inputs[2]
				sampling_ratio = op.inputs[3]
				aligned = op.inputs[4]
				pooled_dims = op.inputs[5]

				result = self.roi_align_lib.RoiAlignGrad(
					grad_output=grad,
					boxes=boxes, 
					spatial_scale=spatial_scale, 
					sampling_ratio=sampling_ratio,
					aligned=aligned,
					forward_image_input=forward_image_input,
					pooled_dims=pooled_dims
				)

				return [result, None, None, None, None, None]

		else:
			self.fun = self.roi_align
		
		self.i_dtype = tf.int32
		self.f_dtype = tf.float32

	def __call__(self, input_, boxes, output_size=(64, 64), spatial_scale = 1.0, sampling_ratio=-1, aligned = False):
		return self.fun(input_, boxes, output_size, spatial_scale, sampling_ratio, aligned)

	def roi_align_fast(self, input_, boxes, output_size=(64, 64), spatial_scale = 1.0, sampling_ratio=-1, aligned = False):
		'''
		A fast implementation that uses binary CPU or CUDA kernel.
		'''
		result = self.roi_align_lib.RoiAlign(
			input=input_,
			boxes=boxes,
			spatial_scale=tf.constant([spatial_scale], tf.float32),
			sampling_ratio=tf.constant([sampling_ratio], tf.int32),
			aligned=tf.constant([aligned], tf.bool),
			pooled_dims=tf.constant(output_size, tf.int32)
		)
		return result

	def bilinear_interpolate(self, input_, y, x, ihdim, iwdim, g_bidx, g_chidx):
		lo_threshold_y = tf.zeros_like(y)
		lo_threshold_x = tf.zeros_like(x)
		
		y = tf.reduce_max(tf.stack([y, lo_threshold_y]), axis=0)
		x = tf.reduce_max(tf.stack([x, lo_threshold_x]), axis=0)

		y_low = tf.cast(y, self.i_dtype)
		x_low = tf.cast(x, self.i_dtype)

		hi_threshold_y = tf.ones_like(y, self.i_dtype)*ihdim-1
		hi_threshold_x = tf.ones_like(x, self.i_dtype)*iwdim-1

		y_high = y_low + 1
		x_high = x_low + 1

		y_high = tf.reduce_min(tf.stack([y_high, hi_threshold_y]), axis=0)
		x_high = tf.reduce_min(tf.stack([x_high, hi_threshold_x]), axis=0)

		y = tf.reduce_min(tf.stack([y, tf.cast(hi_threshold_y, self.f_dtype)]), axis=0)
		x = tf.reduce_min(tf.stack([x, tf.cast(hi_threshold_x, self.f_dtype)]), axis=0)

		ly = y - tf.cast(y_low, self.f_dtype)
		lx = x - tf.cast(x_low, self.f_dtype)
		hy = 1. - ly
		hx = 1. - lx

		v1_idx = tf.stack([g_bidx, y_low, x_low, g_chidx], axis=1)
		v2_idx = tf.stack([g_bidx, y_low, x_high, g_chidx], axis=1)
		v3_idx = tf.stack([g_bidx, y_high, x_low, g_chidx], axis=1)
		v4_idx = tf.stack([g_bidx, y_high, x_high, g_chidx], axis=1)

		v1 = tf.gather_nd(input_, v1_idx)
		v2 = tf.gather_nd(input_, v2_idx)
		v3 = tf.gather_nd(input_, v3_idx)
		v4 = tf.gather_nd(input_, v4_idx)

		w1 = hy * hx
		w2 = hy * lx
		w3 = ly * hx
		w4 = ly * lx

		return w1*v1 + w2*v2 + w3*v3 + w4*v4

	def compute_grid_resolution(self, boxes, spatial_scale, sampling_ratio, offset, pooled_height, pooled_width):
		if sampling_ratio > 0:
			roi_bin_grid_h = sampling_ratio
			roi_bin_grid_w = sampling_ratio
		else:
			box_start_w = boxes[:, 1] * spatial_scale - offset
			box_start_h = boxes[:, 2] * spatial_scale - offset
			box_end_w = boxes[:, 3] * spatial_scale - offset
			box_end_h = boxes[:, 4] * spatial_scale - offset
			
			# shape: (N,)
			box_height = box_end_h - box_start_h
			box_width = box_end_w - box_start_w

			roi_bin_grid_h = tf.cast(tf.math.ceil(box_height / pooled_height), self.i_dtype)
			roi_bin_grid_w = tf.cast(tf.math.ceil(box_width / pooled_width), self.i_dtype)

		return roi_bin_grid_h, roi_bin_grid_w

	def roi_align(self, input_, boxes, output_size=(1, 1), spatial_scale = 1.0, sampling_ratio=-1, aligned = False):
		'''
		A pure Tensorflow but slow implementation of RoiAlign.

		input	(B, H, W, C)
		boxes	(N, 5): (x1, y2, x2, y2)
		return	(N, output_size[0], output_size[1])
		'''
		
		if len(output_size) > 2:
			sys.exit("Current implementation of plain TF roi align only supports 2D.")

		if aligned:
			offset = 0.5
		else:
			offset = 0.0
		
		# Input params
		chdim = input_.shape[-1]
		ihdim = input_.shape[1]
		iwdim = input_.shape[2]

		image_ids = boxes[:, 0]

		# Output params
		pooled_height = output_size[0]
		pooled_width = output_size[1]

		roi_bin_grid_h, roi_bin_grid_w = self.compute_grid_resolution(boxes, spatial_scale, sampling_ratio, offset, pooled_height, pooled_width)

		# Number of sampling grid points for each roi
		roi_grid_pixels = roi_bin_grid_h*roi_bin_grid_w*pooled_height*pooled_width*chdim
		roi_grid_start_indices = tf.cumsum(roi_grid_pixels, exclusive=True)
		all_roi_pixels = tf.reduce_sum(roi_grid_pixels)

		arr = tf.stack(
			[
				roi_bin_grid_h,
				roi_bin_grid_w,
				tf.cast(image_ids, self.i_dtype)
			]
		)

		arrs = tf.repeat(arr, roi_grid_pixels, axis=1)

		g_roi_bin_grid_h = arrs[0, :]
		g_roi_bin_grid_w = arrs[1, :]
		g_bidx = arrs[2, :]

		arr = tf.stack([
			boxes[:, 1] * spatial_scale - offset,
			boxes[:, 2] * spatial_scale - offset,
			boxes[:, 3] * spatial_scale - offset- boxes[:, 1] * spatial_scale - offset,
			boxes[:, 4] * spatial_scale - offset - boxes[:, 2] * spatial_scale - offset
		])

		g_roi_start_w = tf.repeat(boxes[:, 1] * spatial_scale - offset, roi_grid_pixels)
		g_roi_start_h = tf.repeat(boxes[:, 2] * spatial_scale - offset, roi_grid_pixels)

		g_roi_width = tf.repeat(boxes[:, 3] * spatial_scale - offset, roi_grid_pixels) - g_roi_start_w
		g_roi_height = tf.repeat(boxes[:, 4] * spatial_scale - offset, roi_grid_pixels) - g_roi_start_h

		# Create the sampling grid.
		# We need to create a unique sampling grid for each roi as the number of 
		# sampling points depends on the roi size.
		g_yidx = []
		g_xidx = []
		g_chidx = []
		n_rois = tf.shape(image_ids)[0]
		for roi_id in range(n_rois):
			image_id = tf.cast(image_ids[roi_id], self.i_dtype)
			b, y, x, ch = tf.meshgrid(
				image_id, 
				tf.range(pooled_height*roi_bin_grid_h[roi_id]), 
				tf.range(pooled_width*roi_bin_grid_w[roi_id]), 
				tf.range(chdim), indexing='ij')

			g_yidx.append(tf.cast(tf.reshape(y, (-1,)), self.f_dtype))
			g_xidx.append(tf.cast(tf.reshape(x, (-1,)), self.f_dtype))
			g_chidx.append(tf.cast(tf.reshape(ch, (-1,)), self.i_dtype))

		g_yidx = tf.concat(g_yidx, axis=0)
		g_xidx = tf.concat(g_xidx, axis=0)
		g_chidx = tf.concat(g_chidx, axis=0)

		if not aligned:
			# Force malformed ROIs to be 1x1
			g_roi_width = tf.reduce_max(tf.stack([g_roi_width, tf.ones_like(g_roi_width)], axis=1), axis=1)
			g_roi_height = tf.reduce_max(tf.stack([g_roi_height, tf.ones_like(g_roi_height)], axis=1), axis=1)

		g_bin_size_h = g_roi_height / pooled_height
		g_bin_size_w = g_roi_width / pooled_width

		y = g_roi_start_h + (g_yidx+0.5)*(g_bin_size_h/tf.cast(g_roi_bin_grid_h, self.f_dtype))
		x = g_roi_start_w + (g_xidx+0.5)*(g_bin_size_w/tf.cast(g_roi_bin_grid_w, self.f_dtype))

		elems = self.bilinear_interpolate(input_, y, x, ihdim, iwdim, g_bidx, g_chidx)

		# Finally, average pool the bins
		result_rois = []
		for roi_id in range(n_rois):
			act_roi_bin_h = roi_bin_grid_h[roi_id]
			act_roi_bin_w = roi_bin_grid_w[roi_id]		
			act_roi = elems[roi_grid_start_indices[roi_id]:roi_grid_start_indices[roi_id]+roi_grid_pixels[roi_id]]
			act_roi = tf.reshape(act_roi, (1, pooled_height*act_roi_bin_h, pooled_width*act_roi_bin_w, -1))
			act_roi = tf.nn.avg_pool2d(act_roi, (act_roi_bin_h, act_roi_bin_w), (act_roi_bin_h, act_roi_bin_w), padding='SAME')
			result_rois.append(act_roi)

		result = tf.concat(result_rois, axis=0)

		return result
