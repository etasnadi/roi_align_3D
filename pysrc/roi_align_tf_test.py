import statistics
import time
import math

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import roi_align_tf as roi_align

# Test gradients
def roi_align_test_3D_local():
	# --- Prepare inputs for RoiAlign  ---
	# (1, 4, 8, 8, 1)
	z, x, y = np.meshgrid(np.arange(4), np.arange(8), np.arange(8), indexing='ij')
	input_image = x+y+z
	input_image = np.expand_dims(np.expand_dims(input_image.astype(np.float32), 0), -1)
	
	#input_image = np.zeros((1, 1, 8, 8, 1), np.float32)
	input_image_var = tf.Variable(input_image)
	
	z1 = 1.0; z2 = 3.0
	boxes = np.array([
		[0, 3, 3, z1, 6, 6, z2]
	], np.float32)
	output_size = np.array([3, 4, 4], np.int32)

	# --- Compute forward value and gradient manually  ---

	# The point to be tested on the aligned roi:
	pw = 2; ph = 3; pd = 2

	roi_start_w = boxes[0][1]
	roi_start_h = boxes[0][2]
	roi_start_d = boxes[0][3]
	roi_width = boxes[0][4] - roi_start_w
	roi_height = boxes[0][5] - roi_start_h
	roi_depth = boxes[0][6] - roi_start_d
	
	pooled_depth = output_size[0]
	pooled_height = output_size[1]
	pooled_width = output_size[2]
	
	bin_size_w = roi_width / pooled_width
	bin_size_h = roi_height / pooled_height
	bin_size_d = roi_depth / pooled_depth
	
	roi_bin_grid_w = math.ceil(roi_width / pooled_width)
	roi_bin_grid_h = math.ceil(roi_height / pooled_height)
	roi_bin_grid_d = math.ceil(roi_depth / pooled_depth)	

	xp = roi_start_w + pw*bin_size_w + 0.5*bin_size_w/roi_bin_grid_w
	yp = roi_start_h + ph*bin_size_h + 0.5*bin_size_h/roi_bin_grid_h
	zp = roi_start_d + pd*bin_size_d + 0.5*bin_size_d/roi_bin_grid_d

	def bilinear(zp, yp, xp):
		xl = int(xp); yl = int(yp); zl = int(zp)
		xh = xl+1; yh = yl+1; zh = zl + 1
		xlc = xp - xl; ylc = yp - yl; zlc = zp - zl
		xhc = 1.-xlc; yhc = 1.-ylc; zhc = 1.-zlc
		coords = [
			[zl, yl, xl],
			[zl, yl, xh],
			[zl, yh, xl],
			[zl, yh, xh],
			[zh, yl, xl],
			[zh, yl, xh],
			[zh, yh, xl],
			[zh, yh, xh],
		]
		coeffs = [
			zhc*yhc*xhc,
			zhc*yhc*xlc,
			zhc*ylc*xhc,
			zhc*ylc*xlc,
			zlc*yhc*xhc,
			zlc*yhc*xlc,
			zlc*ylc*xhc,
			zlc*ylc*xlc,
		]
		return coords, coeffs

	pts, coeffs = bilinear(zp, yp, xp)

	assumed_grad = np.zeros_like(input_image)

	assumemd_forward_value = 0.0
	for pt, coeff in zip(pts, coeffs):
		assumemd_forward_value += input_image[0, pt[0], pt[1], pt[2], 0] * coeff
		assumed_grad[0, pt[0], pt[1], pt[2], 0] += coeff

	# --- Compute forward value and gradient using the Op and theck the differences ---

	ra = roi_align.RoiAlign()
	with tf.GradientTape() as tape:
		# 1. Check forward value at point (pd, ph, pw) on the output feature map
		
		# Image shape:	(1, 4, 8, 8, 1)
		# Roi shape:	(1, 2, 4, 4, 1)
		roi = ra.roi_align_fast(
			input_image_var, 
			tf.constant(boxes), 
			tf.constant(output_size))
		print('--> Computed forward value: %f assumed forward value: %f' % (roi[0, pd, ph, pw, 0], assumemd_forward_value))

		# 2. Check the input gradient with respect to output at (pd, ph, pw):

		output_gradients = np.zeros_like(roi)
		# Point of interest: (0, 2, 2)
		output_gradients[0, pd, ph, pw, 0] = 1.0
		# Grad shape: (1, 4, 8, 8, 1)
		grad = tape.gradient(roi, [input_image_var], output_gradients=tf.constant(output_gradients))

		print('--> Difference of the computed gradient and the assumed gradient:', tf.reduce_mean((assumed_grad[0]-grad)**2).numpy())
		

def roi_align_test_im_3D():
	depth = 32
	height = 32
	width = 32
	image = np.ones((1, depth, height, width, 1), np.float32)
	for z in range(depth):
		image[:, z, ...] = float(z)
	z1 = 8.
	z2 = 16.
	boxes = [
		[0, 8, 8, z1, 16, 16, z2],
		[0, 8, 8, z1, 16, 16, z2]
	]

	boxes_check = [
		[0, 8, 8, 16, 16],
		[0, 8, 8, 16, 16]
	]

	out_dims = [8, 8, 8]
	plt_dtype = np.float32

	image_t = tf.Variable(tf.convert_to_tensor(image))
	boxes_t = tf.Variable(tf.constant(boxes, tf.float32))

	# Slice at x=0
	image_check_t = tf.Variable(tf.convert_to_tensor(image[:, :, :, 0, :]))
	boxes_check_t = tf.Variable(tf.constant(boxes_check, tf.float32))
	out_dims_check = [8, 8]

	ra = roi_align.RoiAlign()

	measure_performance = False

	if measure_performance:
		tt = []
		for i in range(10):
			st = time.time()
			roi = ra.roi_align_fast(image_t, boxes_t, out_dims)
			tt.append(time.time()-st)

		print('Expired time: ', statistics.mean(tt)*len(tt))
	else:

		with tf.GradientTape() as tape:
			roi = ra.roi_align_fast(image_t, boxes_t, out_dims)
			roi_check = ra.roi_align_fast(image_check_t, boxes_check_t, out_dims_check)
			#print('Roi shape: ', roi.shape)
			g = tape.gradient(roi, [image_t, boxes_t])

	b = 0
	roi_b = 0

	z = 1

	plt.subplot(321)
	plt.imshow(image[b, z, ...].astype(plt_dtype))
	
	plt.subplot(322)
	plt.imshow(image[b, z+1, ...].astype(plt_dtype))

	plt.subplot(323)
	plt.imshow(roi[roi_b, 0, ...].numpy().astype(plt_dtype))
	
	plt.subplot(324)
	plt.imshow(roi[roi_b, 1, ...].numpy().astype(plt_dtype))

	plt.subplot(325)
	plt.imshow(roi_check[roi_b, ...].numpy().astype(plt_dtype))

	#plt.subplot(133)
	#plt.imshow((g[0][b].numpy()*255.0).astype(np.uint8))

	plt.show()

def roi_align_test_im():
	image = imageio.v2.imread('sample.png').astype(np.float32)

	image = np.expand_dims(image, 0)

	print('Image: shape', image.shape)

	image_t = tf.Variable(tf.convert_to_tensor(image))
	boxes_t = tf.Variable(tf.constant([
		[0, 0, 200.5, 450, 265],	# 50x50
		[0, 100, 100, 400.25, 400],	# 300x300
	], tf.float32))

	ra = roi_align.RoiAlign()

	with tf.GradientTape() as tape:
		roi = ra(image_t, boxes_t, [64, 64])
		g = tape.gradient(roi, [image_t, boxes_t])

	b = 0

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

	ax1.set_title('Input')
	ax1.imshow(image[b].astype(np.uint8))
	ax2.set_title('Gradient')
	ax2.imshow(g[0][b].numpy())
	ax3.set_title('Roi #1')
	ax3.imshow(roi[0].numpy().astype(np.uint8))
	ax4.set_title('Roi #2')
	ax4.imshow(roi[1].numpy().astype(np.uint8))

	plt.show()

def roi_align_test_simple():
	ibdim = 4
	ihdim = 100
	iwdim = 100
	ichdim = 1

	image = tf.zeros((ibdim, ihdim, iwdim, ichdim), tf.float32)
	
	# ROIs
	# [(b, x1, y1, x2, y2]

	# 10:20:4 --> 
	boxes = tf.constant([
		[0, 10, 12, 20, 24],
	], tf.float32)
	
	ra = roi_align.RoiAlign()
	roi = ra.roi_align(image, boxes, [4, 4])

if __name__ == '__main__':
	roi_align_test_im()
	#roi_align_test_im_3D()
	#roi_align_test_3D_local()
