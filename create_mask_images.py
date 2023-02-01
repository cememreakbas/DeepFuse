import tifffile as tf
import numpy as np
import os

# creates single-mask images on-the-fly and returns them in a single array!

def create_train_data(input_path, gt_path):
    mask_imgs = np.zeros((144, 1024, 1024))  
	# first dimension is the number of available GT single-mask images (Gold segmentation annotations)
	# if mask images contain multiple masks, they should be split into single-mask images beforehand.
	# second&third dim. is the dimension of input images
    ii = 0
    print('Loading masks from ', input_path)
    for image_name in os.listdir(gt_path):  # Go through all the .tif files in the folder
        if image_name.endswith(".tif"):
            participant_image_name = 'mask' + image_name.split('man_seg')[1][:3] + '.tif'
            im_gt = tf.imread(os.path.join(gt_path, image_name))
            im_allmasks = tf.imread(os.path.join(input_path, participant_image_name))
            mask_im = np.zeros((im_allmasks.shape[0], im_allmasks.shape[1]), dtype='double')
            cur_label = np.unique(im_gt)[-1]
            mask_im[im_allmasks == cur_label] = 1.0	# to produce a binary image for each label since DeepFuse is a 2-class classifier
            mask_imgs[ii, :, :] = mask_im
            ii = ii+1

    mask_imgs = mask_imgs[..., np.newaxis]
    return mask_imgs

# creates single-mask GT images on-the-fly and returns them in a single array!

def create_gt_data(gt_path):
    mask_imgs = np.zeros((144, 1024, 1024))  
	# first dimension is the number of available GT single-mask images (Gold segmentation annotations)
	# if mask images contain multiple masks, they should be split into single-mask images beforehand.
	# second&third dim. is the dimension of input images
    gt_mask_imgs = np.zeros((144, 1024, 1024))
    ii = 0
    for image_name in os.listdir(gt_path):  # Go through all the .tif files in the folder
        if image_name.endswith(".tif"):
            print('Reading image', image_name)
            im_allmasks = tf.imread(os.path.join(gt_path, image_name))
            im_allmasks [im_allmasks > 0.5] = 1.0 # to produce a binary image
            im_allmasks [im_allmasks < 0.5] = 0.0
            gt_mask_imgs[ii, :, :] = im_allmasks
            ii = ii+1

    gt_mask_imgs = gt_mask_imgs[..., np.newaxis]
    return gt_mask_imgs
