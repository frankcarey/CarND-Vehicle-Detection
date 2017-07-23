# Lesson functions

import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog

def convert_color(image, colorspace):
    if colorspace != 'RGB':
        if colorspace == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif colorspace == 'LUV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif colorspace == 'YUV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif colorspace == 'YCrCb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return image


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     hog_channel, feature_vec, vis=False):

    hog_params = {
        'orientations': orient,
        'pixels_per_cell': (pix_per_cell, pix_per_cell),
        'cells_per_block': (cell_per_block, cell_per_block),
        'transform_sqrt': False,
        'visualise': vis,
        'feature_vector': feature_vec,
    }

    # If we want all the channels, but not vectorized, then return one hog feature per channel.
    if hog_channel == 'ALL':
        hog_features = []
        hog_params['visualise'] = False

        # For every color channel, output
        for channel in range(img.shape[2]):
            #print('channel', channel)
            hog_features.append(hog(img[:, :, channel], **hog_params))

        if feature_vec == True:
            hog_features = np.ravel(hog_features)

        return hog_features

    else:
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img[:, :, hog_channel], **hog_params)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img[:, :, hog_channel], **hog_params)
            return features


def bin_spatial(img, size):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins, bins_range):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



# Define a function to extract features from a list of images
def extract_features(imgs, colorspace, orient, pix_per_cell, cell_per_block, hog_channel,
                     spatial_size, hist_bins, hist_range, use_spatial=True, use_hist=True, use_hog=True, debug=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        tmp_feat = []
        # Read in each one by one
        img = mpimg.imread(file)
        img = prepare_image(img, colorspace=colorspace)
        if use_hog:
            hog_features = get_hog_features(img, orient, pix_per_cell, cell_per_block, hog_channel=hog_channel, feature_vec=True)
            tmp_feat.append(hog_features)
            if debug:
                print('hog_features_shape', hog_features.shape)
        if use_spatial:
            spatial_features = bin_spatial(img, size=spatial_size)
            tmp_feat.append(spatial_features)
            if debug:
                print('spatial_features_shape', spatial_features.shape)
        if use_hist:
            hist_features = color_hist(img, nbins=hist_bins, bins_range=hist_range)
            tmp_feat.append(hist_features)
            if debug:
                print('hist_features_shape', hist_features.shape)
        features.append(np.concatenate(tmp_feat))
    return features


def prepare_image(img, factor=(1 / 255.), offset=-0.5, colorspace=None, reader='mpimg', normalize=False):

    img2 = img.copy()
    # First, Standardize everyting to RGB/0-255/uint8
    # matplotlib.imread loads png files as RGB, but as a float from 0-1.
    max_value = img2.max()
    min_value = img2.min()
    if max_value > 0:
        if normalize:
            #scaling seems to hurt?
            img2 = ((img2 - min_value) * (255.0 / (max_value - min_value))).astype(np.uint8)
        elif max_value <= 1:
            img2 = (img2 * 255.0).astype(np.uint8)

    else:
        print("normalizer: max value of image is not greater than 0. Is the image black?")

    if reader == 'cv2':
        # opencv loads images as BGR, so we need to convert to RGB first.
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if colorspace and colorspace != "RGB":
        img2 = convert_color(img2, colorspace)
    img2 = (img2.astype(np.float64) * factor) + offset

    #print(np.isfinite(img2).min())

    return img2


def get_subimg(img, ystart, ystop, scale):
    subimg = img[ystart:ystop, :, :]
    if scale != 1:
        imshape = subimg.shape
        subimg = cv2.resize(subimg, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
    return subimg


def get_search_plan(img, pix_per_cell, cell_per_block, window, cells_per_step):

    # Define blocks and steps as above
    nxblocks = (img.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (img.shape[0] // pix_per_cell) - cell_per_block + 1

    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    return nblocks_per_window, nxsteps, nysteps


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy