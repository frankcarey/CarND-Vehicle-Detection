# Lesson functions

import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog

def convert_color(image, colorspace='YCrCb'):
    feature_image = None
    if colorspace != 'RGB':
        if colorspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif colorspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif colorspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif colorspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    return feature_image


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     hog_channel='ALL', vis=False, feature_vec=True):
    hog_params = {
        'orientations': orient,
        'pixels_per_cell': (pix_per_cell, pix_per_cell),
        'cells_per_block': (cell_per_block, cell_per_block),
        'transform_sqrt': True,
        'visualise': vis,
        'feature_vector': feature_vec
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


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



# Define a function to extract features from a list of images
def extract_features(imgs, colorspace, orient, pix_per_cell, cell_per_block, hog_channel,
                     spatial_size, hist_bins, use_spatial=True, use_hist=True, use_hog=True, debug=False):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        tmp_feat = []
        # Read in each one by one
        img = mpimg.imread(file)
        img = normalize_image(img, factor=(1 / 255.), colorspace=colorspace)
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
            hist_features = color_hist(img, nbins=hist_bins)
            tmp_feat.append(hist_features)
            if debug:
                print('hist_features_shape', hist_features.shape)
        features.append(np.concatenate(tmp_feat))
    return features


def normalize_image(img, factor, colorspace):
    if colorspace != "RGB":
        img = convert_color(img, colorspace)
    return img.astype(np.float32) * factor


def get_subimg(img, ystart, ystop, scale):
    subimg = img[ystart:ystop, :, :]
    if scale != 1:
        imshape = subimg.shape
        ctrans_tosearch = cv2.resize(subimg, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
    return subimg

def get_search_plan(img, pix_per_cell, cell_per_block, window=64, cells_per_step=2):

    # Define blocks and steps as above
    nxblocks = (img.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (img.shape[0] // pix_per_cell) - cell_per_block + 1

    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    return nblocks_per_window, nxsteps, nysteps

def find_cars(img):
    use_hog = True
    use_hist = True
    use_spatial = True
    colorspace = 'YCrCb'
    cells_per_step=2
    orient = 9
    window=64
    color_feature_size=(64, 64)
    pix_per_cell=8
    cell_per_block = 4
    ystart = 400
    ystop = 656
    scale = 1.5
    spatial_size = (8, 8)
    hist_bins = 16

    subimg_features = []
    hog_features = None


    # Make a copy so we can draw on it later. TODO: MOVE THIS OUT
    draw_img = np.copy(img)

    img = normalize_image(img, factor=(1 / 255.), colorspace=colorspace)

    img = get_subimg(img, ystart, ystop, scale)

    if use_hog:
        hog_features = get_hog_features(img, orient, pix_per_cell, cell_per_block, hog_channel='ALL', feature_vec=False)

    nblocks_per_window, nxsteps, nysteps = get_search_plan(img, pix_per_cell, cell_per_block, window=window, cells_per_step=cells_per_step)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            if use_hog:
                # Extract HOG for this patch
                hog_feat1 = hog_features[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog_features[1][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog_features[2][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                subimg_features.append(hog_features)

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Get color features
            if use_hog or use_hist:
                # Extract the image patch to prepare feature extraction.
                subimg = cv2.resize(img[ytop:ytop + window, xleft:xleft + window], color_feature_size)

                if use_spatial:
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    subimg_features.append(spatial_features)
                if use_hist:
                    hist_features = color_hist(subimg, nbins=hist_bins)
                    subimg_features.append(hist_features)
            return np.concatenate(subimg_features)