import os
import torch
from models.models_compared import GLU_Net
import argparse
import imageio
from matplotlib import pyplot as plt
from Utils.pixel_wise_mapping import remap_using_flow_fields
import numpy as np
import cv2
import scipy.io as sio
import pandas as pd
import seaborn as sb


def sort_by_dist(elem):
    return [x[0] for x in elem]


def calculate_distances(map_x, map_y, map_x2, map_y2, matches):
    d = []
    X_l = []
    X_r = []
    row = map_x.shape[0]
    col = map_x.shape[1]
    threshold = 2
    for i in range(row):
        for j in range(col):
            x1 = int(round(map_x[i][j]))
            y1 = int(round(map_y[i][j]))
            if 0 <= x1 < col and 0 <= y1 < row:
                x2 = map_x2[y1][x1]
                y2 = map_y2[y1][x1]
                dist = np.sqrt((i - y2) ** 2 + (j - x2) ** 2)
                if dist < threshold:
                    d.append([(dist, x1, y1, j, i)])
    d.sort(key=sort_by_dist)
    insert(d, map_x, X_l, X_r)
    match = {'size_l': map_x.shape, 'size_r': map_y.shape, 'X_l': X_l, 'X_r': X_r}
    matches["Matches"].append(match)
    return d, matches


def insert(list, map_x, X_l, X_r):
    isAllowed = np.zeros(map_x.shape)
    row = map_x.shape[0]
    col = map_x.shape[1]
    radius = 15
    amount_of_matches = 300
    for m in range(len(list)):
        elem = list[m]
        x1 = [x[1] for x in elem][0]
        y1 = [x[2] for x in elem][0]
        j = [x[3] for x in elem][0]
        i = [x[4] for x in elem][0]
        if isAllowed[y1][x1] == 0:
            for k in range(-radius, radius):
                for l in range(-radius, radius):
                    if 0 <= y1 + k < row and 0 <= x1 + l < col:
                        isAllowed[y1 + k][x1 + l] = 1
            X_l.append([x1, y1])
            X_r.append([j, i])
            if len(X_l) == amount_of_matches:
                return


def pad_to_same_shape(im1, im2):
    # pad to same shape both images with zero
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, pad_y_1, 0, pad_x_1, 0, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, pad_y_2, 0, pad_x_2, 0, cv2.BORDER_CONSTANT)

    return im1, im2


parser = argparse.ArgumentParser(description='Test GLUNet on a pair of images')
parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/',
                    help='Directory containing the pre-trained-models.')
parser.add_argument('--pre_trained_model', type=str, default='DPED_CityScape_ADE',
                    help='Name of the pre-trained-model.')
args = parser.parse_args()
torch.cuda.empty_cache()
torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # either gpu or cpu
dataset_name = 'CPC'
pairs_which_dataset = pd.read_csv(dataset_name + '/pairs_which_dataset.txt')
pairs_gts = np.loadtxt(dataset_name + '/pairs_with_gt.txt')
pairs_gts = pd.DataFrame(pairs_gts)
pairs = pairs_gts[pairs_gts.columns[0:2]]
l_pairs = pairs[pairs.columns[0]]
r_pairs = pairs[pairs.columns[1]]
size = np.size(pairs_gts[0])
matches = {'Matches': []}
with torch.no_grad():
    network = GLU_Net(path_pre_trained_models=args.pre_trained_models_dir,
                      model_type=args.pre_trained_model,
                      consensus_network=False,
                      cyclic_consistency=True,
                      iterative_refinement=True,
                      apply_flipping_condition=False)
    for idx in range(size):
        print(idx)
        l = "{:08}".format(int(l_pairs[idx]))
        r = "{:08}".format(int(r_pairs[idx]))
        folder = pairs_which_dataset.iloc[idx]
        I1 = dataset_name + '/' + folder[0] + 'Images/' + l + '.jpg'
        I2 = dataset_name + '/' + folder[0] + 'Images/' + r + '.jpg'
        # I2 = imageio.imread(pairs_which_dataset(idx) + 'Images/' sprintf('%.8d.jpg', r)]);

        try:
            source_image = imageio.imread(I1)
            target_image = imageio.imread(I2)
            source_image, target_image = pad_to_same_shape(source_image, target_image)
        except:
            raise ValueError('It seems that the path for the images you provided does not work ! ')

        if np.ndim(source_image) == 2:
            source_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2RGB)
        if np.ndim(target_image) == 2:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)
        # convert numpy to torch tensor and put it in right shape
        source_image_ = torch.from_numpy(source_image).permute(2, 0, 1).unsqueeze(0)
        target_image_ = torch.from_numpy(target_image).permute(2, 0, 1).unsqueeze(0)
        # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
        # specific pre-processing (/255 and rescaling) are done within the function.

        # source to target
        estimated_flow = network.estimate_flow(source_image_, target_image_, device, mode='channel_first')
        warped_source_image, map_X, map_Y = remap_using_flow_fields(source_image,
                                                                    estimated_flow.squeeze()[0].cpu().numpy(),
                                                                    estimated_flow.squeeze()[1].cpu().numpy())

        # target to source
        estimated_flow2 = network.estimate_flow(target_image_, source_image_, device, mode='channel_first')
        warped_target_image, map_X2, map_Y2 = remap_using_flow_fields(target_image,
                                                                      estimated_flow2.squeeze()[0].cpu().numpy(),
                                                                      estimated_flow2.squeeze()[1].cpu().numpy())
        dd, matches = calculate_distances(map_X, map_Y, map_X2, map_Y2, matches)
sio.savemat('SIFT-RT.mat', matches)
