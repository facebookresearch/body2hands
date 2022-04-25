# Copyright (c) Facebook, Inc. and its affiliates.
import json
import numpy as np
import os, sys
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.spatial.transform import Rotation as R

from shutil import copyfile
from PIL import Image,ImageDraw
from torchvision import transforms
import torch


FEATURE_MAP = {
    'arm2wh':((6*6), 42*6),
}

ARMS_ONLY = [12,13,14,15,16,17] #arms for mtc
EPSILON = 1e-10

## helper for calculating mean and standard dev
def mean_std(feat, data, rot_idx):
    if feat == 'wh':
       mean = data.mean(axis=2).mean(axis=0)[np.newaxis,:, np.newaxis]
       std =  data.std(axis=2).std(axis=0)[np.newaxis,:, np.newaxis]
       std += EPSILON
    else:
        mean = data.mean(axis=2).mean(axis=0)[np.newaxis,:, np.newaxis]
        std = np.array([[[data.std()]]]).repeat(data.shape[1], axis=1)
    return mean, std


## helper for calculating standardization stats
def calc_standard(train_X, train_Y, pipeline):
    rot_idx = -6
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    body_mean_X, body_std_X = mean_std(in_feat, train_X, rot_idx)
    if in_feat == out_feat:
        body_mean_Y = body_mean_X
        body_std_Y = body_std_X
    else:
        body_mean_Y, body_std_Y = mean_std(out_feat, train_Y, rot_idx)
    return body_mean_X, body_std_X, body_mean_Y, body_std_Y


## utility check if object is float
def is_float(n):
    try:
        float(n)
        return True
    except:
        return False


## utility function to convert from r6d space to axis angle
def rot6d_to_aa(r6ds):
    res = np.zeros((r6ds.shape[0], 3))
    for i,row in enumerate(r6ds):
        np_r6d = np.expand_dims(row, axis=0)
        np_mat = np.reshape(np_rot6d_to_mat(np_r6d)[0], (3,3))
        np_mat = R.from_matrix(np_mat)
        aa = np_mat.as_rotvec()
        res[i,:] = aa
    return res


def np_mat_to_rot6d(np_mat):
    """ Get 6D rotation representation for rotation matrix.
        Implementation base on
            https://arxiv.org/abs/1812.07035
        [Inputs]
            flattened rotation matrix (last dimension is 9)
        [Returns]
            6D rotation representation (last dimension is 6)
    """
    shape = np_mat.shape

    if not ((shape[-1] == 3 and shape[-2] == 3) or (shape[-1] == 9)):
        raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
            but found tensor with shape {}".format(shape[-1]))

    np_mat = np.reshape(np_mat, [-1, 3, 3])
    np_r6d = np.concatenate([np_mat[...,0], np_mat[...,1]], axis=-1)

    if len(shape) == 1:
        np_r6d = np.reshape(np_r6d, [6])

    return np_r6d


## utility function to convert from axis angle to r6d space
def aa_to_rot6d(vecs):
    res = np.zeros((vecs.shape[0], 6))
    for i,row in enumerate(vecs):
        np_mat = R.from_rotvec(row)
        np_mat = np_mat.as_dcm()
        np_mat = np.expand_dims(np_mat, axis=0) #e.g. batch 1
        np_r6d = np_mat_to_rot6d(np_mat)[0]
        res[i,:] = np_r6d
    return res


## utility function to convert from r6d space to rotation matrix
def np_rot6d_to_mat(np_r6d):
    shape = np_r6d.shape
    np_r6d = np.reshape(np_r6d, [-1,6])
    x_raw = np_r6d[:,0:3]
    y_raw = np_r6d[:,3:6]

    x = x_raw / np.linalg.norm(x_raw, ord=2, axis=-1)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, ord=2, axis=-1)
    y = np.cross(z, x)

    x = np.reshape(x, [-1,3,1])
    y = np.reshape(y, [-1,3,1])
    z = np.reshape(z, [-1,3,1])
    np_matrix = np.concatenate([x,y,z], axis=-1)

    if len(shape) == 1:
        np_matrix = np.reshape(np_matrix, [9])
    else:
        output_shape = shape[:-1] + (9,)
        np_matrix = np.reshape(np_matrix, output_shape)

    return np_matrix


## utility to load windows from outside files
def load_windows(data_dir, pipeline, num_samples=None, use_euler=False, require_image=False, require_audio=False, hand3d_image=False, use_lazy=False, test_smpl=False, temporal=False):
    preload_path = os.path.join(data_dir, 'filepaths.npy')
    if os.path.exists(preload_path):
        filepaths = np.load(preload_path, allow_pickle=True)
        feats = pipeline.split('2')
        in_feat, out_feat = feats[0], feats[1]
        p0_size, p1_size = FEATURE_MAP[pipeline]

        if os.path.exists(os.path.join(data_dir, 'full_bodies2.npy')):
            print('using super quick load', data_dir)
            p1_windows = np.load(os.path.join(data_dir, 'full_hands2.npy'), allow_pickle=True)
            p0_windows = np.load(os.path.join(data_dir, 'full_bodies2.npy'), allow_pickle=True)
            B,T = p0_windows.shape[0], p0_windows.shape[1]
            if in_feat == 'arm':
                p0_windows = np.reshape(p0_windows, (B,T,-1,6))
                p0_windows = p0_windows[:,:,ARMS_ONLY,:]
                p0_windows = np.reshape(p0_windows, (B,T,-1))
            if require_image:
                image_windows = np.load(os.path.join(data_dir, 'full_resnet.npy'), allow_pickle=True)

        if require_image:
            p0_windows = (p0_windows, image_windows)

        return p0_windows, p1_windows, filepaths, None


## utility to save results
def save_results(paths, output, pipeline, base_path, tag=''):
    feats = pipeline.split('2')
    out_feat = feats[1]
    paths = np.array(paths)

    for i in range(paths.shape[0]):
        print('working on', paths[i,0,0])
        for j in range(paths.shape[1]):
            vid_path, pnum, frame_idx = paths[i][j]
            vid_path = os.path.join(base_path, vid_path)
            if not os.path.exists(os.path.join(vid_path, 'results/')):
                os.makedirs(os.path.join(vid_path, 'results/'))

            if out_feat == 'wh':
                pred_dir = os.path.join(vid_path, 'results/{}predicted_body_3d_frontal/'.format(tag))
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                pred_path = os.path.join(pred_dir, '{:04d}.txt'.format(int(frame_idx)))

                ## set the ground truth estimated full body pose parameters for viewing
                gt_path = os.path.join(vid_path, 'body_3d_frontal/{:04d}.txt'.format(int(frame_idx)))
                with open(gt_path) as f:
                    lines = f.readlines()
                cam = lines[0]
                cam = [float(n) for n in cam.split(' ') if is_float(n)]
                pose = lines[1]
                pose = [float(n) for n in pose.split(' ') if is_float(n)]
                shape = lines[2]
                shape = [float(n) for n in shape.split(' ') if is_float(n)]
                idk = lines[3]
                idk = [float(n) for n in idk.split(' ') if is_float(n)]
                ## DONE set the ground truth estimated full body pose parameters for viewing


                ## fill in the predicted hands to the full body pose
                pose = np.reshape(pose, (62,3))
                if out_feat == 'wh':
                    hands_r6d = np.reshape(output[i][j],(42,6))
                    hands = rot6d_to_aa(hands_r6d)
                    pose[-42:,:] = hands
                pose = np.reshape(pose, (-1))
                ## DONE fill in the predicted hands to the full body pose


                ## writing prediciton to file
                with open(pred_path, 'w') as f:
                    for item in cam:
                        f.write("%s "%item)
                    f.write("\n")
                    for item in pose:
                        f.write("%s "%item)
                    f.write("\n")
                    for item in shape:
                        f.write("%s "%item)
                    f.write("\n")
                    for item in idk:
                        f.write("%s "%item)
                ## DONE writing prediciton to file
