# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import json
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import pickle

import utils.modelZoo as modelZoo
from utils.load_utils import *

ARMS_ONLY = [12,13,14,15,16,17]
N = 4


## main function demo script to run body2hands on frankmocap (smplx) predictions
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    print('> checkpoint', args.checkpoint)

    pipeline = args.pipeline
    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]
    pretrain_model = args.checkpoint

    tag = args.tag

    ######################################
    # Setup model
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    checkpoint_dir = os.path.split(args.checkpoint)[0]
    model_tag = os.path.basename(args.checkpoint).split(args.pipeline)[0]
    preprocess = np.load(os.path.join(checkpoint_dir,'{}{}_preprocess_core.npz'.format(model_tag, args.pipeline)))
    args.model = 'regressor_fcn_bn_32'
    model = getattr(modelZoo,args.model)()
    model.build_net(feature_in_dim, feature_out_dim)
    model.cuda()
    # Create model
    loaded_state = torch.load(pretrain_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded_state['state_dict'], strict=False)
    model.eval()

    test_X, total_body, total_cam = load_smplx(args.data_dir)

    ###### swap axis ######
    print("seq len", test_X.shape)
    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)
    
    ###### standardize ######
    body_mean_X = preprocess['body_mean_X']
    body_std_X = preprocess['body_std_X']
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_X = (test_X - body_mean_X) / body_std_X


    ##### convert to tensor ######
    inputData = Variable(torch.from_numpy(test_X)).cuda()

    # ===================forward=====================
    output = model(inputData) 

    # De-standardaize
    output_np = output.data.cpu().numpy() 
    output_np = output_np * body_std_Y + body_mean_Y
    output_np = np.swapaxes(output_np, 1, 2).astype(np.float32)

    ### saving as output in MTC format
    save_output(output_np, total_body, total_cam, 'models/', args.pipeline, tag=args.tag)


## process to save smplx based prediction to mtc format
def save_output(output, total_body, total_cam, model_path, pipeline, tag):
    feats = pipeline.split('2')
    out_feat = feats[1]
    start = 0
    
    for j in range(N): 
        frame_idx = start+j
        save_dir = os.path.join(args.data_dir, 'results/{}predicted_body_3d_frontal'.format(tag))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, '{:04d}.txt'.format(int(frame_idx)))
        
        ## note camera differences for visualization between MTC and frankmocap, 
        ## so we just use a frontal default camera.
        cam = [-12.9248, 51.8431, 209.5]
        shape = np.zeros(30)
        idk = np.zeros(200)

        ## load output from smpl body pose
        pose = np.zeros((62,3))
        pose[:20,:] = np.reshape(total_body[0][j], (-1,3))[:20,:]

        ## load predicted hands (convert from 6d to 3d)
        hands_r6d = np.reshape(output[0][j],(42,6))
        hands = rot6d_to_aa(hands_r6d)
        pose[-42:,:] = hands
        pose = np.reshape(pose, (-1))
        
        ## save in MTC format
        with open(save_path, 'w') as f:
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


## function to load smplx data from frankmocap plugin
def load_smplx(data_dir):
    result = np.zeros((N,36))
    body_result = np.zeros((N,72))
    cam_result = np.zeros((N,3))
    start = 0
    ## body_result contains original full body smpl (in original aa)
    ## result contains arms only smpl (in r6d)
    for i in range(N):
        file_path = os.path.join(args.data_dir, '{:05d}_prediction_result.pkl'.format(i+start))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        cam = data['pred_output_list'][0]['pred_camera']
        cam_result[i,:] = cam
        body = data['pred_output_list'][0]['pred_body_pose']
        body *= -1
        body_result[i,:] = body
        # convert aa to r6d
        body = np.reshape(body, (-1, 3))
        body = aa_to_rot6d(body)
        body = np.reshape(body[ARMS_ONLY,:], (-1))
        result[i,:] = body

    ## apply additional smoothing to original smpl for nice visualization
    body_result = body_result[np.newaxis,:,:]
    outputs_smoothed = np.copy(body_result)
    cam_result = cam_result[np.newaxis,:,:]
    cam_smoothed = np.copy(cam_result)
    for i in range(2, body_result.shape[1]-2):
        outputs_smoothed[:,i,:] = body_result[:,i-2,:]*0.1 + body_result[:,i-1,:]*0.2 + body_result[:,i,:]*0.4 + body_result[:,i+1,:]*0.2 + body_result[:,i+2,:]*0.1
        cam_smoothed[:,i,:] = cam_result[:,i-2,:]*0.1 + cam_result[:,i-1,:]*0.2 + cam_result[:,i,:]*0.4 + cam_result[:,i+1,:]*0.2 + cam_result[:,i+2,:]*0.1

    return result[np.newaxis,:,:], outputs_smoothed, cam_smoothed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--data_dir', type=str, required=True, help='input data directory with frankmocap output')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline to run')
    parser.add_argument('--tag', type=str, default='mocap_')
    args = parser.parse_args()
    print(args)
    main(args)
