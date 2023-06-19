'''Pose estimation (OpenPose)'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

openpose_dir = Path('./src/PoseEstimation/')

import sys
sys.path.append(str(openpose_dir))
sys.path.append('./src/utils')

print('make_source.py(2)')
# openpose
#from network.rtpose_vgg import gopenpose_diret_model
from evaluate.coco_eval import get_multiplier, get_outputs
from network.rtpose_vgg import get_model
# utils
from openpose_utils import remove_noise, get_pose


weight_name = './src/PoseEstimation/network/weight/pose_model.pth'

model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

print('make_source.py(3)')
'''make label images for pix2pix'''
test_img_dir = save_dir.joinpath('test_img_ori')
test_img_dir.mkdir(exist_ok=True)
test_label_dir = save_dir.joinpath('test_label_ori')
test_label_dir.mkdir(exist_ok=True)
test_head_dir = save_dir.joinpath('test_head_ori')
test_head_dir.mkdir(exist_ok=True)

print('make_source.py(4)')
from tqdm import tqdm
import os
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
openpose_dir = Path('./src/PoseEstimation/')
sys.path.append(str(openpose_dir))
sys.path.append('./src/utils')

from evaluate.coco_eval import get_multiplier, get_outputs
from network.rtpose_vgg import get_model

from openpose_utils import remove_noise, get_pose


img_dir = Path('./data/source/images')
pose_cords = []
for idx in tqdm(range(len(os.listdir(str(img_dir))))):
    img_path = img_dir.joinpath('{:05}.png'.format(idx)) # './data/source/images' + 'idx.png'
    img = cv2.imread(str(img_path)) # img.shape = (432, 768, 3)
    shape_dst = np.min(img.shape[:2]) # img.shape[:2] = (432, 768); shape_dst = 432

    
    oh = (img.shape[0] - shape_dst) // 2 # 0
    ow = (img.shape[1] - shape_dst ) // 2 # 336//2 == 168
    img = img[oh:oh + shape_dst, ow:ow + shape_dst] # img.shape = (432, 432, 3)
    img = cv2.resize(img, (512, 512)) #  img.shape = (512, 512, 3)

    multiplier = get_multiplier(img) # [0.359375, 0.71875, 1.078125, 1.4375, 1.796875]
    with torch.no_grad():
        paf, heatmap = get_outputs(multiplier, img, model, 'rtpose') # paf, heatmap (512, 512, 38), (512, 512, 19)
    # (512, 512, 18)
    r_heatmap = np.array([remove_noise(ht)
                          for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
        .transpose(1, 2, 0)
    heatmap[:, :, :-1] = r_heatmap # (512, 512, 19)
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    label, cord = get_pose(param, heatmap, paf) # label, cord - (512, 512), (17, 2)
    
    index = 13
    crop_size = 25
    try:
        head_cord = cord[index] # head_cord = [237. 132.] ;shape = 2
    except:
        head_cord = pose_cords[-1] # if there is not head point in picture, use last frame
        
    pose_cords.append(head_cord)
    # head.shape = (50, 50, 3)
    head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
           int(head_cord[0] - crop_size): int(head_cord[0] + crop_size), :]
    # plt.imshow(head) 

    # plt.savefig(str(test_head_dir.joinpath('pose_{}.jpg'.format(idx))))
    # plt.clf()
    cv2.imwrite(str(test_head_dir.joinpath('pose_{}.png'.format(idx))), head)
    cv2.imwrite(str(test_img_dir.joinpath('{:05}.png'.format(idx))), img)
    cv2.imwrite(str(test_label_dir.joinpath('{:05}.png'.format(idx))), label)
    if idx % 100 == 0 and idx != 0:
        pose_cords_arr = np.array(pose_cords, dtype=np.int)
        np.save(str((save_dir.joinpath('pose_source.npy'))), pose_cords_arr)