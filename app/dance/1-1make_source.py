print('1.make_source.py(1)')
print('将脸增加了png格式的存储')
print('设置torch.cuda.set_device(0)，由-1改为0')
print('删除了脸部的jpg')
print('修改512*512_img_path = test_img_ori')
print('head_path = ./data/source/test_head_ori/pose_x.png')
print('img_path = ./data/source/test_img_ori')
print('label_path = ./data/source/test_label_ori')
print('pose_cords_arr_path = ./data/source/pose_source.npy')
'''Download and extract video'''
import cv2
from pathlib import Path
import os
import torch
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)
# torch.cuda.set_device(-1)

data_dir = Path('./data/')
data_dir.mkdir(exist_ok=True)

save_dir = Path('./data/source/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

mp4_path = Path('../../file/mv.mp4')

if len(os.listdir('./data/source/images'))<100:
    cap = cv2.VideoCapture(str(mp4_path))
    i = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if(i >= 100):
            # flag, frame = cap.read()
            if flag == False or i >= 200:
                break
            cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(i-100))), frame)
            if i%100 == 0:
                print('Has generated %d picetures'%i)
        i += 1

'''Pose estimation (OpenPose)'''
import numpy as np
# import matplotlib.pyplot as plt
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
    if idx % 99 == 0 and idx != 0:
        pose_cords_arr = np.array(pose_cords, dtype=np.int)
        np.save(str((save_dir.joinpath('pose_source.npy'))), pose_cords_arr)
    