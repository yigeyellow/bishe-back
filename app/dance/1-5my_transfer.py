print('5.my_transfer.py生成web')
print("在base_model中加了 print('load model:',str(save_path))")
print('在transfer基础上，修改了model_path: checkpoints/target/40_net_G.pth')
print('web_dir: ./results/target/test_40')
# model_path: checkpoints/target/40_net_G.pth
import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
pix2pixhd_dir = Path('./src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from src.pix2pixHD.data.data_loader import CreateDataLoader
from src.pix2pixHD.models.models import create_model
import src.pix2pixHD.util.util as util
from src.pix2pixHD.util.visualizer import Visualizer
from src.pix2pixHD.util import html
import src.config.test_opt as opt

# model_path: checkpoints/target/40_net_G.pth
opt.which_epoch = 40

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

# results_dir='./results/';  name='target'; phase='test'; which_epoch='40'
# web_dir = .\results\target\40, 是webpage保存路径，也是图片保存路径
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)

for data in tqdm(dataset):
    minibatch = 1
    generated = model.inference(data['label'], data['inst'])

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()
