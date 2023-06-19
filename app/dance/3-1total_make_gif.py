# ../data/face/full_fake/
print('6.my_make_gif.py')
print('修改了label_dir和target_label_paths')
print('修改了target_dir和label_dir为模型40的输出图片')
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage import io

import matplotlib.animation as ani
from IPython.display import HTML
import matplotlib

source_dir = Path('./data/source/test_img_ori')
label_dir = Path('./results/target/test_40/images')
target_dir = Path('./data/face/full_fake/')

source_img_paths = sorted(source_dir.iterdir())
target_label_paths = sorted(label_dir.glob('*input_label*'))
target_synth_paths = sorted(target_dir.iterdir())


def animate(nframe):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    source_img = io.imread(source_img_paths[nframe])
    ax1.imshow(source_img)
    ax1.set_xticks([])
    ax1.set_yticks([])

    target_label = io.imread(target_label_paths[nframe])
    ax2.imshow(target_label)
    ax2.set_xticks([])
    ax2.set_yticks([])

    target_synth = io.imread(target_synth_paths[nframe])
    ax3.imshow(target_synth)
    ax3.set_xticks([])
    ax3.set_yticks([])


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

print('here1_制作anim')
anim = ani.FuncAnimation(fig, animate, frames=len(target_label_paths), interval=1000 / 24)
plt.close()

print('here2_在jupyter中播放anim(ignore)')
# js_anim = HTML(anim.to_jshtml())
 
print('here3_保存anim为gif')
anim.save("./face_output.gif", writer="imagemagick")
print('here4')