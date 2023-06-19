import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread

class ImageFolderDataset(Dataset):
    def __init__(self, root, cache=None, is_test=False):
        '''
        修改了dataset中图片路径，不管真假图都修改为'../data/face/fake'
        '''
        '''
        若数据库没建立，则将数据放入数据库，并定义好self.root, self.images, self.size
        若已经有数据库，则从数据库取出self.root, self.images, self.size
        self.root = '../data/face'
        self.images = '0xxxx.png'(图片名，不包含路径)
        self.size = (512, 512)
        '''
        # root = '../data/face'; cache = ../data/face/local.db
        self.is_test = is_test
        # if cache is not None and os.path.isfile(cache):
        # if cache is not None and os.path.isfile(cache):
        #     with open(cache, 'rb') as f:
        #         self.root, self.images, self.size = pickle.load(f)
        # else:
        self.root = root # '../data/face'
        self.images = sorted(os.listdir(os.path.join(root, 'fake'))) # '../data/face/test_real'下的所有文件，一共1069张图
        
        if self.images[0] == " ": # 00000.png
            self.images.pop(0)
        tmp = imread(os.path.join(self.root, 'fake', self.images[0]))
        self.size = tmp.shape[:-1] # (512, 512, 3)[:-1] = (512, 512)
            # if cache is not None:
            #     with open(cache, 'wb') as f:
            #         pickle.dump((self.root, self.images, self.size), f)

    # TODO: rewrite this part
    def __getitem__(self, item):
        name = self.images[item]
        real_img = None if self.is_test else imread(os.path.join(self.root, 'fake', name)) # '../data/face/fake'
        fake_img = imread(os.path.join(self.root, 'fake', name)) # '../data/face/fake'
        return real_img, fake_img

    def __len__(self):
        return len(self.images) # 1069


class FaceCropDataset(Dataset): #TODO FaceCropDataset
    def __init__(self, image_dataset, pose_file, transform, crop_size=96):
        self.image_dataset = image_dataset
        self.transform = transform
        self.crop_size = crop_size

        if not os.path.isfile(pose_file):
            raise(FileNotFoundError('Cannot find pose data...'))
        self.poses = np.load(pose_file) # poses.shape = (1032, 2)

    def get_full_sample(self, item):
        # skip over bad items
        while True:
            real_img, fake_img = self.image_dataset[item]
            head_pos = self.poses[item]
            if head_pos[0] == -1 or head_pos[1] == -1:
                item = (item + 1) % len(self.image_dataset)
            else:
                break

        # crop head image
        size = self.image_dataset.size
        left = int(head_pos[0] - self.crop_size / 2)  # don't suppose left will go out of bound eh?
        left = left if left >= 0 else 0
        left = size[1] - self.crop_size if left + self.crop_size > size[1] else left

        top = int(head_pos[1] - self.crop_size / 2)
        top = top if top >= 0 else 0
        top = size[0] - self.crop_size if top + self.crop_size > size[0] else top


        real_head = None if self.image_dataset.is_test else \
                    self.transform(real_img[top: top + self.crop_size, left: left + self.crop_size,  :])
        fake_head = self.transform(fake_img[top: top + self.crop_size, left: left + self.crop_size,  :])

        # from matplotlib.pyplot import imshow, show
        # imshow(real_head.numpy().transpose((2,1,0)))
        # show()
        # imshow(fake_head.numpy().transpose((2,1,0)))
        # show()

        # keep full fake image to visualize enhancement result
        return real_head, fake_head, \
               top, top + self.crop_size, \
               left, left + self.crop_size, \
               real_img, fake_img

    def __getitem__(self, item):
        real_head, fake_head, _, _, _, _, _, _ = self.get_full_sample(item)
        return {'real_heads': real_head, 'fake_heads': fake_head}

    def __len__(self):
        return len(self.image_dataset) # 1069
