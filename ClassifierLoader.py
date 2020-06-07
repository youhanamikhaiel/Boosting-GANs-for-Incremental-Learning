from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import numbers
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision.transforms import transforms
from PIL import Image
import torch.nn.functional as F

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def _get_image_size(img):
    if isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class ClassifierDataset(Dataset):
	"""ClassifierDataLoader"""

	def __init__(self, root_dir, transform=None):
		"""
		Args:
			root_dir (string): Directory with data
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.labels = np.load(root_dir)['y']
		self.data = np.load(root_dir)['x']/255.0
		#self.weights = np.load(root_dir)['w']
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = self.data[idx]
		label = self.labels[idx]
		label = np.array([label])
		#weight = self.weights[idx]
		#sample = {'image': image, 'label': label, 'weight': weight}
		sample = {'image': image, 'label': label}

		if self.transform:
			image = self.transform(torch.Tensor(sample['image']))

		#sample = {'image': image, 'label': sample['label']}

		return image, label
		
		
		
		
		
class RandomCrop(object):
    
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        #if w == tw and h == th:
            #return 0, 0, h, w

        i = random.randint(0, 6)
        j = random.randint(0, 6)
        return i, j, th, tw

    def __call__(self, img):
        img = torch.nn.functional.pad(img, (3,3,3,3), mode='constant', value=0)
        # pad the width if needed
        #if self.pad_if_needed and img.shape[1] > self.size[0]:
            #img = F.pad(img, (self.size[0] - img.shape[1], 0), self.padding_mode, self.fill)
        # pad the height if needed
        #if self.pad_if_needed and img.shape[2] > self.size[1]:
            #img = F.pad(img, (0, self.size[1] - img.shape[2]), self.padding_mode, self.fill)

        i, j, h, w = self.get_params(img, self.size)

        return img[:,i:i+h,j:j+w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)





class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            return torch.flip(img,dims=[2])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)




		
