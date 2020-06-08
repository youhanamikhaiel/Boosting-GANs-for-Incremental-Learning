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


class GANDataset(Dataset):
	"""ClassifierDataLoader"""

	def __init__(self, root_dir, weights_dir, transform=None):
		"""
		Args:
			root_dir (string): Directory with data
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.labels = np.load(root_dir)['y']
		self.data = np.load(root_dir)['x']
		self.weights = np.load(weights_dir)['w']
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = self.data[idx]
		label = self.labels[idx]
		label = np.array([label])
		weight = self.weights[idx]
		sample = {'image': image, 'label': label, 'weight': weight}
		#sample = {'image': image, 'label': label}

		if self.transform:
			image = self.transform(torch.Tensor(sample['image']))

		#sample = {'image': image, 'label': sample['label']}

		return image, label, weight
