'''
Loads a pretrained model and generate samples from it
'''
import argparse
import functools
import numpy as np
from tqdm import trange
import torch

import utils
import params
import numpy.random as random
from classifier.resnetw import resnet20

import torch.nn as nn


def generate_samples_cond(config,n_samples, model_name, y_class):
    n_samples = int(n_samples)

    # Initializing generator from configuration
    G = utils.initialize(config, model_name)

    # Update batch size setting used for G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_ = utils.prepare_z(G_batch_size, G.dim_z,
                         device='cuda', fp16=config['G_fp16'],
                         z_var=config['z_var'])

    # Preparing fixed y tensor
    y_ = utils.make_y(G_batch_size, y_class)

    # Sample function
    sample = functools.partial(utils.sample_cond, G=G, z_=z_, y_=y_)

    # Sampling a number of images and save them to an NPZ
    print('Sampling %d images from class %d...' % (n_samples, y_class))

    x, y = [], []
    for i in trange(int(np.ceil(n_samples / float(G_batch_size)))):
        with torch.no_grad():
            images, labels = sample()
        x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
        y += [labels.cpu().numpy()]
    x = np.concatenate(x, 0)[:n_samples]
    y = np.concatenate(y, 0)[:n_samples]

    return x, y
	
	
def filter_data(config,data,n_samples,weights_file,y_class):
	"""
	A function that takes the data and the number of samples to filter them according to the
	right class and minimum threshold. The output is the filtered_data and filter_ratio
	"""
	model = resnet20().to('cuda')
	model.load_state_dict(torch.load('./classifier/weights/%s.pth' % weights_file))
	model.eval()
	soft_layer = nn.Softmax(dim=1)
	
	data2 = (data/255.0) - np.array([[[0.4914]],[[0.4822]],[[0.4465]]]) / np.array([[[0.2470]],[[0.2435]],[[0.2616]]])
	data_len = data.shape[0]
	filtered_data = torch.Tensor([])
	y = torch.Tensor([y_class for _ in range(200)])
	thresh = torch.Tensor([0.9])
	for i in range(0,data_len,200):
		batch_data = torch.Tensor(data2[i:i+200]).to('cuda')
		batch_real_data = torch.Tensor(data[i:i+200])
		output = soft_layer(model(batch_data)).to('cpu')
		#output = soft_layer(output)
		filtered_index = torch.nonzero((torch.max(output,dim=1)[0]>thresh) & (torch.argmax(output,dim=1)==y))
		#print(filtered_data.shape)
		batch_real_data = torch.squeeze(batch_real_data[filtered_index].to('cpu'),1)
		filtered_data = torch.cat((filtered_data,batch_real_data),dim=0)
		#print('Batch data: ', y)
		#print('Filtered data: ', batch_data.size())
	
	#print('Data Length: ', data_len)
	filter_ratio = filtered_data.shape[0]/float(data_len)
	
	if filter_ratio >= 0.2:
		filtered_data = filtered_data[0:int(n_samples)]
	
	return filtered_data, filter_ratio
	
	
	
	
	
	
	
	
	
	
	
