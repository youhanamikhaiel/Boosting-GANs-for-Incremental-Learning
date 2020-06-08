import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from classifier.resnetw import resnet20


global resnet_real_feats, dist

#dist = [ torch.Tensor([]).to('cuda') for _ in range(10) ]
#resnet_real_feats, indices = get_real_feats('resnet20')


##Computed only once
def get_real_feats(trainloader, weights_file, n_classes=10):
  """
    Computes the resnet features of cifar10 real image data
    The output is a list of tensors with len(list) = n_classes
    Each tensor inside the list represents the feature vectors of a class
    resnet_real_feats[0] is the feature vectors of class 0 with size (5000,10)
  """
  #load classifier
  model = resnet20().to('cuda')
  model.load_state_dict(torch.load('./classifier/weights/%s.pth' % weights_file))
  #model = resnet20().to('cuda')
  #model.load_state_dict(torch.load('resnet20.pth'))
  model.eval()

  #preparing resnet feature vectors of the real data
  #n_classes = 10
  classified_data = [ torch.Tensor([]).to('cuda') for _ in range(n_classes) ]
  resnet_real_feats = [ torch.Tensor([]).to('cuda') for _ in range(n_classes) ]
  index = [ torch.Tensor([]).to('cuda') for _ in range(n_classes) ]
  print()
  print('Computing feature vectors for real data....')
  #separate different classes instances in the dataset of real images
  for data, label in trainloader:
    for i in range(n_classes):
      index[i] = (label == i).nonzero()
      classified_data[i] = torch.squeeze(data[index[i]])

      #compute the resnet features for each class
      with torch.no_grad():
        for j in range(0,classified_data[i].shape[0],int(classified_data[i].shape[0]/2)):
          output = model(classified_data[i][j:j+int(classified_data[i].shape[0]/2)].to('cuda'))
          resnet_real_feats[i] = torch.cat((resnet_real_feats[i],output),dim=0)
    print('Computing feature vectors for real data done!')
    print()
    return resnet_real_feats, index


#Computed for every generated batch
def get_gen_feats(data,weights_file,n_classes=10):
  """
    Computes the resnet features of generated image data
    The output is a list of tensors with len(list) = n_classes
    Each tensor inside the list represents the feature vectors of a certain class
    resnet_gen_feats[0] is the feature vectors of class 0
  """
  #load classifier
  model = resnet20().to('cuda')
  model.load_state_dict(torch.load('./classifier/weights/%s.pth' % weights_file))
  #model = resnet20().to('cuda')
  #model.load_state_dict(torch.load('resnet20.pth'))
  model.eval()

  resnet_gen_feats = torch.Tensor([]).to('cuda')

    #compute the resnet features for each class
  with torch.no_grad():
    for j in range(0,data.shape[0],int(data.shape[0]/2)):
      output = model(data[j:j+int(data.shape[0]/2)].to('cuda'))
      resnet_gen_feats = torch.cat((resnet_gen_feats,output),dim=0)
  return resnet_gen_feats


def compute_distance(real_data_feats, gen_data_feats):
  """
  Computing the minimum distances of each batch generated data between each of 
  the vectors in the first tesnor and all the vectors in second tensor
  The output is a tensor of size (5000, nb_batches) in case of CIFAR10 
  """
  n_1, n_2 = real_data_feats.size(0), gen_data_feats.size(0)
  norms_1 = torch.sum(real_data_feats**2, dim=1, keepdim=True)
  norms_2 = torch.sum(gen_data_feats**2, dim=1, keepdim=True)
  norms = norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2)
  distances_squared = norms - 2 * real_data_feats.mm(gen_data_feats.t())
  min_distances_squared, _ = torch.min(torch.abs(distances_squared),dim=1)
  return torch.unsqueeze(min_distances_squared,dim=1)


def compute_all_distances(data,weights_file,current_class,n_classes=10):
  ind = current_class
  resnet_gen_feats = get_gen_feats(data,weights_file)

  dist[ind] = torch.cat((dist[ind],compute_distance(resnet_real_feats[ind], resnet_gen_feats)),dim=1)
  dist[ind], _ = torch.min(dist[ind],dim=1)
  dist[ind] = torch.unsqueeze(dist[ind],dim=1)

  return dist[ind]


def get_sample_weights(config):
	for i in range(config['n_classes']):
		filedir = 'samples/samples_class' + str(i) +'.npz'
		data1 = (np.load(filedir)['x']/255.0) - np.array([[[0.4914]],[[0.4822]],[[0.4465]]]) / np.array([[[0.2470]],[[0.2435]],[[0.2616]]])
		dist[i] = compute_all_distances(torch.FloatTensor(data1),'resnet20',i)

	sample_weights = torch.zeros((50000), dtype=torch.float32).to('cuda')
	for i in range(config['n_classes']):
		sample_weights[indices[i]] = dist[i]
		
	ofilew = 'CIFAR10_weights'
	npz_filename = '%s/%s.npz' % ('samples/real_data', ofilew)
	np.savez(npz_filename, **{'w': sample_weights})
		
	return sample_weights
