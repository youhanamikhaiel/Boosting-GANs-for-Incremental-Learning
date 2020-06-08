import torch
import numpy as np
import numpy.random as random
import params
import utils
import FilteredGenerator
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10


def run(config, gan_model, num_instances):

  c_num_instances = num_instances/config['n_classes']
  filter_ratio = []
  for i in range(config['n_classes']):
    file_name = 'samples_class' + str(i)
    data,_ = FilteredGenerator.generate_samples_cond(config,c_num_instances*5,gan_model,i)
    data, ratio = FilteredGenerator.filter_data(config,data,c_num_instances,'resnet20',i)
    filter_ratio.append(float(ratio))
    target_class = np.ones((data.shape[0]), dtype=int)*i
	
    print('Images shape: %s, Labels shape: %s' % (data.shape, target_class.shape))
    #npz_filename = '%s/%s.npz' % (config['samples_root'], 'samples_total')
    npz_filename = '%s/%s.npz' % ('samples', file_name)
    print('Saving npz to %s...' % npz_filename)
    np.savez(npz_filename, **{'x': data.numpy(), 'y': target_class})  

  plt.bar(range(10),filter_ratio)
  print('Filter ratio is: ',filter_ratio)
  plt.show()
	
  del data, target_class

  total_data_x = torch.IntTensor([])
  total_data_y = torch.IntTensor([])
  for i in range(config['n_classes']):
    filedir = 'samples/samples_class' + str(i) +'.npz'
    data1 = np.load(filedir)
    total_data_x = torch.cat((total_data_x,torch.IntTensor(data1['x'])),dim=0)
    total_data_y = torch.cat((total_data_y,torch.IntTensor(data1['y'])),dim=0)
  s = np.arange(total_data_x.shape[0])
  random.shuffle(s)

  print('Images shape: %s, Labels shape: %s' % (total_data_x.shape, total_data_y.shape))
  #npz_filename = '%s/%s.npz' % (config['samples_root'], 'samples_total')
  npz_filename = '%s/%s.npz' % ('samples', 'samples_total')
  print('Saving npz to %s...' % npz_filename)
  np.savez(npz_filename, **{'x': total_data_x[s].numpy(), 'y': total_data_y[s].numpy()})    

  #prepare real data
  print('Preparing real data....')
  (x_train, y_train), (_, _) = cifar10.load_data()
  x_train = np.transpose(x_train,(0,3,1,2))
  y_train = y_train.reshape((-1,))
  ofile = 'CIFAR10_training'
  npz_filename = '%s/%s.npz' % ('samples/real_data', ofile) 
  np.savez(npz_filename, **{'x': x_train, 'y': y_train})
  print('Real data successfully prepared..!!')
	
  #preparing initial weights	
  weights = np.ones((50000,))
  ofilew = 'CIFAR10_weights'
  npz_filename = '%s/%s.npz' % ('samples/real_data', ofilew)
  np.savez(npz_filename, **{'w': weights})
  

def main():
  # Loading configuration
  config = params.params
	
  utils.update_config(config)

  run(config,'BigGAN_C10_seed0_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema',50000)


if __name__ == '__main__':
  main()
