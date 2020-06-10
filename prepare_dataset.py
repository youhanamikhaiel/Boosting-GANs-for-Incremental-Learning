import torch
import numpy as np
import numpy.random as random
import params
import utils
import FilteredGenerator
import matplotlib.pyplot as plt
import BigGANOriginal.utils as utilso

def run(config, gan_model, num_instances):
	
  utils.seed_rng(config['seed'])

  c_num_instances = num_instances/config['n_classes']
  filter_ratio = []
  for i in range(config['n_classes']):
    file_name = 'samples_class' + str(i)
    data,_ = FilteredGenerator.generate_samples_cond(config,c_num_instances*3,gan_model,i)
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
  
  del total_data_x, total_data_y

  #prepare real data
  print('Preparing real data....')
  D_batch_size = 2000
  loaders = utilso.get_data_loaders(**{**config, 'batch_size': D_batch_size, 'shuffle': False})
  x_train = torch.Tensor([])
  y_train = torch.LongTensor([])
  for i,j in loaders[0]:
    x_train = torch.cat((x_train,i),dim=0)
    y_train = torch.cat((y_train,j),dim=0)
	
  ofile = 'CIFAR10_training'
  npz_filename = '%s/%s.npz' % ('samples/real_data', ofile) 
  np.savez(npz_filename, **{'x': x_train.numpy(), 'y': y_train.numpy()})
  print('Real data successfully prepared..!!')

  del x_train, y_train
	
  #preparing initial weights	
  weights = np.ones((50000,))
  ofilew = 'CIFAR10_weights'
  npz_filename = '%s/%s.npz' % ('BigGANOriginal/data_weights', ofilew)
  np.savez(npz_filename, **{'w': weights})
  

def main():
  
  gan_weight_file = 're_BigGAN_C10_seed0_Gch96_Dch96_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema'
	
  # Loading configuration
  if gan_weight_file[23] == '9':
    config = params.params96
  elif gan_weight_file[23] == '6':
    config = params.params64
	
  utils.update_config(config)
	
  run(config,gan_weight_file,50000)
  #run(config,'BigGAN_C10_seed0_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema',50000)

if __name__ == '__main__':
  main()

