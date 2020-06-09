# Boosting GANs for Incremental Learning

- Training BigGAN network on CIFAR10 dataset. 
- Then we generate samples from its generator network. 
- After that, we train a ResNet20 classifier on the generated samples. 
- We compute weights for each real CIFAR10 real data sample according to distance between them and distribution of generated samples.
- We re-train the BigGAN network on CIFAR10 dataset again but this time we give them the weights in the training procedure.

## Training a classifier

Training a classifier requires:
- pre-trained classifier weights in `classifier/weights/model_name.pth`
- pre-trained BigGAN weights in `weights/weights_folder_name/`

## In order to run the whole project, you have to:
- Prepare the dataset: `sh scripts/prepare_dataset.sh`
- Compute weights and train classifier: `sh scripts/train_get_weights.sh`
- Re-train BigGAN network: `sh BigGANOriginal/scripts/retrain_GAN.sh`
