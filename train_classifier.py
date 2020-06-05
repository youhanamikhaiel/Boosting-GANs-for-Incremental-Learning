import argparse
import functools
import numpy as np
from tqdm import trange

import utils
import params
import numpy.random as random

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt




def run():









def main():
    # Loading configuration
    config = params.params
	
    utils.update_config(config)

    run(config,
        num_batches,
        batch_size,
        model_name,
        class_model_name,
        ofile,
        threshold,
        num_workers,
        epochs)


if __name__ == '__main__':
    main()