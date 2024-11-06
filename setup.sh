#!/usr/bin/env bash

conda env create -f environment.yml

# Create data directories
mkdir -p data/cifar-10h
mkdir -p data/cifar-10

# Download CIFAR-10H data
wget -P data/cifar-10h https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy
wget -P data/cifar-10h https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-probs.npy

# Download CIFAR-10 dataset
# This will be handled by torchvision when you first run the code
# But we can pre-download it using curl
wget -P data/cifar-10 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf data/cifar-10/cifar-10-python.tar.gz -C data/cifar-10