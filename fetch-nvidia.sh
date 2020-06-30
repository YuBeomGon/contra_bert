#!/bin/bash -i
  
sudo apt-get install -y gnupg

 

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" >> /etc/apt/sources.list.d/cuda.list'

 

sudo add-apt-repository ppa:graphics-drivers/ppa

 

sudo apt-get install -y nvidia-driver-430
sudo apt-get install -y cuda-10-1
sudo apt-get install -y libcudnn7=7.6.4.38-1+cuda10.1
sudo apt-get install -y libcudnn7-dev=7.6.4.38-1+cuda10.1
