# CNIC

Source code for Chinese image captioning method based on deep multimodal semantic fusion runnable on GPU and CPU.

### License
This code is released under the MIT License (refer to the LICENSE file for details).

## Dependencies
#### 1. tensorflow
The model is trained using Tensorflow, a popular Python framework for training deep neural network. To install Tensorflow, please refer to  [Installing Tensorflow](https://www.tensorflow.org/install/).
#### 2. python libs
The code is written in python, you also need to install following python dependencies:
- bottle==0.12.13
- ipdb==0.10.3
- matplotlib==2.1.0
- numpy==1.13.3
- Pillow==4.3.0
- scikit-image==0.13.1
- scipy==1.0.0
- jieba==0.38

For convenience, you can alse use requirements.txt to install python dependencies:

	pip install -r requirements.txt

To use the evaluation script: see
[coco-caption](https://github.com/tylin/coco-caption) for the requirements.

## Hardware
Though you can run the code on CPU, we highly recommend you to equip a GPU card. To run on cpu, please use

	export CUDA_VISIBLE_DEVICES=""

## Prepare data
To generate training data for Flickr8k-CN, please use build_flickr8k_data.py script:

	python build_flickr8k_data.py
  
## Train single-lable visual encoding model
We use Google Inception V3 for single-lable visual encoding network: see
[Inception](https://github.com/tensorflow/models/tree/master/research/inception) for the instructions.

## Train multi-lable keyword prediction model
Please run train_keyword.py using gpu:

	CUDA_VISIBLE_DEVICES=0 python train_keyword.py
  
## Train multimodal caption generation model
For multimodal caption generation network use train.py:

	python train.py
  
## Generate caption and visualize
Use server.py to load models, and use client.py to request caption generation:

	python server.py
 	python client.py

## Tensorboard visualization
To use tensorboard to monitor training process:

	tensorboard --logdir="MODEL_PATH"
