import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
import cv2

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
m_train = xtrain.shape[1]

batch_size = 1
layers[0]['batch_size'] = batch_size

img = xtest[:,0]
img = np.reshape(img, (28, 28), order='F')
plt.imshow(img.T, cmap='gray')
plt.show()
plt.axis('off')


output = convnet_forward(params, layers, xtest[:,0:1])
output_1 = np.reshape(output[0]['data'], (28,28), order='F')


##### Fill in your code here to plot the features ######

#Extracting Convolution Layer

layer_1_data = output[1]['data']
layer_1_height = output[1]['height']
layer_1_width = output[1]['width']
layer_1_channel = output[1]['channel']
layer_1_batch_size = output[1]['batch_size']

layer_1 = layer_1_data.reshape(
    (layer_1_height, layer_1_width, layer_1_channel, layer_1_batch_size), order = 'F'
)


for i in range(1,21):
    plt.subplot(4,5,i)
    result = layer_1[:,:,i-1,0]
    plt.imshow(result.T, cmap = "gray")
    plt.axis('off')
plt.figure()
plt.show()


#Extracting RelU Layer

layer_2_data = output[2]['data']
layer_2_height = output[2]['height']
layer_2_width = output[2]['width']
layer_2_channel = output[2]['channel']
layer_2_batch_size = output[2]['batch_size']

layer_2 = layer_2_data.reshape(
    (layer_2_height, layer_2_width, layer_2_channel, layer_2_batch_size), order = 'F'
)

#an attempt to increase brightness of output because it is too dark
brightness_factor = 2.0  
for i in range(1,21):
    plt.subplot(4,5,i)
    result = layer_2[:,:,i-1,0]
    result_brightened = np.clip(result * brightness_factor, 0, 255).astype(np.uint8)
    plt.imshow(result_brightened.T, cmap = "gray")
    plt.axis('off')
plt.figure()
plt.show()