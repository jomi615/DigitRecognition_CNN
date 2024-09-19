import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
import cv2
import os
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

folder_path = "../images/real_test"
image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

# sampled_photos = np.zeros([3,5])
all_img = []
layers[0]['batch_size'] = 1
conv=[]
prediction = []
for i in range(len(image_files)):
    #read images
    file = image_files[i]
    file_name = file
    img_src = f'../images/real_test/{file_name}'
    print(f'Reading {img_src}')

    pic = cv2.imread(img_src)
    # print(type(pic))

    #convert image to GRAY
    img = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    img_cor = 255-img 
    img_resized = cv2.resize(img_cor, (28, 28))
    img_resized = img_resized.astype(float) / 255.0
    cptest, P = convnet_forward(params, layers, img_resized, test=True)
    prediction.append(np.argmax(P))
    print("predicted: ", prediction[i])

print("Full prediction: ", prediction)
