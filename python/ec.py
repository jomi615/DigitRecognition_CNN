import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
import re
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

folder_path = "../images"
pattern = r'image\d+\.(JPG|jpg|png)'  # Matches filenames"

image_files = []
for img in os.listdir(folder_path):
    if re.match(pattern, img):
        image_files.append(os.path.join(folder_path, img))

pic = cv2.imread(image_files[3])

#Getting the bounded areas:
# Ref: https://stackoverflow.com/questions/67696370/bounding-boxes-on-handwritten-digits-with-opencv
gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
original = pic.copy()
threshValue, binaryImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ROI_number = 0
cv2.floodFill(binaryImage, None, (0, 0), 0)
contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digit_detected = []  # Create an empty list to store bounding boxes
counter = 1
padding = 10 # Adjust the padding size as needed

for _, c in enumerate(contours):
    # Get the bounding rectangle of the current contour:
    boundRect = cv2.boundingRect(c)

    # Get the bounding rectangle data:
    rectX = boundRect[0]
    rectY = boundRect[1]
    rectWidth = boundRect[2]
    rectHeight = boundRect[3]

    # Estimate the bounding rect area:
    rectArea = rectWidth * rectHeight

    # Set a min area threshold
    minArea = 18

    # Filter blobs by area:
    if rectArea > minArea:
        rectX -= padding
        rectY -= padding
        rectWidth += 2 * padding
        rectHeight += 2 * padding

        # Draw bounding box:
        rectX = max(rectX, 0)
        rectY = max(rectY, 0)
        rectWidth = min(rectWidth, pic.shape[1] - rectX)
        rectHeight = min(rectHeight, pic.shape[0] - rectY)

        color = (0, 255, 0)
        cv2.rectangle(gray, (int(rectX), int(rectY)),
                      (int(rectX + rectWidth), int(rectY + rectHeight)), color, 1)
        # plt.imshow(gray, cmap = 'gray')
        currentCrop = gray[rectY:rectY + rectHeight, rectX:rectX + rectWidth]
        digit_detected.append(currentCrop)
        counter+=1

layers[0]['batch_size'] = 1

for i in range(len(digit_detected)):
    img_cor = 255-digit_detected[i] 
    img_resized = cv2.resize(img_cor, (28, 28))
    img_resized = img_resized.astype(float) / 255.0
    cptest, P = convnet_forward(params, layers, img_resized, test=True)
    # Display the saved digit image
    plt.subplot(3,17,i+1)
    plt.imshow(digit_detected[i], cmap ="gray")
    plt.title(f'{np.argmax(P)}')
    plt.axis('off')
