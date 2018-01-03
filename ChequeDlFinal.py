# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:18:26 2017

@author: Sohan Rai
"""
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

# Setting the image resize value
STANDARD_SIZE = (300, 167)
"""
    takes a filename and turns it into a numpy array of RGB pixels
    Then converts into an array of shape (1, m * n)
    """
def img_to_matrix(filename, verbose=False):
    img = Image.open(filename)
    if verbose==True:
        print ("changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    for i in range(len(img)):
        img[i]=list(img[i])
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]
#Read the train and test image sets
train_img_dir = "C:\\Users\\Sohan Rai\\Dropbox\\Studies\\Github codes\\images\\train\\"
test_img_dir = "C:\\Users\\Sohan Rai\\Dropbox\\Studies\\Github codes\\images\\test\\"
train_images = [train_img_dir+ f for f in os.listdir(train_img_dir)]
test_images = [test_img_dir+ f for f in os.listdir(test_img_dir)]
#Create train and test labels based on the file name
train_labels = ["cheque" if "cheque" in f.split('\\')[-1] else "drivers_license" for f in train_images]
test_labels = ["cheque" if "cheque" in f.split('\\')[-1] else "drivers_license" for f in test_images]

train_data = []
test_data = []
# Convert the train and test images into 1D arrays
for image in train_images:
    img = img_to_matrix(image)
    train_data.append(img)

for image in test_images:
    img = img_to_matrix(image)
    test_data.append(img)

train_data = np.array(train_data)
test_data = np.array(test_data)
########################################################################
# Perform PCA to reduce dimentionality. Adjust the value of n_components to maintain 90 percent variance
pca = RandomizedPCA(n_components=70)
train_x = pca.fit_transform(train_data)
variance = sum(pca.explained_variance_ratio_)
test_x = pca.transform(test_data)
# Classify the test images using KNN Classifier
knn = KNeighborsClassifier()
knn.fit(train_x, train_labels)
pd.crosstab(test_labels,knn.predict(test_x))
knn.predict(test_x)
test_labels
