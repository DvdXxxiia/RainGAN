from keras.models import load_model
from numpy import load
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import cv2
import numpy as np


def load_images(path, size=(256, 256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)
    return asarray(data_list)


# dataset path
path = 'data/'
# load dataset A
dataA1 = load_images(path + 'train/'+'nondefect/')
dataA2 = load_images(path + 'val/'+'nondefect/')
dataA = vstack((dataA1, dataA2))
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB1 = load_images(path + 'train/'+'defect/')
dataB2 = load_images(path + 'val/'+'defect/')
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = 'Nondefect2Defect123_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)


# load and plot the prepared dataset
# load the dataset
data = load('Nondefect2Defect123_256.npz')
dataA, dataB = data['arr_0'], data['arr_1']
print('Loaded: ', dataA.shape, dataB.shape)
# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    return X

# load and prepare training images
def load_real_samples(filename):
    # load the dataset
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]
# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
        pyplot.show()


# load dataset
A_data, B_data = load_real_samples('Nondefect2Defect123_256.npz')
print('Loaded', A_data.shape, B_data.shape)
# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_000600.h5', cust)
model_BtoA = load_model('g_model_BtoA_000600.h5', cust)
# plot A->B->A
A_real = select_sample(A_data, 1)
B_generated = model_AtoB.predict(A_real)
B_generated_grey = cv2.cvtColor(B_generated[0], cv2.COLOR_BGR2GRAY)
A_real_conv=cv2.cvtColor(A_real[0],cv2.COLOR_BGR2GRAY)
A_real_conv=A_real_conv*127.5+127.5
B_generated_grey=B_generated_grey*127.5+127.5
cv2.imshow('generated_grey',np.uint8(B_generated_grey))
cv2.waitKey(10)
cv2.imwrite('A_real.png',A_real_conv)
cv2.imwrite('generated_grey.png',B_generated_grey)
A_reconstructed = model_BtoA.predict(B_generated)
#show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B
B_real = select_sample(B_data, 1)
B_real_conv=cv2.cvtColor(B_real[0],cv2.COLOR_BGR2GRAY)
B_real_conv=B_real_conv*127.5+127.5
cv2.imwrite('B_real.png',B_real_conv)
A_generated = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
#show_plot(B_real, A_generated, B_reconstructed)
cv2.imwrite('rotated_ori5.jpg', B_generated)