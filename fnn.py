from __future__ import print_function, division
import scipy
#from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import optimizers
from keras import initializers
from keras import backend as K
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import os
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
import scipy.misc
from scipy.misc import imsave
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import math

#data process functions
def getConvPath(material, num):
  root = os.getcwd() +"/newData/" + material
  if "Pt_input_images" == material:
    newPath = root +"/Pt_convolution/Pt_convolution_" + str(num) + ".txt"
  elif "Pt-Mo5_input_images" == material:
    newPath = root +"/Pt-Mo_convolution/Pt_Mo5_convolution_" + str(num) +".txt"
  elif "Pt-Mo50_input_images" == material:
    newPath = root + "/Pt281-Mo280-convolution/Pt_Mo50_convolution_" + str(int(num)+2) + ".txt"
  else: print("Material key not found! Please check your spelling.")
  return newPath

def getMultislicePath(material, num):
  root = os.getcwd() + "/newData/" + material
  if "Pt_input_images" == material:
    newPath = root +"/Pt_multislice_16_phonons/Pt_" + str(num) + "_cl160mm_ss.tif"
  elif "Pt-Mo5_input_images" == material:
    newPath = root +"/Pt-Mo_multislice/pt_mo5_" + str(num) +"_cl160mm_ss.tif"
  elif "Pt-Mo50_input_images" == material:
    newPath = root + "/Pt281-Mo280-multislice/pt_mo50_" + str(int(num)+2) + "_cl160mm_ss.tif"
  else: print("material key not found! Please check your spelling.")
  return newPath

def getNumImages(material):
  if "Pt_input_images" == material:
    num = 20
  elif "Pt-Mo5_input_images" == material:
    num = 20
  elif "Pt-Mo50_input_images" == material:
    num = 18
  else:
    num = 0
  return num

def cutImage(image,height,width):
  newImage = image[:height,:width]
  return newImage

#returns list of images cut to be min height and width of the group
def cutImages(images):
  widths = []
  heights = []
  cutImages = []
  for image in images:
    widths.append(len(image[0]))
    heights.append(len(image))
  minWidth = min(widths)
  minHeight = min(heights)
  for i in range(len(images)):
    cutImages.append(cutImage(images[i],minHeight,minWidth))
  return cutImages

def padImage(image, desiredHeight, desiredWidth):
    leftSpace = int((desiredWidth - image.shape[1])/2)
    topSpace = int((desiredHeight - image.shape[0])/2)
    base = np.zeros((desiredHeight,desiredWidth))
    base[topSpace:image.shape[0]+topSpace,leftSpace:image.shape[1]+leftSpace]=image
    return base

#returns list of images with desired heigh and width
def formatImages(images,height,width):
    newImages = []
    for image in roundToZeroes(images):
        if image.shape[0] > height and image.shape[1] > width:
            newImages.append(cutImage(image))
        elif image.shape[0] <= height and image.shape[1] < width:
            newImages.append(padImage(image,height,width))
        elif image.shape[0] >= height and image.shape[1] <= width:
            newImages.append(padImage(image[:height,:],height,width))
        elif image.shape[0] < height and image.shape[1] >= width:
            newImages.append(padImage(image[:,:width],height,width))
    return newImages

# rounds any negative values in the matrix to zero. Requested by Dane
def roundToZeroes(images):
    for image in images:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j] < 0.0:
                    image[i,j] = 0.0
    return images

def cutPadding(image,height,width):
  h_dif = len(image) - height
  w_dif = len(image[0]) - width
  top = h_dif//2
  left = w_dif//2
  if h_dif % 2 == 1:
    bottom = top + 1
  else:
    bottom = top
  if w_dif % 2 == 1:
    right = left + 1
  else:
    right = left
  newImage = image[top:len(image)-bottom ,left:len(image[0])-right]
  return newImage

def kerasSSIM(y_true, y_pred):#may be wrong
    ## mean, std, correlation
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = (sig_x * sig_y)**0.5
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def customLoss(yTrue,yPred):
   # print(backend.shape(yTrue))
   # print(backend.shape(yPred))
    ssimVal = kerasSSIM(yTrue,yPred)
    print(ssimVal)
    return alpha * (1-ssimVal) + (1-alpha) * mean_squared_error(yTrue, yPred)

dirArray = ["Pt-Mo5_input_images", "Pt_input_images", "Pt-Mo50_input_images"]
matl = dirArray[1] #specify desired material here
#Parses image data into ndarrays, then slices each array to be the minimum width and height of the group.
#Thus, formattedConvImages and formattedMultiImages will have arrays of all the same size.
convImages = []
multiImages = []
widths = []
heights = []
for d in range (0, 3):
  matl = dirArray[d]
  for i in range(0,getNumImages(matl)):
    convArr = np.loadtxt(getConvPath(matl, i))
    multiArr = io.imread(getMultislicePath(matl,i))
    #TODO: PLEASE DELETE THIS LINE AFTER DATA PROCESSING
    if (len(convArr[0]) <= 256 and len(convArr) <= 256):
      widths.append(len(convArr[0]))
      heights.append(len(convArr))
      convImages.append(convArr)
      multiImages.append(multiArr)

minWidth = min(widths)
minHeight = min(heights)
print(minWidth)
print(minHeight)

print(len(convImages))
print(len(multiImages))

print(np.min(convImages[0]))
print(np.max(convImages[0]))
print(np.min(multiImages[0]))
print(np.max(multiImages[0]))

#split data using sklearn
x =convImages
y =multiImages
X_train, X_test, Y_train, Y_test = train_test_split(x, y)

#format data into 256 by 256
newX_Train = formatImages(X_train,256,256)
newY_Train = formatImages(Y_train,256,256)
newX_Test = formatImages(X_test,256,256)
newY_Test = formatImages(Y_test,256,256)

formattedX_Train = np.ndarray(shape = (0,256*256))
for i in range(0,len(newX_Train)):
  tempX_Train = newX_Train[i].flatten()
  formattedX_Train = np.vstack([formattedX_Train, tempX_Train])

formattedY_Train = np.ndarray(shape = (0,256*256))
for i in range(0,len(newY_Train)):
  tempY_Train = newY_Train[i].flatten()
  formattedY_Train = np.vstack([formattedY_Train, tempY_Train])

formattedX_Test = np.ndarray(shape = (0,256*256))
for i in range(0,len(newX_Test)):
  tempX_Test = newX_Test[i].flatten()
  formattedX_Test = np.vstack([formattedX_Test, tempX_Test])

formattedY_Test = np.ndarray(shape = (0,256*256))
for i in range(0,len(newY_Test)):
  tempY_Test = newY_Test[i].flatten()
  formattedY_Test = np.vstack([formattedY_Test, tempY_Test])

print(formattedX_Train.shape)
print(formattedY_Train.shape)
print(formattedX_Test.shape)
print(formattedY_Test.shape)



model = Sequential()
model.add(Dense(4096, activation='relu', input_dim=65536, kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(1024, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(256, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(128, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(128, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(128, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(128, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dense(512, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dense(1024, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))
model.add(Dense(65536, activation='relu', kernel_initializer=initializers.he_normal(seed=None)))

alpha = 0.9
"""structural similarity measurement system."""
K1 = 0.01
K2 = 0.03
## L, number of pixels, C1, C2, two constants
L =  255
C1 = math.sqrt(K1 * L)
C2 = math.sqrt(K2 * L)

#sgd = optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=1e-5, beta_1=0.45, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(
    optimizer=adam,
    loss=[customLoss],
    metrics=['accuracy']
)
model.summary()

#Training starts here
# Fit the model
history =  model.fit(formattedX_Train, formattedY_Train, epochs=400, batch_size=3, verbose=2)


# evaluate the model
scores = model.evaluate(formattedX_Train, formattedY_Train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Lossadam.png')

#prediction
y_pred = model.predict(formattedX_Train)

for i in range (14):
  if (i%3 == 0):
    f, axarr = plt.subplots(4, 3)
  #Multislice Images
  image = Y_test[i]
  image = image.clip(min=0)
  axarr[0,i%3].imshow(image, cmap=plt.get_cmap('gray'))
  axarr[0,i%3].set_title("Multislice Image " + str(i))

  #Predicted Images
  image_y = y_pred[i, :].reshape([256, 256])
  image_y = image_y.clip(min=0)
  axarr[1,i%3].imshow(image_y, cmap=plt.get_cmap('gray'))
  axarr[1,i%3].set_title("Prediction Image " + str(i))

  #Convolutional Images
  image_x = formattedX_Test[i, :].reshape([256, 256])
  image_x = image_x.clip(min=0)
  axarr[2,i%3].imshow(image_x, cmap=plt.get_cmap('gray'))
  axarr[2,i%3].set_title("Convolution Image " + str(i))

  #Cut padding Image
  temp = cutPadding(image_y,len(image), len(image[0]))
  axarr[3,i%3].imshow(temp, cmap=plt.get_cmap('gray'))
  axarr[3,i%3].set_title("Prediction W/ Padding " + str(i))

  formatY = formattedY_Test[i, :].reshape([256, 256])

  print(str(i))
  print("The RMSE is "+ str(np.sqrt(np.mean(np.power((image_y-formatY),2)))*100) + " %")
  print("The RMSE w/o padding is "+ str(np.sqrt(np.mean(np.power((temp-image),2)))*100) + " %")
  print("The Max-Min RMSE is "+ str(np.sqrt(np.mean(np.power((image_y-formatY),2)))/(np.max(image) - np.min(image))*100) + " %")
  print("SSIM is :" + str(ssim(image, temp.astype(np.float32), data_range=temp.max() - temp.min())) + "\n")
  print("Standard Dev RMSE is " + str(np.sqrt(np.mean(np.power((image_y-formatY),2)))/(np.std(image_y)*100)) + " %")

  if (i%3 == 2 or i == 14):
    #plt.show()
    plt.savefig('Prediction with adam' + str(i) + '.png')
    plt.clf()


'''
# calculate predictions

  # print(str(i))
  # print("The RMSE is "+ str(np.sqrt(np.mean(np.power((image_y-formatY),2)))*100) + " %")
  # print("The RMSE w/o padding is "+ str(np.sqrt(np.mean(np.power((temp-image),2)))*100) + " %")
  # print("The Max-Min RMSE is "+ str(np.sqrt(np.mean(np.power((image_y-formatY),2)))/(np.max(image) - np.min(image))*100) + " %")
  # print("SSIM is :" + str(ssim(image, temp.astype(np.float32), data_range=temp.max() - temp.min())) + "\n")

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
'''
