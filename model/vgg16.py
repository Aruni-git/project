import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
#Using TensorFlow backend.
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#Provide image size
IMAGE_SIZE = [224, 224]

#Give our training and testing path
training_data = '../chest_xray/train'
testing_data = '../chest_xray/test'

#import vgg16 model, 3 means working with RGB type of images. only two categories as Pneumonia and Normal
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Set the trainable as False, So that all the layers would not be trained.
# use the pre-trained weights for classification.[transfer learning]
for layer in vgg.layers:
    layer.trainable = False

# Finding how many classes present in the train dataset.
folders = glob('../chest_xray/train/*')
x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

#Compiling our model using adam optimizer and optimization metric as accuracy.
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)




# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('../chest_xray/train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')




test_set = test_datagen.flow_from_directory('../chest_xray/test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')

#Fitting the model.
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import tensorflow as tf
from keras.models import load_model

model.save('vgg16.h5')


