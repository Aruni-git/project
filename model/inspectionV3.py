# Base packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil

# More packages
import cv2
import matplotlib.pyplot as plt

# Keras
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# Reading through the metadata
summary = pd.read_csv('../chest_xray/train/*.csv')
df = pd.read_csv('../chest_xray/train/*.csv')

replace_dict = {'Pnemonia':1,
                'Normal':0}
df['Label'] = df['Label'].replace(replace_dict)

train_df = df[df.Dataset_type=='TRAIN']
test_df = df[df.Dataset_type=='TEST']

# Defining the path to Train and Test directories
training_data_path = '../chest_xray/train'
testing_data_path = '../chest_xray/test'

# Funtions for Making nd Removing subdirectories
def create_dir():
    try:
        os.makedirs('/working/train/Pneumonia')
        os.makedirs('/working/train/Normal')
        os.makedirs('/working/val/Pneumonia')
        os.makedirs('/working/val/Normal')
        os.makedirs('/working/test/Pneumonia')
        os.makedirs('/working/test/Normal')
    except:
        pass
def remove_dir():
    try:
        shutil.rmtree('/working/train')
        shutil.rmtree('/working/test')
    except:
        pass

# Seperate dataframes for different labels in test and train
train_pneumonia_df = train_df[train_df.Label==1]
train_normal_df = train_df[train_df.Label==0]
test_pneumonia_df = test_df[test_df.Label==1]
test_normal_df = test_df[test_df.Label==0]

ntrain_p = len(train_pneumonia_df)
ntrain_n = len(train_normal_df)
tntrain = ntrain_p+ntrain_n

#Take 10% from train to be validation

nval_p = round(0.1*ntrain_p)
nval_n = round(0.1*ntrain_n)

print(nval_p)
print(nval_n)

val_pneumonia_df = train_pneumonia_df[0:nval_p]
train_pneumonia_df = train_pneumonia_df[nval_p:]

val_normal_df = train_normal_df[0:nval_n]
train_normal_df = train_normal_df[nval_n:]

# Copying the files to newly created locations. You may use Flow from dataframe attribute and skip all these steps. But I prefer to use flow from directory
remove_dir()
create_dir()

training_images_pneumonia = train_pneumonia_df.X_ray_image_name.values.tolist()
training_images_normal = train_normal_df.X_ray_image_name.values.tolist()

val_images_pneumonia = val_pneumonia_df.X_ray_image_name.values.tolist()
val_images_normal = val_normal_df.X_ray_image_name.values.tolist()

testing_images_pneumonia = test_pneumonia_df.X_ray_image_name.values.tolist()
testing_images_normal = test_normal_df.X_ray_image_name.values.tolist()

for image in training_images_pneumonia:
    train_image_pneumonia = os.path.join(training_data_path, str(image))
    shutil.copy(train_image_pneumonia, '/working/train/Pneumonia')

for image in training_images_normal:
    train_image_normal = os.path.join(training_data_path, str(image))
    shutil.copy(train_image_normal, '/working/train/Normal')

for image in val_images_pneumonia:
    val_image_pneumonia = os.path.join(training_data_path, str(image))
    shutil.copy(val_image_pneumonia, '/working/val/Pneumonia')

for image in val_images_normal:
    val_image_normal = os.path.join(training_data_path, str(image))
    shutil.copy(val_image_normal, '/working/val/Normal')

for image in testing_images_pneumonia:
    test_image_pneumonia = os.path.join(testing_data_path, str(image))
    shutil.copy(test_image_pneumonia, '/working/test/Pneumonia')

for image in testing_images_normal:
    test_image_normal = os.path.join(testing_data_path, str(image))
    shutil.copy(test_image_normal, '/working/test/Normal')

batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   rotation_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory('/working/train',
                                                    target_size=(224,224),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory('/working/val',
                                                    target_size=(224,224),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

#Creating the model
base_model = InceptionV3(include_top=False, weights='imagenet', pooling='max', input_shape=(224,224,3))

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = Dropout(0.3)(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.3)(x)
x = Dense(512, activation = "relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation = "sigmoid")(x)

model = keras.Model(inputs,outputs)

base_model.trainable = False
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(train_generator,
          steps_per_epoch=train_generator.samples//batch_size,
          epochs = 10,
          validation_data=valid_generator,
          validation_steps=valid_generator.samples//batch_size)


base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss='binary_crossentropy',metrics=['accuracy'])

temp = valid_generator.samples//batch_size

model.fit(train_generator,
          steps_per_epoch=train_generator.samples//batch_size,
          epochs = 5,
          validation_data=valid_generator,
          validation_steps=temp)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('/working/test',
                                                    target_size=(224,224),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

model.evaluate(test_generator)
