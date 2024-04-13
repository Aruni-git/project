from keras.models import Model
from keras.layers import Flatten,Dense
from keras.applications.vgg16 import VGG16 #Import all the necessary modules
import matplotlib.pyplot as plot
from glob import glob

#Provide image size as 224 x 224 this is a fixed-size for VGG16 architecture
#3 signifies that we are working with RGB type of images.
IMAGESHAPE = [224, 224, 3]

#import VGG16 model. weights of the imageNet & include_top=False signifies
# that we do not want to classify 1000 different categories present in imageNet as we have
# only two categories as Pneumonia and Normal
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

#Give our training and testing path
training_data = '../chest_xray/train'
testing_data = '../chest_xray/test'

#Set the trainable as False, So that all the layers would not be trained.
# The below code will not train the already trained VGG16 model so that we
# can use the pre-trained weights for classification. This is called transfer learning
# which is used to save a lot of effort and resources for re-training.
for each_layer in vgg_model.layers:
	each_layer.trainable = False

# Finding how many classes present in the train dataset.
classes = glob('../chest_xray/train/*')

flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)

#Combine the VGG output and prediction , this all together will create a model.
final_model = Model(inputs=vgg_model.input, outputs=prediction)

#Displaying the summary
final_model.summary()

#Compiling our model using adam optimizer and optimization metric as accuracy.
final_model.compile(
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy']
)

#importing our dataset to keras using ImageDataGenerator in keras.
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
								shear_range = 0.2,
								zoom_range = 0.2,
								horizontal_flip = True)
testing_datagen = ImageDataGenerator(rescale =1. / 255)

#inserting the images
training_set = train_datagen.flow_from_diretory('../chest_xray/train',
												target_size = (224, 224),
												batch_size = 4,
												class_mode = 'categorical')
test_set = testing_datagen.flow_from_directory('../chest_xray/test',
											target_size = (224, 224),
											batch_size = 4,
											class_mode = 'categorical')
#Fitting the model.
fitted_model = final_model.fit_generator(
training_set,
validation_data=test_set,
epochs=20,
steps_per_epoch=len(training_set),
validation_steps=len(test_set)
)
# plot.plot(fitted_model.history['loss'], label='training loss') #Plotting the accuracies
# plot.plot(fitted_model.history['val_loss'], label='validation loss')
# plot.legend()
# plot.show()
# plot.savefig('LossVal_loss')
# plot.plot(fitted_model.history['acc'], label='training accuracy')
# plot.plot(fitted_model.history['val_acc'], label='validation accuracy')
# plot.legend()
# plot.show()
# plot.savefig('AccVal_acc')

#Saving the model file.
final_model.save('pneumonia_model.h5')
