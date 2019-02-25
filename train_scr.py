
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import keras


train_data_dir      = '/home/techvamp/Documents/Project/retina/training_code/dataset/train'
validation_data_dir = '/home/techvamp/Documents/Project/retina/training_code/dataset/validation'
img_height          = 150
img_width           = 150
steps_per_epoch     = 22
epoch_tf            = 10
epoch_ft	        = 10
batch_size          = 10
#=========================================================================================
#create generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    rotation_range=180,
    zoom_range=[0.5, 1.5],
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='reflect')

test_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')




#=========================================================================================
#TRANSFER LEARNING

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

predictions = Dense(2, activation='softmax')(x)                     #number of classes

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers


#=========================================================================================
#FINETUNEING
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
optimiser = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epoch_ft, validation_data=validation_generator,validation_steps=20)

model.save('model/model_v1_finetuned_INCPT.h5')
