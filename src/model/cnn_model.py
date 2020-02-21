import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.estimator import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pandas as pd
import glob
import pathlib
import cv2

# resghape data
# create model
# Parametres
FILTERS = 64
NUM_CLASSES = 3
DROPOUT_RATE = 0.3
EPOCHS = 15
PATH = "C:/Users/hakim/PycharmProjects/zindi_compet/data"
train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'test')

train_healthy_dir = os.path.join(train_dir, 'healthy_wheat')
train_leaf_dir = os.path.join(train_dir, 'leaf_rust')
train_stem_dir = os.path.join(train_dir, 'stem_rust')

val_healthy_dir = os.path.join(val_dir, 'healthy_wheat')
val_leaf_dir = os.path.join(val_dir, 'leaf_rust')
val_stem_dir = os.path.join(val_dir, 'stem_rust')

# Data size

# TRAIN
num_healthy_tr = len(os.listdir((train_healthy_dir)))
num_leaf_tr = len(os.listdir((train_leaf_dir)))
num_stem_tr = len(os.listdir((train_stem_dir)))

# VAL
num_healthy_val = len(os.listdir((val_healthy_dir)))
num_leaf_val = len(os.listdir((val_leaf_dir)))
num_stem_val = len(os.listdir((val_stem_dir)))

total_train = num_healthy_tr + num_leaf_tr + num_stem_tr
total_val = num_healthy_val + num_leaf_val + num_stem_val

print("total train set", total_train)
print("total validation set", total_val)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMH_WIDTH = 224

train_generator = ImageDataGenerator(rescale=1./255)
val_generator = ImageDataGenerator(rescale=1./255)

train_gen = train_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                directory=train_dir,
                                                classes=['healthy_wheat', 'leaf_rust', 'stem_rust'],
                                                shuffle=True,
                                                color_mode="rgb",
                                                target_size=(IMG_HEIGHT, IMH_WIDTH),
                                                class_mode='categorical')

val_gen = val_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                directory=val_dir,
                                                classes=['healthy_wheat', 'leaf_rust', 'stem_rust'],
                                                shuffle=True,
                                                color_mode="rgb",
                                                target_size=(IMG_HEIGHT, IMH_WIDTH),
                                                class_mode='categorical')

#print (val_gen.shape, "val_shape")


# visulaze training set

sample_training, _ = next(train_gen)

def plotImage(images):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        print(img.shape)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImage((sample_training[:4]))


CLASS_NAMES = ['healthy_wheat', 'leaf_rust', 'stem_rust']

def lr_optimizer(epoch):
    lr = 1e-3
    if (epoch == 30):
        lr *= 1e-1
    elif(epoch == 60):
        lr *= 1e-1
    else:
        lr *= 05e-1


def create_model():

    model = Sequential()

    model.add(Conv2D(FILTERS, kernel_size=(3, 3), padding='same', activation=relu, input_shape=(IMG_HEIGHT, IMH_WIDTH, 3)))
    model.add(MaxPool2D())
    #model.add(Dropout(DROPOUT_RATE))

    model.add(Conv2D(FILTERS, kernel_size=(3, 3), padding='same', activation=relu))
    model.add(MaxPool2D())
    # model.add(Dropout(DROPOUT_RATE))

    model.add(Conv2D(FILTERS, kernel_size=(3, 3), padding='same', activation=relu))
    model.add(MaxPool2D())
    # model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(512, activation=relu))
    # model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(NUM_CLASSES, activation=softmax))

    model.compile(optimizer=Adam(lr_optimizer(0)),
                  loss=sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

model = create_model()
model.summary()
#train_gen = np.reshape(train_gen, (-1, 32, 96))
#val_gen = np.reshape(val_gen, (-1, 32, 96))

STEP_SIZE_TRAIN = total_train // BATCH_SIZE
STEP_SIZE_VAL = total_val // BATCH_SIZE

history = model.fit_generator(train_gen,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              epochs= EPOCHS,
                              validation_data=val_gen,
                              validation_steps=STEP_SIZE_VAL)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# preprocessing Data

def import_data(path):

    data_path = os.path.join(path, '*g')
    files = glob.glob(data_path)
    image = []
    size = 0
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (32, 32))
        #print(img.shape)
        size +=1
        image.append(img)  # tout le dataset
        #plt.imshow(img)
        #plt.show()
    print(size)
    return image

'''x_val = import_data(VAL_PATH)
x_val = np.asarray(x_val)
v = x_val.shape
print(v)

x_train = import_data(TRAIN_PATH)
x_train = np.asarray(x_train)
x = x_train.shape
print(x)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

STEPS_PER_EPOCH = np.ceil(import_data(VAL_PATH)/BATCH_SIZE)
val_data_gen = img_gen.flow_from_directory(directory=str(data_dir),
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           target_size=(IMG_HEIGHT,IMH_WIDTH),
                                           classes=CLASS_NAMES)

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')

image_batch, label_batch = next(val_data_gen)
show_batch(image_batch, label_batch)'''