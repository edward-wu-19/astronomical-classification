import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import astroNN
from keras import utils
from sklearn.model_selection import train_test_split
from astroNN.models import Galaxy10CNN
from astroNN.datasets import load_galaxy10sdss
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup, galaxy10_confusion

images, labels = load_galaxy10sdss()

labels = utils.to_categorical(labels, 10)

img = None
plt.ion()

labels = labels.astype(np.float32)
images = images.astype(np.float32)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(0.25),
  tf.keras.layers.RandomFlip('horizontal')
])

def augment_image(image, label):
    image = data_augmentation(image)
    return image, label

newimages = tf.data.Dataset.from_tensor_slices((images, labels))

newimages = newimages.map(augment_image)

augmented_images = []
augmented_labels = []

for image, label in newimages:
    augmented_images.append(image.numpy())
    augmented_labels.append(label.numpy())

images = np.array(augmented_images)
labels = np.array(augmented_labels)

train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.2)
x_train, y_train, x_test, y_test = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

train_size = 10000
test_size = 500

x_train = x_train[:train_size]
x_test = x_test[:test_size]
y_train = y_train[:train_size]
y_test = y_test[:test_size]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

device = '/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'

with tf.device('/GPU:0'):

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(69, 69, 3))
    
    num_classes = 10
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_data=(x_test, y_test))

with tf.device('/GPU:0'):
    model = model

hist = model.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_data=(x_test, y_test))

trainloss = hist.history['loss']
testloss = hist.history['val_loss']
trainacc = hist.history['accuracy']
testacc = hist.history['val_accuracy']

max_loss = max(np.max(trainloss), np.max(testloss))
plt.plot(trainloss, color='steelblue', label='Training Loss')
plt.plot(testloss, color='red', label='Test Loss')
# plt.title('Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.ylim([0, max_loss + 0.05])
plt.xticks(np.linspace(0,20,11))
plt.legend()
plt.grid()
plt.show()
