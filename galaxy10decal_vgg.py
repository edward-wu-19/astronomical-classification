import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

import tensorflow as tf
import warnings

tf.get_logger().setLevel('ERROR')

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the following GPU(s):")
    for gpu in gpus:
        print(f" - {gpu.name}")
else:
    print("No GPU detected. TensorFlow will use the CPU.")

import h5py
import numpy as np
from tensorflow.keras import utils

with h5py.File('./astro-data/Galaxy10_DECals.h5', 'r') as F: 
    images = np.array(F['images'])
    labels = np.array(F['ans'])

labels = utils.to_categorical(labels, 10)

labels = labels.astype(np.float32)
images = images.astype(np.float32)

images = images / 255.0

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=19)

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

def lr_scheduler(epoch, lr):
    if epoch % 4 == 0 and epoch != 0:
        lr = lr / 2.0
    return lr

checkpoint_callback = ModelCheckpoint(
    'model_checkpoint/model_checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.weights.h5', 
    save_weights_only=True, 
    save_best_only=True,
    verbose=0
)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)

early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

import tensorflow as tf
from tensorflow.keras import layers, models

def vgg16_modified(num_classes=10):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

model = vgg16_modified(num_classes=10)
model.summary()
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=24,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback, early_stopping, lr_callback],
    verbose=2
)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

model.save(
    "model_checkpoint/vgg16_best.keras"
)
