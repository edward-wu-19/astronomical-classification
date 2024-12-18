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

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

def lr_scheduler(epoch, lr):
    if epoch % 8 == 0 and epoch != 0:
        lr = lr / 3.0
    return lr

checkpoint_callback = ModelCheckpoint(
    'model_checkpoint/model_checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.weights.h5', 
    save_weights_only=True, 
    save_best_only=True,
    verbose=0
)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=19)

import tensorflow as tf
from tensorflow.keras import layers, models

def build_googlenet(input_shape=(256, 256, 3), num_classes=10):
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    inception_3a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    inception_3a = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inception_3a)
    
    inception_3b = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    inception_3b = layers.Conv2D(256, (5, 5), activation='relu', padding='same')(inception_3b)
    
    inception_3c = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    inception_3c = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(inception_3c)

    inception_3 = layers.concatenate([inception_3a, inception_3b, inception_3c], axis=-1)

    inception_4a = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(inception_3)
    inception_4a = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inception_4a)

    inception_4b = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(inception_3)
    inception_4b = layers.Conv2D(256, (5, 5), activation='relu', padding='same')(inception_4b)

    inception_4c = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inception_3)
    inception_4c = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(inception_4c)

    inception_4 = layers.concatenate([inception_4a, inception_4b, inception_4c], axis=-1)

    x = layers.GlobalAveragePooling2D()(inception_4)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = build_googlenet(input_shape=(256, 256, 3), num_classes=10)
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=56,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback, early_stopping, lr_callback],
    verbose=2
)

model.evaluate(X_test, y_test, verbose=2)

model.save(
    "model_checkpoint/googlenet_best.keras"
)
