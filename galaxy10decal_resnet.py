import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
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

dataset = tf.data.Dataset.from_tensor_slices((images, labels))

batch_size = 32
dataset = dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder = True)

from torch.utils.data import Dataset, DataLoader
import torch

class Galaxy10Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).float()
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

dataset = Galaxy10Dataset(images, labels)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=19)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size = 10000).batch(32, drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32, drop_remainder=True)

for batch_images, batch_labels in train_dataset.take(1):
    print("Train batch shape:", batch_images.shape)

for batch_images, batch_labels in test_dataset.take(1):
    print("Test batch shape:", batch_images.shape)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.summary()

def lr_scheduler(epoch, lr):
    if epoch % 4 == 0 and epoch != 0:
        lr = lr / 2.0
    return lr

initial_lr = 0.005
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    'model_checkpoint/model_checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.weights.h5', 
    save_weights_only=True, 
    save_best_only=True,
    verbose=0
)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback, early_stopping, lr_callback],
    verbose=2
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

model.save(
    "model_checkpoint/resnet50_best.keras"
)
