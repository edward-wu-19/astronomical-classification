import h5py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers

with h5py.File('/home/tz2264/.astroNN/datasets/Galaxy10_DECals.h5', 'r') as F: 
    images = np.array(F['images'])
    labels = np.array(F['ans'])

images = images.astype(np.float32) / 255.0

images_resized = tf.image.resize(images, (256, 256))

class_names = [
    "Class 0", "Class 1", "Class 2", "Class 3", "Class 4",
    "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"
]

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.load_weights('resnet50_best_weights_NEW.weights.h5')

num_examples = 10
random_indices = np.random.choice(len(images), num_examples, replace=False)
example_images = images_resized.numpy()[random_indices]
true_labels = labels[random_indices]

predictions = model.predict(example_images)
predicted_classes = np.argmax(predictions, axis=1)

image_dir = "galaxy10_images"
label_file = "galaxy10_labels.txt"
os.makedirs(image_dir, exist_ok=True)

with open(label_file, "w") as f:
    for i, image in enumerate(example_images):
        image_path = os.path.join(image_dir, f"image_{i + 1}.npy")
        np.save(image_path, image)
        
        true_label = class_names[true_labels[i]]
        predicted_label = class_names[predicted_classes[i]]
        f.write(f"Image: image_{i + 1}.npy, True: {true_label}, Predicted: {predicted_label}\n")
        
        print(f"Saved: {image_path}")

print(f"Labels saved to: {label_file}")
