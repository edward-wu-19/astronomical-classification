import os
import numpy as np
import matplotlib.pyplot as plt

image_dir = "galaxy10_images"
label_file = "galaxy10_labels.txt"

images = []
true_labels = []
predicted_labels = []

with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split(", ")
        image_file = parts[0].split(": ")[1]
        true_label = parts[1].split(": ")[1]
        predicted_label = parts[2].split(": ")[1]

        image_path = os.path.join(image_dir, image_file)
        image = np.load(image_path)
        
        images.append(image)
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

images = np.array(images)

print("Images shape:", images.shape)
print("True labels:", true_labels[:5])
print("Predicted labels:", predicted_labels[:5])

def display_images_with_labels(images, true_labels, predicted_labels, num_images=10):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
  
display_images_with_labels(images, true_labels, predicted_labels, num_images=10)
