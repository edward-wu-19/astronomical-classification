# astronomical-classification
Classifying Astronomical Bodies Project

Our goal is to classify 10 different groups of galaxies from the galaxy10 DECaLS dataset. We used a CNN classification model, and performed hyperparametertuning and a neural architecture search for both. We wanted to explore the tradeoff between the sizes of various architectures and their accuracies. We tested four architectures, ResNet50, GoogLeNet, LeNet5, and VGG16, modified the architectures slightly to adapt to our dataset, and used random search to perform hyperparameter tuning on each of them to find the optimal values that yielded the highest accuracies.

