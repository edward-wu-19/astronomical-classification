# astronomical-classification
Classifying Astronomical Bodies Project
DS-UA 301, Fall 2024
by Edward Wu, Tyler Zhang

In astronomy, one way to study dark matter is to observe the effects of its forces on visible objects. For example, we can map the positions of galaxies over time to calculate the forces exerted. We are interested in a classification model that can sort galaxies quickly and accurately, so that the galaxies can be mapped more efficiently. We are interested in how well the model can perform while being relatively light in terms of model size.

We selected four different architectures for a CNN classification model, and performed hyperparameter tuning for each. We wanted to explore the tradeoff between the sizes of various architectures and their accuracies. We tested ResNet50, GoogLeNet, LeNet5, and VGG16, and used random search to perform hyperparameter tuning on each of these to find the optimal values that yielded the highest accuracies. Due to the size of the dataset, we were forced to perform random search manually, by switching the values of our hyperparameters after each model was made.
