# astronomical-classification
Classifying Astronomical Bodies Project
DS-UA 301, Fall 2024
by Edward Wu, Tyler Zhang

In astronomy, one way to study dark matter is to observe the effects of its forces on visible objects. For example, we can map the positions of galaxies over time to calculate the forces exerted. We are interested in a classification model that can sort galaxies quickly and accurately, so that the galaxies can be mapped more efficiently. We are interested in how well the model can perform while being relatively light in terms of model size.

## Methodology

We selected four different architectures for a CNN classification model, and performed hyperparameter tuning for each. We wanted to explore the tradeoff between the sizes of various architectures and their accuracies. We tested ResNet50, GoogLeNet, LeNet5, and VGG16, and used random search to perform hyperparameter tuning on each of these to find the optimal values that yielded the highest accuracies. Due to the size of the dataset, we were forced to perform random search manually, by switching the values of our hyperparameters after each model was made. Then we compared the results with WaveMix and Astroformer, two of the top models in the field.

## Results

We created a graph of the best accuracy obtained for each architecture and the number of parameters in the model. We also calculated the space efficiency, which we defined as the accuracy divided by the number of parameters. We observe three main conclusions.
- WaveMix and ResNet50 were of about the same size. However, WaveMix had access to lots more data, which allowed it to increase their accuracy from our observed score of 0.8 to 0.95.
- GoogLeNet is the most space efficient of the models, which is likely attributed to the inception modules that it uses.
- While the images all looked relatively similar, even a simple model like LeNet5 was able to achieve an accuracy of 50%. This is surprising because when all the images are very similar, it is more difficult to pick out the differences between similar images.
