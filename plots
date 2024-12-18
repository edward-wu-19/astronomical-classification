import numpy as np
import matplotlib.pyplot as plt

loss_google = 0.8748
acc_google = 0.7069
params_google = 2574410

loss_lenet = 1.4724
acc_lenet = 0.4645
params_lenet = 7394550

loss_resnet = 0.7525
acc_resnet = 0.7979
params_resnet = 24114826

losses = [loss_google, loss_lenet, loss_resnet]
accs = [acc_google, acc_lenet, acc_resnet]
params = [params_google, params_lenet, params_resnet]

labels = ["GoogleNet", "LeNet", "ResNet"]

plt.scatter(params, accs, s=20)
for i, txt in enumerate(labels):
    plt.text(params[i]+2e5, accs[i]-0.001, labels[i], fontsize=12, ha='left', va='top')
plt.xlabel("Number of Parameters")
plt.ylabel("Best Accuracy")
plt.title("Accuracies of Various Architectures")

losses = [loss_google, loss_lenet, loss_resnet]
accs = [acc_google, acc_lenet, acc_resnet]
params = [params_google, params_lenet, params_resnet]

labels = ["GoogleNet", "LeNet", "ResNet"]

plt.scatter(params, losses, s=20)
for i, txt in enumerate(labels):
    plt.text(params[i]+3e5, losses[i]-0.003, labels[i], fontsize=12, ha='left', va='top')
plt.xlabel("Number of Parameters")
plt.ylabel("Best Loss")
plt.title("Losses of Various Architectures")

acc_af = 0.9486
params_af = 272 * 1e6

acc_wave = 0.9542
params_wave = 28 * 1e6

accs = [acc_google, acc_lenet, acc_resnet, acc_wave, acc_af]
params = [params_google, params_lenet, params_resnet, params_wave, params_af]

labels = ["GoogleNet", "LeNet", "ResNet", "WaveMix", "Astroformer"]
colors = ['steelblue', 'steelblue', 'steelblue', 'red', 'red']

plt.scatter(params, accs, c=colors, s=20)
for i, txt in enumerate(labels):
    plt.text(params[i]+3e6, accs[i]-0.003, labels[i], fontsize=12, ha='left', va='top')
plt.xlabel("Number of Parameters")
plt.ylabel("Best Accuracy")
plt.title("Accuracies of Various Architectures")
