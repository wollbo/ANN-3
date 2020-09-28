import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.functions import *
import sys

plot = True
sequential = True
idx = 9   #9 can be completed!
print_freq = 50
data = np.reshape(pd.read_csv('data/pict.dat', sep=',', header=None).to_numpy(), (11, 1024))


if plot:
    axes = []
    fig = plt.figure()
    pix = np.reshape(data[0:3, :], (3, 32, 32))
    axes.append(fig.add_subplot(1, 3, 1))
    axes[-1].set_title(f'p1')
    plt.imshow(pix[0].T)
    plt.axis('off')
    axes.append(fig.add_subplot(1, 3, 2))
    axes[-1].set_title(f'p2')
    plt.imshow(pix[1].T)
    plt.axis('off')
    axes.append(fig.add_subplot(1, 3, 3))
    axes[-1].set_title(f'p3')
    plt.imshow(pix[2].T)
    plt.axis('off')

    #plt.imshow(pix[idx].T)
    plt.show()
    sys.exit(0)

training = data[0:3, :].T
weight_matrix = np.zeros((1, 1))
axes = []
fig = plt.figure()

if sequential:
    axes.append(fig.add_subplot(4, 5, 1))
    axes[-1].set_title(f'{0}')
    pix = np.reshape(data[idx, :], (32, 32))
    plt.imshow(pix.T)
    plt.axis('off')
    weight_matrix = train_hebb(training, weight_matrix=weight_matrix, max_iter=50)

    for n_iter in range(19):
        data[idx, :] = train_hebb_sequential(data[idx, :], weight_matrix=weight_matrix, max_iter=print_freq)
        intermediate = np.reshape(data[idx, :], (32, 32))
        axes.append(fig.add_subplot(4, 5, n_iter + 2))
        axes[-1].set_title(f'{(n_iter+1) * print_freq}')
        plt.imshow(intermediate.T)
        plt.axis('off')
    plt.show()
    sys.exit(0)
#pix = np.reshape(np.sign(weight_matrix @ data[9, :]), (32, 32))
#plt.imshow(pix.T)
#plt.imshow(np.reshape(data[9, :], (32, 32)).T)
#plt.show()
else:
    weight_matrix = train_hebb(training, weight_matrix=weight_matrix, max_iter=50)
    axes.append(fig.add_subplot(1, 2, 1))
    axes[-1].set_title(f'Original')
    plt.imshow(np.reshape(data[idx, :], (32, 32)).T)
    plt.axis('off')

    axes.append(fig.add_subplot(1, 2, 2))
    axes[-1].set_title(f'Reconstructed')
    pix = np.reshape(np.sign(weight_matrix @ data[idx, :]), (32, 32))
    plt.imshow(pix.T)
    plt.axis('off')

    plt.show()