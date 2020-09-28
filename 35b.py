import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.functions import *
import sys
import copy

n_train = 300
n_neurons = 150
print_freq = 50
data = np.sign(np.random.normal(0.0, 1, size=(n_train, n_neurons))) #biased
#data = np.sign(np.random.normal(0, 1, size=(n_train, n_neurons)))
#data = np.sign(np.random.randint(low=0, high=2, size=(n_neurons, n_train))) # wrong
data[data == 0] = -1
weight_matrix = np.zeros((data.shape[1], data.shape[1]))
res = []
weight_matrix = train_hebb3(data.T, weight_matrix, max_iter=1)
for i in range(n_train):
    training = copy.deepcopy(data[0:i+1, :])
    training = np.reshape(training, (i+1, n_neurons))
    #weight_matrix = train_hebb3(training.T, max_iter=1)
    result = np.array([hamming_distance(train, iterate_data(flip_bits(train, prop=0.01), weight_matrix, max_iter=1)) for train in training]) # count how many elements
    res.append(len(result) - np.count_nonzero(result))

plt.plot(res)
plt.title('Number of stable patterns vs trained patterns')
plt.show()

