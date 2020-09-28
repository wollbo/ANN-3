import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.functions import *
import sys
import copy


n_train = 100
n_neurons = 20
print_freq = 50
data = np.sign(np.random.normal(0, 1, size=(n_train, n_neurons)))
#data = np.sign(np.random.randint(low=0, high=2, size=(n_neurons, n_train))) # wrong
data[data == 0] = -1

#uniq = np.unique(data, axis=0)
#print(uniq-data) WTF

weight_matrix = np.zeros((1, 1))
axes = []
fig = plt.figure()
res = []

wmax = train_hebb(data.T, max_iter=10)
pred = [hamming_distance(dat, iterate_data(flip_bits(dat), wmax, max_iter=100)) for dat in data]


for i in range(n_train):
    training = data[:, 0:i+1].copy().T
    weight_matrix = train_hebb(training, weight_matrix)
    print('asd')
    result = np.array([hamming_distance(train, iterate_data(train, weight_matrix, max_iter=10)) for train in training.copy().T]) # count how many elements
    res.append(len(result) - np.count_nonzero(result))

plt.plot(res)
plt.title('Number of stable patterns vs trained patterns')
plt.show()

