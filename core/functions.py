import numpy as np
import matplotlib.pyplot as plt
import copy


def train_hebb(data, weight_matrix=np.zeros((1, 1)), residual=1, max_iter=1):
    data_dim = data.shape[0]
    n_samples = data.shape[1]
    i = 0
    weight_matrix = np.zeros((data_dim, data_dim)) if not weight_matrix.any() else weight_matrix
    while residual and i < max_iter:
        for col in data.T:
            weight_matrix = weight_matrix + np.outer(col.T, col.T)
        np.fill_diagonal(weight_matrix, 0)
        weight_matrix = weight_matrix / n_samples
        residual = sum([np.sum(pattern-np.sign(weight_matrix @ pattern)) for pattern in data.T])
        i += 1
    return weight_matrix


def train_hebb3(data, weight_matrix=np.zeros((1, 1)), residual=1, max_iter=1):
    data_dim = data.shape[0]
    n_samples = copy.deepcopy(data.shape[1])
    i = 0
    weight_matrix = np.zeros((data_dim, data_dim)) if not weight_matrix.any() else weight_matrix
    while residual and i < max_iter:
        for col in data.T:
            weight_matrix = weight_matrix + np.outer(col.T, col.T)
        #np.fill_diagonal(weight_matrix, 0)
        weight_matrix = weight_matrix / n_samples # np.sum(weight_matrix+1, axis=1)
        residual = sum([np.sum(pattern-np.sign(weight_matrix @ pattern)) for pattern in data.T])
        i += 1
    return weight_matrix


def train_hebb_zero(data, weight_matrix=np.zeros((1, 1)), residual=1, max_iter=1):
    data_dim = data.shape[0]
    n_samples = copy.deepcopy(data.shape[1])
    i = 0
    weight_matrix = np.zeros((data_dim, data_dim)) if not weight_matrix.any() else weight_matrix
    avg_act = np.mean(data)
    while residual and i < max_iter:
        for col in data.T:
            weight_matrix = weight_matrix + np.outer(col.T-avg_act, col.T-avg_act)
        np.fill_diagonal(weight_matrix, 0)
        weight_matrix = weight_matrix / n_samples # np.sum(weight_matrix+1, axis=1)
        residual = sum([np.sum(pattern-np.sign(weight_matrix @ pattern)) for pattern in data.T])
        i += 1
    return weight_matrix


def train_hebb2(data, weight_matrix=np.zeros((1, 1))):
    data_dim = data.shape[0]
    n_samples = data.shape[1]
    weight_matrix = np.zeros((data_dim, data_dim)) if not weight_matrix.any() else weight_matrix
    for col in data:
        weight_matrix = weight_matrix + np.outer(col.T, col.T)
        np.fill_diagonal(weight_matrix, 0)
    weight_matrix = weight_matrix / n_samples
    return weight_matrix


def train_hebb_sequential(data, weight_matrix=np.zeros((1, 1)), residual=1, max_iter=1000): # Fundamentally flawed. Update happens after matrix is trained
    i = 0
    while i < max_iter:
        order = np.random.permutation(data.size)
        np.random.shuffle(order)
        for idx in order:
            data[idx] = np.sign(weight_matrix @ data)[idx]
            i += 1
    return data


def iterate_data(data, weight_matrix, max_iter=1):
    i = 0
    difference = 1
    while i < max_iter and difference:
        previous = copy.deepcopy(data)
        data = np.sign(weight_matrix @ data)
        difference = hamming_distance(data, previous)
        i += 1
    return np.sign(weight_matrix @ data).astype(int)


def iterate_data_zero(data, weight_matrix, theta, max_iter=1):
    i = 0
    difference = 1
    while i < max_iter and difference:
        previous = copy.deepcopy(data)
        data = np.round(0.5+0.5*np.sign((weight_matrix @ data)-theta))
        difference = hamming_distance(data, previous)
        i += 1
    return np.round(0.5+0.5*np.sign((weight_matrix @ data)-theta)).astype(int)


def update_sequential(data, weight_matrix, n_iterations=2):
    energy = []
    for n in range(n_iterations):
        order = np.random.permutation(data.size)
        for idx in order:
            data[idx] = np.sign(weight_matrix @ data)[idx]
            energy.append(simple_energy(data, weight_matrix))
    return data, energy


def find_spurious(data): # not really working
    n_dim = data.shape[0]
    n_patterns = data.shape[1]
    signs = np.stack(np.meshgrid([-1, 1], [-1, 1], [-1, 1]), -1).reshape(-1, n_patterns)
    mvec = np.zeros((n_dim, signs.shape[0]))
    for idx, sig in enumerate(signs):
        rsum = np.zeros(n_dim).T
        for i in range(n_patterns):
            rsum = np.add(rsum, sig[i] * data[:, i])
        mvec[idx, :] = np.sign(rsum+0.000001)
    return np.vstack((mvec, -mvec)).astype(int)


def hamming_distance(a, b):
    return len(np.nonzero(a != b)[0])


def simple_energy(data, weight_matrix):
    return - data.T @ weight_matrix @ data


def flip_bits(data, prop=0.1):
    return data - 2*data*np.random.binomial(1, prop, data.shape)
