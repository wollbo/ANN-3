import numpy as np
import pandas as pd
from core.functions import *
import sys


patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1], [-1, -1, -1, -1, -1, 1, -1, -1], [-1, 1, 1, -1, -1, 1, -1, 1]])
patterns = patterns.transpose()
weight_matrix = train_hebb(patterns)

test_patterns = np.array([[1, -1, 1, -1, 1, -1, -1, 1], [1, 1, -1, -1, -1, 1, -1, -1], [1, 1, 1, -1, 1, 1, -1, 1]])
test_patterns = test_patterns.transpose()

test_patterns_denoised = iterate_data(test_patterns, weight_matrix)
print(test_patterns_denoised.T)
print(test_patterns_denoised.T-iterate_data(test_patterns_denoised, weight_matrix).T)


for idx, pattern in enumerate(test_patterns.T):
    print(f'Noisy pattern {pattern.T}')
    print(f'True pattern {patterns.T[idx, :]}')
    print(f'Denoised pattern {test_patterns_denoised.T[idx,:]}')
    print(f'Hamming distance {hamming_distance(patterns.T[idx, :], test_patterns_denoised.T[idx,:])}')


# spurious patterns can be found as any arbitrary linear combination of the three attractor states, i.e.
# we can expect to find at most 9 other patterns


