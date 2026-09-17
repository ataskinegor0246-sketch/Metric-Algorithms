import numpy as np
from collections import Counter

a = np.array([[1,23,0],[1,2,0], [1,1,1]])
b =a[:,-1] 
counts = {}
for neighbor in b:
    counts[neighbor] = counts.get(neighbor, 0) + 1
print(counts) 

print(max(counts, key = counts.get))