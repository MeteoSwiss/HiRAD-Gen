import os
import sys
import numpy as np

dir = sys.argv[1]
files = os.listdir(dir)
data = np.load(dir + files[0])
shape = data.shape
for i in range(len(files)):
    if i % 10000 == 0:
        print(i)
    try:
        data = np.load(dir + files[i])
    except:
        print(f'{files[i]} does not load') 
    if data.shape != shape:
        print(f'{files[i]} has shape {data.shape}')

    