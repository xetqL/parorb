import matplotlib.pyplot as plt
import numpy as np
from os import listdir


files = [fname for fname in listdir('.') if '.particles' in fname]

for f in files:
    i = f.split('.')[0]
    p = np.loadtxt(f)
    plt.scatter(p[:,0],p[:,1], c='C'+i, s=0.001)

plt.axis('square')
plt.show()
