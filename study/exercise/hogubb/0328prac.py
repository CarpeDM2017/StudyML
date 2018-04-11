import numpy as np
import matplotlib.pyplot as plt

data=[[0.7, 0.2, 0.1],[0.3, 0.3, 0.4],[0.6, 0.2, 0.2]]
print(data)

arraydata=np.array(data)
print(arraydata)

np.dot(arraydata,arraydata)

for i in range(120):
    arraydata2=np.dot(arraydata2,arraydata)

print(arraydata2)
