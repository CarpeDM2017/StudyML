import numpy as np
import matplotlib.pyplot as plt

data = [[0.7,0.2,0.1],[0.3,0.3,0.4],[0.6,0.2,0.2]]
np.array(data)
print(data)

mine=np.array(data)
print(mine)

mine2=mine

for i in range(119):
    mine2 = np.dot(mine2, mine)
    print(mine2)
