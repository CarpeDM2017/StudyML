import numpy as np 
import matplotlib.pyplot as plt

#1번문제
data = [[0.7,0.2,0.1],[0.3,0.3,0.4],[0.6,0.2,0.2]]
print(data)
data_array = np.array(data)
print(data_array)
np.dot(data_array,data_array)
data_array2 = data_array
for i in range(119):
    data_array2=np.dot(data_array2,data_array)
    print(data_array2)


#2번문제
new_data = [[]]
