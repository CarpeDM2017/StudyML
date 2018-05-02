# session02.py
import numpy as np
import matplotlib.pyplot as plt

# 1
TransposedTransitionMatrix = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.6, 0.2, 0.2]]) # bull bear stagnant x bull bear stagnant
TransitionMatrix = np.transpose(TransposedTransitionMatrix) # transposed the array above to make usual transition matrix
State1 = np.transpose(np.array([0,0,1]))

for i in range(120):
	State1 = np.dot(TransitionMatrix, State1) # multiply transition matrix 10*12 times
	
print(State1)

# 2

State2 = np.transpose(np.array([0,0,1]))
Answer2 = [State2[2]]
for i in range(12):
	State2 = np.dot(TransitionMatrix, State2)
	NotInStag = 1 - State2[2]
	Answer2.append(NotInStag)

print(Answer2)

# now plotting
t1 = np.arange(0, 13, 1)
t2 = np.arange(0, 1, 0.001)

fig = plt.subplot(111)
fig.plot(t1, np.transpose(Answer2))
plt.title('Not in Stagnation n months later')
plt.xlabel('Months passed')
plt.show()
