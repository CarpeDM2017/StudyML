import numpy as np
import matplotlib.pyplot as plt


# ------------ Q1 ------------ #
ANSWER_TEMPLATE = {
	0: "Bull",
	1: "Bear",
	2: "Stagnant"
}

P = np.array([
	[0.7,0.2,0.1],
	[0.3,0.3,0.4],
	[0.6,0.2,0.2]
	])

x = np.array([0,0,1])

answer = np.argmax(x.dot(np.linalg.matrix_power(P, 10 * 12)))
print(ANSWER_TEMPLATE[answer]) # Bull

# ------------ Q2 ------------ #
y = [1]
for i in range(12):
	x = x.dot(P)
	# explicitly set prob for stag to 0.
	x[2] = 0
	y.append(np.sum(x))

plt.plot(y)
plt.show()








