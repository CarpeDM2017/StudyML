"""
2018.04.04
My answers for problems in session02

"""
import numpy as np
import matplotlib.pyplot as plt


names = {0:'Bull', 1:'Stagnant', 2:'Bear'}
prob_matrix = np.array([[7, 6, 3],
                        [1, 2, 4],
                        [2, 2, 3]])/10

current_state = np.array([0, 1, 0])     # Stagnant


def Q1():
    next_state = np.eye(3)              # Identity Matrix
    for i in range(12*10):
        next_state = np.dot(next_state, prob_matrix)
    next_state = np.dot(next_state, current_state)
    return names[np.argmax(next_state)]


def Q2():
    x = np.arange(0,13)
    y = np.zeros(13)
    prob_vector = current_state
    for i in range(len(y)):
        if i==0:
            y[i] = 1
        else:
            prob_vector[1] = 0          # Remove possibility of Stagnant
            y[i] = prob_vector[0]+prob_vector[2]    # Residual probability
        prob_vector = np.dot(prob_matrix, prob_vector)
    plt.plot(x,y)
