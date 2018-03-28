# mjh
# session02 exercise

import numpy as np
import matplotlib.pyplot as plt

transition_mat = np.array([[0.7, 0.2, 0.1],
                           [0.3, 0.3, 0.4],
                           [0.6, 0.2, 0.2]])
# transition_mat[i][j] = probability of transitioning from state i to state j
init_state = np.array([[0, 0, 1]])
state_dict = {0: 'Bull', 1: 'Bear', 2: 'Stagnant'}

print('1번 문제...\n')

state = init_state

for i in range(10 * 12):
    state = np.dot(state, transition_mat)

print(state)
print('10년 뒤 가장 확률이 높은 상태는 %s입니다.' % state_dict[state.argmax()])

print()
print('2번 문제...\n')

state = init_state

prob_list = [np.sum(state)]

for i in range(12):
    state = np.dot(state, transition_mat)
    state[0][2] = 0  # 이걸 0으로 바꾸지 않으면 '해당 월까지 다시 횡보장에 들어서지 않을 확률'이 아니라 '해당 월에 횡보장이 아닐 확률'을 구하게 됨
    prob_list.append(np.sum(state))

plt.plot(prob_list, 'bo')
plt.show()



