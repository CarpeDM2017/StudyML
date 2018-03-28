import numpy as np
import matplotlib.pyplot as plt

prob = [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.6, 0.2, 0.2]]
prob_array = np.array(prob)
init_market_state = np.array([0,0,1])

def calculate(market_state, months):
    assert isinstance(months, int), "months should be an integer"
    assert months >= 0, "months should be positive"
    if months == 0:
        return market_state
    else:
        next_market_state = np.dot(market_state, prob_array)
        return calculate(next_market_state, months-1)

print(calculate(init_market_state,120))

def calculate_not_stagnate(market_state, months):
    result = np.zeros(months)
    next_market_state = market_state
    for i in range(0, months):
        next_market_state = np.dot(next_market_state, prob_array)
        result[i] = (1-next_market_state[2])
    return result

t1 = np.arange(13)

fig = plt.figure()
fig.suptitle("session02", size = 15, y = 1.0)

ax1 = plt.subplot(211)
ax1.set_title("problem 2")
ax1.plot(t1, calculate_not_stagnate(init_market_state, 13), 'bo')

plt.subplots_adjust(hspace=0.5)
plt.show()