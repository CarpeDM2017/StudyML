# QUESTION01===================================
import numpy as np
#상황설정
market_conditions = ['Stagnant', 'Bull','Bear']
prob = np.array([[0.2,0.6,0.2],[0.1,0.7,0.2],[0.4,0.3,0.3]])
condition =np.array([1,0,0])
#10년 후 각  market condition의 확률
for i in range(120):
    condition = np.dot(condition, prob)
#가장 확률이 높은 market condition
answer_1 = np.argmax(condition)
print(answer_1) #1 = 'Bull'



# QUETION02====================================
import matplotlib.pyplot as plt
#Stagnant가 아닐 확률
select = np.array([1,0,0])
answer_2 = [np.sum(select)]
for i in range(12):
    select = np.dot(select, prob)
    select[0] = 0
    answer_2.append(np.sum(select))
#그래프 설정
fig = plt.figure()
plt.axis([0,12,0,1])
plt.plot(answer_2,'k>--')
plt.show()
