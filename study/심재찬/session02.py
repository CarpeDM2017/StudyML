#1번문제

import numpy as np

TM = [[0.7, 0.6, 0.3],[0.1, 0.2 ,0.4],[0.2, 0.2, 0.3]]
TM = np.array(TM)
IM = [0, 1, 0]


# IM * TM => 2년 뒤
for i in range(10):
    IM = np.dot(TM, IM)

print(IM)



#2번문제
#1년 뒤까지 횡보장에 들어서지 않을 확률
# = 1 - 1년 뒤 횡보장에 들어설 확률

import numpy as np

TM = [[0.7, 0.6, 0.3],[0.1, 0.2 ,0.4],[0.2, 0.2, 0.3]]
TM = np.array(TM)
IM = [0, 1, 0]
IM = np.array(IM)

P = list()

for i in range(1):
    IM = np.dot(TM, IM)
    prob = IM[0] + IM[2]
    type(prob)
    P.append(prob)

print(P)
