import numpy as np
import matplotlib.pyplot as pyplot

"""
(시간관계상)유튜브에서 다루지 않은 Numpy 클래스 객체와 메소드를 활용하여
작성하고 있습니다

"""

markov = np.array([[7,2,1],[3,3,4],[6,2,2]])/10
        # 이 행렬이 각 시장상태를 설명하는 확률행렬입니다

names = {0:'상승장', 1:'하락장', 2:'횡보장'}
        # 각 인덱스에 따른 시장 상태입니다


tmp = markov #1개월 째의 시장상태
print(tmp)

for i in range(119):            #곱할때마다
    tmp = np.dot(tmp, markov)   #개월이 하나씩 추가 (총 120개월)

prob = np.dot(markov, np.array([0,0,1]))
                                #횡보장이 세번째이므로

print(names[np.argmax(prob)]) #가장 큰 인덱스 출력
