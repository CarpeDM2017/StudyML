import numpy as np

"""
3차시 문제풀이를 위한 예시입니다.

"""

learning_rate = 0.1
momentum = 0.9
velocity = 0
epsilon = 1e-3   # 0.001

def Nesterov(F, x):
    # F는 함수, x는 넘파이 벡터라고 가정하겠습니다.
    # 상수들은 함수 안에서 정의해주어야 합니다 (global을 사용하지 않으면 인식하지 못합니다)
    # def 밖에 있는 변수와 안에 있는 변수는 사실 다른 변수가 됩니다.

    global learning_rate    # def 안에 있는 변수 : local 변수
    global momentum         # def 밖에 있는 (그 밖 무언가에도 들어있지 않은) 변수 : global 변수
    global velocity         # global이라고 지정해줌으로써 global 변수를 local로 데려올 수 있게 됩니다.
    global epsilon          # local 변수를 매 시행마다 저장해주기 위해서는 함수보다는 class를 활용하는 편이 더욱 편리합니다.
                            # ex) Nesterov 객체를 만들어두면 self.velocity는 해당 객체에 귀속되어 계속 따라다닙니다.


    dim = x.shape[0]            # x의 차원
    grad = np.zeros(x.shape)    # x의 미분계수를 저장할 공간
    eps = np.zeros(x.shape)     # x의 미분계수를 구하기 위해, x에 대한 작은 차분들을 저장할 공간

    for i in range(dim):        # x의 각 원소에 대해 반복합니다.
        eps[i] = epsilon        # i번째 원소를 아주 작은 값 epsilon으로 만듭니다.
        grad[i] = (F(x - momentum*velocity + eps) \
                 - F(x - momentum*velocity))/epsilon       # Nesterov 방식으로 미분합니다. momentum*velocity 해주셔야 해요
                                                           # 그냥 velocity를 해도 수렴합니다만 다른 방식의 알고리즘이 됩니다.
        eps[i] = 0              # 다시 원래대로 돌려줍니다.
    velocity = momentum*velocity + learning_rate*grad      # 가속도 업데이트
    x = x - velocity                                       # x값 업데이트
    return x

# 임의의 x값과 F에 대해 테스트
def F(x): return np.sum(x)
x = np.random.randn(100)

for i in range(100):
    x = Q1(F,x)
    print(x)
