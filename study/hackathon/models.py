# 아래와 같이 이용하고자 하는 모형을 여기에 정의해주세요.
# 분류 문제이므로 Classifier 클래스를 이용하시면 됩니다.

from sklearn.neural_network import MLPClassifier

class MyModel:
    def __init__(self):
        args = {'hidden_layer_sizes':(100,),
                'activation':'relu',
                'solver':'sgd',
                'warm_start':True,
                'verbose':True}
        self.model = MLPClassifier(**args)

    def train(self, images, labels):
        """이미지와 레이블을 주었을 때, 모형이 학습하는 방법을 정의해주세요.
        """
        self.model.fit(images, labels)

    def val(self, images, labels):
        """이미지와 레이블을 주었을 때, 그 정확도를 판단하는 방법을 정의해주세요.
           단 모형의 최종 정확도는 맞힌 개수 / 전체 개수만으로 평가합니다.
        """
        return self.model.score(images, labels)
