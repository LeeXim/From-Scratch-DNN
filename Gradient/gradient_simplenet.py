# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

#   정규분포로 초기화
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

#   softmax 함수를 사용
#   cross entropy 를 평가(손실함수)의 지표로 사용
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
#   정답 레이블 
t = np.array([0, 0, 1]) 

net = simpleNet()

#   net.lost() 를 새로운 함수로 정의하지 않고 lambda 를 써서 간단히 표기
#   numerical_gradient()를 통해 가중치 매개변수의 기울기를 구한다. 
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)