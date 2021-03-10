# coding: utf-8
import sys, os
#   상위 디렉토리 경로 읽기
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

#   신경망의 추론 처리

#   flatten = True : 읽어 들인 이미지를 1차원 numpy 배열로 저장
#   normalize = True : 0~255 범위의 픽셀을 0.0 ~ 1.0 범위로 변환 (Pre-Processing)
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

#   pickle: 프로그램 실행 중 특정 객체를 파일로 저장하는 기능
#   저장해둔 pickle 파일을 로드하면 실행 당시의 객체를 즉시 복원 가능
#   이를 통해 이미 학습된 가중치 매개변수 정보가 담긴 sample_weight 를 읽음
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

#   각 레이블의 확률을 넘파이 배열로 반환
#   np.dot(a,b) = a x b
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


#   Step of Accuracy Evaluation
#   MNIST 데이터셋을 얻은 뒤 Network 생성
x, t = get_data()
network = init_network()

#   for 문을 돌며 x에 저장된 이미지를 꺼내 Predict()를 통해 분류
#   즉, x 이미지가 0~9까지의 수에 대한 확률을 numpy 배열로 저장
#   np.argmax()를 통해 확률이 가장 높은 값을 반환
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

#   최종적으로 신경망이 예측한 답변의 총 count를 전체 배열로 나누어 정확도 평가
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))