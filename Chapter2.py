# 선형 회귀를 확장하여 분류 모델을 만들어보기
# 퍼셉트론은 선형 함수에 계단 함수만 추가한것
# 아달린은 퍼셉트론을 개선한 것으로 선형 함수의 결괏값으로 가중치를 학습
# 로지스틱 회귀는 아달린에 화성화 함수를 추가
#%%
# 로지스틱 함수를 그래프를 그려보면 로짓 함수의 가로와 세로축을 반대로 뒤집어 놓은 모양이 됨
import numpy as np
import matplotlib.pylab as plt
def softstep_func(x):
    return 1 / (1 + np.exp(-x))
x = np.arange(-10, 10, 0.01)
plt.plot(x, softstep_func(x), linestyle='-', label="softstep_func")
plt.ylim(0, 1)
plt.legend()
plt.show()
# 이 모양을 착안하여 로지스틱 함수를 시그모이드 함수라고도 부름
# 로지스틱 회귀는 가중치의 업데이트를 위해 로지스틱 손실 함수가 필요
# 로지스틱 손실 함수를 가중치에 대하여 미분할 때 연쇄 법칙 사용가능
# 로지스틱 손실 함수의 미분 결과는 제곱 오차 손실 함수의 미분 결과와 동일

# %%
# 분류용 데이터 세트 준비 
# 유방암 데이터 세트를 준비하는데 이 데이터 세트에는 유방암 세트의 특징 10개에 대하여 평균, 표준오차 최대 이상치가 기록되어있음
# 여기서 해결할 문제는 유방암 데이터샘플이 악성종양(True) 혹은 정상종양(False)인지를 구분하는 이진 분류 문제

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# %%
# 입력 데이터 확인
print(cancer.data.shape, cancer.target.shape)
# 569개의 샘플과 30개의 특성이 있다는 것을 알수있음

# %%
# 이중에 처음 개의 샘플 출력
cancer.data[:3]
# 실수 범위의 값이고 양수로 이루어짐
# 대괄호 1쌍으로 묶은 것이 샘플 총 30개

# %%
# 박스 플롯으로 특성의 사분위 관찰
import matplotlib.pyplot as plt
plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()
# 4, 14, 24번째 특성이 다른 특성보다 값의 분포가 훨씬 크다는 것을 알수있음

# %%
# 4, 14, 24번째 특성의 인덱스를 리스트로 묶어 전달하면 각 인덱스의 특성을 확인할 수 있음
cancer.feature_names[[3,13,23]]

# %%
# 타깃데이터 확인하기
# 넘파이의 unique() 함수를 사용하면 고유한 값을 찾아 반환
# 이때 return_counts 매개변수를 True로 지정하면 고유한 값이 등장하는 횟수까지 세어 반환
np.unique(cancer.target, return_counts=True)
# 두 덩어리의 값을 반환하고있는데 왼쪽값은 cancer.target에 들어있는 고유한 값(0,1)을 의미 

# %%
# 예제 데이터세트를 x,y 변수에 저장
x = cancer.data
y = cancer.target

# %%
# 로지스틱 회귀를 위한 뉴런 만들기


##############################################################################################################################
# 훈련 데이터 세트를 훈련 세트와 테스트 세트로 나누는 규칙
# 훈련 데이터 세트를 나눌 때는 테스트 세트보다 훈련 세트가 더 많아야됨
# 훈련 데이터 세트를 나누기 전에 양성, 음성 클래스가 훈련 세트나 테스트 세트의 어느한쪽에 몰리지 않도록 골고루 섞어야 한다
##############################################################################################################################

# train_test_split()함수로 훈련 데이터 세트 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train ,y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
# stratify = 훈련 데이터를 나눌때 클래스 비율을 동일하게 만듬 일부 클래스 비율이 불균형 한 경우에는 해당 함수를 y로 지정
# test_size = train_test_split() 함수는 기본적으로 훈련 데이터 세트를 75:25 사이즈로 나눔 그래서 따로 비율을 조절할때 
# random_state = 난수 초깃값 42을 지정했는데 실무에선 사용할  필요가없음


# %%
# 결과 확인하기
print(x_train.shape, x_test.shape)

# %%
# unique() 함수로 훈련 세트의 타깃 확인
np.unique(y_train, return_counts=True)

# %%
# 훈련 세트가 준비되었으니 로지스틱 회귀구현
# 로지스틱회귀는 정방향으로 데이터가 흘러가는 과정과 가중치를 업데이트하기 위해 역방향으로 데이터가 흘러가는 과정을 구현해야됨

class LogisticNeuron():

    def __init__(self):
        self.w = None
        self.b = None
    
    def forpass(self, x):
        z = np.sum(x * self.w) + self.b # 직선 방정식 계산
        return z 

    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그레이디언트 계산
        b_grad = 1 * err    # 절편에 대한 그레이디언트 계산
        return w_grad, b_grad
    def activation(self, z):
        z = np.clip(z , -100, None)  # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))      # 시그모이드 계산
        return a 
    
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])    # 가중치를 초기화
        self.b = 0                           # 절편을 초기화
        for i in range(epochs):           # epochs만큼 반복
            for x_i, y_i in zip(x, y ):       # 모든 샘플에 대해 반복
                z = self.forpass(x_i)       # 정방향 계산
                a = self.activation(z)      # 활성화 함수 적용
                err = -(y_i -a)              # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err)    # 역방향 계산
                self.w -= w_grad     # 가중치 업데이트
                self.b -= b_grad      # 절편 업데이트
    def predict(self, x):
        z = [self.forpass(x_i) for  x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
# chaper1와 다른점은 __init()__ 메서드는 가중치와 절편을 미리 초기화 하지않음
# 가중치는 나중에 입력 데이터를 보고 특성 개수에 맞게 결정
# forpass() 메서드에는 넘파이 함수를 사용

#%%
# 로지스틱 회귀 모델 훈련시키기
neuron = LogisticNeuron(  )
neuron.fit(x_train, y_train)

# %%
# 테스트 세트 사용해 모델의 정확도 평가
np.mean(neuron.predict(x_test) == y_test)

# %%
# 로지스틱 회귀 뉴런으로 단일층 신경망 만들기
# 앞에서 구현한 LogisticNeruron 클래스는 이미 단일층 신경망의 역활을 할 수 있으므로 학습을 위해 단일층 신경망을 또 구현할 필요는 없다. 
# 하지만 여기서 단일층 신경망을 다시 구현하는 이유는 몇가지 유용한 기능을 추가하기위해
# 추가기능을 위해 LogisticNeuron 클래스 복사후 이름을 SingleLayer로 변경

class SingleLayer():
    
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []
    
    def forpass(self, x):
        z = np.sum(x * self.w) + self.b # 직선 방정식 계산
        return z 

    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그레이디언트 계산
        b_grad = 1 * err    # 절편에 대한 그레이디언트 계산
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z , -100, None)  # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))      # 시그모이드 계산
        return a 
    
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])    # 가중치를 초기화
        self.b = 0                           # 절편을 초기화
        for i in range(epochs):           # epochs만큼 반복
            loss=0
            indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞는다
            for i in indexes: # 모든 샘플에 대해 반복
                z = self.forpass(x[i])  # 정방향 계산
                a = self.activation(z) # 활성화 함수 적용
                err = -(y[i] -a ) # 오차계산
                w_grad, b_grad = self.backprop(x[i], err) # 역방향 계산
                self.w -= w_grad # 가중치 업데이트
                self.b -= b_grad # 절편 업데이트
                a = np.clip(a, 1e-10, 1-1e-10) # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적

                loss += -(y[i]*np.log(a) + (1-y[i])* np.log(1-a)) # 에포크 마다 평균 손실을 저장함
            self.losses.append(loss/len(y))

    def predict(self, x):
        z = [self.forpass(x_i) for  x_i in x]
        return np.array(z) > 0

    def score(self , x ,y ):
        return np.mean(self.predict(x) == y )

# %%
# 단일층 신경망 훈련하고 정확도 출력하기
# SingleLayer 객체를 만들고 훈련세트로 신경망을 훈련한다음 score() 메서드로 정확도 출력
layer = SingleLayer(  )
layer.fit(x_train, y_train)
layer.score(x_test, y_test)
# 정확도가 매우 높게 나왔는데 이유는 에포크마다 훈련세트를 무작위로 섞어 손실 함수의 값을 줄였기 때문

# %%
# 손실 함수 누적값 확인하기
plt.plot(layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %%
# 사이킷런으로 로지스틱 회귀를 수행
# 1. 로지스틱 손실 함수 지정하기
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42)

#%%
# 2. 사이킷런으로 훈련하고 평가하기
sgd.fit(x_train, y_train)
sgd.score(x_test, y_test)

# %%
# 3. 사이킷런으로 예측하기
sgd.predict(x_test[0:10])

# %%
