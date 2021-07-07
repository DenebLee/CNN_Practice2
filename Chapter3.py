# 훈련 노하루 배우기
# 검증 세트를 나누고 전처리 과정 배우기
# cancer 데이터세트를 읽어 들여 훈련 세트와 테스트 세트로 나눔
#%%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x , y, stratify=y, test_size=0.2, random_state=42)

# %%
# SGDClassifier 클래스를 이용하여 로지스틱 회귀 모델 훈련하기
# fi() aptjemdp x_train_all, y_train_all을 전달하여 모델을 훈련 
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log', random_state=42)
sgd.fit(x_train_all, y_train_all)
sgd.score(x_test, y_test)
# 테스트 세트에서 정확도는 약 83% 

# %%
# 위 성능이 만족스럽지 않다면 다른 손실 함수를 사용해도됨 
# loss와 같은 매개변수의 값은 가중치나 절편처럼 알아서 학습되는 것이 아님 
# 즉 사용자가 직접 선택해야되는데 이런 값을 하이퍼파라미터라고 부름
# SGDClassifier 클래스의 loss 매개변수를 hinge로 바꾸면 선형 서포트 벡터 머신 문제를 푸는 모델이 만들어짐
# 이렇게 해당 클래스의 다른 매개변수들을 바꿔보면 되는데 이 작업을 모델을 튜닝한다고함
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='hinge', random_state=42)
sgd.fit(x_train_all, y_train_all)
sgd.score(x_test, y_test)

# %%
# 테스트 테스로 모델을 튜닝하면 실전에서 좋은 성능을 기대하기 어려움
# 마치 학생에게 답안지를 외우게 하는것과 비슷함
# 검증 세트를 준비해서 학습시킨다 단 훈련세트가 너무 작아져도 곤란
# 위스콘신 유방암 데이터 준비
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
x = cancer.data
y= cancer.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# %%
# 검증세트 분할하기
x_train, x_val, y_train,y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)
print(len(x_train), len(x_val))

# %%
# 검증 세트 사용해 모델 평가하기
sgd =SGDClassifier(loss='log', random_state=42)
sgd.fit(x_train, y_train)
sgd.score(x_val, y_val)

# %%
# 데이터 전처리와 특성의 스케일
# 사이킷런과 같은 머신러닝 패키지에 준비되어있는 데이터는 대부분 실습을 위한것이므로 가공이 잘되어있음
# 실전에서 수집된 데이터들은 블규칙하고 정제가 안되어있다. 
# 이런 데이터들을 전처리 과정을 통해 훈련을 진행할 수있다.
# 훈련 데이터 준비하고 스케일 비교하기

import matplotlib.pyplot as plt
from Chapter2 import SingleLayer # Chapter2에서 만든 단일층 신경망 모델 가져오기
print(cancer.feature_names[[2,3,]])
plt.boxplot(x_train[ : , 2:4])
plt.xlabel('feature')
plt.ylabel('value')
plt.show()

# %%
# 가중치를 기록할 변수와 학습률 파라미터 추가하기
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np
class SingleLayer():
    
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.losses = []
        self.w_history = []
        self.lr = learning_rate # 학습률 파라미터
    
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
        self.w_history.append(self.w.copy())
        np.random.seed(42)
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

                self.w_history.append(self.w.copy())
                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적|
                a = np.clip(a, 1e-10, 1-1e-10) # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적

                loss += -(y[i]*np.log(a) + (1-y[i])* np.log(1-a)) # 에포크 마다 평균 손실을 저장함
            self.losses.append(loss/len(y))
    def predict(self, x):
        z = [self.forpass(x_i) for  x_i in x]
        return np.array(z) > 0

    def score(self , x ,y ):
        return np.mean(self.predict(x) == y )
# %%
# 모델 훈련하고 평가하기
layer1 = SingleLayer(  )
layer1.fit(x_train, y_train)
layer1.score(x_val, y_val)

# %%
# layer1 객체의 인스턴스 변수 w_history에는 100번의 에포크 동안 변경된 가중치가 모두 기록되어있음
w2 = []
w3 = []
for w in layer1.w_history:
    w2.append(w[2])
    w3.append(w[3])
plt.plot(w2, w3)
plt.plot(w2[-1], w3[-1], 'ro')
plt.xlabel('w[2]')
plt.ylabel('w[3]')
plt.show()
# 그래프를 보면 mean perimeter에 비해 mean area의 스케일이 크므로 w3값이 학습 과정에서 큰 폭으로 흔들리며 변화하고있음.
# 반면에 w3에 대한 그레이디언트가 크기떄문에 w3축을 따라 가중치가 크게 요동치고 있따라고 말함
#즉 가중치의 최적값에 도달하는 동안 w3값이 크게 요동치므로 모델이 불안정하게 수렴한다는것을 알수있음
# 이런 현상을 줄일수 있는 방법은 스케일을 조정하면됨

# %%
# 스케일을 조정해 모델을 훈련
# 스케일을 조정하는 방법은 많지만 신경망에서 자주 사용하는 스케일 조정 방법중 하나는 표준화

# 1. 넘파이로 표준화 구현하기
train_mean = np.mean(x_train, axis =0)
train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - train_mean) / train_std

# %%
# 모델 훈련
# 스케일을 조정한 데이터 세트로 단일층 신경망을 다시 훈련시키고 가중치를 그래프로 출력
layer2 = SingleLayer(  )
layer2.fit(x_train_scaled, y_train)
w2 = []
w3 = []
for w in layer2.w_history:
    w2.append(w[2])
    w3.append(w[3])
plt.plot(w2, w3)
plt.plot(w2[-1], w3[-1], 'ro')
plt.xlabel('w[2]')
plt.ylabel('w[3]')
plt.show()

# %%
# 모델 성능 평가
layer2.score(x_val, y_val)
# 검증 세트의 스케일을 바꾸지 않았기 때문에 성능이 좋지않은것

# %%
# 검증세트 표준화 처리
val_mean =np.mean(x_val, axis=0)
val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - val_mean) / val_std
layer2.score(x_val_scaled ,y_val)

# %%
# 스케일을 조정한 다음에 실수하기 쉬운 함정 알아보기
# 여기서 말하는 함정이란 훈련세트와 검증 세트가 다른 비율로 스케일이 조정된 경우를 말함

# 1. 원본 훈련 세트와 검증 세트로 산점도 그리기
plt.plot(x_train[:50, 0], x_train[:50, 1], 'bo')
plt.plot(x_val[:50, 0], x_val[:50, 1], 'ro')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend(['train_set', 'val. set'])
plt.show()
# 파란색이 훈련 세트 빨간 점이 검증세트

# %%
# 전처리한 훈련 세트와 검증 세트로 산점도 그리기
plt.plot(x_train_scaled[:50, 0], x_train_scaled[:50, 1], 'bo')
plt.plot(x_val_scaled[:50, 0], x_val_scaled[:50, 1], 'ro')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend(['train_set', 'val. set'])
plt.show()
# 앞서 그래프와 조금 미세하지만 훈련 세트와 검증 세트가 각각 다른 비율로 변환되었음을 알 수 있음
# 데이터를 제대로 전처리 했으면 (스케일을 조정했다면) 훈련 세트와 검증 세트의 거리가 그대로 유지되어야 함

# %% 
# 올바르게 검증 세트 전처리하기
x_val_scaled = (x_val - train_mean) / train_std
plt.plot(x_train_scaled[:50, 0], x_train_scaled[:50, 1], 'bo')
plt.plot(x_val_scaled[:50, 0], x_val_scaled[:50, 1], 'ro')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend(['train_set', 'val. set'])
plt.show()
# 원본 데이터의 산점도와 스케일 조정 이후의 산점도가 같아졌음 

# %%
# 모델평가
layer2.score(x_val_scaled, y_val)

# %%
# 과대 적합과 과소적합 
# 과소적합된 모델은 편향되었다라고 하고 과대적합된 모델은 분산이 크다라고 한다
# 과소적합된 모델과 과대적합된 모델 사이의 관게를 편향-분산 트레이드오프라고함
# 직역하면 하나를 얻기 위해서는 다른 하나를 희생해야하기 떄문이다
# 즉 편향- 분산 트레이드오프란 편향을 줄이면(훈련 세트의 성능을 높이면) 분산이 커지고(검증 세트와 성능 차이가 커지고) 반대로 분산을 줄이면 편향이 커지는 것을 말함

# 검증 손실을 기록하기 위한 변수추가하기
class SingleLayer():
    
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses = []
        self.w_history = []
        self.lr = learning_rate # 학습률 파라미터
    
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
    
    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        self.w = np.ones(x.shape[1])    # 가중치를 초기화
        self.b = 0                           # 절편을 초기화
        self.w_history.append(self.w.copy())
        np.random.seed(42)
        for i in range(epochs):           # epochs만큼 반복
            loss=0
            indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞는다
            for i in indexes: # 모든 샘플에 대해 반복
                z = self.forpass(x[i])  # 정방향 계산
                a = self.activation(z) # 활성화 함수 적용
                err = -(y[i] -a ) # 오차계산
                w_grad, b_grad = self.backprop(x[i], err) # 역방향 계산
                self.w -= self.lr * w_grad # 가중치 업데이트
                self.b -= b_grad # 절편 업데이트

                self.w_history.append(self.w.copy())
                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적|
                a = np.clip(a, 1e-10, 1-1e-10) # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적

                loss += -(y[i]*np.log(a) + (1-y[i])* np.log(1-a)) # 에포크 마다 평균 손실을 저장함
            self.losses.append(loss/len(y))
            self.update_val_loss(x_val, y_val)
    def predict(self, x):
        z = [self.forpass(x_i) for  x_i in x]
        return np.array(z) > 0

    def score(self , x ,y ):
        return np.mean(self.predict(x) == y )

    def update_val_loss(self, x_val, y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a, 1e-10, 1-1e-10)
            val_loss += -(y_val[i]*np.log(a)+ (1-y_val[i])*np.log(1-a))
        self.val_losses.append(val_loss/len(y_val))

# fit() 메서드에 검증 세트를 전달 받을 수 있도록 x_val, y_val 매개변수를 추가
# 검증 세트의 손실은 다음과 같이 update_val_loss() 메서드에 계산

# %%
# 모델 훈련하기
layer3 = SingleLayer( )
layer3.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val =y_val)

# %%
# 손실값으로 그래프 그려 에포크 횟수 지정하기
plt.ylim(0, 0.3)
plt.plot(layer3.losses)
plt.plot(layer3.val_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# %%
# 훈련 조기 종료하기
layer4 = SingleLayer()
layer4.fit(x_train_scaled, y_train, epochs=20)
layer4.score(x_val_scaled, y_val)

# %%
# 로지스틱 회귀에 규제를 적용
# 실무에서는 규제 효과가 뛰어난 L2규제를 주로 사용함
# 그레이디언트 업데이트 수식에 페널티 항 반영하기
class SingleLayer():
    
    def __init__(self, learning_rate=0.1, l1=0, l2=0):
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses = []
        self.w_history = []
        self.lr = learning_rate # 학습률 파라미터
        self.l1 = l1 # L1 규제
        self.l2 = l2 # L2 규제
    
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
    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        self.w = np.ones(x.shape[1])    # 가중치를 초기화
        self.b = 0                           # 절편을 초기화
        self.w_history.append(self.w.copy())
        np.random.seed(42)
        for i in range(epochs):           # epochs만큼 반복
            loss=0
            indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞는다
            for i in indexes: # 모든 샘플에 대해 반복
                z = self.forpass(x[i])  # 정방향 계산
                a = self.activation(z) # 활성화 함수 적용
                err = -(y[i] -a ) # 오차계산
                w_grad, b_grad = self.backprop(x[i], err) # 역방향 계산
                # 그레이디언트에서 페널티 항의 미분값을 더함
                w_grad += self.l1 * np.sign(self.w) + self.l2 * self.w
                self.w -=self.lr * w_grad # 가중치 업데이트
                self.b -= b_grad # 절편 업데이트

                self.w_history.append(self.w.copy())
                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적|
                a = np.clip(a, 1e-10, 1-1e-10) # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적

                loss += -(y[i]*np.log(a) + (1-y[i])* np.log(1-a)) # 에포크 마다 평균 손실을 저장함
            self.losses.append(loss/len(y) + self.reg_loss())

            self.update_val_loss(x_val, y_val)
    def reg_loss(self):
        return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w**2)

    def predict(self, x):
        z = [self.forpass(x_i) for  x_i in x]
        return np.array(z) > 0

    def score(self , x ,y ):
        return np.mean(self.predict(x) == y )

    def update_val_loss(self, x_val, y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a, 1e-10, 1-1e-10)
            val_loss += -(y_val[i]*np.log(a)+ (1-y_val[i])*np.log(1-a))
        self.val_losses.append(val_loss/len(y_val)+ self.reg_loss())

# %%
# cancer데이터 세트에 L1 규제 적용
l1_list = [0.0001, 0.001, 0.01]

for l1 in l1_list:
    lyr = SingleLayer(l1=l1)
    lyr.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val =y_val)

    plt.plot(lyr.losses)
    plt.plot(lyr.val_losses)
    plt.title('Learning Curve (l1={}'.format(l1))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.ylim(0, 0.3)
    plt.show()

    plt.plot(lyr.w, 'bo')
    plt.title('Weight (l1={})'.format(l1))
    plt.ylabel('value')
    plt.xlabel('weight')
    plt.ylim(-4, 4 )
    plt.show()

    # 학습 곡선 그래프를 보면 규제가 더 커질수록 훈련 세트의 순실과 검증 세트의 손실이 모두 높아짐 -> 과소적합 현상

# %%
layer5 = SingleLayer()
layer5.fit(x_train_scaled, y_train, epochs=20)
layer5.score(x_val_scaled, y_val)
# 이 데이터 세트는 작기 떄문에 규제 효과가 크게 나타나지않음

# %%
# cancer 데이터 세트에 L2 규제 적용
l2_list = [0.000, 0.001, 0.01]

for l2 in l2_list:
    lyr = SingleLayer(l2=l2)
    lyr.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val =y_val)

    plt.plot(lyr.losses)
    plt.plot(lyr.val_losses)
    plt.title('Learning Curve (l1={}'.format(l2))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.ylim(0, 0.3)
    plt.show()

    plt.plot(lyr.w, 'bo')
    plt.title('Weight (l1={})'.format(l2))
    plt.ylabel('value')
    plt.xlabel('weight')
    plt.ylim(-4, 4 )
    plt.show()
# 두 그래프를 보면 L2규제도 L1 규제와 비슷한 양상을 보임
# 하지만 마지막 학습 곡선 그래프를 보면 L2규제는 규제 강도가 강해져도 L1 규제만큼 과소적합이 심해지지는 않는다

# %%
# 훈련하고 성능 평가
layer6 = SingleLayer(l2=0.01)
layer6.fit(x_train_scaled, y_train, epochs=50)
layer6.score(x_val_scaled, y_val)
# L1 규제와 동일함
# 데이터 세트의 샘플 개수가 매우 적어서 L1이든 L2이든 성능에는 큰차이가없음

# %% 
# 예측한 샘플 갯수 
np.sum(layer6.predict(x_val_scaled) == y_val)

# %%
# SGDClassifier 규제 사용하기
sgd = SGDClassifier(loss='log', penalty='l2', alpha=0.001, random_state=42)
sgd.fit(x_train_scaled, y_train)
sgd.score(x_val_scaled, y_val)

# %%
# 교차 검증을 알아보고 사이킷런으로 수행해보기
# 교차 검증은 훈련 세트를 작은 덩어리로 나누어 다음과 진행하는데 이때 훈련 세트를 나눈 작은 덩어리를 폴드 라고부름
# k-폴드 교차 검증을 구현
# 1. 훈련 세트 사용하기
# 전체 데이터 세트를 다시 훈련 세트와 테스트 세트로 1번만 나눈 x_train_all과 y_train_all을 훈련과 검증에 사용
validation_scores = []

# %%
# k-폴드 교차 검증 구현
k =10 
bins =len(x_train) // k

for i in range(k):
    start = i*bins
    end = (i+1)*bins
    val_fold = x_train_all[start:end]
    val_target = y_train_all[start:end]

    train_index = list(range(0, start))+list(range(end, len(x_train)))
    train_fold = x_train_all[train_index]
    train_target = y_train_all[train_index]

    train_mean = np.mean(train_fold, axis=0)
    train_std = np.std(train_fold, axis=0)
    train_fold_scaled = (train_fold - train_mean) / train_std
    val_fold_scaled = (val_fold - train_mean) / train_std

    lyr = SingleLayer(l2=0.01)
    lyr.fit(train_fold_scaled, train_target, epochs=50)
    score = lyr.score(val_fold_scaled, val_target)
    validation_scores.append(score)

print(np.mean(validation_scores))

# %%
# 사이킷런으로 교차검증
# 1. cross_validate() 함수로 교차검증 계산
from sklearn.model_selection import cross_validate
sgd = SGDClassifier(loss='log', penalty='l2' , alpha=0.001, random_state=42)
scores = cross_validate(sgd, x_train_all, y_train_all, cv = 10)
print(np.mean(scores['test_score'])) 

# %%
# 2. Pipeline 클래스 사용해 교차 검증 수행하기
# 사이킷런은 검증 폴드가 전처리 단계에서 누설되지 않도록 전처리 단계와 모델 클래스를 하나로 연결해주는 Pipeline 클래스를 제공

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pipe = make_pipeline(StandardScaler(), sgd)
scores = cross_validate(pipe, x_train_all, y_train_all, cv=10, return_train_score=True)
print(np.mean(scores['test_score']))
# 평균 검증 점수가 높아짐
# 표준화 전처리 단계가 훈련 폴드와 검증 폴드에 올바르게 적용된 결과

# %%
# cross_validate()함수에 return_train_score에 매개변수를 설정하면 훈련 폴드의 점수도 얻을수있다. 
print(np.mean(scores['train_score']))
# %%
