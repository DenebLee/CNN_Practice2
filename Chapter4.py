# 신경망 알고리즘을 벡터화하여 한 번에 전체 샘플을 사용
# 사이킷런의 예제 데이터 세트는 2차원 배열로 저장되어있따. 머신러닝에서 이렇게 훈련 데이터를 2차원 배열로 표현하는 경우가 많음
# 벡터화된 연산은 알고리즘의 성능을 올림
# 지금까지 사용한 경사 하강법 알고리즘 들은 알고리즘을 1번 반복할 떄 1개의 샘플을 사용하면 확률적 상품임

# 벡터 연산과 행렬 연산 -> 구박사한테
# SingleLayer클래스에 배치 경사 하강법 적용
# 위스콘신 유방암 데이터 가져오기
#%%
import numpy as np
import matplotlib.pyplot as plt
# 2. 훈련 검증 테스트 세트로 나누고 데이터 살펴보기
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

# %%
# cancer 데이터 세트의 특성 개수는 30개
# 훈련 세트와 검증 세트의 크기 확인
print(x_train.shape, x_val.shape)

# %%
# forpass(), backprop() 메서드에 배치 경사 하강법 적용
# scalar(스칼라) 는 하나의 실숫값을 의미 스칼라가 여러개 모이면 벡터가 만들어짐
def forpass(self, x):
    z = np.dot(x, self.w) + self.b # 선형 출력을 계산
    return z
def backprop(self, x, err):
    m = len(x)
    w_grad = np.dot(x.T, err) / m # 가중치에 대한 평균 그레이디언트를 계산
    b_grad = np.sum(err) / m # 절편에 대한 평균 그레이디언트를 계산
    return w_grad, b_grad 
# 파이썬의 len() 함수는 넘파이 배열의 행 크기를 반환하므로 이 값을 이용하여 그레이디언트의 평균을 계산
# 절편의 그레이디언트는 오차이므로 오차 행렬의 평균값을 계산

# %%
# SingleLayer 클래스에서 fit() 메서드 수정하기 및 나머지 메서드 수정

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
        z = np.dot(x , self.w) + self.b # 선형 출력을 계산
        return z 

    def backprop(self, x, err):
        m = len(x)
        w_grad =np.dot(x.T, err) / m  # 가중치에 대한 그레이디언트 계산
        b_grad = np.sum(err) / m   # 절편에 대한 그레이디언트 계산
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z , -100, None)  # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))      # 시그모이드 계산
        return a 
        
    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        y = y.reshape(-1,1) # 타깃을 열 벡터로 바꿈
        y_val = y_val.reshape(-1,1) # 검증용 타깃을 열 벡터로 바꿈
        m = len(x) # 샘플 개수를 저장
        self.w = np.ones((x.shape[1],1)) # 가중치를 초기화함
        self.b = 0 # 절편을 초기화함
        self.w_history.append(self.w.copy()) # 가중치를 기록
        for i in range(epochs):# 해당 epoch만큼 반복
            z = self.forpass(x) # 정방향 계산을 수행
            a = self.activation(z) # 활성화 함수를 적용
            err = -(y - a) # 오차를 계산
            w_grad, b_grad = self.backprop(x, err)
            # 오차를 역전파하여 그레이디언트 계산
            w_grad += (self.l1 * np.sign(self.w) + self.l2 * self.w) / m # 그레이디언트에서 페널티 항의 미분값 더하기
            self.w -= self.lr * w_grad # 가중치와 절편 업데이트
            self.b -= self.lr * b_grad
            self.w_history.append(self.w.copy()) # 가중치 기록
            a = np.clip(a , 1e-10 , 1-1e-10) # 로그 손실과 규제 손실을 더하여 리스트에 추가
            loss= np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / m)
            self.update_val_loss(x_val, y_val)
            
    def predict(self, x):
        z = self.forpass(x)
        return  z > 0

    def score(self , x ,y ):
        # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환
        return np.mean(self.predict(x) == y.reshape(-1, 1))

    def reg_loss(self):
        # 가중치에 규제를 적용
        return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w**2)

    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val) # 정방향 계산을 수행
        a = self.activation(z) # 활성화 함수를 적용
        a = np.clip(a, 1e-10, 1-1e-10) # 출력값 클리핑
        # 로그 손실과 규제 손실을 더하여 리스트에 추가
        val_loss = np.sum(-(y_val*np.log(a) + (1-y_val)*np.log(1-a)))
        self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))

# %%
# 훈련 데이터 표준화 전처리하기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)

# %%
# 이 데이터를 SingleLayer클래스 객체에 전달하여 배치 경사 하강법 적용
single_layer = SingleLayer(l2=0.01)
single_layer.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val, epochs=10000)
single_layer.score(x_val_scaled, y_val)

# %%
# 검증 세트로 성능 측정하고 그래프로 비교하기
plt.ylim(0, 0.3)
plt.plot(single_layer.losses)
plt.plot(single_layer.val_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# %%
# 가중치 변화를 그래프로 출력
w2 = []
w3 = []
for w in single_layer.w_history:
    w2.append(w[2])
    w3.append(w[3])
plt.plot(w2,w3)
plt.plot(w2[-1], w3[-1],'ro')
plt.xlabel('w[2')
plt.ylabel('w[3')
plt.show()
# 배치 경사 하강법을 적용하니 가중치를 찾는 경로가 다소 부드러운 곡선의 형태를 띠는것 같음
# 가중치의 변화가 연속적이므로 손실값도 안정적으로 수렴됨
# 배치 경사 하강법은 매번 전체 훈련 세트를 사용하므로 연산 비용이 많이들고 최솟값에 수렴하는 시간도 많이 걸림

# %%
# 2개의 층을 가진 신경망 구현하기
# 1. SingleLayer 클래스에 상속한 DualLayer 클래스 만들기
# 클래스 상속하기
class DualLayer(SingleLayer):
    def __init__(self, units=10, learning_rate=0.1, l1=0, l2=0):
        self.units =units   # 은닉층의 뉴런 개수
        self.w1 = None    # 은닉층의 가중치
        self.b1 = None     # 은닉층의 절편
        self.w2 = None    # 출력층의 가중치
        self.b2 = None     # 출력층의 절편
        self.a1 = None     # 은닉층의 활성화 출력
        self.losses = []      # 훈련 손실
        self.val_losses = []  # 검증 손실
        self.lr = learning_rate # 학습률
        self.l1 = l1            # L1 손실 하이퍼파라미터
        self.l2 = l2            # L2 손실 하이퍼파라미터
    
    def forpass(self,x ):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.activation(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        return z2

    def backprop(self, x, err):
        m = len(x) # 샘플 개수
        # 출력층의 가중치와 절편에 대한 그레이디언트 계산
        w2_grad = np.dot(self.a1.T, err) / m
        b2_grad = np.sum(err) / m
        # 시그모이드 함수까지 그레이디언트 계산
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
        # 은닉층의 가중치와 절편에 대한 그레이디언트 계산
        w1_grad = np.dot(x.T, err_to_hidden)
        b1_grad = np.sum(err_to_hidden, axis=0) / m
        return w1_grad, b1_grad, w2_grad, b2_grad

    def init_weights(self, n_features):
        self.w1 = np.ones((n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.ones((self.units, 1))
        self.b2 = 0

    def fit(self, x , y, epochs=100, x_val=None, y_val = None):
        y = y.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        m = len(x)
        self.init_weights(x.shape[1])
        for i in range(epochs):
            a = self.training(x, y ,m)
            a = np.clip(a, 1e-10, 1-1e-10)
            # 로그 손실과 규제 손실을 더하여 리스트 추가
            loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / m)
            self.update_val_loss(x_val, y_val)

    def training(self, x, y, m):
        z = self.forpass(x)
        a = self.activation(z)
        err = -(y-a)
        # 오차를 역전파하여 그레이디언트 계산            
        w1_grad, b1_grad,w2_grad, b2_grad = self.backprop(x, err)
        w1_grad += (self.l1 * np.sign(self.w1)+ self.l2 * self.w1) / m
        w2_grad += (self.l1 * np.sign(self.w2)+ self.l2 * self.w2) / m            
        self.w1 -= self.lr * w1_grad    
        self.b1 -= self.lr * b1_grad
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad
        return a
        
    def reg_loss(self):
        # 은닉층과 출력층의 가중치에 규제를 적용
        return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + self.l2 / 2 * (np.sum(self.w1**2) + np.sum(self.w2**2))
            
# %%
# 모델 훈련하기
dual_layer = DualLayer(l2=0.01)
dual_layer.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val, epochs=20000)
dual_layer.score(x_val_scaled, y_val)

# %%
# 훈련 손실과 검증 손실 그래프 분석
plt.ylim(0, 0.3)
plt.plot(dual_layer.losses)
plt.plot(dual_layer.val_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# %%
# 가중치 초기화 개선
# 손실 그래프를 보니 초기 손실값이 감소하는 곡선이 매끄럽지 않음
# 아마 손실 함수가 감소하는 방향을 올바르게 찾는 데 시간이 많이 소요된거 같음
# 지금까지 가중치를 1로 놓고 훈련을 했는데 이번에는 넘파이의 random.nomal() 함수를 사용하여 정규 분포를 따르는 무작위 수로 가중치를 초기화

# 1. 가중치 초기화를 위한 init_weights 메서드 수정
class RandomInitNetwork(DualLayer):
    def init_weights(self, n_features):
        np.random.seed(42)
        self.w1 = np.random.normal(0,1,(n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.random.normal(0, 1, (self.units, 1))
        self.b2 = 0

# %%
# 재훈련 및 그래프 출력
random_init_net = RandomInitNetwork(l2=0.01)
random_init_net.fit(x_train_scaled, y_train, x_val= x_val_scaled, y_val=y_val, epochs=500)
plt.ylim(0, 0.3)
plt.plot(random_init_net.losses)
plt.plot(random_init_net.val_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss, val_loss'])
plt.show()

# %%
# 딥러닝에서는 종종 아주 많은 양의 데이터를 사용하는데 배치 경사 하강법은 이런 경우에 사용하기 어려움
# 실무에선 확률적 경사 하강법과 배치 경사하강법의 장점을 절충한 미니배치경사 하강법이 널리 사용됨
# 미니배치 경사하강법 사용해보기
class MiniabatchNetwork(RandomInitNetwork):

    def __init__(self, units=10, batch_size=32, learning_rate=0.1, l1=0, l2=0):
        super().__init__(units, learning_rate, l1, l2)
        self.batch_size = batch_size
        # 파이썬에선 super() 함수로 부모클래스를 참조할수있음

    def fit(self, x,y, epochs=100, x_val=None, y_val= None):
        y_val = y_val.reshape(-1, 1)
        self.init_weights(x.shape[1])
        np.random.seed(42)
        # epoch만큼 반복
        for i in range(epochs):
            loss = 0
            # 제너레이터 함수에서 반환한 미니 배치를 순환
            for x_batch, y_batch in self.gen_batch(x , y):
                y_batch = y_batch.reshape(-1, 1)
                m = len(x_batch)
                a = self.training(x_batch, y_batch, m)
                # 안전한 로그 계산을 위해 클리핑
                a = np.clip(a, 1e-10, 1-1e-10)
                # 로그 손실과 규제 손실을 더하야 리스트에 추가
                loss += np.sum(-(y_batch*np.log(a) + (1-y_batch)*np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / len(x))
            # 검증 세트에 대한 손실을 계산
            self.update_val_loss(x_val, y_val)

    def gen_batch(self,x,y):
        length = len(x)
        bins = length // self.batch_size # 미니 배치 횟수
        if length % self.batch_size:
            bins += 1
        indexes = np.random.permutation(np.arange(len(x)))
        # 인덱스섞기
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start =self.batch_size * i
            end = self.batch_size * (i +1)
        yield x[start:end], y[start:end] # batch_size만큼 슬라이싱하여 반환

            
# %%
# 학습및 그래프 출력
minibatch_net = MiniabatchNetwork(l2=0.01, batch_size=32)
minibatch_net.fit(x_train_scaled, y_train, x_val=x_val_scaled,y_val=y_val, epochs=500)
minibatch_net.score(x_val_scaled, y_val)
#%%

plt.plot(minibatch_net.losses)
plt.plot(minibatch_net.val_losses)
plt.xlabel('loss')
plt.ylabel('iteration')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# %%
# 미니 배치 크기를 늘려 다시 시도
minibatch_net = MiniabatchNetwork(l2=0.01, batch_size=128)
minibatch_net.fit(x_train_scaled, y_train, x_val=x_val_scaled,y_val=y_val, epochs=500)
minibatch_net.score(x_val_scaled, y_val)
# %%
plt.plot(minibatch_net.losses)
plt.plot(minibatch_net.val_losses)
plt.xlabel('loss')
plt.ylabel('iteration')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# %%
# 사이킷런 사용해 다층 신경망 훈련하기
# 사이킷런에는 이미 신경망 알고리즘이 구현되어있는데 MLPClassifier작업을 통해 결과 비교
# MLPClassifier의 객체 만들기
from sklearn.neural_network import MLPClassifier
mip = MLPClassifier(hidden_layer_sizes=(10, ), activation='logistic', solver='sgd', alpha=0.01, batch_size=32, learning_rate_init=0.1, max_iter=500)

# 은닉층의 크기를 정의하는 hidden_layer_sizes

# %%
# 모델 훈련하기
# 모델 객체를 만든 후에 스케일이 조정된 훈련 세트인 x_train_scaled, y_train모델을 훈련
mip.fit(x_train_scaled, y_train)
mip.score(x_val_scaled, y_val)
# %%
