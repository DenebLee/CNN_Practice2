# 여러개의 이미지를 분류하는 다층 신경망 생성
# 다중 분류 신경망을 만들기 위해서는 소프트맥스함수와 크로스 엔트로피 손실 함수라는 새로운 함수를 알아야함

# 자세한 설명은 구박사에게 
# 다중 클래스 분류를 위 MultiClassNetwork 클래스 수정및 보완

#%%
import numpy as np

class MultiClassNetwork:

    def __init__(self, units=10, batch_size=32, learning_rate=0.1, l1=0, l2=0):
        self.units = units # 은닉층의 뉴런 개수
        self.batch_size = batch_size  # 배치크기
        self.w1 = None # 은닉층의 가중치
        self.w2 = None # 출력층의 가중치
        self.b1 = None # 은닉층의 절편
        self.b2 = None # 출력층의 절편
        self.a1 = None # 은닉층의 활성화 출력
        self.losses = [] # 훈련 손실
        self.val_losses = [] # 검증 손실
        self.lr = learning_rate # 학습률
        self.l1 = l1 # L1 손실 하이퍼파라미터
        self.l2 = l2 # L2 손실 하이퍼파라미터

    def forpass(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        return z2
    
    def backprop(self, x, err):
        m = len(x) # 샘플 개수
        # 출력층의 가중치와 절편에 대한 그레이디언트를 계산
        w2_grad = np.dot(self.a1.T, err) / m
        b2_grad = np.sum(err) / m
        # 시그모이드 함수까지 그레이디언트 계산
        err_to_hidden = np.dot(err, self.w2.T) * self.a1 * (1 - self.a1)
        # 은닉층의 가중치와 절편에 대한 그레이디언트를 계산
        w1_grad = np.dot(x.T, err_to_hidden) / m 
        b1_grad = np.sum(err_to_hidden, axis=0) /m 
        return w1_grad, b1_grad, w2_grad, b2_grad

    def sigmoid(self, z):
        z = np.clip(z, -100, None)
        a = 1 / (1 + np.exp(-z))
        return a 
    
    def softmax(self, z):
        # 소프트맥스함수
        z = np.clip(z, -100, None)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)


    def init_weights(self, n_features, n_classes):
        self.w1 = np.random.normal(0, 1, (n_features, self.units)) 
        self.b1 = np.zeros(self.units)
        self.w2 = np.random.normal(0, 1, (self.units, n_classes))
        self.b2 = np.zeros(n_classes)

    def fit(self, x, y ,epochs=100, x_val=None, y_val=None):
        np.random.seed(42)
        self.init_weights(x.shape[1], y.shape[1]) # 은닉층과 출력층의 가중치를 초기화
        for i in range(epochs):
            loss = 0
            print('.', end='')
            # 제너레이터 함수에서 반환한 미니 배치를 순환
            for x_batch, y_batch in self.gen_batch(x,y):
                a = self.training(x_batch, y_batch)
                # 안전한 로그 계산을 위해 클리핑
                a = np.clip(a, 1e-10, 1-1e-10)
                # 로그 손실과 규제 손실을 더하여 리스트에 추가
                loss += np.sum(-y_batch*np.log(a))
                self.losses.append((loss + self.reg_loss()) / len(x))
                # 검증 세트에 대한 손실을 계산
                self.update_val_loss(x_val, y_val)

    # 미니 배치 제너레이터 함수
    def gen_batch(self, x, y):
        length = len(x)
        bins = length // self.batch_size # 미니 배치 횟수
        if length % self.batch_size:
            bins += 1 # 나누어 떨어지지 않을때
        indexes = np.random.permutation(np.arange(len(x)))
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start =self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end] # batch_size만큼 슬라이싱하여 반환

    def training(self, x, y):
        m = len(x) # 샘플 개수 저장
        z = self.forpass(x) # 정방향 계산을 구행
        a = self.softmax(z) # 활성화 함수 적용
        err = -(y -a) # 오차를 계산
        # 오차를 역전파하여 그레이디언트 계산
        w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x, err)
        # 그레이디언트에서 페널티 항의 미분값을 뺀다
        w1_grad += (self.l1 * np.sign(self.w1) + self.l2 * self.w1) /m
        w2_grad += (self.l1 * np.sign(self.w2) + self.l2 * self.w2) /m
        # 은닉층의 가중치와 절편을 업데이트
        self.w1 -= self.lr * w1_grad
        self.b1 -= self.lr * b1_grad
        # 출력층의 가중치와 절편을 업데이트
        self.w2 -= self.lr * w2_grad
        self.b2 -= self.lr * b2_grad
        return a

    def predict(self, x):
        z = self.forpass(x) # 정방향 계산 수행
        return np.argmax(z, axis=1) # 가장 큰 값의 인덱스 반환

    def score(self, x, y ):
        # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환
        return np.mean(self.predict(x) == np.argmax(y, axis =1))

    def reg_loss(self):
        # 은닉층과 출력층의 가중치에 규제를 적용
        return self.l1 * (np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + self.l2 /2 * (np.sum(self.w1 **2) + np.sum(self.w2**2))

    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)
        a = self.softmax()
        a = np.clip( a, 1e-10, 1-1e-10)
        # 크로스 엔트로피 손실과 규제 손실을 더하여 리스트에 추가
        val_loss = np.sum(-y_val*np.log(a))
        self.val_losses.append((val_loss + self.reg_loss())) / len(y_val) 

# %%
# 1. 의류 데이터 준비
# 텐서플로 임포트
import tensorflow as tf
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# %%
# 훈련 세트의 크기 확인
print(x_train_all.shape, y_train_all.shape)

# %%
# imshow() 함수로 샘플 이미지를 확인할수있다 해보자
import matplotlib.pyplot as plt
plt.imshow(x_train_all[0], cmap='gray')
plt.show()

# %%
# 타깃의 내용과 의미 확인 
print(y_train_all[:10]) 
class_names = ['티셔츠/윗도리', '바지', '스웨터', '드레스', '코트', '샌들', '셔츠', '스니커즈', '가방', '앵클부츠']
print(class_names[y_train_all[0]])

# %%
# 훈련 검증 세트로 나누기
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

# %%
# 훈련 검증 세트 레이블 확인
np.bincount(y_train)

# %%
np.bincount(y_val)

# %%
# 입력 데이터 정규화 하기
# 이미지 데이터는 픽셀마다 0~255 사이의 값을 가지므로 이값을 255로 나눠 0~1 사이로 맞춘다
x_train = x_train /255
x_val = x_val /255

# %%
# 훈련 세트와 검증 세트의 차원 변경하기
x_train = x_train.reshape(-1, 784)
x_val =x_val.reshape(-1, 784)

# %%
# 차원 확인
print(x_train.shape, x_val.shape) 

# %%
# to_categorical 함수 사용해 원-핫 인코딩하기
tf.keras.utils.to_categorical([0,1,3])

# %%
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)
print(y_train_encoded.shape, y_val_encoded.shape)
# 첫번째 레이블 출력
print(y_train[0], y_train_encoded[0])

# %%
# MultiClassNetwork 클래스로 다중 분류 신경망 훈련
fc =MultiClassNetwork(units=100,batch_size=256)
fc.fit(x_train, y_train_encoded,x_val = x_val, y_val= y_val_encoded, epochs=40)

# %%
