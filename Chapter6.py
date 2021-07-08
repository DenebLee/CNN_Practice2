# 합성곱 구현하기
# 1. 넘파이 배열을 정의하고 배열 하나 선택해 뒤집기
#%%
import numpy as np
from numpy import core
w = np.array([2,1,5,3])
x = np.array([2,8,3,7,1,2,0,4,5])

# %%
# 넘파이 사용
w_r = np.flip(w)
print(w_r)

# %%
# 파이썬의 슬라이스 연산자로 배열을 뒤집을수도 있음 당연히
w_r = w[::-1]
print(w_r)

# %%    
# 넘파이의 점 곱으로 합성곱 수행하기
for i in range(6):
    print(np.dot(x[i:i+4],w_r))

# %%
# 합성곱 신경망은 진짜 합성곱을 사용하지 않음
# 대부분의 딥러닝 패키지들은 합성곱 신경망을 만들 때 합성곱이 아니라 교차 상관을 사용함
# 합성곱과 교차 상관은 매우 비슷함
# 교차 상관은 합성곱과 동일한 방법으로 연산이 진행되지만 '미끄러지는 배열을 뒤집지 않는다'라는 점이 다름
# 교촤 상관 역시 싸이파이의 correlate() 함수를 사용하면 간단히 계산가능
from scipy.signal import correlate
correlate(x, w, mode='valid')

# %%
# 패딩은 원본 배열의 양끝에 빈원소를 추가하는 것을 말하고 스트라이드는 미끄러지는 배열의 간격을 조절하는 것을 말함
correlate(x, w, mode='full')

# %%
# 세임패딩 적용
correlate(x, w, mode='same')

# %%
# 싸이파이의 correlate2d() 함수를 사용하여 2차원 배열의 합성곱을 계산
x = np.array([[1,2,3,], [4,5,6], [7,8,9]])
w = np.array([[2,0],[0,0]])
from scipy.signal import correlate2d
correlate2d(x, w, mode='valid')

# %%
# 2차원 배열에서 패딩과 스트라이드
correlate2d(x,w, mode='same')
# 스트라이드의 경우 미끄러지는 방향은 그대로 유지하면서 미끄러지는 간격의 크기만 커짐
# %%
# 합성곱 신경망의 입력은 일반적으로 4차원 배열
# 텐서플로에서 2차원 합성곱을 수행하는 함수는 conv2d이다
# 2차원 배열을 4차원 배열로 바꿔 합성곱을 수행
import tensorflow as tf
x_4d = x.astype(np.float).reshape(1,3,3,1)
w_4d = w.reshape(2,2,1,1)

c_out = tf.nn.conv2d(x_4d, w_4d, strides=1, padding='SAME')
# %%
# 출력
c_out.numpy().reshape(3,3)

# %%
# 텐서플로의 max_pool2d() 함수를 사용하여 최대 풀링 수행
x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12],[13,14,15,16]])
x = x.reshape(1,4,4,1)

# %%
p_out = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
p_out.numpy().reshape(2,2)

# %%
# 렐루함수 구현하기
def relu(x):
    return np.maximum(x, 0)
    # 잘 작동하는지 샘플 데이터 넣기
    
# %%
x= np.array([-1,2,-3,4,-5])
relu(x)

# %%
r_out = tf.nn.relu(x)
r_out.numpy()


# %%
# 합성곱 신경망 만들고 훈륜
# 합성곱 적용
import tensorflow as tf

class ConvolutionNetwork:
    
    def __init__(self, n_kernels=10, units=10, batch_size=32, learning_rate=0.1):
        self.n_kernels = n_kernels  # 합성곱의 커널 개수
        self.kernel_size = 3        # 커널 크기
        self.optimizer = None       # 옵티마이저
        self.conv_w = None          # 합성곱 층의 가중치
        self.conv_b = None          # 합성곱 층의 절편
        self.units = units          # 은닉층의 뉴런 개수
        self.batch_size = batch_size  # 배치 크기
        self.w1 = None              # 은닉층의 가중치
        self.b1 = None              # 은닉층의 절편
        self.w2 = None              # 출력층의 가중치
        self.b2 = None              # 출력층의 절편
        self.a1 = None              # 은닉층의 활성화 출력
        self.losses = []            # 훈련 손실
        self.val_losses = []        # 검증 손실
        self.lr = learning_rate     # 학습률

    def forpass(self, x):
        # 3x3 합성곱 연산을 수행합니다.
        c_out = tf.nn.conv2d(x, self.conv_w, strides=1, padding='SAME') + self.conv_b
        # 렐루 활성화 함수를 적용합니다.
        r_out = tf.nn.relu(c_out)
        # 2x2 최대 풀링을 적용합니다.
        p_out = tf.nn.max_pool2d(r_out, ksize=2, strides=2, padding='VALID')
        # 첫 번째 배치 차원을 제외하고 출력을 일렬로 펼칩니다.
        f_out = tf.reshape(p_out, [x.shape[0], -1])
        z1 = tf.matmul(f_out, self.w1) + self.b1     # 첫 번째 층의 선형 식을 계산합니다
        a1 = tf.nn.relu(z1)                          # 활성화 함수를 적용합니다
        z2 = tf.matmul(a1, self.w2) + self.b2        # 두 번째 층의 선형 식을 계산합니다.
        return z2
    
    def init_weights(self, input_shape, n_classes):
        g = tf.initializers.glorot_uniform()
        self.conv_w = tf.Variable(g((3, 3, 1, self.n_kernels)))
        self.conv_b = tf.Variable(np.zeros(self.n_kernels), dtype=float)
        n_features = 14 * 14 * self.n_kernels
        self.w1 = tf.Variable(g((n_features, self.units)))          # (특성 개수, 은닉층의 크기)
        self.b1 = tf.Variable(np.zeros(self.units), dtype=float)    # 은닉층의 크기
        self.w2 = tf.Variable(g((self.units, n_classes)))           # (은닉층의 크기, 클래스 개수)
        self.b2 = tf.Variable(np.zeros(n_classes), dtype=float)     # 클래스 개수
        
    def fit(self, x, y, epochs=100, x_val=None, y_val=None):
        self.init_weights(x.shape, y.shape[1])    # 은닉층과 출력층의 가중치를 초기화합니다.
        self.optimizer = tf.optimizers.SGD(learning_rate=self.lr)
        # epochs만큼 반복합니다.
        for i in range(epochs):
            print('에포크', i, end=' ')
            # 제너레이터 함수에서 반환한 미니배치를 순환합니다.
            batch_losses = []
            for x_batch, y_batch in self.gen_batch(x, y):
                print('.', end='')
                self.training(x_batch, y_batch)
                # 배치 손실을 기록합니다.
                batch_losses.append(self.get_loss(x_batch, y_batch))
            print()
            # 배치 손실 평균내어 훈련 손실 값으로 저장합니다.
            self.losses.append(np.mean(batch_losses))
            # 검증 세트에 대한 손실을 계산합니다.
            self.val_losses.append(self.get_loss(x_val, y_val))

    # 미니배치 제너레이터 함수
    def gen_batch(self, x, y):
        bins = len(x) // self.batch_size                   # 미니배치 횟수
        indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞습니다.
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]   # batch_size만큼 슬라이싱하여 반환합니다.
            
    def training(self, x, y):
        m = len(x)                    # 샘플 개수를 저장합니다.
        with tf.GradientTape() as tape:
            z = self.forpass(x)       # 정방향 계산을 수행합니다.
            # 손실을 계산합니다.
            loss = tf.nn.softmax_cross_entropy_with_logits(y, z)
            loss = tf.reduce_mean(loss)

        weights_list = [self.conv_w, self.conv_b,
                        self.w1, self.b1, self.w2, self.b2]
        # 가중치에 대한 그래디언트를 계산합니다.
        grads = tape.gradient(loss, weights_list)
        # 가중치를 업데이트합니다.
        self.optimizer.apply_gradients(zip(grads, weights_list))
   
    def predict(self, x):
        z = self.forpass(x)                 # 정방향 계산을 수행합니다.
        return np.argmax(z.numpy(), axis=1) # 가장 큰 값의 인덱스를 반환합니다.
    
    def score(self, x, y):
        # 예측과 타깃 열 벡터를 비교하여 True의 비율을 반환합니다.
        return np.mean(self.predict(x) == np.argmax(y, axis=1))

    def get_loss(self, x, y):
        z = self.forpass(x)                 # 정방향 계산을 수행합니다.
        # 손실을 계산하여 저장합니다.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
        return loss.numpy()
# %%
x = tf.Variable(np.array([1.0, 2.0, 3.0]))
with tf.GradientTape() as tape:
    y = x ** 3 + 2 * x + 5

# 그래디언트
print(tape.gradient(y, x))
x = tf.Variable(np.array([1.0, 2.0, 3.0]))
with tf.GradientTape() as tape:
    y = tf.nn.softmax(x)

# 그래디언트를 계산합니다.
print(tape.gradient(y, x))# %%

# %%
# 텐서 플로를 사용해 패션 MNIST 데이터 세트불러오기
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# %%
# 훈련데이터 세트를 훈련 세트와 검증 세트로 나누니
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all,test_size=0.2, random_state=42)

# %%
# 타깃을 원-핫 인코딩으로 변환
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)


# %%
# 입력데이터 준비
x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)
x_train.shape

# %%
# 입력 데이터 표준화 전처리하기

x_train = x_train / 255
x_val = x_val / 255

# %%
# 모델훈련하기
cn = ConvolutionNetwork(n_kernels=10, units=100, batch_size=128, learning_rate=0.01)
cn.fit(x_train, y_train_encoded,x_val=x_val, y_val=y_val_encoded, epochs=20)

# %%
# 훈련, 검증 손실 그래프 그리고 검증 세트의 정확도 확인

import matplotlib.pyplot as plt
plt.plot(cn.losses)
plt.plot(cn.val_losses)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# %%
# 성능 확인
cn.score(x_val, y_val_encoded)

# 케라스로 합성곱 신경망 만들기
# 필요한 클래스들 임포트하기
#%%
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

# %%
# 1. 합성곱 쌓기
conv1 = tf.keras.Sequential()
conv1.add(Conv2D(10, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))

# 2. 풀링층 쌓기
conv1.add(MaxPooling2D((2, 2)))

# 3. 완전 연결층에 주입할 수 있도록 특성 맵 펼치기
conv1.add(Flatten())

# 4. 완전 연결층 쌓기
conv1.add(Dense(100, activation='relu'))
conv1.add(Dense(10, activation='softmax'))

# 5. 모델 구조 살펴보기
conv1.summary()
# 출력 결과를 보면 합성곱층의 출력 크기는 배치 차원을 제외하고 28x28x10
# 플링층과 특성맵을 완전 연결층에 펼쳐서 주입하기 위해 추가한 Flatten 층에는 가중치가 없다

# %%
# 합성곱 신경망 모델 훈련하기
# 아담 옵티마이저 사용
conv1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# %%
# 훈련
history = conv1.fit(x_train, y_train_encoded, epochs=20,validation_data=(x_val, y_val_encoded))

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()

# %%
loss, accuracy = conv1.evaluate(x_val, y_val_encoded, verbose=0)

# %%
print(accuracy)

# %%
# 드롭 아웃을 적용해 합성곱 신경망을 구현
# 1. 케라스로 만든 합성곱 신경망에 드롭아웃 적용
from tensorflow.keras.layers import Dropout

conv2 = tf.keras.Sequential()
conv2.add(Conv2D(10, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
conv2.add(MaxPooling2D((2, 2)))
conv2.add(Flatten())
conv2.add(Dropout(0.5))
conv2.add(Dense(100, activation='relu'))
conv2.add(Dense(10, activation='softmax'))

# %%
# 2. 드롭아웃층 확인
conv2.summary()

# %%
# 훈련하기

conv2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = conv2.fit(x_train, y_train_encoded, epochs=20,validation_data=(x_val, y_val_encoded))

# %%
# 마찬가지로 손실 그래프와 정확도 그래프 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()

#%%
loss, accuracy = conv2.evaluate(x_val, y_val_encoded, verbose=0)
print(accuracy)
# %%
