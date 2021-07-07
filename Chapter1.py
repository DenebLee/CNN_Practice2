# 선형회귀 
# 머신러닝 알고리즘 중 가장 간단하면서도 딥러닝의 기초가 되는 선형 회귀
# 선형회귀는 기울기와 절편을 찾아냄

# 실질적 문제 해결을 위한 당뇨병 환자의 데이터 준비하기
# 사이킷런에서 당뇨병 환자 데이터 가져오기
# load_diabetes() 함수로 당뇨병 데이터 준비하기
#%%
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# %%
print(diabetes.data.shape, diabetes.target.shape)
# 442 x 10 크기의 2차원 배열이고 target은 44개의 요소를 가진 1차원 배열
# 여기서 행은 샘플이고 열은 샘플의 특성
# 샘플이란 당뇨병 환자에 대한 특성으로 이루어진 데이터 1세트를 의미하고 특성은 당뇨병 데이터의 여러 특징들을 의미

# %%
# 입력 데이터 자세히 보기
diabetes.data[0:3]

# %%
# 타깃 데이터 자세히 보기
diabetes.target[:3]
# 타깃 데이터는 10개의 요소로 구서오딘 샘플 1개에 대응
# 실무에서는 데이터 준비에 많은 공을 들임
# %%
# 가져온 데이터 시각화하기
# 산점도 그래프
import matplotlib.pyplot as plt
plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
# 훈련 데이터 준비하기
x = diabetes.data[: , 2]
y = diabetes.target
#################################################################################################
# 선형 회귀의 목표는 입력 데이터와 타깃 데이터를 통해 기울기와 절편을 찾는것
# 경사 하강법은 선형회귀의 목표를 통해 모델이 데이터를 잘 표현할 수 있또록 기울기를 사용하여 모델을 조금씩 조정하는 최적화 알고리즘
# 훈련 데이터에 잘 맞는 가중치와 절편을 찾는방법
# 1. 무작위로 w와 b를 정함
# 2. x에서 샘플 하나를 선택하여 예측값 y^을 계산
# 3. y^랑 선택한 샘플의 진짜 y를 비교
# 4. y^이 y와 더 가까워지도록 w,b 조정하기
# 5. 모든 샘플을 처리할 떄까지 다시 2~4항목 반복
################################################################################################

# 훈련 데이터에 맞는 w와 b 찾아보기
# 1. w와 b 초기화
w = 1.0
b = 1.0
# %%
# 훈련 데이터의 첫 번째 샘플 데이터로 y^ 얻기
y_hat = x[0] * w + b
print(y_hat)

# %%
# 3. 타깃과 예측 데이터 비교하기
print(y[0])

# %%
# 4. w값 조절해 예측값 바꾸기
w_inc = w + 0.1
y_hat_inc = x[0] * w_inc + b 
print(y_hat_inc)

# %%
# 5. w값 조정한 후 예측값 증가 정도 확인
w_rate =(y_hat_inc - y_hat) / (w_inc -w)
print(w_rate)

# %%
# 변화율로 가중치 업데이트하기
# 변화율이 양수일때 w가 증가하면 y_hat도 증가함  
# 이때 변화율이 양수인점을 이용하여 변화율을 w에 더하는 방법으로 w를 증가 시킬수 있음
#  반대로 변화율이 음수일때는 w가 증가하면 y_hat은 감소한다 w가 감소하면 반대로 y_hat은 증가할것이니 그냥 더하면됨

w_new = w+ w_rate
print(w_new)

# %%
# 변화율로 절편 업데이트하기
# 절편 b에 대한 변화율을 구한 다음 변화율로 b를 업데이트
b_inc = b + 0.1 
y_hate_inc = x[0] * w + b_inc
print(y_hat_inc)

b_rate = (y_hat_inc - y_hat) / (b_inc -b)
print(b_rate)

# %%
# 변화율로 가중치 업데이트 하기
# b_rate 값이 1이기때문에 b를 업데이트 하기 위해서는 단순히 1을 더하면됨
b_new = b+1
print(b_new)

# %%
# 오차 역전파로 가중치와 절편을 더 적절하게 업데이트
# 오차 역전파는 y^와 y의 차이를 이용하여 w와 b를 업데이트
# 이름에서 알수 있듯이 오차파 연이퍼 전파되는 모습으로 수행

# 가중치와 절편을 더욱 적절하게 업데이트 하는방법


# 1. 오차와 변화율을 곱하여 가중치 업데이트하기
err = y[0] - y_hat
w_new = w_rate * err
b_new = b+1 *err
print(w_new, b_new)

# %%
# 2. 두 번째 샘플 x[1]을 사용하여 오차를 구하고 새로운 w와 b를 구해보기
y_hat = x[1] * w_new + b_new
err = y[1] - y_hat_inc
w_rate = x[1]
w_new = w_new + w_rate * err
b_new =b_new +1 * err
print(w_new, b_new)
# w는 4만큼 커지고 b는 절반으로 줄어들었음
# 이런 방식으로 모든 샘플을 사용해 가중치와 절편을 업데이트

# %%
# 3. 전체 샘플을 반복하기
for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b 
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print(w,b)
# 파이썬의 zip() 함수는 여러개의 배열에서 동시에 요소를 하나씩 꺼내줍니다. 입력 x와 타깃 y배열에서 요소를 하나씩 꺼내어 err를 계산하고 w와 b를 업데이트

# %%
# 4. 과정3을 통해 얻어낸 모델이 전체 데이터 세트를 잘 표현하는지 그래프 출력
plt.scatter(x,y)
pt1 = (-0.1, -0.1 * w + b )
pt2 = (0.15, 0.15 * w + b )
plt.plot([pt1[0], pt2[0]],[pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
# 5. 여러 에포크 반복하기
# 보통 경사 하강법에서는 주어진 훈련 데이터로 학습을 여러 번 반복
# 이렇게 전체 훈련데이터를 모두 이용하여 한 단위의 작업을 진행하는것을 에포크(epoch)
for i in range(1 , 100):
    for x_i, y_i in zip( x , y ):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err
print( w , b )
# 100번의 에포크를 반복하여 찾은 w와 b는 약 913.6, 123,4

# %%
# 그래프로 출력
plt.scatter(x,y)
pt1 = (-0.1, -0.1 * w + b )
pt2 = (0.15, 0.15 * w + b )
plt.plot([pt1[0], pt2[0]],[pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# 직선이 전체 데이터의 경향을 제법 잘따라가는걸 확인할 수있음
# 여기까지 했던 과정을 통해 얻은 모델
# y^ = 913.6x + 123.4

# %%
# 6. 모델로 예측하기
x_new = 0.18
y_pred = x_new * w + b 
print(y_pred)

#%%
# 위 데이터를 산점도 위에 나타내기
plt.scatter(x,y)
plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
# 지금까지 모델을 만든과정
# 1. w와 b를 임의의 값으로 초기화하고 훈련 데이터의 샘플을 하나씩 대입하여 y와 y^오차를 구함
# 2. 1에서 구한 오차를 w와 b의 변화율에 곱하고 이 값을 이용하여 w와 b를 업데이트
# 3. 만약 y^ 이 y보다 커지면 오차는 음수가 되어 자동으로  w와 b가 줄어드는 방향으로 업데이트
# 4. 반대로 y^이 y보다 작으면 오차는 암수가 되고 w와 b는 더 커지도록 업데이트


#%%
# 손실 함수부터 
# 제곱 오차는 타깃값과 예측값을 뺀 다음 제곱한 것에 불과
# 제곱오차가 최소가 되면 산점도 그래프를 가장 잘 표현한 직선이 그려짐


# 가중치에 대하여 제곱 오차 미분하기 및 편미분 등 공식은 구박사한테물어보시길
# Neuron 클래스 만들기
class Neuron:
    def __init__(self):
        self.w = 1.0 # 가중치 및 절편 초기화
        self.b = 1.0 # 학습할때 가중치 w와 절편 b는 시작값이 있어야 하므로 임으로 1.0을 지정함
        # 정방향 계산을 위한 메서드를 정의  
    def forpass(self, x): #직선 방정식 계산
        y_hat = x * self.w + self.b
        return y_hat
        # 정방향 계산을 통해 얻어낸 y^과 y의 차이 즉 오차가 뉴런의 오른쪽 방향에서 왼쪽으로 감
        # 이를 역방향 계산이라고 부름
    def backprop(self, x, err):
        w_grad = x * err # 가중치에 대한 그레이디언트 계산
        b_grad = 1 * err # 절편에 대한 그레이디언트 계산
        return w_grad, b_grad
        # 편미분을 통해 얻었던 수식을 그대로 코드에 옮김
        # 즉 가중치의 그레이디언트는 x와 오차를 곱하고 절편의 그레이디언트는 1과 오차를 곱하면됨
    def fit(self, x, y, epochs=100): 
        for i in range(epochs): # 해당 에포크만큼 반복
            for x_i, y_i in zip(x , y): # 모든 샘플에 대해 반복
                y_hat = self.forpass(x_i) # 정방향 계산
                err = -(y_i - y_hat) # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
                self.w-= w_grad
                self.b -= b_grad
neuron = Neuron( )
neuron.fit( x , y )

# %%
plt.scatter( x , y )
plt1 = (-0.1, -0.1 * neuron.w + neuron.b)
plt2 = (0.15, 0.15 * neuron.w + neuron.b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
