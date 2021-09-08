# 📌 TensorFlow_Machine_Learning
## 모두를 위한 딥러닝 정리 및 머신러닝 공부

## supervised learning
- image labeling
- Email spam filter
- Predicting exam score

supervised learning에는 Training data set이 존재한다.

## Types of supervised learning

### - regression

|x(hours)|y(score)|
|---|---|
|10|90|
|9|80|
|3|50|
|2|30|

(Linear) Hypothesis(가설)을 정한다.  
> ex) 7시간 공부하면 점수가 얼마가 나올까??   

#### Cost(Loss) function

`H(x) - y` 를 쓰지않고 `(H(x) -y)^2`를 쓴다.   
 왜냐하면 가설의 값과 실제 데이터의 차가 양수나 음수 무엇이 나올지 모르며 이것들이 합쳐져서 상쇄가 될 수 있기 때문이다.
 > 3 + (-3) + 2 + (-2) = 0  <- Feature끼리 상쇄가 되어 제대로된 Feature를 찾지 못 할 수도 있다.
 
 <img width="400" alt="스크린샷 2021-09-06 오후 6 41 18" src="https://user-images.githubusercontent.com/46950334/132196808-6b1dc77c-45bd-4b05-ba27-9a6f344533fb.png">
 
 `H(x) = Wx + b`   
 **point.1  *MINIMIZE COST(W, b)***
 
 ### linear regression과 cost(loss) function을 정리해보자면 결국 H(x)의 값이 무엇인지 hypothesis이 궁금할떄 minimize cost_func을 , 가장 기본적으로 mse(오차제곱합)을 사용하며 W의 값과 b의 값을 최소화 하기 위해 gradient descent algorithm을 사용하는 것이다.

### - binary classification   ex) yes/no, pass/ non-pass

|x(hours)|y(pass/fail)|
|---|---|
|10|P|
|9|P|
|3|F|
|2|F|

binary classification도 linear regression으로 구분을 지어보려했지만 x의 값이 커지거나,작아지면 손실비용이 크게 조정되기 때문에 이것을 해결하기 위해 곡선 'S'모앙인 logistic func(sigmoid func) 고안했다.

logistic regression에서는 cost를 mse, gredient descent algorithm을 사용하면 local variable, global variable이 존재하여 사용하기 불편하다. 
그래서 
cost func을 mse로 선택하여 gredient descent형태로 나타내면 안되고 cross entropy를 사용하여 2차곡선의 형태로 만든 후 gredient descent algorithm을 사용해야한다.
### - multi-label classification  ex)학점

|x(hours)|y(grade)|
|---|---|
|10|A|
|9|B|
|3|C|
|2|D|


## why activation function
##### 그저 가중치와 편향만 구해서 더하면 선형연결이 되는데 이것을 비션형으로 만들어 주기 위해서 activation function이 필요하다.


### 1. sigmoid

* ### The range of the sigmoid function is from 0 to 1

### backpropagation에서는 sigmoid가 위험 할 수가 있다.

역전파 알고리즘을 진행할때 기울기가 손실되는 상황이 발생(**vanishing gradient problem**)


## 2. Relu(Rectified Linear Units

* ### It converges faster than sigmoid

#### image는 0 에서 255의 값을 가짐, 그래서 마이너스의 값을 받으면 의미가 없다고 판단, 즉 0으로 바꾸는게 맞다고 생각해서 Relu가 처리가 잘된다고 생각함 

* ### Dhing ReLU / the linear form in the positive range

#### 음수의 값이 많이 나오면 파라미터 업데이터가 잘 안됨


딥러닝, 머신러닝 공부 방향
1. ian goodfellow의 deep learning을 통해서 이론을 마스터하기.

2. 머신러닝책을 병행하여 공부하기
 
## 나의 목표는 GAN을 연구하는 것이다.



참고 : https://gomguard.tistory.com/184?category=712467

 
