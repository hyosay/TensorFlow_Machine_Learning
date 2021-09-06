# 📌 TensorFlow_Machine_Learning
## 모두를 위한 딥러닝 정리 및 머신러닝 공부

## supervised learning
- image labeling
- Email spam filter
- Predicting exam score

supervised learning에는 Training data set이 존재한다.

## Types of supervised learning

- regression

|x(hours)|y(score)|
|---|---|
|10|90|
|9|80|
|3|50|
|2|30|

(Linear) Hypothesis을 정한다.  
> 7시간 공부하면 점수가 얼마가 나올까??   
> H(*x*) = Wx + b

### Cost(Loss) function

`H(x) - y` 를 쓰지않고 (H(*x*) -y)<sup>2</sup>를 쓴다.   
 왜냐하면 가설의 값과 실제 데이터의 차가 양수나 음수 무엇이 나올지 모르며 이것들이 합쳐져서 상쇄가 될 수 있기 때문이다.  
 ex 3 + (-3) + 2 + (-2) = 0  <- Feature를 제대로 찾지 못 할 수도 있다.
 

- binary classification ex)yes/no, pass/ non-pass

|x(hours)|y(pass/fail)|
|---|---|
|10|P|
|9|P|
|3|F|
|2|F|
- multi-label classification  ex)학점

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

 
