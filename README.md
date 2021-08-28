# 📌 TensorFlow_Machine_Learning


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

 
