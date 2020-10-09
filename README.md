# TensorFlow_Machine_Learning

# why activation function

* #### 그저 가중치와 편향만 구해서 더하면 선형연결이 되는데
#### 이것을 비션형으로 만들어 주기 위해서 activation function이 필요하다.


## 1. sigmoid

* ### The range of the sigmoid function is from 0 to 1



### backpropagation에서는 sigmoid가 위험 할 수가 있다.

* ### 즉 Vanishing gradeints

## 2. Relu(Rectified Linear Units

* ### It converges faster than sigmoid

#### image는 0 에서 255의 값인데 마이너스의 값을 받으면 의미가 없기 떄문에 0으로 바꾸는게 맞다고 생각해서 Relu가 처리가 잘된따고 생각함 

* ### Dhing ReLU / the linear form in the positive range

#### 음수의 값이 많이 나오면 파라미터 업데이터가 잘 안됨
 
