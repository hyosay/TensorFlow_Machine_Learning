#선형 회귀 모델
#선형 회귀 모델을 구축한다
#데이터 3개이상일 떄 의미
#비용(cost) : 가설이 얼마나 정확한 지 판단하는 기준
#=> (예측 값 - 실제 값) ** 2

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.Variable(5)
b = tf.Variable(3)
c = tf.multiply(a,b)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(c)

