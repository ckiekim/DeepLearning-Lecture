import numpy as np
import matplotlib.pyplot as plt

def sigmo(x, b, ):   #
    array = []   # 빈 array를 만들어 줍니다.
    for itr in x:   # parameter로 전달 받은 x 변수(여기서는 array 만큼) 반복 실행
        array.append(1/(1+np.exp(-itr+b)))  # 반복 횟수는 x array만큼이며, 변경되는 값은 itr입니다.
    return array  # 함수의 반환 값

x = np.linspace(-8, 8, 100)  # numpy의  linspace (start, end, num)를 이용해서 -8과 8사이에 100개의 값 생성

sig0 = sigmo(x, 0)  # 앞에서 define했던 sigmo함수 호출 (파라메터로 위에서 생성한 x array 전달)
sig_2 = sigmo(x, -2)
sig4 = sigmo(x, 4)

plt.plot(x, sig0)  # matplotlib의 plot 기능을 이용하여 (x, y) 좌표 표수
plt.plot(x, sig_2)
plt.plot(x, sig4)
plt.show()    # plot를 화면에서 볼 수 있도록  show()