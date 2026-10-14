print("Systems start")
# micrograd is basically an autograd engine and autograd stands for automatic gradient, this implements backpropagation.

import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

print(f(3.0))

xs = np.arange(-5,5,0.25)
print(xs)
ys = f(xs)
print(ys)
plt.plot(xs,ys)
plt.show()

h = 0.0001
x = -3.0
print((f(x+h)- f(x))/h)

# derivative w.r.t. each input of d = a*b + c, nudge one input at a time
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
a += h
d2 = a*b + c
print('slope wrt a', (d2 - d1)/h)  # should be b = -3

a = 2.0
d1 = a*b + c
b += h
d2 = a*b + c
print('slope wrt b', (d2 - d1)/h)  # should be a = 2

b = -3.0
d1 = a*b + c
c += h
d2 = a*b + c
print('slope wrt c', (d2 - d1)/h)  # should be 1