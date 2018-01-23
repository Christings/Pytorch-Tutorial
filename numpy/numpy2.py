import numpy as np

a=np.array([1,2,3])
print(a)  # no--','

b=np.array([1,2,3],dtype=np.float32)
print(b.dtype)  # no--','

c = np.array([[1, 2, 3], [2, 3, 4]])
print(c)

d = np.zeros((3,4))
print(d)

e=np.ones((3,4),dtype=np.int)
print(e)

f=np.empty((3,4))
print(f)

g=np.arange(10,20,2)
print(g)

h=np.arange(12).reshape((3,4))
print(h)

i=np.linspace(1,10,6).reshape((2,3))
print(i)