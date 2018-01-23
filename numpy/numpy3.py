import numpy as np

a = np.array([10, 20, 30, 40])
b = np.arange(4)

print(a, b)
c = 10 * np.sin(a)
print(b == 3)

d = np.array([[1, 2], [1, 3]])
f = np.arange(4).reshape((2, 2))
print(d)
print(f)

g=d*f
g_dot=np.dot(d,f)
g_dot_2=d.dot(f)
print(g)
print(g_dot)
print(g_dot_2)

h=np.random.random((2,4))
print(h)

print(np.sum(h,axis=1))
print(np.min(h,axis=0))
print(np.max(h,axis=1))