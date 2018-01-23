import numpy as np

a=np.arange(4)
print(a)
b=a
c=a
d=b
a[0]=22
print(a)
print(b)
print(b is a)
print(d is a)

d[1:3]=[22,33]
print(d)
print(a)
print(b)

b=a.copy()
print(b)
a[3]=44
print(b)