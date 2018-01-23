import numpy as np

A=np.arange(3,15).reshape(3,4)
print(A)
print(A[2])
print(A[2][1])
print(A[2,1])
print(A[2,:])
print(A.flat)
print(A.flatten())

for row in A:
    print(row)
for column in A.T:
    print(column)
for item in A.flat:
    print(item)
