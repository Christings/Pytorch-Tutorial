import numpy as np

A=np.arange(14,2,-1).reshape((3,4))

print(A)
print(np.argmin(A))
print(np.argmax(A))
print(np.mean(A))
print(A.mean())
print(np.average(A))
print(np.median(A))
print(np.cumsum(A))
print(np.diff(A))
print(np.nonzero(A))
print(np.sort(A))
print(np.transpose(A))
print(A.T)
print(np.clip(A,5,9))
print(np.mean(A,axis=0))



