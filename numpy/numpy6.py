import numpy as np

A=np.array([1,1,1])[:,np.newaxis]
B=np.array([2,2,2])[:,np.newaxis]

C=np.vstack((A,B))
D=np.hstack((A,B))

print(A.shape,B.shape)
print(C)
print(D)
print(C.shape)
print(D.shape)

# print(A.T)
# print(A[:,np.newaxis].shape)

E=np.concatenate((A,B,B,A),axis=1)
print(E)