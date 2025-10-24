from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes
import numpy as np

# a = Tensor.randn((4, 5))
# q,r,p = np.linalg.qr(a, mode='reduced', pivoting=True)
# q2, r2 = np.linalg.qr(r.T)
# x = r2.T

a = Tensor.randn((2, 4, 5))
print(a.numpy())
print(a[..., :1].numpy())
print(100 * "-")

def qr(A: Tensor) -> Tensor:
    pass

def jacobi_rotation(A:Tensor, i:int, j:int) -> Tensor:
    """one round naive jacobi rotation"""
    # rotate using the cis transformation matrix.
    p = A[..., i]
    pass


def svd(A: Tensor) -> Tensor:
    assert A.ndim > 1, f"expected two or more dimensions but got {A.ndim}"
    b, m, n = A.shape[:-2], int(A.shape[-2]), int(A.shape[-1])
    # apply qr factorization
    for i in range(m):
        for j in range(i+1, n):
            jacobi_rotation(A, i, j)

A = Tensor.randn((7, 4)) # (b, h, c, w)
b, m, n = A.shape[:-2], int(A.shape[-2]), int(A.shape[-1])
num = min(m, n)
# print((1,)*len(b) + (num, num))
print(Tensor.eye(num).reshape((1,)*len(b) + (num, num)).expand(b + (num, num)).numpy())
print(b + (num, num))
V = Tensor.eye(num,).reshape((1,) * len(b) + (num, num)).expand(b + (num, num)).contiguous()

# permute = Tensor.arange(0, 10, dtype=dtypes.int)
# inv_permute = Tensor.zeros(10, dtype=dtypes.int).contiguous()
# print(permute[num//2:num].numpy())
# print(permute[num//2:num].flip(0).numpy()) # reverse half of permute
# print(permute.numpy())
permute, inverse_permute = Tensor.arange(0, num, dtype=dtypes.int), Tensor.zeros(num, dtype=dtypes.int).contiguous()
permute[num//2:num] = permute[num//2:num].flip(0)
print(inverse_permute.numpy())
inverse_permute[permute] = Tensor.arange(num, dtype=dtypes.int)
# print(inverse_permute.numpy())

V_permuted, runoff_V = (V[..., permute].split(num - 1, -1)) if num % 2 == 1 else (V[..., permute], None)
print(V_permuted.numpy())
# print(runoff_V.numpy())
print(V[..., -1].numpy())
# V_left, V_right = V_permuted.split(num//2, -1)
# U_permuted, runoff_U = (U[..., permute].split(num - 1, -1)) if num % 2 == 1 else (U[..., permute], None)
# U_left, U_right = U_permuted.split(num//2, -1)
print(V[:, [3, 4]].numpy())
print(permute[0].reshape(1).cat(((permute[1:num] - 2) % (num - 1)) + 1).numpy())
print(permute.numpy())
print(((permute - 1) % num).numpy())
# 0, 1, 2, 3, 4  # p
# -1, 0, 1, 2, 3 # p - 1
# (a + b) % m = (a % m + b % m) % m
"""
----->
[x, y] | [c,  s]
       | [-s, c]

= out shape (m x 2 @ 2 x 2) => m x 2
= [x * c - y * s, x * s + y * c]
"""
# print(inverse_permute.numpy())
# inverse_permute = inverse_permute.scatter(0,((permute - 1) % num),Tensor.arange(num,dtype=dtypes.int32))
# print(inverse_permute.numpy())


print(100 * "=")

# a = Tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[1, 2, 3], [4, 5, 6]]]]) # b, c, h, w
# print(a.shape)
# print(a[..., 1:2].numpy())