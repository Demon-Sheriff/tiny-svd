from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes
import numpy as np

np.random.seed(2)
Tensor.manual_seed(2)

an = np.arange(20).reshape((4, 5))
a = Tensor.arange(20).reshape((4, 5))
print(a.sum().numpy())
# q,r,p = np.linalg.qr(a, mode='reduced', pivoting=True)
# q2, r2 = np.linalg.qr(r.T)
# x = r2.T

# a = Tensor.randn((2, 4, 5))
print(a.numpy())
print(a[..., :, 2].square())
# print(a[..., :, 0].numpy())
# print(100 * "-")

# along the column axis(...., -2, -1)
# Tensor.zeros_like(a, )
# a.where()
# a.scatter(-2, index=Tensor.arange())
print(a.square().sum(axis=-2).sqrt().numpy())
print((a.T @ a).diagonal().diag().numpy())
# print((a - a.diag(keepdims=True)).numpy())
def qr(A: Tensor) -> Tensor:
    pass

def jacobi_rotation(A:Tensor, V:Tensor, i:int, j:int) -> Tensor:
    """one round naive jacobi rotation"""
    # compute the cis transformation matrix.
    p = A[..., :, i].clone()
    q = A[..., :, j].clone()
    a = p.square().sum()
    b = q.square().sum()
    c = (p * q).sum() 
    if c.item() == 0: return A, V 
    tau = (b - a) / (2*c)
    t = tau.sign() / (tau.abs() + (1 + tau.square()).sqrt())
    c_ = 1 / (1 + t.square()).sqrt()
    s_ = c_ * t 
    # apply the rotations
    A[..., :, i] = p * c_ - q * s_
    A[..., :, j] = p * s_ + q * c_

    vi = V[..., :, i].clone()
    vj = V[..., :, j].clone()
    V[..., :, i] = c_ * vi - s_ * vj
    V[..., :, j] = s_ * vi + c_ * vj
    return A, V

def svd(A: Tensor, tol=1e-12) -> Tensor:
    assert A.ndim > 1, f"expected two or more dimensions but got {A.ndim}"
    b, m, n = A.shape[:-2], int(A.shape[-2]), int(A.shape[-1])
    converged = False
    V = Tensor.eye(n).contiguous()
    while not converged:
        for i in range(n):
            for j in range(i+1, n):
                A, V = jacobi_rotation(A, V, i, j)
        X = A.T @ A
        a_norm = A.square().sum().sqrt()
        off_norm = (X - X.diagonal().diag()).square().sum().sqrt()
        # off_norm = ((X - X.diagonal(offset=0, dim1=-2, dim2=-1).diag_embed()) ** 2).sum().sqrt()
        if off_norm.item() < a_norm.item() * tol: converged = True

    sig = A.square().sum(axis=-2).sqrt()
    U = A / sig[..., None, :]
    return U, sig, V

# A = Tensor.randn((7, 4)) # (b, h, c, w)
# b, m, n = A.shape[:-2], int(A.shape[-2]), int(A.shape[-1])
# num = min(m, n)
# # print((1,)*len(b) + (num, num))
# print(Tensor.eye(num).reshape((1,)*len(b) + (num, num)).expand(b + (num, num)).numpy())
# print(b + (num, num))
# V = Tensor.eye(num,).reshape((1,) * len(b) + (num, num)).expand(b + (num, num)).contiguous()

# permute = Tensor.arange(0, 10, dtype=dtypes.int)
# inv_permute = Tensor.zeros(10, dtype=dtypes.int).contiguous()
# print(permute[num//2:num].numpy())
# print(permute[num//2:num].flip(0).numpy()) # reverse half of permute
# print(permute.numpy())
# permute, inverse_permute = Tensor.arange(0, num, dtype=dtypes.int), Tensor.zeros(num, dtype=dtypes.int).contiguous()
# permute[num//2:num] = permute[num//2:num].flip(0)
# print(inverse_permute.numpy())
# inverse_permute[permute] = Tensor.arange(num, dtype=dtypes.int)
# print(inverse_permute.numpy())

# V_permuted, runoff_V = (V[..., permute].split(num - 1, -1)) if num % 2 == 1 else (V[..., permute], None)
# print(V_permuted.numpy())
# print(runoff_V.numpy())
# print(V[..., -1].numpy())
# V_left, V_right = V_permuted.split(num//2, -1)
# U_permuted, runoff_U = (U[..., permute].split(num - 1, -1)) if num % 2 == 1 else (U[..., permute], None)
# U_left, U_right = U_permuted.split(num//2, -1)
# print(V[:, [3, 4]].numpy())
# print(permute[0].reshape(1).cat(((permute[1:num] - 2) % (num - 1)) + 1).numpy())
# print(permute.numpy())
# print(((permute - 1) % num).numpy())
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


# print(100 * "=")

# a = Tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[1, 2, 3], [4, 5, 6]]]]) # b, c, h, w
# print(a.shape)
# print(a[..., 1:2].numpy())
print(100 *"-")
print(an)
print("for matrix : ")
print(a.numpy())
print("computing tiny-svd")
u, sig, v = svd(a)
print("computing numpy svd")
un, sn, vn = np.linalg.svd(an)
print(un)
print(sn)
print("computing tinygrad-svd")
U, S, V = a.realize().svd()
print("comparisons")
print(u.numpy(), U.numpy())
print(sig.numpy(), S.numpy())