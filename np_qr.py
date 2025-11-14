import numpy as np
from tinygrad import Tensor
from tinygrad.dtype import dtypes

def householder_qr_np(A):
    """
    Batched Householder QR for A with shape (..., m, n).
    Returns Q, R where Q is (..., m, m) orthogonal and R is (..., m, n) upper-triangular.
    Numerically stable choices:
      - alpha = -sign(x0) * ||x|| to avoid catastrophic cancellation
      - handle zero/near-zero norms with small eps
      - fully batch-aware (no .item() or scalar extraction)
    """
    A = np.array(A, copy=True)  # avoid mutating input
    if A.ndim < 2:
        raise ValueError("expected matrix with at least 2 dimensions")

    *b_shape, m, n = A.shape
    batch_shape = tuple(b_shape)
    R = A.astype(np.float64)  # use high precision internally
    Q = np.broadcast_to(np.eye(m), batch_shape + (m, m)).copy().astype(np.float64)

    min_k = min(m, n)
    for i in range(min_k):
        # x is the vector to zero below diagonal, shape (..., k) where k = m - i
        x = R[..., i:, i]  # shape batch + (k,)
        # vector norm along last axis, keep dims for broadcasting
        norm_x = np.linalg.norm(x, axis=-1, keepdims=True)  # shape batch + (1,)

        # sign of first element, with zeros mapped to +1
        x0 = x[..., :1]  # shape batch + (1,)
        sign = np.sign(x0)
        sign = np.where(sign == 0, 1.0, sign)

        # alpha chooses direction to avoid cancellation
        alpha = -sign * norm_x  # shape batch + (1,)

        # v = x - alpha * e1 (with only first component changed)
        v = x.copy()
        v_first = x[..., :1] - alpha  # shape batch + (1,)
        v[..., :1] = v_first

        # beta = v^T v (norm squared). If beta == 0 => reflector is identity
        beta = np.sum(v * v, axis=-1, keepdims=True)  # shape batch + (1,)

        # tolerance to treat extremely small beta as zero
        eps = np.finfo(R.dtype).eps * (m)  # scale with dimension
        zero_mask = (beta <= eps)

        # tau = 2 / beta where beta != 0
        tau = np.where(zero_mask, 0.0, 2.0 / beta)  # shape batch + (1,)

        # Build H_k = I_k - tau * v v^T  (small matrix of size k x k)
        k = m - i
        # v has shape batch + (k,)
        # Build outer product v v^T -> shape batch + (k, k)
        vvT = v[..., :, None] * v[..., None, :]  # broadcasting
        # tau shape batch + (1,) -> reshape to batch + (1,1) to broadcast with vvT
        tau_mat = tau[..., None, None]
        Hk = np.eye(k)[(None,) * len(batch_shape)] - tau_mat * vvT  # shape batch + (k,k)
        # For small batches, np.eye broadcast may require copying:
        # apply Hk to the appropriate subblocks
        # Update R[..., i:, i:] = Hk @ R[..., i:, i:]
        R_block = R[..., i:, i:]  # shape batch + (k, n-i)
        # matrix multiply: (batch, k, k) @ (batch, k, n-i) -> (batch, k, n-i)
        R[..., i:, i:] = np.matmul(Hk, R_block)

        # Update Q[..., :, i:] = Q[..., :, i:] @ Hk^T
        Q_block = Q[..., :, i:]  # shape batch + (m, k)
        # We need to left-multiply by identity sized m x m reflector that is identity except Hk on trailing subspace.
        # Equivalent: Q[..., :, i:] = Q[..., :, i:] @ Hk
        Q[..., :, i:] = np.matmul(Q_block, Hk)

    # Now R may have tiny subdiagonal residuals due to numeric rounding: zero them explicitly
    tril_idx = np.tril_indices(m, -1)
    print(tril_idx)
    print(R)
    R[..., tril_idx[0], tril_idx[1]] = 0.0

    # Return Q (orthogonal) and R (upper triangular)
    return Q.astype(A.dtype), R.astype(A.dtype)

def qr(a) -> tuple[Tensor, Tensor]:
    """
    Batched Householder QR for A with shape (..., m, n).
    Returns Q, R where Q is (..., m, m) orthogonal and R is (..., m, n) upper-triangular.

    Numerically stable:
      - alpha = -sign(x0) * ||x||
      - Handles zero/near-zero norms with eps
      - Fully batch-aware
    """
    assert a.ndim > 1, "expected at least 2D matrix input"

    # --- promote integer input to float
    A = a.cast(dtypes.float32) if a.dtype.name.startswith("int") else a
    b_shape, m, n = A.shape[:-2], int(A.shape[-2]), int(A.shape[-1])
    R = A.clone()
    Q = Tensor.eye(m, dtype=A.dtype).reshape((1,) * len(b_shape) + (m, m)).expand(b_shape + (m, m)).contiguous()

    min_k = min(m, n)
    # print(dtypes.finfo(dtypes.float32))
    exp_bits, mant_bits = dtypes.finfo(A.dtype)
    eps_val = 2.0 ** (-mant_bits)
    eps = Tensor([eps_val * m], dtype=A.dtype).realize()  # broadcastable epsilon

    # eps = (Tensor.ones(1, dtype=A.dtype) * dtypes.finfo(dtypes.float32).eps * m).realize()  # broadcastable epsilon

    for i in range(min_k):
        # --- x is the current column below diagonal
        x = R[..., i:, i]  # shape (..., k)
        norm_x = (x.square().sum(-1, keepdim=True)).sqrt()  # (...,1)

        # --- sign of first element (replace zeros with +1)
        x0 = x[..., :1]
        sign = x0.sign()
        sign = sign.where(sign != 0, Tensor.ones_like(sign))

        # --- alpha and v
        alpha = -sign * norm_x  # (...,1)
        v = x.clone()
        v[..., :1] = x0 - alpha  # modify first component

        # --- beta = v^T v
        beta = v.square().sum(-1, keepdim=True)
        zero_mask = beta <= eps
        tau = Tensor.where(zero_mask, Tensor.zeros_like(beta), 2.0 / beta)

        # --- build the Householder small Hk
        k = m - i
        vvT = v.unsqueeze(-1) * v.unsqueeze(-2)  # (..., k, k)
        Hk = Tensor.eye(k, dtype=A.dtype).reshape((1,) * len(b_shape) + (k, k)).expand(b_shape + (k, k)) - tau[..., None, None] * vvT

        # --- apply Hk to R[..., i:, i:]
        R_block = R[..., i:, i:]
        R[..., i:, i:] = Hk @ R_block

        # --- apply to Q[..., :, i:]
        Q_block = Q[..., :, i:]
        Q[..., :, i:] = Q_block @ Hk

    # --- zero out small subdiagonal entries for clean R
    tril_i, tril_j = np.tril_indices(m, -1)
    mask = Tensor.zeros_like(R)
    mask[..., tril_i, tril_j] = 1.0
    R = R * (1 - mask)

    return Q, R


# a = Tensor.arange(20, dtype="float32").reshape(5,4)
a = Tensor([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.],
       [12., 13., 14., 15.],
       [16., 17., 18., 19.]])
Q,R = a.qr()
print("Q:", Q.numpy())
print("R:", R.numpy())
print("Check A≈QR:", np.allclose(a.numpy(), (Q@R).numpy())); print(a.numpy()); qn, rn = np.linalg.qr(a.numpy()); print("Check A≈QR for numpy QR:", np.allclose(a.numpy(), qn@rn)); print(qn); print(rn); print(a.dtype);
print("check custom")
qc, rc = householder_qr_np(a.numpy())
print("Check A≈QR for custom:", np.allclose(a.numpy(), qc@rc))
print(qn.shape)
print(qc.shape)
print(qn)
print(qc)
# print(np.allclose(qn, qc))