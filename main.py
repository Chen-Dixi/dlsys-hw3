import sys
sys.path.append('./python')
import numpy as np
import needle as ndl
from needle import backend_ndarray as nd

device = nd.cuda()
dims = [
    (8,9,10),
    (9,9,10),
]
if __name__ == '__main__':
    for (m, n, p) in dims:
        _A = np.random.randn(m, n)
        _B = np.random.randn(n, p)
        # 实验组
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        # np.testing.assert_allclose((A @ B).numpy(), _A @ _B, rtol=1e-5, atol=1e-5)
        C = (A @ B).numpy()
        _C = (A.matmul_vanilla(B)).numpy()
        print(C)
        print(_C)
        print(C - _C)