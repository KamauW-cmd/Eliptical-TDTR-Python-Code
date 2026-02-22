import numpy as np
from scipy.io import loadmat

'''
def lgwt_V4(N, a, b):
    """
    Computes the Legendre-Gauss nodes and weights for integration over [a, b].
    
    Parameters:
        N (int): number of quadrature points
        a (float): lower bound of interval
        b (float): upper bound of interval
    
    Returns:
        x (np.ndarray): quadrature nodes
        w (np.ndarray): quadrature weights
    """
    N1 = N
    N = N - 1
    N2 = N + 2

    xu = np.linspace(-1, 1, N1).reshape(-1,1)

    # Initial guess using Chebyshev nodes
    #y = np.cos((2 * np.arange(0, N1) + 1) * np.pi / (2 * N1)) + (0.27 / N1) * np.sin(np.pi * xu * N / N2)

    #y = np.cos((2 * np.arange(0, N1).reshape(-1, 1) + 1) * np.pi / (2 * N2)) + (0.27 / N1) * np.sin(np.pi * xu * N / N2)
    
    n = np.arange(0, N1).reshape(-1,1)
    y = np.cos((2 * n + 1) * np.pi / (2*N +2)) + (0.27 / N1) * np.sin(np.pi * xu * N / N2)
    y = y.flatten()

    # Legendre-Gauss Vandermonde Matrix and its derivative
    L = np.zeros((N1, N2))
    Lp = np.zeros((N1, N2))

    y0 = 2 * np.ones_like(y)

    y = y.astype(np.longdouble)
    L = np.zeros((N1, N2), dtype=np.longdouble)
    Lp = np.zeros_like(L)

    # Newton-Raphson iteration
   
    while np.max(np.abs(y - y0)) > np.finfo(float).eps:
        L[:, 0] = 1
        L[:, 1] = y
        
        for k in range(1, N1 - 1):
            L[:, k + 1] = ((2 * k + 1) * y * L[:, k] - k * L[:, k - 1]) / (k + 1)

        mat = loadmat('L.mat')
        mat_matrix = mat['L']
        comp_mat = L

        print(L.dtype)

        for i in range(comp_mat.shape[1]):
            diff = comp_mat[:,i]-mat_matrix[:,i]
            print("Max difference:", np.max(diff))
            print("Mean difference:", np.mean(diff))
        

        Lp = N2 * (L[:, N] - y * L[:, N + 1]) / (1 - y**2)
        y0 = y
        y = y0 - L[:, N + 1] / Lp
    
    # Map from [-1, 1] to [a, b]
    x = (a * (1 - y) + b * (1 + y)) / 2

    # Compute weights
    w = (b - a) / ((1 - y**2) * Lp**2) * (N2 / N1)**2
    return x, w

'''

import numpy as np

def lgwt_V4(N, a, b):
    N = N-1
    N1 = N+1
    N2 = N+2

    xu = np.linspace(-1,1,N1)

    y = np.cos((2*np.arange(0,N+1)+1)*np.pi/(2*N+2))+(0.27/N1)*np.sin(np.pi*xu*N/N2)

    L = np.zeros((N1,N2))
    Lp = np.zeros((N1,N2))

    y0 = 2

    while np.max(np.abs(y-y0)) > np.finfo(float).eps:
        L[:,0] = 1
        #Lp[:,0] = 0

        L[:,1] = y
        #Lp[:,1]=1

        for k in range(1,N1):
            L[:,k+1] = ((2 * k + 1) * y * L[:,k] -k * L[:,k-1]) / (k+1)
    
        Lp = N2 * ( L[:,N1-1]- y * L[:,N2-1]) / (1 - y**2)

        y0 = y
        y = y0 - L[:,N2-1]/Lp

    x = (a * (1 - y) + b*(1+y))/2

    w = (b-a)/((1 - y**2 )* Lp**2) * (N2/N1)**2

    #x = x.reshape(-1,1)
    #w = w.reshape(-1,1)

    '''
    mat = loadmat('x.mat')
    mat_matrix = mat['x']  # shape should be (35, 4721)
    comp_mat = x
    print(comp_mat.shape)
    print(mat_matrix.shape)  # should be (35, 4721)
    print("Same shape:", mat_matrix.shape == comp_mat.shape)

    # Element-wise difference
    diff = np.abs(mat_matrix - comp_mat)

    print("Max difference:", np.max(diff))
    print("Mean difference:", np.mean(diff))
    print("All close?:", np.allclose(mat_matrix, comp_mat, rtol=1e-5, atol=1e-8))
    '''
    return x, w