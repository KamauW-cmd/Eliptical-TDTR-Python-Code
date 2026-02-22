import numpy as np
from numpy.polynomial.hermite import hermgauss

ii =1j

def lambdas(s,r,C,w,Kx,Ky,Kz,Kxy,Kyz,Kxz,wx2,wy2):
    wx = np.sqrt(wx2)
    wy = np.sqrt(wy2)
    temp_11 = (ii*C*w)/Kz 
    temp_12 = (4/Kz)*((Kx/wx2)*s**2 + ((2*Kxy)/(wx*wy))*s*r+ (Ky/wy2)*r**2)
    lambda_1 = temp_11 + temp_12
    temp_21 = (4*ii)/Kz
    temp_22 = (Kxz*s)/wx + (Kyz*r)/wy
    lambda_2 = temp_21*temp_22

    return lambda_1,lambda_2

def u_s(lambda_1,lambda_2):
    uplus = (-lambda_2 + np.sqrt(lambda_2**2+4*lambda_1))/2
    uminus = (-lambda_2 - np.sqrt(lambda_2**2+4*lambda_1))/2
    if np.real(uplus)<np.real(uminus):
        uplus,uminus = uminus,uplus
    return uplus,uminus

def N_mat(Kz,uplus,uminus,z):
    temp_n1 = np.array([[1,1],[-Kz*uplus, -Kz*uminus]])
    temp_n2 = np.array([[np.exp(uplus*z), 0],[0, np.exp(uminus*z)]])
    N = temp_n1@temp_n2
    return N

def M_mat(Kz,uplus,uminus):
    temp_m1 = 1/(Kz*(uplus-uminus))
    temp_m2 = np.array([[-Kz*uminus, -1],[Kz*uplus, 1]])
    M = temp_m1*temp_m2

    return M

def R_mat(G):
    R = np.array([[1,-1/G],[0,1]])
    return R

#kvectin, freq, Lambda, C, h, eta,r_pumpx,r_pumpy,r_probex, r_probey, P_pump, X_heat, X_temp

def H_mat(s,r,C,freq, Lambda,G, h, eta,r_pumpx,r_pumpy,r_probex, r_probey, P_pump, X_heat, X_temp,num_layers):
    
    w = 2*np.pi*freq
    pi2 = np.pi**2
    wx2 = (r_pumpx**2+r_probex**2)/2
    wy2 = (r_pumpy**2+r_probey**2)/2

    for n in range(num_layers):
        if n == 0:
            z = h[n]
            C_f = C[n]
            Kx,Ky,Kz,Kxy,Kyz,Kxz = Lambda[n]
            lambda_1,lambda_2 = lambdas(s,r,C_f,w,Kx,Ky,Kz,Kxy,Kyz,Kxz,wx2,wy2)
            uplus,uminus = u_s(lambda_1,lambda_2)
            N1 = N_mat(Kz,uplus,uminus,z)
            M1 = M_mat(Kz,uplus,uminus)
            R1 = R_mat(G)
            
        else:
            C_s = C[n]
            Kx,Ky,Kz,Kxy,Kyz,Kxz = Lambda[n]
            lambda_1,lambda_2 = lambdas(s,r,C_s,w,Kx,Ky,Kz,Kxy,Kyz,Kxz,wx2,wy2)
            uplus,uminus = u_s(lambda_1,lambda_2)
            #N2 = N_mat(Kz,uplus,uminus,z)
            M2 = M_mat(Kz,uplus,uminus)

    T = M2 @ R1 @ N1 @ M1
    Cmat,D = T[1,0],T[1,1]

    H = -D/Cmat

    return H


    
def Integrand(N,C,freq, Lambda,G, h, eta,r_pumpx,r_pumpy,r_probex, r_probey, P_pump, X_heat, X_temp, num_layers):
    ii = 1j
    pi2 = np.pi**2

    wx2 = (r_pumpx**2+r_probex**2)/2
    wy2 = (r_pumpy**2+r_probey**2)/2

    wx = np.sqrt(wx2)
    wy = np.sqrt(wy2)

    s_nodes,w_s = hermgauss(N)
    r_nodes,w_r = hermgauss(N)
    Integrand = 0.0+0.0j

    Jacobian = 1/(pi2*wx*wy)

    for si,s in enumerate(s_nodes):
        for rj,r in enumerate(r_nodes):
            H = H_mat(s,r,C,freq, Lambda,G, h, eta,r_pumpx,r_pumpy,r_probex, r_probey, P_pump, X_heat, X_temp,num_layers)
            Integrand += w_s[si]*w_r[rj]*H


    return Jacobian*Integrand



