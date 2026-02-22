import numpy as np
import matplotlib.pyplot as plt
from Integrand import conduction_ratio_uv
from Integrator import Integrator

#from Sub_B import TDTR_Bidirectional_SUB_B

def TDTR_Bidirectional_SUB_C(X, Ratio_data, tdelay, tau_rep, f,
                              Lambda, C, h, eta, r_pump, r_probe,
                              P_pump, nnodes,
                              FITNLambda, FITNC, FITNh,
                              X_heat, X_temp, AbsProf):
        
    # Update parameters based on fit indices

    for i, idx in enumerate(FITNLambda):
        Lambda[idx-1] = X[i]

    for i, idx in enumerate(FITNC):
        C[idx] = X[len(FITNLambda) + i]

    for i, idx in enumerate(FITNh):
        h[idx] = X[len(FITNLambda) + len(FITNC) + i]

    layer_props = np.zeros((2,6))

    layer_props[0,0] = 133e-9
    layer_props[1,0] = 1e-3

    layer_props[0,4] = 129
    layer_props[1,4] = 710

    layer_props[0,5] = 19.32
    layer_props[1,5] = 2.33

    for i in range(3):
        layer_props[1,i] = X[1]
        layer_props[0,i] = X[0]

    #layer_props = np.array([[133e-9, 310, 310, 310, 129, 19.32],
    #                  [1e-3, 149, 149, 149 ,710 , 2.33]],dtype = float)
    

    pump_props = np.array([r_pump,r_pump])
    probe_props = np.array([r_probe, r_probe])

    # Call model function (you must define this elsewhere)
    Ts,_ = Integrator(layer_props.shape[0],nnodes, layer_props, [0.3], pump_props, probe_props, f, tau_rep, tdelay)

    #print(Ratio_data.shape())

    # Absorption-weighted temperature response
    print(Ts.shape)
    print(AbsProf.shape)
    Tin_model = np.real(Ts) # @ AbsProf / (np.ones(AbsProf.shape).T @ AbsProf)
    Tout_model = np.imag(Ts) # @ AbsProf / (np.ones(AbsProf.shape).T @ AbsProf)
    Ratio_model = -Tin_model / Tout_model


    # Residuals and error metric
    res = ((Ratio_model - Ratio_data) / Ratio_model)**2
    Z = np.sqrt(np.sum(res)) / len(res)

    # Debug print
    print(f"Z = {Z}")
    print("X =", X)

    # Plotting
    '''
    plt.figure(10)
    plt.clf()
    plt.semilogx(tdelay, -np.real(Ts) / np.imag(Ts), 'g', label='Model - Re/Im')
    plt.semilogx(tdelay, Ratio_data, 'ob', label='Experimental Ratio')
    plt.semilogx(tdelay, Ratio_model, 'k', label='Modeled Ratio')
    plt.xlabel('Time Delay (ps)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.pause(0.01)
    '''

    # Plotting
    plt.figure(10)
    plt.clf()

    model_matrix = -np.real(Ts) / np.imag(Ts)  # shape: (len(tdelay), k)

    # Draw all green lines but label only the first one
    lines = plt.semilogx(tdelay, model_matrix, 'g', linewidth=0.8)
    lines[0].set_label('Model - Re/Im')        # one legend entry only

    plt.semilogx(tdelay, Ratio_data, 'ob', label='Experimental Ratio')
    plt.semilogx(tdelay, Ratio_model, 'k', label='Modeled Ratio')

    plt.xlabel('Time Delay (ps)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.pause(0.01)
    return Z, Ratio_model
