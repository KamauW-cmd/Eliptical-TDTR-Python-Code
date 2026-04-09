import numpy as np
import matplotlib.pyplot as plt
from Integrand import conduction_ratio_uv
from Integrator import Integrator
import sys

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

    layer_props[0,4] = 2.492e6 #129 * (19.32 * 1000)
    layer_props[1,4] = 1.62e6 #710 * (2.33 * 1000)

    layer_props[0,5] = 19.32
    layer_props[1,5] = 2.33



    G = X[0]/h[1]
    interface_props = np.array([G])

    for i in range(1,4):
        layer_props[1,i] = X[1]
        layer_props[0,i] = 315 #X[0]

    

    pump_props = np.array([r_pump,r_pump])
    probe_props = np.array([r_probe, r_probe])

    # Call model function (you must define this elsewhere)
    #Tin_model, Tout_model = Integrator(layer_props.shape[0],nnodes, layer_props, interface_props, pump_props, probe_props, f, tau_rep, tdelay)

    _,Ratio_model = Integrator(tdelay, tau_rep, f, Lambda, C, h, eta,r_pump, r_probe, P_pump, nnodes, X_heat, X_temp, pump_props, probe_props, layer_props, interface_props)

    #print(Ratio_data.shape())

    # Absorption-weighted temperature response
    
    
    #Tin_model = np.real(Ts) # @ AbsProf / (np.ones(AbsProf.shape).T @ AbsProf)
    #Tout_model = np.imag(Ts) # @ AbsProf / (np.ones(AbsProf.shape).T @ AbsProf)
    #Ratio_model = -Tin_model / Tout_model

    '''
    correct_arr = np.load("correct_ratio_model.npy")
    count = True

    diff = np.abs(Ratio_model - correct_arr)

    print("model shape: ",Ratio_model.shape)
    print("correct shape", correct_arr.shape)
    print("diff: min/mean/max =", diff.min(), diff.mean(), diff.max())
    print("model: min/mean/max =", Ratio_model.min(), Ratio_model.mean(), Ratio_model.max())
    print("correct: min/mean/max =", correct_arr.min(), correct_arr.mean(), correct_arr.max())

    print("mean(model-correct) =", np.mean(Ratio_model - correct_arr))
    print("std(model-correct)  =", np.std(Ratio_model - correct_arr))

    rel = diff / (np.maximum(np.abs(correct_arr), 1e-12))
    print("rel err: min/mean/max =", rel.min(), rel.mean(), rel.max())

    sys.exit()
    '''

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

    model_matrix = Ratio_model #-np.real(Ts) / np.imag(Ts)  # shape: (len(tdelay), k)

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
