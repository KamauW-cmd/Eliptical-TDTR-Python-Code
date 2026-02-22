import numpy as np
from scipy.io import loadmat

data_py = loadmat('TDTR_Var_B_fixed.mat')
vars_py = {k: v for k, v in data_py.items() if not k.startswith('__')}

data_mat = loadmat('TDTR_vars_B.mat')
vars_mat = {k: v for k, v in data_mat.items() if not k.startswith('__')}

max_diff = {}
mean_diff = {}
check_diff = {}
off_maxdiff = {}
off_meandiff = {}


for name in vars_py:
    if name in  ['pi', 'mat','mat_matrix','comp_mat','diff','test_split','Bp_down', 'Bm_down', 'un_minus','r_pump_max','Xh']:
        continue
    print(name)
    if name in vars_mat:
        pyt = vars_py[name]
        mat = vars_mat[name]

        if np.isscalar(pyt):
            print('Scalar',name)
        elif len(pyt.shape) > 2:
            for i in range(len(pyt.shape)):
                if pyt.shape[i] == 1:
                    pop = i
            pyt = np.squeeze(pyt, axis = i)

        elif mat.shape != pyt.shape:
            pyt = pyt.reshape(-1,1)
        # Element-wise difference

        diff = np.abs(pyt - mat)

        max_diff[name] = np.max(diff)
        mean_diff[name] = np.mean(diff)

        if np.mean(diff) < 1e-9 and np.max(diff) < 1e-9:
            check_diff[name] = 1
        else:
            check_diff[name] = 0
            off_maxdiff[name] = np.max(diff)
            off_meandiff[name] = np.mean(diff)


    print(np.allclose(pyt, mat, rtol=1e-6, atol=1e-8))
        

print(max_diff)
print(mean_diff)

'''
mat = loadmat('weights.mat')
mat_matrix = mat['weights']  # shape should be (35, 4721)
comp_mat = weights.reshape(-1,1)

print('weights')

print(mat_matrix.shape)  # should be (35, 4721)
print(comp_mat.shape)
print("Same shape:", mat_matrix.shape == comp_mat.shape)

# Element-wise difference
diff = np.abs(mat_matrix -comp_mat)

print("Max difference:", np.max(diff))
print("Mean difference:", np.mean(diff))
print("All close?:", np.allclose(mat_matrix, comp_mat, rtol=1e-5, atol=1e-8))
'''



