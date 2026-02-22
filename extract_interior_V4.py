import numpy as np

def extract_interior_V4(t_data, f_data, t_min, t_max):
    """
    Extracts all points in range t_min <= t_data <= t_max, and corresponding values in f_data.
    
    Parameters:
        t_data (np.ndarray): 1D array of time delay values
        f_data (np.ndarray): 1D array of function values corresponding to t_data
        t_min (float): lower bound of time delay range
        t_max (float): upper bound of time delay range
        
    Returns:
        t_interior (np.ndarray): time delays within the range [t_min, t_max]
        f_interior (np.ndarray): corresponding values from f_data
    """
    mask = (t_data >= t_min) & (t_data <= t_max)
    t_interior = t_data[mask]
    f_interior = f_data[mask]
    return t_interior, f_interior
