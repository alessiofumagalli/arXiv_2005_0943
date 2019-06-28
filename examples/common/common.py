import numpy as np

# reaction function, parameter: solute (u), precipitate (w), temperature (theta)
def reaction_fct(u, w, theta, tol=1e-15):
    r = np.power(u, 2)
    return lmbda(theta) * ((w>tol)*np.maximum(1 - r, 0) - np.maximum(r - 1, 0))

# temperature dependent ?????????
def lmbda(theta):
    return 8.37e-6*np.exp(-6e4/8.314/theta);

