import numpy as np
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from reaction import Reaction

# reaction function, parameter: solute (u), precipitate (w), temperature (theta)
def reaction_fct(u, w, theta, tol=1e-15):
    r = np.power(u, 2)
    return lmbda(theta) * ((w>tol)*np.maximum(1 - r, 0) - np.maximum(r - 1, 0))

# temperature dependent ?????????
def lmbda(theta):
    return 8.37e-6*np.exp(-6e4/8.314/theta)

def main():

    # time loop
    num_steps = 5
    end_time = 2
    deltaT = end_time / float(num_steps)

    tol_react = 1e-12

    param = {
        "length": 1, # reference length
        "velocity": 1e-13, # reference Darcy velocity
        "gamma_eq": 0.16, # reference concentration of solute at equilibrium
        "theta": 100+273, # reference temperature (K)
        "reaction": reaction_fct # reaction functions
    }

    r = Reaction(tol_react)
    r.set_data(param)

    u = np.array([1, 0.2, 2])
    w = np.array([0.2, 0.01, 0.2])
    uw = np.stack((u, w))

    for _ in np.arange(num_steps):
        uw = r.step(uw, deltaT)
        print(uw)

if __name__ == "__main__":
    main()

