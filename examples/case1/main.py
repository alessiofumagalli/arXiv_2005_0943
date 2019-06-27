import numpy as np
import sys; sys.path.insert(0, "../../src/")

import porepy as pp

from reaction import Reaction


# temperature dependent ?????????
def lmbda(theta):
    return 8.37e-6*np.exp(-6e4/8.314/theta);

# reaction function, parameter: solute (u), precipitate (w), temperature (theta)
def reaction_fct(u, w, theta):
    r = np.power(u, 2)
    #rint(w) <- nel caso di Anna qui ho un 1e-17 mentre io ho 0 e quindi vengo punito
    return lmbda(theta) * ((w>1e-15)*np.maximum(1 - r, 0) - np.maximum(r - 1, 0))

def set_data():
    data = {}

    # reference name of variable
    data["solute"] = "solute"
    data["precipitate"] = "precipitate"

    # reference length
    reference = {}
    reference["length"] = 1
    # reference Darcy velocity
    reference["velocity"] = 1e-13
    # reference concentration of solute at equilibrium
    reference["gamma_eq"] = 0.16
    # reference temperature (K)
    reference["theta"] = 100+273 #pp.CELSIUS_to_KELVIN(100)
    data["reference"] = reference

    # reaction functions
    data["reaction"] = reaction_fct

    # initial datum
    #gamma_0=@(x) 0*(x>L/3).*(x<2*L/3);
    #C_0=@(x) 4*(x>L/3).*(x<2*L/3);

    #Nx=100;

    # time loop
    data["num_steps"] = 5
    data["end_time"] = 2
    data["deltaT"] = 0.002 #data["end_time"]/data["num_steps"]

    #xx=[1/Nx*0.5: 1/Nx: 1-0.5/Nx];
    #dif=1.0e-13;

    #Vm=0.0226;

    return data

def main():

    data = set_data()
    r = Reaction(**data)

    u = np.array([1, 0.2, 2])
    w = np.array([0.2, 0.01, 0.2])
    uw = np.stack((u, w))

    for _ in np.arange(data["num_steps"]):
        uw = r.step(uw, data["deltaT"])
        print(uw)

if __name__ == "__main__":
    main()
