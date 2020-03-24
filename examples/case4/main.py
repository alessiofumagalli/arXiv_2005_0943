import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from scheme import Scheme

from data import get_param

# ------------------------------------------------------------------------------#

import time as t
def time_cost(fct):
    start = t.time()
    fct()
    end = t.time()
    print(fct.__name__, str(end-start))

# ------------------------------------------------------------------------------#

def create_gb(file_name, mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    # assign the flag for the low permeable fractures
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}
    gb = network.mesh(mesh_kwargs)

    for _, d in gb:
        d[pp.PRIMARY_VARIABLES] = {}
        d[pp.DISCRETIZATION] = {}
        d[pp.DISCRETIZATION_MATRICES] = {}

    for _, d in gb.edges():
        d[pp.PRIMARY_VARIABLES] = {}
        d[pp.COUPLING_DISCRETIZATION] = {}
        d[pp.DISCRETIZATION_MATRICES] = {}

    return gb

# ------------------------------------------------------------------------------#

def main():

    mesh_size = np.power(2., -6)
    gb = create_gb("network.csv", mesh_size)

    param = get_param()

    scheme = Scheme(gb)
    scheme.set_data(param)

    # exporter
    save = pp.Exporter(gb, "case4", folder_name="solution")
    vars_to_save = scheme.vars_to_save()

    # post process
    scheme.extract()
    save.write_vtk(vars_to_save, time_step=0)

    for i in np.arange(param["time"]["num_steps"]):

        start = t.time()
        print("processing", i, "step")

        # -- DO THE FLOW PART -- #

        # update the data from the previous time step
        time_cost(scheme.update_flow)

        # solve the flow part
        time_cost(scheme.solve_flow)

        # -- DO THE TRANSPORT PART -- #

        # update the data from the previous time step
        time_cost(scheme.update_solute_precipitate)
        # solve the advection and diffusion part
        time_cost(scheme.solve_solute_precipitate_advection_diffusion)
        ## solve the reaction part
        time_cost(scheme.solve_solute_precipitate_rection)

        ## -- DO THE POROSITY PART -- #

        ## solve the porosity part
        time_cost(scheme.solve_porosity)

        ## solve the aperture part
        time_cost(scheme.solve_aperture)

        # post process
        time_cost(scheme.extract)
        time = param["time"]["step"]*(i+1)
        time_cost(lambda : save.write_vtk(vars_to_save, time_step=time))


    time = np.arange(param["time"]["num_steps"]+1)*param["time"]["step"]
    save.write_pvd(time)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------#
