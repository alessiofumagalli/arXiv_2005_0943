import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from scheme import Scheme

from data import get_param
from soultz import soultz

# ------------------------------------------------------------------------------#

def create_gb(tol):

    domain = {"xmin": -1200, "xmax": 500, "ymin": -600,
              "ymax": 600, "zmin": 600, "zmax": 5500}

    # some fractures create problem in the meshing
    fracs = np.hstack((np.arange(20), [22], np.arange(25, 39)))
    network = soultz(fracs=fracs, num_points=10, domain=domain, tol=tol)

    # create the mesh
    mesh_kwargs = {"mesh_size_frac": 75, "mesh_size_bound": 200,
                   "mesh_size_min": 10, "meshing_algorithm": 4, "tol": tol}

    # assign the flag for the low permeable fractures
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

    tol = 1e-6
    gb = create_gb(tol)
    save = pp.Exporter(gb, "case6", folder_name="grid")
    save.write_vtk()
    a=e

    param = get_param()

    scheme = Scheme(gb)
    scheme.set_data(param)

    # exporter
    save = pp.Exporter(gb, "case6", folder_name="solution")
    vars_to_save = scheme.vars_to_save()

    # post process
    scheme.extract()
    save.write_vtk(vars_to_save, time_step=0)

    for i in np.arange(param["time"]["num_steps"]):

        print("processing", i, "step")

        # -- DO THE FLOW PART -- #

        # update the data from the previous time step
        scheme.update_flow()
        # solve the flow part
        scheme.solve_flow()

        # -- DO THE TRANSPORT PART -- #

        # update the data from the previous time step
        scheme.update_solute_precipitate()
        # solve the advection and diffusion part
        scheme.solve_solute_precipitate_advection_diffusion()
        ## solve the reaction part
        scheme.solve_solute_precipitate_rection()

        ## -- DO THE POROSITY PART -- #

        ## solve the porosity part
        scheme.solve_porosity()

        ## solve the aperture part
        scheme.solve_aperture()

        # post process
        scheme.extract()
        time = param["time"]["step"]*(i+1)
        save.write_vtk(vars_to_save, time_step=time)

    time = np.arange(param["time"]["num_steps"]+1)*param["time"]["step"]
    save.write_pvd(time)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------#
