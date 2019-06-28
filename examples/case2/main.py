import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from flow import Flow

from concentration import Concentration
from reaction import Reaction

# ------------------------------------------------------------------------------#

def bc_flag_flow(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > 1 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels[in_flow + out_flow] = "dir"
    bc_val[b_faces[in_flow]] = 0
    bc_val[b_faces[out_flow]] = 10

    return labels, bc_val

# ------------------------------------------------------------------------------#

def bc_flag_solute(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > 1 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels[in_flow + out_flow] = "dir"
    bc_val[b_faces[in_flow]] = 0
    bc_val[b_faces[out_flow]] = 1

    return labels, bc_val

# ------------------------------------------------------------------------------#


def create_gb(file_name, mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    # create the mesh
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # assign the flag for the low permeable fractures
    return network.mesh(mesh_kwargs)

# ------------------------------------------------------------------------------#

def main():
    tol = 1e-6

    mesh_size = np.power(2., -2)

    end_time = 1
    num_steps = 20
    time_step = end_time / num_steps

    # the flow problem
    param_flow = {
        "tol": tol,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        "mass_weight": 1.0,
    }

    # the solute problem
    param_solute = {
        "tol": tol,
        "l": 1,
        "aperture": 1e-2, "lf_t": 1e2, "lf_n": 1e2,
        "mass_weight": 1.0/time_step,  # inverse of the time step
    }

    gb = create_gb("network.csv", mesh_size)
    gb.set_porepy_keywords()

    # -- darcy -- #

    # declare the modles
    model_flow = "flow"
    model_solute = "solute"
    model_precipitate = "precipitate"

    discr_flow = Flow(gb, model_flow, tol)

    # set the data
    discr_flow.set_data(param_flow, bc_flag_flow)

    # compute the matrices and rhs
    A_flow, M_flow, b_flow = discr_flow.matrix_rhs()

    # solve the linear system
    x_flow = sps.linalg.spsolve(A_flow, b_flow)

    # post processing
    discr_flow.extract(x_flow)
    discr_flow.export()

    # -- transport -- #

    # declare the solute and precipitate
    discr_conc = Concentration(gb, [model_solute, model_precipitate], tol)

    # set the data
    param_conc = {model_solute: param_solute, model_precipitate: param_solute}
    bc_flag_conc = {model_solute: bc_flag_solute, model_precipitate: bc_flag_solute}
    discr_conc.set_data(param_conc, bc_flag_conc)

    # set the flux from the flow equation
    discr_conc.set_flux(discr_flow.flux, discr_flow.mortar)

    # time loop
    x_conc = np.zeros(discr_conc.shape())
    for i in np.arange(num_steps):

        # do one step
        x_conc = discr_conc.one_step(x_conc, i)

        # post process
        discr_conc.extract(x_conc)
        discr_conc.export(i)

    discr_conc.export_pvd(np.arange(num_steps))

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------#
