import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
from transport import Transport

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

def bc_flag_scalar(g, data, tol):
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

    mesh_size = 2e-2

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

    # the scalar problem
    param_scalar = {
        "tol": tol,
        "l": 1,
        "aperture": 1e-2, "lf_t": 1e2, "lf_n": 1e2,
        "mass_weight": 1.0/time_step,  # inverse of the time step
    }

    gb = create_gb("network.csv", mesh_size)

    # -- darcy -- #

    # declare the flow
    model_flow = "flow"
    model_scalar = "scalar"

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

    # declare the scalar
    discr_scalar = Transport(gb, model_scalar, tol)

    # set the data
    discr_scalar.set_data(param_scalar, bc_flag_scalar)
    discr_scalar.set_flux(discr_flow.flux, discr_flow.mortar)

    # compute the matrices and rhs
    A_scalar, M_scalar, b_scalar = discr_scalar.matrix_rhs()

    # time loop
    x_scalar = np.zeros(b_scalar.size)
    for i in np.arange(num_steps):
        rhs = M_scalar * x_scalar + b_scalar
        x_scalar = sps.linalg.spsolve(M_scalar + A_scalar, rhs)

        # post process
        discr_scalar.extract(x_scalar)
        discr_scalar.export(i)

    discr_scalar.export_pvd(np.arange(num_steps))

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------#
