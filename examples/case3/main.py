import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")

from flow import Flow
from transport import Transport
from reaction import Reaction

# ------------------------------------------------------------------------------#

# reaction function, parameter: solute (u), precipitate (w), temperature (theta)
def reaction_fct(u, w, theta, tol=1e-15):
    r = np.power(u, 1)
    return (w>tol)*np.maximum(1 - r, 0) - np.maximum(r - 1, 0)

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
    bc_val[b_faces[in_flow]] = 0.5
    bc_val[b_faces[out_flow]] = 0

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
    labels_diff = np.array(["neu"] * b_faces.size)
    labels_adv = np.array(["neu"] * b_faces.size)

    labels_diff[in_flow] = "dir"
    labels_adv[in_flow + out_flow] = "dir"

    bc_val = np.zeros(g.num_faces)
    bc_val[b_faces[in_flow]] = 0
    bc_val[b_faces[out_flow]] = 0

    return labels_diff, labels_adv, bc_val

# ------------------------------------------------------------------------------#


def create_gb(file_name, mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    constraints = {"points": np.array([[0.4, 0.4, 0.6, 0.6],
                                       [0.4, 0.6, 0.6, 0.4]]),
                   "edges": np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).T}
    # create the mesh
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # assign the flag for the low permeable fractures
    return network.mesh(mesh_kwargs, constraints=constraints)

# ------------------------------------------------------------------------------#

def main():
    tol = 1e-6
    tol_react = 1e-12

    mesh_size = np.power(2., -6)

    end_time = 1
    num_steps = 100
    time_step = end_time / float(num_steps)

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

    # the reaction part
    param_react = {
        "length": 1,
        "velocity": 1,
        "gamma_eq": 1,
        "theta": 0,
        "reaction": reaction_fct,
    }

    gb = create_gb("network.csv", mesh_size)
    gb.set_porepy_keywords()

    # -- darcy -- #

    # declare the modles
    model_flow = "flow"
    model_solute = "solute"
    model_reaction = "reaction"

    # exporter
    save = pp.Exporter(gb, "case2", folder="solution")
    save_vars = ["pressure", "P0_darcy_flux", "solute", "precipitate"]

    # -- flow part -- #

    # declare the flow
    discr_flow = Flow(gb, model_flow, tol)

    # set the data
    discr_flow.set_data(param_flow, bc_flag_flow)

    # -- transport part -- #

    # declare the solute and precipitate
    discr_solute = Transport(gb, model_solute, tol)
    discr_react = Reaction(tol_react)

    # set the data
    discr_solute.set_data(param_solute, bc_flag_solute)
    discr_react.set_data(param_react)

    # time loop
    x_conc = np.ones((2, discr_solute.shape()))

    for g, _ in gb:
        if g.dim == 2:
            x_conc[1, :g.num_cells] = 0.8*(np.logical_and(np.abs(g.cell_centers[0, :] - 0.5) < 0.1,
                                                          np.abs(g.cell_centers[1, :] - 0.5) < 0.1))
    x_conc_old = x_conc.copy()

    # post process
    discr_flow.extract(np.zeros(discr_flow.shape()))
    discr_solute.extract(x_conc[0, :], "solute")
    discr_solute.extract(x_conc[1, :], "precipitate")

    save.write_vtk(save_vars, time_step=0)

    for i in np.arange(num_steps):

        print("processing", i, "step")

        # -- do the flow part -- #

        # udpate the flow parameters that depends on the precipitate
        param_flow = {
            "k": np.power(1 - x_conc[1, :], 2),
            "source": (x_conc[1, :] - x_conc_old[1, :])/time_step,
            "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
        }
        discr_flow.update_data(param_flow)

        # compute the matrices and rhs # FORSE NON SERVE LA M
        A_flow, M_flow, b_flow = discr_flow.matrix_rhs()

        # solve the linear system
        x_flow = sps.linalg.spsolve(A_flow, b_flow)

        # set the flux from the flow equation
        discr_solute.set_flux(discr_flow.flux, discr_flow.mortar)

        # -- do the transport part -- #

        # compute the matrices and rhs
        A_conc, M_conc, b_conc = discr_solute.matrix_rhs()

        # solve the advective and diffusive step
        x_conc_old = x_conc.copy()
        x_conc[0, :] = sps.linalg.spsolve(M_conc + A_conc, M_conc * x_conc[0, :] + b_conc)

        # do one step
        x_conc = discr_react.step(x_conc, time_step)

        # post process
        discr_flow.extract(x_flow)
        discr_solute.extract(x_conc[0, :], "solute")
        discr_solute.extract(x_conc[1, :], "precipitate")

        save.write_vtk(save_vars, time_step=time_step*(i+1))

    save.write_pvd(np.arange(num_steps+1)*time_step)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------#
