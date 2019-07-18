import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")

from flow import Flow
from transport import Transport
from reaction import Reaction
from porosity import Porosity

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

    labels[in_flow] = "dir"
    labels[out_flow] = "dir"
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
    labels_adv[in_flow] = "dir"
    labels_adv[out_flow] = "dir"

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

def set_initial_condition(gb, discr_react):
    dof = np.cumsum(np.append(0, np.asarray(discr_react.assembler.full_dof)))

    values = np.zeros(dof[-1])
    for pair, bi in discr_react.assembler.block_dof.items():
        g = pair[0]
        if isinstance(g, pp.Grid):
            data = gb.node_props(g)
            values[dof[bi] : dof[bi + 1]] = 0.8*(np.logical_and(\
                                                np.abs(g.cell_centers[0, :] - 0.5) < 0.1,
                                                np.abs(g.cell_centers[1, :] - 0.5) < 0.1))
    return values

# ------------------------------------------------------------------------------#

def main():
    tol = 1e-6

    mesh_size = np.power(2., -3)

    end_time = 1
    num_steps = end_time * 100
    time_step = end_time / float(num_steps)

    gb = create_gb("network.csv", mesh_size)
    gb.set_porepy_keywords()

    # data problem
    param = {
        "tol": tol,
        "time_step": time_step,

        "aperture": 1e-2,

        # flow
        "k": 1,
        "kf_t": 1e4, "kf_n": 1e4,

        # transport of solute
        "l": 1e-6,
        "lf_t": 1e-6, "lf_n": 1e-6*1e2,
        "mass_weight": 1.0/time_step,

        # reaction of solute and precipitate
        "length": 1,
        "velocity": 1,
        "gamma_eq": 1,
        "theta": 0,
        "reaction": reaction_fct,
        "tol_reaction": 1e-12,
        "max_iter": 1e2,

        # porosity
        "eta": 1,
    }

    # -- darcy -- #

    # exporter
    save = pp.Exporter(gb, "case3", folder="solution")
    save_vars = ["pressure", "P0_darcy_flux", "solute", "precipitate", "porosity"]

    # -- flow part -- #
    discr_flow = Flow(gb)
    discr_flow.set_data(param, bc_flag_flow)

    # -- transport part -- #
    discr_solute = Transport(gb)
    discr_solute.set_data(param, bc_flag_solute)

    discr_react = Reaction()
    discr_react.set_data(param)

    # -- porosity part -- #
    discr_poro = Porosity()
    discr_poro.set_data(param)

    # time loop
    x_flow = np.zeros(discr_flow.shape())
    x_conc = np.zeros((2, discr_solute.shape()))
    x_poro = np.zeros(discr_solute.shape())

    x_conc[1, :] = set_initial_condition(gb, discr_solute)
    x_poro = 1 - x_conc[1, :] ##############

    # save the old solution
    x_conc_old = x_conc.copy()
    x_poro_old = x_poro.copy()
    x_poro_star = discr_poro.extrapolate(x_poro, x_poro_old)

    discr_solute.extract(x_conc[0, :], "solute")
    discr_solute.extract(x_conc[1, :], "precipitate")

    discr_solute.extract(x_poro, "porosity")
    discr_solute.extract(x_poro_old, "porosity_old")
    discr_solute.extract(x_poro_star, "porosity_star")

    # post process
    discr_flow.extract(np.zeros(discr_flow.shape()))
    save.write_vtk(save_vars, time_step=0)

    for i in np.arange(num_steps):

        print("processing", i, "step")

        # -- do the flow part -- #

        # udpate the flow parameters that depends on the precipitate
        discr_flow.update_data()

        # compute the matrices and rhs
        A_flow, b_flow = discr_flow.matrix_rhs()

        # solve the linear system
        x_flow = sps.linalg.spsolve(A_flow, b_flow)

        # -- do the transport part -- #

        # set the flux from the flow equation
        discr_solute.set_flux(discr_flow.flux, discr_flow.mortar)

        # update the data
        discr_solute.update_data()

        # compute the matrices and rhs
        A_conc, M_conc, b_conc = discr_solute.matrix_rhs()

        # solve the advective and diffusive step
        x_conc_old = x_conc.copy()
        # da controllare i termini modificati
        lhs_conc = M_conc * sps.diags(x_poro_star, 0) + A_conc
        rhs_conc = M_conc * sps.diags(x_poro_old, 0) *  x_conc[0, :] + b_conc
        x_conc[0, :] = sps.linalg.spsolve(lhs_conc, rhs_conc)

        # do one step
        x_conc = discr_react.step(x_conc)

        # -- do the porosity part -- #

        x_poro_old = x_poro.copy()
        x_poro = discr_poro.step(x_poro, x_conc[1, :], x_conc_old[1, :])
        x_poro_star = discr_poro.extrapolate(x_poro, x_poro_old)

        # post process
        discr_flow.extract(x_flow)

        discr_solute.extract(x_conc[0, :], "solute")
        discr_solute.extract(x_conc[1, :], "precipitate")

        discr_solute.extract(x_poro, "porosity")
        discr_solute.extract(x_poro_old, "porosity_old")
        discr_solute.extract(x_poro_star, "porosity_star")

        save.write_vtk(save_vars, time_step=time_step*(i+1))

    save.write_pvd(np.arange(num_steps+1)*time_step)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------#
