import numpy as np

# ------------------------------------------------------------------------------#

def get_param():
    # data problem

    tol = 1e-6
    end_time = 2
    num_steps = end_time * 20
    time_step = end_time / float(num_steps)

    return {
        "tol": tol,

        "time": {
            "end_time": end_time,
            "num_steps": num_steps,
            "step": time_step
        },

        # porosity
        "porosity": {
            "eta": 0.5,
            "initial": initial_porosity
        },

        # aperture
        "aperture": {
            "eta": 1,
            "initial": initial_aperture
        },

        # flow
        "flow": {
            "tol": tol,
            "k": 1,
            "k_t": 1e6, "k_n": 1e6,
            "bc": bc_flow,
        },

        # advection and diffusion of solute
        "solute_advection_diffusion": {
            "tol": tol,
            "l": 1e-6,
            "l_t": 1e-6, "l_n": 1e-6,
            "mass_weight": 1.0,
            "bc": bc_solute,
            "initial_solute": initial_solute,
            "initial_precipitate": initial_precipitate,
        },

        # reaction of solute and precipitate
        "solute_precipitate_reaction": {
            "tol": tol,
            "length": 1,
            "velocity": 1,
            "gamma_eq": 1,
            "theta": 0,
            "reaction": reaction_fct,
            "tol_reaction": 1e-12,
            "max_iter": 1e2,
        },

        # temperature
        "temperature": {
            "initial": initial_temperature
        },

    }

# ------------------------------------------------------------------------------#

# reaction function, parameter: solute (u), precipitate (w), temperature (theta)
def reaction_fct(u, w, theta, tol=1e-15):
    l = 1+0*100*np.exp(-4/theta)
    r = np.power(u, 1)
    return l*((w>tol)*np.maximum(1 - r, 0) - np.maximum(r - 1, 0))

# ------------------------------------------------------------------------------#

def bc_flow(g, data, tol):
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
    bc_val[b_faces[in_flow]] = 1
    bc_val[b_faces[out_flow]] = 0

    return labels, bc_val

# ------------------------------------------------------------------------------#

def bc_solute(g, data, tol):
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

def initial_solute(g, data, tol):
    solute = np.zeros(g.num_cells)
    return solute

# ------------------------------------------------------------------------------#

def initial_precipitate(g, data, tol):
    x, y = g.cell_centers[0, :], g.cell_centers[1, :]
    precipitate = 0.8*(np.logical_and(np.abs(x-0.5) < 0.1, np.abs(y-0.5) < 0.1))
    return precipitate

# ------------------------------------------------------------------------------#

def initial_porosity(g, data, tol):
    if g.dim == 2: # NOTE: we are in 2d as dim_max
        x, y = g.cell_centers[0, :], g.cell_centers[1, :]
        return 1-0.8*(np.logical_and(np.abs(x-0.5) < 0.1, np.abs(y-0.5) < 0.1))
    else:
        # we set a zero porosity, meaning it is not active anymore
        # the temporal scheme considered keeps this variable null
        return np.zeros(g.num_cells)

# ------------------------------------------------------------------------------#

def initial_aperture(g, data, tol):
    if g.dim == 2: # NOTE: we are in 2d as dim_max
        # we set a zero aperture, meaning it is not active
        # the temporal scheme considered keeps this variable null
        return np.zeros(g.num_cells)
    else:
        return 1e-2*np.ones(g.num_cells)

# ------------------------------------------------------------------------------#

def initial_temperature(g, data, tol):
    return np.zeros(g.num_cells)

# ------------------------------------------------------------------------------#
