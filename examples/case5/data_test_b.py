import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#

def create_gb(mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    file_name = "network_split_with_constraints.csv"
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

def get_param():
    # data problem

    tol = 1e-6
    end_time = 5
    num_steps = int(end_time * 4)
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
            "eta": 2,
            "initial": initial_aperture
        },

        # flow
        "flow": {
            "tol": tol,
            "k": 1,
            "k_t": 1e8, "k_n": 1e8,

            "bc": bc_flow,
        },

        # temperature
        "temperature": {
            "tol": tol,
            "l_w": 10.,
            "l_s": 1e-1,
            "rc_w": 10, # rho_w \cdot c_w
            "rc_s": 1.0,
            "mass_weight": 1.0,

            "bc": bc_temperature,
            "initial": initial_temperature,
        },

        # advection and diffusion of solute
        "solute_advection_diffusion": {
            "tol": tol,
            "d": 1e-1,
            "d_t": 1e4, "d_n": 1e4,
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
            "tol_consider_zero": 1e-30,
            "max_iter": 1e2,
        },

    }

# ------------------------------------------------------------------------------#

# reaction function, parameter: solute (u), precipitate (w), temperature (theta)
def reaction_fct(u, w, theta, tol=1e-15):
    l = 10*np.exp(-4/theta)
    r = np.power(u, 2)
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

def bc_temperature(g, data, tol):
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
    bc_val[b_faces[in_flow]] = 1.5
    bc_val[b_faces[out_flow]] = 0

    return labels_diff, labels_adv, bc_val

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

def initial_temperature(g, data, tol):
    temperature = np.ones(g.num_cells)
    return temperature

# ------------------------------------------------------------------------------#

def initial_solute(g, data, tol):
    solute = 0 * np.ones(g.num_cells)
    return solute

# ------------------------------------------------------------------------------#

def initial_precipitate(g, data, tol):
    if g.dim == 1: #is_flag(g):
        precipitate = 10*np.ones(g.num_cells)
    else:
        precipitate = np.zeros(g.num_cells)
    return precipitate

# ------------------------------------------------------------------------------#

def initial_porosity(g, data, tol):
    if g.dim == 2: # NOTE: we are in 2d as dim_max
        return 0.2
    else:
        # we set a zero porosity, meaning it is not active anymore
        # the temporal scheme considered keeps this variable null
        return np.zeros(g.num_cells)

# ------------------------------------------------------------------------------#

def initial_aperture(g, data, tol):
    if g.dim == 2: # NOTE: we are in 2d as dim_max
        # we set a zero aperture, meaning it is not active
        # the temporal scheme considered keeps this variable null
        aperture = np.zeros(g.num_cells)
    else:
        aperture = 1e-2*np.ones(g.num_cells)
#        if is_flag(g):
#            aperture = 1e-3*np.ones(g.num_cells)
#        else:
#            aperture = 1e-2*np.ones(g.num_cells)
    return aperture

# ------------------------------------------------------------------------------#

def is_flag(g):
    tol = 1e-3
    # set the key for the low peremable fractures
    if g.dim == 1:

#        f_0 = (g.nodes[0, :] - 0.05)/(0.2200 - 0.05) - (g.nodes[1, :] - 0.4160)/(0.0624 - 0.4160)
#        if np.sum(np.abs(f_0)) < tol:
#            d["frac_num"] = 0
#            d["k_t"] = 1e4
#
#        f_1 = (g.nodes[0, :] - 0.05)/(0.2500 - 0.05) - (g.nodes[1, :] - 0.2750)/(0.1350 - 0.2750)
#        if np.sum(np.abs(f_1)) < tol:
#            d["frac_num"] = 1
#            d["k_t"] = 1e4
#
#        f_2 = (g.nodes[0, :] - 0.15)/(0.4500 - 0.15) - (g.nodes[1, :] - 0.6300)/(0.0900 - 0.6300)
#        if np.sum(np.abs(f_2)) < tol:
#            d["frac_num"] = 2
#            d["k_t"] = 1e4

        f_3 = (g.nodes[0, :] - 0.15)/(0.4 - 0.15) - (g.nodes[1, :] - 0.9167)/(0.5 - 0.9167)
        if np.sum(np.abs(f_3)) < tol:
            return True

        f_4 = (g.nodes[0, :] - 0.65)/(0.849723 - 0.65) - (g.nodes[1, :] - 0.8333)/(0.167625 - 0.8333)
        if np.sum(np.abs(f_4)) < tol:
            return True

#        f_5 = (g.nodes[0, :] - 0.70)/(0.849723 - 0.70) - (g.nodes[1, :] - 0.2350)/(0.167625 - 0.2350)
#        if np.sum(np.abs(f_5)) < tol:
#            d["frac_num"] = 5
#            d["k_t"] = 1e4
#
#        f_6 = (g.nodes[0, :] - 0.60)/(0.8500 - 0.60) - (g.nodes[1, :] - 0.3800)/(0.2675 - 0.3800)
#        if np.sum(np.abs(f_6)) < tol:
#            d["frac_num"] = 6
#            d["k_t"] = 1e4
#
#        f_7 = (g.nodes[0, :] - 0.35)/(0.8000 - 0.35) - (g.nodes[1, :] - 0.9714)/(0.7143 - 0.9714)
#        if np.sum(np.abs(f_7)) < tol:
#            d["frac_num"] = 7
#            d["k_t"] = 1e4
#
#        f_8 = (g.nodes[0, :] - 0.75)/(0.9500 - 0.75) - (g.nodes[1, :] - 0.9574)/(0.8155 - 0.9574)
#        if np.sum(np.abs(f_8)) < tol:
#            d["frac_num"] = 8
#            d["k_t"] = 1e4
#
#        f_9 = (g.nodes[0, :] - 0.15)/(0.4000 - 0.15) - (g.nodes[1, :] - 0.8363)/(0.9727 - 0.8363)
#        if np.sum(np.abs(f_9)) < tol:
#            d["frac_num"] = 9
#            d["k_t"] = 1e4

    return False

# ------------------------------------------------------------------------------#
