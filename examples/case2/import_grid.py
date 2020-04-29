import glob
import warnings
import os
import numpy as np
from scipy import sparse as sps

import porepy as pp

def import_grid(folder, fname, dim, **kwargs):

    index_shift = kwargs.get("index_shift", 1)

    # load the nodes
    fnodes_root = kwargs.get("nodes", "points")
    fnodes = folder + "/" + fnodes_root + fname + ".txt"
    nodes = np.loadtxt(fnodes, dtype=np.float)
    nodes = np.atleast_2d(nodes).T
    nodes = np.vstack([nodes[1:], np.zeros(nodes.shape[1])])

    # load the face to node map
    fface_nodes_root = kwargs.get("face_nodes", "face2node")
    fface_nodes = folder + "/" + fface_nodes_root + fname + ".txt"
    face_nodes = np.loadtxt(fface_nodes, dtype=np.int).T

    col = face_nodes[0] - index_shift
    row = face_nodes[1] - index_shift
    data = face_nodes[2].astype(np.bool)
    face_nodes = sps.csc_matrix((data, (row, col)))

    # load the cell to face map
    fcell_faces_root = kwargs.get("cell_faces", "cell2face")
    fcell_faces = folder + "/" + fcell_faces_root + fname + ".txt"
    cell_faces = np.loadtxt(fcell_faces, dtype=np.int).T

    col = cell_faces[0] - index_shift
    row = cell_faces[1] - index_shift
    data = cell_faces[2].astype(np.float)
    cell_faces = sps.csc_matrix((data, (row, col)))

    # it's not really triangular grid but it's useful somewhere
    name = "TriangleGrid" if dim == 2 else "TensorGrid"
    g = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
    g.name.append(fname)
    g.compute_geometry()

    # load the point to local to global map
    fglobal_root = kwargs.get("global", "glob2loc")
    fglobal = folder + "/" + fglobal_root + fname + ".txt"
    if glob.glob(fglobal):
        g.global_point_ind = np.loadtxt(fglobal, dtype=np.int).T[-1]
    else:
        g.global_point_ind = np.arange(g.num_nodes) + 1
        warnings.warn("Attention default global to local mapping used for grid\n" + str(g))

    return g

def import_grid_0d(folder, fname, **kwargs):

    # load the node
    fnodes_root = kwargs.get("nodes", "InterCellsCoord")
    fnodes = folder + "/" + fnodes_root + fname + ".txt"
    nodes = np.loadtxt(fnodes, dtype=np.float)
    nodes = np.atleast_2d(nodes).T
    nodes = np.vstack([nodes, np.zeros(nodes.shape[1])])

    name = "PointGrid"
    g = pp.PointGrid(nodes, name)
    g.compute_geometry()

    # load the point to local to global map
    fglobal_root = kwargs.get("global", "InterCellsGlobalID")
    fglobal = folder + "/" + fglobal_root + fname + ".txt"
    g.global_point_ind = np.loadtxt(fglobal, dtype=np.int).T

    return g

def import_gb(folder, max_dim, **kwargs):

    index_names = kwargs.get("index_names", 1)
    index_shift = kwargs.get("index_shift", 1)

    # bulk grid
    fname_bulk = kwargs.get("bulk", "_bulk")
    grid_bulk = [import_grid(folder, fname_bulk, max_dim, **kwargs)]

    # fracture grids
    fname_fracture = kwargs.get("fracture", "_fracture_")
    grid_fracture = []

    # check the number of fractures
    fname = "/" + kwargs.get("nodes", "points") + fname_fracture + "*.txt"
    num_frac = len(glob.glob(folder + fname))
    print("there are " + str(num_frac) + " fractures")

    for idx in np.arange(num_frac):
        fname = fname_fracture + str(idx+index_names)
        g = import_grid(folder, fname, max_dim-1, **kwargs)
        grid_fracture.append(g)

    # intersection grids
    fname_intersection = kwargs.get("intersection", "_")
    grid_intersection = []

    # check the number of intersections
    fname = "/" + kwargs.get("nodes", "InterCellsCoord") + fname_intersection + "*.txt"
    num_int = len(glob.glob(folder + fname))
    print("there are " + str(num_int) + " intersections")

    for idx in np.arange(num_int):
        fname = fname_intersection + str(idx+index_names)
        g = import_grid_0d(folder, fname, **kwargs)
        grid_intersection.append(g)

    grids = [grid_bulk, grid_fracture, grid_intersection]
    #grids = [[], grid_fracture, []]#, grid_intersection]
    gb = pp.meshing.grid_list_to_grid_bucket(grids)
    pp.plot_grid(gb, alpha=0)

    for _, d in gb:
        d[pp.PRIMARY_VARIABLES] = {}
        d[pp.DISCRETIZATION] = {}
        d[pp.DISCRETIZATION_MATRICES] = {}

    for _, d in gb.edges():
        d[pp.PRIMARY_VARIABLES] = {}
        d[pp.COUPLING_DISCRETIZATION] = {}
        d[pp.DISCRETIZATION_MATRICES] = {}

    return gb
