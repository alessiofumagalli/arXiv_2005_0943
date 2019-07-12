import numpy as np
import porepy as pp

class Flow(object):

    def __init__(self, gb, model, tol):

        self.model = model
        self.gb = gb
        self.data = None
        self.assembler = None

        # discretization operator name
        self.discr_name = self.model + "_flux"
        self.discr = pp.RT0(self.model)

        self.mass_name = self.model + "_mass"
        self.mass = pp.MixedMassMatrix(self.model)

        # coupling operator
        self.coupling_name = self.discr_name + "_coupling"
        self.coupling = pp.RobinCoupling(self.model, self.discr)

        # source
        self.source_name = self.model + "_source"
        self.source = pp.DualScalarSource(self.model)

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar = self.model + "_lambda"

        # post process variables
        self.pressure = "pressure"
        self.flux = "darcy_flux"  # it has to be this one
        self.P0_flux = "P0_darcy_flux"

        # tolerance
        self.tol = tol

    def set_data(self, data, bc_flag):
        self.data = data

        for g, d in self.gb:
            param = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["is_tangential"] = True
            d["tol"] = self.tol

            # assign permeability
            if g.dim < self.gb.dim_max():
                kxx = data["kf_t"] * unity
                perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)
                aperture = data["aperture"] * unity

            else:
                kxx = data["k"] * unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
                aperture = unity

            param["second_order_tensor"] = perm
            param["aperture"] = aperture
            param["mass_weight"] = data["mass_weight"] * unity

            param["source"] = g.cell_volumes * 0

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels, bc_val = bc_flag(g, data, self.tol)
                param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
            else:
                bc_val = np.zeros(g.num_faces)
                param["bc"] = pp.BoundaryCondition(g, empty, empty)

            param["bc_values"] = bc_val

            d[pp.PARAMETERS].update(pp.Parameters(g, self.model, param))

        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]

            mg = d["mortar_grid"]
            check_P = mg.slave_to_mortar_avg()

            aperture = self.gb.node_props(g_l, pp.PARAMETERS)[self.model]["aperture"]
            gamma = check_P * aperture
            kn = data["kf_n"] * np.ones(mg.num_cells) / gamma
            param = {"normal_diffusivity": kn}

            d[pp.PARAMETERS].update(pp.Parameters(e, self.model, param))

        # set now the discretization

        # set the discretization for the grids
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.variable: {"cells": 1, "faces": 1}})
            d[pp.DISCRETIZATION].update({self.variable: {self.discr_name: self.discr,
                                                         self.mass_name: self.mass,
                                                         self.source_name: self.source}})

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES].update({self.mortar: {"cells": 1}})
            d[pp.COUPLING_DISCRETIZATION].update({
                self.coupling_name: {
                    g_slave: (self.variable, self.discr_name),
                    g_master: (self.variable, self.discr_name),
                    e: (self.mortar, self.coupling),
                }
            })

        # assembler
        variables = [self.variable, self.mortar]
        self.assembler = pp.Assembler(self.gb, active_variables=variables)

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # empty the matrices
        for g, d in self.gb:
            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        for e, d in self.gb.edges():
            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        block_A, block_b = self.assembler.assemble_matrix_rhs(add_matrices=False)

        # unpack the matrices just computed
        coupling_name = self.coupling_name + (
            "_" + self.mortar + "_" + self.variable + "_" + self.variable
        )
        discr_name = self.discr_name + "_" + self.variable
        mass_name = self.mass_name + "_" + self.variable
        source_name = self.source_name + "_" + self.variable

        # need a sign for the convention of the conservation equation
        M = - block_A[mass_name]

        if self.gb.size() > 1:
            A = block_A[discr_name] + block_A[coupling_name]
            b = block_b[discr_name] + block_b[coupling_name] + block_b[source_name]
        else:
            A = block_A[discr_name]
            b = block_b[discr_name] + block_b[source_name]

        return A, M, b

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        self.assembler.distribute_variable(x)
        for g, d in self.gb:
            d[pp.STATE][self.pressure] = self.discr.extract_pressure(g, d[pp.STATE][self.variable], d)
            d[pp.STATE][self.flux] = self.discr.extract_flux(g, d[pp.STATE][self.variable], d)

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, self.discr, self.flux, self.P0_flux, self.mortar)

    # ------------------------------------------------------------------------------#
