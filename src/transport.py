import numpy as np
import porepy as pp

class Transport(object):

    def __init__(self, gb, model, tol):

        self.model = model
        self.gb = gb
        self.data = None
        self.assembler = None

        # discretization operator name
        self.diff_name = self.model + "_diff"
        self.diff = pp.Tpfa(self.model)

        self.adv_name = self.model + "_adv"
        self.adv = pp.Upwind(self.model)

        self.mass_name = self.model + "_mass"
        self.mass = pp.MassMatrix(self.model)

        # coupling operator
        self.coupling_diff_name = self.diff_name + "_coupling"
        self.coupling_diff = pp.RobinCoupling(self.model, self.diff)

        self.coupling_adv_name = self.adv_name + "_coupling"
        self.coupling_adv = pp.UpwindCoupling(self.model)

        self.source_name = self.model + "_source"
        self.source = pp.ScalarSource(self.model)

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar_diff = self.diff_name + "_lambda"
        self.mortar_adv = self.adv_name + "_lambda"

        # post process variables
        self.scalar = self.model + "_scalar"
        self.flux = "darcy_flux"

        # tolerance
        self.tol = tol

        # exporter
        self.save = pp.Exporter(self.gb, self.model, folder="solution")

    def set_data(self, data, bc_flag):
        self.data = data

        for g, d in self.gb:
            param = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["Aavatsmark_transmissibilities"] = True
            d["tol"] = self.tol

            # assign permeability
            if g.dim < self.gb.dim_max():
                kxx = data["lf_t"] * unity
                aperture = data["aperture"] * unity
            else:
                kxx = data["l"] * unity
                aperture = unity

            param["second_order_tensor"] = pp.SecondOrderTensor(3, kxx=kxx)
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
            kn = data["lf_n"] * np.ones(mg.num_cells) / gamma
            param = {"normal_diffusivity": kn}

            d[pp.PARAMETERS].update(pp.Parameters(e, self.model, param))

        # set now the discretization

        # set the discretization for the grids
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.variable: {"cells": 1}})
            d[pp.DISCRETIZATION].update({self.variable: {self.diff_name: self.diff,
                                                         self.adv_name: self.adv,
                                                         self.mass_name: self.mass,
                                                         self.source_name: self.source}})

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES].update({self.mortar_diff: {"cells": 1},
                                            self.mortar_adv: {"cells": 1}})
            d[pp.COUPLING_DISCRETIZATION].update({
                self.coupling_diff_name: {
                    g_slave: (self.variable, self.diff_name),
                    g_master: (self.variable, self.diff_name),
                    e: (self.mortar_diff, self.coupling_diff),
                }
            })
            d[pp.COUPLING_DISCRETIZATION].update({
                self.coupling_adv_name: {
                    g_slave: (self.variable, self.adv_name),
                    g_master: (self.variable, self.adv_name),
                    e: (self.mortar_adv, self.coupling_adv),
                }
            })

    # ------------------------------------------------------------------------------#

    def set_flux(self, flux_name, mortar_name):
        for _, d in self.gb:
            d[pp.PARAMETERS][self.model][self.flux] = d[pp.STATE][flux_name]

        for _, d in self.gb.edges():
            d[pp.PARAMETERS][self.model][self.flux] = d[pp.STATE][mortar_name]

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # empty the matrices
        for g, d in self.gb:
            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        for e, d in self.gb.edges():
            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        # solution of the darcy problem
        variables = [self.variable, self.mortar_diff, self.mortar_adv]
        self.assembler = pp.Assembler(self.gb, active_variables=variables)

        block_A, block_b = self.assembler.assemble_matrix_rhs(add_matrices=False)

        # unpack the matrices just computed
        coupling_diff_name = self.coupling_diff_name + (
            "_" + self.mortar_diff + "_" + self.variable + "_" + self.variable
        )
        coupling_adv_name = self.coupling_adv_name + (
            "_" + self.mortar_adv + "_" + self.variable + "_" + self.variable
        )
        diff_name = self.diff_name + "_" + self.variable
        adv_name = self.adv_name + "_" + self.variable
        mass_name = self.mass_name + "_" + self.variable
        source_name = self.source_name + "_" + self.variable

        # extract the matrices
        M = block_A[mass_name]

        if self.gb.size() > 1:
            A = block_A[diff_name] + block_A[coupling_diff_name] +\
                block_A[adv_name] + block_A[coupling_adv_name]
            b = block_b[diff_name] + block_b[coupling_diff_name] +\
                block_b[adv_name] + block_b[coupling_adv_name] +\
                block_b[source_name]
        else:
            A = block_A[discr_name] + block_A[adv_name]
            b = block_b[discr_name] + block_b[adv_name] + block_b[source_name]

        return A, M, b

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        self.assembler.distribute_variable(x)
        for _, d in self.gb:
            d[pp.STATE][self.scalar] = d[pp.STATE][self.variable]

    # ------------------------------------------------------------------------------#

    def export(self, time_step=None):
        self.save.write_vtk([self.scalar], time_step=time_step)

    # ------------------------------------------------------------------------------#

    def export_pvd(self, steps):
        self.save.write_pvd(steps)

    # ------------------------------------------------------------------------------#
