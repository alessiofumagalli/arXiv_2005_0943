import numpy as np
import porepy as pp

class Transport(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, model="solute"):

        self.model = model
        self.gb = gb
        self.data = None
        self.data_time = None
        self.assembler = None

        # discretization operator name
        self.diff_name = self.model + "_diff"
        self.diff = pp.Tpfa(self.diff_name)

        self.adv_name = self.model + "_adv"
        self.adv = pp.Upwind(self.adv_name)

        self.mass_name = self.model + "_mass"
        self.mass = pp.MassMatrix(self.mass_name)

        # coupling operator
        self.coupling_diff_name = self.diff_name + "_coupling"
        self.coupling_diff = pp.RobinCoupling(self.diff_name, self.diff)

        self.coupling_adv_name = self.adv_name + "_coupling"
        self.coupling_adv = pp.UpwindCoupling(self.adv_name)

        self.source_name = self.model + "_source"
        self.source = pp.ScalarSource(self.source_name)

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar_diff = self.diff_name + "_lambda"
        self.mortar_adv = self.adv_name + "_lambda"

        # post process variables
        self.scalar = self.model + "_scalar"
        self.flux = "darcy_flux"

        # set the discretizaton
        self.set_discr()

    # ------------------------------------------------------------------------------#

    def set_discr(self):

        # set the discretization for the grids
        for _, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.variable: {"cells": 1}})
            d[pp.DISCRETIZATION].update({self.variable: {self.diff_name: self.diff,
                                                         self.adv_name: self.adv,
                                                         self.mass_name: self.mass,
                                                         self.source_name: self.source}})
            d[pp.DISCRETIZATION_MATRICES].update({self.diff_name: {}, self.adv_name: {},
                                                  self.mass_name: {}, self.source_name: {}})

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
            d[pp.DISCRETIZATION_MATRICES].update({self.diff_name: {}, self.adv_name: {},
                                                  self.mass_name: {}, self.source_name: {}})

        # assembler
        variables = [self.variable, self.mortar_diff, self.mortar_adv]
        self.assembler = pp.Assembler(self.gb, active_variables=variables)

    # ------------------------------------------------------------------------------#

    def set_data(self, data, data_time):
        self.data = data
        self.data_time = data_time

        for g, d in self.gb:
            param_diff = {}
            param_adv = {}
            param_mass = {}
            param_source = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["Aavatsmark_transmissibilities"] = True
            d["tol"] = data["tol"]

            # assign permeability
            if g.dim < self.gb.dim_max():
                aperture_star = d[pp.STATE]["aperture_star"]

                l = data["l_t"] * aperture_star
                diff = pp.SecondOrderTensor(l)
            else:
                poro_star = d[pp.STATE]["porosity_star"]

                l = poro_star * data["l"]
                diff = pp.SecondOrderTensor(l)

            param_diff["second_order_tensor"] = diff

            param_mass["mass_weight"] = data["mass_weight"] / data_time["step"]
            param_source["source"] = zeros

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels_diff, labels_adv, bc_val = data["bc"](g, data, data["tol"])
                param_diff["bc"] = pp.BoundaryCondition(g, b_faces, labels_diff)
                param_adv["bc"] = pp.BoundaryCondition(g, b_faces, labels_adv)
            else:
                bc_val = np.zeros(g.num_faces)
                param_diff["bc"] = pp.BoundaryCondition(g, empty, empty)
                param_adv["bc"] = pp.BoundaryCondition(g, empty, empty)

            param_diff["bc_values"] = bc_val
            param_adv["bc_values"] = bc_val

            models = [self.diff_name, self.adv_name, self.mass_name, self.source_name]
            params = [param_diff, param_adv, param_mass, param_source]
            for model, param in zip(models, params):
                pp.initialize_data(g, d, model, param)

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            g_l = self.gb.nodes_of_edge(e)[0]

            check_P = mg.slave_to_mortar_avg()
            aperture_star = self.gb.node_props(g_l, pp.STATE)["aperture_star"]

            l = 2 * check_P * (self.data["l_n"] / aperture_star)

            models = [self.diff_name, self.adv_name]
            params = [{"normal_diffusivity": l}, {}]
            for model, param in zip(models, params):
                pp.initialize_data(mg, d, model, param)

    # ------------------------------------------------------------------------------#

    def update_data(self):
        # NOTE the self.data stores the original values

        for g, d in self.gb:
            param_diff = {}

            # assign permeability
            if g.dim < self.gb.dim_max():
                aperture_star = d[pp.STATE]["aperture_star"]

                l = aperture_star * self.data["l_t"]
                diff = pp.SecondOrderTensor(l)
            else:
                poro_star = d[pp.STATE]["porosity_star"]

                l = poro_star * self.data["l"]
                try:
                    diff = pp.SecondOrderTensor(l)
                except:
                    import pdb; pdb.set_trace()
                    print(diff)

            param_diff["second_order_tensor"] = diff
            d[pp.PARAMETERS][self.diff_name].update(param_diff)

        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]

            check_P = d["mortar_grid"].slave_to_mortar_avg()
            aperture_star = self.gb.node_props(g_l, pp.STATE)["aperture_star"]

            l = 2 * check_P * (self.data["l_n"] / aperture_star)
            d[pp.PARAMETERS][self.diff_name].update({"normal_diffusivity": l})

    # ------------------------------------------------------------------------------#

    def shape(self):
        return self.gb.num_cells() + 2*self.gb.num_mortar_cells()

    # ------------------------------------------------------------------------------#

    def set_flux(self, flux_name, mortar_name):
        for _, d in self.gb:
            d[pp.PARAMETERS][self.adv_name][self.flux] = d[pp.STATE][flux_name]

        for _, d in self.gb.edges():
            d[pp.PARAMETERS][self.adv_name][self.flux] = d[pp.STATE][mortar_name]

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # empty the matrices
        for g, d in self.gb:
            d[pp.DISCRETIZATION_MATRICES].update({self.diff_name: {}, self.adv_name: {},
                                                  self.mass_name: {}, self.source_name: {}})

        for e, d in self.gb.edges():
            d[pp.DISCRETIZATION_MATRICES].update({self.diff_name: {}, self.adv_name: {},
                                                  self.mass_name: {}, self.source_name: {}})

        self.assembler.discretize()
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
            A = block_A[diff_name] + block_A[adv_name]
            b = block_b[diff_name] + block_b[adv_name] + block_b[source_name]

        return A, M, b

    # ------------------------------------------------------------------------------#

    def extract(self, x, name=None):
        self.assembler.distribute_variable(x)
        if name is None:
            name = self.scalar
        for _, d in self.gb:
            d[pp.STATE][name] = d[pp.STATE][self.variable]

    # ------------------------------------------------------------------------------#
