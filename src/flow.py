import numpy as np
import porepy as pp

class Flow(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, model="flow"):

        self.model = model
        self.gb = gb
        self.data = None
        self.data_time = None
        self.assembler = None

        # discretization operator name
        self.discr_name = self.model + "_flux"
        self.discr = pp.RT0

        # coupling operator
        self.coupling_name = self.discr_name + "_coupling"
        self.coupling = pp.RobinCoupling

        # source
        self.source_name = self.model + "_source"
        self.source = pp.DualScalarSource

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar = self.model + "_lambda"

        # post process variables
        self.pressure = "pressure"
        self.flux = "darcy_flux"  # it has to be this one
        self.P0_flux = "P0_darcy_flux"

        # set the discretizaton
        self.set_discr()

    # ------------------------------------------------------------------------------#

    def set_discr(self):

        # set the discretization for the grids
        for _, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.variable: {"cells": 1, "faces": 1}})

            discr = self.discr(self.model)
            source = self.source(self.model)
            d[pp.DISCRETIZATION].update({self.variable: {self.discr_name: discr,
                                                         self.source_name: source}})

            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES].update({self.mortar: {"cells": 1}})

            coupling = self.coupling(self.model, self.discr(self.model))
            d[pp.COUPLING_DISCRETIZATION].update({
                self.coupling_name: {
                    g_slave: (self.variable, self.discr_name),
                    g_master: (self.variable, self.discr_name),
                    e: (self.mortar, coupling),
                }
            })

            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        # assembler
        variables = [self.variable, self.mortar]
        self.assembler = pp.Assembler(self.gb, active_variables=variables)

    # ------------------------------------------------------------------------------#

    def set_data(self, data, data_time):
        self.data = data
        self.data_time = data_time

        for g, d in self.gb:
            param = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            alpha = self.data.get("alpha", 2)

            d["is_tangential"] = True
            d["tol"] = data["tol"]

            # assign permeability
            if g.dim < self.gb.dim_max():
                aperture = d[pp.STATE]["aperture"]

                k = np.power(aperture, alpha+1) * self.data["k_t"]
                perm = pp.SecondOrderTensor(kxx=k, kyy=1, kzz=1)
            else:
                porosity = d[pp.STATE]["porosity"]

                k = np.power(porosity, alpha) * self.data["k"]
                perm = pp.SecondOrderTensor(kxx=k, kyy=k, kzz=1)

            # no source term is assumed by the user
            param["second_order_tensor"] = perm
            param["source"] = zeros

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels, param["bc_values"] = data["bc"](g, data, data["tol"])
                param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
            else:
                param["bc_values"] = np.zeros(g.num_faces)
                param["bc"] = pp.BoundaryCondition(g, empty, empty)

            pp.initialize_data(g, d, self.model, param)

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            g_l = self.gb.nodes_of_edge(e)[0]

            check_P = mg.slave_to_mortar_avg()
            aperture = self.gb.node_props(g_l, pp.STATE)["aperture"]

            k = 2 * check_P * (np.power(aperture, alpha-1) * self.data["k_n"])
            pp.initialize_data(mg, d, self.model, {"normal_diffusivity": k})

    # ------------------------------------------------------------------------------#

    def update_data(self):
        # NOTE the self.data stores the original values

        for g, d in self.gb:
            param = {}
            unity = np.ones(g.num_cells)

            alpha = self.data.get("alpha", 2)

            # assign permeability
            if g.dim < self.gb.dim_max():
                aperture = d[pp.STATE]["aperture"]
                aperture_star = d[pp.STATE]["aperture_star"]

                source = (aperture_star - aperture) / self.data_time["step"]
                k = np.power(aperture_star, alpha+1) * self.data["k_t"]

                # aperture and permeability check
                if np.any(aperture < 0) or np.any(aperture_star < 0) or np.any(k < 0):
                    import pdb; pdb.set_trace()
                    raise ValueError(str(np.any(aperture < 0)) + " " +
                                     str(np.any(aperture_star < 0)) + " " +
                                     str(np.any(k < 0)))

                perm = pp.SecondOrderTensor(kxx=k, kyy=1, kzz=1)
            else:
                poro = d[pp.STATE]["porosity"]
                poro_star = d[pp.STATE]["porosity_star"]

                source = (poro_star - poro) / self.data_time["step"]
                k = np.power(poro_star, alpha) * self.data["k"]

                # porosity and permeability check
                if np.any(poro < 0) or np.any(poro_star < 0) or np.any(k < 0):
                    raise ValueError(str(np.any(poro < 0)) + " " +
                                     str(np.any(poro_star < 0)) + " " +
                                     str(np.any(k < 0)))

                perm = pp.SecondOrderTensor(kxx=k, kyy=k, kzz=1)

            param["second_order_tensor"] = perm
            param["source"] = g.cell_volumes * source
            d[pp.PARAMETERS][self.model].update(param)

        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]

            check_P = d["mortar_grid"].slave_to_mortar_avg()
            aperture_star = self.gb.node_props(g_l, pp.STATE)["aperture_star"]

            k = 2 * check_P * (np.power(aperture_star, alpha-1) * self.data["k_n"])
            d[pp.PARAMETERS][self.model].update({"normal_diffusivity": k})

    # ------------------------------------------------------------------------------#

    def shape(self):
        return self.gb.num_cells() + self.gb.num_faces() + self.gb.num_mortar_cells()

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # empty the matrices
        for g, d in self.gb:
            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        for e, d in self.gb.edges():
            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        self.assembler.discretize()
        return self.assembler.assemble_matrix_rhs()

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        self.assembler.distribute_variable(x)

        discr = self.discr(self.model)
        for g, d in self.gb:
            var = d[pp.STATE][self.variable]
            d[pp.STATE][self.pressure] = discr.extract_pressure(g, var, d)
            d[pp.STATE][self.flux] = discr.extract_flux(g, var, d)

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, discr, self.flux, self.P0_flux, self.mortar)

    # ------------------------------------------------------------------------------#
