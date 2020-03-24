import numpy as np
import scipy.sparse as sps
import porepy as pp

from flow import Flow
from transport import Transport
from reaction import Reaction
from porosity import Porosity
from aperture import Aperture

class Scheme(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb):
        self.gb = gb

        # -- flow -- #
        self.discr_flow = Flow(gb)

        shape = self.discr_flow.shape()
        self.flux_pressure = np.zeros(shape)

        # -- solute and precipitate -- #
        self.discr_solute_advection_diffusion = Transport(gb)
        self.discr_solute_precipitate_reaction = Reaction(gb)

        shape = self.discr_solute_advection_diffusion.shape()
        self.solute = np.zeros(shape)
        self.precipitate = np.zeros(shape)
        self.solute_old = np.zeros(shape)
        self.precipitate_old = np.zeros(shape)

        # -- temperature -- #
        self.temperature = np.zeros(shape)

        # -- porosity -- #
        self.discr_porosity = Porosity(gb)

        shape = self.discr_porosity.shape()
        self.porosity = np.zeros(shape)
        self.porosity_old = np.zeros(shape)
        self.porosity_star = np.zeros(shape)

        # -- aperture -- #
        self.discr_aperture = Aperture(gb)

        shape = self.discr_aperture.shape()
        self.aperture = np.zeros(shape)
        self.aperture_old = np.zeros(shape)
        self.aperture_star = np.zeros(shape)

        # map to identify the rock and the fractures
        self.is_rock = np.zeros(shape, dtype=np.bool)

    # ------------------------------------------------------------------------------#

    def update_flow(self):
        self.discr_flow.update_data()

    # ------------------------------------------------------------------------------#

    def solve_flow(self):
        A, b = self.discr_flow.matrix_rhs()
        self.flux_pressure = sps.linalg.spsolve(A, b)

    # ------------------------------------------------------------------------------#

    def update_solute_precipitate(self):
        # set the flux from the flow equation
        self.discr_solute_advection_diffusion.set_flux(self.discr_flow.flux, self.discr_flow.mortar)

        # update the data
        self.discr_solute_advection_diffusion.update_data()

    # ------------------------------------------------------------------------------#

    def solve_solute_precipitate_advection_diffusion(self):
        # conc 0 e' il soluto, conc 1 e' il precipitato
        A, M, b = self.discr_solute_advection_diffusion.matrix_rhs()

        # the mass term which considers both the porosity and aperture contribution
        M_star = M * sps.diags(self.porosity_star + self.aperture_star, 0)
        M_old = M * sps.diags(self.porosity_old + self.aperture_old, 0)

        # construct the left and right-hand side
        lhs = M_star + A
        rhs = M_old *  self.solute + b

        # save the old solute
        self.solute_old = self.solute.copy()

        # compute the new solute
        self.solute = sps.linalg.spsolve(lhs, rhs)

    # ------------------------------------------------------------------------------#

    def solve_solute_precipitate_rection(self):
        # save the old precipitate
        self.precipitate_old = self.precipitate.copy()
        # APPROPOSITUO DEL SOLUTO?? QUALE TENGO?

        # compute the new solute and precipitate
        self.solute, self.precipitate = self.discr_solute_precipitate_reaction.step(
                                        self.solute, self.precipitate, self.temperature)

    # ------------------------------------------------------------------------------#

    def solve_porosity(self):
        self.porosity_old = self.porosity.copy()

        self.porosity = self.discr_porosity.step(
                        self.porosity, self.precipitate, self.precipitate_old)

        self.porosity_star = self.discr_porosity.extrapolate(
                             self.porosity, self.porosity_old)

    # ------------------------------------------------------------------------------#

    def solve_aperture(self):
        self.aperture_old = self.aperture.copy()

        self.aperture = self.discr_aperture.step(
                        self.aperture, self.precipitate, self.precipitate_old)

        self.aperture_star = self.discr_aperture.extrapolate(
                             self.aperture, self.aperture_old)

    # ------------------------------------------------------------------------------#

    def set_data(self, param):

        # set the initial condition
        assembler = self.discr_solute_advection_diffusion.assembler
        dof = np.cumsum(np.append(0, np.asarray(assembler.full_dof)))
        for (g, _), bi in assembler.block_dof.items():
            #g = pair[0]
            if isinstance(g, pp.Grid):
                dof_loc = slice(dof[bi], dof[bi+1])

                data = param["solute_advection_diffusion"]["initial_solute"]
                self.solute[dof_loc] = data(g, param, param["tol"])

                data = param["solute_advection_diffusion"]["initial_precipitate"]
                self.precipitate[dof_loc] = data(g, param, param["tol"])

                data = param["porosity"]["initial"]
                self.porosity[dof_loc] = data(g, param, param["tol"])

                data = param["aperture"]["initial"]
                self.aperture[dof_loc] = data(g, param, param["tol"])

                data = param["temperature"]["initial"]
                self.temperature[dof_loc] = data(g, param, param["tol"])

                self.is_rock[dof_loc] = g.dim == self.gb.dim_max()

        # set the old variables
        self.solute_old = self.solute.copy()
        self.precipitate_old = self.precipitate.copy()
        self.porosity_old = self.porosity.copy()
        self.aperture_old = self.aperture.copy()

        # set the star variables
        self.porosity_star = self.discr_porosity.extrapolate(
                             self.porosity, self.porosity_old)

        self.aperture_star = self.discr_aperture.extrapolate(
                             self.aperture, self.aperture_old)

        # extract the initialized variables, useful for setting the data
        self.extract()

        # set now the data for each scheme
        self.discr_flow.set_data(param["flow"], param["time"])
        self.discr_solute_advection_diffusion.set_data(param["solute_advection_diffusion"], param["time"])
        self.discr_solute_precipitate_reaction.set_data(param["solute_precipitate_reaction"], param["time"])
        self.discr_porosity.set_data(param["porosity"])
        self.discr_aperture.set_data(param["aperture"])


    # ------------------------------------------------------------------------------#

    def extract(self):

        self.discr_flow.extract(self.flux_pressure)
        self.discr_solute_advection_diffusion.extract(self.solute, "solute")
        self.discr_solute_advection_diffusion.extract(self.precipitate, "precipitate")
        self.discr_solute_advection_diffusion.extract(self.temperature, "temperature")

        self.discr_porosity.extract(self.porosity, "porosity")
        self.discr_porosity.extract(self.porosity_old, "porosity_old")
        self.discr_porosity.extract(self.porosity_star, "porosity_star")

        self.discr_aperture.extract(self.aperture, "aperture")
        self.discr_aperture.extract(self.aperture_old, "aperture_old")
        self.discr_aperture.extract(self.aperture_star, "aperture_star")

    # ------------------------------------------------------------------------------#

    def vars_to_save(self):
        name = ["solute", "precipitate", "porosity", "aperture", "temperature"]
        return name + [self.discr_flow.pressure, self.discr_flow.P0_flux]

    # ------------------------------------------------------------------------------#
