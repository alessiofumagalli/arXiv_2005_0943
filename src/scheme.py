import numpy as np
import scipy.sparse as sps
import porepy as pp

from flow import Flow
from transport import Transport
from reaction import Reaction
from porosity import Porosity
from aperture import Aperture
from heat import Heat

class Scheme(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb):
        self.gb = gb

        # -- flow -- #
        self.discr_flow = Flow(gb)

        shape = self.discr_flow.shape()
        self.flux_pressure = np.zeros(shape)

        # -- temperature -- #
        self.discr_temperature = Heat(gb)

        shape = self.discr_temperature.shape()
        self.temperature = np.zeros(shape)
        self.temperature_old = np.zeros(shape)

        # -- solute and precipitate -- #
        self.discr_solute_advection_diffusion = Transport(gb)
        self.discr_solute_precipitate_reaction = Reaction(gb)

        shape = self.discr_solute_advection_diffusion.shape()
        self.solute = np.zeros(shape)
        self.precipitate = np.zeros(shape)
        self.solute_old = np.zeros(shape)
        self.precipitate_old = np.zeros(shape)

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

        # -- composite variables -- #
        self.porosity_aperture_times_solute = np.zeros(shape)
        self.porosity_aperture_times_precipitate = np.zeros(shape)

    # ------------------------------------------------------------------------------#

    def compute_flow(self):
        A, b = self.discr_flow.matrix_rhs()
        return sps.linalg.spsolve(A, b)

    # ------------------------------------------------------------------------------#

    def compute_temperature(self, porosity_star, aperture_star):
        # compute the matrices
        A, M, b = self.discr_temperature.matrix_rhs()

        # compute the effective thermal capacity, keeping in mind that the fracture
        # contains only water
        rc_w = self.discr_temperature.data["rc_w"]
        rc_s = self.discr_temperature.data["rc_s"]

        c_star = rc_w * (porosity_star + aperture_star) + rc_s * (1 - porosity_star)
        c_old = rc_w * (self.porosity_old + self.aperture_old) + rc_s * (1 - self.porosity_old)

        # the mass term which considers the contribution from the effective thermal capacity
        M_star = M * sps.diags(c_star, 0)
        M_old = M * sps.diags(c_old, 0)

        # compute the new temperature
        return sps.linalg.spsolve(M_star + A, M_old * self.temperature_old + b)

    # ------------------------------------------------------------------------------#

    def compute_solute_precipitate_advection_diffusion(self, porosity_star, aperture_star):
        # compute the matrices
        A, M, b = self.discr_solute_advection_diffusion.matrix_rhs()

        # the mass term which considers both the porosity and aperture contribution
        M_star = M * sps.diags(porosity_star + aperture_star, 0)
        M_old = M * sps.diags(self.porosity_old + self.aperture_old, 0)

        # compute the new solute
        return sps.linalg.spsolve(M_star + A, M_old * self.solute_old + b)

    # ------------------------------------------------------------------------------#

    def compute_solute_precipitate_rection(self, solute_half, precipitate_half):
        # the dof associated to the porous media and fractures, are the first
        dof = self.gb.num_cells()

        # temporary solution vectors
        solute = np.zeros(self.discr_solute_advection_diffusion.shape())
        precipitate = np.zeros(self.discr_solute_advection_diffusion.shape())

        # compute the new solute and precipitate
        solute[:dof], precipitate[:dof] = self.discr_solute_precipitate_reaction.step(
                                                                    solute_half[:dof],
                                                                    precipitate_half[:dof],
                                                                    self.temperature[:dof])
        return solute, precipitate

    # ------------------------------------------------------------------------------#

    def set_old_variables(self):
        self.temperature_old = self.temperature.copy()
        self.solute_old = self.solute.copy()
        self.precipitate_old = self.precipitate.copy()
        self.porosity_old = self.porosity.copy()
        self.aperture_old = self.aperture.copy()

    # ------------------------------------------------------------------------------#

    def set_data(self, param):

        # set the initial condition
        assembler = self.discr_solute_advection_diffusion.assembler
        dof = np.cumsum(np.append(0, np.asarray(assembler.full_dof)))
        for (g, _), bi in assembler.block_dof.items():
            #g = pair[0]
            if isinstance(g, pp.Grid):
                dof_loc = slice(dof[bi], dof[bi+1])

                data = param["temperature"]["initial"]
                self.temperature[dof_loc] = data(g, param, param["tol"])

                data = param["solute_advection_diffusion"]["initial_solute"]
                self.solute[dof_loc] = data(g, param, param["tol"])

                data = param["solute_advection_diffusion"]["initial_precipitate"]
                self.precipitate[dof_loc] = data(g, param, param["tol"])

                data = param["porosity"]["initial"]
                self.porosity[dof_loc] = data(g, param, param["tol"])

                data = param["aperture"]["initial"]
                self.aperture[dof_loc] = data(g, param, param["tol"])

        # set the old variables
        self.set_old_variables()

        # extract the initialized variables, useful for setting the data
        self.extract()

        # set now the data for each scheme
        self.discr_flow.set_data(param["flow"], param["time"])
        self.discr_temperature.set_data(param["temperature"], param["time"])
        self.discr_solute_advection_diffusion.set_data(param["solute_advection_diffusion"], param["time"])
        self.discr_solute_precipitate_reaction.set_data(param["solute_precipitate_reaction"], param["time"])
        self.discr_porosity.set_data(param["porosity"])
        self.discr_aperture.set_data(param["aperture"])

    # ------------------------------------------------------------------------------#

    def extract(self):

        self.discr_flow.extract(self.flux_pressure)
        self.discr_temperature.extract(self.temperature, "temperature")
        self.discr_solute_advection_diffusion.extract(self.solute, "solute")
        self.discr_solute_advection_diffusion.extract(self.precipitate, "precipitate")

        self.discr_porosity.extract(self.porosity, "porosity")
        self.discr_porosity.extract(self.porosity_old, "porosity_old")

        self.discr_aperture.extract(self.aperture, "aperture")
        self.discr_aperture.extract(self.aperture_old, "aperture_old")

        self.discr_solute_advection_diffusion.extract(self.porosity_aperture_times_solute,
            "porosity_aperture_times_solute")
        self.discr_solute_advection_diffusion.extract(self.porosity_aperture_times_precipitate,
            "porosity_aperture_times_precipitate")

    # ------------------------------------------------------------------------------#

    def vars_to_save(self):
        name = ["solute", "precipitate", "porosity", "aperture", "temperature"]
        name += ["porosity_aperture_times_solute", "porosity_aperture_times_precipitate"]
        return name + [self.discr_flow.pressure, self.discr_flow.P0_flux]

    # ------------------------------------------------------------------------------#


    def one_step_splitting_scheme(self):

        # the dof associated to the porous media and fractures, are the first
        dof = slice(0, self.gb.num_cells())

        # POINT 1) extrapolate the precipitate to get a better estimate of porosity
        precipitate_star = 2*self.precipitate - self.precipitate_old

        # POINT 2) compute the porosity and aperture star
        porosity_star = self.discr_porosity.step(self.porosity_old, precipitate_star, self.precipitate_old)
        self.discr_porosity.extract(porosity_star, "porosity_star")

        aperture_star = self.discr_aperture.step(self.aperture_old, precipitate_star, self.precipitate_old)
        self.discr_aperture.extract(aperture_star, "aperture_star")

        # -- DO THE FLOW PART -- #

        # POINT 3) update the data from the previous time step
        self.discr_flow.update_data()

        # POINT 4) solve the flow part
        self.flux_pressure = self.compute_flow()
        self.discr_flow.extract(self.flux_pressure)

        # -- DO THE HEAT PART -- #

        # POINT 5) set the flux and update the data from the previous time step
        self.discr_temperature.set_flux(self.discr_flow.flux, self.discr_flow.mortar)
        self.discr_temperature.update_data()

        # POINT 5) solve the temperature part
        self.temperature = self.compute_temperature(porosity_star, aperture_star)

        # -- DO THE TRANSPORT PART -- #

        # set the flux and update the data from the previous time step
        self.discr_solute_advection_diffusion.set_flux(self.discr_flow.flux, self.discr_flow.mortar)
        self.discr_solute_advection_diffusion.update_data()

        # POINT 6) solve the advection and diffusion part to get the intermediate solute solution
        solute_half = self.compute_solute_precipitate_advection_diffusion(porosity_star, aperture_star)

        # POINT 7) Since in the advection-diffusion step we have accounted for porosity changes using
        # phi_star, the new solute concentration accounts for the change in pore volume, thus, the
        # precipitate needs to be updated accordingly
        factor = np.zeros(self.porosity_old.size)
        factor[dof] = (self.porosity_old[dof] + self.aperture_old[dof]) / (porosity_star[dof] + aperture_star[dof])
        precipitate_half = self.precipitate_old * factor

        # POINT 8) solve the reaction part
        solute_star_star, precipitate_star_star = self.compute_solute_precipitate_rection(solute_half, precipitate_half)

        # -- DO THE POROSITY PART -- #

        # POINT 9) solve the porosity and aperture part with the true concentration of precipitate
        self.porosity = self.discr_porosity.step(self.porosity, precipitate_star_star, self.precipitate_old)
        self.aperture = self.discr_aperture.step(self.aperture, precipitate_star_star, self.precipitate_old)

        # POINT 10) finally, we correct the concentrations to account for the difference between the extrapolated
        # and "true" new porosity to ensure mass conservation
        factor = np.zeros(self.porosity_old.size)
        factor[dof] = (porosity_star[dof] + aperture_star[dof]) / (self.porosity[dof] + self.aperture[dof])
        self.solute = solute_star_star * factor
        self.precipitate = precipitate_star_star * factor

        # set the old variables
        self.set_old_variables()

        # compute composite variables
        factor = self.porosity + self.aperture
        self.porosity_aperture_times_solute = factor * self.solute
        self.porosity_aperture_times_precipitate = factor * self.precipitate

        # extract all the variables, useful for exporting
        self.extract()

    # ------------------------------------------------------------------------------#
