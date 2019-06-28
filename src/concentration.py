import numpy as np
import scipy.sparse as sps
import porepy as pp

from transport import Transport

class Concentration(object):

    def __init__(self, gb, models, tol):

        self.gb = gb
        self.tol = tol

        models = np.atleast_1d(models)
        self.discr = {model: Transport(self.gb, model, self.tol) for model in models}

    # ------------------------------------------------------------------------------#

    def set_data(self, data, bc_flag):
        [discr.set_data(data[model], bc_flag[model]) for model, discr in self.discr.items()]

    # ------------------------------------------------------------------------------#

    def set_flux(self, flux_name, mortar_name):
        [discr.set_flux(flux_name, mortar_name) for _, discr in self.discr.items()]

    # ------------------------------------------------------------------------------#

    def shape(self):
        return [len(self.discr), self.gb.num_cells() + 2*self.gb.num_mortar_cells()]

    # ------------------------------------------------------------------------------#

    def one_step(self, x, time):
        # do the advective and diffusive step for each specie separately
        for pos, discr in enumerate(self.discr.values()):

            # compute the matrices and rhs
            A, M, b = discr.matrix_rhs()

            # solve the advective and diffusive step
            x[pos, :] = sps.linalg.spsolve(M + A, M * x[pos, :] + b)

        # do the reactive step

        return x

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        [discr.extract(x[pos, :]) for pos, discr in enumerate(self.discr.values())]

    # ------------------------------------------------------------------------------#

    def export(self, time_step=None):
        [discr.export(time_step) for discr in self.discr.values()]

    # ------------------------------------------------------------------------------#

    def export_pvd(self, steps):
        [discr.export_pvd(steps) for discr in self.discr.values()]

    # ------------------------------------------------------------------------------#
