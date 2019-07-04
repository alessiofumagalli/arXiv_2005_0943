import numpy as np
import scipy.sparse as sps
import porepy as pp

from transport import Transport
from reaction import Reaction

class Concentration(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, models_trans, model_react, tol):

        self.gb = gb
        self.tol = tol

        # set up the transport models
        self.models_trans = np.atleast_1d(models_trans)
        self.discr_trans = {m: Transport(self.gb, m, self.tol) for m in self.models_trans}

        # set up the reaction models
        # we assume only one reaction model
        self.model_react = model_react
        self.discr_react = Reaction(self.tol)

    # ------------------------------------------------------------------------------#

    def set_data(self, data, bc_flag):
        [d.set_data(data[m], bc_flag[m]) for m, d in self.discr_trans.items()]
        self.discr_react.set_data(data[self.model_react])

    # ------------------------------------------------------------------------------#

    def set_flux(self, flux_name, mortar_name):
        [d.set_flux(flux_name, mortar_name) for _, d in self.discr_trans.items()]

    # ------------------------------------------------------------------------------#

    def shape(self):
        return [len(self.discr_trans), self.gb.num_cells() + 2*self.gb.num_mortar_cells()]

    # ------------------------------------------------------------------------------#

    def one_step(self, x, time):
        # do the advective and diffusive step for each specie separately
        for pos, discr in enumerate(self.discr_trans.values()):

            # compute the matrices and rhs
            A, M, b = discr.matrix_rhs()

            # solve the advective and diffusive step
            x[pos, :] = sps.linalg.spsolve(M + A, M * x[pos, :] + b)

        # do the reactive step
        #x = self.discr_react.step(x, time)

        return x

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        [d.extract(x[p, :]) for p, d in enumerate(self.discr_trans.values())]

    # ------------------------------------------------------------------------------#

    def export(self, time_step=None):
        [d.export(time_step) for d in self.discr_trans.values()]

    # ------------------------------------------------------------------------------#

    def export_pvd(self, steps):
        [d.export_pvd(steps) for d in self.discr_trans.values()]

    # ------------------------------------------------------------------------------#
