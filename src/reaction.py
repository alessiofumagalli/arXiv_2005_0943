import numpy as np

class Reaction(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, tol=1e-16, max_it=1e2):
        self.tol = tol
        self.max_it = max_it

    # ------------------------------------------------------------------------------#

    def set_data(self, data):
        # dimensionless parameter
        self.adim = data["length"]/data["gamma_eq"]/data["velocity"]

        self.theta = data["theta"]
        self.reaction = data["reaction"]

    # ------------------------------------------------------------------------------#

    def step(self, uw_0, deltaT):
        # solution at the end of the procedure
        uw_n = np.zeros(uw_0.shape)

        # compute the reaction rate with the solute and precipitate
        reaction_0 = self.reaction_fct(uw_0)

        # update the solute and precipitate
        uw_1 = uw_0 + 0.5 * deltaT * reaction_0

        # we need to check now if the precipitate went negative
        neg_1 = np.where(uw_1[1, :] < 0)[0]
        pos_1 = np.where(uw_1[1, :] >= 0)[0]

        # for them compute the time step such that they become null
        deltaT_corrected = np.tile(uw_0[1, neg_1] / np.abs(reaction_0[1, neg_1]), (2, 1))

        # correct the step
        uw_corrected = uw_0[:, neg_1] + 0.5 * deltaT_corrected * reaction_0[:, neg_1]
        uw_eta = uw_0[:, neg_1] + deltaT_corrected * self.reaction_fct(uw_corrected)

        # then finish with the last part of the deltaT
        uw_corrected = uw_eta + 0.5 * (deltaT - deltaT_corrected) * self.reaction_fct(uw_eta)
        uw_n[:, neg_1] = uw_eta + (deltaT - deltaT_corrected) * self.reaction_fct(uw_corrected)

        # consider now the positive and progress with the scheme
        uw_n[:, pos_1] = uw_0[:, pos_1] + deltaT * self.reaction_fct(uw_1[:, pos_1])

        # if during the second step something goes wrong we still need to work on it
        neg_n = pos_1[np.where(uw_n[1, pos_1] < 0)[0]]

        for i in neg_n:
            # initialization of the bisection algorithm
            if uw_n[1, i] < 0:
                err = np.abs(uw_n[1, i])
                it = 0
                deltaT_a, deltaT_b = 0, deltaT

                while err > self.tol and it < self.max_it:
                    deltaT_mid = 0.5*(deltaT_a + deltaT_b)
                    # re-do the step
                    uw_bisec = uw_0[:, i, np.newaxis] + 0.5 * deltaT_mid * reaction_0[:, i, np.newaxis]
                    uw_bisec = uw_0[:, i, np.newaxis] + deltaT_mid * self.reaction_fct(uw_bisec)

                    if uw_bisec[1, :] > 0:
                        deltaT_a = deltaT_mid
                    else:
                        deltaT_b = deltaT_mid

                    it += 1
                    err = np.abs(uw_bisec[1, :])


                # we are at convergence, we want to avoid negative numbers
                uw_bisec[1, :] = 0
                # then finish with the last part of the deltaT
                uw_corrected = uw_bisec + 0.5 * (deltaT - deltaT_mid) * self.reaction_fct(uw_bisec)
                uw_n[:, i, np.newaxis] = uw_bisec + (deltaT - deltaT_mid) * self.reaction_fct(uw_corrected)

        return uw_n

    # ------------------------------------------------------------------------------#

    def reaction_fct(self, uw):
        # set the reaction rate, we suppose given as a function with
        # (solute, precipitate) as argument

        sign = np.ones(uw.shape)
        sign[1, :] = -1

        val = self.reaction(uw[0, :], uw[1, :], self.theta)
        return self.adim * np.einsum("ij,j->ij", sign, val)

    # ------------------------------------------------------------------------------#
