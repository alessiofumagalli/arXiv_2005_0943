class Porosity(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, model="porosity"):

        self.model = model
        self.data = None
        self.eta = None

    # ------------------------------------------------------------------------------#

    def set_data(self, data):

        self.data = data
        self.eta = data["eta"]

    # ------------------------------------------------------------------------------#

    def extrapolate(self, x, x_old):
        return 2*x - x_old

    # ------------------------------------------------------------------------------#

    def step(self, x_old, v, v_old):
        #v is the precipitate
        return x_old - self.eta*(v - v_old)
        #return x_old / (1 + self.eta*(v - v_old))

    # ------------------------------------------------------------------------------#
