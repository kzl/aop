import numpy as np

class Perturber():

    def __init__(self, params):
        self.params = params

    def perturb(self, act):
        perturbed_act = act

        if self.params['type'] == 'rotation':
            # Useful in particle maze environments
            theta = self.params['theta']
            x, y = act[0], act[1]
            perturbed_act[0] = np.cos(theta) * x + np.sin(theta) * y
            perturbed_act[1] = -np.sin(theta) * x + np.cos(theta) * y

        elif self.params['type'] == 'zero':
            # i.e. a joint is disabled
            zero_inds = self.params['zero_inds']
            perturbed_act[zero_inds] = 0

        elif self.params['type'] == 'eye':
            pass

        return perturbed_act
