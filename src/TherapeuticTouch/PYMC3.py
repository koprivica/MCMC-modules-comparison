import pymc3 as pm
import numpy as np

class TT_PYMC3:

    def __init__(self, ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N):
        self.Nsubj = len(N)

        with pm.Model() as model:
            omega = pm.Beta('omega', 1, 1, testval=meanThetaInit)
            kappaMinusTwo = pm.Gamma('kappa', 0.01, 0.01, testval=kappaInitMinusTwo)

            kappa = kappaMinusTwo + 2

            theta = pm.Beta('theta', omega * (kappa - 2) + 1, (1 - omega) * (kappa - 2) + 1, shape=self.Nsubj,
                            testval=np.array(ThetaI))

            self.z = pm.Binomial('z', p=theta, n=np.array(N), observed=np.array(zz))

            self.samples = pm.sample(133666, step=pm.NUTS(), njobs=3, random_seed=-1, progressbar=False)