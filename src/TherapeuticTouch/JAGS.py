import pyjags

class TT_JAGS:

    def __init__(self, ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N):
        self.code = '''
        model {
            for (s in 1:Nsubj) {
                z[s] ~ dbin(theta[s], N[s])
                theta[s] ~ dbeta (omega*(kappa-2)+1 , (1-omega)*(kappa-2)+1 )
            }
            omega ~ dbeta(1,1)
            kappa <- kappaMinusTwo + 2
            kappaMinusTwo ~ dgamma(0.01 , 0.01) # mean =1 , sd=10
        }
        '''

        self.model = pyjags.Model(self.code, data=dict(z=zz, N=N, Nsubj=len(N)), init=dict(theta=ThetaI, omega=meanThetaInit,
                                                                                 kappaMinusTwo=kappaInitMinusTwo),
                             chains=3, adapt=500)

        # 500 warmup / burn-in iterations, not used for inference.
        self.model.sample(500, vars=[])

        self.samples = self.model.sample(133333, vars=['theta', 'omega', 'kappa'], thin=20)