import pystan

class TT_STAN:

    def __init__(self, ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N):

        self.modelString = '''
            data {
                int<lower=1> Nsubj;
                int N [Nsubj];
                int z [Nsubj];
            }

            parameters {
                real<lower=0, upper=1> theta[Nsubj];

                real<lower=0, upper=1> omega;
                real<lower=0> kappaMinusTwo ;
            }

            transformed parameters {
                real<lower=0> kappa ;
                kappa = kappaMinusTwo + 2 ;
          }

            model {
                omega ~ beta(1,1) ;

                kappaMinusTwo ~ gamma(0.01 , 0.01) ; # mean =1 , sd=10
                theta ~ beta (omega*(kappa-2)+1 , (1-omega)*(kappa-2)+1 ) ;
                z ~ binomial (N, theta);  // naopako u odnosu na JAGS
            }
        '''

        self.dat = {'Nsubj': len(N),
                       'N': N,
                       'z': zz}

        self.init_dat = [{'theta': ThetaI,
                     'omega': meanThetaInit,
                     'kappaMinusTwo': kappaInitMinusTwo}] * 3

        self.samples = pystan.stan(model_code=self.modelString, data=self.dat, init=self.init_dat, iter=134333, warmup=1000, thin=20, chains=3)


def getStan(ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N):
    return TT_STAN(ThetaI, meanThetaInit, kappaInitMinusTwo,zz,N)