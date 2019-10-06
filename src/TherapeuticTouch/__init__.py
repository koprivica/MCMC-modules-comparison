import pandas as pd
import numpy as np
# from .JAGS import * as jg
# from .STAN import * as st
import TherapeuticTouch.JAGS as jg
import TherapeuticTouch.STAN as st

import time
import matplotlib.pyplot as plt
import itertools
import scipy

def main():
    # Data - Creating Panas Data frame with two columns
    my_data = pd.read_csv("../data/TherapeuticTouchData.csv", dtype={'y': '<i4', 's': 'category'})

    y = my_data.y

    # Creating Pandas Index, necessary for next step
    cat_columns = my_data.select_dtypes(['category']).columns

    # change category names to numbers
    my_data[cat_columns] = my_data[cat_columns].apply(lambda x: x.cat.codes)

    # Make 's' again a category
    my_data['s'] = my_data['s'].astype('category')

    s = my_data.s.cat.codes

    zz = my_data.groupby('s')['y'].sum()
    N = my_data.groupby('s')['y'].count()

    # Initialize the chain by finding MLE
    def getInitValues(values, sumOFValues, groupCount):
        # def getInitValues (my_data, groupCount):
        ThetaI, initList = [0] * len(groupCount), [0] * len(groupCount)

        for ss in range(0, len(groupCount)):
            initList[ss] = np.random.choice((my_data.y[my_data.s == ss]).values, size=groupCount[ss], replace=True,
                                            p=None)
            ThetaI[ss] = sum(initList[ss]) / len(initList[ss])
            ThetaI[ss] = 0.001 + 0.998 * ThetaI[ss]  # keep away from 0,1
            ss += 1

        meanThetaInit = np.mean(ThetaI)
        kappaInit = 100  # lazy, start high and let burn-in find better value
        return ThetaI, meanThetaInit, kappaInit - 2

    ThetaI, meanThetaInit, kappaInitMinusTwo = getInitValues(my_data, zz, N)


    time1 = time.time()
    jags = jg.TT_JAGS(ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N)
    time2 = time.time()

    time3 = time.time()
    stan = st.TT_STAN(ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N)
    time4 = time.time()

    # Necessary because of overlaping issues
    import TherapeuticTouch.PYMC3 as pymc
    # from .PYMC3 import TT_PYMC3 as pymc

    time5 = time.time()
    pymc = pymc.TT_PYMC3(ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N)
    time6 = time.time()

    print(time2 - time1)
    print(time4 - time3)
    print(time6 - time5)


    bins = np.linspace(0, 1, 50)
    bins2 = np.linspace(0, 300, 45)


    '''THETA 1 - THETA 28'''

    fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)
    big_axes[0].set_title("JAGS : {} seconds".format(time2 - time1), fontsize=16)
    big_axes[0].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    big_axes[0]._frameon = False

    ax0 = fig.add_subplot(3, 2, 1)
    ax0.set_title('theta 1')
    ax0.hist(list(itertools.chain.from_iterable(jags.samples['theta'][0, :])), bins, alpha=0.5, ec='white', density=1)

    ax01 = fig.add_subplot(3, 2, 2)
    ax01.set_title('theta 28')
    ax01.hist(list(itertools.chain.from_iterable(jags.samples['theta'][27, :])), bins, alpha=0.5, ec='white', density=1)

    big_axes[1].set_title("STAN : {} seconds".format(time4 - time3), fontsize=16)
    big_axes[1].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    big_axes[1]._frameon = False

    ax1 = fig.add_subplot(3, 2, 3)
    ax1.set_title('theta 1')
    ax1.hist(stan.samples.extract(permuted=True)['theta'][:, 0], bins, alpha=0.5, ec='white', density=1)

    ax11 = fig.add_subplot(3, 2, 4)
    ax11.set_title('theta 28')
    ax11.hist(stan.samples.extract(permuted=True)['theta'][:, 27], bins, alpha=0.5, ec='white', density=1)

    big_axes[2].set_title("PYMC3 : {} seconds".format(time6 - time5), fontsize=16)
    big_axes[2].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    big_axes[2]._frameon = False

    ax2 = fig.add_subplot(3, 2, 5)
    ax2.set_title('theta 1')
    ax2.hist(pymc.samples['theta'][:, 0][1000::20], bins, alpha=0.5, ec='white', density=1)

    ax21 = fig.add_subplot(3, 2, 6)
    ax21.set_title('theta 28')
    ax21.hist(pymc.samples['theta'][:, 27][1000::20], bins, alpha=0.5, ec='white', density=1)

    fig.set_facecolor('w')
    plt.tight_layout()
    plt.show()
    plt.savefig('../figures/Therapeutic-Touch_THETA.png')


    '''OMEGA - KAPPA'''

    fig1, big_axes1 = plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)
    big_axes1[0].set_title("JAGS", fontsize=16)
    big_axes1[0].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    big_axes1[0]._frameon = False

    plt.suptitle('OMEGA - KAPPA', fontsize=20)

    ax0 = fig1.add_subplot(3, 2, 1)
    ax0.hist(np.concatenate(list(itertools.chain.from_iterable(jags.samples['omega'][:])), axis=0), bins, alpha=0.5,
             ec='white', density=1)
    ax0.set_title('omega')

    ax01 = fig1.add_subplot(3, 2, 2)
    ax01.hist(np.concatenate(list(itertools.chain.from_iterable(jags.samples['kappa'][:])), axis=0), bins2, alpha=0.5,
              ec='white', density=1)  # FIX BEANS
    ax01.set_title('kappa')

    big_axes1[1].set_title("STAN", fontsize=16)
    big_axes1[1].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    big_axes1[1]._frameon = False

    ax1 = fig1.add_subplot(3, 2, 3)
    ax1.hist(stan.samples.extract(permuted=True)['omega'][:], bins, alpha=0.5, ec='white', density=1)
    ax1.set_title('omega')

    ax11 = fig1.add_subplot(3, 2, 4)
    ax11.hist(stan.samples.extract(permuted=True)['kappa'][:], bins2, alpha=0.5, ec='white', density=1)
    ax11.set_title('kappa')

    big_axes1[2].set_title("PYMC3", fontsize=16)
    big_axes1[2].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    big_axes1[2]._frameon = False

    ax2 = fig1.add_subplot(3, 2, 5)
    ax2.hist(pymc.samples['omega'][:][1000::20], bins, alpha=0.5, ec='white', density=1)
    ax2.set_title('omega')

    ax21 = fig1.add_subplot(3, 2, 6)
    ax21.hist(pymc.samples['kappa'][:][1000::20], bins2, alpha=0.5, ec='white', density=1)
    ax21.set_title('kappa')

    plt.tight_layout()
    plt.show()
    plt.savefig('../figures/Therapeutic-Touch_OMEGA-KAPPA.png')


    '''T-test on samples'''

    jags_theta1 = list(itertools.chain.from_iterable(jags.samples['theta'][0, :]))
    stan_theta1 = stan.samples.extract(permuted=True)['theta'][:, 0]
    pymc_theta1 = pymc.samples['theta'][:, 0][1000::20]

    jags_stan1 = scipy.stats.ttest_ind(jags_theta1, stan_theta1, axis=0, equal_var=False)
    jags_pymc1 = scipy.stats.ttest_ind(jags_theta1, pymc_theta1, axis=0, equal_var=False)
    stan_pymc1 = scipy.stats.ttest_ind(stan_theta1, pymc_theta1, axis=0, equal_var=False)

    jags_theta28 = list(itertools.chain.from_iterable(jags.samples['theta'][27, :]))
    stan_theta28 = stan.samples.extract(permuted=True)['theta'][:, 27]
    pymc_theta28 = pymc.samples['theta'][:, 27][1000::20]

    jags_stan28 = scipy.stats.ttest_ind(jags_theta28, stan_theta28, axis=0, equal_var=False)
    jags_pymc28 = scipy.stats.ttest_ind(jags_theta28, pymc_theta28, axis=0, equal_var=False)
    stan_pymc28 = scipy.stats.ttest_ind(stan_theta28, pymc_theta28, axis=0, equal_var=False)