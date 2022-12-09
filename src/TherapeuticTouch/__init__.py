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

def main(fileNumber):
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


    # time1 = time.time()
    # jags = jg.TT_JAGS(ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N)
    # time2 = time.time()
    #
    # time3 = time.time()
    # stan = st.TT_STAN(ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N)
    # time4 = time.time()

    # Necessary because of overlaping issues
    import TherapeuticTouch.PYMC3 as pymc
    # from .PYMC3 import TT_PYMC3 as pymc

    time5 = time.time()
    pymc = pymc.TT_PYMC3(ThetaI, meanThetaInit, kappaInitMinusTwo, zz, N)
    time6 = time.time()

    # print(time2 - time1)
    # print(time4 - time3)
    print('PYMC3:' + str(time6 - time5))


    bins = np.linspace(0, 1, 50)
    bins2 = np.linspace(0, 300, 45)


    # '''THETA 1 - THETA 28'''
    #
    # fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)
    # big_axes[0].set_title("JAGS : {} seconds".format(time2 - time1), fontsize=16)
    # big_axes[0].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # big_axes[0]._frameon = False
    #
    # ax0 = fig.add_subplot(3, 2, 1)
    # ax0.set_title('theta 1')
    # ax0.hist(list(itertools.chain.from_iterable(jags.samples['theta'][0, :])), bins, alpha=0.5, ec='white', density=1)
    #
    # ax01 = fig.add_subplot(3, 2, 2)
    # ax01.set_title('theta 28')
    # ax01.hist(list(itertools.chain.from_iterable(jags.samples['theta'][27, :])), bins, alpha=0.5, ec='white', density=1)
    #
    # big_axes[1].set_title("STAN : {} seconds".format(time4 - time3), fontsize=16)
    # big_axes[1].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # big_axes[1]._frameon = False
    #
    # ax1 = fig.add_subplot(3, 2, 3)
    # ax1.set_title('theta 1')
    # ax1.hist(stan.samples.extract(permuted=True)['theta'][:, 0], bins, alpha=0.5, ec='white', density=1)
    #
    # ax11 = fig.add_subplot(3, 2, 4)
    # ax11.set_title('theta 28')
    # ax11.hist(stan.samples.extract(permuted=True)['theta'][:, 27], bins, alpha=0.5, ec='white', density=1)
    #
    # big_axes[2].set_title("PYMC3 : {} seconds".format(time6 - time5), fontsize=16)
    # big_axes[2].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # big_axes[2]._frameon = False
    #
    # ax2 = fig.add_subplot(3, 2, 5)
    # ax2.set_title('theta 1')
    # ax2.hist(pymc.samples['theta'][:, 0][1000::20], bins, alpha=0.5, ec='white', density=1)
    #
    # ax21 = fig.add_subplot(3, 2, 6)
    # ax21.set_title('theta 28')
    # ax21.hist(pymc.samples['theta'][:, 27][1000::20], bins, alpha=0.5, ec='white', density=1)
    #
    # fig.set_facecolor('w')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('../figures/Therapeutic-Touch_THETA.png')


    # '''OMEGA - KAPPA'''
    #
    # fig1, big_axes1 = plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)
    # big_axes1[0].set_title("JAGS", fontsize=16)
    # big_axes1[0].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # big_axes1[0]._frameon = False
    #
    # plt.suptitle('OMEGA - KAPPA', fontsize=20)
    #
    # ax0 = fig1.add_subplot(3, 2, 1)
    # ax0.hist(np.concatenate(list(itertools.chain.from_iterable(jags.samples['omega'][:])), axis=0), bins, alpha=0.5,
    #          ec='white', density=1)
    # ax0.set_title('omega')
    #
    # ax01 = fig1.add_subplot(3, 2, 2)
    # ax01.hist(np.concatenate(list(itertools.chain.from_iterable(jags.samples['kappa'][:])), axis=0), bins2, alpha=0.5,
    #           ec='white', density=1)  # FIX BEANS
    # ax01.set_title('kappa')
    #
    # big_axes1[1].set_title("STAN", fontsize=16)
    # big_axes1[1].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # big_axes1[1]._frameon = False
    #
    # ax1 = fig1.add_subplot(3, 2, 3)
    # ax1.hist(stan.samples.extract(permuted=True)['omega'][:], bins, alpha=0.5, ec='white', density=1)
    # ax1.set_title('omega')
    #
    # ax11 = fig1.add_subplot(3, 2, 4)
    # ax11.hist(stan.samples.extract(permuted=True)['kappa'][:], bins2, alpha=0.5, ec='white', density=1)
    # ax11.set_title('kappa')
    #
    # big_axes1[2].set_title("PYMC3", fontsize=16)
    # big_axes1[2].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # big_axes1[2]._frameon = False
    #
    # ax2 = fig1.add_subplot(3, 2, 5)
    # ax2.hist(pymc.samples['omega'][:][1000::20], bins, alpha=0.5, ec='white', density=1)
    # ax2.set_title('omega')
    #
    # ax21 = fig1.add_subplot(3, 2, 6)
    # ax21.hist(pymc.samples['kappa'][:][1000::20], bins2, alpha=0.5, ec='white', density=1)
    # ax21.set_title('kappa')
    #
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('../figures/Therapeutic-Touch_OMEGA-KAPPA.png')


    # '''T-test on samples'''
    #
    # jags_theta1 = list(itertools.chain.from_iterable(jags.samples['theta'][0, :]))
    # stan_theta1 = stan.samples.extract(permuted=True)['theta'][:, 0]
    # pymc_theta1 = pymc.samples['theta'][:, 0][1000::20]
    #
    # jags_stan1 = scipy.stats.ttest_ind(jags_theta1, stan_theta1, axis=0, equal_var=False)
    # jags_pymc1 = scipy.stats.ttest_ind(jags_theta1, pymc_theta1, axis=0, equal_var=False)
    # stan_pymc1 = scipy.stats.ttest_ind(stan_theta1, pymc_theta1, axis=0, equal_var=False)
    #
    # jags_theta28 = list(itertools.chain.from_iterable(jags.samples['theta'][27, :]))
    # stan_theta28 = stan.samples.extract(permuted=True)['theta'][:, 27]
    # pymc_theta28 = pymc.samples['theta'][:, 27][1000::20]
    #
    # jags_stan28 = scipy.stats.ttest_ind(jags_theta28, stan_theta28, axis=0, equal_var=False)
    # jags_pymc28 = scipy.stats.ttest_ind(jags_theta28, pymc_theta28, axis=0, equal_var=False)
    # stan_pymc28 = scipy.stats.ttest_ind(stan_theta28, pymc_theta28, axis=0, equal_var=False)

    '''Save results to file'''

    import csv

    # ''' JAGS '''
    # np.savetxt('../results/jags/theta/test_' + str(fileNumber) + '.csv', np.c_[list(itertools.chain.from_iterable(jags.samples['theta'][0, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][1, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][2, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][3, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][4, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][5, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][6, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][7, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][8, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][9, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][10, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][11, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][12, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][13, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][14, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][15, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][16, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][17, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][18, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][19, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][20, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][21, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][22, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][23, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][24, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][25, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][26, :])),
    #                                                            list(itertools.chain.from_iterable(jags.samples['theta'][27, :])) #,
    #                                                            # list(itertools.chain.from_iterable(jags.samples['omega'][:])),
    #                                                            # list(itertools.chain.from_iterable(jags.samples['kappa'][:]))
    #                                                            ],
    #            header="theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9,theta10,theta11,theta12,theta13,theta14,theta15,theta16,theta17,theta18,theta19,theta20,theta21,theta22,theta23,theta24,theta25,theta26,theta27", # ,omega,kappa",
    #            delimiter=',',comments='')
    #
    # np.savetxt('../results/jags/omega/test_' + str(fileNumber) + '.csv', np.c_[list(itertools.chain.from_iterable(jags.samples['omega'][:]))],header="omega",delimiter=',',comments='')
    #
    # np.savetxt('../results/jags/kappa/test_' + str(fileNumber) + '.csv', np.c_[list(itertools.chain.from_iterable(jags.samples['kappa'][:]))],header="kappa",delimiter=',',comments='')
    #
    # ''' STAN '''
    #
    # np.savetxt('../results/stan/theta/test_' + str(fileNumber) + '.csv', np.c_[stan.samples.extract(permuted=True)['theta'][:, 0],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 1],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 2],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 3],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 4],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 5],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 6],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 7],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 8],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 9],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 10],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 11],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 12],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 13],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 14],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 15],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 16],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 17],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 18],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 19],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 20],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 21],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 22],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 23],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 24],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 25],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 26],
    #                                                               stan.samples.extract(permuted=True)['theta'][:, 27] #,
    #                                                               # stan.samples.extract(permuted=True)['omega'][:],
    #                                                               # stan.samples.extract(permuted=True)['kappa'][:]
    #                                                            ],
    #            header="theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9,theta10,theta11,theta12,theta13,theta14,theta15,theta16,theta17,theta18,theta19,theta20,theta21,theta22,theta23,theta24,theta25,theta26,theta27", #,omega,kappa",
    #            delimiter=',',comments='')
    #
    # np.savetxt('../results/stan/omega/test_' + str(fileNumber) + '.csv',
    #            np.c_[stan.samples.extract(permuted=True)['omega'][:]], header="omega", delimiter=',',
    #            comments='')
    #
    # np.savetxt('../results/stan/kappa/test_' + str(fileNumber) + '.csv',
    #            np.c_[stan.samples.extract(permuted=True)['kappa'][:]], header="kappa", delimiter=',',
    #            comments='')

    ''' PYMC '''

    np.savetxt('../results/pymc/theta/test_' + str(fileNumber) + '.csv', np.c_[pymc.samples['theta'][:, 0][::20],
                                                               pymc.samples['theta'][:, 1][::20],
                                                               pymc.samples['theta'][:, 2][::20],
                                                               pymc.samples['theta'][:, 3][::20],
                                                               pymc.samples['theta'][:, 4][::20],
                                                               pymc.samples['theta'][:, 5][::20],
                                                               pymc.samples['theta'][:, 6][::20],
                                                               pymc.samples['theta'][:, 7][::20],
                                                               pymc.samples['theta'][:, 8][::20],
                                                               pymc.samples['theta'][:, 9][::20],
                                                               pymc.samples['theta'][:, 10][::20],
                                                               pymc.samples['theta'][:, 11][::20],
                                                               pymc.samples['theta'][:, 12][::20],
                                                               pymc.samples['theta'][:, 13][::20],
                                                               pymc.samples['theta'][:, 14][::20],
                                                               pymc.samples['theta'][:, 15][::20],
                                                               pymc.samples['theta'][:, 16][::20],
                                                               pymc.samples['theta'][:, 17][::20],
                                                               pymc.samples['theta'][:, 18][::20],
                                                               pymc.samples['theta'][:, 19][::20],
                                                               pymc.samples['theta'][:, 20][::20],
                                                               pymc.samples['theta'][:, 21][::20],
                                                               pymc.samples['theta'][:, 22][::20],
                                                               pymc.samples['theta'][:, 23][::20],
                                                               pymc.samples['theta'][:, 24][::20],
                                                               pymc.samples['theta'][:, 25][::20],
                                                               pymc.samples['theta'][:, 26][::20],
                                                               pymc.samples['theta'][:, 27][::20] #,
                                                               # pymc.samples['omega'][:][1000::20],
                                                               # pymc.samples['kappa'][:][1000::20]
                                                               ],
               header="theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9,theta10,theta11,theta12,theta13,theta14,theta15,theta16,theta17,theta18,theta19,theta20,theta21,theta22,theta23,theta24,theta25,theta26,theta27", #,omega,kappa",
               delimiter=',',comments='')

    np.savetxt('../results/pymc/omega/test_' + str(fileNumber) + '.csv',
               np.c_[pymc.samples['omega'][:][::20]], header="omega", delimiter=',',
               comments='')

    np.savetxt('../results/pymc/kappa/test_' + str(fileNumber) + '.csv',
               np.c_[pymc.samples['kappa'][:][::20]], header="kappa", delimiter=',',
               comments='')

    # jags_f = open("../results/jags_" + fileNumber + ".csv", "w+")
    #
    # jags_f.close()
    #
    # stan_f = open("../results/stan_" + fileNumber + ".csv", "w+")
    #
    # stan_f.close()
    #
    # pymc_f = open("../results/pymc_" + fileNumber + ".csv", "w+")
    #
    # pymc_f.close()
