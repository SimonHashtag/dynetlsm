"""
Runs the time-inhomogeneous simulations found in the
paper 'A Bayesian nonparametric latent space approach to modeling evolving
communities in dynamic networks' by Joshua Loyal and Yuguo Chen
The synthetic network has weighted, directed edges.
"""

import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import dirname, join, abspath

from deco import concurrent, synchronized
from sklearn.utils import check_random_state
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from dynetlsm import DynamicNetworkHDPLPCM
from dynetlsm.datasets import synthetic_dynamic_network, synthetic_time_homogeneous_dynamic_network
from dynetlsm.model_selection.approx_bic import calculate_cluster_counts
from dynetlsm.metrics import variation_of_information, pseudo_R_squared
from dynetlsm.network_statistics import density, modularity
from dynetlsm.plots import plot_latent_space_synthetic


def counts_per_time_step(z):
    n_time_steps = z.shape[0]
    group_counts = np.zeros(n_time_steps, dtype=int)
    for t in range(n_time_steps):
        group_counts[t] = np.unique(z[t]).shape[0]

    return group_counts


def posterior_per_time_step(model):
    n_time_steps = model.Y_fit_.shape[0]
    probas = np.zeros((n_time_steps, model.n_components + 1))
    for t in range(n_time_steps):
        freq = model.posterior_group_counts_[t]
        index = model.posterior_group_ids_[t]
        probas[t, index] = freq / freq.sum()

    return probas

@concurrent
def benchmark_single(n_iter=10000, burn=5000, tune=1000,
                     outfile_name='benchmark', nufile_name=None,
                     iin_file_name=None, iout_file_name=None,
                     radii_in_file_name=None, radii_out_file_name=None,
                     random_state=None):
    iteration = random_state
    random_state = check_random_state(random_state)

    # generate simulated networks
    Y, X, z, intercept, radii, nu, _, _ = synthetic_dynamic_network(
        n_time_steps=9, n_nodes=100, is_directed=True,
        is_weighted=True, lmbda=0.8, intercept=1.0,
        sticky_const=20, random_state=random_state)

    # for t in range(9):
    #     fig, _ = plot_latent_space_synthetic(Y, X, z, radii, t=t, only_show_connected=False, show_edges=True)
    #     fig.show()

    # fit HDP-LPCM
    model = DynamicNetworkHDPLPCM(n_iter=n_iter,
                                  burn=burn,
                                  tune=tune,
                                  tune_interval=1000,
                                  is_directed=True,
                                  is_weighted=True,
                                  n_components=10,
                                  selection_type='vi',
                                  random_state=random_state).fit(Y)

    if os.path.exists(nufile_name + '.csv'):
        nu = pd.read_csv(nufile_name + '.csv')
        nu['nu_{}'.format(iteration)] = model.nus_[burn:]
    else:
        nu = pd.DataFrame(model.nus_[burn:], columns=['nu_{}'.format(iteration)])
    nu.to_csv(nufile_name + '.csv', index=False)

    if os.path.exists(iin_file_name + '.csv'):
        iin = pd.read_csv(iin_file_name + '.csv')
        iin['intercept_in_{}'.format(iteration)] = model.intercepts_[burn:, 0]
    else:
        iin = pd.DataFrame(model.intercepts_[burn:, 0], columns=['intercept_in_{}'.format(iteration)])
    iin.to_csv(iin_file_name + '.csv', index=False)

    if os.path.exists(iout_file_name + '.csv'):
        iout = pd.read_csv(iout_file_name + '.csv')
        iout['intercept_out_{}'.format(iteration)] = model.intercepts_[burn:, 1]
    else:
        iout = pd.DataFrame(model.intercepts_[burn:, 1], columns=['intercept_out_{}'.format(iteration)])
    iout.to_csv(iout_file_name + '.csv', index=False)

    if os.path.exists(radii_in_file_name + '.csv'):
        radii_in = pd.read_csv(radii_in_file_name + '.csv')
        radii_in['radii_in_{}'.format(iteration)] = model.radiis_[burn:, 0]
    else:
        radii_in = pd.DataFrame(model.radiis_[burn:, 0])
    radii_in.to_csv(radii_in_file_name + '.csv', index=False)

    if os.path.exists(radii_out_file_name + '.csv'):
        radii_out = pd.read_csv(radii_out_file_name + '.csv')
        radii_out['radii_out_{}'.format(iteration)] = model.radiis_[burn:, 1]
    else:
        radii_out = pd.DataFrame(model.radiis_[burn:, 1])
    radii_out.to_csv(radii_out_file_name + '.csv', index=False)

    # MAP: number of clusters per time point
    map_counts = counts_per_time_step(model.z_)

    # Posterior group count probabilities
    probas = posterior_per_time_step(model)

    # create dataframe of results
    results = pd.DataFrame(probas)
    results['map_counts'] = map_counts

    # goodness-of-fit metrics for MAP
    results['pseudo_R_sq'] = pseudo_R_squared(intercepts_mean=model.intercepts_mean_,
                                               X_mean=model.X_mean_,
                                               radii_mean=model.radii_mean_,
                                               nu_mean=model.nu_mean_)

    # clustering results
    results['vi'] = variation_of_information(z.ravel(), model.z_.ravel())
    results['vi_t1'] = variation_of_information(z[:3].ravel(), model.z_[:3].ravel())
    results['vi_t2'] = variation_of_information(z[3:6].ravel(), model.z_[3:6].ravel())
    results['vi_t3'] = variation_of_information(z[6:].ravel(), model.z_[6:].ravel())

    # time average VI
    vi = 0.
    for t in range(Y.shape[0]):
        vi += variation_of_information(z[t], model.z_[t])
    results['vi_avg'] = vi / Y.shape[0]

    results['rand_index'] = adjusted_rand_score(z.ravel(), model.z_.ravel())
    results['rand_t1'] = adjusted_rand_score(z[:3].ravel(), model.z_[:3].ravel())
    results['rand_t2'] = adjusted_rand_score(z[3:6].ravel(), model.z_[3:6].ravel())
    results['rand_t3'] = adjusted_rand_score(z[6:].ravel(), model.z_[6:].ravel())

    # time average rand
    adj_rand = 0.
    for t in range(Y.shape[0]):
        adj_rand += adjusted_rand_score(z[t], model.z_[t])
    results['rand_avg'] = adj_rand / Y.shape[0]

    results['AMI'] = adjusted_mutual_info_score(z.ravel(), model.z_.ravel())
    results['AMI_t1'] = adjusted_mutual_info_score(z[:3].ravel(), model.z_[:3].ravel())
    results['AMI_t2'] = adjusted_mutual_info_score(z[3:6].ravel(), model.z_[3:6].ravel())
    results['AMI_t3'] = adjusted_mutual_info_score(z[6:].ravel(), model.z_[6:].ravel())

    # time average adjusted mutual information (AMI)

    AMI = 0.
    for t in range(Y.shape[0]):
        AMI += adjusted_mutual_info_score(z[t], model.z_[t])
    results['AMI_avg'] = AMI / Y.shape[0]

    # info about simulated networks
    results['modularity'] = modularity(Y, z)
    results['modularity_t1'] = modularity(Y[:3], z[:3])
    results['modularity_t2'] = modularity(Y[3:6], z[3:6])
    results['modularity_t3'] = modularity(Y[6:], z[6:])

    results['density'] = density(Y)
    results['density_t1'] = density(Y[:3])
    results['density_t2'] = density(Y[3:6])
    results['density_t3'] = density(Y[6:])

    results.to_csv(outfile_name + '.csv', index=False)


@synchronized
def run_simulations(n_reps=30):
    # NOTE: This is meant to be run in parallel on a computer cluster!
    out_dir = 'results'

    # create a directory to store the results
    if not os.path.exists('results'):
        os.mkdir(out_dir)
    if not os.path.exists('results/benchmarks'):
        os.mkdir(os.path.join(out_dir, 'benchmarks'))

    for i in range(n_reps):
        benchmark_single(n_iter=100000, burn=80000, tune=20000, random_state=i,
                         outfile_name=os.path.join(
                            out_dir, 'benchmarks/benchmark_large_{}'.format(i)),
                         nufile_name=os.path.join('results/nu_large'),
                         iin_file_name=os.path.join('results/intercept_in_large'),
                         iout_file_name=os.path.join('results/intercept_out_large'),
                         radii_in_file_name=os.path.join('results/radii_in_large'),
                         radii_out_file_name=os.path.join('results/radii_out_large'))


def plot_simulations():
    figpath = abspath(join(dirname(__file__), '..', 'images', 'simulation'))

    # calculate median metric values
    n_time_steps = 9
    n_groups = 10

    n_files = len(glob.glob('results/*'))
    stat_names = ['pseudo_R_sq', 'vi_avg', 'vi_t1', 'vi_t2', 'vi_t3',
                  'rand_avg', 'rand_t1', 'rand_t2', 'rand_t3', 'AMI_avg', 'AMI_t1', 'AMI_t2', 'AMI_t3']
    data = np.zeros((n_files, len(stat_names)))
    for i, file_name in enumerate(glob.glob('results/benchmarks/*')):
        df = pd.read_csv(file_name)
        data[i] = df.loc[0, stat_names].values

    data = pd.DataFrame(data, columns=stat_names)
    print('Median Metrics:')
    print(data.median(axis=0))
    print('Metrics SD:')
    print(data.std(axis=0))

    data = data[['pseudo_R_sq', 'rand_avg', 'rand_t1', 'rand_t2', 'rand_t3', 'AMI_avg', 'AMI_t1', 'AMI_t2', 'AMI_t3']]
    data.columns = ['Pseudo R^2', 'Avg. ARI', 'ARI (t = 1 - 3)', 'ARI (t = 4 - 6)',
                    'ARI (t = 7 - 9)', 'Avg. AMI', 'AMI (t = 1 - 3)', 'AMI (t = 4 - 6)', 'AMI (t = 7 - 9)']

    # boxplots of the metrics
    plt.rc('font', family='sans-serif', size=24)
    fig, ax = plt.subplots(figsize=(10, 6))
    g = sns.boxplot(x='variable', y='value',
                    data=pd.melt(data), fliersize=2.0, ax=ax)
    g.tick_params(labelsize=8)
    sns.despine()
    ax.set(xlabel='', ylabel='')
    plt.savefig(join(figpath, 'performance_plot.png'), dpi=300)

    # clear figure
    plt.clf()

    # plot posterior boxplots
    data = {'probas': [], 'cluster_number': [], 't': []}
    for file_name in glob.glob('results/benchmarks/*'):
        df = pd.read_csv(file_name)
        for t in range(n_time_steps):
            for i in range(1, n_groups):
                data['probas'].append(df.iloc[t, i])
                data['cluster_number'].append(i)
                data['t'].append(t + 1)

    data = pd.DataFrame(data)

    plt.rc('font', family='sans-serif', size=16)
    g = sns.catplot(x='cluster_number', y='probas', col='t',
                    col_wrap=3, kind='box', data=data)

    for ax in g.axes:
        ax.set_ylabel('posterior probability')
        ax.set_xlabel('# of clusters')

    g.fig.tight_layout()

    plt.savefig(join(figpath, 'cluster_posterior.png'), dpi=300)

    # clear figure
    plt.clf()

    # plot selected number of groups for each simulation
    data = np.zeros((n_time_steps, n_groups), dtype=int)
    for sim_id, file_name in enumerate(glob.glob('results/benchmarks/*')):
        df = pd.read_csv(file_name)
        for t in range(n_time_steps):
            data[t, df.iloc[t, n_groups + 1] - 1] +=1

    data = pd.DataFrame(data,
        columns=range(1, n_groups + 1), index=range(1, n_time_steps + 1))
    mask = data.values == 0

    g = sns.heatmap(data, annot=True, cmap="Blues", cbar=False, mask=mask)
    g.set_xlabel('# of clusters')
    g.set_ylabel('t')
    plt.savefig(join(figpath, 'num_clusters.png'), dpi=300)

    # clear figure
    plt.clf()

    # plot posterior distributions of nu^2
    nu = pd.read_csv(os.path.join('results/nu_large.csv'))
    g = sns.kdeplot(data=nu, shade=False, legend=False)
    g.axvline(4, 0, 1, color='black', lw=2, ls='--')
    plt.savefig(join(figpath, 'nu_distribution_l.png'), dpi=300)

    # clear figure
    plt.clf()

    # plot posterior distributions of nu^2
    iin = pd.read_csv(os.path.join('results/intercept_in_large.csv'))
    g = sns.kdeplot(data=iin, shade=False, legend=False)
    g.axvline(3, 0, 1, color='black', lw=2, ls='--')
    plt.savefig(join(figpath, 'intercept_in_distribution_l.png'), dpi=300)

    # clear figure
    plt.clf()

    # plot posterior distributions of nu^2
    iout = pd.read_csv(os.path.join('results/intercept_out_large.csv'))
    g = sns.kdeplot(data=iout, shade=False, legend=False)
    g.axvline(1, 0, 1, color='black', lw=2, ls='--')
    plt.savefig(join(figpath, 'intercept_out_distribution_l.png'), dpi=300)

    # clear figure
    plt.clf()

    # Check for multicollinearity

    iout = pd.read_csv(os.path.join('results/intercept_out_large.csv'))
    iin = pd.read_csv(os.path.join('results/intercept_in_large.csv'))

    sum = []
    for i in range(iout.shape[1]):
        plus = iin.iloc[:,i].values + iout.iloc[:,i].values
        sum.append(plus)
    sum = pd.DataFrame(sum).transpose()
    g = sns.kdeplot(data=sum, shade=False, legend=False)
    g.axvline(4, 0, 1, color='black', lw=2, ls='--')
    plt.savefig(join(figpath, 'Sum1.png'), dpi=300)

    # clear figure
    plt.clf()

    iout = iout.stack().reset_index(drop=True)
    iin = iin.stack().reset_index(drop=True)

    corr = iin.corr(iout)
    print('The correlation between both intercept variables is:', corr)

    radii_out = pd.read_csv(os.path.join('results/radii_out_large.csv'))
    radii_in = pd.read_csv(os.path.join('results/radii_in_large.csv'))
    radii_out = radii_out.stack().reset_index(drop=True)
    radii_in = radii_in.stack().reset_index(drop=True)

    corr_radii = radii_in.corr(radii_out)
    print('The correlation between both radii distributions is:', corr_radii)


if __name__ == "__main__":
    run_simulations(10)
    plot_simulations()
