from dynetlsm.datasets.load_armstrade import importData, loadDeals
from dynetlsm import DynamicNetworkHDPLPCM
from dynetlsm import DynamicNetworkLSM
from dynetlsm.plots import (
    plot_traces,
    alluvial_plot,
    plot_latent_space,
    plot_adjacency_matrix,
    create_summaryplots,
    create_networkplots
)
from dynetlsm.datasets import synthetic_dynamic_network, synthetic_time_homogeneous_dynamic_network, synthetic_clusterfree_dynamic_network
from dynetlsm.metrics import variation_of_information, pseudo_R_squared

from os.path import dirname, join, abspath
import pickle
import numpy as np
import pandas as pd

# Specification of directories and paths
currentPath = dirname(__file__)
datapath = abspath(join(currentPath, '..', 'dynetlsm', 'datasets', 'raw_data', 'armstrade'))
figpath = abspath(join(currentPath, '..', 'images', 'armstrade'))
#modelpath = 'D:\Dokumente\Studium\Masterarbeit'

# Specification of color palettes
# colors2 = ["#08D63D", "black", "purple", "black", "#FF5733", "black", "#ff0000", "#0847D6", "#FFC300", "black"]
# colors = ["#08D63D", "black", "purple", "black", "#FF5733", "black", "#fa6d6d", "#73a8f4", "#FFC300", "black"]
colorslarge = ["#FFC300", "#00A300", "#fa6d6d", "#73a8f4"]
# colors3 = ['#FFC300', '#0000FF', '#FF0000', '#0000FF']

# green: #08D63D; dark green: #00A300; orange: #FF5733; light red: #fa6d6d, light blue: #73a8f4, yellow: #FFC300

#importData(filename='armstrade1950_2020.csv', weighted=True, dropcountry=True)
L, names = loadDeals()

 # 1. Compute new model and dump it in an object file in datapath
model = DynamicNetworkHDPLPCM(n_iter=10000,
                              tune=5000,
                              burn=5000,
                              tune_interval=1000,
                              random_state=42,
                              n_components=10,
                              selection_type='vi',
                              is_directed=True,
                              is_weighted=True).fit(L)

# model = DynamicNetworkLSM(n_iter=1000,
#                           n_features=2,
#                           tune=1000,
#                           burn=1000,
#                           is_directed=True,
#                           is_weighted=True,
#                           random_state=42).fit(L)

#
# modelStorage = open(join(modelpath, 'model.obj'), 'wb')
# pickle.dump(model, modelStorage)

# # 2. Retrieve old model
# modelStorage = open(join(modelpath, 'modellarge.obj'), 'rb')
# model = pickle.load(modelStorage)

# Summary plots similar to Akerman & Seim (2014)
create_summaryplots(csv=join(datapath, 'cleanarmstrade1950_2020.csv'), figpath=figpath, rolling=False)

# Network plots without clustering
create_networkplots(csv=join(datapath, 'cleanarmstrade1950_2020.csv'), yearstart=1950, yearend=1954, figpath=figpath)
create_networkplots(csv=join(datapath, 'cleanarmstrade1950_2020.csv'), yearstart=2015, yearend=2019, figpath=figpath)

# Trace plots
fig, ax = plot_traces(model, figsize=(10, 15))
fig.savefig(join(figpath, 'armstrade_no_clusters.png'), dpi=300)

# Alluvial diagram
fig, ax = alluvial_plot(model.z_, colors=colorslarge, figsize=(10, 5))
fig.savefig(join(figpath, 'armstrade_alluvial.png'), dpi=300)

# Adjacency matrix
fig, ax = plot_adjacency_matrix(L[0,:,:], model.z_[0,:])
fig.savefig(join(figpath, 'armstrade_adjacency.png'), dpi=300)

# latent space visualizations

for t in range(L.shape[0]):
    fig, ax = plot_latent_space(
        model, figsize=(15, 15), t=t,
        node_size=35,
        mutation_scale=10,
        linewidth=1.0,
        connectionstyle='arc3,rad=0.2',
        node_names=names,
        node_textsize=6.5,
        repel_strength=0.0019,
        mask_groups=[0, 1],  # NOTE: this may not be background on other settings!
        only_show_connected=True,
        number_nodes=True,
        use_radii=False,
        border=0,
        colors=colorslarge,
        title_text=''
    )
    fig.savefig(join(figpath, 'armstrade_latent_space_t{}.png'.format(t)), dpi=300)



# #Determining the self-transition probability and the share of zero-valued links in the simulation study
# percentage0=[]
# selftrans=0
#
# for random_state in range(30):
#     Y, X, z, intercept, radii, nu, _, _ = synthetic_dynamic_network(
#         n_time_steps=9, n_nodes=120, is_directed=True,
#         is_weighted=True, lmbda=0.8, intercept=1.0,
#         sticky_const=20, random_state=random_state)
#     percentage0.append(len(Y[Y==0])/(9*120*120))
#     for i in range(z.shape[1]):
#         for j in range(z.shape[0]-1):
#             if z[j, i] == z[j + 1, i]:
#                 selftrans += 1
#
# selftrans/=(9*120*30)
# percavg = np.mean(percentage0)
# std = np.std(percentage0)

