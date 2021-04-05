from dynetlsm.datasets.load_armstrade import importData, loadDeals
from dynetlsm import DynamicNetworkHDPLPCM
from dynetlsm.plots import (
    plot_traces,
    alluvial_plot,
    plot_latent_space,
    plot_adjacency_matrix
)

from os.path import dirname, join, abspath
import pickle

# Specification of directories and paths
currentPath = dirname(__file__)
figpath = abspath(join(currentPath, '..', 'images', 'armstrade'))

importData(filename= 'armstrade1950_2020.csv', weighted = False, dropcountry = True)
L, names = loadDeals()

# # 1. Compute new model and dump it in an object file in datapath
# model = DynamicNetworkHDPLPCM(n_iter=1000,
#                               tune=500,
#                               burn=500,
#                               tune_interval=1000,
#                               random_state=42,
#                               n_components=25,
#                               selection_type='vi',
#                               is_directed=True).fit(L)
#
# modelStorage = open(join(currentPath, 'model.obj'), 'wb')
# pickle.dump(model, modelStorage)

# 2. Retrieve old model
modelStorage = open(join(currentPath, 'model.obj'), 'rb')
model = pickle.load(modelStorage)

# Trace plots
fig, ax = plot_traces(model, figsize=(10, 12))
fig.savefig(join(figpath, 'armstrade_traces.png'), dpi=300)

# alluvial diagram
fig, ax = alluvial_plot(model.z_, figsize=(10, 5))
fig.savefig(join(figpath, 'armstrade_alluvial.png'), dpi=300)

# # Adjacency matrix
# fig, ax = plot_adjacency_matrix(L[1], model.z_)
# fig.savefig(join(figpath, 'armstrade_adjacency.png'), dpi=300)

# latent space visualizations
for t in range(L.shape[0]):
    fig, ax = plot_latent_space(
        model, figsize=(15, 15), t=t,
        textsize=15,
        node_size=5,
        mutation_scale=10,
        linewidth=1.0,
        connectionstyle='arc3,rad=0.2',
        title_text=None,
        plot_group_sigma=True,
        node_names=names,
        node_textsize=5,
        repel_strength=0.0012,
        #mask_groups=[[0, 3]],  # NOTE: this may not be background on other settings!
        only_show_connected=True,
        number_nodes=True,
        use_radii=True,
        border=0)
    fig.savefig(join(figpath, 'armstrade_latent_space_t{}.png'.format(t)), dpi=300)

# # countries to drop
# dropcountry = ['Unknown supplier(s)', 'European Union**', 'Europe multi-state', '(multiple sellers)', 'African Union**',
#                'Unknown recipient(s)', 'Unknown country', 'NATO**', 'Ukraine Rebels*', 'United Wa State (Myanmar)*', 'Houthi rebels (Yemen)*', 'United Nations**',
#                'Darfur rebels (Sudan)*', 'Hamas (Palestine)*', 'PRC (Israel/Palestine)*', 'NTC (Libya)*', 'Syria rebels*', 'OSCE**',
#                'UIC (Somalia)*', 'Hezbollah (Lebanon)*', 'Northern Alliance (Afghanistan)*', 'Southern rebels (Yemen)*',
#                'LTTE (Sri Lanka)*', 'MTA (Myanmar)*', 'SNA (Somalia)*', 'FMLN (El Salvador)*', 'SLA (Lebanon)*', 'Mujahedin (Afghanistan)*',
#                'LF (Lebanon)*', 'RUF (Sierra Leone)*', 'Regional Security System**', 'Khmer Rouge (Cambodia)*', 'PLO (Israel)*',
#                'UNITA (Angola)*', 'GUNT (Chad)*', 'Provisional IRA (UK)*', 'ANC (South Africa)*', 'Amal (Lebanon)*',
#                'Lebanon Palestinian rebels*', 'Contras (Nicaragua)*', 'Katanga', 'FNLA (Angola)*', 'Viet Minh (France)*',
#                'MNLF (Philippines)*', 'Pathet Lao (Laos)*', 'Viet Cong (South Vietnam)*', 'ZAPU (Zimbabwe)*', 'ELF (Ethiopia)*',
#                'Anti-Castro rebels (Cuba)*', 'Armas (Guatemala)*', 'Haiti rebels*', 'Indonesia rebels*']
# pd.DataFrame(dropcountry).to_csv(abspath(join(currentPath, '..', 'dynetlsm', 'datasets', 'raw_data', 'armstrade', 'dropcountry.csv')), index=None, header=['Countries'])