import numpy as np
import pandas as pd
import networkx as nx

from os.path import dirname, join, abspath

__all__ = ['importData', 'loadDeals']

def loadDeals(min_degree: int = 1, remove_periphery:bool = True):
    currentPath = dirname(__file__)
    dataPath = abspath(join(currentPath, 'raw_data', 'armstrade'))

    countNodes = 199
    countYears = 5
    L = np.zeros((countYears, countNodes, countNodes))

    file_fmt = 'AdjacencyMatrix_{}.npy'

    for t, year in enumerate(range(1995,2020, 5)):
        L[t] = np.load(join(dataPath, file_fmt.format(year)))

    if remove_periphery:
        for t in range(L.shape[0]):
            G = nx.from_numpy_array(L[t])
            core_id = np.asarray(list(nx.core_number(G).values()))
            mask = np.where(core_id <= 2)[0]
            L[t, mask] = 0
            L[t, :, mask] = 0

    # a country must be active for at least min_degree
    active_ids = np.where(
        (L.sum(axis=(0, 1)) + L.sum(axis=(0, 2))) >= min_degree)[0]
    L = np.ascontiguousarray(L[:, active_ids][:, :, active_ids])

    # load country names
    names = pd.read_csv(join(dataPath, 'names.csv'))
    names = names.values.ravel()[active_ids]

    return np.ascontiguousarray(L), names

def importData(filename:str, weighted:bool = False, minimal: bool = True, dropcountry: bool = True, oceania: bool = False, caribbean: bool = False):
    currentPath = dirname(__file__)
    dataPath = abspath(join(currentPath, 'raw_data', 'armstrade'))
    filePath = join(dataPath, filename)

    drop = pd.read_csv(join(dataPath, 'dropcountry.csv')).values.ravel()

    data = pd.read_csv(filePath, sep=';', decimal='.', header=0)
    data.loc[data['Order date'].isna(), 'Order date'] = data.loc[
        data['Order date'].isna(), 'Order date is estimate']
    data.loc[data['Numbers delivered'].isna(), 'Numbers delivered'] = data.loc[
        data['Numbers delivered'].isna(), 'Numbers delivered is estimate']
    data.loc[data['Delivery year'].isna(), 'Delivery year'] = data.loc[
        data['Delivery year'].isna(), 'Delivery year is estimate']
    data.drop(['Designation', 'Description', 'Armament category', 'Order date is estimate',
               'Numbers delivered is estimate', 'Delivery year is estimate', 'Status', 'SIPRI estimate',
               'TIV deal unit', 'Local production'], inplace=True, axis=1)
    data.dropna(axis=0, inplace=True)
    data[['Seller', 'Buyer']] = data[['Seller', 'Buyer']].astype(str)
    data[['Order date', 'Numbers delivered', 'Delivery year']] = data[
        ['Order date', 'Numbers delivered', 'Delivery year']].astype(int)

    if minimal:
        data.drop(['Deal ID', 'Order date', 'Numbers delivered'], inplace=True, axis=1)

    if dropcountry:
        for entry in drop:
            data = data.drop(data[(data['Seller'] == entry) | (data['Buyer'] == entry)].index)

    # Summarise oceanic countries as oceania if oceania==True. Analogous for caribbean further below.
    if oceania:
        ocean = ['Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Palau', 'Papua New Guinea', 'Samoa',
                 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu', 'Nauru']
        for entry in ocean:
            data.loc[data['Seller'] == entry, 'Seller'] = 'Oceania'
            data.loc[data['Buyer'] == entry, 'Buyer'] = 'Oceania'

    if caribbean:
        carib = ['Antigua & Barbuda', 'Aruba', 'Bahamas', 'Barbados', 'Cayman Islands', 'Cuba', 'Dominica',
                 'Dominican Republic', 'Grenada', 'Guadeloupe', 'Haiti', 'Jamaica', 'Martinique', 'Puerto Rico',
                 'Saint Barthélemy', 'St. Kitts & Nevis', 'Saint Kitts & Nevis', 'St. Lucia', 'Saint Lucia',
                 'St. Vincent and the Grenadines', 'Saint Vincent and the Grenadines', 'Saint Vincent',
                 'Trinidad & Tobago', 'Turks & Caicos Islands', 'Virgin Islands', 'Trinidad and Tobago']
        for entry in carib:
            data.loc[data['Seller'] == entry, 'Seller'] = 'Caribbean'
            data.loc[data['Buyer'] == entry, 'Buyer'] = 'Caribbean'

    # Summarise two country names as Libya
    data.loc[(data['Seller'] == 'Libya HoR') | (data['Seller'] == 'Libya GNC'), 'Seller'] = 'Libya'
    data.loc[(data['Buyer'] == 'Libya HoR') | (data['Buyer'] == 'Libya GNC'), 'Buyer'] = 'Libya'
    # Drop two transactions that are components of the network
    data = data.drop(data[(data['Seller'] == 'Costa Rica') & (data['Buyer'] == 'Guyana')].index)
    data = data.drop(data[(data['Seller'] == 'Malawi') & (data['Buyer'] == 'Cabo Verde')].index)

    cleanPath = join(dataPath, 'clean' + filename)
    data.to_csv(cleanPath, index=None)

    nodelist = nx.from_pandas_edgelist(data, 'Seller', 'Buyer', edge_attr='TIV delivery values', create_using=nx.DiGraph()).nodes()
    pd.DataFrame(nodelist).to_csv(join(dataPath, 'names.csv'), index=None, header = ['Countries'])

    for t, year in enumerate(range(1950, 2020, 5)):
        periodEnd = year + 4
        deals = data.loc[(data['Delivery year'] >= year) & (data['Delivery year'] <= periodEnd)]
        if weighted == True:
            deals = deals.groupby(['Seller', 'Buyer'])['TIV delivery values'].sum().reset_index()
            G = nx.from_pandas_edgelist(deals, 'Seller', 'Buyer', edge_attr='TIV delivery values',
                                        create_using=nx.DiGraph())
            adjMatrix = nx.to_numpy_array(G, weight='TIV delivery values', nodelist = nodelist)
        else:
            deals = deals.value_counts(['Seller', 'Buyer']).reset_index(name='Number of deals')
            G = nx.from_pandas_edgelist(deals, 'Seller', 'Buyer', edge_attr=None,
                                        create_using=nx.DiGraph())
            adjMatrix = nx.to_numpy_array(G, weight=None, nodelist = nodelist)
        file_fmt = 'AdjacencyMatrix_{}.npy'
        adjPath = join(dataPath, file_fmt.format(year))
        np.save(adjPath, adjMatrix)