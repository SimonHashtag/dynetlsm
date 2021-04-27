import numpy as np

from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.metrics.cluster import entropy

from .array_utils import triu_indices_from_3d
from .array_utils import nondiag_indices_from_3d
from .latent_space import calculate_distances


def network_auc(Y_true, Y_pred, is_directed=False):
    if is_directed:
        indices = nondiag_indices_from_3d(Y_true)
    else:
        indices = triu_indices_from_3d(Y_true, 1)

    y_fit = Y_pred[indices]
    y_true = Y_true[indices]
    return roc_auc_score(y_true, y_fit)


def _network_auc_directed():
    y_true, y_fit = [], []

    indices = np.triu_indices_from(Y_true[0], 1)
    for t in range(Y_true.shape[0]):
        y_fit.append(Y_pred[t][indices])
        y_true.append(Y_true[t][indices])

    return roc_auc_score(np.hstack(y_true), np.hstack(y_fit))


def _network_auc_undirected(Y_true, Y_pred):
    y_true, y_fit = [], []

    indices = triu_indices_from_3d(Y_true, 1)
    y_fit = Y_pred[indices]

    return roc_auc_score(np.hstack(y_true), np.hstack(y_fit))


def variation_of_information(labels_true, labels_pred):
    entropy_true = entropy(labels_true)
    entropy_pred = entropy(labels_pred)
    mutual_info = mutual_info_score(labels_true, labels_pred)

    return entropy_true + entropy_pred - 2 * mutual_info

def pseudo_R_squared(intercepts_mean, X_mean, radii_mean, nu_mean):
    n_time_steps, n_nodes, _ = X_mean.shape
    distances_mean = calculate_distances(X_mean, squared=False)
    y_squared = 0.
    y = 0.
    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    y_squared += (intercepts_mean[0] * (1 - distances_mean[t,i,j]/radii_mean[j]) + intercepts_mean[1] * (1 - distances_mean[t,i,j]/radii_mean[i]))**2
                    y += (intercepts_mean[0] * (1 - distances_mean[t,i,j]/radii_mean[j]) + intercepts_mean[1] * (1 - distances_mean[t,i,j]/radii_mean[i]))
    y_bar = y / (n_time_steps*n_nodes*(n_nodes-1))
    numerator = y_squared - (n_time_steps*n_nodes*(n_nodes-1)) * y_bar**2
    denominator = numerator + n_time_steps * n_nodes * (n_nodes - 1) * nu_mean

    return float(numerator / denominator)