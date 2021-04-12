import numpy as np

from sklearn.utils import check_random_state

from .network_likelihoods import (
    dynamic_network_loglikelihood_directed,
    dynamic_network_loglikelihood_undirected,
    approx_directed_network_loglikelihood,
    dynamic_network_loglikelihood_directed_weighted,
    dynamic_network_loglikelihood_undirected_weighted
)


def sample_intercepts(Y, X, intercepts, intercept_prior,
                      intercept_variance_prior, samplers, radii=None, nu=None,
                      dist=None, is_weighted=False, is_directed=False, case_control_sampler=None,
                      squared=False, random_state=None):
    rng = check_random_state(random_state)

    if is_weighted:
        if is_directed:
            # sample intercept_in
            def logp(x):
                if case_control_sampler is not None:
                    # TODO: Loglikelihood for case_control_sampler in weighted case
                    raise ValueError('The case-control likelihood currently only '
                                     'supported for non-weighted directed networks.')
                else:
                    loglik = dynamic_network_loglikelihood_directed_weighted(
                        Y, X,
                        intercept_in=x[0], intercept_out=intercepts[1],
                        radii=radii,
                        nu=nu,
                        squared=squared,
                        dist=dist)
                loglik -= ((x[0] - intercept_prior[0]) ** 2 /
                           (2 * intercept_variance_prior))
                return loglik

            intercepts[0] = samplers[0].step(
                                    np.array([intercepts[0]]), logp, rng)[0]

            # sample intercept_out
            def logp(x):
                if case_control_sampler is not None:
                    # TODO: Loglikelihood for case_control_sampler in weighted case
                    raise ValueError('The case-control likelihood currently only '
                                     'supported for non-weighted directed networks.')
                else:
                    loglik = dynamic_network_loglikelihood_directed_weighted(
                        Y, X,
                        intercept_in=intercepts[0], intercept_out=x[0],
                        radii=radii,
                        nu=nu,
                        squared=squared,
                        dist=dist)
                loglik -= ((x[0] - intercept_prior[1]) ** 2 /
                           (2 * intercept_variance_prior))
                return loglik

            intercepts[1] = samplers[1].step(
                                np.array([intercepts[1]]), logp, rng)[0]
        else:
            def logp(x):
                loglik = dynamic_network_loglikelihood_undirected_weighted(Y, X,
                                                                  intercept=x, nu=nu,
                                                                  squared=squared,
                                                                  dist=dist)
                loglik -= ((x - intercept_prior) ** 2 /
                           (2 * intercept_variance_prior))
                return loglik

            intercepts = samplers[0].step(intercepts, logp, rng)
    else:
        if is_directed:
            # sample intercept_in
            def logp(x):
                if case_control_sampler is not None:
                    # TODO: we do not cache distances here, decrease by
                    #       factor of 2 if we do this
                    loglik = approx_directed_network_loglikelihood(
                        X=X,
                        radii=radii,
                        in_edges=case_control_sampler.in_edges_,
                        out_edges=case_control_sampler.out_edges_,
                        degree=case_control_sampler.degrees_,
                        control_nodes=case_control_sampler.control_nodes_out_,
                        intercept_in=x[0],
                        intercept_out=intercepts[1],
                        squared=squared)
                else:
                    loglik = dynamic_network_loglikelihood_directed(
                        Y, X,
                        intercept_in=x[0], intercept_out=intercepts[1],
                        radii=radii,
                        squared=squared,
                        dist=dist)
                loglik -= ((x[0] - intercept_prior[0]) ** 2 /
                           (2 * intercept_variance_prior))
                return loglik

            intercepts[0] = samplers[0].step(
                np.array([intercepts[0]]), logp, rng)[0]

            # sample intercept_out
            def logp(x):
                if case_control_sampler is not None:
                    # TODO: we do not cache distances here, decrease by
                    #       factor of 2 if we do this
                    loglik = approx_directed_network_loglikelihood(
                        X=X,
                        radii=radii,
                        in_edges=case_control_sampler.in_edges_,
                        out_edges=case_control_sampler.out_edges_,
                        degree=case_control_sampler.degrees_,
                        control_nodes=case_control_sampler.control_nodes_out_,
                        intercept_in=intercepts[0],
                        intercept_out=x[0],
                        squared=squared)
                else:
                    loglik = dynamic_network_loglikelihood_directed(
                        Y, X,
                        intercept_in=intercepts[0], intercept_out=x[0],
                        radii=radii,
                        squared=squared,
                        dist=dist)
                loglik -= ((x[0] - intercept_prior[1]) ** 2 /
                           (2 * intercept_variance_prior))
                return loglik

            intercepts[1] = samplers[1].step(
                np.array([intercepts[1]]), logp, rng)[0]
        else:
            def logp(x):
                loglik = dynamic_network_loglikelihood_undirected(Y, X,
                                                                  intercept=x,
                                                                  squared=squared,
                                                                  dist=dist)
                loglik -= ((x - intercept_prior) ** 2 /
                           (2 * intercept_variance_prior))
                return loglik

            intercepts = samplers[0].step(intercepts, logp, rng)

    return intercepts


def sample_radii(Y, X, intercepts, radii, sampler, nu=None, dist=None,
                 is_weighted=False, case_control_sampler=None, squared=False,
                 random_state=None):
    rng = check_random_state(random_state)

    if is_weighted:
        def logp(x):
            # NOTE: dirichlet prior (this is constant for alpha = 1.0
            if case_control_sampler:
                # TODO: Loglikelihood for case_control_sampler in weighted case
                raise ValueError('The case-control likelihood currently only '
                                 'supported for non-weighted directed networks.')
            else:
                loglik = dynamic_network_loglikelihood_directed_weighted(
                    Y, X,
                    intercept_in=intercepts[0],
                    intercept_out=intercepts[1],
                    radii=x,
                    nu=nu,
                    squared=squared,
                    dist=dist)

            return loglik
    else:
        def logp(x):
            # NOTE: dirichlet prior (this is constant for alpha = 1.0
            if case_control_sampler:
                # TODO: we do not cache distances here, decrease by
                #       factor of 2 if we do this
                loglik = approx_directed_network_loglikelihood(
                    X=X,
                    radii=x,
                    in_edges=case_control_sampler.in_edges_,
                    out_edges=case_control_sampler.out_edges_,
                    degree=case_control_sampler.degrees_,
                    control_nodes=case_control_sampler.control_nodes_out_,
                    intercept_in=intercepts[0],
                    intercept_out=intercepts[1],
                    squared=squared)
            else:
                loglik = dynamic_network_loglikelihood_directed(
                    Y, X,
                    intercept_in=intercepts[0],
                    intercept_out=intercepts[1],
                    radii=x,
                    squared=squared,
                    dist=dist)

            return loglik

    return sampler.step(radii, logp, rng)

def sample_nu(Y, X, delta, zeta_sq, intercepts, nu, sampler, radii=None, dist=None,
                 is_directed=False, case_control_sampler=None, squared=False, random_state=None):
    rng = check_random_state(random_state)

    if is_directed:
        def logp(x):
            # sample nu squared: distributed as inverse gamma (IG(a, b).
            # a = 2 + delta, b = (1 + delta) * tau_squared with small delta and pos. const. tau**2
            if case_control_sampler:
                # TODO: Loglikelihood for case_control_sampler in weighted case
                raise ValueError('The case-control likelihood currently only '
                                 'supported for non-weighted directed networks.')
            else:
                loglik = dynamic_network_loglikelihood_directed_weighted(
                    Y, X,
                    intercept_in=intercepts[0],
                    intercept_out=intercepts[1],
                    radii=radii,
                    nu=x,
                    squared=squared,
                    dist=dist)

            loglik -= ((2 + delta) + 1) * np.log(x) + ((1 + delta) * zeta_sq) / x
            return loglik
    else:
        def logp(x):
            # sample nu squared: distributed as inverse gamma (IG(a, b).
            # a = 2 + delta, b = (1 + delta) * tau_squared with small delta and pos. const. tau**2
            loglik = dynamic_network_loglikelihood_undirected_weighted(
                Y, X,
                intercept=intercepts,
                nu=x,
                squared=squared,
                dist=dist)
            loglik -= ((2 + delta) + 1) * np.log(x) + ((1 + delta) * zeta_sq) / x
            return loglik

    return sampler.step(nu, logp, rng)
