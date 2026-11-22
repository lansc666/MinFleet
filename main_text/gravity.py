"""Population gravity model fitted to aggregate OD flows."""


def gravity_model_popnum(dist, alpha, beta, gamma, K):
    return (
        K
        * (dist["pop_o"] ** alpha)
        * (dist["pop_d"] ** beta)
        / ((dist["t"] + 1) ** gamma)
    )
