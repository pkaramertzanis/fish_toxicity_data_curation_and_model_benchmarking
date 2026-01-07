import logger
log = logger.get_logger(__name__)

import numpy as np
from scipy import odr
from tqdm import tqdm
from sklearn.decomposition import PCA

# --- Deming regression function ---
def deming_regression(x, y, lambda_ratio=1.0):
    """
    Deming regression (generalized orthogonal regression).
    lambda_ratio = variance_x / variance_y
    lambda_ratio=1 -> standard orthogonal regression
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    Sxx = np.var(x, ddof=1)
    Syy = np.var(y, ddof=1)
    Sxy = np.cov(x, y, ddof=1)[0, 1]
    delta = Syy - lambda_ratio * Sxx
    beta = (delta + np.sqrt(delta ** 2 + 4 * lambda_ratio * Sxy ** 2)) / (2 * Sxy)
    alpha = y_mean - beta * x_mean
    return alpha, beta

# --- Orthogonal regression function ---
def orthogonal_regression(x, y):
    XY = np.vstack([x, y]).T
    XY_mean = XY.mean(axis=0)
    U, S, Vt = np.linalg.svd(XY - XY_mean)
    direction = Vt.T[:, 0]
    slope = direction[1] / direction[0]
    intercept = XY_mean[1] - slope * XY_mean[0]
    return intercept, slope

# orthogonal regression using scipy
def orthogonal_regression_scipy(x, y, n_bootstrap=1000):
    '''
    Orthogonal regression using scipy.odr

    The exact solution is computed using principal component analysis (PCA) as implemented in scikit learn
    :param x: independent variable
    :param y: dependent variable
    :param n_bootstrap: number of bootstrap samples to estimate the confidence intervals
    :return: tuple (intercept, slope, exact_pca)
    '''
    # set the numpy seed for reproducibility
    np.random.seed(1)
    x = np.array(x)
    y = np.array(y)
    # define a model function, e.g., a line y = m*x + c
    def linear_model(B, x):
        return B[0] * x + B[1]
    # create a Model object
    model = odr.Model(linear_model)
    intercepts = []
    slopes = []
    for _ in tqdm(range(n_bootstrap)):
        # sample with replacement
        indices = np.random.choice(len(x), size=len(x), replace=True)
        x_sample = x[indices]
        y_sample = y[indices]
        # create a Data object
        data = odr.RealData(x_sample, y_sample)
        # set up ODR with model and data
        odr_obj = odr.ODR(data, model, beta0=[1., 0.], maxit=1000)  # initial guess for m and c
        # run the regression
        output = odr_obj.run()
        intercepts.append(output.beta[1])
        slopes.append(output.beta[0])

    # compute exact orthogonal regression
    data = odr.RealData(x, y)
    odr_obj = odr.ODR(data, model, beta0=[1., 0.], maxit=1000)  # initial guess for m and c
    output = odr_obj.run()
    slope_exact = output.beta[0]
    intercept_exact = output.beta[1]

    # compute R2
    distances = np.abs(slope_exact * x - y + intercept_exact) / np.sqrt(slope_exact ** 2 + 1)
    # .. vertical distances
    SSR = np.sum(distances ** 2)
    # .. distances from the dataset centroid
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    SST = np.sum(((x - x_mean) ** 2 + (y - y_mean) ** 2) )
    # SST = np.sum((y - y_mean) ** 2)
    R2_exact = 1 - SSR / SST

    # compute RMSE and MAE
    RMSE = np.sqrt(np.mean((y - x) ** 2))
    MAE = np.mean(np.abs(y - x))

    # compute the exact orthogonal regression using PCA (for verification)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(np.stack([x, y], axis=1))
    mu = pca.mean_
    # .. direction of first principal component
    v = pca.components_[0]  # already unit length
    # .. compute slope and intercept
    slope_pca = v[1] / v[0]
    intercept_pca = mu[1] - slope_pca * mu[0]
    variance_first_principal_pca = pca.explained_variance_ratio_[0]

    # return medians and 95% confidence intervals for intercept and slope
    return {'intercept': {'median': np.median(intercepts), '2.5%': np.percentile(intercepts, 2.5), '97.5%': np.percentile(intercepts, 97.5)},
            'slope': {'median': np.median(slopes), '2.5%': np.percentile(slopes, 2.5), '97.5%': np.percentile(slopes, 97.5)},
            'exact': {'intercept ODR': intercept_exact, 'slope ODR': slope_exact, 'R2 ODR': R2_exact, 'RMSE': RMSE, 'MAE': MAE,
                      'intercept PCA': intercept_pca, 'slope PCA': slope_pca, 'variance first principal PCA': variance_first_principal_pca}
            }
