import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

def calculate_frechet_distance(real_data, fake_data, eps=1e-6, n_components=128):
    """
    Computes the Fréchet Distance between two 1D distributions.
    This is the mathematical equivalent of FID, but calculated directly
    on spectrum data (1D) instead of Inception image features (3D) .
    """
    fake_data = np.squeeze(fake_data) # (N, 1, L) -> (N, L)

    # --- PCA INTEGRATION BLOCK ---
    # reduce the dimensionality from L = 16384 to a stable number (e.g., 128)
    # n_components must be smaller than N.
    pca = PCA(n_components=n_components)

    # We fit the PCA basis strictly on the REAL data to define the principal axes,
    # and then project both real and fake data into this exact same space.
    real_data = pca.fit_transform(real_data)
    fake_data = pca.transform(fake_data)

    mu1 = np.mean(real_data, axis=0)
    sigma1 = np.cov(real_data, rowvar=False)

    mu2 = np.mean(fake_data, axis=0)
    sigma2 = np.cov(fake_data, rowvar=False)

    diff = mu1 - mu2

    # Calculate the square root of the dot product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical instability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps # Identity matrix * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Fréchet Distance formula
    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fd)


def calculate_polynomial_mmd(real_data, fake_data, degree=3, coef0=1):
    """
    Computes the Unbiased Maximum Mean Discrepancy using a polynomial kernel.
    This is the mathematical foundation of KID, calculated directly on the spectra.
    """
    fake_data = np.squeeze(fake_data) # (N, 1, L) -> (N, L)

    # Gamma is typically 1 / feature_dimension
    gamma = 1.0 / real_data.shape[1]

    # Calculate kernels
    K_XX = (gamma * (real_data @ real_data.T) + coef0) ** degree
    K_YY = (gamma * (fake_data @ fake_data.T) + coef0) ** degree
    K_XY = (gamma * (real_data @ fake_data.T) + coef0) ** degree

    m = real_data.shape[0]
    n = fake_data.shape[0]

    # Unbiased MMD estimators (removing the diagonal)
    mmd_xx = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    mmd_yy = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    mmd_xy = np.sum(K_XY) / (m * n)

    kid_equivalent = mmd_xx + mmd_yy - 2 * mmd_xy
    return float(kid_equivalent)