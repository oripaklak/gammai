import argparse
import os

import numpy as np
from scipy import linalg
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def calculate_fid(real_data, fake_data, eps=1e-6, n_components=128):
    """Computes the Fréchet Distance between two 1D distributions using PCA.

    Calculates the 1D equivalent of FID directly on spectrum data reduced via
    PCA.
    """
    pca = PCA(n_components=n_components)

    # Fit PCA strictly on real data, then project both real and fake spectra
    real_data = pca.fit_transform(real_data)
    fake_data = pca.transform(fake_data)

    mu1 = np.mean(real_data, axis=0)
    sigma1 = np.cov(real_data, rowvar=False)

    mu2 = np.mean(fake_data, axis=0)
    sigma2 = np.cov(fake_data, rowvar=False)

    diff = mu1 - mu2

    # Calculate square root of product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = (
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covmean)
    )
    return float(fd)


def calculate_kid(real_data, fake_data, degree=3, coef0=1):
    """Computes Unbiased Maximum Mean Discrepancy (MMD) using a polynomial
    kernel."""
    fake_data = np.squeeze(fake_data)

    gamma = 1.0 / real_data.shape[1]

    K_XX = (gamma * (real_data @ real_data.T) + coef0) ** degree
    K_YY = (gamma * (fake_data @ fake_data.T) + coef0) ** degree
    K_XY = (gamma * (real_data @ fake_data.T) + coef0) ** degree

    m = real_data.shape[0]
    n = fake_data.shape[0]

    mmd_xx = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    mmd_yy = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    mmd_xy = np.sum(K_XY) / (m * n)

    kid_equivalent = mmd_xx + mmd_yy - 2 * mmd_xy
    return float(kid_equivalent)


def calculate_ndb(
    real_data, fake_data, n_clusters=50, alpha=0.05, random_state=42
):
    """Computes Number of Statistically-Different Bins (NDB) for 1D spectra."""
    X = np.atleast_2d(real_data)
    Y = np.atleast_2d(fake_data)

    m = X.shape[0]
    n = Y.shape[0]

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init="auto"
    )
    kmeans.fit(X)

    real_labels = kmeans.predict(X)
    fake_labels = kmeans.predict(Y)

    real_counts = np.bincount(real_labels, minlength=n_clusters)
    fake_counts = np.bincount(fake_labels, minlength=n_clusters)

    ndb_score = 0
    bin_results = []

    for i in range(n_clusters):
        count_real = real_counts[i]
        count_fake = fake_counts[i]

        p_p = count_real / m
        p_q = count_fake / n

        P_pooled = (count_real + count_fake) / (m + n)

        if P_pooled == 0 or P_pooled == 1:
            p_val = 1.0
        else:
            se = np.sqrt(P_pooled * (1 - P_pooled) * (1 / m + 1 / n))

            if se == 0:
                p_val = 1.0
            else:
                z = (p_p - p_q) / se
                p_val = 2.0 * norm.cdf(-np.abs(z))

        is_different = p_val < alpha
        if is_different:
            ndb_score += 1

        bin_results.append({
            "bin": i,
            "real_count": count_real,
            "fake_count": count_fake,
            "p_value": p_val,
            "significant_diff": is_different,
        })

    return {
        "ndb_score": ndb_score,
        "max_bins": n_clusters,
        "bin_details": bin_results,
    }


def calculate_f_mem(gen_data, real_data, k=1 / 3):
    """Calculates the number and percentage of memorized generated samples."""
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(real_data)

    distances, _ = nn.kneighbors(gen_data)

    d1 = distances[:, 0]
    d2 = distances[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(d2 == 0, 0.0, d1 / d2)

    is_memorized = ratios < k

    f_mem_percentage = float(np.mean(is_memorized)) * 100.0

    min_d1, max_d1 = float(np.min(d1)), float(np.max(d1))
    min_d2, max_d2 = float(np.min(d2)), float(np.max(d2))
    min_ratio, max_ratio = float(np.min(ratios)), float(np.max(ratios))

    print("\n   [F_MEM Distance Stats]")
    print(
        f"   -> 1st Nearest Neighbor (d1) | Lowest: {min_d1:8.2f} | Max: {max_d1:8.2f}"
    )
    print(
        f"   -> 2nd Nearest Neighbor (d2) | Lowest: {min_d2:8.2f} | Max: {max_d2:8.2f}"
    )
    print(
        f"   -> Ratio (d1/d2)             | Lowest: {min_ratio:8.4f} | Max: {max_ratio:8.4f}"
    )

    return f_mem_percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate baseline metrics on a spectrum matrix."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to spectrum_matrix.npy",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if data_path is None:
        if os.path.exists("data/Baq/spectrum_matrix.npy"):
            data_path = "data/Baq/spectrum_matrix.npy"
        elif os.path.exists("data/IAEA/spectrum_matrix.npy"):
            data_path = "data/IAEA/spectrum_matrix.npy"
        else:
            data_path = "data/Baq/spectrum_matrix.npy"

    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path).astype(np.float32)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please check the path.")
        exit(1)

    print("Applying log1p and global max normalization...")
    log_matrix = np.log1p(data)
    max_val = np.max(log_matrix)
    data = log_matrix / max_val

    print(f"Total data shape: {data.shape}")
    N = data.shape[0]

    if N < 4:
        print("Error: Not enough data points to split and calculate metrics.")
        exit(1)

    np.random.seed(0)
    indices = np.random.permutation(N)
    shuffled_data = data[indices]

    midpoint = N // 2
    half_a = shuffled_data[:midpoint]
    half_b = shuffled_data[midpoint:]

    print(
        f"Split data into Half A (shape: {half_a.shape}) and Half B (shape: {half_b.shape})"
    )

    target_components = 128
    safe_components = min(target_components, half_a.shape[0])

    if safe_components < target_components:
        print(
            f"Note: Reduced PCA components to {safe_components} due to small sample size."
        )

    print("\n--- Calculating Baselines (Real vs Real) ---")

    baseline_fd = calculate_fid(half_a, half_b, n_components=safe_components)
    print(f"Baseline Fréchet Distance (FD): {baseline_fd:.6f}")

    baseline_kid = calculate_kid(half_a, half_b)
    print(f"Baseline Polynomial MMD (KID):  {baseline_kid:.6f}")

    baseline_f_mem = calculate_f_mem(half_a, half_b)
    print(f"Baseline fractional memorization (f_mem):  {baseline_f_mem:.6f}")

    baseline_ndb_data = calculate_ndb(half_a, half_b, n_clusters=30)
    ndb_score = baseline_ndb_data["ndb_score"]
    max_bins = baseline_ndb_data["max_bins"]
    print(f"Baseline NDB Score:             {ndb_score} / {max_bins} bins")