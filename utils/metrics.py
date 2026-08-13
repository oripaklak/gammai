import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import binomtest

def calculate_fid(real_data, fake_data, eps=1e-6, n_components=128):
    """
    Computes the Fréchet Distance between two 1D distributions.
    This is the mathematical equivalent of FID, but calculated directly
    on spectrum data (1D) instead of Inception image features (3D) .
    """

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


def calculate_kid(real_data, fake_data, degree=3, coef0=1):
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


def calculate_ndb(real_data, fake_data, n_clusters=50, alpha=0.05, random_state=42):
    """
    Computes the Number of Statistically-Different Bins (NDB) metric for 1D spectra.

    Parameters:
    -----------
    real_data : np.ndarray, shape (N_real, L)
        Real spectra vectors.
    fake_data : np.ndarray, shape (N_fake, L)
        Generated spectra vectors.
    n_clusters : int, optional
        Number of clusters (bins) $K$ to partition the data space. Default is 50.
    alpha : float, optional
        Significance level for the binomial hypothesis test. Default is 0.05.
    random_state : int, optional
        Random seed for reproducibility of KMeans.

    Returns:
    --------
    dict
        Contains the NDB score and diagnostic details per bin.
    """
    X = np.atleast_2d(real_data)
    Y = np.atleast_2d(fake_data)

    m = X.shape[0]  # Number of real samples
    n = Y.shape[0]  # Number of fake samples

    # Step 1: Fit KMeans clustering exclusively on the REAL data space
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(X)

    # Step 2: Assign both real and fake samples to the nearest cluster bins
    real_labels = kmeans.predict(X)
    fake_labels = kmeans.predict(Y)

    # Count occurrences in each bin
    real_counts = np.bincount(real_labels, minlength=n_clusters)
    fake_counts = np.bincount(fake_labels, minlength=n_clusters)

    # Expected proportion of samples falling into each bin from the real dataset
    p_real = real_counts / m

    # Step 3: Perform binomial tests for each bin
    # Null hypothesis: The probability of a generated sample falling into bin i is p_real[i].
    ndb_score = 0
    bin_results = []

    for i in range(n_clusters):
        k_success = fake_counts[i]  # Number of fake samples in bin i
        p_prob = p_real[i]  # Expected probability from real distribution

        # If a bin has zero real samples, handle edge cases
        if p_prob == 0:
            p_val = 0.0 if k_success == 0 else 1.0
        else:
            # Two-sided binomial test
            res = binomtest(k_success, n=n, p=p_prob)
            p_val = res.pvalue

        # Check if difference is statistically significant
        is_different = (p_val < alpha)
        if is_different:
            ndb_score += 1

        bin_results.append({
            'bin': i,
            'real_count': real_counts[i],
            'fake_count': fake_counts[i],
            'p_value': p_val,
            'significant_diff': is_different
        })

    return {
        'ndb_score': ndb_score,
        'max_bins': n_clusters,
        'bin_details': bin_results
    }

if __name__ == "__main__":
    # 1. Load the data
    data_path = "data/spectrum_matrix_2023_2024.npy"
    print(f"Loading data from {data_path}...")
    try:
        data = np.load(data_path).astype(np.float32)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please check the path.")
        exit()

    # --- NEW: Apply the exact normalization from SignalDataset ---
    print("Applying log1p and global max normalization...")
    log_matrix = np.log1p(data)
    max_val = np.max(log_matrix)
    data = log_matrix / max_val
    # -----------------------------------------------------------

    print(f"Total data shape: {data.shape}")
    N = data.shape[0]

    if N < 4:
        print("Error: Not enough data points to split and calculate metrics.")
        exit()

    # 2. Shuffle the data
    np.random.seed(0)
    indices = np.random.permutation(N)
    shuffled_data = data[indices]

    # 3. Split into two halves
    midpoint = N // 2
    half_a = shuffled_data[:midpoint]
    half_b = shuffled_data[midpoint:]

    print(f"Split data into Half A (shape: {half_a.shape}) and Half B (shape: {half_b.shape})")

    target_components = 128
    safe_components = min(target_components, half_a.shape[0])

    if safe_components < target_components:
        print(f"Note: Reduced PCA components to {safe_components} due to small sample size.")

    # 4. Calculate Baselines
    print("\n--- Calculating Baselines (Real vs Real) ---")

    # Fixed function names to match your definitions
    baseline_fd = calculate_fid(half_a, half_b, n_components=safe_components)
    print(f"Baseline Fréchet Distance (FD): {baseline_fd:.6f}")

    baseline_kid = calculate_kid(half_a, half_b)
    print(f"Baseline Polynomial MMD (KID):  {baseline_kid:.6f}")

    # --- NEW: Calculate Baseline NDB ---
    # Using the default 50 clusters/bins defined in your function
    baseline_ndb_data = calculate_ndb(half_a, half_b, n_clusters=30)
    ndb_score = baseline_ndb_data['ndb_score']
    max_bins = baseline_ndb_data['max_bins']
    print(f"Baseline NDB Score:             {ndb_score} / {max_bins} bins")