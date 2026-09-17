import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class TemporalSIFT:
    """
    1D SIFT Feature Extractor based on Bailly et al.
    Performs dense scale-space sampling and gradient descriptor generation.
    """

    def __init__(self, tau_step=15, num_scales=9, sigma_0=1.0, k_scale=1.4, n_b=4, a=2):
        self.tau_step = tau_step  # Time step between sampled keypoints
        self.num_scales = num_scales  # Number of Gaussian scales
        self.sigma_0 = sigma_0  # Base scale
        self.k_scale = k_scale  # Scale step multiplier
        self.n_b = n_b  # Number of sub-blocks in a window
        self.a = a  # Number of temporal samples per block

    def extract_descriptors(self, signal: np.ndarray) -> np.ndarray:
        """
        Extracts 1D SIFT descriptors from a single 1D signal across multiple scales.
        Returns: Array of shape (N_descriptors, 2 * n_b)
        """
        signal = np.asarray(signal, dtype=np.float32)
        window_size = self.n_b * self.a
        half_w = window_size // 2
        descriptors = []

        if len(signal) < window_size:
            return np.empty((0, 2 * self.n_b))

        # 1. Scale-Space loop
        for s in range(self.num_scales):
            sigma = self.sigma_0 * (self.k_scale ** s)
            smoothed = gaussian_filter1d(signal, sigma=sigma)
            grad = np.gradient(smoothed)

            # 2. Dense temporal sampling
            for t in range(half_w, len(signal) - half_w, self.tau_step):
                grad_window = grad[t - half_w: t - half_w + window_size]

                # 3. Descriptor construction per block
                desc_vector = []
                for b in range(self.n_b):
                    sub_grad = grad_window[b * self.a: (b + 1) * self.a]
                    pos_sum = np.sum(np.maximum(0, sub_grad))
                    neg_sum = np.sum(np.maximum(0, -sub_grad))
                    desc_vector.extend([pos_sum, neg_sum])

                desc_vector = np.array(desc_vector, dtype=np.float32)
                norm = np.linalg.norm(desc_vector)

                # Normalize descriptor vector (L2 norm)
                if norm > 1e-6:
                    desc_vector /= norm
                descriptors.append(desc_vector)

        return np.array(descriptors, dtype=np.float32)


class SIFTBagOfWords:
    def __init__(self, extractor: TemporalSIFT = None, n_clusters: int = 64):
        self.extractor = extractor if extractor is not None else TemporalSIFT()
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.nn = None
        self.is_fitted = False

    def fit_codebook(self, signals: list):
        all_descriptors = []
        for sig in signals:
            desc = self.extractor.extract_descriptors(sig)
            if len(desc) > 0:
                all_descriptors.append(desc)

        if len(all_descriptors) == 0:
            raise ValueError("No descriptors extracted. Check signal lengths or parameters.")

        all_descriptors = np.vstack(all_descriptors)
        self.kmeans.fit(all_descriptors)
        self.nn = NearestNeighbors(n_neighbors=1).fit(self.kmeans.cluster_centers_)
        self.is_fitted = True

    def save_codebook(self, filepath: str):
        """Saves the fitted codebook and extractor parameters to a file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted codebook.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'nn': self.nn,
                'n_clusters': self.n_clusters,
                'extractor_params': {
                    'tau_step': self.extractor.tau_step,
                    'num_scales': self.extractor.num_scales,
                    'sigma_0': self.extractor.sigma_0,
                    'k_scale': self.extractor.k_scale,
                    'n_b': self.extractor.n_b,
                    'a': self.extractor.a
                }
            }, f)
        print(f"SIFT Codebook successfully saved to: {filepath}")

    def load_codebook(self, filepath: str):
        """Loads a pre-fitted codebook from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Codebook file not found at {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.kmeans = data['kmeans']
        self.nn = data['nn']
        self.n_clusters = data['n_clusters']

        # Restore extractor parameters if saved
        params = data.get('extractor_params', {})
        if params:
            self.extractor = TemporalSIFT(**params)

        self.is_fitted = True
        print(f"SIFT Codebook successfully loaded from: {filepath}")

    def signal_to_histogram(self, signal: np.ndarray) -> np.ndarray:
        """
        Converts a 1D signal into an L1-normalized K-dimensional histogram.
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook must be fitted before extracting histograms.")

        descriptors = self.extractor.extract_descriptors(signal)

        if len(descriptors) == 0:
            return np.zeros(self.n_clusters, dtype=np.float32)

        # Assign each descriptor to the nearest codeword (cluster)
        word_indices = self.nn.kneighbors(descriptors, return_distance=False).flatten()

        # Build histogram of word occurrences
        counts = np.bincount(word_indices, minlength=self.n_clusters)
        histogram = counts.astype(np.float32)

        # L1 Normalization (ensures length-invariance)
        total_counts = np.sum(histogram)
        if total_counts > 0:
            histogram /= total_counts

        return histogram