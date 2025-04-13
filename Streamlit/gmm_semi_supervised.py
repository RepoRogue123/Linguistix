import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import dill as pickle

class GaussianMixtureModel:
    """Enhanced Gaussian Mixture Model implementation for speaker recognition"""

    def __init__(self, max_iters=100, tol=1e-4, reg_covar=1e-6, covariance_type='full',
                 init_method='kmeans++', semi_supervised=False, supervision_weight=0.5):
        self.max_iters = max_iters
        self.tol = tol  # convergence threshold
        self.reg_covar = reg_covar  # regularization parameter for covariance matrices
        self.weights = None  # mixture weights
        self.means = None    # component means
        self.covars = None   # component covariance matrices
        self.n_components = None
        self.covariance_type = covariance_type  # 'full', 'diagonal', 'tied', or 'spherical'
        self.init_method = init_method  # 'random', 'kmeans++', or 'kmeans'
        self.semi_supervised = semi_supervised  # whether to use semi-supervised learning
        self.supervision_weight = supervision_weight  # weight for known labels in semi-supervised learning
        self.label_to_component_map = None  # mapping from labels to components for semi-supervised learning

    def _initialize_parameters(self, X, n_components, y=None):
        n_samples, n_features = X.shape
        self.n_components = n_components

        # Initialize means
        if self.init_method == 'random':
            indices = np.random.choice(n_samples, n_components, replace=False)
            self.means = X[indices]
        elif self.init_method in ['kmeans++', 'kmeans']:
            kmeans = KMeans(n_clusters=n_components, init='k-means++', n_init=10)
            kmeans.fit(X)
            self.means = kmeans.cluster_centers_

        # Initialize covariances based on covariance type
        if self.covariance_type == 'full':
            self.covars = np.array([np.cov(X.T) + self.reg_covar * np.eye(n_features)] * n_components)
        elif self.covariance_type == 'diagonal':
            variance = np.var(X, axis=0)
            self.covars = np.array([np.diag(variance + self.reg_covar)] * n_components)
        elif self.covariance_type == 'tied':
            self.covars = np.cov(X.T) + self.reg_covar * np.eye(n_features)
        elif self.covariance_type == 'spherical':
            variance = np.mean(np.var(X, axis=0))
            self.covars = np.array([variance * np.eye(n_features)] * n_components)

        # Initialize weights uniformly
        self.weights = np.ones(n_components) / n_components

        # Setup semi-supervised learning if enabled
        if self.semi_supervised and y is not None:
            self._setup_semi_supervised(X, y)

    def _setup_semi_supervised(self, X, y):
        unique_labels = np.unique(y)
        n_labels = len(unique_labels)
        if self.n_components >= n_labels:
            components_per_label = self.n_components // n_labels
            self.label_to_component_map = {}
            component_idx = 0
            for label in unique_labels:
                self.label_to_component_map[label] = []
                for _ in range(components_per_label):
                    if component_idx < self.n_components:
                        self.label_to_component_map[label].append(component_idx)
                        component_idx += 1
            remaining_label = 0
            while component_idx < self.n_components:
                self.label_to_component_map[unique_labels[remaining_label]].append(component_idx)
                component_idx += 1
                remaining_label = (remaining_label + 1) % n_labels
            # Initialize means for each label's components using samples from that label
            for label, components in self.label_to_component_map.items():
                label_samples = X[y == label]
                n_label_samples = len(label_samples)
                if n_label_samples >= len(components):
                    kmeans = KMeans(n_clusters=len(components), init='k-means++', n_init=3)
                    kmeans.fit(label_samples)
                    for i, comp_idx in enumerate(components):
                        self.means[comp_idx] = kmeans.cluster_centers_[i]
                else:
                    for i, comp_idx in enumerate(components):
                        if i < n_label_samples:
                            self.means[comp_idx] = label_samples[i]
        else:
            self.label_to_component_map = {}
            for i, label in enumerate(unique_labels[:self.n_components]):
                self.label_to_component_map[label] = [i]
                label_samples = X[y == label]
                if len(label_samples) > 0:
                    self.means[i] = np.mean(label_samples, axis=0)

    def _e_step(self, X, y=None):
        n_samples = X.shape[0]
        weighted_log_prob = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            try:
                if self.covariance_type == 'full':
                    dist = multivariate_normal(mean=self.means[k], cov=self.covars[k])
                    log_prob = dist.logpdf(X)
                elif self.covariance_type == 'diagonal':
                    cov_diag = np.diag(self.covars[k]) if not isinstance(self.covars[k], np.ndarray) else np.diag(self.covars[k])
                    log_prob = self._log_multivariate_normal_density_diag(X, self.means[k], cov_diag)
                elif self.covariance_type == 'tied':
                    dist = multivariate_normal(mean=self.means[k], cov=self.covars)
                    log_prob = dist.logpdf(X)
                elif self.covariance_type == 'spherical':
                    variance = self.covars[k][0, 0] if isinstance(self.covars[k], np.ndarray) else self.covars[k]
                    log_prob = self._log_multivariate_normal_density_spherical(X, self.means[k], variance)
                weighted_log_prob[:, k] = log_prob + np.log(self.weights[k])
            except Exception as e:
                print(f"Error in E-step for component {k}: {e}")
                cov_diag = np.diag(np.diag(self.covars[k] if self.covariance_type != 'tied' else self.covars))
                cov_diag = cov_diag + self.reg_covar * np.eye(X.shape[1])
                log_prob = self._log_multivariate_normal_density_diag(X, self.means[k], cov_diag)
                weighted_log_prob[:, k] = log_prob + np.log(self.weights[k])
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        resp = np.exp(log_resp)
        if self.semi_supervised and y is not None and self.label_to_component_map is not None:
            resp = self._adjust_responsibilities_semi_supervised(resp, y)
        return resp, np.sum(log_prob_norm)

    def _log_multivariate_normal_density_diag(self, X, mean, diag_cov):
        n_samples, n_dim = X.shape
        diag_cov = np.maximum(diag_cov, self.reg_covar)
        log_det = np.sum(np.log(diag_cov))
        precisions = 1.0 / diag_cov
        log_prob = -0.5 * (n_dim * np.log(2 * np.pi) + log_det)
        log_probs = np.array([log_prob - 0.5 * np.sum((X[i] - mean) ** 2 * precisions) for i in range(n_samples)])
        return log_probs

    def _log_multivariate_normal_density_spherical(self, X, mean, variance):
        n_samples, n_dim = X.shape
        variance = max(variance, self.reg_covar)
        log_det = n_dim * np.log(variance)
        precision = 1.0 / variance
        log_prob = -0.5 * (n_dim * np.log(2 * np.pi) + log_det)
        log_probs = np.array([log_prob - 0.5 * precision * np.sum((X[i] - mean) ** 2) for i in range(n_samples)])
        return log_probs

    def _adjust_responsibilities_semi_supervised(self, resp, y):
        modified_resp = resp.copy()
        labeled_indices = ~np.isnan(y) if np.issubdtype(y.dtype, np.floating) else y >= 0
        for i in np.where(labeled_indices)[0]:
            label = y[i]
            if label in self.label_to_component_map:
                label_components = self.label_to_component_map[label]
                mask = np.ones(self.n_components, dtype=bool)
                mask[label_components] = False
                modified_resp[i, mask] *= (1 - self.supervision_weight)
                modified_resp[i] /= np.sum(modified_resp[i])
        return modified_resp

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights = nk / n_samples
        self.means = np.dot(resp.T, X) / nk[:, np.newaxis]
        if self.covariance_type == 'full':
            self.covars = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                weighted_diff = resp[:, k, np.newaxis] * diff
                self.covars[k] = np.dot(diff.T, weighted_diff) / nk[k]
                self.covars[k] += self.reg_covar * np.eye(n_features)
        elif self.covariance_type == 'diagonal':
            self.covars = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                weighted_diff = resp[:, k, np.newaxis] * diff
                self.covars[k] = np.sum(diff * weighted_diff, axis=0) / nk[k]
                self.covars[k] += self.reg_covar
        elif self.covariance_type == 'tied':
            self.covars = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                weighted_diff = resp[:, k, np.newaxis] * diff
                self.covars += np.dot(diff.T, weighted_diff) / n_samples
            self.covars += self.reg_covar * np.eye(n_features)
        elif self.covariance_type == 'spherical':
            self.covars = np.zeros(self.n_components)
            for k in range(self.n_components):
                diff = X - self.means[k]
                weighted_diff = resp[:, k, np.newaxis] * diff
                self.covars[k] = np.sum(diff * weighted_diff) / (n_features * nk[k])
                self.covars[k] += self.reg_covar

    def fit(self, X, n_components, y=None, n_init=1):
        best_log_likelihood = -np.inf
        best_params = None
        best_log_likelihoods = None
        for init in range(n_init):
            print(f"Initialization {init+1}/{n_init}")
            self._initialize_parameters(X, n_components, y if self.semi_supervised else None)
            log_likelihoods = []
            prev_log_likelihood = None
            for iteration in range(self.max_iters):
                resp, log_likelihood = self._e_step(X, y if self.semi_supervised else None)
                log_likelihoods.append(log_likelihood)
                if prev_log_likelihood is not None:
                    if abs(log_likelihood - prev_log_likelihood) < self.tol:
                        print(f"Converged after {iteration+1} iterations")
                        break
                prev_log_likelihood = log_likelihood
                self._m_step(X, resp)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = (self.weights.copy(), self.means.copy(), self.covars.copy())
                best_log_likelihoods = log_likelihoods
        self.weights, self.means, self.covars = best_params
        resp, _ = self._e_step(X, y if self.semi_supervised else None)
        labels = resp.argmax(axis=1)
        return labels, best_log_likelihoods

    def predict(self, X):
        resp, _ = self._e_step(X)
        return resp.argmax(axis=1)

    def predict_proba(self, X):
        resp, _ = self._e_step(X)
        return resp

    def calculate_bic(self, X):
        n_samples, n_features = X.shape
        if self.covariance_type == 'full':
            n_params = (self.n_components - 1)
            n_params += self.n_components * n_features
            n_params += self.n_components * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'diagonal':
            n_params = (self.n_components - 1)
            n_params += self.n_components * n_features
            n_params += self.n_components * n_features
        elif self.covariance_type == 'tied':
            n_params = (self.n_components - 1)
            n_params += self.n_components * n_features
            n_params += n_features * (n_features + 1) // 2
        elif self.covariance_type == 'spherical':
            n_params = (self.n_components - 1)
            n_params += self.n_components * n_features
            n_params += self.n_components
        _, log_likelihood = self._e_step(X)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        return bic

def save(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
