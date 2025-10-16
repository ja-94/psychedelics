import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Alternative pre-processing step to satisfy gaussianity assumption in PCA (non-linear transformation to make spike counts normally distibuted within neurons)
# transformer = PowerTransformer(method='yeo-johnson', standardize=True)  # note: standardization/ rescaling is included

# ~class PCA(PCA):

    # ~def __init__(self, scaler=StandardScaler, scaler_kwargs={}, **kwargs):
        # ~super().__init__(**kwargs)
        # ~self.scaler = scaler(**scaler_kwargs)

    # ~@property
    # ~def X_norm(self):
        # ~return self.scaler.fit_transform(self.X)


def _apply_PCA(population_counts, scaler=StandardScaler, scaler_kwargs={}):

    res = {}

    # Put spike counts for neurons into a samples x features matrix
    X = np.column_stack(population_counts)

    # Normalize the data
    X_rescaled = scaler().fit_transform(X)

    # Fit the PCA
    pca = PCA()
    pca.fit(X_rescaled)

    # Add explained variance and loadings to the data dict
    res[f'variance'] = pca.explained_variance_ratio_
    res[f'components'] = pca.components_
