"""
Linear regression using a regularization parameter (ridge regression).
Different values for the regularization parameter are evaluated efficiently by
employing generalized cross-validation. If the number of features exceeds the
number of examples, kernel regression is used to speed up computations.

Adapted from the scikit-learn 0.18 code by Marijn van Vliet to suppert the
alpha_per_target parameter. Maybe this change will be merged into scikit-learn
some day, see: https://github.com/scikit-learn/scikit-learn/pull/6624

Example usage:

# Let X be a numpy array of size #examples x #features.
# Let y be a numpy array of size #examples x #labels.
model = RidgeCV().fit(X, y)
predicted_y = model.predict(X)

See the docstring of the RidgeGCV class for more information.
"""

import warnings

import numpy as np
from scipy import linalg
from scipy import sparse

from sklearn.linear_model.base import LinearModel, _rescale_data
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_X_y
from sklearn.metrics.scorer import check_scoring


class RidgeGCV(LinearModel):
    """Ridge regression with built-in Generalized Cross-Validation

    It allows efficient Leave-One-Out cross-validation.

    This class is not intended to be used directly. Use RidgeCV instead.

    Notes
    -----

    We want to solve (K + alpha*Id)c = y,
    where K = X X^T is the kernel matrix.

    Let G = (K + alpha*Id)^-1.

    Dual solution: c = Gy
    Primal solution: w = X^T c

    Compute eigendecomposition K = Q V Q^T.
    Then G = Q (V + alpha*Id)^-1 Q^T,
    where (V + alpha*Id) is diagonal.
    It is thus inexpensive to inverse for many alphas.

    Let loov be the vector of prediction values for each example
    when the model was fitted with all examples but this example.

    loov = (KGY - diag(KG)Y) / diag(I-KG)

    Let looe be the vector of prediction errors for each example
    when the model was fitted with all examples but this example.

    looe = y - loov = c / diag(G)

    References
    ----------
    http://cbcl.mit.edu/projects/cbcl/publications/ps/MIT-CSAIL-TR-2007-025.pdf
    http://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf
    """

    def __init__(self, alphas=(0.1, 1.0, 10.0),
                 fit_intercept=True, normalize=False,
                 scoring=None, copy_X=True,
                 gcv_mode=None, store_cv_values=False,
                 alpha_per_target=False):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target

    def _pre_compute(self, X, y, centered_kernel=True):
        # even if X is very sparse, K is usually very dense
        K = safe_sparse_dot(X, X.T, dense_output=True)
        # the following emulates an additional constant regressor
        # corresponding to fit_intercept=True
        # but this is done only when the features have been centered
        if centered_kernel:
            K += np.ones_like(K)
        v, Q = linalg.eigh(K)
        QT_y = np.dot(Q.T, y)
        return v, Q, QT_y

    def _decomp_diag(self, v_prime, Q):
        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        return (v_prime * Q ** 2).sum(axis=-1)

    def _diag_dot(self, D, B):
        # compute dot(diag(D), B)
        if len(B.shape) > 1:
            # handle case where B is > 1-d
            D = D[(slice(None), ) + (np.newaxis, ) * (len(B.shape) - 1)]
        return D * B

    def _errors_and_values_helper(self, alpha, y, v, Q, QT_y):
        """Helper function to avoid code duplication between self._errors and
        self._values.

        Notes
        -----
        We don't construct matrix G, instead compute action on y & diagonal.
        """
        w = 1. / (v + alpha)
        constant_column = np.var(Q, 0) < 1.e-12
        # detect constant columns
        w[constant_column] = 0  # cancel the regularization for the intercept
        w[v == 0] = 0
        c = np.dot(Q, self._diag_dot(w, QT_y))
        G_diag = self._decomp_diag(w, Q)
        # handle case where y is 2-d
        if len(y.shape) != 1:
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def _errors(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return (c / G_diag) ** 2, c

    def _values(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return y - (c / G_diag), c

    def _pre_compute_svd(self, X, y, centered_kernel=True):
        if sparse.issparse(X):
            raise TypeError("SVD not supported for sparse matrices")
        if centered_kernel:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        # to emulate fit_intercept=True situation, add a column on ones
        # Note that by centering, the other columns are orthogonal to that one
        U, s, _ = linalg.svd(X, full_matrices=0)
        v = s ** 2
        UT_y = np.dot(U.T, y)
        return v, U, UT_y

    def _errors_and_values_svd_helper(self, alpha, y, v, U, UT_y):
        """Helper function to avoid code duplication between self._errors_svd
        and self._values_svd.
        """
        constant_column = np.var(U, 0) < 1.e-12
        # detect columns colinear to ones
        w = ((v + alpha) ** -1) - (alpha ** -1)
        w[constant_column] = - (alpha ** -1)
        # cancel the regularization for the intercept
        c = np.dot(U, self._diag_dot(w, UT_y)) + (alpha ** -1) * y
        G_diag = self._decomp_diag(w, U) + (alpha ** -1)
        if len(y.shape) != 1:
            # handle case where y is 2-d
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def _errors_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return (c / G_diag) ** 2, c

    def _values_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return y - (c / G_diag), c

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or array-like of shape [n_samples]
            Sample weight

        Returns
        -------
        self : Returns self.
        """
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float64,
                         multi_output=True, y_numeric=True)
        n_samples, n_features = X.shape

        if hasattr(LinearModel, '_preprocess_data'):
            # Scikit-learn 0.18 and up
            X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
                X, y, self.fit_intercept, self.normalize, self.copy_X,
                sample_weight=sample_weight)
        else:
            X, y, X_offset, y_offset, X_scale = LinearModel._center_data(
                X, y, self.fit_intercept, self.normalize, self.copy_X,
                sample_weight=sample_weight)

        gcv_mode = self.gcv_mode
        with_sw = len(np.shape(sample_weight))

        if gcv_mode is None or gcv_mode == 'auto':
            if sparse.issparse(X) or n_features > n_samples or with_sw:
                gcv_mode = 'eigen'
            else:
                gcv_mode = 'svd'
        elif gcv_mode == "svd" and with_sw:
            # FIXME non-uniform sample weights not yet supported
            warnings.warn("non-uniform sample weights unsupported for svd, "
                          "forcing usage of eigen")
            gcv_mode = 'eigen'

        if gcv_mode == 'eigen':
            _pre_compute = self._pre_compute
            _errors = self._errors
            _values = self._values
        elif gcv_mode == 'svd':
            # assert n_samples >= n_features
            _pre_compute = self._pre_compute_svd
            _errors = self._errors_svd
            _values = self._values_svd
        else:
            raise ValueError('bad gcv_mode "%s"' % gcv_mode)

        if sample_weight is not None:
            X, y = _rescale_data(X, y, sample_weight)

        # Ensure that y is a 2D array: n_samples x n_targets
        flat_y = y.ndim == 1
        if flat_y:
            y = np.atleast_2d(y).T
        n_targets = y.shape[1]

        centered_kernel = not sparse.issparse(X) and self.fit_intercept
        v, Q, QT_y = _pre_compute(X, y, centered_kernel)
        cv_values = np.zeros((n_samples, n_targets, len(self.alphas)))
        C = []

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        error = scorer is None

        for i, alpha in enumerate(self.alphas):
            if error:
                out, c = _errors(alpha, y, v, Q, QT_y)
            else:
                out, c = _values(alpha, y, v, Q, QT_y)
            cv_values[:, :, i] = out
            C.append(c)

        if self.store_cv_values:
            self.cv_values_ = cv_values

        if error:
            if self.alpha_per_target:
                # Find the best alpha for each target
                best = cv_values.mean(axis=0).argmin(axis=1)
            else:
                # Find the best alpha overall
                best = np.mean(cv_values.reshape(-1, len(self.alphas)),
                               axis=0).argmin()
        else:
            # The scorer wants an object that will make the predictions, but
            # they are already computed efficiently by RidgeGCV. This
            # identity_estimator will just return them
            def identity_estimator():
                pass
            identity_estimator.decision_function = lambda y_predict: y_predict
            identity_estimator.predict = lambda y_predict: y_predict

            if self.alpha_per_target:
                out = [scorer(identity_estimator, target, cv_values[:, i, j])
                       for j in range(len(self.alphas))
                       for i, target in enumerate(y.T)]
                best = np.argmax(out, axis=1)
            else:
                out = [scorer(identity_estimator, y.ravel(),
                              cv_values[:, :, i].ravel())
                       for i in range(len(self.alphas))]
                best = np.argmax(out)

        self.alpha_ = self.alphas[best]

        if self.alpha_per_target:
            self.dual_coef_ = np.vstack([C[j][:, i]
                                         for i, j in enumerate(best)]).T
        else:
            self.dual_coef_ = C[best]

        self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)

        # If the original y was flat, remove some dimensions to match
        if flat_y:
            if self.store_cv_values:
                self.cv_values_ = cv_values.reshape(n_samples,
                                                    len(self.alphas))
            self.coef_ = self.coef_.ravel()

        self._set_intercept(X_offset, y_offset, X_scale)

        return self
