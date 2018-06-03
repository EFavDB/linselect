import numpy as np
from .base import Base


class RevSelect(Base):
    """
    RevSelect -- Efficient Reverse Stepwise Linear Regression

    A class for carrying out reverse, single-step linear feature selection
    protocols.  At each step, the feature that is selected is that which
    reduces the total COD (coefficient of determination, aka R^2) by the least
    amount.  The feature ordering and CODs are stored, allowing for review.

    Special Attributes
    ------------------
    ordered_features: list
        List of the feature indices.  The ordering is the reverse of that in
        which the features were removed from the predictor set during
        selection.

    ordered_cods: list
        This list's index i specifies the COD that results if only the first i
        features of ordered_features are taken as predictors (large COD
        values are better and a perfect score = n_targets).
    """
    def __init__(self, dtype=np.float32):
        """
        Parameters
        ----------
        dtype: numeric variable type
            Computations will be carried out using this level of precision.
            Note: Lower precision types result in faster computation. However,
            for nearly redundant data sets these can sometimes result in nan
            results populating the ordered_cods.
        """
        super(RevSelect, self).__init__(dtype)

    def fit(self, X, y=None):
        """
        Method fits passed data, evaluates self.ordered_features and
        self.ordered_cods.

        Parameters
        ----------
        X : np.array (n_examples, n_features)
            Data array of features containing samples across all features. Must
            be numeric.

        y : np.array (n_examples, n_targets), default None
            Array of label values for each example. If n_targets > 1 we seek
            the features that maximize the sum total COD over the separate
            labels.  If None passed, we carry out unsupervised selection,
            treating all features as targets.  If passed, must be numeric.

        Returns
        -------
        self : fitted instance
        """
        # setup
        if y is not None:
            X = np.hstack((X, y))
            self._setup(X)
            self.mobile[-y.shape[1]:] = False
            self.targets = ~self.mobile
        else:
            self._setup(X)
        self.s = self.mobile.copy()
        self._set_efroymson()
        self._set_cod()
        self.ordered_cods = [self.cod]
        self.ordered_features = list()
        # carry out stepwise procedure
        for _ in range(np.sum(self.mobile)):
            opt_index = self._reverse_step()
            self.ordered_features.append(opt_index)
            self.ordered_cods.append(self.cod)
        # reverse list orders for consistency with FwdSelect
        self.ordered_features = self.ordered_features[::-1]
        self.ordered_cods = self.ordered_cods[-2::-1]
        return self
