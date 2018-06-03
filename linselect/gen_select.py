from itertools import cycle
import numpy as np
from .base import Base


class GenSelect(Base):
    """
    GenSelect -- Efficient General Stepwise Linear Regression

    A class for carrying out general, single-step linear feature selection
    protocols: protocols that include both forward and reverse search steps.
    Best results seen so far are stored, allowing for review.  This also allows
    for a search to be continued with repositioning or step protocol
    adjustments, as desired.

    Special Attributes
    ------------------
    best_results: dict
        Keys of this dict correspond to feature subset size.  The value for a
        given key is also a dict -- one characterizing the best subset seen so
        far of this size.  These inner dicts have two keys, `s` and `cod`.  The
        first holds a Boolean array specifying which features were included in
        the subset and the second holds the corresponding COD.
    """
    def __init__(self, dtype=np.float32):
        """
        input
        -----
        dtype: numeric variable type
            Computations will be carried out using this level of precision.
            Note: Lower precision types can result in faster computation.
            However, for nearly redundant data sets these can sometimes result
            in nan results populating the ordered_cods.
        """
        super(GenSelect, self).__init__(dtype)
        self.best_results = {}

    def _update_best_results(self):
        """
        If current COD is larger than prior best at current predictor set size,
        update best_results dict.
        """
        # do not update if current COD is nan
        if np.isnan(self.cod):
            return
        # possible update
        size_s = np.sum(self.s)
        if size_s in self.best_results.keys():
            if self.cod <= self.best_results[size_s]["cod"]:
                # better result before, do not update
                return
        new_dict = {"s": self.s.copy(), "cod": self.cod}
        self.best_results[size_s] = new_dict

    def position(self, X=None, s=None, mobile=None, targets=None):
        """
        Set the operating conditions of the stepwise search.

        Parameters
        ----------
        X: np.array, (n_examples, n_features), default None
            The data set to be fit.  This must be passed in the first call to
            this method, but should not need to be passed again in any
            following repositioning call.  Must be numeric.

        s: np.array, (n_features), default None
            This is a Boolean array that specifies which predictor set to use
            when we begin (or continue) our search.  If the index i is set to
            True, the corresponding feature i will be included in the initial
            predictor set.

        mobile: np.array, (n_features), default None
            This is a Boolean array that specifies which of the features are
            locked into or out of our fit -- if the index i is set to True, the
            corresponding feature i can be moved into or out of the predictor
            set.  Otherwise, the feature i is locked in the set specified by
            the passed s argument.

        targets: np.array, (n_features), default None
            This is a Boolean array that specifies which of the columns of X
            are to be fit -- analogs of y in the FwdSelect and RevSelect
            algorithms.  If the index i is set to True, the corresponding
            column i will be placed in the target set.  Once set, this should
            not be passed again in any following repositioning call.
        """
        self._setup(X, s, mobile, targets)
        self._set_efroymson()
        self._set_cod()
        self._update_best_results()

    def search(self, protocol=(2, 1), steps=1):
        """
        (Continue) stepwise search and update elements of best_results
        throughout.

        Parameters
        ----------
        protocol: tuple of 2 ints
            First element is number of forward steps to take each iteration.
            The second is the number of reverse.  E.g., default is (2, 1).
            This results in two forward steps and one reverse step being taken
            each iteration.  E.g., if (1, 0) is passed, one forward step is
            taken followed by zero reverse steps.  That is, we do a forward
            search, etc.

        steps: int
            The total number of steps to take.  Note that if a step is
            requested but there are no mobile moves available no step will be
            taken.  This can happen, e.g., when requesting a forward step, but
            all mobile features are already in the predictor set, s.
        """
        # write out full protocol for the current search request
        protocol = [i[0] for i in zip(
            cycle('f' * protocol[0] + 'r' * protocol[1]),
            range(steps))]

        # carry out stepwise search
        for p in protocol:
            if p == 'f':
                self._forward_step()
            else:
                self._reverse_step()
            self._update_best_results()

    def forward_cods(self):
        """
        Returns the COD increase that would result from each possible movement
        of an element outside of the predictor set inside.

        Returns
        -------
        cod_gains : np.array (self.dimension, )
            This array's index i specifies the gain in target COD that would
            result if feature i were to move into s.  Values corresponding to
            unavailable moves are set to 0.
        """
        cod_gains = np.full(self.dimension, 0, dtype=self.dtype)
        cod_gains[np.ix_(~self.s * self.mobile)] = np.einsum('ij,ij->i',
            (self.C - self.M)[np.ix_(
                ~self.s * self.mobile, ~self.s * self.targets)],
            (self.C - self.M)[np.ix_(
                ~self.s * self.mobile, ~self.s * self.targets)]) / (
                self.dtype(1.0) - self.C.diagonal()[np.ix_(
                    ~self.s * self.mobile)])
        return cod_gains

    def reverse_cods(self):
        """
        Returns the COD decrease that would result from each possible movement
        of an element inside of the predictor set outside.

        Returns
        -------
        cod_costs : np.array (self.dimension, )
            This array's index i specifies the drop in target COD that would
            result if feature i were to move outside of s.  Values
            corresponding to unavailable moves are set to 0.
        """
        cod_costs = np.full(self.dimension, 0, dtype=self.dtype)
        cod_costs[np.ix_(self.s * self.mobile)] = (
            self.targets[self.s * self.mobile]
            + np.einsum('ij,ij->i',
                self.C[np.ix_(
                    self.s * self.mobile, ~self.s * self.targets)],
                self.C[np.ix_(
                    self.s * self.mobile, ~self.s * self.targets)])
            ) / self.C.diagonal()[self.s * self.mobile]
        return cod_costs
