import numpy as np


class Base(object):
    """
    Base class for the stepwise selection algorithms.

    Attributes common to all child classes
    --------------------------------------
    M : np.array (self.dimension, self.dimension)
        The correlation matrix of all variables.  Numpy automatically scales
        all variables so that each has mean 0 and variance 1, as needed.

    C : np.array (self.dimension, self.dimension)
        This holds all information needed to identify the gains and costs
        associated with moving features into or out of the predictor set at
        each step of the process.

    s : np.array, Boolean (self.dimension, )
        This array's index i specifies whether variable i is currently in the
        predictor set.

    mobile : np.array, Boolean (self.dimension, )
        This array's index i specifies whether variable i is free to move in
        and out of the predictor set.

    targets : np.array, Boolean (self.dimension, )
        This array's index i specifies whether variable i is in the target set,
        i.e., one of the variables we are trying to fit.

    dimension : int
        This is the number of variables, including both the features in X and
        the targets in y (when passed).

    dtype : numeric variable type
        Computations will be carried out using this level of precision.
        Note: Lower precision types result in faster computation. However, for
        nearly redundant data sets these can sometimes result in nan results
        populating the ordered_cods.

    Private methods
    ---------------
    _setup(self, X, s, mobile, targets)
        Sets passed search parameters.

    _set_efroymson(self)
        Sets the Efroymson matrix C.

    _set_cod(self)
        Evaluates current COD of the target set.

    _forward_step(self)
        Identifies optimal forward step and COD gain associated with this.

    _reverse_step(self)
        Identifies optimal reverse step and COD cost associated with this.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        self.M = None
        self.s = None
        self.mobile = None
        self.targets = None

    def _setup(self, X=None, s=None, mobile=None, targets=None):
        """
        Reset various parameters if passed.  If not passed and values are not
        currently set, default to forward selection initial placements.
        """
        if X is None and self.M is None:
            raise ValueError('Must pass X to initialize')
        if X is not None:
            if self.M is not None:
                raise ValueError('X was previously passed')
            X = np.array(X)
            self.dimension = X.shape[1]
            self.indices = np.arange(self.dimension)
            self.M = np.corrcoef(np.transpose(X)).astype(self.dtype)
        if s is not None:
            self.s = np.array(s)
        elif self.s is None:
            self.s = np.full(self.dimension, False, dtype=bool)
        if mobile is not None:
            self.mobile = np.array(mobile)
        elif self.mobile is None:
            self.mobile = np.full(self.dimension, True, dtype=bool)
        if targets is not None:
            if self.targets is not None:
                raise ValueError('targets was previously set')
            self.targets = np.array(targets)
        elif self.targets is None:
            self.targets = np.full(self.dimension, True, dtype=bool)

    def _set_efroymson(self):
        """
        Set the Efroymson matrix, C.
        """
        self.C = (np.diag(~self.s).dot(self.M) + np.diag(self.s)).dot(
            np.linalg.inv(np.diag(self.s).dot(self.M).dot(np.diag(self.s))
                          + np.diag(~self.s)) - np.diag(~self.s)).dot(
            self.M.dot(np.diag(~self.s)) + np.diag(self.s))

    def _set_cod(self):
        """
        Evaluate current COD of target set.
        """
        self.cod = (np.diag(self.C)[~self.s * self.targets].sum()
            + self.dtype(np.sum(self.s * self.targets)))

    def _forward_step(self):
        """
        Take optimal forward step and update C, s, and COD.  Return optimal
        index.
        """
        # check there are mobile options, exit if not
        if np.sum(self.mobile * ~self.s) == 0:
            return

        # identify COD gain of each candidate move, select best
        gains = np.einsum('ij,ij->i',
            (self.C - self.M)[np.ix_(
                ~self.s * self.mobile, ~self.s * self.targets)],
            (self.C - self.M)[np.ix_(
                ~self.s * self.mobile, ~self.s * self.targets)]) / (
                self.dtype(1.0) - self.C.diagonal()[np.ix_(
                    ~self.s * self.mobile)])
        opt_index = np.argmax(gains)
        opt_gain = gains[opt_index]

        # map opt_index to index in full feature set
        opt_index = self.indices[np.ix_(
            ~self.s * self.mobile)][opt_index]

        # save what the moved column should change to
        C_opt_index_update = (
            self.M[opt_index] * ~self.s - self.C[opt_index]) / (
                self.dtype(1.0) - self.C[opt_index, opt_index])
        C_opt_index_update[opt_index] = self.dtype(1.0) / (
            self.dtype(1.0) - self.C[opt_index, opt_index])

        # update C
        x = self.C[opt_index] - self.M[opt_index] * ~self.s
        for index in range(self.dimension):
            self.C[index] -= x[index] * x / x[opt_index]

        # fix the opt_index column and row
        self.C[opt_index, :] = C_opt_index_update
        self.C[:, opt_index] = C_opt_index_update

        # update s
        self.s[opt_index] = True

        # update COD, refresh if update gives nan
        test_cod = self.cod + opt_gain
        if np.isnan(test_cod):
            self._set_cod()
        else:
            self.cod = test_cod

        # return opt_index
        return opt_index

    def _reverse_step(self):
        """
        Take optimal reverse step and update C, s, and COD.  Return optimal
        index.
        """
        # check there are mobile options, exit otherwise
        if np.sum(self.mobile * self.s) == 0:
            return

        # evaluate COD costs
        costs = (
            self.targets[self.s * self.mobile]
            + np.einsum('ij,ij->i',
                self.C[np.ix_(
                    self.s * self.mobile, ~self.s * self.targets)],
                self.C[np.ix_(
                    self.s * self.mobile, ~self.s * self.targets)])
            ) / (self.C.diagonal()[self.s * self.mobile])
        opt_index = np.argmin(costs)
        opt_cost = costs[opt_index]

        # map opt_index to index in full feature set
        opt_index = self.indices[np.ix_(self.s * self.mobile)][opt_index]

        # save what the moved column should change to
        C_opt_index_update = (
            self.M[opt_index] * ~self.s - self.C[opt_index] / (
            self.C[opt_index, opt_index]))
        C_opt_index_update[opt_index] = self.M[opt_index, opt_index] - (
            self.dtype(1.0) / self.C[opt_index, opt_index])

        # update C
        x = self.C[opt_index].copy()
        for index in range(self.dimension):
            self.C[index] -= x[index] * x / x[opt_index]

        # fix the top_index column and row
        self.C[opt_index, :] = C_opt_index_update
        self.C[:, opt_index] = C_opt_index_update

        # update s
        self.s[opt_index] = False

        # update COD, refresh if update gives nan
        test_cod = self.cod - opt_cost
        if np.isnan(test_cod):
            self._set_cod()
        else:
            self.cod = test_cod

        # return opt_index
        return opt_index
