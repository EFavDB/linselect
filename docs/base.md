# `Base` 

Base class for the stepwise selection algorithms.

## Attributes common to all child classes

#### `M : np.array (self.dimension, self.dimension)`
The correlation matrix of all variables.  Numpy automatically scales all
variables so that each has mean 0 and variance 1, as needed.

#### `C : np.array (self.dimension, self.dimension)`
This holds all information needed to identify the gains and costs associated
with moving features into or out of the predictor set at each step of the
process.

#### `s : np.array, Boolean (self.dimension, )`
This array's index `i` specifies whether variable `i` is currently in the
predictor set.

#### `mobile : np.array, Boolean (self.dimension, )`
This array's index `i` specifies whether variable `i` is free to move in and
out of the predictor set.

#### `targets : np.array, Boolean (self.dimension, )`
This array's index `i` specifies whether variable `i` is in the target set,
i.e., one of the variables we are trying to fit.

#### `dimension : int`
This is the number of variables, including both the features in `X` and the
targets in `y` (when passed).

#### `dtype : numeric variable type`
Computations will be carried out using this level of precision.  Note: Lower
precision types result in faster computation. However, for nearly redundant
data sets these can sometimes result in `nan` results populating the
`ordered_cods`.
