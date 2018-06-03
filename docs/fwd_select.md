# `FwdSelect` 

A class for carrying out forward, single-step linear feature selection
protocols.  At each step, the feature that is selected is that which increases
the total COD (coefficient of determination, aka R^2) by the largest amount.
The feature ordering and CODs are stored, allowing for review.

## Special Attributes
#### `ordered_features: list`
  List of the feature indices.  The ordering is that in which the features
  were added to the predictor set during selection.

#### `ordered_cods: list`
  This list's index `i` specifies the COD that results if only the first `i`
  features of `ordered_features` are taken as predictors (large COD
  values are better and a perfect score = `n_targets`).

## Methods
#### `__init__(self, dtype=np.float32)`

*Parameters*

 * `dtype`: numeric variable type

    Computations will be carried out using this level of precision. Note: Lower
    precision types result in faster computation. However, for nearly redundant
    data sets these can sometimes result in `nan` results populating the
    `ordered_cods`.

#### `fit(self, X, y)`

Method fits passed data, evaluates `self.ordered_features` and
`self.ordered_cods`.
  
*Parameters*

 * `X : np.array (n_examples, n_features)`

    Data array of features containing samples across all features.  Must be
    numeric.
  
 * `y : np.array (n_examples, n_targets)`, default `None`
  
    Array of label values for each example. If `n_targets > 1` we seek the
    features that maximize the sum total COD over the separate labels.  If
    `None` passed, we carry out unsupervised selection, treating all features
    as targets.  If passed, must be numeric.
  
*Returns*

 * `self` : fitted instance

## Supervised example
The code below carries out a forward stepwise selection procedure on a
supervised example:  We construct a random `X` array and then define a `y`
array that is a specified linear combination of the columns of `X`.  Passing
these to a `FwdSelect` instance's `fit` method, the `ordered_features` and
`ordered_cods` lists are then evaluated and stored.

```python
import numpy as np
from linselect import FwdSelect

# Generate a random data set
M = 1000    # example count
N = 4       # feature count
X = np.random.rand(M, N)
y = np.dot(X, np.arange(N).reshape([-1, 1])) 

# Carry out forward selection
selector = FwdSelect()
selector.fit(X, y)

# Get the ordered features and COD lists
print selector.ordered_features
# [3, 2, 1, 0]

print selector.ordered_cods
# [0.64, 0.93, 1.00, 1.00]
```

## Unsupervised example
The code below carries out a forward stepwise selection procedure on an
unsupervised example:  We construct a random `X` array and then pass this to a
`FwdSelect` instance's `fit` method, the `ordered_features` and `ordered_cods`
lists are then evaluated and stored.  Note that to generate this example, we
use numpy's `multivariate_normal` sampler, after first generating a random,
non-negative definite covariance matrix.  Apparently, the zeroth feature in
this sample was able to account for about 2.76 / 4.00 of the full variance
exhibited in the data set.

```python
import numpy as np
from linselect import FwdSelect

# Constants
M = 100    # example count
N = 4      # feature count

# Generate a random, correlated data set
c = np.random.rand(N, N)
C = np.dot(c.T, c)
X = np.random.multivariate_normal(np.zeros(N), C, M)

# Carry out forward selection
selector = FwdSelect()
selector.fit(X)

# Get the ordered features and COD lists
print selector.ordered_features
# [0, 3, 1, 2] 

print selector.ordered_cods
# [2.76, 3.54, 3.99, 4.00]
```
