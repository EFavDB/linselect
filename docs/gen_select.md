# `GenSelect` 

A class for carrying out general, single-step linear feature selection
protocols: protocols that include both forward and reverse search steps.  Best
results seen so far are stored, allowing for review.  This also allows for a
search to be continued with repositioning or step protocol adjustments, as
desired.

## Special Attributes
#### `best_results: dict`
  Keys of this dict correspond to feature subset size.  The value for a given
  key is also a dict -- one characterizing the best subset seen so far of this
  size.  These inner dicts have two keys, `s` and `cod`.  The first holds a
  Boolean array specifying which features were included in the subset and the
  second holds the corresponding COD.

## Methods
#### `__init__(self, dtype=np.float32)`
  *Parameters*

  * `dtype`: numeric variable type
  
    Computations will be carried out using this level of precision. Note: Lower
    precision types result in faster computation. However, for nearly redundant
    data sets these can sometimes result in nan results populating the
    `ordered_cods`.

#### `position(self, X=None, s=None, mobile=None, targets=None)`

Set the operating conditions of the stepwise search.

*Parameters*
  
 *  `X: np.array, (n_examples, n_features)`

    The data set to be fit.  This must be passed in the first call to this
    method, but should not need to be passed again in any following
    repositioning call.  Must be numeric.

 *  `s: np.array, (n_features)`

    This is a Boolean array that specifies which predictor set to use when we
    begin (or continue) our search.  If the index `i` is set to `True`, the
    corresponding feature `i` will be included in the initial predictor set.

 *  `mobile: np.array, (n_features)`

    This is a Boolean array that specifies which of the features are locked
    into or out of our fit -- if the index `i` is set to `True`, the
    corresponding feature `i` can be moved into or out of the predictor set.
    Otherwise, the feature `i` is locked in the set specified by the passed `s`
    argument.


 * `targets: np.array, (n_features)`

    This is a Boolean array that specifies which of the columns of `X` are to
    be fit -- analogs of `y` in the `FwdSelect` and `RevSelect` algorithms.  If
    the index `i` is set to `True`, the corresponding column `i` will be placed
    in the target set.  Once set, this should not be passed again in any
    following repositioning call.

#### `search(self, protocol=(2,1), steps=1)`

(Continue) stepwise search and update elements of `best_results` throughout. 

  *Parameters*

  * `protocol: tuple of 2 ints`

    First element is number of forward steps to take each iteration.  The
    second is the number of reverse.  E.g., default is `(2, 1)`.  This results
    in two forward steps and one reverse step being taken each iteration.
    E.g., if `(1, 0)` is passed, one forward step is taken followed by zero
    reverse steps.  That is, we do a forward search, etc.

  * `steps: int`

    The total number of steps to take.  Note that if a step is
    requested but there are no mobile moves available no step will be
    taken.  This can happen, e.g., when requesting a forward step, but
    all mobile features are already in the predictor set `s`.

#### `forward_cods(self)`

Returns the COD increase that would result from each possible movement of an
element outside of the predictor set inside.

  *Returns*

  * `cod_gains : np.array (self.dimension, )`

    This array's index `i` specifies the gain in target COD that would result
    if feature `i` were to move into `s`.  Values corresponding to unavailable
    moves are set to 0.

#### `reverse_cods(self)`

Returns the COD decrease that would result from each possible movement of an
element inside of the predictor set outside.

  *Returns*

  * `cod_costs : np.array (self.dimension, )`


    This array's index `i` specifies the drop in target COD that would result
    if feature `i` were to move outside of `s`.  Values corresponding to
    unavailable moves are set to 0.

## Supervised example
The code below carries out a forward and a reverse sweep on a random,
correlated dataset, `X`:  After generating the data set, we initialize the
search by setting the last column of `X` as the target and fix this as a
non-predictor.  We start the search at the point where no features are
included as predictors and take a forward sweep.  Finally, we continue the
search with a backwards sweep.  Because we did not reposition before this, the
reverse sweep starts off where we left off -- i.e., at the point where all
allowed features are included as predictors. Notice that the second sweep
resulted in an improved three feature fit.

```python
import numpy as np
from linselect import GenSelect

# Constants
M = 50    # example count 
N = 5     # feature count 
K = 3     # best results sample point

# Generate a random, correlated data set
c = np.random.rand(N, N)
C = np.dot(c.T, c)
X = np.random.multivariate_normal(np.zeros(N), C, M)

# Initialize the selector and position search with last column as target.
s = np.array([False for i in range(N)])    # initially no predictors
targets = s.copy(); targets[-1] = True     # last feature is target
mobile = ~targets                          # fix target as non-predictor
selector = GenSelect()
selector.position(X, s=s, targets=targets, mobile=mobile)

# Take a forward sweep and print best K feature model found
selector.search(protocol=(1, 0), steps=N-1)
print selector.best_results[K]
# {'s': array([False,  True,  True,  True, False], dtype=bool), 'cod': 0.912}

# Continue search with a reverse sweep
selector.search(protocol=(0, 1), steps=N-1)
print selector.best_results[K]
# {'s': array([ True,  True,  True, False, False], dtype=bool), 'cod': 0.914}
```

## Unsupervised example
The code below carries out an unsupervised selection analysis on a random,
correlated data set, `X`.  After generating the data set, we initialize the
selector and position it by passing in `X`.  The arguments `s`, `mobile`, and
`targets` are not passed, so are set to their default values.  This starts the
search with no features as predictors, all features mobile, and all features
as targets -- i.e., unsupervised selection.  We take a forward sweep,
reposition at the best three feature fit seen in that sweep, then sweep back
and forth a few times.  Note that the best three feature model found beats
that from the first sweep.

```python
import numpy as np
from linselect import GenSelect

# Constants
M = 50    # example count 
N = 5     # feature count 
K = 3     # best results sample point

# Generate a random, correlated data set
c = np.random.rand(N, N)
C = np.dot(c.T, c)
X = np.random.multivariate_normal(np.zeros(N), C, M)

# Initialize the selector and position search with last column as target.
selector = GenSelect()
selector.position(X)

# Take a forward sweep and print best K feature model found
selector.search(protocol=(1, 0), steps=N-1)
print selector.best_results[K]
# {'s': array([ True, False,  True, False,  True], dtype=bool), 'cod': 4.89}

# Continue search with a few reverse and forward sweeps
selector.position(s=selector.best_results[K]['s'])
selector.search(protocol=(N-1, N-1), steps=6*(N-1))
print selector.best_results[K]
# {'s': array([ True,  True, False,  True, False], dtype=bool), 'cod': 4.95}
```
