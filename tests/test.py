from unittest import TestCase
import numpy as np
import linselect


def _shift_scale(x):
    """
    Subtract mean and normalize an array.
    Returns one with mean zero and variance
    one.
    """
    mean = np.mean(x)
    sigma = np.std(x)
    return (x - mean) / sigma


def _generate_normalized_array(m, n):
    """
    m = number of rows, n number of columns
    """
    print m, type(m)
    print n, type(n)
    x = np.random.randn(m, n)
    for col in range(n):
        x[:, col] = _shift_scale(x[:, col])
    return x


class TestSelectionMethods():
    """
    Check that the cod returned by our method
    agrees with external package fit.
    """
    n = 11
    m = 2000
    N = 5
    clusters = 3
    cluster_size = 5

    # First tests check we report correct COD with all classes.
    def test_gen_supervised_cod(TestCase, m=m, n=n, N=N):
        # generate a data set
        X = _generate_normalized_array(m, n)

        # start with first two features in s
        s = [False for i in range(n)]
        s[0] = True
        s[1] = True

        # make last column the target variable
        targets = [False for i in range(n)]
        targets[-1] = True

        # make all mobile except first two and target
        mobile = [True for i in range(n)]
        mobile[0] = False
        mobile[1] = False
        mobile[-1] = False

        # now search with a particular protocol
        selector = linselect.GenSelect()
        selector.position(
            X=X, s=s, mobile=mobile, targets=targets)
        selector.search(protocol=(3, 2), steps=25)

        # check best cod found using N features
        s_at_N = selector.best_results[N]['s']
        cod_at_N = selector.best_results[N]['cod']

        # compare to cod of fit from numpy using same features
        X_at_N = X[:, s_at_N]
        y = X[:, -1]
        squared_error = np.linalg.lstsq(X_at_N, y)[1]
        assert(np.isclose(cod_at_N, 1 - squared_error[0] / m, atol=1e-05))

    def test_gen_unsupervised_cod(TestCase, m=m, n=n, N=N):
        # generate a data set
        X = _generate_normalized_array(m, n)

        # now search with default protocol
        selector = linselect.GenSelect()
        selector.position(X=X)
        selector.search(protocol=(3, 2), steps=25)

        # check best cod found using N features
        s_at_N = selector.best_results[N]['s']
        cod_at_N = selector.best_results[N]['cod']

        # compare to cod of fit from numpy using same features
        X_at_N = X[:, s_at_N]
        y_at_N = X[:, ~s_at_N]
        squared_error = np.linalg.lstsq(X_at_N, y_at_N)[1]
        assert(np.isclose(cod_at_N, n - np.sum(squared_error) / m, atol=1e-05))

    def test_fwd_supervised_cod(TestCase, m=m, n=n, N=N):
        # generate a data set with three target (y) vars.
        X = _generate_normalized_array(m, n)
        y = _generate_normalized_array(m, 3)

        # now carry out reverse selection
        selector = linselect.FwdSelect()
        selector.fit(X, y)

        # check best cod found using N features
        s_at_N = selector.ordered_features[:N]
        cod_at_N = selector.ordered_cods[N - 1]

        # compare to cod of fit from numpy using same features
        X_at_N = X[:, s_at_N]
        squared_error = np.linalg.lstsq(X_at_N, y)[1]
        assert(np.isclose(cod_at_N, 3 - np.sum(squared_error) / m, atol=1e-05))

    def test_fwd_unsupervised_cod(TestCase, m=m, n=n, N=N):
        # generate a data set with two target (y) vars.
        X = _generate_normalized_array(m, n)

        # now carry out reverse selection
        selector = linselect.FwdSelect()
        selector.fit(X)

        # check best cod found using N features
        cod_at_N = selector.ordered_cods[N - 1]

        # compare to cod of fit from numpy using same features
        X_at_N = X[:, selector.ordered_features[:N]]
        y_at_N = X[:, selector.ordered_features[N:]]
        squared_error = np.linalg.lstsq(X_at_N, y_at_N)[1]
        assert(np.isclose(cod_at_N, n - np.sum(squared_error) / m, atol=1e-05))

    def test_rev_supervised_cod(TestCase, m=m, n=n, N=N):
        # generate a data set with two target (y) vars.
        X = _generate_normalized_array(m, n)
        y = _generate_normalized_array(m, 2)

        # now carry out reverse selection
        selector = linselect.RevSelect()
        selector.fit(X, y)

        # check best cod found using N features
        cod_at_N = selector.ordered_cods[N - 1]

        # compare to cod of fit from numpy using same features
        X_at_N = X[:, selector.ordered_features[:N]]
        squared_error = np.linalg.lstsq(X_at_N, y)[1]
        assert(np.isclose(cod_at_N, 2 - np.sum(squared_error) / m, atol=1e-05))

    def test_rev_unsupervised_cod(TestCase, m=m, n=n, N=N):
        # generate a data set with two target (y) vars.
        X = _generate_normalized_array(m, n)

        # now carry out reverse selection
        selector = linselect.RevSelect()
        selector.fit(X)

        # check best cod found using N features
        cod_at_N = selector.ordered_cods[N - 1]

        # compare to cod of fit from numpy using same features
        X_at_N = X[:, selector.ordered_features[:N]]
        y_at_N = X[:, selector.ordered_features[N:]]
        squared_error = np.linalg.lstsq(X_at_N, y_at_N)[1]
        assert(np.isclose(cod_at_N, n - np.sum(squared_error) / m, atol=1e-05))

    # Tests below ensure we select the best candidate each time.
    def test_fwd_supervised_ordering(TestCase, m=m, n=n):
        # Take y linear in X's columns, with coefficient increasing with index.
        X = _generate_normalized_array(m, n)
        y = np.dot(X, np.arange(1, n + 1)).reshape(-1, 1)

        # Forward selection
        selector = linselect.FwdSelect()
        selector.fit(X, y)

        # Ensure correct feature order
        assert(selector.ordered_features == range(n)[::-1])

    def test_fwd_unsupervised_ordering(
            TestCase, m=m, clusters=clusters, cluster_size=cluster_size):
        # Generate well-separated clusters of features
        centroids = 100 * np.random.rand(m, clusters)
        X = np.random.rand(m, clusters * cluster_size)
        for i in range(clusters):
            X[:, i * cluster_size:(i+1) * cluster_size] += centroids[:, [i]]

        # Forward selection
        selector = linselect.FwdSelect()
        selector.fit(X)

        # Ensure top features are each from a different cluster
        first_features = selector.ordered_features[:clusters]
        first_clusters = sorted([c // cluster_size for c in first_features])
        assert first_clusters == range(clusters)

    def test_rev_supervised_ordering(TestCase, m=m, n=n):
        # Take y linear in X's columns, with coefficient increasing with index.
        X = _generate_normalized_array(m, n)
        y = np.dot(X, np.arange(1, n + 1)).reshape(-1, 1)

        # Reverse selection
        selector = linselect.RevSelect()
        selector.fit(X, y)

        # Ensure correct feature order
        assert(selector.ordered_features == range(n)[::-1])

    def test_rev_unsupervised_ordering(
            TestCase, m=m, clusters=clusters, cluster_size=cluster_size):
        # Generate well-separated clusters of features
        centroids = 100 * np.random.rand(m, clusters)
        X = np.random.rand(m, clusters * cluster_size)
        for i in range(clusters):
            X[:, i * cluster_size:(i+1) * cluster_size] += centroids[:, [i]]

        # Reverse selection
        selector = linselect.RevSelect()
        selector.fit(X)

        # Ensure top features are each from a different cluster
        first_features = selector.ordered_features[:clusters]
        first_clusters = sorted([c // cluster_size for c in first_features])
        assert first_clusters == range(clusters)

    def test_gen_supervised_ordering(TestCase, m=m, n=n):
        # Take last col linear others, with coefficient increasing with index.
        X = _generate_normalized_array(m, n)
        X[:, -1] = np.dot(X[:, :-1], np.arange(1, n))

        # General selection set up
        mobile = np.array([True for i in range(n)])
        mobile[-1] = False
        targets = ~mobile
        selector = linselect.GenSelect()
        selector.position(X, mobile=mobile, targets=targets)

        # Now sweep back and forth a few times
        selector.search(protocol=(1, 0), steps=n)
        selector.search(protocol=(0, 1), steps=n)
        selector.search(protocol=(1, 0), steps=n)

        # Ensure correct features included with each subset size
        for k in range(n):
            assert np.all(
                selector.best_results[k]['s'][-(k + 1):-1])

    def test_gen_unsupervised_ordering(
            TestCase, m=m, clusters=clusters, cluster_size=cluster_size):
        # Generate well-separated clusters of features
        centroids = 100 * np.random.rand(m, clusters)
        X = np.random.rand(m, clusters * cluster_size)
        for i in range(clusters):
            X[:, i * cluster_size:(i+1) * cluster_size] += centroids[:, [i]]

        # Now sweep back and forth a few times
        selector = linselect.GenSelect()
        selector.position(X)
        selector.search(protocol=(1, 0), steps=clusters * cluster_size)
        selector.search(protocol=(0, 1), steps=clusters * cluster_size)
        selector.search(protocol=(1, 0), steps=clusters * cluster_size)

        # Ensure top features are each from a different cluster
        first_features = np.where(selector.best_results[clusters]['s'])[0]
        first_clusters = sorted([c // cluster_size for c in first_features])
        assert first_clusters == range(clusters)

    # Tests below ensure dtype is respected throughout algorithms
    def test_gen_dtype_maintained(TestCase, m=m, n=n):
        X = _generate_normalized_array(m, n)

        # General selection set up
        mobile = np.array([True for i in range(n)])
        mobile[-1] = False
        targets = ~mobile
        selector = linselect.GenSelect(dtype=np.float32)
        selector.position(X, mobile=mobile, targets=targets)

        # Now sweep back and forth a few times
        selector.search(protocol=(1, 0), steps=n)
        selector.search(protocol=(0, 1), steps=n)
        selector.search(protocol=(1, 0), steps=n)

        # Ensure dtype respected at each best result outcome
        for br in selector.best_results:
            assert isinstance(selector.best_results[br]['cod'], selector.dtype)

    def test_fwd_dtype_maintained(TestCase, m=m, n=n):
        X = _generate_normalized_array(m, n)
        selector = linselect.FwdSelect(dtype=np.float32)
        selector.fit(X)
        for cod in selector.ordered_cods:
            assert isinstance(cod, selector.dtype)

    def test_rev_dtype_maintained(TestCase, m=m, n=n):
        X = _generate_normalized_array(m, n)
        selector = linselect.RevSelect(dtype=np.float32)
        selector.fit(X)
        for cod in selector.ordered_cods:
            assert isinstance(cod, selector.dtype)
