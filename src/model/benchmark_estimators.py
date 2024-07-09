"""
Suite of benchmark estimators.

Mainly wrappers around existing packages for IV estimation.
"""
import numpy as np
import pandas as pd

from linearmodels.iv.model import IV2SLS, IVLIML, _OLS

from ..model.base_estimator import BaseEstimator
class TSLS(BaseEstimator):
    def estimate(self, T, X, Z, Y, **kwargs):
        self.model = IV2SLS(dependent=Y, 
                            exog=X, 
                            endog=T, 
                            instruments=Z,
                            **kwargs)
        results = self.model.fit()
        return {'tau': results.params['T'], 'se': results.std_errors['T']}


class LIML(BaseEstimator):
    def estimate(self, T, X, Z, Y, **kwargs):
        self.model = IVLIML(dependent=Y, 
                            exog=X, 
                            endog=T, 
                            instruments=Z,
                            **kwargs)
        results = self.model.fit()
        return {'tau': results.params['T'], 'se': results.std_errors['T']}


class OLS(BaseEstimator):
    def estimate(self, T, X, Z, Y, **kwargs):
        self.model = _OLS(dependent=Y, 
                        exog=pd.concat([X, T], axis=1),
                        **kwargs)
        results = self.model.fit()
        return {'tau': results.params['T'], 'se': results.std_errors['T']}


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

class MHML(BaseEstimator):
    """Implements the sample-split machine learning IV from Chen 2020:
    'Mostly Harmless ML' (MHML)
    
    See https://gist.github.com/jiafengkevinchen/439ae146d235834efa884533d1ff9915
    """

    """
    Implementation plan:

    - allow for choice of arbitrary ML esetimator?
    - cross-fit, choose number of folds
    - then pass into a TSLS 
    - bootstrap sampling for bias and variance??
    """
    n_folds = None
    estimator = None

    def __init__(self, n_folds=2, estimator=None):
        """
        Args:
        - n_folds (int): number of cross-fitted folds to generate, default 2
        - estimator (sklearn.estimator): instantiation of an sklearn estimator,
            defaults to RandomForestClassifier with default params

        TODO hyperparameter tuning
        """
        self.n_folds = n_folds
        if estimator is not None:
            self.estimator = estimator
        else:
            self.estimator = RandomForestRegressor(n_jobs=8)

    def _sample_split():
        pass

    def estimate(self, T, X, Z, Y, **kwargs):
        """
        Fits a predicted Z using the given ML estimator, then returns TSLS results.

        TODO parallelize 
        """
        k_fold = KFold(n_splits=self.n_folds)

        pred_Z = np.zeros_like(T)
        for train_idx, test_idx in k_fold.split(T):
            train_data = Z.values[train_idx]
            train_outcome = T.values[train_idx]

            test_data = Z.values[test_idx]
            # TODO some R^2 diagnostics?
            #test_outcome = T[test_idx]
            self.estimator.fit(train_data, train_outcome)
            fold_preds = self.estimator.predict(test_data)

            pred_Z[test_idx] = fold_preds

        # then run TSLS on the predicted instrument
        tsls = TSLS()
        return tsls.estimate(T=T, X=X, Z=pred_Z, Y=Y)
