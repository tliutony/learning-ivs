"""
Suite of benchmark estimators.

Mainly wrappers around existing packages for IV estimation.
"""
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