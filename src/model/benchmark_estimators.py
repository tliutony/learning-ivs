"""
Suite of benchmark estimators.

Mainly wrappers around existing packages for IV estimation.
"""

from base_estimator import BaseEstimator

from linearmodels.iv import IV2SLS, IVLIML, _OLS

class TSLS(BaseEstimator):
    def estimate(self, T, X, Z, Y, **kwargs):
        
        self.model = IV2SLS(dependent=Y, 
                            exog=X, 
                            endog=T, 
                            instruments=Z,
                            **kwargs)
        self.model.fit()
        return {'tau': self.model.params['T'], 'se': self.model.std_errors['T']}