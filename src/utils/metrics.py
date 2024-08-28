import numpy as np

__all__ = ['mse_decomp']

def mse_decomp(result_df, ground_truth=1):
    """
    Computes MSE, along with bias and variance breakdown.

    Args:
        result_df: pd.DataFrame of results
        ground_truth: if not a scalar, needs to be same shape as preds
    """
    preds = result_df['tau'].values

    # note: bias-var is only meaningful if ground truth is a scalar
    assert np.isscalar(ground_truth) or preds.shape == ground_truth.shape

    mse = np.mean((preds - ground_truth)**2)
    
    bias = np.mean(preds) - ground_truth
    var = np.var(preds)

    if np.isscalar(ground_truth):
        assert np.isclose(bias**2 + var, mse)
    else:
        bias = np.nan
        var = np.nan

    return {
        'mse': mse, 
        'bias^2': bias**2, 
        'var': var
    }
