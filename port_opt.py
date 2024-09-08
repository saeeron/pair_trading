
import numpy as np
import scipy.optimize as sco



class MVO:

    def __init__(self, approach, **kwargs):
        self._approach = approach

        self._expected_returns = kwargs.get("expected_returns", None)
        self._cov_matrix = kwargs.get("cov_matrix", None)
        self._risk_free_rate =  kwargs.get("_risk_free_rate", None)
        if self._risk_free_rate is None:
            self._risk_free_rate = 0


    def optim(self, bounds: tuple = (0, 1), sum_constraint_abs: bool = False, regularization_coeff = 1e-7, method = "SLSQP"):
        
        obj_fun = self._return_obj_func()
        const_fun = self._return_constraint_fun(sum_constraint_abs)
        boundaries = bounds = tuple(bounds for _ in range(len(self._expected_returns)))
        n = len(self._expected_returns)
        initial_guess = np.array([1/n] * n)

        self._cov_matrix += np.eye(self._cov_matrix.shape[0]) * regularization_coeff

        result = sco.minimize(obj_fun, initial_guess, 
                      args=(self._expected_returns, self._cov_matrix, self._risk_free_rate), 
                      method=method, bounds=boundaries, constraints=const_fun)

        assert result.success, f"cannot converge!"
        return result.x

    def _return_obj_func(self):
        
        def max_sharpe_obj_fun(weights, expected_returns, cov_matrix, risk_free_rate):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return - sharpe_ratio

        allobj = {"MaxSharpe": max_sharpe_obj_fun}
        
        return allobj.get("MaxSharpe")


    def _return_constraint_fun(self, sum_constraint_abs):

        def abs_sum_constraint(weights):
                return np.sum(np.abs(weights)) - 1
        
        def sum_constraint(weights):
                return np.sum(weights) - 1

        if sum_constraint_abs:
            return ({'type': 'eq', 'fun': abs_sum_constraint})
        else:
            return ({'type': 'eq', 'fun': sum_constraint})


