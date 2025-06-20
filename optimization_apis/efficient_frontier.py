import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class EfficientFrontier:
    def __init__(self, expected_returns, cov_matrix):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.frontier_df = None
    
    def calculate_frontier(self, num_points=50, risk_free_rate=0.03):
        n_assets = len(self.expected_returns)
        target_returns = np.linspace(
            self.expected_returns.min(), 
            self.expected_returns.max(), 
            num_points
        )
        frontier_data = []
        
        for target_return in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.expected_returns) - target_return}
            ]
            bounds = [(0, 1) for _ in range(n_assets)]
            init_guess = n_assets * [1. / n_assets]
            
            opt_result = minimize(
                lambda w: np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w))),
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if opt_result.success:
                weights = opt_result.x
                volatility = opt_result.fun
                sharpe = (target_return - risk_free_rate) / volatility
                
                frontier_data.append({
                    'return': target_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'weights': weights
                })
        
        self.frontier_df = pd.DataFrame(frontier_data)
        return self.frontier_df
    
    def plot_frontier(self):
        if self.frontier_df is None:
            raise RuntimeError("Run calculate_frontier() first")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.frontier_df['volatility'], self.frontier_df['return'], 'b-', linewidth=2)
        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Return (Annualized)')
        ax.grid(True)
        return fig
    
    def summary(self):
        if self.frontier_df is None:
            raise RuntimeError("Run calculate_frontier() first")
            
        max_sharpe = self.frontier_df.loc[self.frontier_df['sharpe'].idxmax()]
        min_vol = self.frontier_df.loc[self.frontier_df['volatility'].idxmin()]
        
        return {
            "max_sharpe_return": max_sharpe['return'],
            "max_sharpe_volatility": max_sharpe['volatility'],
            "min_vol_return": min_vol['return'],
            "min_vol_volatility": min_vol['volatility'],
            "frontier_points": len(self.frontier_df)
        }