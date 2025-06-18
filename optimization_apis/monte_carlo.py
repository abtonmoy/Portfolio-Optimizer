import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Dict, List, Optional
import logging

class MonteCarloSimulation:

    def __init__(self, log_returns:pd.DataFrame, expected_returns:pd.Series, cov_matrix: pd.DataFrame):
        """
        Initialize the simulator with prepared data
        
        Args:
            log_returns (pd.DataFrame): Log returns of assets
            expected_returns (pd.Series): Annualized expected returns
            cov_matrix (pd.DataFrame): Annualized covariance matrix
        """
        self.log_returns = log_returns
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.simulation_df = None
        self.max_sharpe = None
        self.min_vol = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def monte_carlo_simulation(self, num_portfolios: int =1000, risk_free_rate: float=0.03) -> None:
        """
        Run Monte Carlo portfolio simulation
        
        Args:
            num_portfolios (int): Number of portfolios to simulate
            risk_free_rate (float): Risk-free rate for Sharpe ratio
        """
        num_assets = len(self.expected_returns)

        # init arrays
        weights_arr = np.zeros(num_portfolios, num_assets)
        return_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)

        for i in range(num_portfolios):
            # gen random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_arr[i, :] = weights

            #calc expected return
            exp_return = np.sum(self.expected_returns*weights)
            return_arr[i] = exp_return

            #calc exp vol
            exp_vol = np.sqrt(np.dot(weights.T, 
                                     np.dot(self.cov_matrix, weights)))
            vol_arr[i] = exp_vol

            # calc the sharpe ratio
            sharpe_arr[i] = (exp_return - risk_free_rate)/exp_vol if exp_vol != 0 else 0

        
        #create simulation results
        self.simulation_df = pd.DataFrame({
            'Return': return_arr,
            'Volatility': vol_arr,
            'Sharpe Ratio': sharpe_arr,
            'Portfolio Weights': list(weights_arr)
        })

        # find the optimal portfolio
        self.max_sharpe = self.simulation_df.loc[self.simulation_df['Sharpe Ratio'].idxmax()]
        self.min_vol = self.simulation_df.loc[self.simulation_df['Volatility'].idxmin()]
        self.logger.info(f'Monte Carlo simulation completed with {num_portfolios} portfolios')



    def get_optimal_portfolios(self) -> Tuple[pd.Series, pd.Series]:
        """
        Get optimal portfolios (max Sharpe ratio and min volatility)
        
        Returns:
            Tuple: (max_sharpe, min_vol) portfolio details
        """
        if self.max_sharpe is None or self.min_vol is None:
            raise RuntimeError("Run simulation first using monte_carlo_simulation()")
        
        return self.max_sharpe, self.min_vol
    

    
    def plot_simulation(self, figsize: tuple = (10, 8)) -> plt.Figure:
        """
        Plot Monte Carlo simulation results
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if self.simulation_df is None:
            raise RuntimeError("Run simulation first using monte_carlo_simulation()")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot of all portfolios
        scatter = ax.scatter(
            self.simulation_df['Volatility'],
            self.simulation_df['Returns'],
            c=self.simulation_df['Sharpe Ratio'],
            cmap='viridis',
            alpha=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        # Highlight optimal portfolios
        ax.scatter(
            self.max_sharpe['Volatility'],
            self.max_sharpe['Returns'],
            marker='*',
            color='r',
            s=200,
            label='Max Sharpe Ratio'
        )
        ax.scatter(
            self.min_vol['Volatility'],
            self.min_vol['Returns'],
            marker='*',
            color='g',
            s=200,
            label='Min Volatility'
        )
        
        # Set labels and title
        ax.set_title('Monte Carlo Portfolio Simulation')
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Return (Annualized)')
        ax.legend()
        plt.tight_layout()
        
        return fig    