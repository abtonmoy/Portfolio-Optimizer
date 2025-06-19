import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional
import logging

class MonteCarloSimulation:

    def __init__(self, log_returns: pd.DataFrame, expected_returns: pd.Series, cov_matrix: pd.DataFrame):
        self.log_returns = log_returns
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.simulation_df = None
        self.max_sharpe = None
        self.min_vol = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def monte_carlo_simulation(self, num_portfolios: int = 1000, risk_free_rate: float = 0.03) -> None:
        num_assets = len(self.expected_returns)
        weights_arr = np.zeros((num_portfolios, num_assets))
        return_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)

        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_arr[i, :] = weights

            exp_return = np.sum(self.expected_returns * weights)
            exp_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = (exp_return - risk_free_rate) / exp_vol if exp_vol != 0 else 0

            return_arr[i] = exp_return
            vol_arr[i] = exp_vol
            sharpe_arr[i] = sharpe

        self.simulation_df = pd.DataFrame({
            'Return': return_arr,
            'Volatility': vol_arr,
            'Sharpe Ratio': sharpe_arr,
            'Portfolio Weights': list(weights_arr)
        })

        self.max_sharpe = self.simulation_df.loc[self.simulation_df['Sharpe Ratio'].idxmax()]
        self.min_vol = self.simulation_df.loc[self.simulation_df['Volatility'].idxmin()]
        self.logger.info(f'Monte Carlo simulation completed with {num_portfolios} portfolios')

    def get_optimal_portfolios(self) -> Tuple[pd.Series, pd.Series]:
        if self.max_sharpe is None or self.min_vol is None:
            raise RuntimeError("Run simulation first using monte_carlo_simulation()")
        return self.max_sharpe, self.min_vol

    def plot_simulation(self, figsize: tuple = (10, 8)) -> plt.Figure:
        if self.simulation_df is None:
            raise RuntimeError("Run simulation first using monte_carlo_simulation()")

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(
            self.simulation_df['Volatility'],
            self.simulation_df['Return'],
            c=self.simulation_df['Sharpe Ratio'],
            cmap='viridis',
            alpha=0.5
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')

        ax.scatter(
            self.max_sharpe['Volatility'],
            self.max_sharpe['Return'],
            marker='*',
            color='r',
            s=200,
            label='Max Sharpe Ratio'
        )
        ax.scatter(
            self.min_vol['Volatility'],
            self.min_vol['Return'],
            marker='*',
            color='g',
            s=200,
            label='Min Volatility'
        )

        ax.set_title('Monte Carlo Portfolio Simulation')
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Return (Annualized)')
        ax.legend()
        plt.tight_layout()

        return fig

    def calculate_efficient_frontier(self, risk_free_rate: float = 0.03, num_points: int = 50) -> pd.DataFrame:
        num_assets = len(self.expected_returns)
        target_returns = np.linspace(self.expected_returns.min(), self.expected_returns.max(), num_points)

        frontier_data = []

        for target_return in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.expected_returns) - target_return}
            ]
            bounds = [(0, 1) for _ in range(num_assets)]
            init_guess = num_assets * [1. / num_assets]

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
                sharpe = (target_return - risk_free_rate) / volatility if volatility > 0 else 0

                frontier_data.append({
                    'weights': weights,
                    'return': target_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe
                })

        self.logger.info(f"Efficient frontier calculated with {num_points} points")
        return pd.DataFrame(frontier_data)

    def plot_efficient_frontier(self, frontier_df: pd.DataFrame, figsize: tuple = (10, 6)) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frontier_df['volatility'], frontier_df['return'], 'b-', linewidth=2)

        if self.max_sharpe is not None:
            ax.scatter(
                self.max_sharpe['Volatility'],
                self.max_sharpe['Return'],
                marker='*',
                color='r',
                s=200,
                label='Max Sharpe Ratio'
            )
        if self.min_vol is not None:
            ax.scatter(
                self.min_vol['Volatility'],
                self.min_vol['Return'],
                marker='*',
                color='g',
                s=200,
                label='Min Volatility'
            )

        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Return (Annualized)')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        return fig

    def plot_combined_simulation_and_frontier(self, frontier_df: pd.DataFrame, figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot both Monte Carlo simulation and Efficient Frontier in one figure.
        """
        if self.simulation_df is None:
            raise RuntimeError("Run monte_carlo_simulation() first")

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(
            self.simulation_df['Volatility'],
            self.simulation_df['Return'],
            c=self.simulation_df['Sharpe Ratio'],
            cmap='viridis',
            alpha=0.4,
            label='Simulated Portfolios'
        )

        ax.plot(frontier_df['volatility'], frontier_df['return'], 'b--', linewidth=2, label='Efficient Frontier')

        ax.scatter(
            self.max_sharpe['Volatility'],
            self.max_sharpe['Return'],
            marker='*',
            color='red',
            s=200,
            label='Max Sharpe Ratio'
        )
        ax.scatter(
            self.min_vol['Volatility'],
            self.min_vol['Return'],
            marker='*',
            color='green',
            s=200,
            label='Min Volatility'
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')

        ax.set_title('Combined Monte Carlo Simulation & Efficient Frontier')
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Return (Annualized)')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        return fig

    def calculate_var(self, method: str = 'historical',
                      weights: Optional[np.ndarray] = None,
                      confidence_level: float = 0.95,
                      horizon: int = 1,
                      portfolio_value: float = 1e6,
                      simulations: int = 10000) -> float:

        if weights is None:
            weights = np.ones(len(self.expected_returns)) / len(self.expected_returns)

        returns = self.log_returns.dot(weights)

        if method == 'historical':
            return self._historical_var(returns, confidence_level, portfolio_value)
        elif method == 'monte_carlo':
            return self._monte_carlo_var(returns, confidence_level, horizon, portfolio_value, simulations)
        elif method == 'parametric':
            return self._parametric_var(returns, confidence_level, portfolio_value)
        else:
            raise ValueError("Invalid method. Choose 'historical', 'monte_carlo', or 'parametric'")

    def _historical_var(self, returns: pd.Series, confidence_level: float, portfolio_value: float) -> float:
        return -portfolio_value * np.percentile(returns, 100 * (1 - confidence_level))

    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float,
                         horizon: int, portfolio_value: float, simulations: int) -> float:
        mu = returns.mean()
        sigma = returns.std()
        simulated_returns = np.random.normal(mu, sigma, (horizon, simulations))
        portfolio_paths = portfolio_value * (1 + simulated_returns).cumprod(axis=0)
        final_values = portfolio_paths[-1, :]
        sorted_values = np.sort(final_values)
        index = int((1 - confidence_level) * simulations)
        return portfolio_value - sorted_values[index]

    def _parametric_var(self, returns: pd.Series, confidence_level: float, portfolio_value: float) -> float:
        mu = returns.mean()
        sigma = returns.std()
        z_score = norm.ppf(1 - confidence_level)
        return -portfolio_value * (mu + z_score * sigma)



# Example usage
if __name__ == "__main__":
    # First prepare data using DataPreparer
    # (Assuming you've already prepared data)
    
    # Initialize simulator
    # simulator = PortfolioSimulator(log_returns, expected_returns, cov_matrix)
    
    # Run Monte Carlo simulation
    # simulator.monte_carlo_simulation()
    
    # Get optimal portfolios
    # max_sharpe, min_vol = simulator.get_optimal_portfolios()
    
    # Plot results
    # fig = simulator.plot_simulation()
    # fig.savefig('monte_carlo_simulation.png')
    
    # Calculate efficient frontier
    # frontier_df = simulator.calculate_efficient_frontier()
    # frontier_fig = simulator.plot_efficient_frontier(frontier_df)
    # frontier_fig.savefig('efficient_frontier.png')
    
    # Calculate VaR
    # var = simulator.calculate_var('monte_carlo', weights=max_sharpe['Portfolio Weights'])
    # print(f"Value at Risk: ${var:,.2f}")
    pass        
