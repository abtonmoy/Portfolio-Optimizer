import pandas as pd
import numpy as np
import logging
from typing import Tuple

class DataPreparer:
    """
    Prepares data for portfolio simulations from a CSV file
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the DataPreparer with path to CSV file
        
        Args:
            csv_path (str): Path to CSV file containing price data
        """
        self.csv_path = csv_path
        self.pivot_df = None
        self.returns = None
        self.log_returns = None
        self.expected_returns = None
        self.cov_matrix = None
        self.corr_matrix = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_and_prepare(self) -> None:
        """
        Load data from CSV and prepare all necessary data structures
        """
        try:
            # Load data from CSV
            self.logger.info(f"Loading data from {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            # Validate required columns
            required_columns = ['date', 'symbol', 'close']
            if not all(col in df.columns for col in required_columns):
                missing = set(required_columns) - set(df.columns)
                raise ValueError(f"CSV missing required columns: {missing}")
            
            # Pivot to create symbol columns
            self.pivot_df = df.pivot(index='date', columns='symbol', values='close')
            
            # Calculate returns
            self.returns = self.pivot_df.pct_change().dropna()
            
            # Calculate log returns
            self.log_returns = np.log(1 + self.returns)
            
            # Calculate annualized metrics
            self.expected_returns = self.returns.mean() * 252
            self.cov_matrix = self.returns.cov() * 252
            self.corr_matrix = self.returns.corr()
            
            self.logger.info("Data preparation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def get_simulation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Get data needed for simulations
        
        Returns:
            Tuple: (pivot_df, log_returns, expected_returns, cov_matrix)
        """
        if self.pivot_df is None:
            raise RuntimeError("Data not prepared. Call load_and_prepare() first.")
            
        return (
            self.pivot_df,
            self.log_returns,
            self.expected_returns,
            self.cov_matrix
        )
    
    def get_full_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                    pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Get all prepared data
        
        Returns:
            Tuple: (pivot_df, returns, log_returns, expected_returns, cov_matrix, corr_matrix)
        """
        if self.pivot_df is None:
            raise RuntimeError("Data not prepared. Call load_and_prepare() first.")
            
        return (
            self.pivot_df,
            self.returns,
            self.log_returns,
            self.expected_returns,
            self.cov_matrix,
            self.corr_matrix
        )

# Example usage
if __name__ == "__main__":
    # Initialize with path to your CSV file
    preparer = DataPreparer("path/to/your/data.csv")
    
    # Load and prepare data
    preparer.load_and_prepare()
    
    # Get data for simulations
    pivot_df, log_returns, expected_returns, cov_matrix = preparer.get_simulation_data()
    
    print("Pivot DataFrame:")
    print(pivot_df.head())
    print("\nLog Returns:")
    print(log_returns.head())