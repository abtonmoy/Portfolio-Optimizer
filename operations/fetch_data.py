'''
An api that collects historic data from nasdaq/Alpha Vantage
Structure: 
class PriceHistory -> holds all the following functions
    func build_URL -> builds the url to collect the data
    func build_df -> builds an unified dataframe of historic data which will be used for analysis and predictions
    func symbols -> holds and returns the symbols for specific stocks and bonds etc
    func fetch_prices -> fetch the prices for the returned symbols from the url, clean it, and dump it in the dataframe
'''

import pandas as pd
import requests
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

class PriceHistory:
    """
    A comprehensive API class for collecting historic data from Alpha Vantage
    """
    
    def __init__(self, api_key: str, symbols: Union[List[str], Dict] = None, rate_limit: float = 0.2):
        """
        Initialize the PriceHistory class
        
        Args:
            api_key (str): Alpha Vantage API key
            symbols (Union[List[str], Dict], optional): Symbols to work with. Can be a list or dictionary
            rate_limit (float): Delay between API calls in seconds (default 0.2 for 5 calls/min free tier)
        """
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.base_url = "https://www.alphavantage.co/query"
        self.df = pd.DataFrame()
        self._symbols = symbols if symbols is not None else []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def symbols(self, category: Optional[str] = None, subcategory: Optional[str] = None) -> Union[Dict, List]:
        """
        Returns the symbols passed during initialization
        
        Args:
            category (str, optional): Category key if symbols is a dictionary
            subcategory (str, optional): Subcategory key if symbols is a nested dictionary
        
        Returns:
            Union[Dict, List]: The symbols or subset based on category/subcategory
        """
        if self._symbols is None or (isinstance(self._symbols, list) and len(self._symbols) == 0):
            self.logger.warning("No symbols have been set. Please initialize with symbols or use set_symbols()")
            return []
        
        # If symbols is a simple list
        if isinstance(self._symbols, list):
            return self._symbols
        
        # If symbols is a dictionary
        if isinstance(self._symbols, dict):
            if category is None:
                return self._symbols
            
            if category not in self._symbols:
                self.logger.warning(f"Category '{category}' not found. Available categories: {list(self._symbols.keys())}")
                return []
            
            # Handle nested dictionaries
            if isinstance(self._symbols[category], dict) and subcategory:
                if subcategory in self._symbols[category]:
                    return self._symbols[category][subcategory]
                else:
                    self.logger.warning(f"Subcategory '{subcategory}' not found in {category}. Available: {list(self._symbols[category].keys())}")
                    return []
            
            return self._symbols[category]
        
        return self._symbols
    
    def set_symbols(self, symbols: Union[List[str], Dict]) -> None:
        """
        Set or update the symbols to work with
        
        Args:
            symbols (Union[List[str], Dict]): Symbols to work with
        """
        self._symbols = symbols
        self.logger.info(f"Updated symbols. Type: {type(symbols)}, Count: {len(symbols) if isinstance(symbols, (list, dict)) else 'N/A'}")
    
    def build_URL(self, symbol: str, function: str = "TIME_SERIES_DAILY", 
                  outputsize: str = "full", datatype: str = "json") -> str:
        """
        Builds the URL to collect data from Alpha Vantage API
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            function (str): API function to call
            outputsize (str): 'compact' (last 100 days) or 'full' (all available)
            datatype (str): 'json' or 'csv'
        
        Returns:
            str: Complete API URL
        """
        params = {
            'function': function,
            'symbol': symbol,
            'outputsize': outputsize,
            'datatype': datatype,
            'apikey': self.api_key
        }
        
        url = f"{self.base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
        return url
    
    def fetch_prices(self, symbols: Union[str, List[str]], 
                    function: str = "TIME_SERIES_DAILY",
                    outputsize: str = "full") -> pd.DataFrame:
        """
        Fetch prices for the given symbols, clean data, and add to dataframe
        
        Args:
            symbols (Union[str, List[str]]): Single symbol or list of symbols
            function (str): Alpha Vantage function to use
            outputsize (str): Data size to retrieve
        
        Returns:
            pd.DataFrame: Cleaned price data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        all_data = []
        
        for symbol in symbols:
            try:
                self.logger.info(f"Fetching data for {symbol}...")
                
                # Build URL and fetch data
                url = self.build_URL(symbol, function, outputsize)
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors
                if "Error Message" in data:
                    error_msg = data["Error Message"]
                    self.logger.error(f"API Error for {symbol}: {error_msg}")
                    continue
                
                if "Note" in data:
                    note_msg = data["Note"]
                    self.logger.warning(f"API Note for {symbol}: {note_msg}")
                    continue
                
                # Extract time series data based on function type
                time_series_key = self._get_time_series_key(data, function)
                if time_series_key not in data:
                    self.logger.error(f"No time series data found for {symbol}")
                    continue
                
                time_series = data[time_series_key]
                
                # Convert to DataFrame and clean
                df_symbol = pd.DataFrame(time_series).T
                df_symbol = self._clean_price_data(df_symbol, symbol)
                
                if not df_symbol.empty:
                    all_data.append(df_symbol)
                    self.logger.info(f"Successfully fetched {len(df_symbol)} records for {symbol}")
                
                # Rate limiting
                time.sleep(self.rate_limit)
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error fetching {symbol}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        else:
            self.logger.warning("No data was successfully fetched")
            return pd.DataFrame()
    
    def _get_time_series_key(self, data: Dict, function: str) -> str:
        """Helper method to get the correct time series key from API response"""
        key_mapping = {
            'TIME_SERIES_DAILY': 'Time Series (Daily)',
            'TIME_SERIES_WEEKLY': 'Weekly Time Series',
            'TIME_SERIES_MONTHLY': 'Monthly Time Series',
            'TIME_SERIES_INTRADAY': 'Time Series (5min)'  # Default for intraday
        }
        
        # For intraday, the key varies by interval
        if function == 'TIME_SERIES_INTRADAY':
            for key in data.keys():
                if 'Time Series' in key and 'min' in key:
                    return key
        
        return key_mapping.get(function, 'Time Series (Daily)')
    
    def _clean_price_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and standardize price data
        
        Args:
            df (pd.DataFrame): Raw price data
            symbol (str): Stock symbol
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        if df.empty:
            return df
        
        # Standardize column names
        column_mapping = {
            '1. open': 'open',
            '2. high': 'high', 
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume',
            '1. Open': 'open',
            '2. High': 'high',
            '3. Low': 'low', 
            '4. Close': 'close',
            '5. Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert price columns to numeric
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert volume to numeric
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Convert index to datetime and sort
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Reset index to make date a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        
        # Remove any rows with all NaN values
        df = df.dropna(how='all')
        
        # Calculate additional metrics
        if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            df['daily_return'] = df['close'].pct_change()
            df['price_range'] = df['high'] - df['low']
            df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        return df
    
    def build_df(self, symbols: Union[str, List[str]] = None, 
                 update_existing: bool = True) -> pd.DataFrame:
        """
        Build a unified dataframe of historic data for analysis and predictions
        
        Args:
            symbols (Union[str, List[str]], optional): Symbols to fetch. If None, uses class symbols
            update_existing (bool): Whether to update existing dataframe or create new
        
        Returns:
            pd.DataFrame: Unified dataframe with all historic data
        """
        if symbols is None:
            # Use class symbols if no specific symbols provided
            if isinstance(self._symbols, list):
                symbols = self._symbols
            elif isinstance(self._symbols, dict):
                # Flatten dictionary to get all symbols
                symbols = []
                for value in self._symbols.values():
                    if isinstance(value, list):
                        symbols.extend(value)
                    elif isinstance(value, dict):
                        for subvalue in value.values():
                            if isinstance(subvalue, list):
                                symbols.extend(subvalue)
            else:
                self.logger.error("No symbols available. Please provide symbols or set them during initialization.")
                return pd.DataFrame()
        
        # Fetch new data
        new_data = self.fetch_prices(symbols)
        
        if update_existing and not self.df.empty:
            # Combine with existing data
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            # Remove duplicates based on date and symbol
            self.df = self.df.drop_duplicates(subset=['date', 'symbol'], keep='last')
        else:
            self.df = new_data
        
        # Sort by symbol and date
        if not self.df.empty:
            self.df = self.df.sort_values(['symbol', 'date']).reset_index(drop=True)
            self.logger.info(f"Built dataframe with {len(self.df)} total records for {self.df['symbol'].nunique()} symbols")
        
        return self.df
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the collected data"""
        if self.df.empty:
            return {"message": "No data available"}
        
        summary = {
            "total_records": len(self.df),
            "symbols_count": self.df['symbol'].nunique(),
            "symbols": list(self.df['symbol'].unique()),
            "date_range": {
                "start": self.df['date'].min().strftime('%Y-%m-%d'),
                "end": self.df['date'].max().strftime('%Y-%m-%d')
            },
            "avg_daily_return": self.df['daily_return'].mean() if 'daily_return' in self.df.columns else None,
            "avg_volume": self.df['volume'].mean() if 'volume' in self.df.columns else None
        }
        
        return summary

# Example usage:
if __name__ == "__main__":
    # Initialize with your Alpha Vantage API key and symbols
    api_key = 'JQFVMO4N0BHJ66FK'  # Replace with your actual API key
    
    # Example 1: Simple list of symbols
    my_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    price_history = PriceHistory(api_key, symbols=my_symbols)
    
    # Example 2: Organized dictionary of symbols
    organized_symbols = {
        'tech': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        'finance': ['JPM', 'BAC', 'WFC'],
        'etfs': ['SPY', 'QQQ', 'VTI']
    }
    # price_history = PriceHistory(api_key, symbols=organized_symbols)
    
    # Example 3: Nested dictionary
    nested_symbols = {
        'stocks': {
            'tech': ['AAPL', 'GOOGL', 'MSFT'],
            'finance': ['JPM', 'BAC']
        },
        'etfs': ['SPY', 'QQQ']
    }
    # price_history = PriceHistory(api_key, symbols=nested_symbols)
    
    # Get symbols
    all_symbols = price_history.symbols()
    print("All symbols:", all_symbols)
    
    # Build URL example
    url = price_history.build_URL('AAPL')
    print("URL:", url)
    
    # Fetch prices for specific symbols
    df = price_history.fetch_prices(['AAPL', 'GOOGL'])
    
    # Build unified dataframe (will use class symbols if none provided)
    unified_df = price_history.build_df()
    
    # Or build with specific symbols
    unified_df = price_history.build_df(['AAPL', 'MSFT'])
    
    # Update symbols after initialization
    price_history.set_symbols(['AAPL', 'TSLA', 'NVDA'])
    
    # Get summary
    summary = price_history.get_data_summary()
    print("Data Summary:", summary)