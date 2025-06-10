'''
We need an api that collects historic data from nasdaq/Alpha Vantage
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
import logging

class PriceHistory:


    def __init__(self, api_key: str, symbols:Union[List[str], Dict] = None , rate_limit:float=0.2):
        """
        Initialize the PriceHistory class
        
        Args:
            api_key (str): Alpha Vantage API key
            rate_limit (float): Delay between API calls in seconds (default 0.2 for 5 calls/min free tier)
            symbols (Union[List[str], Dict], optional): Symbols to work with. Can be a list or dictionary
        """

        self.api_key = api_key
        self.rate_limit =rate_limit
        self.base_url = 'https://www.alphavantage.co/query'
        self.df = pd.DataFrame()
        self._symbols = symbols if symbols is not None else []

        #setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        
    def symbols(self, category:  Optional[str]=None, subcategory: Optional[str]=None)-> Union[Dict, List]:
        """
        Returns the symbols passed during initialization
        
        Args:
            category (str, optional): Category key if symbols is a dictionary
            subcategory (str, optional): Subcategory key if symbols is a nested dictionary
        
        Returns:
            Union[Dict, List]: The symbols or subset based on category/subcategory
        """
        if self._symbols is None or (isinstance(self._symbols, list) and len(self._symbols)==0):
            self.logger.warning("No symbols have been set. Please initialize with symbols or use set_symbols()")
            return []
        
        # symbols is a list
        if isinstance(self._symbols, list):
            return self._symbols
        
        # symbols is dictionary
        if isinstance(self._symbols, dict):
            if category is None:
                return self._symbols
            
            if category not in self._symbols:
                self.logger.warning(f"Category '{category}' not found. Available categories: {list(self._symbols.keys())}")
                return []
            
            if isinstance(self._symbols[category], dict) and subcategory:
                if subcategory in self._symbols[category]:
                    return self._symbols[category][subcategory]
                else:
                    self.logger.warning(f"Subcategory '{subcategory}' not found in {category}. Available: {list(self._symbols[category].keys())}")
                    return []
                
            return self._symbols[category]
        
        return self._symbols
    


    def set_symbols(self, symbols:Union[List[str], Dict])->None:
        """
        Set or update the symbols to work with
        
        Args:
            symbols (Union[List[str], Dict]): Symbols to work with
        """
        self._symbols = symbols
        self.logger.info(f"Updated symbols. Type: {type(symbols)}, Count: {len(symbols) if isinstance(symbols, (list, dict)) else 'N/A'}")


    def build_url(self, symbol:str, function: str ='TIME_SERIES_DAILY', outputsize:str = 'full', datatype: str ='json')-> str:
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

        url =  f'{self.base_url}?'+'&'.join([f'{k}={v}'for k,v in params.items()])
        return url
    

    def fetch_prices(self, symbols:Union[str, List[str]], function:str ='TIME_sERIES_DAILY', outputsize:str='full')->pd.DataFrame:
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
                self.logger.info(f'Fetching data for {symbol}...')

                #build url and fetch data
                url = self.build_url(symbol, function, outputsize)
                response = requests.get(url, timeout=30)

                data =response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    self.logger.error(f'API Error for {symbol}: {data['Error Message']}')
                    continue

                if 'Note' in data:
                    self.logger.warning(f"API Note for {symbol}: {data['Note']}")
                    continue

                # Extract time series data based on function type
                time_series_key = self._get_time_series_key(data, function)
                if time_series_key not in data:
                    self.logger.error(f"No time series data found for {symbol}")
                    continue

                time_series = data[time_series_key]

                #convert to df and clean

                df_symbol = pd.DataFrame(time_series).T
                df_symbol = self._clean_price_data(df_symbol, symbol)

                if not df_symbol.empty:
                    all_data.append(df_symbol)
                    self.logger.info(f"Successfully fetched {len(df_symbol)} records for {symbol}")

                #rate limiting
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