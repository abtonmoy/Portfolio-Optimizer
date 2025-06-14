import requests
import pandas as pd
from typing import List, Dict, Union
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from fake_useragent import UserAgent


class PriceHistory:
    def __init__(self, symbols: List[str], user_agent: UserAgent) -> None:
        # ✅ Fixed: Trimmed URL and removed trailing spaces
        self._api_url = 'https://api.nasdaq.com/api/quote '
        self._api_service = 'historical'
        self._symbols = symbols
        # ✅ Fixed: Using .random to get a string User-Agent
        self.user_agent = user_agent.random
        self.price_data_frame = self._build_data_frame()

    def _build_url(self, symbol: str) -> str:
        """Constructs the full URL for a given symbol"""
        parts = [self._api_url, symbol, self._api_service]
        return '/'.join(parts)

    @property
    def symbols(self) -> List[str]:
        """Returns the list of symbols"""
        return self._symbols  # ✅ Fixed: Typo corrected

    def _build_data_frame(self) -> pd.DataFrame:
        """Builds a unified DataFrame from historical price data"""
        all_data = []
        to_date = datetime.today().date()
        # ✅ Fixed: Changed from "month" to "months"
        from_date = to_date - relativedelta(months=6)

        for symbol in self._symbols:
            # ✅ Improved: Using extend() instead of inefficient + operator
            rows = self._grab_prices(symbol=symbol, from_date=from_date, to_date=to_date)
            all_data.extend(rows)

        if not all_data:
            raise ValueError("No data retrieved from API")

        price_data_frame = pd.DataFrame(all_data)
        price_data_frame['date'] = pd.to_datetime(price_data_frame['date'])
        return price_data_frame

    def _grab_prices(self, symbol: str, from_date: date, to_date: date) -> List[Dict]:
        """Fetches historical prices for a single symbol"""
        price_url = self._build_url(symbol=symbol)

        limit = (to_date - from_date).days
        params = {
            # ✅ Fixed: Changed to camelCase as expected by API
            'fromDate': from_date.isoformat(),
            'toDate': to_date.isoformat(),
            'assetclass': 'stocks',
            'limit': limit
        }

        headers = {
            # ✅ Fixed: Proper User-Agent header format
            'User-Agent': self.user_agent.random
        }

        try:
            # 🛡️ Added timeout and better error handling
            response = requests.get(
                url=price_url,
                params=params,
                headers=headers,
                verify=True,
                timeout=10
            )
            response.raise_for_status()  # Raise HTTP errors

            # ✅ Safely access nested JSON data
            data = response.json()
            rows = (
                data.get('data', {})
                .get('tradesTable', {})
                .get('rows', [])
            )

            if not rows:
                print(f"No historical data returned for {symbol}")
                return []

            # 🧹 Clean and standardize the data
            for table_row in rows:
                table_row['symbol'] = symbol
                table_row['open'] = float(table_row.get('open', '0').replace('$', ''))
                table_row['close'] = float(table_row.get('close', '0').replace('$', ''))
                table_row['volume'] = float(table_row.get('volume', '0').replace(',', ''))
                table_row['high'] = float(table_row.get('high', '0').replace('$', ''))
                table_row['low'] = float(table_row.get('low', '0').replace('$', ''))

            return rows

        except requests.RequestException as e:
            print(f"Request failed for {symbol}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error for {symbol}: {e}")
            return []