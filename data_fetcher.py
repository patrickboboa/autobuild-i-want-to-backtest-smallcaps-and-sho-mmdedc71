import os
import time
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from polygon import RESTClient
from polygon.rest.models import Agg


class DataFetcher:
    """Handles Polygon API requests with caching and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "./cache"):
        """
        Initialize DataFetcher with Polygon API key.
        
        Args:
            api_key: Polygon API key. If None, reads from environment.
            cache_dir: Directory for caching downloaded data.
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key not provided. Set POLYGON_API_KEY environment variable.")
        
        self.client = RESTClient(self.api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Rate limiting: Polygon free tier allows ~5 calls/min, paid tiers allow more
        self.request_delay = 0.25  # 250ms between requests (conservative for paid tier)
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, 
                       timespan: str = "day") -> str:
        """Generate cache key for data request."""
        key_string = f"{symbol}_{start_date}_{end_date}_{timespan}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_key}: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key}: {e}")
    
    def fetch_daily_data(self, symbol: str, start_date: str, end_date: str,
                        use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume, vwap
            or None if data unavailable
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date, "day")
        
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Fetch from API
        try:
            self._rate_limit()
            
            aggs = []
            for agg in self.client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=50000
            ):
                aggs.append(agg)
            
            if not aggs:
                return None
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'date': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York').tz_localize(None),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': agg.vwap if hasattr(agg, 'vwap') else None,
                    'transactions': agg.transactions if hasattr(agg, 'transactions') else None
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_intraday_data(self, symbol: str, date: str, 
                           timespan: str = "minute", multiplier: int = 1,
                           use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch intraday data for a specific date.
        
        Args:
            symbol: Stock ticker symbol
            date: Date in YYYY-MM-DD format
            timespan: 'minute', 'hour', etc.
            multiplier: Aggregation multiplier (e.g., 5 for 5-minute bars)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with intraday OHLCV data or None if unavailable
        """
        cache_key = self._get_cache_key(symbol, date, date, f"{multiplier}{timespan}")
        
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            self._rate_limit()
            
            # Fetch for the specific date
            next_day = (pd.Timestamp(date) + timedelta(days=1)).strftime('%Y-%m-%d')
            
            aggs = []
            for agg in self.client.list_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=date,
                to=next_day,
                limit=50000
            ):
                aggs.append(agg)
            
            if not aggs:
                return None
            
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': pd.Timestamp(agg.timestamp, unit='ms', tz='America/New_York'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': agg.vwap if hasattr(agg, 'vwap') else None,
                    'transactions': agg.transactions if hasattr(agg, 'transactions') else None
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            if use_cache:
                self._save_to_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching intraday data for {symbol} on {date}: {e}")
            return None
    
    def get_market_open_price(self, symbol: str, date: str,
                              use_cache: bool = True) -> Optional[float]:
        """
        Get the market open price for a symbol on a specific date.
        Uses the first regular trading hour bar (9:30 AM ET).
        
        Args:
            symbol: Stock ticker symbol
            date: Date in YYYY-MM-DD format
            use_cache: Whether to use cached data
            
        Returns:
            Open price at market open or None if unavailable
        """
        # First try to get from daily data
        daily_data = self.fetch_daily_data(symbol, date, date, use_cache)
        if daily_data is not None and len(daily_data) > 0:
            return daily_data.iloc[0]['open']
        
        # If daily data unavailable, try intraday
        intraday_data = self.fetch_intraday_data(symbol, date, "minute", 1, use_cache)
        if intraday_data is not None and len(intraday_data) > 0:
            # Filter for market open (9:30 AM ET)
            intraday_data['time'] = intraday_data['timestamp'].dt.time
            market_open_time = pd.Timestamp('09:30:00').time()
            
            # Get first bar at or after 9:30
            open_bars = intraday_data[intraday_data['time'] >= market_open_time]
            if len(open_bars) > 0:
                return open_bars.iloc[0]['open']
        
        return None
    
    def fetch_ticker_details(self, symbol: str) -> Optional[Dict]:
        """
        Fetch ticker details including market cap, sector, etc.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with ticker details or None
        """
        try:
            self._rate_limit()
            details = self.client.get_ticker_details(symbol)
            
            return {
                'symbol': symbol,
                'name': details.name if hasattr(details, 'name') else None,
                'market_cap': details.market_cap if hasattr(details, 'market_cap') else None,
                'sector': details.sic_description if hasattr(details, 'sic_description') else None,
                'exchange': details.primary_exchange if hasattr(details, 'primary_exchange') else None,
                'currency': details.currency_name if hasattr(details, 'currency_name') else 'USD',
                'outstanding_shares': details.share_class_shares_outstanding if hasattr(details, 'share_class_shares_outstanding') else None,
                'description': details.description if hasattr(details, 'description') else None
            }
        except Exception as e:
            print(f"Error fetching ticker details for {symbol}: {e}")
            return None
    
    def get_smallcap_universe(self, date: str, max_market_cap: float = 2e9,
                              min_price: float = 1.0, min_volume: float = 100000) -> List[str]:
        """
        Get list of smallcap tickers based on market cap threshold.
        Note: This is a simplified version. In production, you'd want to use
        a more comprehensive screening approach or historical constituent data.
        
        Args:
            date: Date for screening (YYYY-MM-DD)
            max_market_cap: Maximum market cap for smallcap classification
            min_price: Minimum stock price
            min_volume: Minimum average daily volume
            
        Returns:
            List of ticker symbols
        """
        # This is a placeholder - Polygon doesn't have a direct screener API
        # In practice, you would:
        # 1. Maintain a list of potential smallcaps
        # 2. Use a separate data source for historical market cap
        # 3. Use Polygon to validate they were trading at the time
        
        print("Warning: get_smallcap_universe is a simplified implementation.")
        print("For production use, maintain historical constituent lists to avoid survivor bias.")
        return []
    
    def fetch_stock_splits(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock splits for adjustment purposes.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with split information
        """
        try:
            self._rate_limit()
            
            splits = []
            for split in self.client.list_splits(
                ticker=symbol,
                execution_date_gte=start_date,
                execution_date_lte=end_date
            ):
                splits.append({
                    'date': split.execution_date,
                    'split_from': split.split_from if hasattr(split, 'split_from') else None,
                    'split_to': split.split_to if hasattr(split, 'split_to') else None,
                    'ratio': split.split_to / split.split_from if hasattr(split, 'split_from') and hasattr(split, 'split_to') else None
                })
            
            return pd.DataFrame(splits)
            
        except Exception as e:
            print(f"Error fetching splits for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_dividends(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch dividend information.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with dividend information
        """
        try:
            self._rate_limit()
            
            dividends = []
            for div in self.client.list_dividends(
                ticker=symbol,
                ex_dividend_date_gte=start_date,
                ex_dividend_date_lte=end_date
            ):
                dividends.append({
                    'ex_date': div.ex_dividend_date,
                    'pay_date': div.pay_date if hasattr(div, 'pay_date') else None,
                    'amount': div.cash_amount if hasattr(div, 'cash_amount') else None,
                    'declaration_date': div.declaration_date if hasattr(div, 'declaration_date') else None
                })
            
            return pd.DataFrame(dividends)
            
        except Exception as e:
            print(f"Error fetching dividends for {symbol}: {e}")
            return pd.DataFrame()
    
    def check_ticker_active(self, symbol: str, date: str) -> bool:
        """
        Check if a ticker was actively trading on a specific date.
        Helps avoid survivor bias.
        
        Args:
            symbol: Stock ticker symbol
            date: Date to check (YYYY-MM-DD)
            
        Returns:
            True if ticker was trading, False otherwise
        """
        data = self.fetch_daily_data(symbol, date, date, use_cache=True)
        return data is not None and len(data) > 0
    
    def batch_fetch_daily_data(self, symbols: List[str], start_date: str, 
                               end_date: str, use_cache: bool = True,
                               chunk_size: int = 10) -> Dict[