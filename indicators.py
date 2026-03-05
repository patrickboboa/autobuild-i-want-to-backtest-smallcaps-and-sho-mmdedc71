import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for backtesting, avoiding look-ahead bias."""
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame, use_previous_day: bool = True) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price).
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume, timestamp
            use_previous_day: If True, uses previous day's VWAP for comparison at open
                            to avoid look-ahead bias
        
        Returns:
            Series with VWAP values
        """
        if df.empty:
            return pd.Series(dtype=float)
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative typical price * volume
        df['tp_volume'] = df['typical_price'] * df['volume']
        
        # Group by date if we have intraday data
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp'].dt.date
            
            # Calculate daily VWAP
            vwap_values = []
            for date, group in df.groupby('date'):
                cumulative_tp_volume = group['tp_volume'].cumsum()
                cumulative_volume = group['volume'].cumsum()
                
                # Avoid division by zero
                daily_vwap = np.where(
                    cumulative_volume > 0,
                    cumulative_tp_volume / cumulative_volume,
                    group['typical_price']
                )
                vwap_values.extend(daily_vwap)
            
            vwap_series = pd.Series(vwap_values, index=df.index)
        else:
            # Daily data - use rolling calculation
            df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()
            
            vwap_series = np.where(
                df['cumulative_volume'] > 0,
                df['cumulative_tp_volume'] / df['cumulative_volume'],
                df['typical_price']
            )
            vwap_series = pd.Series(vwap_series, index=df.index)
        
        # Shift forward by 1 to avoid look-ahead bias if requested
        if use_previous_day:
            vwap_series = vwap_series.shift(1)
        
        return vwap_series
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, use_previous: bool = True) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period (default 14)
            use_previous: If True, shifts RSI to avoid look-ahead bias
        
        Returns:
            Series with RSI values
        """
        if df.empty or 'close' not in df.columns:
            return pd.Series(dtype=float)
        
        close = df['close'].copy()
        
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate exponential moving average of gains and losses
        avg_gains = gains.ewm(span=period, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, min_periods=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Replace infinity and NaN values
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        
        # Shift to avoid look-ahead bias
        if use_previous:
            rsi = rsi.shift(1)
        
        return rsi
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = 'close', period: int = 20, 
                     use_previous: bool = True) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate SMA on
            period: SMA period
            use_previous: If True, shifts SMA to avoid look-ahead bias
        
        Returns:
            Series with SMA values
        """
        if df.empty or column not in df.columns:
            return pd.Series(dtype=float)
        
        sma = df[column].rolling(window=period, min_periods=period).mean()
        
        if use_previous:
            sma = sma.shift(1)
        
        return sma
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = 'close', period: int = 20,
                     use_previous: bool = True) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate EMA on
            period: EMA period
            use_previous: If True, shifts EMA to avoid look-ahead bias
        
        Returns:
            Series with EMA values
        """
        if df.empty or column not in df.columns:
            return pd.Series(dtype=float)
        
        ema = df[column].ewm(span=period, min_periods=period, adjust=False).mean()
        
        if use_previous:
            ema = ema.shift(1)
        
        return ema
    
    @staticmethod
    def calculate_volume_metrics(df: pd.DataFrame, period: int = 20,
                                use_previous: bool = True) -> Dict[str, pd.Series]:
        """
        Calculate volume-based metrics.
        
        Args:
            df: DataFrame with 'volume' column
            period: Period for average calculations
            use_previous: If True, shifts metrics to avoid look-ahead bias
        
        Returns:
            Dictionary with volume metrics: avg_volume, volume_ratio, volume_spike
        """
        if df.empty or 'volume' not in df.columns:
            return {
                'avg_volume': pd.Series(dtype=float),
                'volume_ratio': pd.Series(dtype=float),
                'volume_spike': pd.Series(dtype=bool)
            }
        
        # Calculate average volume
        avg_volume = df['volume'].rolling(window=period, min_periods=period).mean()
        
        # Calculate volume ratio (current volume / average volume)
        volume_ratio = df['volume'] / avg_volume
        volume_ratio = volume_ratio.replace([np.inf, -np.inf], np.nan)
        
        # Identify volume spikes (volume > 2x average)
        volume_spike = volume_ratio > 2.0
        
        if use_previous:
            avg_volume = avg_volume.shift(1)
            volume_ratio = volume_ratio.shift(1)
            volume_spike = volume_spike.shift(1)
        
        return {
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'volume_spike': volume_spike
        }
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, column: str = 'close',
                                 period: int = 20, num_std: float = 2.0,
                                 use_previous: bool = True) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate bands on
            period: Period for moving average
            num_std: Number of standard deviations
            use_previous: If True, shifts bands to avoid look-ahead bias
        
        Returns:
            Dictionary with upper_band, middle_band, lower_band
        """
        if df.empty or column not in df.columns:
            return {
                'upper_band': pd.Series(dtype=float),
                'middle_band': pd.Series(dtype=float),
                'lower_band': pd.Series(dtype=float)
            }
        
        # Calculate middle band (SMA)
        middle_band = df[column].rolling(window=period, min_periods=period).mean()
        
        # Calculate standard deviation
        std_dev = df[column].rolling(window=period, min_periods=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (num_std * std_dev)
        lower_band = middle_band - (num_std * std_dev)
        
        if use_previous:
            upper_band = upper_band.shift(1)
            middle_band = middle_band.shift(1)
            lower_band = lower_band.shift(1)
        
        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band
        }
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14,
                     use_previous: bool = True) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame with high, low, close columns
            period: ATR period
            use_previous: If True, shifts ATR to avoid look-ahead bias
        
        Returns:
            Series with ATR values
        """
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            return pd.Series(dtype=float)
        
        # Calculate True Range components
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR using exponential moving average
        atr = true_range.ewm(span=period, min_periods=period, adjust=False).mean()
        
        if use_previous:
            atr = atr.shift(1)
        
        return atr
    
    @staticmethod
    def calculate_price_change(df: pd.DataFrame, column: str = 'close',
                              periods: int = 1, use_previous: bool = True) -> pd.Series:
        """
        Calculate price change over specified periods.
        
        Args:
            df: DataFrame with price data
            column: Column to calculate change on
            periods: Number of periods to look back
            use_previous: If True, shifts result to avoid look-ahead bias
        
        Returns:
            Series with percentage change values
        """
        if df.empty or column not in df.columns:
            return pd.Series(dtype=float)
        
        pct_change = df[column].pct_change(periods=periods) * 100
        
        if use_previous:
            pct_change = pct_change.shift(1)
        
        return pct_change
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            config: Optional configuration dictionary with indicator parameters
        
        Returns:
            DataFrame with all indicators added as columns
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to calculate_all_indicators")
            return df
        
        df = df.copy()
        
        # Default configuration
        if config is None:
            config = {}
        
        rsi_period = config.get('rsi_period', 14)
        sma_period = config.get('sma_period', 20)
        ema_period = config.get('ema_period', 20)
        volume_period = config.get('volume_period', 20)
        bb_period = config.get('bb_period', 20)
        bb_std = config.get('bb_std', 2.0)
        atr_period = config.get('atr_period', 14)
        
        try:
            # VWAP
            df['vwap'] = TechnicalIndicators.calculate_vwap(df)
            
            # RSI
            df['rsi'] = TechnicalIndicators.calculate_rsi(df, period=rsi_period)
            
            # Moving averages
            df['sma'] = TechnicalIndicators.calculate_sma(df, period=sma_period)
            df['ema'] = TechnicalIndicators.calculate_ema(df, period=ema_period)
            
            # Volume metrics
            volume_metrics = TechnicalIndicators.calculate_volume_metrics(df, period=volume_period)
            df['avg_volume'] = volume_metrics['avg_volume']
            df['volume_ratio'] = volume_metrics['volume_ratio']
            df['volume_spike'] = volume_metrics['volume_spike']
            
            # Bollinger Bands
            bb = TechnicalIndicators.calculate_bollinger_bands(
                df, period=bb_period, num_std=bb_std
            )
            df['bb_upper'] = bb['upper_band']
            df['bb_middle'] = bb['middle_band']
            df['bb_lower'] = bb['lower_band']
            
            # ATR
            df['atr'] = TechnicalIndicators.calculate_atr(df, period=atr_period)
            
            # Price changes
            df['price_change_1d'] = TechnicalIndicators.calculate_price_change(df, periods=1)
            df['price_change_5d'] = TechnicalIndicators.calculate_price_change(df, periods=5)
            df['price_change_20d'] = TechnicalIndicators.calculate_price_change(df, periods=20)
            
            logger.info(f"Successfully calculated all indicators for {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
        
        return df
    
    @staticmethod
    def get_indicator_at_market_open(df: pd.DataFrame, date: pd.Timestamp,
                                    indicator: str) -> Optional[float]:
        """
        Get indicator value available at market open (avoids look-ahead bias).
        
        Args:
            df: DataFrame with indicators already calculated
            date: Date for which to get the indicator
            indicator: Name of the indicator column
        
        Returns:
            Indicator value or None if not available
        """
        if df.empty or indicator not in df.columns:
            return None
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns and df.index.name != 'timestamp':
            logger.warning("No timestamp column found in DataFrame")
            return None
        
        # Use index if timestamp is the index
        if df.index.name == 'timestamp':
            time_col = df.index
        else:
            time_col = pd.to_datetime(df['timestamp'])
        
        #