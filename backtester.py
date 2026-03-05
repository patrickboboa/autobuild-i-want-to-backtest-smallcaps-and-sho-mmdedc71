import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single short trade"""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    entry_value: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    borrow_fee_daily: float = 0.0
    total_borrow_fees: float = 0.0
    slippage_cost: float = 0.0
    commission_entry: float = 0.0
    commission_exit: float = 0.0
    hold_days: int = 0
    exit_reason: str = ''
    

@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    initial_capital: float = 100000.0
    position_size_pct: float = 0.1  # % of portfolio per position
    max_positions: int = 10
    commission_pct: float = 0.001  # 0.1% per trade
    borrow_fee_annual: float = 0.05  # 5% annual borrow fee
    margin_requirement: float = 1.5  # 150% margin for short sales
    slippage_pct: float = 0.001  # 0.1% slippage on entry/exit
    max_position_size: float = 50000.0  # Max $ per position for liquidity
    market_open_time: time = field(default_factory=lambda: time(9, 30))
    market_close_time: time = field(default_factory=lambda: time(16, 0))
    

class Backtester:
    """
    Core backtesting engine for short selling strategies.
    Handles trade execution, position management, and performance tracking.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.daily_equity: List[Tuple[datetime, float]] = []
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
        self.reserved_margin = 0.0
        
    def reset(self):
        """Reset backtester state"""
        self.trades = []
        self.open_positions = {}
        self.daily_equity = []
        self.cash = self.initial_capital
        self.reserved_margin = 0.0
        
    def calculate_position_size(self, price: float, avg_volume: float) -> int:
        """
        Calculate position size based on available capital and liquidity constraints.
        
        Args:
            price: Entry price
            avg_volume: Average daily volume
            
        Returns:
            Number of shares to short
        """
        # Available capital for new position
        available = self.cash - self.reserved_margin
        target_value = min(
            available * self.config.position_size_pct,
            self.config.max_position_size
        )
        
        # Liquidity constraint: don't short more than 1% of daily volume
        max_shares_liquidity = int(avg_volume * 0.01) if avg_volume > 0 else 0
        max_value_liquidity = max_shares_liquidity * price
        
        # Take minimum of capital and liquidity constraints
        position_value = min(target_value, max_value_liquidity)
        
        if position_value < price:
            return 0
            
        shares = int(position_value / price)
        
        # Check margin requirements
        required_margin = shares * price * self.config.margin_requirement
        if self.cash - self.reserved_margin < required_margin:
            # Reduce shares to meet margin requirement
            max_shares_margin = int((self.cash - self.reserved_margin) / (price * self.config.margin_requirement))
            shares = min(shares, max_shares_margin)
            
        return max(0, shares)
    
    def calculate_slippage(self, price: float, shares: int, is_entry: bool) -> float:
        """
        Calculate slippage cost for trade execution.
        Short entry = selling, so slippage reduces proceeds
        Short exit = buying, so slippage increases cost
        
        Args:
            price: Base price
            shares: Number of shares
            is_entry: True for entry, False for exit
            
        Returns:
            Slippage cost (always positive)
        """
        base_slippage = price * shares * self.config.slippage_pct
        # Small caps have higher slippage
        return base_slippage
    
    def calculate_commission(self, value: float) -> float:
        """Calculate commission cost"""
        return value * self.config.commission_pct
    
    def calculate_borrow_fee(self, value: float, days: int) -> float:
        """Calculate borrow fee for holding short position"""
        daily_rate = self.config.borrow_fee_annual / 365.0
        return value * daily_rate * days
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        if len(self.open_positions) >= self.config.max_positions:
            return False
            
        available = self.cash - self.reserved_margin
        min_position = 1000.0  # Minimum $1000 position
        
        return available > min_position * self.config.margin_requirement
    
    def enter_short(
        self,
        symbol: str,
        date: datetime,
        price: float,
        avg_volume: float
    ) -> Optional[Trade]:
        """
        Enter a short position at market open.
        
        Args:
            symbol: Stock symbol
            date: Entry date
            price: Entry price (open price)
            avg_volume: Average daily volume for liquidity check
            
        Returns:
            Trade object if successful, None otherwise
        """
        # Check if already in position
        if symbol in self.open_positions:
            logger.debug(f"Already in short position for {symbol}")
            return None
            
        # Check if can open new position
        if not self.can_open_position():
            logger.debug(f"Cannot open new position - max positions or insufficient capital")
            return None
            
        # Calculate position size
        shares = self.calculate_position_size(price, avg_volume)
        if shares == 0:
            logger.debug(f"Position size too small for {symbol}")
            return None
            
        # Calculate costs
        entry_value = shares * price
        slippage = self.calculate_slippage(price, shares, is_entry=True)
        commission = self.calculate_commission(entry_value)
        
        # Actual proceeds from short sale (reduced by slippage)
        proceeds = entry_value - slippage
        
        # Calculate margin requirement
        margin_required = entry_value * self.config.margin_requirement
        
        # Check if we have enough cash for margin
        if self.cash < margin_required + commission:
            logger.debug(f"Insufficient cash for margin requirement: {symbol}")
            return None
            
        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            shares=shares,
            entry_value=entry_value,
            borrow_fee_daily=self.config.borrow_fee_annual / 365.0,
            slippage_cost=slippage,
            commission_entry=commission
        )
        
        # Update cash and margin
        self.cash += proceeds - commission
        self.reserved_margin += margin_required
        
        # Add to open positions
        self.open_positions[symbol] = trade
        
        logger.info(f"Entered short: {symbol} @ ${price:.2f}, {shares} shares, value ${entry_value:.2f}")
        
        return trade
    
    def exit_short(
        self,
        symbol: str,
        date: datetime,
        price: float,
        reason: str = 'signal'
    ) -> Optional[Trade]:
        """
        Exit a short position.
        
        Args:
            symbol: Stock symbol
            date: Exit date
            price: Exit price
            reason: Reason for exit (signal, stop_loss, take_profit, etc.)
            
        Returns:
            Completed trade object if successful, None otherwise
        """
        if symbol not in self.open_positions:
            logger.warning(f"Attempted to exit non-existent position: {symbol}")
            return None
            
        trade = self.open_positions[symbol]
        
        # Calculate holding period
        hold_days = (date - trade.entry_date).days
        if hold_days < 0:
            logger.error(f"Exit date before entry date for {symbol}")
            return None
            
        # Calculate costs
        exit_value = trade.shares * price
        slippage = self.calculate_slippage(price, trade.shares, is_entry=False)
        commission = self.calculate_commission(exit_value)
        borrow_fees = self.calculate_borrow_fee(trade.entry_value, hold_days)
        
        # Cost to buy back shares (including slippage)
        buyback_cost = exit_value + slippage + commission
        
        # Update trade
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_value = exit_value
        trade.commission_exit = commission
        trade.total_borrow_fees = borrow_fees
        trade.hold_days = hold_days
        trade.exit_reason = reason
        
        # Calculate P&L: proceeds from short sale - buyback cost - fees
        trade.pnl = (trade.entry_value - trade.slippage_cost - trade.commission_entry) - \
                    (exit_value + slippage + commission) - borrow_fees
        trade.pnl_pct = (trade.pnl / trade.entry_value) * 100
        
        # Update cash and margin
        margin_released = trade.entry_value * self.config.margin_requirement
        self.cash -= buyback_cost
        self.reserved_margin -= margin_released
        
        # Move to completed trades
        self.trades.append(trade)
        del self.open_positions[symbol]
        
        logger.info(f"Exited short: {symbol} @ ${price:.2f}, P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%), reason: {reason}")
        
        return trade
    
    def update_positions_eod(self, date: datetime, prices: Dict[str, float]):
        """
        Update open positions at end of day for margin calculations and borrow fees.
        
        Args:
            date: Current date
            prices: Dictionary of symbol -> closing price
        """
        for symbol, trade in list(self.open_positions.items()):
            if symbol not in prices:
                logger.warning(f"Missing price data for {symbol} on {date}")
                continue
                
            current_price = prices[symbol]
            
            # Check for margin call (position moved against us significantly)
            current_value = trade.shares * current_price
            margin_required = current_value * self.config.margin_requirement
            
            # If current loss exceeds 50% of initial margin, force close
            unrealized_loss = current_value - trade.entry_value
            if unrealized_loss > trade.entry_value * self.config.margin_requirement * 0.5:
                logger.warning(f"Margin call - force closing {symbol}")
                self.exit_short(symbol, date, current_price, reason='margin_call')
    
    def calculate_equity(self, date: datetime, prices: Dict[str, float]) -> float:
        """
        Calculate total equity (cash + unrealized P&L).
        
        Args:
            date: Current date
            prices: Dictionary of symbol -> price
            
        Returns:
            Total equity value
        """
        equity = self.cash
        
        # Add unrealized P&L from open positions
        for symbol, trade in self.open_positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                days_held = (date - trade.entry_date).days
                
                # Unrealized P&L
                current_value = trade.shares * current_price
                unrealized_pnl = trade.entry_value - current_value
                
                # Subtract accrued borrow fees
                accrued_fees = self.calculate_borrow_fee(trade.entry_value, days_held)
                
                equity += unrealized_pnl - accrued_fees
                
        return equity
    
    def run_backtest(
        self,
        signals: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Run the backtest over the specified period.
        
        Args:
            signals: DataFrame with columns [date, symbol, signal, avg_volume]
                    signal: 1 for short entry, -1 for exit, 0 for hold
            price_data: Dictionary mapping symbol -> DataFrame with OHLCV data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        # Ensure signals are sorted by date
        signals = signals.sort_values('date').reset_index(drop=True)
        
        # Get unique dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = [d for d in dates if d.weekday() < 5]  # Trading days only
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        for current_date in dates:
            # Get signals for this date
            daily_signals = signals[signals['date'] == current_date]
            
            # Collect closing prices for EOD updates
            eod_prices = {}
            
            # Process entry signals
            entry_signals = daily_signals[daily_signals['signal'] == 1]
            for _, row in entry_signals.iterrows():
                symbol = row['symbol']
                
                if symbol not in price_data:
                    continue
                    
                symbol_data = price_data[symbol]
                day_data = symbol_data[symbol_data['date'] == current_date]
                
                if day_data.empty:
                    continue
                    
                open_price = day_data.iloc[0]['open']
                avg_volume = row.get('avg_volume', day_data.iloc[0].get('volume', 0))
                
                self.enter_short(symbol, current_date, open_price, avg_volume)
                
                # Store close price for EOD
                eod_prices[symbol] = day_data.iloc[0]['close']
            
            # Process exit signals
            exit_signals = daily_signals[daily_signals['signal'] == -1]
            for _, row in exit_signals.iterrows():
                symbol = row['symbol']
                
                if symbol not in self.open_positions:
                    continue
                    
                if symbol not in price_data:
                    continue
                    
                symbol_data = price_data[symbol]
                day_data = symbol_data[symbol_data['date'] == current_date]
                
                if day_data.empty:
                    continue
                    
                # Exit at open price (