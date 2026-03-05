from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np


@dataclass
class Position:
    """Represents a short position in the portfolio"""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    entry_value: float
    current_price: float = 0.0
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    borrow_fee_rate: float = 0.0
    margin_requirement: float = 0.5
    holding_days: int = 0
    accumulated_borrow_fees: float = 0.0
    slippage_cost: float = 0.0
    
    def update_price(self, current_price: float, current_date: datetime):
        """Update position with current market price"""
        self.current_price = current_price
        if self.entry_date:
            self.holding_days = (current_date - self.entry_date).days
        
        # Calculate unrealized PnL for short position (profit when price drops)
        price_change = self.entry_price - current_price
        self.unrealized_pnl = price_change * self.shares
        
        # Calculate accumulated borrow fees (daily fee on position value)
        if self.holding_days > 0 and self.borrow_fee_rate > 0:
            daily_fee_rate = self.borrow_fee_rate / 365
            self.accumulated_borrow_fees = self.entry_value * daily_fee_rate * self.holding_days
        
        # Subtract fees from unrealized PnL
        self.unrealized_pnl -= self.accumulated_borrow_fees
    
    def close_position(self, exit_price: float, exit_date: datetime, slippage_pct: float = 0.0):
        """Close the position and calculate realized PnL"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.current_price = exit_price
        
        # Apply slippage (for shorts, slippage increases exit price)
        slippage_adjustment = exit_price * slippage_pct
        adjusted_exit_price = exit_price + slippage_adjustment
        self.slippage_cost = slippage_adjustment * self.shares
        
        # Calculate holding period
        self.holding_days = (exit_date - self.entry_date).days
        
        # Calculate final borrow fees
        if self.borrow_fee_rate > 0:
            daily_fee_rate = self.borrow_fee_rate / 365
            self.accumulated_borrow_fees = self.entry_value * daily_fee_rate * self.holding_days
        
        # Realized PnL for short: entry_price - exit_price, minus fees and slippage
        price_change = self.entry_price - adjusted_exit_price
        self.realized_pnl = (price_change * self.shares) - self.accumulated_borrow_fees
        self.unrealized_pnl = 0.0
    
    def get_margin_requirement_amount(self) -> float:
        """Calculate the margin requirement for this position"""
        return self.entry_value * self.margin_requirement
    
    def to_dict(self) -> dict:
        """Convert position to dictionary for reporting"""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'shares': self.shares,
            'entry_value': self.entry_value,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'current_price': self.current_price,
            'holding_days': self.holding_days,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'borrow_fees': self.accumulated_borrow_fees,
            'slippage_cost': self.slippage_cost,
            'return_pct': (self.realized_pnl / self.entry_value * 100) if self.exit_date and self.entry_value > 0 else 0.0
        }


@dataclass
class PortfolioState:
    """Tracks the complete state of the portfolio at a point in time"""
    date: datetime
    cash: float
    equity: float
    total_value: float
    positions_value: float
    num_positions: int
    margin_used: float
    margin_available: float
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0


class Portfolio:
    """Manages portfolio state, positions, PnL, and risk metrics for short selling strategy"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.05,
        max_positions: int = 20,
        default_borrow_fee_rate: float = 0.05,
        margin_requirement: float = 0.5,
        max_slippage_pct: float = 0.005,
        commission_per_share: float = 0.0
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.equity = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.default_borrow_fee_rate = default_borrow_fee_rate
        self.margin_requirement = margin_requirement
        self.max_slippage_pct = max_slippage_pct
        self.commission_per_share = commission_per_share
        
        # Active and closed positions
        self.active_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Historical tracking
        self.daily_states: List[PortfolioState] = []
        self.trade_log: List[dict] = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
        # Daily returns for metrics calculation
        self.daily_returns: List[float] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    def can_open_position(self, symbol: str, price: float) -> Tuple[bool, str]:
        """Check if a new position can be opened"""
        # Check if already holding position
        if symbol in self.active_positions:
            return False, f"Already holding position in {symbol}"
        
        # Check max positions limit
        if len(self.active_positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"
        
        # Calculate position size
        position_value = self.initial_capital * self.position_size_pct
        shares = int(position_value / price)
        
        if shares == 0:
            return False, f"Price too high for position size: ${price}"
        
        # Calculate margin requirement
        margin_needed = position_value * self.margin_requirement
        
        # Check if enough margin available
        current_margin_used = sum(pos.get_margin_requirement_amount() for pos in self.active_positions.values())
        available_margin = self.cash - current_margin_used
        
        if margin_needed > available_margin:
            return False, f"Insufficient margin: need ${margin_needed:.2f}, available ${available_margin:.2f}"
        
        return True, "OK"
    
    def open_position(
        self,
        symbol: str,
        entry_date: datetime,
        entry_price: float,
        borrow_fee_rate: Optional[float] = None,
        slippage_pct: Optional[float] = None
    ) -> Optional[Position]:
        """Open a new short position"""
        can_open, reason = self.can_open_position(symbol, entry_price)
        if not can_open:
            return None
        
        # Use default rates if not specified
        if borrow_fee_rate is None:
            borrow_fee_rate = self.default_borrow_fee_rate
        if slippage_pct is None:
            slippage_pct = self.max_slippage_pct
        
        # Calculate position sizing
        position_value = self.initial_capital * self.position_size_pct
        shares = int(position_value / entry_price)
        
        # Apply slippage to entry price (for shorts, slippage increases entry cost)
        slippage_adjustment = entry_price * slippage_pct
        adjusted_entry_price = entry_price + slippage_adjustment
        actual_entry_value = adjusted_entry_price * shares
        
        # Apply commission
        commission = shares * self.commission_per_share
        total_entry_cost = commission
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=adjusted_entry_price,
            shares=shares,
            entry_value=actual_entry_value,
            current_price=adjusted_entry_price,
            borrow_fee_rate=borrow_fee_rate,
            margin_requirement=self.margin_requirement,
            slippage_cost=slippage_adjustment * shares
        )
        
        # Update portfolio state
        self.active_positions[symbol] = position
        self.cash -= total_entry_cost
        self.total_trades += 1
        
        # Log trade
        self.trade_log.append({
            'date': entry_date,
            'action': 'OPEN_SHORT',
            'symbol': symbol,
            'price': adjusted_entry_price,
            'shares': shares,
            'value': actual_entry_value,
            'slippage': slippage_adjustment * shares,
            'commission': commission
        })
        
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_date: datetime,
        exit_price: float,
        slippage_pct: Optional[float] = None
    ) -> Optional[Position]:
        """Close an existing short position"""
        if symbol not in self.active_positions:
            return None
        
        if slippage_pct is None:
            slippage_pct = self.max_slippage_pct
        
        position = self.active_positions[symbol]
        position.close_position(exit_price, exit_date, slippage_pct)
        
        # Apply commission
        commission = position.shares * self.commission_per_share
        position.realized_pnl -= commission
        
        # Update portfolio state
        self.cash += position.realized_pnl
        self.total_pnl += position.realized_pnl
        
        # Track win/loss
        if position.realized_pnl > 0:
            self.winning_trades += 1
        elif position.realized_pnl < 0:
            self.losing_trades += 1
        
        # Move to closed positions
        del self.active_positions[symbol]
        self.closed_positions.append(position)
        
        # Log trade
        self.trade_log.append({
            'date': exit_date,
            'action': 'CLOSE_SHORT',
            'symbol': symbol,
            'price': exit_price,
            'shares': position.shares,
            'pnl': position.realized_pnl,
            'return_pct': (position.realized_pnl / position.entry_value * 100) if position.entry_value > 0 else 0.0,
            'holding_days': position.holding_days,
            'borrow_fees': position.accumulated_borrow_fees,
            'slippage': position.slippage_cost,
            'commission': commission
        })
        
        return position
    
    def update_positions(self, current_date: datetime, price_data: Dict[str, float]):
        """Update all active positions with current market prices"""
        for symbol, position in self.active_positions.items():
            if symbol in price_data:
                position.update_price(price_data[symbol], current_date)
    
    def calculate_equity(self) -> float:
        """Calculate current total equity"""
        positions_value = sum(pos.entry_value for pos in self.active_positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
        return self.cash + unrealized_pnl
    
    def update_daily_state(self, current_date: datetime):
        """Record daily portfolio state"""
        # Calculate current equity
        self.equity = self.calculate_equity()
        
        # Calculate positions value
        positions_value = sum(pos.entry_value for pos in self.active_positions.values())
        
        # Calculate margin usage
        margin_used = sum(pos.get_margin_requirement_amount() for pos in self.active_positions.values())
        margin_available = self.cash - margin_used
        
        # Calculate daily PnL
        previous_equity = self.daily_states[-1].equity if self.daily_states else self.initial_capital
        daily_pnl = self.equity - previous_equity
        
        # Calculate daily return
        if previous_equity > 0:
            daily_return = (self.equity - previous_equity) / previous_equity
            self.daily_returns.append(daily_return)
        
        # Track equity curve
        self.equity_curve.append((current_date, self.equity))
        
        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Create state snapshot
        state = PortfolioState(
            date=current_date,
            cash=self.cash,
            equity=self.equity,
            total_value=self.equity,
            positions_value=positions_value,
            num_positions=len(self.active_positions),
            margin_used=margin_used,
            margin_available=margin_available,
            daily_pnl=daily_pnl,
            cumulative_pnl=self.equity - self.initial_capital
        )
        
        self.daily_states.append(state)
    
    def get_performance_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""
        if not self.daily_returns or not self.closed_positions:
            return self._empty_metrics()
        
        # Basic metrics
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        total_return_pct = total_return * 100
        
        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        # Average trade metrics
        avg_trade_pnl = self.total_pnl / len(self.closed_positions) if self.closed_positions else 0.0
        
        winning_pnls = [pos.realized_pnl for pos in self.closed_positions if pos.realized_pnl > 0]
        losing_pnls = [pos.realized