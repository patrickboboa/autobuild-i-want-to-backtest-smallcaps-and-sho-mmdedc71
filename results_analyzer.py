import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes backtest results and generates comprehensive performance metrics."""
    
    def __init__(self, trades: List[Dict], portfolio_values: pd.Series, 
                 benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize the results analyzer.
        
        Args:
            trades: List of trade dictionaries with entry/exit details
            portfolio_values: Time series of portfolio values
            benchmark_returns: Optional benchmark returns for comparison
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.trades = trades
        self.portfolio_values = portfolio_values
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.metrics = {}
        
    def analyze(self) -> Dict:
        """
        Run complete analysis and return all metrics.
        
        Returns:
            Dictionary containing all performance metrics
        """
        logger.info("Starting results analysis...")
        
        if len(self.portfolio_values) == 0:
            logger.warning("No portfolio data to analyze")
            return self._empty_metrics()
            
        if len(self.trades) == 0:
            logger.warning("No trades to analyze")
            
        self.metrics = {
            'summary': self._calculate_summary_stats(),
            'returns': self._calculate_return_metrics(),
            'risk': self._calculate_risk_metrics(),
            'drawdown': self._calculate_drawdown_metrics(),
            'trade_analysis': self._analyze_trades(),
            'time_analysis': self._analyze_time_periods(),
            'short_specific': self._analyze_short_metrics(),
        }
        
        logger.info("Analysis complete")
        return self.metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'summary': {},
            'returns': {},
            'risk': {},
            'drawdown': {},
            'trade_analysis': {},
            'time_analysis': {},
            'short_specific': {},
        }
    
    def _calculate_summary_stats(self) -> Dict:
        """Calculate high-level summary statistics."""
        initial_value = self.portfolio_values.iloc[0]
        final_value = self.portfolio_values.iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate period
        start_date = self.portfolio_values.index[0]
        end_date = self.portfolio_values.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        
        return {
            'initial_capital': initial_value,
            'final_capital': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': days,
            'duration_years': years,
            'total_trades': len(self.trades),
        }
    
    def _calculate_return_metrics(self) -> Dict:
        """Calculate return-based metrics."""
        returns = self.portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Annualization factor (assuming daily data)
        trading_days_per_year = 252
        annualization_factor = np.sqrt(trading_days_per_year)
        
        # Calculate CAGR
        years = self.metrics.get('summary', {}).get('duration_years', 1)
        if years > 0:
            initial_value = self.portfolio_values.iloc[0]
            final_value = self.portfolio_values.iloc[-1]
            cagr = (final_value / initial_value) ** (1 / years) - 1
        else:
            cagr = 0
        
        return {
            'cagr': cagr,
            'cagr_pct': cagr * 100,
            'mean_daily_return': returns.mean(),
            'mean_daily_return_pct': returns.mean() * 100,
            'median_daily_return': returns.median(),
            'std_daily_return': returns.std(),
            'annualized_return': returns.mean() * trading_days_per_year,
            'annualized_volatility': returns.std() * annualization_factor,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'positive_day_ratio': (returns > 0).mean(),
        }
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk-adjusted performance metrics."""
        returns = self.portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        trading_days_per_year = 252
        annualization_factor = np.sqrt(trading_days_per_year)
        
        # Sharpe Ratio
        excess_returns = returns - (self.risk_free_rate / trading_days_per_year)
        sharpe_ratio = np.sqrt(trading_days_per_year) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(trading_days_per_year) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio (CAGR / Max Drawdown)
        cagr = self.metrics.get('returns', {}).get('cagr', 0)
        max_dd = abs(self._calculate_drawdown_metrics().get('max_drawdown', 1))
        calmar_ratio = cagr / max_dd if max_dd > 0 else 0
        
        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'downside_deviation': downside_std * annualization_factor,
        }
    
    def _calculate_drawdown_metrics(self) -> Dict:
        """Calculate drawdown statistics."""
        cumulative = self.portfolio_values
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()
        
        # Find the peak before max drawdown
        peak_idx = cumulative[:max_drawdown_idx].idxmax()
        
        # Find recovery date (if recovered)
        recovery_idx = None
        peak_value = cumulative[peak_idx]
        after_trough = cumulative[max_drawdown_idx:]
        recovered = after_trough[after_trough >= peak_value]
        if len(recovered) > 0:
            recovery_idx = recovered.index[0]
        
        # Calculate drawdown duration
        drawdown_duration = None
        if recovery_idx:
            drawdown_duration = (recovery_idx - peak_idx).days
        else:
            drawdown_duration = (cumulative.index[-1] - peak_idx).days
        
        # Average drawdown
        drawdowns = drawdown[drawdown < 0]
        avg_drawdown = drawdowns.mean() if len(drawdowns) > 0 else 0
        
        # Count number of drawdown periods
        drawdown_periods = self._count_drawdown_periods(drawdown)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_date': max_drawdown_idx,
            'peak_date': peak_idx,
            'recovery_date': recovery_idx,
            'drawdown_duration_days': drawdown_duration,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_pct': avg_drawdown * 100,
            'num_drawdown_periods': drawdown_periods,
            'current_drawdown': drawdown.iloc[-1],
            'current_drawdown_pct': drawdown.iloc[-1] * 100,
        }
    
    def _count_drawdown_periods(self, drawdown: pd.Series) -> int:
        """Count distinct drawdown periods."""
        is_drawdown = drawdown < 0
        periods = 0
        in_drawdown = False
        
        for dd in is_drawdown:
            if dd and not in_drawdown:
                periods += 1
                in_drawdown = True
            elif not dd:
                in_drawdown = False
                
        return periods
    
    def _analyze_trades(self) -> Dict:
        """Analyze individual trade statistics."""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate P&L for each trade
        if 'pnl' in trades_df.columns:
            pnls = trades_df['pnl']
        elif 'exit_price' in trades_df.columns and 'entry_price' in trades_df.columns:
            # For shorts: profit when exit < entry
            pnls = trades_df['entry_price'] - trades_df['exit_price']
            if 'quantity' in trades_df.columns:
                pnls = pnls * trades_df['quantity']
            if 'total_costs' in trades_df.columns:
                pnls = pnls - trades_df['total_costs']
        else:
            pnls = pd.Series([0] * len(trades_df))
        
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]
        breakeven_trades = pnls[pnls == 0]
        
        # Calculate returns
        if 'return_pct' in trades_df.columns:
            returns = trades_df['return_pct']
        elif 'exit_price' in trades_df.columns and 'entry_price' in trades_df.columns:
            # For shorts: return = (entry - exit) / entry
            returns = (trades_df['entry_price'] - trades_df['exit_price']) / trades_df['entry_price']
        else:
            returns = pd.Series([0] * len(trades_df))
        
        # Winning/losing streaks
        win_streak, loss_streak = self._calculate_streaks(pnls > 0)
        
        # Average holding period
        if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
            trades_df['holding_period'] = (
                pd.to_datetime(trades_df['exit_date']) - 
                pd.to_datetime(trades_df['entry_date'])
            ).dt.days
            avg_holding_period = trades_df['holding_period'].mean()
            max_holding_period = trades_df['holding_period'].max()
            min_holding_period = trades_df['holding_period'].min()
        else:
            avg_holding_period = 0
            max_holding_period = 0
            min_holding_period = 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        win_rate = len(winning_trades) / len(pnls) if len(pnls) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'breakeven_trades': len(breakeven_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'loss_rate': len(losing_trades) / len(pnls) if len(pnls) > 0 else 0,
            'avg_profit': pnls.mean(),
            'median_profit': pnls.median(),
            'total_profit': pnls.sum(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': winning_trades.max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades.min() if len(losing_trades) > 0 else 0,
            'avg_return_pct': returns.mean() * 100,
            'median_return_pct': returns.median() * 100,
            'best_return_pct': returns.max() * 100,
            'worst_return_pct': returns.min() * 100,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_consecutive_wins': win_streak,
            'max_consecutive_losses': loss_streak,
            'avg_holding_period_days': avg_holding_period,
            'max_holding_period_days': max_holding_period,
            'min_holding_period_days': min_holding_period,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
        }
    
    def _calculate_streaks(self, is_winner: pd.Series) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if len(is_winner) == 0:
            return 0, 0
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for win in is_winner:
            if win: