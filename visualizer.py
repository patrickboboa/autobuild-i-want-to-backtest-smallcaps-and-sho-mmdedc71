import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Creates charts and visualizations for backtesting results"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save output images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.size'] = 10
        
    def create_full_report(self, results: Dict, save: bool = True) -> None:
        """
        Create a comprehensive visualization report
        
        Args:
            results: Dictionary containing backtest results
            save: Whether to save the figure to file
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1, results)
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_drawdown(ax2, results)
        
        # Trade distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_trade_distribution(ax3, results)
        
        # Monthly returns heatmap
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_monthly_returns(ax4, results)
        
        plt.suptitle('Backtest Results Report', fontsize=16, fontweight='bold', y=0.995)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"backtest_report_{timestamp}.png"
            plt.savefig(filepath, bbox_inches='tight')
            print(f"Report saved to {filepath}")
        
        plt.show()
        
    def _plot_equity_curve(self, ax: plt.Axes, results: Dict) -> None:
        """Plot the equity curve with buy & hold comparison"""
        equity_curve = results.get('equity_curve', pd.Series())
        
        if equity_curve.empty:
            ax.text(0.5, 0.5, 'No equity curve data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Ensure index is datetime
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # Plot strategy equity
        ax.plot(equity_curve.index, equity_curve.values, 
               label='Strategy', linewidth=2, color='#2E86AB')
        
        # Add benchmark if available
        benchmark = results.get('benchmark_equity', None)
        if benchmark is not None and not benchmark.empty:
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                benchmark.index = pd.to_datetime(benchmark.index)
            ax.plot(benchmark.index, benchmark.values, 
                   label='Buy & Hold', linewidth=1.5, alpha=0.7, 
                   color='#A23B72', linestyle='--')
        
        # Highlight drawdown periods
        underwater = results.get('underwater', pd.Series())
        if not underwater.empty:
            if not isinstance(underwater.index, pd.DatetimeIndex):
                underwater.index = pd.to_datetime(underwater.index)
            dd_periods = underwater < -0.1  # Highlight 10%+ drawdowns
            if dd_periods.any():
                ax.fill_between(underwater.index, equity_curve.min(), 
                               equity_curve.max(), where=dd_periods,
                               alpha=0.1, color='red', label='Drawdown > 10%')
        
        # Formatting
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontweight='bold')
        ax.set_title('Equity Curve', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
    def _plot_drawdown(self, ax: plt.Axes, results: Dict) -> None:
        """Plot drawdown over time"""
        underwater = results.get('underwater', pd.Series())
        
        if underwater.empty:
            ax.text(0.5, 0.5, 'No drawdown data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Ensure index is datetime
        if not isinstance(underwater.index, pd.DatetimeIndex):
            underwater.index = pd.to_datetime(underwater.index)
        
        # Plot underwater (drawdown) percentage
        ax.fill_between(underwater.index, 0, underwater.values * 100, 
                        alpha=0.5, color='#C73E1D', label='Drawdown')
        ax.plot(underwater.index, underwater.values * 100, 
               color='#8B0000', linewidth=1.5)
        
        # Mark maximum drawdown
        max_dd_idx = underwater.idxmin()
        max_dd_val = underwater.min() * 100
        ax.plot(max_dd_idx, max_dd_val, 'o', color='red', 
               markersize=8, label=f'Max DD: {max_dd_val:.2f}%')
        ax.annotate(f'{max_dd_val:.2f}%', 
                   xy=(max_dd_idx, max_dd_val),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Formatting
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontweight='bold')
        ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_ylim(bottom=min(max_dd_val * 1.2, -5))
        
    def _plot_trade_distribution(self, ax: plt.Axes, results: Dict) -> None:
        """Plot distribution of trade returns"""
        trades = results.get('trades', [])
        
        if not trades:
            ax.text(0.5, 0.5, 'No trades executed', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract returns
        returns = [t.get('return_pct', 0) for t in trades]
        
        if not returns:
            ax.text(0.5, 0.5, 'No return data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create histogram
        n, bins, patches = ax.hist(returns, bins=50, alpha=0.7, 
                                   color='#2E86AB', edgecolor='black')
        
        # Color bars based on positive/negative
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#C73E1D')
            else:
                patch.set_facecolor('#06A77D')
        
        # Add vertical line at zero
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add mean and median lines
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        ax.axvline(mean_return, color='blue', linestyle='--', 
                  linewidth=1.5, label=f'Mean: {mean_return:.2f}%')
        ax.axvline(median_return, color='green', linestyle='--', 
                  linewidth=1.5, label=f'Median: {median_return:.2f}%')
        
        # Formatting
        ax.set_xlabel('Return (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Trade Return Distribution', fontsize=12, fontweight='bold', pad=10)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_monthly_returns(self, ax: plt.Axes, results: Dict) -> None:
        """Plot monthly returns heatmap"""
        equity_curve = results.get('equity_curve', pd.Series())
        
        if equity_curve.empty:
            ax.text(0.5, 0.5, 'No equity data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Ensure index is datetime
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # Calculate monthly returns
        monthly_equity = equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100
        
        if len(monthly_returns) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for monthly returns', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create pivot table for heatmap
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='month', columns='year', values='return')
        
        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax, 
                   linewidths=0.5, linecolor='gray')
        
        # Format month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticklabels([month_labels[int(i)-1] for i in pivot.index], rotation=0)
        
        # Formatting
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Month', fontweight='bold')
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold', pad=10)
        
    def plot_trade_timeline(self, results: Dict, save: bool = True) -> None:
        """
        Plot trades on a timeline with entry/exit points
        
        Args:
            results: Dictionary containing backtest results
            save: Whether to save the figure to file
        """
        trades = results.get('trades', [])
        equity_curve = results.get('equity_curve', pd.Series())
        
        if not trades or equity_curve.empty:
            print("No trade data available for timeline plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                       sharex=True, height_ratios=[2, 1])
        
        # Ensure index is datetime
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # Plot equity curve
        ax1.plot(equity_curve.index, equity_curve.values, 
                color='#2E86AB', linewidth=2, label='Portfolio Value')
        
        # Plot trade markers
        for trade in trades:
            entry_date = pd.to_datetime(trade.get('entry_date'))
            exit_date = pd.to_datetime(trade.get('exit_date'))
            
            # Get equity values at entry and exit
            entry_equity = equity_curve.asof(entry_date)
            exit_equity = equity_curve.asof(exit_date)
            
            # Determine color based on profit/loss
            trade_return = trade.get('return_pct', 0)
            color = '#06A77D' if trade_return > 0 else '#C73E1D'
            
            # Plot entry point
            ax1.plot(entry_date, entry_equity, 'v', color=color, 
                    markersize=8, alpha=0.7)
            
            # Plot exit point
            ax1.plot(exit_date, exit_equity, '^', color=color, 
                    markersize=8, alpha=0.7)
            
            # Draw line connecting entry and exit
            ax1.plot([entry_date, exit_date], [entry_equity, exit_equity],
                    color=color, alpha=0.3, linewidth=1)
        
        ax1.set_ylabel('Portfolio Value ($)', fontweight='bold')
        ax1.set_title('Trade Timeline', fontsize=14, fontweight='bold', pad=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot cumulative trade count
        trade_dates = [pd.to_datetime(t['exit_date']) for t in trades]
        trade_counts = list(range(1, len(trades) + 1))
        
        ax2.step(trade_dates, trade_counts, where='post', 
                color='#2E86AB', linewidth=2)
        ax2.fill_between(trade_dates, 0, trade_counts, step='post',
                        alpha=0.3, color='#2E86