import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from dotenv import load_dotenv

from config import Config
from data_fetcher import DataFetcher
from strategy_parser import StrategyParser
from backtester import Backtester
from portfolio import Portfolio
from results_analyzer import ResultsAnalyzer
from visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class BacktestRunner:
    def __init__(self):
        load_dotenv()
        
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
        
        self.config = Config()
        self.data_fetcher = DataFetcher(self.api_key, self.config)
        self.strategy_parser = StrategyParser()
        self.visualizer = Visualizer()
        
        logger.info("BacktestRunner initialized successfully")
    
    def get_user_input(self) -> Dict:
        """Get backtest parameters from user"""
        print("\n" + "="*70)
        print("SMALLCAP SHORT STRATEGY BACKTESTER")
        print("="*70)
        
        print("\n1. Strategy Definition")
        print("-" * 70)
        print("Enter your strategy in plain English. Examples:")
        print("  - Short stocks with market cap under 500M and volume over 1M")
        print("  - Short if price is above VWAP by 5% and RSI is over 70")
        print("  - Short stocks with negative earnings and debt ratio over 2")
        print()
        
        strategy_text = input("Your strategy: ").strip()
        if not strategy_text:
            strategy_text = "Short stocks with market cap under 500M and volume over 1M"
            print(f"Using default: {strategy_text}")
        
        print("\n2. Backtest Period")
        print("-" * 70)
        end_date_input = input("End date (YYYY-MM-DD) [default: today]: ").strip()
        if end_date_input:
            try:
                end_date = datetime.strptime(end_date_input, "%Y-%m-%d")
            except ValueError:
                print("Invalid date format, using today")
                end_date = datetime.now()
        else:
            end_date = datetime.now()
        
        years_back = input("Years to backtest [default: 5]: ").strip()
        try:
            years_back = int(years_back) if years_back else 5
        except ValueError:
            years_back = 5
            print("Invalid input, using 5 years")
        
        start_date = end_date - timedelta(days=years_back * 365)
        
        print("\n3. Portfolio Settings")
        print("-" * 70)
        initial_capital = input("Initial capital [default: 100000]: ").strip()
        try:
            initial_capital = float(initial_capital) if initial_capital else 100000.0
        except ValueError:
            initial_capital = 100000.0
            print("Invalid input, using $100,000")
        
        max_positions = input("Maximum concurrent positions [default: 20]: ").strip()
        try:
            max_positions = int(max_positions) if max_positions else 20
        except ValueError:
            max_positions = 20
            print("Invalid input, using 20 positions")
        
        position_size = input("Position size as % of portfolio [default: 5]: ").strip()
        try:
            position_size = float(position_size) if position_size else 5.0
        except ValueError:
            position_size = 5.0
            print("Invalid input, using 5%")
        
        print("\n4. Short Selling Costs")
        print("-" * 70)
        borrow_fee = input("Annual borrow fee % [default: 8.0]: ").strip()
        try:
            borrow_fee = float(borrow_fee) if borrow_fee else 8.0
        except ValueError:
            borrow_fee = 8.0
            print("Invalid input, using 8%")
        
        print("\n5. Stock Universe")
        print("-" * 70)
        print("Options:")
        print("  1. Auto-detect smallcaps (market cap < $2B)")
        print("  2. Provide ticker list")
        print("  3. Use watchlist file")
        
        universe_option = input("Choose option [1-3, default: 1]: ").strip()
        
        tickers = []
        if universe_option == "2":
            ticker_input = input("Enter comma-separated tickers: ").strip()
            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        elif universe_option == "3":
            filename = input("Enter watchlist filename: ").strip()
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                    tickers = [t.strip().upper() for t in content.replace('\n', ',').split(',') if t.strip()]
            except FileNotFoundError:
                print(f"File {filename} not found, using auto-detect")
                universe_option = "1"
        
        return {
            'strategy_text': strategy_text,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'max_positions': max_positions,
            'position_size_pct': position_size,
            'borrow_fee_annual': borrow_fee,
            'universe_option': universe_option,
            'tickers': tickers
        }
    
    def discover_smallcaps(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Discover smallcap stocks for the backtest period"""
        logger.info("Discovering smallcap stocks...")
        print("\nDiscovering smallcap stocks (this may take a few minutes)...")
        
        tickers = self.data_fetcher.get_smallcap_universe(
            start_date, 
            end_date,
            max_market_cap=2_000_000_000  # $2B
        )
        
        logger.info(f"Found {len(tickers)} smallcap stocks")
        print(f"Found {len(tickers)} smallcap stocks in the universe")
        
        return tickers
    
    def run_backtest(self, params: Dict):
        """Execute the backtest with given parameters"""
        logger.info("Starting backtest execution")
        print("\n" + "="*70)
        print("RUNNING BACKTEST")
        print("="*70)
        
        try:
            # Parse strategy
            print("\n[1/6] Parsing strategy...")
            strategy_criteria = self.strategy_parser.parse(params['strategy_text'])
            logger.info(f"Parsed strategy: {strategy_criteria}")
            print(f"Strategy parsed: {len(strategy_criteria['conditions'])} conditions identified")
            
            # Get stock universe
            print("\n[2/6] Building stock universe...")
            if params['universe_option'] == '1':
                tickers = self.discover_smallcaps(
                    params['start_date'], 
                    params['end_date']
                )
            else:
                tickers = params['tickers']
            
            if not tickers:
                raise ValueError("No tickers found in universe")
            
            print(f"Universe contains {len(tickers)} stocks")
            
            # Fetch historical data
            print("\n[3/6] Fetching historical data...")
            print("This may take several minutes depending on data volume...")
            historical_data = self.data_fetcher.fetch_historical_data(
                tickers,
                params['start_date'],
                params['end_date']
            )
            
            available_tickers = len(historical_data)
            print(f"Successfully fetched data for {available_tickers} stocks")
            
            if available_tickers == 0:
                raise ValueError("No historical data available for any tickers")
            
            # Initialize portfolio
            portfolio = Portfolio(
                initial_capital=params['initial_capital'],
                max_positions=params['max_positions'],
                position_size_pct=params['position_size_pct'],
                borrow_fee_annual=params['borrow_fee_annual']
            )
            
            # Run backtest
            print("\n[4/6] Running backtest simulation...")
            backtester = Backtester(
                portfolio=portfolio,
                strategy_criteria=strategy_criteria,
                config=self.config
            )
            
            results = backtester.run(historical_data, params['start_date'], params['end_date'])
            
            # Analyze results
            print("\n[5/6] Analyzing results...")
            analyzer = ResultsAnalyzer(results, portfolio)
            metrics = analyzer.calculate_metrics()
            
            # Display results
            print("\n[6/6] Generating visualizations...")
            self.display_results(metrics, results, params)
            
            # Generate visualizations
            self.visualizer.create_dashboard(results, metrics, params)
            
            # Save results
            self.save_results(results, metrics, params)
            
            logger.info("Backtest completed successfully")
            print("\n" + "="*70)
            print("BACKTEST COMPLETED SUCCESSFULLY")
            print("="*70)
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}", exc_info=True)
            print(f"\n❌ ERROR: {str(e)}")
            raise
    
    def display_results(self, metrics: Dict, results: Dict, params: Dict):
        """Display backtest results to console"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS SUMMARY")
        print("="*70)
        
        print("\n📊 PERFORMANCE METRICS")
        print("-" * 70)
        print(f"Total Return:          {metrics['total_return']:.2f}%")
        print(f"Annual Return (CAGR):  {metrics['cagr']:.2f}%")
        print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:          {metrics['max_drawdown']:.2f}%")
        print(f"Win Rate:              {metrics['win_rate']:.2f}%")
        
        print("\n💰 PROFIT & LOSS")
        print("-" * 70)
        print(f"Initial Capital:       ${params['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
        print(f"Total P&L:             ${metrics['total_pnl']:,.2f}")
        
        print("\n📈 TRADING STATISTICS")
        print("-" * 70)
        print(f"Total Trades:          {metrics['total_trades']}")
        print(f"Winning Trades:        {metrics['winning_trades']}")
        print(f"Losing Trades:         {metrics['losing_trades']}")
        print(f"Average Win:           ${metrics['avg_win']:,.2f}")
        print(f"Average Loss:          ${metrics['avg_loss']:,.2f}")
        print(f"Profit Factor:         {metrics['profit_factor']:.2f}")
        
        print("\n💸 COST ANALYSIS")
        print("-" * 70)
        print(f"Total Borrow Fees:     ${metrics.get('total_borrow_fees', 0):,.2f}")
        print(f"Total Slippage:        ${metrics.get('total_slippage', 0):,.2f}")
        print(f"Total Commission:      ${metrics.get('total_commission', 0):,.2f}")
        
        print("\n🎯 RISK METRICS")
        print("-" * 70)
        print(f"Volatility (Annual):   {metrics['volatility']:.2f}%")
        print(f"Sortino Ratio:         {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:          {metrics['calmar_ratio']:.2f}")
        print(f"Max Consecutive Loss:  {metrics.get('max_consecutive_losses', 0)}")
        
        if 'best_trade' in metrics and metrics['best_trade']:
            print("\n🏆 BEST TRADE")
            print("-" * 70)
            bt = metrics['best_trade']
            print(f"Ticker:                {bt['ticker']}")
            print(f"Entry Date:            {bt['entry_date']}")
            print(f"Exit Date:             {bt['exit_date']}")
            print(f"P&L:                   ${bt['pnl']:,.2f} ({bt['return_pct']:.2f}%)")
        
        if 'worst_trade' in metrics and metrics['worst_trade']:
            print("\n💔 WORST TRADE")
            print("-" * 70)
            wt = metrics['worst_trade']
            print(f"Ticker:                {wt['ticker']}")
            print(f"Entry Date:            {wt['entry_date']}")
            print(f"Exit Date:             {wt['exit_date']}")
            print(f"P&L:                   ${wt['pnl']:,.2f} ({wt['return_pct']:.2f}%)")
        
        print("\n" + "="*70)
    
    def save_results(self, results: Dict, metrics: Dict, params: Dict):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "backtest_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(results_dir, f"results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'parameters': {
                    'strategy': params['strategy_text'],
                    'start_date': params['start_date'].strftime("%Y-%m-%d"),
                    'end_date': params['end_date'].strftime("%Y-%m-%d"),
                    'initial_capital': params['initial_capital'],
                    'max_positions': params['max_positions'],
                    'position_size_pct': params['position_size_pct'],
                    'borrow_fee_annual': params['borrow_fee_annual']
                },
                'metrics': metrics,
                'trades': results.get('trades', [])
            }, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save trades to CSV
        if results.get('trades'):
            import csv
            csv_file = os.path.join(results_dir, f"trades_{timestamp}.csv")
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results['trades'][0].keys())
                writer.writeheader()
                writer.writerows(results['trades'])
            print(f"📊 Trade log saved to: {csv_file}")


def main():
    """Main entry