# Smallcap Short Backtesting System

A comprehensive backtesting framework for shorting smallcap stocks at market open using Polygon.io data. Supports plain-English strategy definitions with robust handling of look-ahead bias, survivor bias, and realistic short-selling costs.

## Features

- **Plain-English Strategy Parser**: Define strategies in natural language
- **Historical Data Fetching**: 5-year lookback using Polygon.io API with intelligent caching
- **Short-Selling Realism**: Includes borrow fees, margin requirements, and slippage modeling
- **Survivor Bias Mitigation**: Accounts for delisted stocks during backtest period
- **Look-Ahead Bias Prevention**: Ensures indicators use only data available at market open
- **Performance Analytics**: Comprehensive metrics including Sharpe ratio, max drawdown, win rate
- **Visualization**: Interactive charts for equity curves, drawdowns, and trade analysis

## Installation

### Prerequisites

- Python 3.8+
- Polygon.io API key (get one at https://polygon.io)

### Setup

1. Clone or download this project in Replit

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` and add your Polygon API key:
```
POLYGON_API_KEY=your_api_key_here
```

## Quick Start

```bash
python main.py
```

This will run a default strategy shorting smallcaps with negative VWAP momentum.

## Configuration

Edit `config.py` to customize:

### API Settings
```python
API_KEY = os.getenv('POLYGON_API_KEY')
API_RATE_LIMIT = 5  # requests per minute (adjust based on your plan)
CACHE_ENABLED = True  # Enable data caching to avoid re-fetching
CACHE_DIR = '.cache'  # Directory for cached data
```

### Backtest Parameters
```python
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'
INITIAL_CAPITAL = 100000
MAX_POSITIONS = 10  # Maximum concurrent short positions
POSITION_SIZE_PCT = 0.1  # 10% of portfolio per position
```

### Short Selling Costs
```python
BORROW_FEE_ANNUAL = 0.05  # 5% annual borrow fee (will vary by stock)
MARGIN_REQUIREMENT = 1.5  # 150% margin requirement for shorts
SLIPPAGE_BPS = 10  # 10 basis points slippage
```

### Smallcap Definition
```python
MIN_MARKET_CAP = 300_000_000  # $300M
MAX_MARKET_CAP = 2_000_000_000  # $2B
MIN_VOLUME = 100_000  # Minimum average daily volume
```

## Strategy Syntax

Define strategies in plain English using the strategy parser. The system converts natural language into executable rules.

### Basic Syntax

```
Short stocks where [CONDITION] and [CONDITION]
Exit when [CONDITION] or [CONDITION]
```

### Supported Conditions

#### Price-Based
- `price below $X`
- `price above $X`
- `price between $X and $Y`
- `close below open` (opening gap down)
- `close above open` (opening gap up)

#### Volume-Based
- `volume above X shares`
- `volume below average` (20-day average)
- `volume above average by X%`
- `low liquidity` (below MIN_VOLUME threshold)

#### VWAP Indicators
- `price below VWAP`
- `price above VWAP`
- `VWAP declining` (negative VWAP momentum)
- `VWAP rising` (positive VWAP momentum)
- `distance from VWAP greater than X%`

#### Technical Indicators
- `RSI below X` (14-period RSI)
- `RSI above X`
- `RSI oversold` (RSI < 30)
- `RSI overbought` (RSI > 70)
- `price below SMA(X)` (Simple Moving Average)
- `price above SMA(X)`
- `SMA(X) below SMA(Y)` (moving average crossover)
- `ATR above X` (Average True Range - volatility)

#### Market Cap & Fundamentals
- `market cap below $XM`
- `market cap above $XM`
- `is smallcap` (uses MIN/MAX_MARKET_CAP from config)
- `is microcap` (< $300M)

#### Time-Based
- `first hour of trading`
- `last hour of trading`
- `held for X days`
- `Monday` / `Tuesday` / etc.

#### Exit Conditions
- `profit above X%`
- `loss above X%`
- `price moved X%` (either direction)
- `days held equals X`
- `stop loss X%` (from entry price)
- `take profit X%` (from entry price)

### Example Strategies

#### Strategy 1: Negative VWAP Momentum
```
Short stocks where:
  - is smallcap
  - VWAP declining
  - volume above average by 50%
  - RSI below 45

Exit when:
  - profit above 10%
  - loss above 5%
  - held for 5 days
```

#### Strategy 2: Gap-Down Continuation
```
Short stocks where:
  - close below open by 3%
  - price below VWAP
  - market cap below $1000M
  - volume above 200000 shares

Exit when:
  - take profit 8%
  - stop loss 4%
  - held for 3 days
```

#### Strategy 3: Overbought Reversal
```
Short stocks where:
  - RSI overbought
  - price above SMA(20) by 15%
  - is smallcap
  - low liquidity

Exit when:
  - RSI below 50
  - profit above 12%
  - loss above 6%
```

#### Strategy 4: Momentum Exhaustion
```
Short stocks where:
  - price above SMA(50)
  - VWAP declining
  - ATR above 2.0
  - first hour of trading

Exit when:
  - price below SMA(50)
  - held for 7 days
```

### Strategy File Format

Save strategies as text files in the `strategies/` directory:

**strategies/my_strategy.txt**
```
Name: Smallcap VWAP Short
Description: Short smallcaps showing negative VWAP momentum with high volume

Entry Rules:
Short stocks where:
  - is smallcap
  - VWAP declining
  - volume above average by 50%
  - price below $50

Exit Rules:
Exit when:
  - profit above 10%
  - loss above 5%
  - held for 5 days
```

Load and run:
```python
from strategy_parser import StrategyParser

parser = StrategyParser()
strategy = parser.parse_file('strategies/my_strategy.txt')
```

## Usage Examples

### Example 1: Run Default Strategy

```python
python main.py
```

### Example 2: Custom Strategy via Code

```python
from backtester import Backtester
from strategy_parser import StrategyParser
import config

# Parse strategy
parser = StrategyParser()
strategy_text = """
Short stocks where:
  - is smallcap
  - VWAP declining
  - volume above average

Exit when:
  - profit above 10%
  - loss above 5%
"""
strategy = parser.parse(strategy_text)

# Run backtest
backtester = Backtester(
    strategy=strategy,
    start_date=config.START_DATE,
    end_date=config.END_DATE,
    initial_capital=config.INITIAL_CAPITAL
)

results = backtester.run()

# Analyze results
from results_analyzer import ResultsAnalyzer
analyzer = ResultsAnalyzer(results)
analyzer.print_summary()
analyzer.plot_equity_curve()
```

### Example 3: Backtest Multiple Strategies

```python
from pathlib import Path
from backtester import Backtester
from strategy_parser import StrategyParser

strategies_dir = Path('strategies')
results_dict = {}

for strategy_file in strategies_dir.glob('*.txt'):
    parser = StrategyParser()
    strategy = parser.parse_file(strategy_file)
    
    backtester = Backtester(strategy=strategy)
    results = backtester.run()
    
    results_dict[strategy.name] = results

# Compare strategies
for name, results in results_dict.items():
    print(f"{name}: Return={results.total_return:.2%}, Sharpe={results.sharpe_ratio:.2f}")
```

### Example 4: Fetch Historical Data Only

```python
from data_fetcher import DataFetcher
import config

fetcher = DataFetcher(api_key=config.API_KEY)

# Get smallcap universe
tickers = fetcher.get_smallcap_universe(
    min_market_cap=config.MIN_MARKET_CAP,
    max_market_cap=config.MAX_MARKET_CAP,
    date='2023-01-01'
)

# Fetch data for specific ticker
df = fetcher.fetch_daily_data(
    ticker='ABC',
    start_date='2022-01-01',
    end_date='2023-01-01'
)
```

## Data Caching

To minimize API calls and avoid rate limits, the system caches all fetched data:

- **Location**: `.cache/` directory
- **Format**: Parquet files (efficient storage)
- **Behavior**: Automatic cache check before API calls
- **Clear cache**: Delete `.cache/` directory or run:

```python
from data_fetcher import DataFetcher
fetcher = DataFetcher()
fetcher.clear_cache()
```

## API Rate Limiting

Polygon.io has rate limits based on your subscription tier:

- **Free**: 5 requests/minute
- **Starter**: 100 requests/minute
- **Developer**: 1000 requests/minute

The system automatically throttles requests based on `API_RATE_LIMIT` in config.py. Adjust this based on your plan.

## Handling Survivor Bias

The system accounts for delisted stocks by:

1. Fetching historical ticker lists from Polygon for each date
2. Including stocks that were listed during the backtest period but later delisted
3. Properly handling delisting events as forced exits

Enable survivor bias mitigation:
```python
config.SURVIVOR_BIAS_MITIGATION = True
```

## Look-Ahead Bias Prevention

To ensure realistic backtesting:

- **Market Open Timing**: Uses official market open prices (9:30 AM ET)
- **Indicator Calculation**: All indicators calculated using data available before trade entry
- **VWAP Calculation**: Uses previous day's VWAP for "VWAP declining" conditions
- **No Future Data**: Strict enforcement that decision at time T only uses data from T-1 and earlier

## Performance Metrics

The results analyzer provides:

### Return Metrics
- Total Return
- Annualized Return
- Daily/Monthly/Yearly Returns
- Alpha and Beta (vs SPY benchmark)

### Risk Metrics
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR (CVaR)

### Trade Metrics
- Total Trades
- Win Rate
- Average Win/Loss
- Profit Factor
- Average Hold Time

### Short-Specific Metrics
- Total Borrow Fees Paid
- Average Borrow Fee per Trade
- Margin Requirements Met
- Forced Liquidations

## Visualization

Generate charts:

```python
from visualizer import Visualizer

viz = Visualizer(results)

# Equity curve
viz.plot_equity_curve()

# Drawdown chart
viz.plot_drawdown()

# Monthly returns heatmap
viz.plot_monthly_returns()

# Trade distribution
viz.plot_trade_distribution()

# Save all charts
viz.save_all_charts('output/')
```

## Troubleshooting

### API Key Issues
**Error**: "Invalid API key"
**Solution**: Verify your key in `.env` matches your Polygon.io account

### Rate Limit Exceeded
**Error**: "Rate limit exceeded"
**Solution**: 
- Reduce `API_RATE_LIMIT` in config.py
- Enable caching with `CACHE_ENABLED = True`
- Upgrade your Polygon.io plan

### Memory Issues
**Error**: "MemoryError" or system slowdown
**Solution**:
- Reduce backtest date range
- Decrease `MAX_POSITIONS`
- Enable data chunking: `config.USE_CHUNKING = True`

### No Trades Generated
**Issue**: Backtest completes but shows 0 trades
**Solution**:
- Verify strategy conditions aren't too restrictive
- Check that tickers meet smallcap criteria during backtest period
- Ensure data is available for selected date range

### Inaccurate Results
**Issue**: Results seem unrealistic
**Solution**:
- Verify `SLIPPAGE_BPS` accounts for low liquidity
- Check `BORROW_FEE_ANNUAL` is realistic for smallcaps (typically 5-20%)
- Enable survivor bias mitigation
- Review trade logs for data quality issues

## Best Practices

1. **Start Small**: Test strategies on 1-2 years before running full 5-year backtest
2. **Use Caching**: Always enable caching to minimize API costs
3. **Realistic Costs**: Smallcap borrow fees can be 10-20%+, adjust accordingly
4. **Liquidity Constraints**: Set `MIN_VOLUME` to avoid illiquid stocks
5. **Position Sizing**: Keep `MAX_POSITIONS` reasonable (5-15) for smallcaps
6. **Validate Strategies**: Run paper trading before live implementation
7. **Monitor API Usage**: Check Polygon.io dashboard for rate limit status

## Advanced Configuration

### Custom Indicators

Add custom indicators in `indicators.py`:

```python
def custom_indicator(df, period=14):
    """
    Your custom indicator logic
    Returns: pandas Series with indicator values
    """
    return result

# Register in indicator registry
INDICATOR_REGISTRY['custom_indicator'] = custom_indicator
```

### Custom Exit Logic

Implement custom exit conditions in `portfolio.py`:

```python
def custom_exit_check(position, current_data):
    """
    Custom exit logic
    Returns: bool (True to exit)
    """
    # Your logic here
    return should_exit
```

### Slippage Models

Customize slippage in `config.py`:

```python
def custom_slippage(price, volume, shares):
    """
    Model slippage based on order size vs volume
    """
    impact = (shares / volume) * 0.1  # 10% impact factor
    return price * (1 + impact)

config.SLIPPAGE_MODEL = custom_slippage
```

## Project Structure

```
.
├── main.py                 # Entry point
├── config.py              # Configuration settings
├── data_fetcher.py        # Polygon API integration
├── indicators.py          # Technical indicators
├── strategy_parser.py     # Plain English parser
├── backtester.py          # Backtesting engine
├── portfolio.py           # Position management
├── results_analyzer.py    # Performance metrics
├── visualizer.py          # Charting and plots
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── .replit                # Replit configuration
└── README.md              # This file
```

## Support & Contributing

For issues or questions:
1. Check troubleshooting section above
2. Review Polygon.io API documentation
3. Verify configuration settings match your use case

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This backtesting system is for educational purposes only. Past performance does not guarantee future results. Short selling involves significant risk including the potential for