import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta


@dataclass
class PolygonConfig:
    """Polygon API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("POLYGON_API_KEY", ""))
    base_url: str = "https://api.polygon.io"
    max_requests_per_minute: int = 5
    request_delay: float = 0.2
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_dir: str = "./cache"
    cache_enabled: bool = True
    cache_expiry_days: int = 7


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365*5))
    end_date: datetime = field(default_factory=lambda: datetime.now())
    initial_capital: float = 100000.0
    position_size_pct: float = 0.10
    max_positions: int = 20
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_pct: float = 0.001
    
    # Short selling specific parameters
    short_borrow_fee_annual: float = 0.05
    hard_to_borrow_fee_annual: float = 0.25
    margin_requirement: float = 1.5
    margin_interest_rate_annual: float = 0.08
    
    # Market timing
    market_open_time: str = "09:30"
    market_close_time: str = "16:00"
    use_pre_market: bool = False
    entry_delay_seconds: int = 0
    
    # Risk management
    max_position_size: float = 50000.0
    min_position_size: float = 1000.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_daily_loss_pct: float = 0.05
    max_portfolio_loss_pct: float = 0.20
    
    # Data handling
    chunk_size_days: int = 90
    max_memory_mb: int = 2048
    
    # Survivor bias handling
    include_delisted: bool = True
    delisting_liquidation_pct: float = 0.90


@dataclass
class SmallCapCriteria:
    """Criteria for identifying small-cap stocks"""
    min_market_cap: float = 50_000_000
    max_market_cap: float = 2_000_000_000
    min_price: float = 1.0
    max_price: float = 50.0
    min_avg_volume: int = 100_000
    min_trading_days: int = 60
    exchanges: List[str] = field(default_factory=lambda: ["XNYS", "XNAS", "ARCX"])
    exclude_otc: bool = True


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    vwap_lookback_days: int = 1
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20
    
    # Look-ahead bias prevention
    use_prior_day_close: bool = True
    calculate_on_close: bool = True


@dataclass
class StrategyParserConfig:
    """Configuration for plain English strategy parser"""
    supported_indicators: List[str] = field(default_factory=lambda: [
        "vwap", "sma", "ema", "rsi", "volume", "price", "gap", 
        "bollinger", "atr", "market_cap", "avg_volume"
    ])
    supported_operators: List[str] = field(default_factory=lambda: [
        "above", "below", "greater than", "less than", "between", 
        "crosses above", "crosses below", "is", "equals"
    ])
    supported_conditions: List[str] = field(default_factory=lambda: [
        "and", "or"
    ])
    max_conditions: int = 10
    allow_complex_logic: bool = True


@dataclass
class LiquidityConfig:
    """Configuration for modeling market impact and liquidity"""
    enable_market_impact: bool = True
    
    # Market impact model parameters
    impact_coefficient: float = 0.1
    participation_rate: float = 0.05
    
    # Volume-based constraints
    max_volume_participation: float = 0.10
    min_avg_spread_bps: float = 5.0
    
    # Slippage tiers based on market cap
    slippage_tiers: Dict[str, float] = field(default_factory=lambda: {
        "micro": 0.005,
        "small": 0.002,
        "medium": 0.001
    })
    
    # Order execution
    order_execution_time_minutes: int = 5
    partial_fill_probability: float = 0.1


@dataclass
class ReportingConfig:
    """Configuration for results analysis and reporting"""
    output_dir: str = "./results"
    save_trades: bool = True
    save_daily_portfolio: bool = True
    save_metrics: bool = True
    generate_plots: bool = True
    
    # Metrics to calculate
    calculate_sharpe: bool = True
    calculate_sortino: bool = True
    calculate_max_drawdown: bool = True
    calculate_win_rate: bool = True
    
    # Visualization
    plot_equity_curve: bool = True
    plot_drawdown: bool = True
    plot_monthly_returns: bool = True
    plot_trade_distribution: bool = True
    
    # Risk-free rate for Sharpe/Sortino
    risk_free_rate_annual: float = 0.04


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_file: str = "./logs/backtest.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_console: bool = True
    log_to_file: bool = True


class Config:
    """Main configuration class"""
    def __init__(self):
        self.polygon = PolygonConfig()
        self.backtest = BacktestConfig()
        self.smallcap = SmallCapCriteria()
        self.indicators = IndicatorConfig()
        self.parser = StrategyParserConfig()
        self.liquidity = LiquidityConfig()
        self.reporting = ReportingConfig()
        self.logging = LoggingConfig()
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.polygon.api_key:
            errors.append("POLYGON_API_KEY not set")
        
        if self.backtest.start_date >= self.backtest.end_date:
            errors.append("start_date must be before end_date")
        
        if self.backtest.initial_capital <= 0:
            errors.append("initial_capital must be positive")
        
        if not (0 < self.backtest.position_size_pct <= 1.0):
            errors.append("position_size_pct must be between 0 and 1")
        
        if self.backtest.max_positions <= 0:
            errors.append("max_positions must be positive")
        
        if self.smallcap.min_market_cap >= self.smallcap.max_market_cap:
            errors.append("min_market_cap must be less than max_market_cap")
        
        if self.smallcap.min_price >= self.smallcap.max_price:
            errors.append("min_price must be less than max_price")
        
        if not (0 < self.liquidity.max_volume_participation <= 1.0):
            errors.append("max_volume_participation must be between 0 and 1")
        
        return errors
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        self.polygon.api_key = os.getenv("POLYGON_API_KEY", self.polygon.api_key)
        
        if os.getenv("INITIAL_CAPITAL"):
            self.backtest.initial_capital = float(os.getenv("INITIAL_CAPITAL"))
        
        if os.getenv("POSITION_SIZE_PCT"):
            self.backtest.position_size_pct = float(os.getenv("POSITION_SIZE_PCT"))
        
        if os.getenv("MAX_POSITIONS"):
            self.backtest.max_positions = int(os.getenv("MAX_POSITIONS"))
        
        if os.getenv("SHORT_BORROW_FEE"):
            self.backtest.short_borrow_fee_annual = float(os.getenv("SHORT_BORROW_FEE"))
        
        if os.getenv("SLIPPAGE_PCT"):
            self.backtest.slippage_pct = float(os.getenv("SLIPPAGE_PCT"))
        
        if os.getenv("START_DATE"):
            self.backtest.start_date = datetime.strptime(os.getenv("START_DATE"), "%Y-%m-%d")
        
        if os.getenv("END_DATE"):
            self.backtest.end_date = datetime.strptime(os.getenv("END_DATE"), "%Y-%m-%d")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            "polygon": self.polygon.__dict__,
            "backtest": {k: str(v) if isinstance(v, datetime) else v 
                        for k, v in self.backtest.__dict__.items()},
            "smallcap": self.smallcap.__dict__,
            "indicators": self.indicators.__dict__,
            "parser": self.parser.__dict__,
            "liquidity": self.liquidity.__dict__,
            "reporting": self.reporting.__dict__,
            "logging": self.logging.__dict__
        }


# Default configuration instance
config = Config()