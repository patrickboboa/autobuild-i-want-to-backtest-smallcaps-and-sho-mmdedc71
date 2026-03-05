import re
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class ComparisonOperator(Enum):
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class LogicalOperator(Enum):
    AND = "and"
    OR = "or"


@dataclass
class Condition:
    left_operand: str
    operator: ComparisonOperator
    right_operand: str
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        left_value = self._resolve_operand(self.left_operand, context)
        right_value = self._resolve_operand(self.right_operand, context)
        
        if left_value is None or right_value is None:
            return False
            
        try:
            if self.operator == ComparisonOperator.GREATER_THAN:
                return float(left_value) > float(right_value)
            elif self.operator == ComparisonOperator.LESS_THAN:
                return float(left_value) < float(right_value)
            elif self.operator == ComparisonOperator.GREATER_EQUAL:
                return float(left_value) >= float(right_value)
            elif self.operator == ComparisonOperator.LESS_EQUAL:
                return float(left_value) <= float(right_value)
            elif self.operator == ComparisonOperator.EQUAL:
                return float(left_value) == float(right_value)
            elif self.operator == ComparisonOperator.NOT_EQUAL:
                return float(left_value) != float(right_value)
        except (ValueError, TypeError):
            return False
        
        return False
    
    def _resolve_operand(self, operand: str, context: Dict[str, Any]) -> Optional[float]:
        operand = operand.strip()
        
        # Check if it's a numeric literal
        try:
            return float(operand)
        except ValueError:
            pass
        
        # Check if it's a percentage
        if operand.endswith('%'):
            try:
                return float(operand[:-1]) / 100.0
            except ValueError:
                pass
        
        # Check if it's a variable in context
        if operand in context:
            value = context[operand]
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, pd.Series) and len(value) > 0:
                return float(value.iloc[-1])
        
        # Check for nested attributes (e.g., "price.close")
        if '.' in operand:
            parts = operand.split('.')
            value = context
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
            
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, pd.Series) and len(value) > 0:
                return float(value.iloc[-1])
        
        return None


@dataclass
class Rule:
    conditions: List[Condition]
    logical_operators: List[LogicalOperator]
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        if not self.conditions:
            return False
        
        if len(self.conditions) == 1:
            return self.conditions[0].evaluate(context)
        
        result = self.conditions[0].evaluate(context)
        
        for i, operator in enumerate(self.logical_operators):
            if i + 1 >= len(self.conditions):
                break
            
            next_result = self.conditions[i + 1].evaluate(context)
            
            if operator == LogicalOperator.AND:
                result = result and next_result
            elif operator == LogicalOperator.OR:
                result = result or next_result
        
        return result


class StrategyParser:
    def __init__(self):
        self.entry_rules: List[Rule] = []
        self.exit_rules: List[Rule] = []
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.time_stop: Optional[int] = None
        
        # Indicator aliases
        self.indicator_aliases = {
            'rsi': 'RSI',
            'macd': 'MACD',
            'signal': 'MACD_signal',
            'histogram': 'MACD_histogram',
            'sma': 'SMA',
            'ema': 'EMA',
            'bollinger_upper': 'BB_upper',
            'bollinger_lower': 'BB_lower',
            'bollinger_middle': 'BB_middle',
            'bb_upper': 'BB_upper',
            'bb_lower': 'BB_lower',
            'bb_middle': 'BB_middle',
            'atr': 'ATR',
            'volume': 'volume',
            'price': 'close',
            'close': 'close',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'vwap': 'VWAP',
            'adx': 'ADX',
            'stochastic': 'Stochastic_K',
            'stoch_k': 'Stochastic_K',
            'stoch_d': 'Stochastic_D',
        }
        
    def parse(self, strategy_text: str) -> None:
        """Parse plain English strategy description into executable rules."""
        self.entry_rules = []
        self.exit_rules = []
        self.stop_loss = None
        self.take_profit = None
        self.time_stop = None
        
        lines = strategy_text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            line_lower = line.lower()
            
            # Detect sections
            if 'entry' in line_lower and ':' in line:
                current_section = 'entry'
                # Check if condition is on the same line
                if ':' in line:
                    condition_part = line.split(':', 1)[1].strip()
                    if condition_part:
                        rule = self._parse_rule(condition_part)
                        if rule:
                            self.entry_rules.append(rule)
                continue
            elif 'exit' in line_lower and ':' in line:
                current_section = 'exit'
                if ':' in line:
                    condition_part = line.split(':', 1)[1].strip()
                    if condition_part:
                        rule = self._parse_rule(condition_part)
                        if rule:
                            self.exit_rules.append(rule)
                continue
            elif 'stop loss' in line_lower or 'stoploss' in line_lower:
                self.stop_loss = self._parse_percentage(line)
                continue
            elif 'take profit' in line_lower or 'takeprofit' in line_lower:
                self.take_profit = self._parse_percentage(line)
                continue
            elif 'time stop' in line_lower or 'timestop' in line_lower:
                self.time_stop = self._parse_days(line)
                continue
            
            # Parse conditions based on current section
            if current_section == 'entry':
                rule = self._parse_rule(line)
                if rule:
                    self.entry_rules.append(rule)
            elif current_section == 'exit':
                rule = self._parse_rule(line)
                if rule:
                    self.exit_rules.append(rule)
    
    def _parse_rule(self, text: str) -> Optional[Rule]:
        """Parse a single rule from text."""
        text = text.strip()
        if not text:
            return None
        
        # Split by logical operators while preserving them
        conditions = []
        logical_operators = []
        
        # Replace common phrases
        text = self._normalize_text(text)
        
        # Split by 'and' and 'or' while keeping track of operators
        parts = re.split(r'\s+(and|or)\s+', text, flags=re.IGNORECASE)
        
        for i, part in enumerate(parts):
            part = part.strip()
            if part.lower() == 'and':
                logical_operators.append(LogicalOperator.AND)
            elif part.lower() == 'or':
                logical_operators.append(LogicalOperator.OR)
            else:
                condition = self._parse_condition(part)
                if condition:
                    conditions.append(condition)
        
        if not conditions:
            return None
        
        return Rule(conditions=conditions, logical_operators=logical_operators)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize common phrases to standard format."""
        text = text.lower()
        
        # Replace common phrases
        replacements = {
            'is greater than': '>',
            'is less than': '<',
            'greater than': '>',
            'less than': '<',
            'is above': '>',
            'is below': '<',
            'above': '>',
            'below': '<',
            'exceeds': '>',
            'falls below': '<',
            'drops below': '<',
            'rises above': '>',
            'crosses above': '>',
            'crosses below': '<',
            'is equal to': '==',
            'equals': '==',
            'not equal to': '!=',
            '>=': '>=',
            '<=': '<=',
            'at least': '>=',
            'at most': '<=',
            'no more than': '<=',
            'no less than': '>=',
        }
        
        for phrase, symbol in replacements.items():
            text = text.replace(phrase, symbol)
        
        return text
    
    def _parse_condition(self, text: str) -> Optional[Condition]:
        """Parse a single condition from text."""
        text = text.strip()
        
        # Try to find comparison operators
        operators = ['>=', '<=', '!=', '==', '>', '<']
        
        for op_str in operators:
            if op_str in text:
                parts = text.split(op_str, 1)
                if len(parts) == 2:
                    left = self._normalize_operand(parts[0].strip())
                    right = self._normalize_operand(parts[1].strip())
                    
                    operator_map = {
                        '>': ComparisonOperator.GREATER_THAN,
                        '<': ComparisonOperator.LESS_THAN,
                        '>=': ComparisonOperator.GREATER_EQUAL,
                        '<=': ComparisonOperator.LESS_EQUAL,
                        '==': ComparisonOperator.EQUAL,
                        '!=': ComparisonOperator.NOT_EQUAL,
                    }
                    
                    return Condition(
                        left_operand=left,
                        operator=operator_map[op_str],
                        right_operand=right
                    )
        
        return None
    
    def _normalize_operand(self, operand: str) -> str:
        """Normalize an operand to a standard form."""
        operand = operand.strip().lower()
        
        # Remove common words
        operand = operand.replace('the ', '')
        operand = operand.replace('current ', '')
        operand = operand.replace('stock ', '')
        
        # Map to indicator aliases
        for alias, indicator in self.indicator_aliases.items():
            if alias in operand:
                # Handle period specifications like "RSI(14)" or "SMA_20"
                period_match = re.search(r'\((\d+)\)', operand)
                if period_match:
                    period = period_match.group(1)
                    return f"{indicator}_{period}"
                
                period_match = re.search(r'_(\d+)', operand)
                if period_match:
                    period = period_match.group(1)
                    return f"{indicator}_{period}"
                
                # Check if operand is exactly the alias
                if operand.replace('_', '').replace('(', '').replace(')', '') == alias.replace('_', ''):
                    return indicator
        
        # Return as-is if no mapping found
        return operand
    
    def _parse_percentage(self, text: str) -> Optional[float]:
        """Extract percentage value from text."""
        # Look for patterns like "5%", "5 percent", "0.05"
        match = re.search(r'(\d+\.?\d*)\s*%', text)
        if match:
            return float(match.group(1)) / 100.0
        
        match = re.search(r'(\d+\.?\d*)\s*percent', text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
        
        # Look for decimal values
        match = re.search(r'(\d*\.\d+)', text)
        if match:
            value = float(match.group(1))
            if value < 1:  # Assume it's already in decimal form
                return value
            else:  # Assume it's percentage
                return value / 100.0
        
        return None
    
    def _parse_days(self, text: str) -> Optional[int]:
        """Extract number of days from text."""
        match = re.search(r'(\d+)\s*(day|days|d)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        match = re.search(r'(\d+)', text)
        if match:
            return int(match.group(1))
        
        return None
    
    def check_entry(self, context: Dict[str, Any]) -> bool:
        """Check if entry conditions are met."""
        if not self.entry_rules:
            return False
        
        # All entry rules must be satisfied (AND logic between rules)
        for rule in self.entry_rules:
            if not rule.evaluate(context):
                return False
        
        return True
    
    def check_exit(self, context: Dict[str, Any], entry_price: float, days_held: int) -> Tuple[bool, str]:
        """Check if exit conditions are met."""
        current_price = context.get('close', 0)
        
        if current_price == 0 or entry_price == 0:
            return False, ""
        
        # For short positions, profit is when price goes down
        pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss (for shorts, this triggers when price goes up)
        if self.stop_loss is not None:
            if pnl_pct < -self.stop_loss:
                return True, "stop_loss"
        
        # Check take profit (for shorts, this triggers when price goes down)
        if self.take_profit is not None:
            if pnl_pct >= self.take_profit:
                return True, "take_profit"
        
        # Check time stop
        if self.time_stop is not None:
            if days_held >= self.time_stop:
                return True, "time_stop"
        
        # Check custom exit rules
        if self