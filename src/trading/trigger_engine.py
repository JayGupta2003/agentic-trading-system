from calendar import monthrange
from datetime import datetime, timedelta
from typing import List, Optional

from .types import (
    AgentName,
    AGENT_ROUTING,
    TRIGGER_PRIORITY,
    MarketContext,
    TriggerConfig,
    TriggerResult,
    TriggerType,
)


def is_expiry_week(date: datetime) -> bool:
    """Check if given date is in the last week before monthly expiry.
    
    NSE monthly expiry is on the last Thursday of each month.
    Expiry week is the week containing that Thursday.
    
    Args:
        date: Date to check
        
    Returns:
        True if date is in expiry week
    """
    year = date.year
    month = date.month
    
    last_day = monthrange(year, month)[1]
    last_thursday = None
    
    for day in range(last_day, 0, -1):
        check_date = datetime(year, month, day)
        if check_date.weekday() == 3:
            last_thursday = check_date
            break
    
    if last_thursday is None:
        return False
    
    expiry_week_start = last_thursday - timedelta(days=6)
    
    return expiry_week_start <= date <= last_thursday + timedelta(days=1)


def get_last_trading_day_of_month(date: datetime) -> datetime:
    """Get the last trading day (Friday) of the month.
    
    Args:
        date: Any date in the target month
        
    Returns:
        Last trading day (Friday) of that month
    """
    year = date.year
    month = date.month
    
    last_day = monthrange(year, month)[1]
    
    for day in range(last_day, 0, -1):
        check_date = datetime(year, month, day)
        if check_date.weekday() <= 4:
            return check_date
    
    return datetime(year, month, last_day)


def is_scheduled_review_day(
    date: datetime,
    day_of_month: int = -1
) -> bool:
    """Check if given date is a scheduled review day.
    
    Args:
        date: Date to check
        day_of_month: Day of month for review (-1 = last day)
        
    Returns:
        True if this is a scheduled review day
    """
    if day_of_month == -1:
        last_trading_day = get_last_trading_day_of_month(date)
        return date.date() == last_trading_day.date()
    
    return date.day == day_of_month and date.weekday() <= 4


def check_vix_trigger(
    vix_history: List[float],
    current_vix: float,
    zscore_threshold: float = 2.0,
    window: int = 20
) -> bool:
    """Check if VIX has spiked based on Z-score.
    
    Trigger if VIX exceeds 20-day mean by 2 standard deviations.
    Formula: VIX > μ₂₀ + 2σ₂₀
    
    Args:
        vix_history: List of historical VIX values
        current_vix: Current VIX value
        zscore_threshold: Number of standard deviations (default 2.0)
        window: Rolling window size (default 20)
        
    Returns:
        True if VIX has spiked
    """
    if len(vix_history) < window:
        return False
    
    recent = vix_history[-window:]
    mean = sum(recent) / len(recent)
    
    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
    std = variance ** 0.5
    
    threshold = mean + (zscore_threshold * std)
    return current_vix > threshold


def check_price_drop_trigger(
    current_price: float,
    previous_price: float,
    cap_type: str,
    config: TriggerConfig
) -> bool:
    """Check if price dropped beyond threshold for cap type.
    
    Args:
        current_price: Current stock price
        previous_price: Previous day's closing price
        cap_type: 'large', 'mid', or 'small'
        config: Trigger configuration
        
    Returns:
        True if drop exceeds threshold
    """
    if previous_price <= 0:
        return False
    
    drop_pct = ((previous_price - current_price) / previous_price) * 100
    threshold = config.get_drop_threshold(cap_type)
    
    return drop_pct >= threshold


class TriggerEngine:
    """Evaluates market conditions and determines if LLM review is needed."""
    
    def __init__(self, config: Optional[TriggerConfig] = None):
        """Initialize TriggerEngine with configuration."""
        self.config = config or TriggerConfig()
    
    def evaluate(self, context: MarketContext) -> TriggerResult:
        """Evaluate all triggers and return the highest priority result.
        
        Args:
            context: Market context with current conditions
            
        Returns:
            TriggerResult with decision details
        """
        triggered_types = []
        
        if self._check_scheduled(context):
            triggered_types.append(TriggerType.SCHEDULED)
        
        if self._check_price_drop(context):
            triggered_types.append(TriggerType.PRICE_DROP)
        
        if self._check_vix_spike(context):
            triggered_types.append(TriggerType.VIX_SPIKE)
        
        if self._check_earnings(context):
            triggered_types.append(TriggerType.EARNINGS)
        
        if self._check_news_anomaly(context):
            triggered_types.append(TriggerType.NEWS_ANOMALY)
        
        if self._check_sector_collapse(context):
            triggered_types.append(TriggerType.SECTOR_COLLAPSE)
        
        if self._check_max_drawdown(context):
            triggered_types.append(TriggerType.MAX_DRAWDOWN)
        
        if self._check_expiry_week(context):
            triggered_types.append(TriggerType.EXPIRY_WEEK)
        
        if not triggered_types:
            return TriggerResult(
                should_run_agents=False,
                trigger_type=None,
                agents_to_run=[],
                trigger_details={},
                priority=0
            )
        
        highest_priority = max(
            triggered_types,
            key=lambda t: TRIGGER_PRIORITY.get(t, 0)
        )
        
        agents = AGENT_ROUTING.get(highest_priority, [AgentName.CIO])
        
        trigger_details = self._get_trigger_details(highest_priority, context)
        
        return TriggerResult(
            should_run_agents=True,
            trigger_type=highest_priority,
            agents_to_run=agents,
            trigger_details=trigger_details,
            priority=TRIGGER_PRIORITY.get(highest_priority, 0)
        )
    
    def _check_scheduled(self, context: MarketContext) -> bool:
        """Check if today is a scheduled review day."""
        return is_scheduled_review_day(
            context.current_date,
            self.config.scheduled_review_day
        )
    
    def _check_price_drop(self, context: MarketContext) -> bool:
        """Check if stock price dropped beyond threshold."""
        if context.stock_price is None or context.stock_previous_price is None:
            return False
        
        return check_price_drop_trigger(
            context.stock_price,
            context.stock_previous_price,
            context.stock_cap_type,
            self.config
        )
    
    def _check_vix_spike(self, context: MarketContext) -> bool:
        """Check if VIX has spiked based on Z-score."""
        if not context.vix_history or len(context.vix_history) < 20:
            return False
        
        return check_vix_trigger(
            context.vix_history,
            context.vix,
            self.config.vix_zscore_threshold,
            self.config.vix_history_window
        )
    
    def _check_earnings(self, context: MarketContext) -> bool:
        """Check if any holding has reported earnings."""
        return context.has_earnings
    
    def _check_news_anomaly(self, context: MarketContext) -> bool:
        """Check if there's significant negative news."""
        return context.has_news_anomaly and context.news_sentiment < -0.5
    
    def _check_sector_collapse(self, context: MarketContext) -> bool:
        """Check if any sector index collapsed."""
        for sector, change in context.sector_changes.items():
            if change <= -self.config.sector_collapse_threshold:
                return True
        return False
    
    def _check_max_drawdown(self, context: MarketContext) -> bool:
        """Check if portfolio drawdown exceeded threshold."""
        return context.portfolio_drawdown_pct >= self.config.max_drawdown_threshold
    
    def _check_expiry_week(self, context: MarketContext) -> bool:
        """Check if we're in expiry week."""
        return is_expiry_week(context.current_date)
    
    def _get_trigger_details(
        self,
        trigger_type: TriggerType,
        context: MarketContext
    ) -> dict:
        """Get detailed information about triggered condition."""
        details = {"trigger_type": trigger_type.value}
        
        if trigger_type == TriggerType.PRICE_DROP:
            if context.stock_price and context.stock_previous_price:
                drop_pct = ((context.stock_previous_price - context.stock_price) 
                           / context.stock_previous_price * 100)
                details["drop_pct"] = round(drop_pct, 2)
                details["threshold"] = self.config.get_drop_threshold(
                    context.stock_cap_type
                )
                details["cap_type"] = context.stock_cap_type
        
        elif trigger_type == TriggerType.VIX_SPIKE:
            details["current_vix"] = context.vix
            if len(context.vix_history) >= 20:
                recent = context.vix_history[-20:]
                details["vix_mean_20"] = round(sum(recent) / len(recent), 2)
                variance = sum((x - details["vix_mean_20"]) ** 2 
                              for x in recent) / len(recent)
                details["vix_std_20"] = round(variance ** 0.5, 2)
        
        elif trigger_type == TriggerType.MAX_DRAWDOWN:
            details["drawdown_pct"] = context.portfolio_drawdown_pct
            details["threshold"] = self.config.max_drawdown_threshold
        
        elif trigger_type == TriggerType.SECTOR_COLLAPSE:
            collapsed = {k: v for k, v in context.sector_changes.items() 
                        if v <= -self.config.sector_collapse_threshold}
            details["collapsed_sectors"] = collapsed
        
        elif trigger_type == TriggerType.NEWS_ANOMALY:
            details["sentiment"] = context.news_sentiment
        
        return details
    
    def get_agents_for_trigger(self, trigger_type: TriggerType) -> List[AgentName]:
        """Get list of agents to run for a specific trigger type."""
        return AGENT_ROUTING.get(trigger_type, [AgentName.CIO])
    
    def should_skip_agent(
        self,
        trigger_type: TriggerType,
        agent: AgentName
    ) -> bool:
        """Check if an agent should be skipped for a trigger type.
        
        Args:
            trigger_type: The trigger type
            agent: The agent to check
            
        Returns:
            True if agent should be skipped
        """
        agents_to_run = self.get_agents_for_trigger(trigger_type)
        return agent not in agents_to_run


def trigger_decision(context: MarketContext) -> TriggerResult:
    """Convenience function to evaluate triggers.
    
    Args:
        context: Market context with current conditions
        
    Returns:
        TriggerResult with decision
    """
    engine = TriggerEngine()
    return engine.evaluate(context)
