import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .types import TradingSignal


class SignalStore:
    """Persists and retrieves trading signals with JSON storage."""
    
    def __init__(self, storage_path: str = "data/signals"):
        """Initialize SignalStore with storage path."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._signals: Dict[str, TradingSignal] = {}
        self._load_all()
    
    def _get_signal_path(self, symbol: str) -> Path:
        """Get file path for a symbol's signal."""
        return self.storage_path / f"{symbol.upper()}.json"
    
    def _load_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Load a single signal from disk."""
        path = self._get_signal_path(symbol)
        if not path.exists():
            return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return TradingSignal.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading signal for {symbol}: {e}")
            return None
    
    def _load_all(self) -> None:
        """Load all signals from disk."""
        self._signals = {}
        for path in self.storage_path.glob("*.json"):
            symbol = path.stem
            signal = self._load_signal(symbol)
            if signal:
                self._signals[symbol] = signal
    
    def _save_signal(self, signal: TradingSignal) -> None:
        """Save a signal to disk."""
        path = self._get_signal_path(signal.symbol)
        with open(path, 'w') as f:
            json.dump(signal.to_dict(), f, indent=2)
    
    def save(self, signal: TradingSignal) -> None:
        """Save a trading signal."""
        self._signals[signal.symbol.upper()] = signal
        self._save_signal(signal)
    
    def get(self, symbol: str) -> Optional[TradingSignal]:
        """Get a trading signal for a symbol."""
        return self._signals.get(symbol.upper())
    
    def get_all(self) -> List[TradingSignal]:
        """Get all trading signals."""
        return list(self._signals.values())
    
    def get_active(self) -> List[TradingSignal]:
        """Get all active (non-expired) trading signals."""
        return [s for s in self._signals.values() if not s.is_expired()]
    
    def get_open_positions(self) -> List[TradingSignal]:
        """Get all signals with open positions."""
        return [s for s in self._signals.values() if s.position_open]
    
    def delete(self, symbol: str) -> bool:
        """Delete a trading signal."""
        symbol = symbol.upper()
        if symbol not in self._signals:
            return False
        
        path = self._get_signal_path(symbol)
        if path.exists():
            path.unlink()
        
        del self._signals[symbol]
        return True
    
    def update_execution_state(
        self,
        symbol: str,
        position_open: bool = None,
        entry_executed_at: float = None,
        quantity: int = None,
        highest_price: float = None
    ) -> bool:
        """Update execution state for a signal."""
        signal = self.get(symbol)
        if not signal:
            return False
        
        if position_open is not None:
            signal.position_open = position_open
        if entry_executed_at is not None:
            signal.entry_executed_at = entry_executed_at
        if quantity is not None:
            signal.quantity = quantity
        if highest_price is not None:
            signal.highest_price_since_entry = highest_price
        
        self.save(signal)
        return True
    
    def get_by_cap_type(self, cap_type: str) -> List[TradingSignal]:
        """Get all signals for a specific cap type."""
        return [s for s in self._signals.values() if s.cap_type == cap_type]
    
    def get_signal_count(self) -> Dict[str, int]:
        """Get count of signals by status."""
        return {
            "total": len(self._signals),
            "active": len(self.get_active()),
            "open_positions": len(self.get_open_positions()),
            "expired": len(self._signals) - len(self.get_active())
        }
    
    def clear_expired(self) -> int:
        """Remove expired signals from storage."""
        expired = [s for s in self._signals.values() if s.is_expired()]
        for signal in expired:
            self.delete(signal.symbol)
        return len(expired)
    
    def get_summary(self) -> Dict:
        """Get a summary of all stored signals."""
        if not self._signals:
            return {
                "total_signals": 0,
                "by_cap_type": {},
                "by_signal_type": {},
                "open_positions": 0,
                "last_updated": None
            }
        
        by_cap = {}
        by_signal = {}
        open_positions = 0
        
        for signal in self._signals.values():
            by_cap[signal.cap_type] = by_cap.get(signal.cap_type, 0) + 1
            by_signal[signal.signal] = by_signal.get(signal.signal, 0) + 1
            if signal.position_open:
                open_positions += 1
        
        last_updated = max(
            (s.generated_at for s in self._signals.values()),
            default=None
        )
        
        return {
            "total_signals": len(self._signals),
            "by_cap_type": by_cap,
            "by_signal_type": by_signal,
            "open_positions": open_positions,
            "last_updated": last_updated.isoformat() if last_updated else None
        }
    
    def __len__(self) -> int:
        return len(self._signals)
    
    def __contains__(self, symbol: str) -> bool:
        return symbol.upper() in self._signals
