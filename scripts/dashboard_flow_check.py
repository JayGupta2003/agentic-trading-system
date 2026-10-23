"""
Test script for Today's Picks dashboard flow.
Manual end-to-end check (needs network access and a running Ollama). Run: python scripts/dashboard_flow_check.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_screening():
    """Test step 1: Screening"""
    print("\n=== TEST 1: SCREENING ===")
    from src.screening.screener import StockShortlister
    
    screener = StockShortlister()
    result = screener.run_screening(
        universe='NIFTY50',
        min_fscore=3,
        max_pe=50,
        min_momentum=-20,
        top_n=10,
        save_output=True,
    )
    
    stocks = result.get('stocks', [])
    print(f"Found {len(stocks)} stocks")
    if stocks:
        print(f"First 5: {[s['symbol'] for s in stocks[:5]]}")
    return stocks


def check_analysis(symbols):
    """Test step 2: Analysis"""
    print("\n=== TEST 2: ANALYSIS ===")
    from src.pipeline.orchestrator import PipelineOrchestrator
    from datetime import datetime, timedelta
    from src.trading.signal_store import SignalStore
    from src.trading.types import TradingSignal
    
    results = []
    orch = PipelineOrchestrator()
    
    for i, symbol in enumerate(symbols[:3]):  # Test with 3 stocks
        print(f"Analyzing {symbol}...")
        try:
            res = orch.analyze_single_stock(symbol)
            
            agent_outputs = res.get('agent_outputs', {})
            cio = agent_outputs.get('cio', {})
            
            verdict = cio.get('verdict', 'HOLD')
            conviction = cio.get('conviction_score', 5)
            thesis = cio.get('thesis', '')
            risk_params = cio.get('risk_params', {})
            
            print(f"  -> {verdict} (conviction: {conviction})")
            
            # Save signal
            signal = TradingSignal(
                symbol=symbol,
                signal=verdict,
                conviction=conviction,
                entry_price=risk_params.get('target', 0) or 0,
                stop_loss=risk_params.get('stop_loss', 0) or 0,
                target_price=risk_params.get('target', 0) or 0,
                thesis=thesis,
                generated_at=datetime.now(),
                valid_until=datetime.now(),
                review_type='ai_analysis',
                cap_type='large',
                agent_outputs=agent_outputs,
                full_cio_analysis=cio.get('analysis', ''),
            )
            
            store = SignalStore()
            store.save(signal)
            results.append({'symbol': symbol, 'verdict': verdict, 'status': 'success'})
            print(f"  -> Signal saved for {symbol}")
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({'symbol': symbol, 'verdict': 'ERROR', 'status': 'error', 'error': str(e)})
    
    print(f"Analyzed: {len(results)} stocks")
    return results


def check_signal_store():
    """Test step 3: Signal store"""
    print("\n=== TEST 3: SIGNAL STORE ===")
    from src.trading.signal_store import SignalStore
    
    store = SignalStore()
    
    print(f"All signals in store: {len(store.get_all())}")
    print(f"Active signals: {len(store.get_active())}")
    
    active = store.get_active()
    for s in active[:5]:
        print(f"  {s.symbol}: {s.signal} (conviction: {s.conviction})")
    
    return active


def main():
    print("Testing Today's Picks Dashboard Flow")
    print("=" * 50)
    
    # Step 1: Screening
    stocks = check_screening()
    if not stocks:
        print("No stocks passed screening!")
        return
    
    # Step 2: Analysis  
    symbols = [s['symbol'] for s in stocks]
    results = check_analysis(symbols)
    
    # Step 3: Check signal store
    active = check_signal_store()
    
    print("\n" + "=" * 50)
    print("DONE!")
    print(f"Active signals: {len(active)}")


if __name__ == "__main__":
    main()
