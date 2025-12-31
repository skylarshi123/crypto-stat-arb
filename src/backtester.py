"""
Backtesting engine for cryptocurrency statistical arbitrage strategy.
Tests multiple holding periods with realistic transaction costs.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatArbBacktester:
    """
    Backtester for statistical arbitrage mean-reversion strategy.
    
    Features:
    - Multiple holding periods (4h, 8h, 12h, 24h)
    - Realistic transaction costs (20 bps per trade)
    - Buy-and-hold benchmark comparison
    - Long-only positions with no overlaps
    """
    
    def __init__(self, signals_file: str = 'results/signals.csv', 
                 results_dir: str = 'results'):
        """Initialize backtester with signals and results paths."""
        self.signals_file = signals_file
        self.results_dir = results_dir
        
        # Trading parameters (as per PROJECT_SPEC.md)
        self.transaction_cost_bps = 20  # 20 basis points per trade (0.20%)
        self.round_trip_cost = 2 * self.transaction_cost_bps / 10000  # Total cost: 40 bps = 0.40%
        
        # Test holding periods (in hours)
        self.holding_periods = [4, 8, 12, 24]
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_signals(self) -> pd.DataFrame:
        """
        Load signals from CSV file and prepare for backtesting.
        
        Returns:
            DataFrame with signals and market data
        """
        if not os.path.exists(self.signals_file):
            raise FileNotFoundError(f"Signals file not found: {self.signals_file}")
        
        df = pd.read_csv(self.signals_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        logger.info(f"Loaded signals: {len(df)} records")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Total signals: {df['signal'].sum()}")
        
        return df
    
    def get_exit_price(self, df: pd.DataFrame, entry_time: pd.Timestamp, 
                      symbol: str, holding_hours: int) -> float:
        """
        Get exit price for a trade after specified holding period.
        
        Args:
            df: Complete market data
            entry_time: When position was entered
            symbol: Trading symbol
            holding_hours: How long to hold position
            
        Returns:
            Exit price, or None if no data available
        """
        exit_time = entry_time + pd.Timedelta(hours=holding_hours)
        
        # Filter for the specific symbol and time
        symbol_data = df[df['symbol'] == symbol]
        
        # Find the closest available exit time
        available_times = symbol_data.index[symbol_data.index >= exit_time]
        
        if len(available_times) == 0:
            return None  # No data available for exit
        
        actual_exit_time = available_times[0]
        exit_price = symbol_data.loc[actual_exit_time, 'close']
        
        return exit_price
    
    def backtest_holding_period(self, df: pd.DataFrame, holding_hours: int) -> Dict:
        """
        Backtest strategy for a specific holding period.
        
        Args:
            df: Complete market data with signals
            holding_hours: Hours to hold each position
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Backtesting {holding_hours}-hour holding period...")
        
        # Find all signals
        signals = df[df['signal'] == 1].copy()
        trades = []
        
        # Track when we're in positions to avoid overlaps
        active_positions = {}  # symbol -> exit_time
        
        for signal_time, signal_row in signals.iterrows():
            symbol = signal_row['symbol']
            entry_price = signal_row['close']
            
            # Check if we already have an active position for this symbol
            if symbol in active_positions:
                if signal_time < active_positions[symbol]:
                    continue  # Skip this signal - still in position
                else:
                    del active_positions[symbol]  # Position expired
            
            # Get exit price
            exit_price = self.get_exit_price(df, signal_time, symbol, holding_hours)
            
            if exit_price is None:
                continue  # Can't exit - not enough data
            
            # Calculate trade return
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - self.round_trip_cost
            
            trade = {
                'entry_time': signal_time,
                'exit_time': signal_time + pd.Timedelta(hours=holding_hours),
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': gross_return,
                'net_return': net_return,
                'holding_hours': holding_hours
            }
            
            trades.append(trade)
            
            # Mark this symbol as having an active position
            active_positions[symbol] = signal_time + pd.Timedelta(hours=holding_hours)
        
        # Calculate aggregate statistics
        if not trades:
            return {
                'holding_period': holding_hours,
                'num_trades': 0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'net_return': 0.0,
                'trades': []
            }
        
        trades_df = pd.DataFrame(trades)
        
        num_trades = len(trades_df)
        total_gross_return = trades_df['gross_return'].sum()
        total_net_return = trades_df['net_return'].sum()
        win_rate = (trades_df['net_return'] > 0).mean()
        avg_gross_return = trades_df['gross_return'].mean()
        avg_net_return = trades_df['net_return'].mean()
        
        results = {
            'holding_period': holding_hours,
            'num_trades': num_trades,
            'total_gross_return': total_gross_return,
            'total_net_return': total_net_return,
            'win_rate': win_rate,
            'avg_gross_return': avg_gross_return,
            'avg_net_return': avg_net_return,
            'trades': trades
        }
        
        logger.info(f"{holding_hours}h: {num_trades} trades, "
                   f"{win_rate:.1%} win rate, "
                   f"{avg_net_return:.2%} avg return")
        
        return results
    
    def calculate_buy_hold_benchmark(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate buy-and-hold returns for each symbol over the entire period.
        
        Args:
            df: Complete market data
            
        Returns:
            Dictionary mapping symbols to buy-hold returns
        """
        benchmark_returns = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_index()
            
            if len(symbol_data) < 2:
                continue
            
            start_price = symbol_data['close'].iloc[0]
            end_price = symbol_data['close'].iloc[-1]
            
            buy_hold_return = (end_price - start_price) / start_price
            benchmark_returns[symbol] = buy_hold_return
            
            logger.info(f"{symbol} buy-hold: {buy_hold_return:.2%}")
        
        return benchmark_returns
    
    def run_backtest(self) -> pd.DataFrame:
        """
        Run complete backtest across all holding periods.
        
        Returns:
            DataFrame with results for each holding period
        """
        logger.info("Starting backtesting...")
        
        # Load market data and signals
        df = self.load_signals()
        
        # Calculate buy-and-hold benchmark
        logger.info("\nCalculating buy-and-hold benchmark...")
        benchmark_returns = self.calculate_buy_hold_benchmark(df)
        avg_benchmark = np.mean(list(benchmark_returns.values()))
        logger.info(f"Average buy-hold return: {avg_benchmark:.2%}")
        
        # Test each holding period
        logger.info("\nTesting holding periods...")
        results = []
        
        for holding_hours in self.holding_periods:
            period_results = self.backtest_holding_period(df, holding_hours)
            results.append(period_results)
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'holding_period': result['holding_period'],
                'num_trades': result['num_trades'],
                'total_gross_return': result['total_gross_return'],
                'total_net_return': result['total_net_return'],
                'win_rate': result['win_rate'],
                'avg_gross_return': result['avg_gross_return'],
                'avg_net_return': result['avg_net_return'],
                'benchmark_return': avg_benchmark
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df, results, benchmark_returns
    
    def print_results_summary(self, summary_df: pd.DataFrame, 
                            benchmark_returns: Dict[str, float]) -> None:
        """Print formatted results summary."""
        logger.info("=" * 80)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        # Strategy results
        logger.info("STRATEGY PERFORMANCE BY HOLDING PERIOD:")
        logger.info("-" * 80)
        logger.info(f"{'Period':<8} {'Trades':<8} {'Win Rate':<10} {'Avg Return':<12} "
                   f"{'Net Return':<12} {'vs B&H':<10}")
        logger.info("-" * 80)
        
        avg_benchmark = np.mean(list(benchmark_returns.values()))
        
        for _, row in summary_df.iterrows():
            period = f"{row['holding_period']}h"
            trades = f"{row['num_trades']}"
            win_rate = f"{row['win_rate']:.1%}"
            avg_return = f"{row['avg_net_return']:.2%}"
            total_return = f"{row['total_net_return']:.2%}"
            vs_benchmark = f"{row['total_net_return'] - avg_benchmark:+.2%}"
            
            logger.info(f"{period:<8} {trades:<8} {win_rate:<10} {avg_return:<12} "
                       f"{total_return:<12} {vs_benchmark:<10}")
        
        # Benchmark comparison
        logger.info("-" * 80)
        logger.info("BUY-AND-HOLD BENCHMARK:")
        for symbol, return_val in benchmark_returns.items():
            logger.info(f"{symbol}: {return_val:+.2%}")
        logger.info(f"Average: {avg_benchmark:+.2%}")
        
        # Best performing period
        if len(summary_df) > 0:
            best_period = summary_df.loc[summary_df['total_net_return'].idxmax()]
            logger.info("-" * 80)
            logger.info("BEST PERFORMING PERIOD:")
            logger.info(f"✅ {best_period['holding_period']} hours: "
                       f"{best_period['total_net_return']:.2%} total return "
                       f"({best_period['num_trades']} trades, "
                       f"{best_period['win_rate']:.1%} win rate)")
        
        logger.info("=" * 80)
    
    def save_results(self, summary_df: pd.DataFrame, detailed_results: List[Dict]) -> str:
        """Save backtest results to CSV files."""
        
        # Save summary results
        summary_path = os.path.join(self.results_dir, 'backtest_results.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Save detailed trade-by-trade results
        all_trades = []
        for period_result in detailed_results:
            all_trades.extend(period_result['trades'])
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_path = os.path.join(self.results_dir, 'detailed_trades.csv')
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Detailed trades saved to {trades_path}")
        
        logger.info(f"Summary results saved to {summary_path}")
        return summary_path


def main():
    """Main function to run the backtest."""
    try:
        # Initialize backtester
        backtester = StatArbBacktester()
        
        # Run complete backtest
        summary_df, detailed_results, benchmark_returns = backtester.run_backtest()
        
        # Print results
        backtester.print_results_summary(summary_df, benchmark_returns)
        
        # Save results
        backtester.save_results(summary_df, detailed_results)
        
        logger.info("✅ Backtesting complete!")
        
        # Quick assessment for resume
        if len(summary_df) > 0:
            best_result = summary_df.loc[summary_df['total_net_return'].idxmax()]
            total_trades = summary_df['num_trades'].sum()
            
            logger.info("\n🎯 RESUME METRICS:")
            logger.info(f"Best holding period: {best_result['holding_period']} hours")
            logger.info(f"Best total return: {best_result['total_net_return']:.2%}")
            logger.info(f"Best win rate: {best_result['win_rate']:.1%}")
            logger.info(f"Total trades executed: {total_trades}")
            
            if best_result['total_net_return'] > 0.05:  # 5% threshold
                logger.info("✅ Strategy shows promise for resume!")
            else:
                logger.info("⚠️  Consider parameter tuning for better performance")
                
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        raise


if __name__ == "__main__":
    main()