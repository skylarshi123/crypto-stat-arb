"""
Visualization generator for cryptocurrency statistical arbitrage strategy.
Creates professional charts for strategy performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategyVisualizer:
    """
    Create professional visualizations for trading strategy performance.
    
    Charts generated:
    - Equity curve (cumulative returns over time)
    - Drawdown chart (peak-to-trough declines)
    - Returns distribution (histogram of trade returns)
    """
    
    def __init__(self, trades_file: str = 'results/detailed_trades.csv',
                 results_dir: str = 'results'):
        """Initialize visualizer with data paths."""
        self.trades_file = trades_file
        self.results_dir = results_dir
        self.best_holding_period = 24  # Hours - our best performing period
        
        # Set professional styling
        plt.style.use('seaborn-v0_8')  # Clean, professional style
        sns.set_palette("husl")  # Nice color palette
        
        # Chart styling parameters
        self.chart_params = {
            'figure_size': (12, 8),
            'dpi': 300,
            'grid_alpha': 0.3,
            'title_size': 16,
            'label_size': 12,
            'tick_size': 10
        }
    
    def load_trades_data(self) -> pd.DataFrame:
        """
        Load and prepare trades data for visualization.
        
        Returns:
            DataFrame with trades for the best holding period
        """
        if not os.path.exists(self.trades_file):
            raise FileNotFoundError(f"Trades file not found: {self.trades_file}")
        
        # Load all trades
        trades_df = pd.read_csv(self.trades_file)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Filter for best holding period and sort by entry time
        best_trades = trades_df[trades_df['holding_hours'] == self.best_holding_period].copy()
        best_trades = best_trades.sort_values('entry_time')
        
        logger.info(f"Loaded {len(best_trades)} trades for {self.best_holding_period}-hour holding period")
        
        return best_trades
    
    def create_equity_curve(self, trades_df: pd.DataFrame) -> str:
        """
        Create equity curve showing cumulative returns over time.
        
        Args:
            trades_df: DataFrame with trade results
            
        Returns:
            Path to saved chart
        """
        logger.info("Creating equity curve chart...")
        
        # Calculate cumulative returns
        trades_df['cumulative_return'] = (1 + trades_df['net_return']).cumprod() - 1
        trades_df['cumulative_value'] = 1 + trades_df['cumulative_return']  # Starting with $1
        
        # Create the chart
        fig, ax = plt.subplots(figsize=self.chart_params['figure_size'], 
                              dpi=self.chart_params['dpi'])
        
        # Plot equity curve
        ax.plot(trades_df['entry_time'], trades_df['cumulative_value'], 
                color='#2E8B57', linewidth=2.5, marker='o', markersize=4,
                label=f'Strategy ({self.best_holding_period}h holding)')
        
        # Add horizontal line at starting value
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Initial Value')
        
        # Styling
        ax.set_title('Cryptocurrency Statistical Arbitrage - Equity Curve', 
                    fontsize=self.chart_params['title_size'], fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=self.chart_params['label_size'])
        ax.set_ylabel('Portfolio Value (Starting = $1.00)', fontsize=self.chart_params['label_size'])
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add grid and legend with better positioning
        ax.grid(True, alpha=self.chart_params['grid_alpha'])
        ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
        
        # Add performance annotations in upper right to avoid overlap
        final_value = trades_df['cumulative_value'].iloc[-1]
        total_return = (final_value - 1) * 100
        
        ax.text(0.98, 0.98, f'Total Return: {total_return:.1f}%\nFinal Value: ${final_value:.3f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Tight layout and save
        plt.tight_layout()
        filepath = os.path.join(self.results_dir, 'equity_curve.png')
        plt.savefig(filepath, dpi=self.chart_params['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Equity curve saved to {filepath}")
        return filepath
    
    def create_drawdown_chart(self, trades_df: pd.DataFrame) -> str:
        """
        Create drawdown chart showing peak-to-trough declines.
        
        Args:
            trades_df: DataFrame with trade results
            
        Returns:
            Path to saved chart
        """
        logger.info("Creating drawdown chart...")
        
        # Calculate cumulative returns and drawdowns
        trades_df['cumulative_return'] = (1 + trades_df['net_return']).cumprod() - 1
        trades_df['running_max'] = trades_df['cumulative_return'].cummax()
        trades_df['drawdown'] = trades_df['cumulative_return'] - trades_df['running_max']
        trades_df['drawdown_pct'] = trades_df['drawdown'] * 100
        
        # Create the chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.chart_params['figure_size'], 
                                       dpi=self.chart_params['dpi'], 
                                       height_ratios=[2, 1], sharex=True)
        
        # Top plot: Cumulative returns with running maximum
        ax1.plot(trades_df['entry_time'], trades_df['cumulative_return'] * 100, 
                color='#2E8B57', linewidth=2, label='Cumulative Return')
        ax1.plot(trades_df['entry_time'], trades_df['running_max'] * 100, 
                color='#4169E1', linewidth=2, linestyle='--', alpha=0.8, label='Running Maximum')
        
        ax1.set_title('Strategy Performance with Drawdown Analysis', 
                     fontsize=self.chart_params['title_size'], fontweight='bold')
        ax1.set_ylabel('Return (%)', fontsize=self.chart_params['label_size'])
        ax1.grid(True, alpha=self.chart_params['grid_alpha'])
        ax1.legend(loc='upper left')
        
        # Bottom plot: Drawdown
        ax2.fill_between(trades_df['entry_time'], trades_df['drawdown_pct'], 0, 
                        color='#DC143C', alpha=0.7, label='Drawdown')
        ax2.plot(trades_df['entry_time'], trades_df['drawdown_pct'], 
                color='#8B0000', linewidth=1.5)
        
        # Find and annotate maximum drawdown
        max_dd_idx = trades_df['drawdown_pct'].idxmin()
        max_dd_date = trades_df.loc[max_dd_idx, 'entry_time']
        max_dd_value = trades_df.loc[max_dd_idx, 'drawdown_pct']
        
        ax2.annotate(f'Max DD: {max_dd_value:.1f}%', 
                    xy=(max_dd_date, max_dd_value),
                    xytext=(max_dd_date, max_dd_value - 2),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, ha='center')
        
        ax2.set_xlabel('Date', fontsize=self.chart_params['label_size'])
        ax2.set_ylabel('Drawdown (%)', fontsize=self.chart_params['label_size'])
        ax2.grid(True, alpha=self.chart_params['grid_alpha'])
        ax2.legend(loc='lower right')
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Tight layout and save
        plt.tight_layout()
        filepath = os.path.join(self.results_dir, 'drawdown_chart.png')
        plt.savefig(filepath, dpi=self.chart_params['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Drawdown chart saved to {filepath}")
        return filepath
    
    def create_returns_distribution(self, trades_df: pd.DataFrame) -> str:
        """
        Create histogram of individual trade returns.
        
        Args:
            trades_df: DataFrame with trade results
            
        Returns:
            Path to saved chart
        """
        logger.info("Creating returns distribution chart...")
        
        # Convert returns to percentages
        returns_pct = trades_df['net_return'] * 100
        
        # Create the chart
        fig, ax = plt.subplots(figsize=self.chart_params['figure_size'], 
                              dpi=self.chart_params['dpi'])
        
        # Create histogram with different colors for gains/losses
        positive_returns = returns_pct[returns_pct >= 0]
        negative_returns = returns_pct[returns_pct < 0]
        
        # Plot histograms
        ax.hist(positive_returns, bins=8, alpha=0.7, color='#2E8B57', 
                label=f'Profitable Trades ({len(positive_returns)})', edgecolor='black')
        ax.hist(negative_returns, bins=6, alpha=0.7, color='#DC143C', 
                label=f'Losing Trades ({len(negative_returns)})', edgecolor='black')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Add mean and median lines
        mean_return = returns_pct.mean()
        median_return = returns_pct.median()
        
        ax.axvline(x=mean_return, color='blue', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_return:.1f}%')
        ax.axvline(x=median_return, color='orange', linestyle='--', linewidth=2, 
                  label=f'Median: {median_return:.1f}%')
        
        # Styling
        ax.set_title('Distribution of Trade Returns', 
                    fontsize=self.chart_params['title_size'], fontweight='bold', pad=20)
        ax.set_xlabel('Return per Trade (%)', fontsize=self.chart_params['label_size'])
        ax.set_ylabel('Number of Trades', fontsize=self.chart_params['label_size'])
        
        # Add grid and legend
        ax.grid(True, alpha=self.chart_params['grid_alpha'])
        ax.legend(loc='upper right', fontsize=self.chart_params['tick_size'])
        
        # Add statistics text box
        win_rate = (returns_pct > 0).mean() * 100
        std_dev = returns_pct.std()
        
        stats_text = f'''Trade Statistics:
Win Rate: {win_rate:.1f}%
Mean Return: {mean_return:.2f}%
Std Deviation: {std_dev:.2f}%
Best Trade: {returns_pct.max():.1f}%
Worst Trade: {returns_pct.min():.1f}%'''
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Tight layout and save
        plt.tight_layout()
        filepath = os.path.join(self.results_dir, 'returns_distribution.png')
        plt.savefig(filepath, dpi=self.chart_params['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Returns distribution saved to {filepath}")
        return filepath
    
    def generate_all_charts(self) -> List[str]:
        """
        Generate all visualization charts.
        
        Returns:
            List of paths to saved chart files
        """
        logger.info("Generating all strategy visualization charts...")
        
        # Load trades data
        trades_df = self.load_trades_data()
        
        if len(trades_df) == 0:
            raise ValueError(f"No trades found for {self.best_holding_period}-hour holding period")
        
        # Generate all charts
        chart_files = []
        
        try:
            # 1. Equity curve
            equity_file = self.create_equity_curve(trades_df)
            chart_files.append(equity_file)
            
            # 2. Drawdown chart
            drawdown_file = self.create_drawdown_chart(trades_df)
            chart_files.append(drawdown_file)
            
            # 3. Returns distribution
            returns_file = self.create_returns_distribution(trades_df)
            chart_files.append(returns_file)
            
            logger.info("✅ All visualization charts generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            raise
        
        return chart_files
    
    def print_chart_summary(self, chart_files: List[str]) -> None:
        """Print summary of generated charts."""
        logger.info("=" * 60)
        logger.info("VISUALIZATION CHARTS SUMMARY")
        logger.info("=" * 60)
        
        for i, filepath in enumerate(chart_files, 1):
            filename = os.path.basename(filepath)
            logger.info(f"{i}. {filename}")
        
        logger.info(f"\nAll {len(chart_files)} charts saved to '{self.results_dir}/' directory")
        logger.info("Charts are ready for presentation and portfolio inclusion!")
        logger.info("=" * 60)


def main():
    """Main function to generate all visualization charts."""
    try:
        # Initialize visualizer
        visualizer = StrategyVisualizer()
        
        # Generate all charts
        chart_files = visualizer.generate_all_charts()
        
        # Print summary
        visualizer.print_chart_summary(chart_files)
        
        logger.info("🎨 Visualization generation complete!")
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise


if __name__ == "__main__":
    main()