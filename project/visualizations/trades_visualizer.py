#!/usr/bin/env python3
"""
Simple trades visualizer: reads a `trades.json` file (list of {timestamp, price, quantity})
and plots price over time (with optional volume overlay). Saves `trades_price_timeline.png`.

Usage:
  python visualizations/trades_visualizer.py path/to/trades.json

If a metrics JSON is provided and it contains `trade_series`, it will extract that.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def load_trades(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # If file contains an object with trade_series key
    if isinstance(data, dict) and 'trade_series' in data:
        data = data['trade_series']
    if not isinstance(data, list):
        raise ValueError('Expected a list of trade snapshots or a dict with trade_series')
    return data


def plot_trades(trades, out_path, ma_window=20, show_raw=False, show_volume=False):
    # Parse timestamps and values
    timestamps = []
    prices = []
    quantities = []
    for t in trades:
        ts = t.get('timestamp')
        if ts is None:
            # If timestamp missing, skip
            continue
        # Normalize Z timezone
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            dt = datetime.fromisoformat(str(ts))
        timestamps.append(dt)
        prices.append(float(t.get('price', 0.0)))
        quantities.append(float(t.get('quantity', 0.0)))

    if len(timestamps) == 0:
        print('No timestamps found in trades file; nothing to plot')
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    # Compute moving average (running average) for legibility
    try:
        window = max(1, int(ma_window))
    except Exception:
        window = 20

    if len(prices) >= 2 and window > 1:
        # Use 'same' mode to preserve alignment; boundary values are averaged with available points
        smoothed = np.convolve(prices, np.ones(window) / window, mode='same')
    else:
        smoothed = prices

    if show_raw:
        ax.plot(timestamps, prices, color='#9ecae1', linewidth=1, alpha=0.6, label='Raw')
    ax.plot(timestamps, smoothed, color='#2b8cbe', linewidth=2, label=f'MA (window={window})')
    ax.set_title('Trade Price Timeline')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Volume bars are optional; disabled by default to keep the chart clean.
    if show_volume and any(q > 0 for q in quantities):
        ax2 = ax.twinx()
        # Use a reasonable bar width based on timespan; keep bars faint
        span = max(1e-6, (timestamps[-1].timestamp() - timestamps[0].timestamp()))
        width = 0.0005 * span
        ax2.bar(timestamps, quantities, width=width, alpha=0.2, color='#e41a1c')
        ax2.set_ylabel('Quantity', color='#e41a1c')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Price timeline saved to: {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Plot trade price series from trades.json')
    parser.add_argument('trades_file', help='Path to trades.json or metrics JSON containing trade_series')
    parser.add_argument('--output', help='Output PNG path', default=None)
    parser.add_argument('--ma-window', type=int, default=20, help='Moving average window size (points)')
    parser.add_argument('--raw', action='store_true', help='Also plot raw prices beneath the moving average')
    parser.add_argument('--show-volume', action='store_true', help='Show quantity bars as a volume overlay (disabled by default)')
    args = parser.parse_args()

    trades_path = Path(args.trades_file)
    if not trades_path.exists():
        print(f'File not found: {trades_path}')
        return

    trades = load_trades(trades_path)
    out_path = Path(args.output) if args.output else trades_path.parent / 'trades_price_timeline.png'
    plot_trades(trades, out_path, ma_window=args.ma_window, show_raw=args.raw, show_volume=args.show_volume)


if __name__ == '__main__':
    main()
