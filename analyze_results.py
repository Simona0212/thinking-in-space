#!/usr/bin/env python3
"""
Result Analysis Script for VSI-Bench Evaluation
Aggregates and compares results across multiple models
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all aggregated results from the results directory"""
    results_path = Path(results_dir)
    all_results = {}

    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Find the most recent aggregated results
        agg_files = sorted(model_dir.glob("aggregated_*.json"), reverse=True)

        if agg_files:
            with open(agg_files[0], 'r') as f:
                all_results[model_name] = json.load(f)

    return all_results


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison table of all models"""
    if not results:
        return pd.DataFrame()

    # Get all unique metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())

    # Remove non-metric keys
    all_metrics.discard('tabulated_keys')
    all_metrics.discard('tabulated_results')

    # Create DataFrame
    data = []
    for model_name, model_results in results.items():
        row = {'Model': model_name}
        for metric in sorted(all_metrics):
            row[metric] = model_results.get(metric, None)
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by overall score
    if 'overall' in df.columns:
        df = df.sort_values('overall', ascending=False)

    return df


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table"""
    if df.empty:
        print("No results found!")
        return

    print("\n" + "="*100)
    print("VSI-Bench Evaluation Results Comparison")
    print("="*100)

    # Print overall scores first
    if 'overall' in df.columns:
        print("\nOverall Scores:")
        print("-"*50)
        for _, row in df.iterrows():
            model = row['Model'].replace('_', '/')
            score = row['overall']
            print(f"  {model:50s} {score:6.2f}%")

    # Print per-task breakdown
    print("\nPer-Task Breakdown:")
    print("-"*100)

    # Get task columns (exclude Model and overall)
    task_cols = [col for col in df.columns if col not in ['Model', 'overall']]

    if task_cols:
        # Print header
        print(f"{'Model':<40s}", end='')
        for col in task_cols:
            print(f"{col[:15]:>15s}", end='')
        print()
        print("-"*100)

        # Print rows
        for _, row in df.iterrows():
            model = row['Model'].replace('_', '/')[:38]
            print(f"{model:<40s}", end='')
            for col in task_cols:
                val = row[col]
                if pd.notna(val):
                    print(f"{val:>15.2f}", end='')
                else:
                    print(f"{'N/A':>15s}", end='')
            print()

    print("="*100)


def export_results(df: pd.DataFrame, output_path: str):
    """Export results to CSV and Excel"""
    output_path = Path(output_path)

    # CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nExported to CSV: {csv_path}")

    # Excel (if openpyxl is available)
    try:
        excel_path = output_path.with_suffix('.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"Exported to Excel: {excel_path}")
    except ImportError:
        print("Install openpyxl to export to Excel: pip install openpyxl")


def analyze_task_performance(df: pd.DataFrame):
    """Analyze performance across different task types"""
    print("\n" + "="*100)
    print("Task-Level Analysis")
    print("="*100)

    task_cols = [col for col in df.columns if col not in ['Model', 'overall']]

    if not task_cols:
        print("No task-level metrics found")
        return

    print("\nBest performing model per task:")
    print("-"*50)

    for task in task_cols:
        if task in df.columns:
            best_idx = df[task].idxmax()
            best_model = df.loc[best_idx, 'Model'].replace('_', '/')
            best_score = df.loc[best_idx, task]
            print(f"  {task:40s} {best_model:40s} {best_score:6.2f}%")

    print("\nAverage performance per task:")
    print("-"*50)

    for task in task_cols:
        if task in df.columns:
            avg_score = df[task].mean()
            std_score = df[task].std()
            print(f"  {task:40s} {avg_score:6.2f}% (±{std_score:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze VSI-Bench evaluation results")
    parser.add_argument("--results_dir", type=str, default="./results", help="Results directory")
    parser.add_argument("--export", type=str, default=None, help="Export comparison table to file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed task analysis")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found results for {len(results)} models")

    # Create comparison table
    df = create_comparison_table(results)

    # Print comparison
    print_comparison_table(df)

    # Detailed analysis
    if args.detailed:
        analyze_task_performance(df)

    # Export if requested
    if args.export:
        export_results(df, args.export)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
