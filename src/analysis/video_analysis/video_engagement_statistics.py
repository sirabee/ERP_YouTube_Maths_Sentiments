#!/usr/bin/env python3
"""
Analyze video engagement scores to find min, max, median, mean
for top 20 videos by different metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
RESULTS_DIR = BASE_DIR / "results/analysis/video_analysis"

def load_top_videos_data():
    """Load all top videos datasets."""
    datasets = {
        'engagement': RESULTS_DIR / "top_videos_weighted_engagement_20250828_224715.csv",
        'comments': RESULTS_DIR / "top_videos_by_comment_count_weighted_20250828_231343.csv",
        'likes': RESULTS_DIR / "top_videos_by_like_count_weighted_20250828_231343.csv",
        'views': RESULTS_DIR / "top_videos_by_view_count_weighted_20250828_231343.csv"
    }
    
    loaded_data = {}
    for name, path in datasets.items():
        if path.exists():
            df = pd.read_csv(path)
            # Take top 20 rows
            df_top20 = df.head(20)
            loaded_data[name] = df_top20
            print(f"Loaded {name}: {len(df_top20)} videos")
        else:
            print(f"Warning: {name} file not found at {path}")
    
    return loaded_data

def calculate_statistics(df, metric_name):
    """Calculate statistics for a specific metric."""
    if 'weighted_engagement_score' not in df.columns:
        print(f"Warning: weighted_engagement_score not found in {metric_name}")
        return None
    
    scores = df['weighted_engagement_score'].values
    
    stats = {
        'metric': metric_name,
        'count': len(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'q25': np.percentile(scores, 25),
        'q75': np.percentile(scores, 75)
    }
    
    # Add additional metrics if available
    if 'video_views' in df.columns:
        stats['total_views'] = df['video_views'].sum()
        stats['avg_views'] = df['video_views'].mean()
    
    if 'video_likes' in df.columns:
        stats['total_likes'] = df['video_likes'].sum()
        stats['avg_likes'] = df['video_likes'].mean()
    
    if 'comment_count' in df.columns:
        stats['total_comments'] = df['comment_count'].sum()
        stats['avg_comments'] = df['comment_count'].mean()
    
    return stats

def print_statistics(stats):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"TOP 20 VIDEOS BY {stats['metric'].upper()}")
    print(f"{'='*60}")
    
    print(f"\nEngagement Score Statistics:")
    print(f"  Count:   {stats['count']}")
    print(f"  Min:     {stats['min']:.4f}")
    print(f"  Q25:     {stats['q25']:.4f}")
    print(f"  Median:  {stats['median']:.4f}")
    print(f"  Mean:    {stats['mean']:.4f}")
    print(f"  Q75:     {stats['q75']:.4f}")
    print(f"  Max:     {stats['max']:.4f}")
    print(f"  Std Dev: {stats['std']:.4f}")
    
    if 'total_views' in stats:
        print(f"\nAdditional Metrics:")
        print(f"  Total Views:    {stats['total_views']:,}")
        print(f"  Average Views:  {stats['avg_views']:,.0f}")
    
    if 'total_likes' in stats:
        print(f"  Total Likes:    {stats['total_likes']:,}")
        print(f"  Average Likes:  {stats['avg_likes']:,.0f}")
    
    if 'total_comments' in stats:
        print(f"  Total Comments: {stats['total_comments']:,}")
        print(f"  Average Comments: {stats['avg_comments']:,.0f}")

def create_comparison_table(all_stats):
    """Create a comparison table of all metrics."""
    comparison_data = []
    
    for metric_name, stats in all_stats.items():
        if stats:
            comparison_data.append({
                'Metric': metric_name.title(),
                'Min': f"{stats['min']:.4f}",
                'Q25': f"{stats['q25']:.4f}",
                'Median': f"{stats['median']:.4f}",
                'Mean': f"{stats['mean']:.4f}",
                'Q75': f"{stats['q75']:.4f}",
                'Max': f"{stats['max']:.4f}",
                'Std Dev': f"{stats['std']:.4f}"
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    return df_comparison

def analyze_video_titles(datasets):
    """Analyze common themes in top video titles."""
    print("\n" + "="*60)
    print("TOP VIDEO TITLES ANALYSIS")
    print("="*60)
    
    for name, df in datasets.items():
        if df is not None and 'video_title' in df.columns:
            print(f"\nTop 5 videos by {name.upper()}:")
            for i, row in df.head(5).iterrows():
                score = row.get('weighted_engagement_score', 0)
                title = row['video_title'][:80] + "..." if len(row['video_title']) > 80 else row['video_title']
                print(f"  {i+1}. [{score:.4f}] {title}")

def save_report(all_stats, comparison_df, output_dir):
    """Save analysis report to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"engagement_statistics_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("VIDEO ENGAGEMENT STATISTICS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write("SUMMARY TABLE\n")
        f.write("-"*60 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        for metric_name, stats in all_stats.items():
            if stats:
                f.write(f"TOP 20 VIDEOS BY {metric_name.upper()}\n")
                f.write("-"*60 + "\n")
                f.write(f"Min:    {stats['min']:.4f}\n")
                f.write(f"Q25:    {stats['q25']:.4f}\n")
                f.write(f"Median: {stats['median']:.4f}\n")
                f.write(f"Mean:   {stats['mean']:.4f}\n")
                f.write(f"Q75:    {stats['q75']:.4f}\n")
                f.write(f"Max:    {stats['max']:.4f}\n")
                f.write(f"Std:    {stats['std']:.4f}\n\n")
    
    print(f"\nReport saved to: {report_file}")
    return report_file

def main():
    """Main analysis function."""
    print("VIDEO ENGAGEMENT SCORE ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading top videos data...")
    datasets = load_top_videos_data()
    
    if not datasets:
        print("No data files found!")
        return
    
    # Calculate statistics for each metric
    all_stats = {}
    
    for name, df in datasets.items():
        if df is not None:
            stats = calculate_statistics(df, name)
            if stats:
                all_stats[name] = stats
                print_statistics(stats)
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    
    comparison_df = create_comparison_table(all_stats)
    print("\n" + comparison_df.to_string(index=False))
    
    # Analyze video titles
    analyze_video_titles(datasets)
    
    # Save report
    output_dir = RESULTS_DIR
    save_report(all_stats, comparison_df, output_dir)
    
    print("\nAnalysis complete!")
    
    return all_stats, comparison_df

if __name__ == "__main__":
    stats, comparison = main()