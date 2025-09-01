#!/usr/bin/env python3
"""
Video Dataset Weighted Engagement Analysis Script

Implements weighted engagement scoring where comments are valued 10x more than likes
to better reflect engagement quality across videos of different scales.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
COMMENT_WEIGHT = 10
MIN_VIEWS_THRESHOLD = 1000

def load_video_data():
    """Load the filtered video dataset."""
    file_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/data/processed/videos_gradual_complete_filtered_20250720_223652.csv"
    
    print("Loading video dataset...")
    df = pd.read_csv(file_path)
    
    # Ensure numeric columns
    numeric_columns = ['view_count', 'like_count', 'comment_count']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print(f"Loaded {len(df):,} videos")
    return df

def calculate_engagement_scores(df):
    """Calculate both old and weighted engagement scores."""
    df = df.copy()
    
    # Calculate rates as percentages
    df['like_rate'] = np.where(df['view_count'] > 0, 
                               (df['like_count'] / df['view_count'] * 100), 0)
    df['comment_rate'] = np.where(df['view_count'] > 0, 
                                  (df['comment_count'] / df['view_count'] * 100), 0)
    
    # Calculate scores
    df['engagement_score_old'] = df['like_rate'] + df['comment_rate']
    df['weighted_engagement_score'] = df['like_rate'] + (df['comment_rate'] * COMMENT_WEIGHT)
    
    # Add scale categories
    df['scale_category'] = pd.cut(df['view_count'], 
                                  bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
                                  labels=['Micro', 'Small', 'Medium', 'Large', 'Viral'],
                                  right=False)
    
    return df

def validate_across_scales(df):
    """Validate metric performance across different video scales."""
    print("\n" + "="*60)
    print("METRIC VALIDATION ACROSS VIDEO SCALES")
    print("="*60)
    
    # Filter to minimum threshold
    df_filtered = df[df['view_count'] >= MIN_VIEWS_THRESHOLD]
    
    # Group by scale and calculate statistics
    scale_stats = df_filtered.groupby('scale_category').agg({
        'video_id': 'count',
        'like_rate': 'mean',
        'comment_rate': 'mean',
        'engagement_score_old': ['mean', 'std'],
        'weighted_engagement_score': ['mean', 'std']
    }).round(3)
    
    print("\nEngagement by Video Scale:")
    print("-" * 60)
    print(scale_stats)
    
    return scale_stats

def compare_top_videos(df, top_n=20):
    """Compare top videos between old and weighted metrics."""
    print("\n" + "="*60)
    print("TOP VIDEOS COMPARISON")
    print("="*60)
    
    df_filtered = df[df['view_count'] >= MIN_VIEWS_THRESHOLD]
    
    # Get top videos by each metric
    top_old = set(df_filtered.nlargest(top_n, 'engagement_score_old')['video_id'])
    top_new = set(df_filtered.nlargest(top_n, 'weighted_engagement_score')['video_id'])
    
    overlap = len(top_old & top_new)
    print(f"\nTop {top_n} videos overlap: {overlap}/{top_n} ({overlap/top_n*100:.1f}%)")
    
    # Show examples of videos that gained prominence
    only_new = top_new - top_old
    if only_new:
        print(f"\nVideos gaining prominence with weighted metric ({len(only_new)} videos):")
        gaining = df_filtered[df_filtered['video_id'].isin(only_new)].nlargest(3, 'comment_rate')
        for _, row in gaining.iterrows():
            print(f"- {row['title'][:60]}...")
            print(f"  Comment rate: {row['comment_rate']:.3f}% | Like rate: {row['like_rate']:.2f}%")

def generate_top_videos_report(df, top_n=50):
    """Generate report of top engaged videos."""
    print("\n" + "="*60)
    print(f"TOP {top_n} VIDEOS BY WEIGHTED ENGAGEMENT")
    print("="*60)
    
    df_filtered = df[df['view_count'] >= MIN_VIEWS_THRESHOLD]
    top_videos = df_filtered.nlargest(top_n, 'weighted_engagement_score')
    
    # Select relevant columns
    report_columns = [
        'video_id', 'title', 'search_query', 
        'view_count', 'like_count', 'comment_count',
        'like_rate', 'comment_rate', 
        'engagement_score_old', 'weighted_engagement_score',
        'scale_category'
    ]
    
    report_df = top_videos[report_columns].reset_index(drop=True)
    report_df.index = range(1, len(report_df) + 1)
    report_df.index.name = 'rank'
    
    # Display top 10
    print("\nTop 10 Most Engaging Videos:")
    print("-" * 60)
    for idx, row in report_df.head(10).iterrows():
        print(f"\n{idx}. {row['title'][:70]}...")
        print(f"   Views: {row['view_count']:,} | Likes: {row['like_count']:,} | Comments: {row['comment_count']:,}")
        print(f"   Weighted Score: {row['weighted_engagement_score']:.2f}")
    
    return report_df

def save_results(df, top_videos_df):
    """Save analysis results to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save top videos
    top_filename = f"top_videos_weighted_engagement_{timestamp}.csv"
    top_videos_df.to_csv(top_filename)
    
    # Save full dataset with scores (filtered)
    df_filtered = df[df['view_count'] >= MIN_VIEWS_THRESHOLD]
    full_filename = f"videos_with_weighted_scores_{timestamp}.csv"
    
    columns_to_save = [
        'video_id', 'title', 'search_query', 
        'view_count', 'like_count', 'comment_count',
        'like_rate', 'comment_rate', 
        'engagement_score_old', 'weighted_engagement_score',
        'scale_category'
    ]
    
    df_filtered[columns_to_save].to_csv(full_filename, index=False)
    
    # Save summary statistics
    summary_filename = f"weighted_engagement_summary_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write("WEIGHTED ENGAGEMENT ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Comment Weight: {COMMENT_WEIGHT}x\n")
        f.write(f"Min Views Threshold: {MIN_VIEWS_THRESHOLD:,}\n")
        f.write(f"Videos Analyzed: {len(df_filtered):,}\n\n")
        
        f.write("WEIGHTED SCORE STATISTICS:\n")
        f.write(f"Mean: {df_filtered['weighted_engagement_score'].mean():.3f}\n")
        f.write(f"Median: {df_filtered['weighted_engagement_score'].median():.3f}\n")
        f.write(f"Std Dev: {df_filtered['weighted_engagement_score'].std():.3f}\n")
        f.write(f"Max: {df_filtered['weighted_engagement_score'].max():.3f}\n")
    
    print(f"\nFiles saved:")
    print(f"- {top_filename}")
    print(f"- {full_filename}")
    print(f"- {summary_filename}")
    
    return [top_filename, full_filename, summary_filename]

def main():
    """Main execution function."""
    print("WEIGHTED VIDEO ENGAGEMENT ANALYSIS")
    print("=" * 40)
    print(f"Comment weight: {COMMENT_WEIGHT}x")
    print(f"Min views threshold: {MIN_VIEWS_THRESHOLD:,}")
    
    # Load and process data
    df = load_video_data()
    df = calculate_engagement_scores(df)
    
    # Validate and compare
    validate_across_scales(df)
    compare_top_videos(df)
    
    # Generate report
    top_videos_df = generate_top_videos_report(df)
    
    # Save results
    save_results(df, top_videos_df)
    
    print("\n" + "="*40)
    print("Analysis complete!")
    
    return df

if __name__ == "__main__":
    results = main()