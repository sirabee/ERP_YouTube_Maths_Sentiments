#!/usr/bin/env python3
"""
Create dual line-bar charts showing video count and average engagement scores.
Graph 1: Ordered by highest average engagement score
Graph 2: Ordered by most videos
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_and_calculate_scores():
    """Load dataset and calculate weighted engagement scores with 100-view minimum."""
    file_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/data/processed/videos_gradual_complete_filtered_20250720_223652.csv"
    MIN_VIEWS = 100
    
    print("Loading video dataset...")
    df = pd.read_csv(file_path)
    
    # Ensure numeric columns
    for col in ['view_count', 'like_count', 'comment_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Apply minimum view threshold
    df_filtered = df[df['view_count'] >= MIN_VIEWS].copy()
    print(f"Filtered to {len(df_filtered):,} videos with ≥{MIN_VIEWS} views")
    
    # Calculate weighted engagement
    df_filtered['like_rate'] = (df_filtered['like_count'] / df_filtered['view_count'] * 100).round(2)
    df_filtered['comment_rate'] = (df_filtered['comment_count'] / df_filtered['view_count'] * 100).round(2)
    df_filtered['weighted_engagement_score'] = (df_filtered['like_rate'] + df_filtered['comment_rate'] * 10).round(2)
    
    return df_filtered

def prepare_query_data(df):
    """Calculate query statistics."""
    query_stats = df.groupby('search_query').agg({
        'video_id': 'count',
        'weighted_engagement_score': 'mean'
    }).round(2)
    query_stats.columns = ['video_count', 'avg_engagement']
    query_stats = query_stats.reset_index()
    
    # Filter queries with at least 3 videos for statistical reliability
    query_stats = query_stats[query_stats['video_count'] >= 3]
    
    return query_stats

def create_dual_charts(query_stats):
    """Create two dual line-bar charts."""
    
    # Get top 10 for each ordering
    top_by_engagement = query_stats.nlargest(10, 'avg_engagement')
    top_by_count = query_stats.nlargest(10, 'video_count')
    
    # Create figure with subplots (extra height for two-line labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Query Analysis: Video Count vs Average Engagement Score', fontsize=16, fontweight='bold')
    
    # GRAPH 1: Ordered by highest average engagement
    ax1_twin = ax1.twinx()
    
    # Bar chart for video count
    x1 = np.arange(len(top_by_engagement))
    bars1 = ax1.bar(x1, top_by_engagement['video_count'], 
                    color='lightblue', alpha=0.7, label='Video Count')
    
    # Line chart for average engagement
    line1 = ax1_twin.plot(x1, top_by_engagement['avg_engagement'], 
                         color='red', marker='o', linewidth=2, markersize=6,
                         label='Average Engagement Score')
    
    # Formatting for Graph 1
    ax1.set_xlabel('Search Query (Ordered by Engagement Score)')
    ax1.set_ylabel('Number of Videos', color='blue')
    ax1_twin.set_ylabel('Average Weighted Engagement Score', color='red')
    ax1.set_title('Top 10 Queries by Average Engagement Score')
    
    ax1.set_xticks(x1)
    # Create two-line labels
    labels1 = []
    for q in top_by_engagement['search_query']:
        if len(q) > 12:
            # Find a good break point (space or after 8-15 chars)
            words = q.split()
            if len(words) > 1:
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                labels1.append(f"{line1}\n{line2}")
            else:
                # If no spaces, break at middle
                mid = len(q) // 2
                labels1.append(f"{q[:mid]}\n{q[mid:]}")
        else:
            labels1.append(q)
    
    ax1.set_xticklabels(labels1, rotation=0, ha='center', fontsize=9)
    
    # Color the y-axis labels
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count, engagement in zip(bars1, top_by_engagement['video_count'], top_by_engagement['avg_engagement']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(count)}', ha='center', va='bottom', fontsize=8, color='blue')
    
    # Add value labels on line points
    for i, (engagement, count) in enumerate(zip(top_by_engagement['avg_engagement'], top_by_engagement['video_count'])):
        ax1_twin.text(i, engagement + 1.0, f'{engagement:.1f}', ha='center', va='bottom', 
                     fontsize=8, color='red', fontweight='bold')
    
    # Set y-axis limits with extra space at top
    ax1_twin.set_ylim(0, max(top_by_engagement['avg_engagement']) * 1.2)
    
    # GRAPH 2: Ordered by most videos
    ax2_twin = ax2.twinx()
    
    # Bar chart for video count
    x2 = np.arange(len(top_by_count))
    bars2 = ax2.bar(x2, top_by_count['video_count'], 
                    color='lightgreen', alpha=0.7, label='Video Count')
    
    # Line chart for average engagement
    line2 = ax2_twin.plot(x2, top_by_count['avg_engagement'], 
                         color='darkred', marker='s', linewidth=2, markersize=6,
                         label='Average Engagement Score')
    
    # Formatting for Graph 2
    ax2.set_xlabel('Search Query (Ordered by Video Count)')
    ax2.set_ylabel('Number of Videos', color='green')
    ax2_twin.set_ylabel('Average Weighted Engagement Score', color='darkred')
    ax2.set_title('Top 10 Queries by Video Count')
    
    ax2.set_xticks(x2)
    # Create two-line labels
    labels2 = []
    for q in top_by_count['search_query']:
        if len(q) > 12:
            # Find a good break point (space or after 8-15 chars)
            words = q.split()
            if len(words) > 1:
                mid = len(words) // 2
                line1 = ' '.join(words[:mid])
                line2 = ' '.join(words[mid:])
                labels2.append(f"{line1}\n{line2}")
            else:
                # If no spaces, break at middle
                mid = len(q) // 2
                labels2.append(f"{q[:mid]}\n{q[mid:]}")
        else:
            labels2.append(q)
    
    ax2.set_xticklabels(labels2, rotation=0, ha='center', fontsize=9)
    
    # Color the y-axis labels
    ax2.tick_params(axis='y', labelcolor='green')
    ax2_twin.tick_params(axis='y', labelcolor='darkred')
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, top_by_count['video_count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(count)}', ha='center', va='bottom', fontsize=8, color='green')
    
    # Add value labels on line points
    for i, engagement in enumerate(top_by_count['avg_engagement']):
        ax2_twin.text(i, engagement + 0.3, f'{engagement:.1f}', ha='center', va='bottom', 
                     fontsize=8, color='darkred', fontweight='bold')
    
    # Set y-axis limits with extra space at top
    ax2_twin.set_ylim(0, max(top_by_count['avg_engagement']) * 1.3)
    
    plt.tight_layout()
    return fig, top_by_engagement, top_by_count

def print_comparison_analysis(top_by_engagement, top_by_count):
    """Print comparative analysis of the two orderings."""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    print("\nTOP 10 QUERIES BY AVERAGE ENGAGEMENT SCORE:")
    print("-" * 60)
    print(f"{'Query':<25} {'Avg Score':<12} {'Video Count':<12} {'Score/Video':<12}")
    print("-" * 60)
    for _, row in top_by_engagement.head(10).iterrows():
        score_per_video = row['avg_engagement'] / row['video_count']
        print(f"{row['search_query']:<25} {row['avg_engagement']:<12.2f} {int(row['video_count']):<12} {score_per_video:<12.3f}")
    
    print("\nTOP 10 QUERIES BY VIDEO COUNT:")
    print("-" * 60)
    print(f"{'Query':<25} {'Video Count':<12} {'Avg Score':<12} {'Score/Video':<12}")
    print("-" * 60)
    for _, row in top_by_count.head(10).iterrows():
        score_per_video = row['avg_engagement'] / row['video_count']
        print(f"{row['search_query']:<25} {int(row['video_count']):<12} {row['avg_engagement']:<12.2f} {score_per_video:<12.3f}")
    
    # Find queries that appear in both top 10 lists
    engagement_queries = set(top_by_engagement.head(10)['search_query'])
    count_queries = set(top_by_count.head(10)['search_query'])
    overlap = engagement_queries & count_queries
    
    print(f"\nQueries appearing in both top 10 lists: {len(overlap)}")
    if overlap:
        for query in overlap:
            print(f"- {query}")
    
    # Identify interesting patterns
    high_engagement_low_count = top_by_engagement.head(5)[top_by_engagement.head(5)['video_count'] < 20]
    high_count_low_engagement = top_by_count.head(5)[top_by_count.head(5)['avg_engagement'] < 5]
    
    if len(high_engagement_low_count) > 0:
        print(f"\nHigh engagement, lower volume queries:")
        for _, row in high_engagement_low_count.iterrows():
            print(f"- {row['search_query']}: {row['avg_engagement']:.2f} score ({int(row['video_count'])} videos)")
    
    if len(high_count_low_engagement) > 0:
        print(f"\nHigh volume, lower engagement queries:")
        for _, row in high_count_low_engagement.iterrows():
            print(f"- {row['search_query']}: {int(row['video_count'])} videos ({row['avg_engagement']:.2f} score)")

def main():
    """Main execution."""
    print("CREATING DUAL LINE-BAR CHARTS")
    print("=" * 35)
    
    # Load and process data
    df = load_and_calculate_scores()
    query_stats = prepare_query_data(df)
    
    print(f"Analyzing {len(query_stats)} queries with ≥3 videos each")
    
    # Create visualization
    fig, top_by_engagement, top_by_count = create_dual_charts(query_stats)
    
    # Print analysis
    print_comparison_analysis(top_by_engagement, top_by_count)
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'dual_line_bar_engagement_analysis_{timestamp}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    
    # Show plot
    plt.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()