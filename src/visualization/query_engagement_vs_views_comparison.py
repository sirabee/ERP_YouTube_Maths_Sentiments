#!/usr/bin/env python3
"""
Compare top 10 queries by average engagement score vs top 10 queries by total view count.
Creates visualization showing the inverse relationship between volume and engagement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
DATA_FILE = BASE_DIR / "results/analysis/video_analysis/videos_with_weighted_scores_20250828_224715.csv"
OUTPUT_DIR = BASE_DIR / "results/visualizations"

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_analyze_queries(df):
    """Analyze queries by engagement and views."""
    
    # Group by search query
    query_stats = df.groupby('search_query').agg({
        'weighted_engagement_score': 'mean',  # Average engagement
        'view_count': 'sum',  # Total views
        'video_id': 'count',  # Video count
        'like_count': 'sum',  # Total likes
        'comment_count': 'sum'  # Total comments
    }).reset_index()
    
    query_stats.columns = ['search_query', 'avg_engagement', 'total_views', 
                           'video_count', 'total_likes', 'total_comments']
    
    # Get top 10 by average engagement
    top_engagement = query_stats.nlargest(10, 'avg_engagement')
    
    # Get top 10 by total views
    top_views = query_stats.nlargest(10, 'total_views')
    
    return query_stats, top_engagement, top_views

def format_query_name(query):
    """Format query name for display."""
    query_map = {
        'math_phobia': 'math phobia',
        'mathematics_instructor': 'mathematics\ninstructor',
        'maths_hate': 'maths hate',
        'maths_pedagogy': 'maths\npedagogy',
        'Key': 'Key',
        'stage_maths': 'Stage maths',
        'maths_intergrating': 'maths\nintegrating',
        'girls_and_maths': 'girls\nand maths',
        'maths_in_life': 'maths\nin life',
        'math_important': 'math\nimportant',
        'maths_anxiety': 'maths\nanxiety',
        'financial_maths': 'financial\nmaths',
        'matrices_explained': 'matrices\nexplained',
        'vectors_tutorial': 'vectors\ntutorial',
        'applied_mathematics': 'applied\nmathematics',
        'decimals_help': 'decimals\nhelp',
        'linear_equations': 'linear\nequations',
        'differentiation': 'differentiation',
        'quadratic_formula': 'quadratic',
        'math_for_data_science': 'math for\ndata science',
        'dyscalculia': 'dyscalculia'
    }
    
    # Handle the query formatting
    formatted = query_map.get(query, query.replace('_', '\n'))
    return formatted

def create_comparison_visualization(top_engagement, top_views):
    """Create side-by-side comparison of engagement vs views."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Top 10 by Average Engagement Score
    x1 = np.arange(len(top_engagement))
    bars1 = ax1.bar(x1, top_engagement['video_count'], color='skyblue', alpha=0.8)
    
    # Add engagement score line
    ax1_twin = ax1.twinx()
    line1 = ax1_twin.plot(x1, top_engagement['avg_engagement'], 
                          color='red', marker='o', linewidth=2, markersize=8, 
                          label='Average Engagement Score')
    
    # Labels for left plot
    ax1.set_xlabel('Search Query (Ordered by Engagement Score)', fontsize=11)
    ax1.set_ylabel('Number of Videos', fontsize=11, color='darkblue')
    ax1_twin.set_ylabel('Average Engagement Score', fontsize=11, color='red')
    ax1.set_title('Top 10 Queries by Average Engagement Score', fontsize=12, fontweight='bold')
    
    # Format x-axis labels
    query_labels1 = [format_query_name(q) for q in top_engagement['search_query']]
    ax1.set_xticks(x1)
    ax1.set_xticklabels(query_labels1, rotation=0, ha='center', fontsize=9)
    
    # Add value labels on bars
    for bar, count in zip(bars1, top_engagement['video_count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontsize=9, color='darkblue')
    
    # Add value labels on line points
    for i, (x, y) in enumerate(zip(x1, top_engagement['avg_engagement'])):
        ax1_twin.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom', 
                     fontsize=9, color='red', fontweight='bold')
    
    # Grid
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Right plot: Top 10 by Total View Count
    x2 = np.arange(len(top_views))
    
    # Convert views to millions for readability
    views_millions = top_views['total_views'] / 1_000_000
    bars2 = ax2.bar(x2, views_millions, color='lightgreen', alpha=0.8)
    
    # Add engagement score line
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(x2, top_views['avg_engagement'],
                          color='red', marker='o', linewidth=2, markersize=8,
                          label='Average Engagement Score')
    
    # Labels for right plot
    ax2.set_xlabel('Search Query (Ordered by Total View Count)', fontsize=11)
    ax2.set_ylabel('Total Views (Millions)', fontsize=11, color='darkgreen')
    ax2_twin.set_ylabel('Average Engagement Score', fontsize=11, color='red')
    ax2.set_title('Top 10 Queries by Total View Count', fontsize=12, fontweight='bold')
    
    # Format x-axis labels
    query_labels2 = [format_query_name(q) for q in top_views['search_query']]
    ax2.set_xticks(x2)
    ax2.set_xticklabels(query_labels2, rotation=0, ha='center', fontsize=9)
    
    # Add value labels on bars
    for bar, views in zip(bars2, views_millions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{views:.1f}M', ha='center', va='bottom', fontsize=9, color='darkgreen')
    
    # Add value labels on line points
    for i, (x, y) in enumerate(zip(x2, top_views['avg_engagement'])):
        ax2_twin.text(x, y + 0.2, f'{y:.1f}', ha='center', va='bottom',
                     fontsize=9, color='red', fontweight='bold')
    
    # Grid
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Main title
    fig.suptitle('Query Analysis: Average Engagement Score vs Total View Count', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"query_engagement_vs_views_{timestamp}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    plt.show()
    
    return fig

def print_analysis_summary(top_engagement, top_views, query_stats):
    """Print summary statistics."""
    
    print("\n" + "="*70)
    print("QUERY ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nTOP 10 QUERIES BY AVERAGE ENGAGEMENT SCORE:")
    print("-"*50)
    for i, row in top_engagement.iterrows():
        print(f"{i+1:2}. {row['search_query']:25} | Score: {row['avg_engagement']:6.2f} | Videos: {row['video_count']:3}")
    
    print("\nTOP 10 QUERIES BY TOTAL VIEW COUNT:")
    print("-"*50)
    for i, row in top_views.iterrows():
        print(f"{i+1:2}. {row['search_query']:25} | Views: {row['total_views']/1e6:6.2f}M | Score: {row['avg_engagement']:6.2f}")
    
    # Calculate correlation
    correlation = query_stats['avg_engagement'].corr(query_stats['total_views'])
    print(f"\nCorrelation between engagement and views: {correlation:.3f}")
    
    # Overlap analysis
    top_eng_queries = set(top_engagement['search_query'])
    top_view_queries = set(top_views['search_query'])
    overlap = top_eng_queries.intersection(top_view_queries)
    
    print(f"\nOverlap between top engagement and top views: {len(overlap)} queries")
    if overlap:
        print("Overlapping queries:", ', '.join(overlap))
    
    # Statistical comparison
    print("\n" + "-"*50)
    print("STATISTICAL COMPARISON:")
    print(f"Top Engagement Queries - Avg Score: {top_engagement['avg_engagement'].mean():.2f}")
    print(f"Top View Queries - Avg Score: {top_views['avg_engagement'].mean():.2f}")
    print(f"Score Difference: {top_engagement['avg_engagement'].mean() - top_views['avg_engagement'].mean():.2f}")
    
    print(f"\nTop Engagement Queries - Total Views: {top_engagement['total_views'].sum()/1e6:.2f}M")
    print(f"Top View Queries - Total Views: {top_views['total_views'].sum()/1e6:.2f}M")

def main():
    """Main analysis function."""
    
    print("LOADING DATA...")
    print("="*70)
    
    # Load data
    if not DATA_FILE.exists():
        print(f"Error: Data file not found at {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df):,} videos")
    
    # Analyze queries
    query_stats, top_engagement, top_views = load_and_analyze_queries(df)
    
    # Print summary
    print_analysis_summary(top_engagement, top_views, query_stats)
    
    # Create visualization
    print("\nCREATING VISUALIZATION...")
    fig = create_comparison_visualization(top_engagement, top_views)
    
    print("\nAnalysis complete!")
    
    return query_stats, top_engagement, top_views

if __name__ == "__main__":
    stats, top_eng, top_views = main()