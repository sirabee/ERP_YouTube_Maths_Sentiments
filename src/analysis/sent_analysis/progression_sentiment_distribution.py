#!/usr/bin/env python3
"""
Analyze sentiment distribution within each progression type
for XLM-RoBERTa Enhanced model results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
BASE_DIR = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
DATA_FILE = BASE_DIR / "results/models/sentiment_analysis/xlm_roberta_clean_20250816_132139/enhanced_comments.csv"
OUTPUT_DIR = BASE_DIR / "results/visualizations"

# Color scheme
SENTIMENT_COLORS = {
    'positive': "#7ED481",
    'neutral': "#149DF3", 
    'negative': "#F0928B"
}

PROGRESSION_ORDER = [
    'single_sentence', 
    'stable', 
    'simple_transition',
    'complex_progression', 
    'learning_journey'
]

def load_data():
    """Load and prepare XLM-RoBERTa Enhanced results."""
    print("Loading XLM-RoBERTa Enhanced results...")
    df = pd.read_csv(DATA_FILE)
    df['sentiment'] = df['xlm_sentiment'].str.lower()
    print(f"Loaded {len(df):,} comments")
    return df

def calculate_distribution(subset):
    """Calculate sentiment distribution for a subset of data."""
    total = len(subset)
    if total == 0:
        return None
    
    counts = subset['sentiment'].value_counts()
    percentages = (counts / total * 100).round(1)
    
    return {
        'total': total,
        'counts': counts.to_dict(),
        'percentages': percentages.to_dict()
    }

def format_progression_name(prog_type):
    """Convert progression type to display name."""
    if prog_type == 'learning_journey':
        return 'Strong Transformation'
    return prog_type.replace('_', ' ').title()

def print_distribution(name, stats, total_dataset):
    """Print formatted distribution statistics."""
    print(f"\n{name}:")
    print(f"  Total: {stats['total']:,} ({stats['total']/total_dataset*100:.1f}% of dataset)")
    print(f"  Sentiment Distribution:")
    
    for sentiment in ['positive', 'neutral', 'negative']:
        count = stats['counts'].get(sentiment, 0)
        pct = stats['percentages'].get(sentiment, 0)
        if count > 0:
            print(f"    {sentiment.capitalize():8} : {count:6,} ({pct:5.1f}%)")

def analyze_progressions(df):
    """Analyze sentiment distribution for each progression type."""
    results = {}
    total = len(df)
    
    print("\n" + "="*70)
    print("SENTIMENT DISTRIBUTION BY PROGRESSION TYPE")
    print("="*70)
    
    # Analyze each progression type
    for prog_type in PROGRESSION_ORDER:
        subset = df[df['progression_type'] == prog_type]
        stats = calculate_distribution(subset)
        
        if stats:
            results[prog_type] = stats
            name = format_progression_name(prog_type)
            print_distribution(name, stats, total)
    
    # Analyze broad learning journey detection
    print("\n" + "-"*70)
    print("BROAD LEARNING JOURNEY DETECTION")
    print("-"*70)
    
    journey_subset = df[df['learning_journey'] == True]
    journey_stats = calculate_distribution(journey_subset)
    
    if journey_stats:
        results['broad_journey'] = journey_stats
        print_distribution("All Learning Progressions", journey_stats, total)
    
    return results

def create_visualization(df, results):
    """Create visualization of sentiment distribution by progression type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare data
    labels = []
    positive_counts = []
    neutral_counts = []
    negative_counts = []
    positive_pcts = []
    neutral_pcts = []
    negative_pcts = []
    
    for prog_type in PROGRESSION_ORDER:
        if prog_type in results:
            # Format label
            if prog_type == 'learning_journey':
                label = 'Strong\nTransformation'
            else:
                label = prog_type.replace('_', '\n').title()
            labels.append(label)
            
            # Get data
            stats = results[prog_type]
            positive_counts.append(stats['counts'].get('positive', 0))
            neutral_counts.append(stats['counts'].get('neutral', 0))
            negative_counts.append(stats['counts'].get('negative', 0))
            positive_pcts.append(stats['percentages'].get('positive', 0))
            neutral_pcts.append(stats['percentages'].get('neutral', 0))
            negative_pcts.append(stats['percentages'].get('negative', 0))
    
    x_pos = np.arange(len(labels))
    
    # Plot 1: Absolute counts
    ax1.bar(x_pos, positive_counts, color=SENTIMENT_COLORS['positive'], label='Positive')
    ax1.bar(x_pos, neutral_counts, bottom=positive_counts, 
           color=SENTIMENT_COLORS['neutral'], label='Neutral')
    ax1.bar(x_pos, negative_counts,
           bottom=np.array(positive_counts) + np.array(neutral_counts),
           color=SENTIMENT_COLORS['negative'], label='Negative')
    
    ax1.set_xlabel('Progression Type')
    ax1.set_ylabel('Number of Comments')
    ax1.set_title('Sentiment Distribution by Progression Type (Counts)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add totals on top
    for i, prog_type in enumerate(PROGRESSION_ORDER):
        if prog_type in results:
            total = results[prog_type]['total']
            y_pos = positive_counts[i] + neutral_counts[i] + negative_counts[i]
            ax1.text(i, y_pos, f'{total:,}', ha='center', va='bottom')
    
    # Plot 2: Percentages
    ax2.bar(x_pos, positive_pcts, color=SENTIMENT_COLORS['positive'], label='Positive')
    ax2.bar(x_pos, neutral_pcts, bottom=positive_pcts,
           color=SENTIMENT_COLORS['neutral'], label='Neutral')
    ax2.bar(x_pos, negative_pcts,
           bottom=np.array(positive_pcts) + np.array(neutral_pcts),
           color=SENTIMENT_COLORS['negative'], label='Negative')
    
    ax2.set_xlabel('Progression Type')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Sentiment Distribution by Progression Type (Percentages)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add percentage labels
    for i in range(len(labels)):
        y_pos = 0
        for pct, color in [(positive_pcts[i], 'white'),
                          (neutral_pcts[i], 'black'),
                          (negative_pcts[i], 'white')]:
            if pct > 5:  # Only show if large enough
                ax2.text(i, y_pos + pct/2, f'{pct:.0f}%',
                        ha='center', va='center', color=color, fontweight='bold')
            y_pos += pct
    
    plt.suptitle('XLM-RoBERTa Enhanced: Sentiment Distribution Across Progression Types',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / "progression_sentiment_distribution.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    return fig

def main():
    """Main analysis function."""
    # Load data
    df = load_data()
    
    # Analyze distributions
    results = analyze_progressions(df)
    
    # Create visualization
    print("\nCreating visualization...")
    create_visualization(df, results)
    
    print("\nAnalysis complete!")
    return results

if __name__ == "__main__":
    results = main()