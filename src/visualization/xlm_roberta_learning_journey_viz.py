#!/usr/bin/env python3
"""
XLM-RoBERTa Enhanced Learning Journey Visualization
Creates three key visualizations for thesis: learning journey by topic,
sentiment progression patterns, and overall sentiment distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define consistent color scheme
SENTIMENT_COLORS = {
    'positive': "#7CD67F",    # Green
    'neutral': "#07A4FF",      # Blue
    'negative': "#EC867F"      # Red
}

PROGRESSION_COLORS = {
    'single_sentence': '#80CBC4',      # Light teal
    'stable': '#EF9A9A',               # Light coral
    'simple_transition': '#A5D6A7',    # Light green
    'complex_progression': '#CE93D8',  # Light purple
    'learning_journey': '#FFCC80'      # Light orange
}

def format_topic_name(topic):
    """Convert topic/search_query to readable format."""
    topic_map = {
        'everyday_math': 'Everyday Math',
        'math_important': 'Math Important',
        'mathematician': 'Mathematician',
        'math_love': 'Math Love',
        'math_career': 'Math Career',
        'college_mathematics': 'College Mathematics',
        'mathematics_education': 'Mathematics Education',
        'math_stereotypes': 'Math Stereotypes',
        'math_hard': 'Math Hard',
        'mathematics_breakthroughs': 'Mathematics Breakthroughs',
        'mathematics_study': 'Mathematics Study',
        'mathematics_society': 'Mathematics Society',
        'maths_degree': 'Maths Degree',
        'maths_anxiety': 'Maths Anxiety',
        'dyscalculia': 'Dyscalculia'
    }
    return topic_map.get(topic, topic.replace('_', ' ').title())

def load_data():
    """Load XLM-RoBERTa Clean Enhanced results."""
    base_dir = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
    results_file = base_dir / "results/models/sentiment_analysis/xlm_roberta_clean_20250816_132139/enhanced_comments.csv"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    print(f"Loading data from: {results_file}")
    df = pd.read_csv(results_file)
    
    # Standardize column names
    if 'xlm_sentiment' in df.columns:
        df['sentiment'] = df['xlm_sentiment'].str.lower()
    if 'xlm_confidence' in df.columns:
        df['confidence'] = df['xlm_confidence']
    
    # Ensure learning journey column is boolean
    # Note: 'learning_journey' boolean column indicates ANY negative-to-positive progression
    # while progression_type='learning_journey' is a specific subtype
    df['is_learning_journey'] = df['learning_journey'].astype(bool) if 'learning_journey' in df.columns else False
    
    print(f"Loaded {len(df):,} comments")
    print(f"Learning journeys: {df['is_learning_journey'].sum():,} ({df['is_learning_journey'].mean()*100:.1f}%)")
    
    return df

def create_learning_journey_by_topic(df, output_dir):
    """Create horizontal bar chart of learning journey rates by topic."""
    # Calculate statistics by topic
    stats = df.groupby('search_query').agg(
        journeys=('is_learning_journey', 'sum'),
        total=('is_learning_journey', 'count')
    )
    stats['percentage'] = (stats['journeys'] / stats['total']) * 100
    
    # Get top 15 topics by percentage
    top_topics = stats.nlargest(15, 'percentage')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bars
    y_pos = np.arange(len(top_topics))
    ax.barh(y_pos, top_topics['percentage'], color="#7CD87F", alpha=0.8)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([format_topic_name(t) for t in top_topics.index])
    ax.set_xlabel('Learning Journey Rate (%)')
    ax.set_title('XLM-RoBERTa Enhanced: Learning Journey Rate by Topic', 
                fontweight='bold', pad=20)
    
    # Add grid and adjust limits
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, max(20, top_topics['percentage'].max() * 1.1))
    
    plt.tight_layout()
    output_path = output_dir / 'learning_journey_by_topic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return top_topics

def create_sentiment_progression_patterns(df, output_dir):
    """Create bar chart of sentiment progression patterns."""
    # Count progression types
    counts = df['progression_type'].value_counts()
    
    # Define display order
    order = ['single_sentence', 'stable', 'simple_transition', 
             'complex_progression', 'learning_journey']
    
    # Prepare data in order
    labels = []
    values = []
    colors = []
    
    for prog_type in order:
        if prog_type in counts.index:
            # Special case for learning_journey -> Strong Transformation
            if prog_type == 'learning_journey':
                labels.append('Strong Transformation')
            else:
                labels.append(prog_type.replace('_', ' ').title())
            values.append(counts[prog_type])
            colors.append(PROGRESSION_COLORS.get(prog_type, '#999999'))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=colors, alpha=0.8)
    
    # Set labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of Comments')
    ax.set_title('XLM-RoBERTa Enhanced: Sentiment Progression Patterns',
                fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{int(height):,}', ha='center', va='bottom')
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.1)
    
    plt.tight_layout()
    output_path = output_dir / 'sentiment_progression_patterns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return counts

def create_overall_sentiment_distribution(df, output_dir):
    """Create pie chart of overall sentiment distribution."""
    # Count sentiments
    counts = df['sentiment'].value_counts()
    
    # Prepare data in consistent order
    order = ['positive', 'neutral', 'negative']
    sizes = []
    labels = []
    colors = []
    
    for sentiment in order:
        if sentiment in counts.index:
            sizes.append(counts[sentiment])
            labels.append(sentiment.capitalize())
            colors.append(SENTIMENT_COLORS[sentiment])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=45,
        textprops={'fontsize': 12}
    )
    
    # Bold white percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('XLM-RoBERTa Enhanced: Overall Sentiment Distribution',
                fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'overall_sentiment_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # Print percentages
    total = counts.sum()
    print("\nSentiment Distribution:")
    for sentiment in order:
        if sentiment in counts.index:
            pct = (counts[sentiment] / total) * 100
            print(f"  {sentiment.capitalize()}: {pct:.1f}%")
    
    return counts

def generate_report(df, output_dir):
    """Generate summary statistics report."""
    total = len(df)
    journeys = df['is_learning_journey'].sum()
    journey_rate = (journeys / total) * 100
    
    # Count specific learning journey progression type
    specific_journey = (df['progression_type'] == 'learning_journey').sum()
    
    report = f"""XLM-RoBERTa Enhanced Learning Journey Analysis
==============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Statistics:
- Total Comments: {total:,}
- Comments with Negative-to-Positive Progression: {journeys:,} ({journey_rate:.2f}%)
  (Includes all types of learning progressions)

Sentiment Distribution:
"""
    
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = (count / total) * 100
        report += f"- {sentiment.capitalize()}: {count:,} ({pct:.1f}%)\n"
    
    report += "\nProgression Type Breakdown:\n"
    prog_counts = df['progression_type'].value_counts()
    for prog_type, count in prog_counts.head().items():
        pct = (count / total) * 100
        if prog_type == 'learning_journey':
            label = 'Strong Transformation'
        else:
            label = prog_type.replace('_', ' ').title()
        report += f"- {label}: {count:,} ({pct:.1f}%)\n"
    
    report += f"""
Note: The 'learning_journey' boolean column ({journeys:,}) captures ALL negative-to-positive
progressions, while 'Strong Transformation' ({specific_journey:,}) represents only the most
explicit, pronounced learning breakthroughs with multiple sentiment transitions.
"""
    
    # Save report
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: {report_path}")
    return report

def main():
    """Run complete visualization pipeline."""
    print("\nXLM-ROBERTA ENHANCED LEARNING JOURNEY VISUALIZATION")
    print("=" * 60)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
    output_dir = base_dir / f"results/visualizations/xlm_roberta_learning_journey_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_learning_journey_by_topic(df, output_dir)
    create_sentiment_progression_patterns(df, output_dir)
    create_overall_sentiment_distribution(df, output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(df, output_dir)
    
    print(f"\nComplete! All outputs saved to:\n{output_dir}")

if __name__ == "__main__":
    main()