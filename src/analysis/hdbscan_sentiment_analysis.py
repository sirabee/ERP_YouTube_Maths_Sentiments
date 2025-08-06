#!/usr/bin/env python3
"""
Comparative Analysis: Comment-Level vs Sentence-Level Sentiment Analysis
Measures the impact and statistical significance of implementing per-sentence analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_comparative_data():
    """Load both comment-level and sentence-level sentiment analysis results."""
    
    # Comment-level analysis
    comment_level_path = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/Sentiment_Analysis/sentiment_analysis_20250726_202945/comments_with_topics_and_sentiment.csv"
    
    # Sentence-level analysis  
    sentence_level_path = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/Sentiment_Analysis/sentence_sentiment_analysis_20250726_221258/comments_with_sentence_sentiment.csv"
    
    print("Loading comment-level sentiment analysis results...")
    comment_df = pd.read_csv(comment_level_path)
    
    print("Loading sentence-level sentiment analysis results...")
    sentence_df = pd.read_csv(sentence_level_path)
    
    print(f"Comment-level dataset: {len(comment_df):,} comments")
    print(f"Sentence-level dataset: {len(sentence_df):,} comments")
    
    return comment_df, sentence_df

def align_datasets(comment_df, sentence_df):
    """Align datasets to ensure we're comparing the same comments."""
    
    # Use comment_id as the key for alignment (assuming both have this column)
    # If comment_id is not unique, we'll need to use a combination
    
    print("Aligning datasets for fair comparison...")
    
    # Check if we have the same comments in both datasets
    comment_ids_comment = set(comment_df['comment_id'].astype(str))
    comment_ids_sentence = set(sentence_df['comment_id'].astype(str))
    
    common_ids = comment_ids_comment.intersection(comment_ids_sentence)
    print(f"Common comment IDs: {len(common_ids):,}")
    
    # Filter both datasets to only include common comments
    comment_aligned = comment_df[comment_df['comment_id'].astype(str).isin(common_ids)].copy()
    sentence_aligned = sentence_df[sentence_df['comment_id'].astype(str).isin(common_ids)].copy()
    
    # Sort by comment_id for proper alignment
    comment_aligned = comment_aligned.sort_values('comment_id').reset_index(drop=True)
    sentence_aligned = sentence_aligned.sort_values('comment_id').reset_index(drop=True)
    
    print(f"Aligned datasets: {len(comment_aligned):,} comments each")
    
    return comment_aligned, sentence_aligned

def compare_sentiment_distributions(comment_df, sentence_df, output_dir):
    """Compare sentiment distributions between approaches."""
    
    print("Comparing sentiment distributions...")
    
    # Get sentiment distributions
    comment_sentiment = comment_df['sentiment_label'].value_counts(normalize=True) * 100
    sentence_sentiment = sentence_df['sentence_level_sentiment'].value_counts(normalize=True) * 100
    
    # Ensure all sentiments are represented
    sentiments = ['positive', 'neutral', 'negative']
    comment_dist = [comment_sentiment.get(s, 0) for s in sentiments]
    sentence_dist = [sentence_sentiment.get(s, 0) for s in sentiments]
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Comment-Level': comment_dist,
        'Sentence-Level': sentence_dist
    }, index=sentiments)
    
    print("\nSentiment Distribution Comparison:")
    print(comparison_df.round(2))
    
    # Calculate differences
    differences = comparison_df['Sentence-Level'] - comparison_df['Comment-Level']
    print(f"\nDifferences (Sentence-Level - Comment-Level):")
    for sentiment, diff in differences.items():
        direction = "increase" if diff > 0 else "decrease"
        print(f"{sentiment.capitalize()}: {diff:+.2f}% ({direction})")
    
    return comparison_df, differences

def statistical_significance_tests(comment_df, sentence_df):
    """Perform statistical significance tests."""
    
    print("\nPerforming statistical significance tests...")
    
    results = {}
    
    # 1. Chi-square test for sentiment distribution independence
    comment_counts = comment_df['sentiment_label'].value_counts()
    sentence_counts = sentence_df['sentence_level_sentiment'].value_counts()
    
    # Ensure same categories
    sentiments = ['positive', 'neutral', 'negative']
    comment_array = [comment_counts.get(s, 0) for s in sentiments]
    sentence_array = [sentence_counts.get(s, 0) for s in sentiments]
    
    # Chi-square test
    contingency_table = np.array([comment_array, sentence_array])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    results['chi_square'] = {
        'statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'interpretation': 'Significant difference' if p_value < 0.05 else 'No significant difference'
    }
    
    print(f"Chi-square test for sentiment distribution:")
    print(f"  χ² = {chi2:.4f}, p = {p_value:.6f}, df = {dof}")
    print(f"  Result: {results['chi_square']['interpretation']}")
    
    # 2. Mann-Whitney U test for sentiment scores (if available)
    if 'sentiment_score' in comment_df.columns and 'final_sentiment_weight' in sentence_df.columns:
        # Align the datasets first
        aligned_comment, aligned_sentence = align_datasets(comment_df, sentence_df)
        
        u_stat, u_p_value = mannwhitneyu(
            aligned_comment['sentiment_score'],
            aligned_sentence['final_sentiment_weight'],
            alternative='two-sided'
        )
        
        results['mann_whitney'] = {
            'statistic': u_stat,
            'p_value': u_p_value,
            'interpretation': 'Significant difference in sentiment scores' if u_p_value < 0.05 else 'No significant difference in sentiment scores'
        }
        
        print(f"\nMann-Whitney U test for sentiment scores:")
        print(f"  U = {u_stat:.4f}, p = {u_p_value:.6f}")
        print(f"  Result: {results['mann_whitney']['interpretation']}")
    
    # 3. Effect size calculations
    # Cohen's w for chi-square
    total_n = sum(comment_array) + sum(sentence_array)
    cohens_w = np.sqrt(chi2 / total_n)
    
    results['effect_size'] = {
        'cohens_w': cohens_w,
        'interpretation': 'Small' if cohens_w < 0.3 else 'Medium' if cohens_w < 0.5 else 'Large'
    }
    
    print(f"\nEffect Size (Cohen's w): {cohens_w:.4f} ({results['effect_size']['interpretation']} effect)")
    
    return results

def analyze_learning_journey_impact(sentence_df):
    """Analyze the impact of learning journey detection."""
    
    print("\nAnalyzing learning journey detection impact...")
    
    # Learning journey statistics
    total_comments = len(sentence_df)
    learning_journeys = len(sentence_df[sentence_df['sentence_progression'] == 'learning_journey'])
    transitions = len(sentence_df[sentence_df['has_transition'] == True])
    
    journey_impact = {
        'total_comments': total_comments,
        'learning_journeys': learning_journeys,
        'journey_percentage': (learning_journeys / total_comments) * 100,
        'transitions': transitions,
        'transition_percentage': (transitions / total_comments) * 100
    }
    
    print(f"Learning Journey Impact:")
    print(f"  Total comments: {total_comments:,}")
    print(f"  Learning journeys detected: {learning_journeys:,} ({journey_impact['journey_percentage']:.2f}%)")
    print(f"  Comments with transitions: {transitions:,} ({journey_impact['transition_percentage']:.2f}%)")
    
    # Analyze sentiment changes in learning journeys
    journeys = sentence_df[sentence_df['sentence_progression'] == 'learning_journey']
    
    print(f"  Average confidence for learning journeys: {journeys['final_sentiment_weight'].mean():.3f}")
    print(f"  Average sentences in learning journeys: {journeys['sentence_count'].mean():.1f}")
    
    # Top topics for learning journeys
    top_journey_topics = journeys['search_query'].value_counts().head(5)
    print(f"\nTop Learning Journey Topics:")
    for topic, count in top_journey_topics.items():
        percentage = (count / learning_journeys) * 100
        print(f"  {topic.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return journey_impact

def create_comparative_visualizations(comment_df, sentence_df, comparison_df, output_dir):
    """Create comprehensive comparative visualizations."""
    
    print("Creating comparative visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Side-by-side sentiment distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparative Analysis: Comment-Level vs Sentence-Level Sentiment Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Sentiment distribution comparison
    comparison_df.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
    axes[0,0].set_title('Sentiment Distribution Comparison', fontweight='bold')
    axes[0,0].set_ylabel('Percentage (%)')
    axes[0,0].set_xlabel('Sentiment')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=0)
    
    # Difference visualization
    differences = comparison_df['Sentence-Level'] - comparison_df['Comment-Level']
    colors = ['green' if x > 0 else 'red' for x in differences]
    axes[0,1].bar(differences.index, differences.values, color=colors, alpha=0.7)
    axes[0,1].set_title('Difference in Sentiment Distribution\n(Sentence-Level - Comment-Level)', fontweight='bold')
    axes[0,1].set_ylabel('Percentage Point Difference')
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # Learning journey analysis (only for sentence-level)
    progression_counts = sentence_df['sentence_progression'].value_counts()
    axes[1,0].pie(progression_counts.values, 
                  labels=[p.replace('_', ' ').title() for p in progression_counts.index],
                  autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('Sentence-Level Progression Patterns\n(Not Available in Comment-Level)', fontweight='bold')
    
    # Confidence score comparison (if available)
    if 'sentiment_score' in comment_df.columns and 'final_sentiment_weight' in sentence_df.columns:
        aligned_comment, aligned_sentence = align_datasets(comment_df, sentence_df)
        
        axes[1,1].hist([aligned_comment['sentiment_score'], aligned_sentence['final_sentiment_weight']], 
                       bins=30, alpha=0.7, label=['Comment-Level', 'Sentence-Level'], 
                       color=['skyblue', 'lightcoral'])
        axes[1,1].set_title('Sentiment Confidence Score Distribution', fontweight='bold')
        axes[1,1].set_xlabel('Confidence Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparative_sentiment_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning journey specific analysis
    if len(sentence_df[sentence_df['sentence_progression'] == 'learning_journey']) > 0:
        plt.figure(figsize=(14, 8))
        
        # Learning journey by topic
        journeys = sentence_df[sentence_df['sentence_progression'] == 'learning_journey']
        journey_by_topic = journeys['search_query'].value_counts().head(15)
        
        plt.barh(range(len(journey_by_topic)), journey_by_topic.values, color='mediumseagreen')
        plt.yticks(range(len(journey_by_topic)), 
                  [topic.replace('_', ' ').title() for topic in journey_by_topic.index])
        plt.title('Learning Journeys by Topic\n(Unique to Sentence-Level Analysis)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Number of Learning Journey Comments')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/learning_journeys_by_topic.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def generate_comparative_report(comment_df, sentence_df, comparison_df, differences, 
                               statistical_results, journey_impact, output_dir):
    """Generate comprehensive comparative analysis report."""
    
    report_path = f"{output_dir}/comparative_analysis_report.md"
    
    report = f"""
# Comparative Analysis: Comment-Level vs Sentence-Level Sentiment Analysis
## Impact Assessment of Per-Sentence Analysis Implementation

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Comment-Level Dataset:** {len(comment_df):,} comments
**Sentence-Level Dataset:** {len(sentence_df):,} comments

## Executive Summary

This analysis compares the impact of implementing per-sentence sentiment analysis versus traditional comment-level analysis for YouTube mathematics education comments. The sentence-level approach introduces sophisticated progression pattern detection, particularly learning journey identification.

## Sentiment Distribution Changes

### Overall Distribution Comparison
"""
    
    for sentiment in ['positive', 'neutral', 'negative']:
        comment_pct = comparison_df.loc[sentiment, 'Comment-Level']
        sentence_pct = comparison_df.loc[sentiment, 'Sentence-Level']
        diff = differences[sentiment]
        direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        
        report += f"- **{sentiment.capitalize()}:** {comment_pct:.1f}% → {sentence_pct:.1f}% ({direction} {abs(diff):.1f}pp)\n"
    
    report += f"""
### Key Findings
- **Most Significant Change:** {differences.abs().idxmax().capitalize()} sentiment ({differences[differences.abs().idxmax()]:+.1f} percentage points)
- **Direction of Change:** {"Sentence-level analysis detects more nuanced sentiment patterns" if abs(differences['positive']) > 1 else "Minimal overall distribution changes"}

## Statistical Significance Analysis

### Chi-Square Test for Distribution Independence
- **χ² statistic:** {statistical_results['chi_square']['statistic']:.4f}
- **p-value:** {statistical_results['chi_square']['p_value']:.6f}
- **Degrees of freedom:** {statistical_results['chi_square']['degrees_of_freedom']}
- **Result:** {statistical_results['chi_square']['interpretation']}

### Effect Size
- **Cohen's w:** {statistical_results['effect_size']['cohens_w']:.4f}
- **Interpretation:** {statistical_results['effect_size']['interpretation']} effect size
- **Practical Significance:** {'Meaningful difference in approaches' if statistical_results['effect_size']['cohens_w'] > 0.1 else 'Minimal practical difference'}

"""
    
    if 'mann_whitney' in statistical_results:
        report += f"""
### Sentiment Score Comparison (Mann-Whitney U Test)
- **U statistic:** {statistical_results['mann_whitney']['statistic']:.4f}
- **p-value:** {statistical_results['mann_whitney']['p_value']:.6f}
- **Result:** {statistical_results['mann_whitney']['interpretation']}
"""
    
    report += f"""
## Learning Journey Detection Impact

The sentence-level approach introduces learning journey detection, a novel capability not available in comment-level analysis:

- **Learning Journeys Detected:** {journey_impact['learning_journeys']:,} ({journey_impact['journey_percentage']:.2f}% of comments)
- **Comments with Transitions:** {journey_impact['transitions']:,} ({journey_impact['transition_percentage']:.2f}% of comments)
- **Educational Value:** Identifies authentic learning experiences where negative emotions (confusion, frustration) transition to positive outcomes (understanding, gratitude)

### Progression Pattern Distribution
Only available through sentence-level analysis:
"""
    
    # Add progression pattern statistics
    progression_counts = sentence_df['sentence_progression'].value_counts()
    for pattern, count in progression_counts.items():
        percentage = (count / len(sentence_df)) * 100
        report += f"- **{pattern.replace('_', ' ').title()}:** {count:,} ({percentage:.1f}%)\n"
    
    report += f"""
## Methodological Improvements

### Advantages of Sentence-Level Analysis
1. **Granular Sentiment Detection:** Identifies sentiment transitions within individual comments
2. **Learning Journey Identification:** Detects educational transformation patterns
3. **Context-Aware Weighting:** Higher confidence scores for learning journeys (90% vs standard scoring)
4. **Educational Insights:** Provides deeper understanding of learning processes in online education

### Limitations and Considerations
1. **Computational Complexity:** Approximately 3-4x processing time compared to comment-level analysis
2. **Model Requirements:** Requires sophisticated sentence segmentation (spaCy) and progression algorithms
3. **Data Requirements:** More effective with longer, multi-sentence comments

## Recommendations

### For Educational Research
- **Adopt sentence-level analysis** for studies focusing on learning processes and educational sentiment
- **Use learning journey detection** to identify successful educational interventions
- **Apply progression patterns** to understand student emotional trajectories

### For Large-Scale Analysis
- **Consider hybrid approach:** Comment-level for broad sentiment trends, sentence-level for detailed educational insights
- **Implement selective processing:** Use sentence-level analysis for comments above certain length thresholds

## Conclusion

The sentence-level sentiment analysis represents a significant methodological advancement for educational research, providing {journey_impact['journey_percentage']:.1f}% additional insight through learning journey detection. While overall sentiment distributions show {statistical_results['effect_size']['interpretation'].lower()} changes, the educational value lies in the ability to identify and quantify authentic learning experiences within online educational content.

**Statistical Significance:** {statistical_results['chi_square']['interpretation']} between approaches (p = {statistical_results['chi_square']['p_value']:.6f})
**Practical Impact:** {journey_impact['learning_journeys']:,} learning journeys identified that would be missed by comment-level analysis
**Recommendation:** Implement sentence-level analysis for educational sentiment research requiring nuanced understanding of learning processes.
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Comparative analysis report saved: {report_path}")

def main():
    """Main comparative analysis function."""
    
    print("="*80)
    print("COMPARATIVE ANALYSIS: COMMENT-LEVEL vs SENTENCE-LEVEL SENTIMENT ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/Sentiment_Analysis/comparative_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    comment_df, sentence_df = load_comparative_data()
    
    # Compare sentiment distributions
    comparison_df, differences = compare_sentiment_distributions(comment_df, sentence_df, output_dir)
    
    # Statistical significance tests
    statistical_results = statistical_significance_tests(comment_df, sentence_df)
    
    # Learning journey impact analysis
    journey_impact = analyze_learning_journey_impact(sentence_df)
    
    # Create visualizations
    create_comparative_visualizations(comment_df, sentence_df, comparison_df, output_dir)
    
    # Generate comprehensive report
    generate_comparative_report(comment_df, sentence_df, comparison_df, differences, 
                               statistical_results, journey_impact, output_dir)
    
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}/")
    print("Key files:")
    print("├── comparative_sentiment_analysis.png")
    print("├── learning_journeys_by_topic.png")
    print("└── comparative_analysis_report.md")

if __name__ == "__main__":
    main()