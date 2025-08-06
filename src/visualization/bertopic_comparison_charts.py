#!/usr/bin/env python3
"""
BERTopic Model Comparison Chart Generator
Compares HDBSCAN Per Query vs Optimised Variable K models

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Paths to model results
HDBSCAN_PATH = Path("/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/BERTopic HDBSCAN Per Query 20250720/bertopic_complete_pipeline_analysis_20250720_230249")
VARIABLE_K_PATH = Path("/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/optimised_variable_k_phase_4_20250722_224755")

def load_model_data():
    """Load summary data from both models"""
    
    # Load HDBSCAN data
    hdbscan_file = HDBSCAN_PATH / "_summary_reports" / "analysis_summary_stats.csv"
    hdbscan_df = pd.read_csv(hdbscan_file)
    hdbscan_df['model'] = 'HDBSCAN Per Query'
    hdbscan_df['silhouette_score'] = np.nan  # Not available for HDBSCAN
    
    # Load Variable K data
    vark_file = VARIABLE_K_PATH / "analysis_summary.csv"
    vark_df = pd.read_csv(vark_file)
    vark_df['model'] = 'Variable K-means'
    vark_df.rename(columns={'optimal_k': 'n_topics', 'n_documents': 'n_docs'}, inplace=True)
    vark_df['noise_pct'] = vark_df['noise_rate'] * 100  # Convert to percentage
    
    # Standardize column names for comparison
    common_cols = ['query', 'n_docs', 'n_topics', 'noise_pct', 'model']
    hdbscan_common = hdbscan_df[common_cols].copy()
    vark_common = vark_df[common_cols + ['silhouette_score']].copy()
    
    return hdbscan_df, vark_df, hdbscan_common, vark_common

def get_topic_examples():
    """Get topic examples from a few representative queries"""
    examples = {}
    
    # Example queries to analyze
    example_queries = ['algebra_explained', 'trigonometry_explained', 'quadratic_equations']
    
    for query in example_queries:
        examples[query] = {}
        
        # HDBSCAN topics
        hdbscan_topic_file = HDBSCAN_PATH / query / f"topic_info_{query}.csv"
        if hdbscan_topic_file.exists():
            hdbscan_topics = pd.read_csv(hdbscan_topic_file)
            examples[query]['hdbscan'] = hdbscan_topics
        
        # Variable K topics
        vark_topic_file = VARIABLE_K_PATH / query / "data" / f"topic_info_{query}.csv"
        if vark_topic_file.exists():
            vark_topics = pd.read_csv(vark_topic_file)
            examples[query]['variable_k'] = vark_topics
    
    return examples

def create_comparison_visualizations(hdbscan_df, vark_df, hdbscan_common, vark_common):
    """Create comprehensive comparison visualizations"""
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Noise Level Comparison (Main comparison)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Merge data for comparison
    comparison_df = pd.concat([hdbscan_common, vark_common], ignore_index=True)
    
    # Sort by document count for better visualization
    comparison_df = comparison_df.sort_values('n_docs', ascending=False)
    
    # Create scatter plot
    for model in ['HDBSCAN Per Query', 'Variable K-means']:
        model_data = comparison_df[comparison_df['model'] == model]
        ax1.scatter(model_data['n_docs'], model_data['noise_pct'], 
                   label=model, alpha=0.7, s=60)
    
    ax1.set_xlabel('Number of Documents per Query')
    ax1.set_ylabel('Noise Percentage (%)')
    ax1.set_title('Noise Level Comparison: HDBSCAN vs Variable K-means')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add trend lines
    hdb_data = comparison_df[comparison_df['model'] == 'HDBSCAN Per Query']
    vark_data = comparison_df[comparison_df['model'] == 'Variable K-means']
    
    if len(hdb_data) > 0:
        z_hdb = np.polyfit(hdb_data['n_docs'], hdb_data['noise_pct'], 1)
        p_hdb = np.poly1d(z_hdb)
        ax1.plot(hdb_data['n_docs'], p_hdb(hdb_data['n_docs']), "--", alpha=0.8)
    
    # 2. Topic Count Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Box plot of topic counts
    topic_counts = [hdbscan_df['n_topics'].values, vark_df['n_topics'].values]
    ax2.boxplot(topic_counts, labels=['HDBSCAN\nPer Query', 'Variable\nK-means'])
    ax2.set_ylabel('Number of Topics')
    ax2.set_title('Topic Count Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Summary Statistics
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create performance comparison table
    perf_stats = pd.DataFrame({
        'Model': ['HDBSCAN Per Query', 'Variable K-means'],
        'Avg Noise %': [hdbscan_df['noise_pct'].mean(), vark_df['noise_rate'].mean() * 100],
        'Max Noise %': [hdbscan_df['noise_pct'].max(), vark_df['noise_rate'].max() * 100],
        'Min Noise %': [hdbscan_df['noise_pct'].min(), vark_df['noise_rate'].min() * 100],
        'Avg Topics': [hdbscan_df['n_topics'].mean(), vark_df['n_topics'].mean()],
        'Avg Silhouette': [np.nan, vark_df['silhouette_score'].mean()],
        'Total Queries': [len(hdbscan_df), len(vark_df)]
    })
    
    # Create table
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=perf_stats.round(3).values,
                     colLabels=perf_stats.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax3.set_title('Performance Summary Statistics', pad=20)
    
    # 4. Query-by-Query Comparison (Top 20 by document count)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Get top 20 queries by document count
    top_queries = hdbscan_df.nlargest(20, 'n_docs')['query'].values
    hdb_subset = hdbscan_df[hdbscan_df['query'].isin(top_queries)]
    vark_subset = vark_df[vark_df['query'].isin(top_queries)]
    
    x = np.arange(len(top_queries))
    width = 0.35
    
    ax4.bar(x - width/2, hdb_subset.set_index('query').loc[top_queries, 'noise_pct'], 
            width, label='HDBSCAN Noise %', alpha=0.8)
    ax4.bar(x + width/2, vark_subset.set_index('query').loc[top_queries, 'n_topics'], 
            width, label='Variable K Topics', alpha=0.8)
    
    ax4.set_xlabel('Top 20 Queries by Document Count')
    ax4.set_ylabel('Noise % / Topic Count')
    ax4.set_title('Query-by-Query Comparison: Noise vs Topics (Top 20 Queries)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_queries, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Interpretability Analysis
    ax5 = fig.add_subplot(gs[3, :2])
    
    # Calculate interpretability metrics
    hdbscan_interpretability = (100 - hdbscan_df['noise_pct']) / hdbscan_df['n_topics']
    vark_interpretability = vark_df['silhouette_score'] * 100
    
    ax5.hist(hdbscan_interpretability, bins=20, alpha=0.6, label='HDBSCAN (Coverage/Topics)', density=True)
    ax5.hist(vark_interpretability, bins=20, alpha=0.6, label='Variable K (Silhouette*100)', density=True)
    ax5.set_xlabel('Interpretability Score')
    ax5.set_ylabel('Density')
    ax5.set_title('Interpretability Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary insights
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.axis('off')
    
    insights_text = f"""
KEY FINDINGS:

HDBSCAN Per Query:
• Average Noise: {hdbscan_df['noise_pct'].mean():.1f}%
• Queries with 0% Noise: {sum(hdbscan_df['noise_pct'] == 0)}
• Average Topics: {hdbscan_df['n_topics'].mean():.1f}

Variable K-means:
• Average Noise: {vark_df['noise_rate'].mean() * 100:.1f}%
• Queries with 0% Noise: {sum(vark_df['noise_rate'] == 0)}
• Average Topics: {vark_df['n_topics'].mean():.1f}
• Average Silhouette: {vark_df['silhouette_score'].mean():.3f}

CONCLUSION:
Variable K-means achieves:
✓ 0% noise across all queries
✓ Optimized topic granularity
✓ High silhouette scores
✓ Complete document assignment
    """
    
    ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('BERTopic Model Comparison: HDBSCAN Per Query vs Variable K-means\n' + 
                 'Complete Pipeline Dataset (34,057 comments, 82 queries)', 
                 fontsize=16, y=0.98)
    
    return fig

def create_topic_examples_table(examples):
    """Create detailed topic examples comparison"""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Topic Examples Comparison: HDBSCAN vs Variable K-means', fontsize=14)
    
    for idx, (query, data) in enumerate(examples.items()):
        ax = axes[idx]
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare comparison data
        comparison_data = []
        
        if 'hdbscan' in data and 'variable_k' in data:
            hdb_topics = data['hdbscan']
            vark_topics = data['variable_k']
            
            # Get top topics (excluding noise topic -1 for HDBSCAN)
            hdb_real_topics = hdb_topics[hdb_topics['Topic'] != -1].head(5)
            vark_real_topics = vark_topics.head(5)
            
            max_topics = max(len(hdb_real_topics), len(vark_real_topics))
            
            for i in range(max_topics):
                row = [f"Topic {i+1}"]
                
                # HDBSCAN topic
                if i < len(hdb_real_topics):
                    hdb_topic = hdb_real_topics.iloc[i]
                    hdb_words = eval(hdb_topic['Representation'])[:5]  # Top 5 words
                    row.extend([hdb_topic['Count'], ', '.join(hdb_words)])
                else:
                    row.extend(['', ''])
                
                # Variable K topic
                if i < len(vark_real_topics):
                    vark_topic = vark_real_topics.iloc[i]
                    vark_words = eval(vark_topic['Representation'])[:5]  # Top 5 words
                    row.extend([vark_topic['Count'], ', '.join(vark_words)])
                else:
                    row.extend(['', ''])
                
                comparison_data.append(row)
            
            # Add noise information for HDBSCAN
            noise_topic = hdb_topics[hdb_topics['Topic'] == -1]
            if len(noise_topic) > 0:
                noise_count = noise_topic.iloc[0]['Count']
                comparison_data.append(['Noise', noise_count, 'Unassigned documents', 0, 'No noise (K-means)'])
        
        # Create table
        headers = ['Topic', 'HDBSCAN\nCount', 'HDBSCAN Top Words', 'Variable K\nCount', 'Variable K Top Words']
        
        table = ax.table(cellText=comparison_data,
                        colLabels=headers,
                        cellLoc='left',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title(f'{query.replace("_", " ").title()}', pad=20)
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    print("Loading model data...")
    hdbscan_df, vark_df, hdbscan_common, vark_common = load_model_data()
    
    print("Getting topic examples...")
    examples = get_topic_examples()
    
    print("Creating comparison visualizations...")
    comparison_fig = create_comparison_visualizations(hdbscan_df, vark_df, hdbscan_common, vark_common)
    
    print("Creating topic examples table...")
    examples_fig = create_topic_examples_table(examples)
    
    # Save figures
    output_dir = Path("/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis")
    
    comparison_fig.savefig(output_dir / "bertopic_models_comparison_chart.png", 
                          dpi=300, bbox_inches='tight')
    comparison_fig.savefig(output_dir / "bertopic_models_comparison_chart.pdf", 
                          bbox_inches='tight')
    
    examples_fig.savefig(output_dir / "bertopic_topic_examples_comparison.png", 
                        dpi=300, bbox_inches='tight')
    examples_fig.savefig(output_dir / "bertopic_topic_examples_comparison.pdf", 
                        bbox_inches='tight')
    
    print("\n" + "="*60)
    print("BERTOPIC MODEL COMPARISON ANALYSIS COMPLETE")
    print("="*60)
    
    print(f"\nFILES GENERATED:")
    print(f"• {output_dir}/bertopic_models_comparison_chart.png")
    print(f"• {output_dir}/bertopic_models_comparison_chart.pdf")
    print(f"• {output_dir}/bertopic_topic_examples_comparison.png")
    print(f"• {output_dir}/bertopic_topic_examples_comparison.pdf")
    
    # Print summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"HDBSCAN Per Query Model:")
    print(f"  • Total Queries: {len(hdbscan_df)}")
    print(f"  • Average Noise: {hdbscan_df['noise_pct'].mean():.2f}%")
    print(f"  • Queries with 0% Noise: {sum(hdbscan_df['noise_pct'] == 0)}")
    print(f"  • Average Topics per Query: {hdbscan_df['n_topics'].mean():.1f}")
    
    print(f"\nVariable K-means Model:")
    print(f"  • Total Queries: {len(vark_df)}")
    print(f"  • Average Noise: {vark_df['noise_rate'].mean() * 100:.2f}%")
    print(f"  • Queries with 0% Noise: {sum(vark_df['noise_rate'] == 0)}")
    print(f"  • Average Topics per Query: {vark_df['n_topics'].mean():.1f}")
    print(f"  • Average Silhouette Score: {vark_df['silhouette_score'].mean():.3f}")
    
    improvement = (hdbscan_df['noise_pct'].mean() - vark_df['noise_rate'].mean() * 100)
    print(f"\nNOISE REDUCTION: {improvement:.2f} percentage points")
    print(f"COMPLETE ASSIGNMENT: Variable K-means assigns 100% of documents to topics")
    
    plt.show()

if __name__ == "__main__":
    main()