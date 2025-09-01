#!/usr/bin/env python3
"""
Final Analysis Plots for Complete Pipeline BERTopic Methodology Evolution


Creates focused plots that demonstrate:
1. Kernel K-means performance by kernel type
2. Best performing clustering algorithm per query (pie chart)
3. Overall silhouette score comparison to identify the final method
4. Performance analysis for final algorithm selection

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set professional academic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class FinalAnalysisPlots:
    def __init__(self, base_dir):
        """Initialize final analysis plot generator."""
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save to results/figures directory as requested
        self.output_dir = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/figures"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Final Analysis Plot Generator")
        print("Creating definitive performance analysis plots")
        print(f"Output directory: {self.output_dir}")
        
        # Load data
        self.phase3_data = None
        self.phase4_data = None
        self.load_data()
        
    def load_data(self):
        """Load Phase 3 clustering comparison results."""
        try:
            # Load Phase 3 results
            phase3_dirs = glob.glob(f"{self.base_dir}/complete_pipeline_clustering_comparison_*")
            if phase3_dirs:
                latest_phase3_dir = max(phase3_dirs)
                phase3_files = glob.glob(f"{latest_phase3_dir}/clustering_comparison_detailed_*.csv")
                if phase3_files:
                    latest_phase3_file = max(phase3_files)
                    self.phase3_data = pd.read_csv(latest_phase3_file)
                    print(f"✓ Loaded Phase 3 data: {len(self.phase3_data)} records")
                    
            # Load Phase 4 results for supplementary analysis
            phase4_dirs = glob.glob(f"{self.base_dir}/complete_pipeline_variable_k_*")
            if phase4_dirs:
                latest_phase4_dir = max(phase4_dirs)
                phase4_files = glob.glob(f"{latest_phase4_dir}/variable_k_detailed_*.csv")
                if phase4_files:
                    latest_phase4_file = max(phase4_files)
                    self.phase4_data = pd.read_csv(latest_phase4_file)
                    print(f"✓ Loaded Phase 4 data: {len(self.phase4_data)} records")
                        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def create_kernel_performance_analysis(self):
        """Create comprehensive Kernel K-means performance analysis."""
        if self.phase3_data is None:
            return
            
        # Extract kernel data
        kernel_data = self.phase3_data[self.phase3_data['algorithm_family'] == 'Kernel_KMeans'].copy()
        if kernel_data.empty:
            print("No Kernel K-means data available")
            return
            
        # Parse kernel types and K values
        kernel_data['kernel_type'] = kernel_data['algorithm'].str.extract(r'_([^_]+)$')[0]
        kernel_data['k_value'] = kernel_data['algorithm'].str.extract(r'_K(\d+)_')[0].astype(int)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Kernel K-means Performance Analysis\nComplete Pipeline Dataset', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Kernel Type Performance Comparison (TOP LEFT)
        kernel_performance = kernel_data.groupby('kernel_type').agg({
            'silhouette_score': ['mean', 'std', 'count', 'min', 'max']
        }).round(4)
        
        kernel_types = kernel_performance.index
        means = kernel_performance[('silhouette_score', 'mean')]
        stds = kernel_performance[('silhouette_score', 'std')]
        counts = kernel_performance[('silhouette_score', 'count')]
        
        colors = ['#FF6B35', '#4ECDC4', '#45B7D1']  # Orange, Teal, Blue
        bars1 = ax1.bar(kernel_types, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
        
        ax1.set_title('Kernel Type Performance Comparison\n(Average Silhouette Score)', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Kernel Type', fontsize=12)
        ax1.set_ylabel('Average Silhouette Score', fontsize=12)
        ax1.set_xticklabels([k.upper() for k in kernel_types], fontsize=11)
        
        # Add value labels with sample sizes
        for i, (bar, mean_val, std_val, count) in enumerate(zip(bars1, means, stds, counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean_val:.3f}±{std_val:.3f}\n(n={count})', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(means) * 1.15)
        
        # 2. Kernel Performance by K Value (TOP RIGHT)
        k_values = sorted(kernel_data['k_value'].unique())
        kernel_colors = {'linear': '#FF6B35', 'rbf': '#4ECDC4', 'poly': '#45B7D1'}
        
        for kernel in kernel_types:
            kernel_subset = kernel_data[kernel_data['kernel_type'] == kernel]
            k_performance = kernel_subset.groupby('k_value')['silhouette_score'].mean()
            
            ax2.plot(k_performance.index, k_performance.values, 
                    'o-', label=f'{kernel.upper()} Kernel', linewidth=3, markersize=8,
                    color=kernel_colors.get(kernel, 'gray'))
        
        ax2.set_title('Silhouette Score by K Value\n(Kernel Type Comparison)', 
                     fontweight='bold', fontsize=14)
        ax2.set_xlabel('K Value (Number of Topics)', fontsize=12)
        ax2.set_ylabel('Average Silhouette Score', fontsize=12)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(k_values)
        
        # 3. Kernel Performance Distribution (BOTTOM LEFT)
        kernel_data_list = []
        kernel_labels = []
        
        for kernel in kernel_types:
            kernel_subset = kernel_data[kernel_data['kernel_type'] == kernel]
            kernel_data_list.append(kernel_subset['silhouette_score'].values)
            kernel_labels.append(kernel.upper())
        
        box_plot = ax3.boxplot(kernel_data_list, labels=kernel_labels, patch_artist=True)
        ax3.set_title('Silhouette Score Distribution\n(Kernel Type Comparison)', 
                     fontweight='bold', fontsize=14)
        ax3.set_xlabel('Kernel Type', fontsize=12)
        ax3.set_ylabel('Silhouette Score', fontsize=12)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Best Kernel by K Value (BOTTOM RIGHT)
        best_kernel_by_k = []
        k_values_for_best = []
        
        for k in k_values:
            k_subset = kernel_data[kernel_data['k_value'] == k]
            if not k_subset.empty:
                best_kernel_idx = k_subset['silhouette_score'].idxmax()
                best_kernel = k_subset.loc[best_kernel_idx, 'kernel_type']
                best_kernel_by_k.append(best_kernel)
                k_values_for_best.append(k)
        
        # Create stacked bar chart showing kernel dominance
        kernel_dominance = pd.Series(best_kernel_by_k).value_counts()
        
        bars4 = ax4.bar(kernel_dominance.index, kernel_dominance.values, 
                       color=[kernel_colors.get(k, 'gray') for k in kernel_dominance.index],
                       alpha=0.8)
        
        ax4.set_title('Best Performing Kernel by K Value\n(Frequency of Optimal Performance)', 
                     fontweight='bold', fontsize=14)
        ax4.set_xlabel('Kernel Type', fontsize=12)
        ax4.set_ylabel('Number of K Values Where Best', fontsize=12)
        ax4.set_xticklabels([k.upper() for k in kernel_dominance.index])
        
        # Add value labels
        for bar, value in zip(bars4, kernel_dominance.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/kernel_kmeans_performance_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Kernel K-means performance analysis created")
    
    def create_best_algorithm_distribution(self):
        """Create pie chart showing distribution of best performing algorithms per query."""
        if self.phase3_data is None:
            return
            
        # Find best algorithm per query
        best_per_query = self.phase3_data.loc[self.phase3_data.groupby('query')['silhouette_score'].idxmax()]
        
        # Count by algorithm family
        algorithm_counts = best_per_query['algorithm_family'].value_counts()
        
        # Create more detailed breakdown for Kernel K-means
        detailed_counts = {}
        for _, row in best_per_query.iterrows():
            if row['algorithm_family'] == 'Kernel_KMeans':
                kernel_type = row['algorithm'].split('_')[-1]  # Extract kernel type
                detailed_counts[f'Kernel K-means ({kernel_type.upper()})'] = detailed_counts.get(f'Kernel K-means ({kernel_type.upper()})', 0) + 1
            else:
                algorithm_name = row['algorithm_family'].replace('_', ' ')
                detailed_counts[algorithm_name] = detailed_counts.get(algorithm_name, 0) + 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Best Performing Clustering Algorithms\nDistribution Across Queries', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Algorithm Family Distribution (LEFT)
        colors1 = plt.cm.Set3(np.linspace(0, 1, len(algorithm_counts)))
        wedges1, texts1, autotexts1 = ax1.pie(algorithm_counts.values, 
                                              labels=algorithm_counts.index, 
                                              autopct='%1.1f%%', 
                                              startangle=90, 
                                              colors=colors1,
                                              textprops={'fontsize': 12})
        
        ax1.set_title('Best Algorithm by Family\n(Percentage of Queries)', fontweight='bold', fontsize=14)
        
        # Enhance text visibility
        for autotext in autotexts1:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        # 2. Detailed Algorithm Distribution (RIGHT) 
        detailed_series = pd.Series(detailed_counts)
        colors2 = plt.cm.Set2(np.linspace(0, 1, len(detailed_series)))
        
        wedges2, texts2, autotexts2 = ax2.pie(detailed_series.values, 
                                              labels=detailed_series.index, 
                                              autopct='%1.1f%%', 
                                              startangle=90, 
                                              colors=colors2,
                                              textprops={'fontsize': 11})
        
        ax2.set_title('Detailed Algorithm Distribution\n(Including Kernel Types)', fontweight='bold', fontsize=14)
        
        # Enhance text visibility
        for autotext in autotexts2:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/best_algorithm_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Best algorithm distribution pie charts created")
    
    def create_overall_performance_ranking(self):
        """Create comprehensive ranking of all clustering algorithms."""
        if self.phase3_data is None:
            return
            
        # Calculate comprehensive statistics for each algorithm
        algorithm_stats = self.phase3_data.groupby('algorithm').agg({
            'silhouette_score': ['mean', 'std', 'count', 'min', 'max', 'median']
        }).round(4)
        
        # Flatten column names
        algorithm_stats.columns = ['mean', 'std', 'count', 'min', 'max', 'median']
        algorithm_stats = algorithm_stats.sort_values('mean', ascending=False)
        
        # Add algorithm family information
        algorithm_stats['family'] = self.phase3_data.groupby('algorithm')['algorithm_family'].first()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Comprehensive Clustering Algorithm Performance Ranking\nComplete Pipeline Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Top 15 Algorithms by Mean Silhouette Score (TOP LEFT)
        top_15 = algorithm_stats.head(15)
        
        # Color by family
        family_colors = {
            'HDBSCAN': '#FF6B6B',
            'KMeans': '#4ECDC4', 
            'Kernel_KMeans': '#45B7D1',
            'GMM': '#96CEB4'
        }
        
        colors = [family_colors.get(family, 'gray') for family in top_15['family']]
        
        bars1 = ax1.barh(range(len(top_15)), top_15['mean'], 
                        xerr=top_15['std'], capsize=3, alpha=0.8, color=colors)
        
        ax1.set_title('Top 15 Algorithms by Silhouette Score\n(Mean ± Standard Deviation)', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Average Silhouette Score', fontsize=12)
        ax1.set_yticks(range(len(top_15)))
        ax1.set_yticklabels([alg.replace('_', ' ') for alg in top_15.index], fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, mean_val, std_val) in enumerate(zip(bars1, top_15['mean'], top_15['std'])):
            ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{mean_val:.3f}±{std_val:.3f}', 
                    va='center', fontweight='bold', fontsize=9)
        
        # 2. Algorithm Family Performance Summary (TOP RIGHT)
        family_stats = self.phase3_data.groupby('algorithm_family').agg({
            'silhouette_score': ['mean', 'std', 'count']
        }).round(4)
        
        family_stats.columns = ['mean', 'std', 'count']
        family_stats = family_stats.sort_values('mean', ascending=False)
        
        bars2 = ax2.bar(range(len(family_stats)), family_stats['mean'], 
                       yerr=family_stats['std'], capsize=5, alpha=0.8,
                       color=[family_colors.get(family, 'gray') for family in family_stats.index])
        
        ax2.set_title('Algorithm Family Performance Summary\n(Average Silhouette Score)', 
                     fontweight='bold', fontsize=14)
        ax2.set_ylabel('Average Silhouette Score', fontsize=12)
        ax2.set_xticks(range(len(family_stats)))
        ax2.set_xticklabels([f.replace('_', '\n') for f in family_stats.index], fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels with counts
        for i, (bar, mean_val, std_val, count) in enumerate(zip(bars2, family_stats['mean'], 
                                                                family_stats['std'], family_stats['count'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean_val:.3f}±{std_val:.3f}\n(n={count})', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. Performance Distribution by Family (BOTTOM LEFT)
        families = family_stats.index
        family_data = []
        family_labels = []
        
        for family in families:
            family_subset = self.phase3_data[self.phase3_data['algorithm_family'] == family]
            family_data.append(family_subset['silhouette_score'].values)
            family_labels.append(family.replace('_', '\n'))
        
        violin_parts = ax3.violinplot(family_data, positions=range(len(families)), 
                                     showmeans=True, showextrema=True)
        
        ax3.set_title('Silhouette Score Distribution by Algorithm Family\n(Violin Plot)', 
                     fontweight='bold', fontsize=14)
        ax3.set_ylabel('Silhouette Score', fontsize=12)
        ax3.set_xticks(range(len(families)))
        ax3.set_xticklabels(family_labels, fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Color violin plots
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(family_colors.get(families[i], 'gray'))
            pc.set_alpha(0.7)
        
        # 4. Final Recommendation Panel (BOTTOM RIGHT)
        ax4.axis('off')
        
        # Determine best overall algorithm
        best_algorithm = algorithm_stats.index[0]
        best_score = algorithm_stats['mean'].iloc[0]
        best_family = algorithm_stats['family'].iloc[0]
        
        # Get best family
        best_family_overall = family_stats.index[0]
        best_family_score = family_stats['mean'].iloc[0]
        
        # Performance statistics
        total_algorithms = len(algorithm_stats)
        total_configurations = len(self.phase3_data)
        
        recommendation_text = f"FINAL ALGORITHM RECOMMENDATION\n\n"
        recommendation_text += f"BEST OVERALL ALGORITHM:\n"
        recommendation_text += f"Algorithm: {best_algorithm.replace('_', ' ')}\n"
        recommendation_text += f"Family: {best_family.replace('_', ' ')}\n"
        recommendation_text += f"Silhouette Score: {best_score:.3f}\n\n"
        
        recommendation_text += f"BEST ALGORITHM FAMILY:\n"
        recommendation_text += f"Family: {best_family_overall.replace('_', ' ')}\n"
        recommendation_text += f"Average Score: {best_family_score:.3f}\n\n"
        
        recommendation_text += f"ANALYSIS SUMMARY:\n"
        recommendation_text += f"• Total Algorithms Tested: {total_algorithms}\n"
        recommendation_text += f"• Total Configurations: {total_configurations}\n"
        recommendation_text += f"• Queries Analyzed: {len(self.phase3_data['query'].unique())}\n\n"
        
        # Performance insights
        kernel_subset = algorithm_stats[algorithm_stats['family'] == 'Kernel_KMeans']
        if not kernel_subset.empty:
            best_kernel_alg = kernel_subset.index[0]
            kernel_type = best_kernel_alg.split('_')[-1]
            recommendation_text += f"KERNEL K-MEANS INSIGHTS:\n"
            recommendation_text += f"• Best Kernel: {kernel_type.upper()}\n"
            recommendation_text += f"• Score: {kernel_subset['mean'].iloc[0]:.3f}\n\n"
        
        recommendation_text += f"RECOMMENDED FOR FINAL ANALYSIS:\n"
        recommendation_text += f"→ {best_family_overall.replace('_', ' ')}\n"
        recommendation_text += f"→ Consistent high performance\n"
        recommendation_text += f"→ Reliable topic separation\n"
        recommendation_text += f"→ Suitable for thesis analysis"
        
        ax4.text(0.05, 0.95, recommendation_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.3),
                fontweight='bold', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/overall_performance_ranking.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Overall performance ranking analysis created")
    
    def create_statistical_comparison(self):
        """Create statistical comparison of top performing algorithms."""
        if self.phase3_data is None:
            return
            
        # Get top performing algorithms from each family
        family_best = self.phase3_data.loc[self.phase3_data.groupby('algorithm_family')['silhouette_score'].idxmax()]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Performance Comparison\nTop Algorithms for Final Topic Modeling Selection', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Statistical Summary Table (TOP LEFT)
        ax1.axis('off')
        
        # Calculate detailed statistics for top algorithms
        top_algorithms = []
        for family in self.phase3_data['algorithm_family'].unique():
            family_data = self.phase3_data[self.phase3_data['algorithm_family'] == family]
            family_stats = family_data.groupby('algorithm')['silhouette_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
            top_algorithms.append(family_stats.index[0])  # Best from each family
        
        # Create statistical table
        stats_data = []
        for alg in top_algorithms:
            alg_data = self.phase3_data[self.phase3_data['algorithm'] == alg]
            if not alg_data.empty:
                stats = {
                    'Algorithm': alg.replace('_', ' '),
                    'Mean': f"{alg_data['silhouette_score'].mean():.3f}",
                    'Std': f"{alg_data['silhouette_score'].std():.3f}",
                    'Min': f"{alg_data['silhouette_score'].min():.3f}",
                    'Max': f"{alg_data['silhouette_score'].max():.3f}",
                    'Count': str(len(alg_data))
                }
                stats_data.append(stats)
        
        # Display as formatted table
        table_text = "STATISTICAL PERFORMANCE SUMMARY\n"
        table_text += "Top Algorithm from Each Family\n\n"
        table_text += f"{'Algorithm':<25} {'Mean':<6} {'Std':<6} {'Min':<6} {'Max':<6} {'N':<4}\n"
        table_text += "-" * 65 + "\n"
        
        for stats in stats_data:
            table_text += f"{stats['Algorithm']:<25} {stats['Mean']:<6} {stats['Std']:<6} "
            table_text += f"{stats['Min']:<6} {stats['Max']:<6} {stats['Count']:<4}\n"
        
        ax1.text(0.05, 0.95, table_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                fontfamily='monospace')
        
        ax1.set_title('Statistical Summary of Top Algorithms', fontweight='bold', fontsize=14)
        
        # 2. Performance Comparison Bar Chart (TOP RIGHT)
        comparison_data = []
        comparison_labels = []
        comparison_colors = []
        
        family_colors = {
            'HDBSCAN': '#FF6B6B',
            'KMeans': '#4ECDC4', 
            'Kernel_KMeans': '#45B7D1',
            'GMM': '#96CEB4'
        }
        
        for stats in stats_data:
            comparison_data.append(float(stats['Mean']))
            comparison_labels.append(stats['Algorithm'])
            # Determine family color
            for family, color in family_colors.items():
                if family.replace('_', ' ').lower() in stats['Algorithm'].lower():
                    comparison_colors.append(color)
                    break
            else:
                comparison_colors.append('gray')
        
        bars2 = ax2.bar(range(len(comparison_data)), comparison_data, 
                       color=comparison_colors, alpha=0.8)
        
        ax2.set_title('Top Algorithm Performance Comparison\n(Mean Silhouette Score)', 
                     fontweight='bold', fontsize=14)
        ax2.set_ylabel('Average Silhouette Score', fontsize=12)
        ax2.set_xticks(range(len(comparison_labels)))
        ax2.set_xticklabels([label.replace(' ', '\n') for label in comparison_labels], 
                           fontsize=10, rotation=0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars2, comparison_data):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Highlight the best performer
        best_idx = np.argmax(comparison_data)
        bars2[best_idx].set_edgecolor('red')
        bars2[best_idx].set_linewidth(3)
        
        # 3. Performance by Query Analysis (BOTTOM LEFT)
        # Show consistency across queries
        query_performance = {}
        for family in self.phase3_data['algorithm_family'].unique():
            family_data = self.phase3_data[self.phase3_data['algorithm_family'] == family]
            query_scores = family_data.groupby('query')['silhouette_score'].max()  # Best from family per query
            query_performance[family.replace('_', ' ')] = query_scores.values
        
        # Create box plot
        performance_data = list(query_performance.values())
        performance_labels = list(query_performance.keys())
        
        box_plot = ax3.boxplot(performance_data, labels=performance_labels, patch_artist=True,
                               showmeans=False)
        ax3.set_title('Performance Consistency Across Queries\n(Best Algorithm per Family per Query)', 
                     fontweight='bold', fontsize=14)
        ax3.set_ylabel('Silhouette Score', fontsize=12)
        ax3.set_xlabel('Algorithm Family', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # Color boxes
        colors = [family_colors.get(label.replace(' ', '_'), 'gray') for label in performance_labels]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add median value labels 
        for i, data in enumerate(performance_data):
            median_value = np.median(data)
            # Place median label to the right of each box, aligned with median line
            ax3.text(i + 1.25, median_value, f'{median_value:.3f}', 
                    ha='left', va='center', fontweight='bold', fontsize=9,
                    color='black')
        
        # Add legend for median line
        ax3.plot([], [], marker='_', color='orange', markersize=10, linewidth=3, linestyle='none', label='Median')
        ax3.legend(loc='upper right', fontsize=10)
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Final Recommendation (BOTTOM RIGHT)
        ax4.axis('off')
        
        # Determine final recommendation
        best_overall_idx = np.argmax(comparison_data)
        best_algorithm_name = comparison_labels[best_overall_idx]
        best_score = comparison_data[best_overall_idx]
        
        # Calculate additional metrics
        best_alg_data = self.phase3_data[self.phase3_data['algorithm'].str.replace('_', ' ') == best_algorithm_name]
        if best_alg_data.empty:
            # Try alternative matching
            for alg in self.phase3_data['algorithm'].unique():
                if alg.replace('_', ' ').lower() == best_algorithm_name.lower():
                    best_alg_data = self.phase3_data[self.phase3_data['algorithm'] == alg]
                    break
        
        recommendation_text = f"FINAL RECOMMENDATION FOR THESIS\n\n"
        recommendation_text += f"SELECTED ALGORITHM:\n"
        recommendation_text += f"→ {best_algorithm_name}\n"
        recommendation_text += f"→ Mean Silhouette Score: {best_score:.3f}\n"
        
        if not best_alg_data.empty:
            recommendation_text += f"→ Standard Deviation: {best_alg_data['silhouette_score'].std():.3f}\n"
            recommendation_text += f"→ Performance Range: {best_alg_data['silhouette_score'].min():.3f} - {best_alg_data['silhouette_score'].max():.3f}\n"
            recommendation_text += f"→ Queries Tested: {len(best_alg_data)}\n\n"
        
        recommendation_text += f"JUSTIFICATION:\n"
        recommendation_text += f"• Highest average silhouette score\n"
        recommendation_text += f"• Consistent performance across queries\n"
        recommendation_text += f"• Reliable topic separation quality\n"
        recommendation_text += f"• Suitable for final topic modeling\n\n"
        
        recommendation_text += f"NEXT STEPS:\n"
        recommendation_text += f"1. Apply {best_algorithm_name} to complete dataset\n"
        recommendation_text += f"2. Generate final topic model\n"
        recommendation_text += f"3. Proceed with sentiment analysis\n"
        recommendation_text += f"4. Integrate results into thesis analysis"
        
        ax4.text(0.05, 0.95, recommendation_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.3),
                fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/statistical_comparison_final.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("✓ Statistical comparison for final selection created")
    
    def generate_final_plots(self):
        """Generate all final analysis plots."""
        print("\n" + "="*80)
        print("GENERATING FINAL ANALYSIS PLOTS")
        print("Focused performance analysis for algorithm selection")
        print("="*80)
        
        try:
            self.create_kernel_performance_analysis()
            self.create_best_algorithm_distribution()  
            self.create_overall_performance_ranking()
            self.create_statistical_comparison()
            
            print("\n" + "="*80)
            print("FINAL ANALYSIS PLOTS COMPLETED")
            print("="*80)
            print(f"Output directory: {self.output_dir}")

            
        except Exception as e:
            print(f"Error generating final plots: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    base_dir = "/Users/siradbihi/Desktop/MScDataScience/ERP Maths Sentiments/Complete_Pipeline_Methodology_Evolution_2"
    
    generator = FinalAnalysisPlots(base_dir)
    generator.generate_final_plots()

if __name__ == "__main__":
    main()