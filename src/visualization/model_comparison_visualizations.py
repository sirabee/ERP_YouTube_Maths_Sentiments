#!/usr/bin/env python3
"""
Comprehensive Model Comparison Visualizations
Compares manual annotations with all model predictions:
- Twitter-RoBERTa (original aggregation)
- YouTube-BERT (original aggregation) 
- XLM-RoBERTa Variable K (original aggregation)
- XLM-RoBERTa Enhanced (improved aggregation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

class ModelComparisonVisualizer:
    def __init__(self):
        """Initialize the visualizer with paths and styling."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "visualizations" / f"model_comparison_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"Model Comparison Visualizations")
        print(f"Output directory: {self.output_dir}")
    
    def load_all_data(self):
        """Load all model results and manual annotations."""
        data = {}
        
        # Load manual annotations (consensus)
        consensus_path = self.base_path / "results" / "analysis" / "sent_analysis" / "model_evaluation" / "consensus_annotations_20250813_222930.csv"
        data['manual'] = pd.read_csv(consensus_path)
        print(f"Loaded manual annotations: {len(data['manual'])} samples")
        
        # Load Twitter-RoBERTa results (37% accuracy)
        twitter_path = self.base_path / "results" / "analysis" / "sent_analysis" / "model_evaluation" / "mapping_summary_20250813_222930.json"
        with open(twitter_path, 'r') as f:
            twitter_summary = json.load(f)
        
        # Load aligned data for Twitter-RoBERTa
        aligned_path = self.base_path / "results" / "analysis" / "sent_analysis" / "model_evaluation" / "aligned_annotations_predictions_20250813_222930.csv"
        data['twitter_aligned'] = pd.read_csv(aligned_path)
        
        # Load YouTube-BERT results
        youtube_path = self.base_path / "results" / "analysis" / "sent_analysis" / "model_evaluation" / "youtube_bert_original_matched_annotations_20250816_153905.csv"
        data['youtube_bert'] = pd.read_csv(youtube_path)
        print(f"Loaded YouTube-BERT results: {len(data['youtube_bert'])} samples")
        
        # Load XLM-RoBERTa Variable K results (same aggregation)
        xlm_var_path = self.base_path / "results" / "analysis" / "sent_analysis" / "model_evaluation" / "xlm_roberta_variable_k_matched_annotations_20250816_164500.csv"
        data['xlm_variable'] = pd.read_csv(xlm_var_path)
        print(f"Loaded XLM-RoBERTa Variable K results: {len(data['xlm_variable'])} samples")
        
        # Load XLM-RoBERTa Enhanced results
        xlm_enh_path = self.base_path / "results" / "analysis" / "sent_analysis" / "model_evaluation" / "xlm_roberta_clean_matched_annotations_20250816_141420.csv"
        data['xlm_enhanced'] = pd.read_csv(xlm_enh_path)
        print(f"Loaded XLM-RoBERTa Enhanced results: {len(data['xlm_enhanced'])} samples")
        
        return data, twitter_summary
    
    def create_accuracy_comparison(self, data):
        """Create overall accuracy comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy scores
        models = ['Twitter-RoBERTa\n(Twitter-trained)', 'YouTube-BERT\n(YouTube-trained)', 
                 'XLM-RoBERTa Var K\n(YouTube-trained)', 'XLM-RoBERTa Enhanced\n(YouTube-trained)']
        accuracies = [0.37, 0.73, 0.745, 0.74]
        aggregation_types = ['Original\nAggregation', 'Original\nAggregation', 
                           'Original\nAggregation', 'Enhanced\nAggregation']
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Bar chart
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Manual Annotation Agreement', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Comparison\n(200 Sample Manual Annotation)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 0.8)
        
        # Add value labels on bars
        for bar, acc, agg in zip(bars, accuracies, aggregation_types):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                    agg, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        # Add baseline and improvement lines
        ax1.axhline(y=0.37, color='red', linestyle='--', alpha=0.7, label='Twitter-RoBERTa\nBaseline')
        ax1.legend(loc='upper left', fontsize=9)
        
        # Improvement comparison
        improvements = [0, 36, 37.5, 37]
        bars2 = ax2.bar(models, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_ylabel('Improvement over Twitter-RoBERTa (pp)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Improvement Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 40)
        
        for bar, imp in zip(bars2, improvements):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'+{imp:.1f}pp', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created accuracy comparison chart")
    
    def create_sentiment_distribution_comparison(self, data):
        """Create sentiment distribution comparison across all models."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Add spacing between title and charts
        fig.subplots_adjust(top=0.92)
        
        # Define consistent sentiment order and color mapping
        sentiment_order = ['negative', 'neutral', 'positive']
        color_mapping = {
            'negative': '#FF9999',    # Light red for negative
            'neutral': '#66B2FF',     # Light blue for neutral  
            'positive': '#99FF99'     # Light green for positive
        }
        
        def get_consistent_pie_data(sentiment_counts):
            """Get values and colors in consistent order."""
            values = []
            colors = []
            labels = []
            for sentiment in sentiment_order:
                if sentiment in sentiment_counts.index:
                    values.append(sentiment_counts[sentiment])
                    colors.append(color_mapping[sentiment])
                    labels.append(sentiment)
            return values, colors, labels
        
        # Manual annotations
        manual_dist = data['manual']['consensus_sentiment'].value_counts()
        ax = axes[0]
        values, colors, labels = get_consistent_pie_data(manual_dist)
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        ax.set_title('Manual Annotations\n(Ground Truth)', fontsize=12, fontweight='bold')
        
        # Twitter-RoBERTa
        twitter_dist = data['twitter_aligned']['model_sentiment'].value_counts()
        ax = axes[1]
        values, colors, labels = get_consistent_pie_data(twitter_dist)
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        ax.set_title('Twitter-RoBERTa\n(37% Accuracy)', fontsize=12, fontweight='bold')
        
        # YouTube-BERT
        youtube_dist = data['youtube_bert']['model_sentiment'].value_counts()
        ax = axes[2]
        values, colors, labels = get_consistent_pie_data(youtube_dist)
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        ax.set_title('YouTube-BERT\n(73% Accuracy)', fontsize=12, fontweight='bold')
        
        # XLM-RoBERTa Variable K
        xlm_var_dist = data['xlm_variable']['model_sentiment'].value_counts()
        ax = axes[3]
        values, colors, labels = get_consistent_pie_data(xlm_var_dist)
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        ax.set_title('XLM-RoBERTa Variable K\n(74.5% Accuracy)', fontsize=12, fontweight='bold')
        
        # XLM-RoBERTa Enhanced
        xlm_enh_dist = data['xlm_enhanced']['model_sentiment'].value_counts()
        ax = axes[4]
        values, colors, labels = get_consistent_pie_data(xlm_enh_dist)
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        ax.set_title('XLM-RoBERTa Enhanced\n(74% Accuracy)', fontsize=12, fontweight='bold')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.suptitle('Sentiment Distribution Comparison: Manual vs Model Predictions', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sentiment_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created sentiment distribution comparison")
    
    def create_confusion_matrices(self, data):
        """Create confusion matrices for all models."""
        from sklearn.metrics import confusion_matrix
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        models_data = [
            (data['twitter_aligned'], 'Twitter-RoBERTa (Accuracy: 37%)', 'consensus_sentiment', 'model_sentiment'),
            (data['youtube_bert'], 'YouTube-BERT (Accuracy: 73%)', 'manual_sentiment', 'model_sentiment'),
            (data['xlm_variable'], 'XLM-RoBERTa Variable K (Accuracy: 74.5%)', 'manual_sentiment', 'model_sentiment'),
            (data['xlm_enhanced'], 'XLM-RoBERTa Enhanced (Accuracy: 74%)', 'manual_sentiment', 'model_sentiment')
        ]
        
        labels = ['positive', 'neutral', 'negative']
        
        for i, (df, title, true_col, pred_col) in enumerate(models_data):
            ax = axes[i]
            
            # Create confusion matrix
            cm = confusion_matrix(df[true_col], df[pred_col], labels=labels)
            
            # Normalize to percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create heatmap
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=ax,
                       cbar_kws={'label': 'Percentage'})
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Sentiment', fontweight='bold')
            ax.set_ylabel('True Sentiment', fontweight='bold')
            
            # # Add accuracy score
            # accuracy = np.trace(cm) / np.sum(cm)
            # ax.text(0.02, 0.98, f'Accuracy: {accuracy:.1%}', transform=ax.transAxes,
            #        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Confusion Matrices: Model Predictions vs Manual Annotations', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created confusion matrices comparison")
    
    def create_learning_journey_comparison(self, data):
        """Create learning journey detection comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Learning journey counts
        models = ['Manual\nAnnotations', 'Twitter-RoBERTa', 'YouTube-BERT', 
                 'XLM-RoBERTa Var K', 'XLM-RoBERTa Enhanced']
        
        # Count learning journeys (manual annotations use 'yes', models use True/False)
        manual_journeys = (data['manual']['consensus_journey'] == 'yes').sum()
        twitter_journeys = (data['twitter_aligned']['model_transition'] == 'yes').sum()
        youtube_journeys = data['youtube_bert']['model_journey'].sum()
        xlm_var_journeys = data['xlm_variable']['model_journey'].sum()
        xlm_enh_journeys = data['xlm_enhanced']['model_journey'].sum()
        
        journey_counts = [manual_journeys, twitter_journeys, youtube_journeys, 
                         xlm_var_journeys, xlm_enh_journeys]
        
        colors = ['#333333', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Bar chart
        bars = ax1.bar(models, journey_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Learning Journeys Detected', fontsize=12, fontweight='bold')
        ax1.set_title('Learning Journey Detection Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(journey_counts) * 1.2)
        
        # Add value labels
        for bar, count in zip(bars, journey_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Rotate x-axis labels
        ax1.tick_params(axis='x', rotation=45)
        
        # Add baseline line
        ax1.axhline(y=manual_journeys, color='red', linestyle='--', alpha=0.7, 
                   label=f'Manual Baseline ({manual_journeys})')
        ax1.legend()
        
        # Calculate accuracy scores for learning journey detection
        model_names = ['Twitter-RoBERTa', 'YouTube-BERT', 'XLM-RoBERTa Var K', 'XLM-RoBERTa Enhanced']
        accuracies = []
        
        # For each model, calculate learning journey detection accuracy
        for i, (df, model_name) in enumerate([(data['twitter_aligned'], 'Twitter-RoBERTa'),
                                             (data['youtube_bert'], 'YouTube-BERT'),
                                             (data['xlm_variable'], 'XLM-RoBERTa Var K'),
                                             (data['xlm_enhanced'], 'XLM-RoBERTa Enhanced')]):
            if model_name == 'Twitter-RoBERTa':
                manual_col = 'consensus_transition'
                model_col = 'model_transition'
                manual_true = (df[manual_col] == 'yes')
                model_true = (df[model_col] == 'yes')
            else:
                manual_col = 'manual_journey'
                model_col = 'model_journey'
                manual_true = (df[manual_col] == 'yes')
                model_true = df[model_col]
            
            # Calculate accuracy
            correct = (manual_true == model_true).sum()
            total = len(df)
            accuracy = correct / total
            accuracies.append(accuracy)
        
        # Accuracy bar chart
        bars2 = ax2.bar(model_names, accuracies, color=colors[1:], alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_ylabel('Learning Journey Detection Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Journey Detection Accuracy', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        
        for bar, acc in zip(bars2, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Rotate x-axis labels for both subplots
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_journey_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created learning journey comparison")
    
    def create_disagreement_analysis(self, data):
        """Create disagreement pattern analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        models_data = [
            (data['twitter_aligned'], 'Twitter-RoBERTa', 'consensus_sentiment', 'model_sentiment'),
            (data['youtube_bert'], 'YouTube-BERT', 'manual_sentiment', 'model_sentiment'),
            (data['xlm_variable'], 'XLM-RoBERTa Variable K', 'manual_sentiment', 'model_sentiment'),
            (data['xlm_enhanced'], 'XLM-RoBERTa Enhanced', 'manual_sentiment', 'model_sentiment')
        ]
        
        for i, (df, title, true_col, pred_col) in enumerate(models_data):
            ax = axes[i]
            
            # Find disagreements
            disagreements = df[df[true_col] != df[pred_col]]
            
            if len(disagreements) > 0:
                # Count disagreement patterns
                disagreement_patterns = disagreements.groupby([true_col, pred_col]).size()
                
                # Create labels and values
                labels = []
                values = []
                for (true_sent, pred_sent), count in disagreement_patterns.items():
                    labels.append(f'{true_sent} → {pred_sent}')
                    values.append(count)
                
                # Create horizontal bar chart
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                bars = ax.barh(labels, values, color=colors, alpha=0.8, edgecolor='black')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    width = bar.get_width()
                    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{value}', ha='left', va='center', fontweight='bold')
                
                ax.set_xlabel('Number of Disagreements', fontweight='bold')
                ax.set_title(f'{title}\nDisagreement Patterns', fontsize=12, fontweight='bold')
                
                # Add total disagreements and accuracy (bottom right)
                total_disagreements = len(disagreements)
                accuracy = (len(df) - total_disagreements) / len(df)
                ax.text(0.98, 0.02, f'Total Disagreements: {total_disagreements}\nAccuracy: {accuracy:.1%}',
                       transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No Disagreements', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{title}\nDisagreement Patterns', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'disagreement_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created disagreement analysis")
    
    def create_domain_impact_analysis(self, data):
        """Create visualization showing domain training impact."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Domain training comparison
        training_domains = ['Twitter/Social Media', 'YouTube Comments', 'YouTube Comments', 'YouTube Comments']
        models = ['Twitter-RoBERTa', 'YouTube-BERT', 'XLM-RoBERTa Var K', 'XLM-RoBERTa Enhanced']
        accuracies = [0.37, 0.73, 0.745, 0.74]
        aggregation = ['Original', 'Original', 'Original', 'Enhanced']
        
        # Color by training domain
        colors = ['#FF6B6B' if domain == 'Twitter/Social Media' else '#4ECDC4' for domain in training_domains]
        
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Manual Annotation Agreement', fontsize=12, fontweight='bold')
        ax1.set_title('Domain Training Impact Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 0.8)
        
        # Add annotations
        for bar, acc, agg in zip(bars, accuracies, aggregation):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{agg}\nAggregation', ha='center', va='center', fontsize=8, 
                    color='white', fontweight='bold')
        
        # Legend
        twitter_patch = plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.8)
        youtube_patch = plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.8)
        ax1.legend([twitter_patch, youtube_patch], ['Twitter/Social Media\nTraining', 'YouTube Training'])
        
        # Impact analysis
        factors = ['Domain Training\n(Twitter → YouTube)', 'Model Architecture\n(BERT → XLM-RoBERTa)', 
                  'Aggregation Method\n(Original → Enhanced)']
        improvements = [36, 1.5, -0.5]  # XLM Enhanced is slightly worse than Variable K
        
        colors_impact = ['#FF6B6B', '#45B7D1', '#96CEB4']
        bars2 = ax2.bar(factors, improvements, color=colors_impact, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_ylabel('Performance Improvement (percentage points)', fontsize=12, fontweight='bold')
        ax2.set_title('Factor Impact Analysis', fontsize=14, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylim(-5, 40)  # Add more space for labels
        
        for bar, imp in zip(bars2, improvements):
            height = bar.get_height()
            label_y = height + 1.5 if height > 0 else height - 1.5
            ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{imp:+.1f}pp', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=11)
        
        # Rotate x-axis labels for both subplots
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created domain impact analysis")
    
    def create_summary_dashboard(self, data):
        """Create comprehensive summary dashboard."""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # Title (reduced font size)
        fig.suptitle('Model Comparison Dashboard: Educational Sentiment Analysis\n' +
                    'Manual Annotation Agreement (200 Samples)', fontsize=16, fontweight='bold', y=0.95)
        
        # 1. Overall accuracy comparison
        ax1 = fig.add_subplot(gs[0, :2])
        models = ['Twitter-RoBERTa', 'YouTube-BERT', 'XLM-RoBERTa Var K', 'XLM-RoBERTa Enhanced']
        accuracies = [0.37, 0.73, 0.745, 0.74]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_ylim(0, 0.8)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sentiment distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        manual_dist = data['manual']['consensus_sentiment'].value_counts()
        twitter_dist = data['twitter_aligned']['model_sentiment'].value_counts()
        youtube_dist = data['youtube_bert']['model_sentiment'].value_counts()
        xlm_var_dist = data['xlm_variable']['model_sentiment'].value_counts()
        xlm_enh_dist = data['xlm_enhanced']['model_sentiment'].value_counts()
        
        sentiments = ['positive', 'neutral', 'negative']
        x = np.arange(len(sentiments))
        width = 0.15
        
        ax2.bar(x - 2*width, [manual_dist.get(s, 0) for s in sentiments], width, 
               label='Manual', color='#333333', alpha=0.8)
        ax2.bar(x - width, [twitter_dist.get(s, 0) for s in sentiments], width, 
               label='Twitter-RoBERTa', color='#FF6B6B', alpha=0.8)
        ax2.bar(x, [youtube_dist.get(s, 0) for s in sentiments], width, 
               label='YouTube-BERT', color='#4ECDC4', alpha=0.8)
        ax2.bar(x + width, [xlm_var_dist.get(s, 0) for s in sentiments], width, 
               label='XLM-Var K', color='#45B7D1', alpha=0.8)
        ax2.bar(x + 2*width, [xlm_enh_dist.get(s, 0) for s in sentiments], width, 
               label='XLM-Enhanced', color='#96CEB4', alpha=0.8)
        
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Sentiment Distribution Comparison', fontweight='bold')
        # Move xlabel further down to avoid overlap
        ax2.set_xlabel('Sentiment', fontweight='bold', labelpad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(sentiments)
        ax2.legend()
        
        # 3. Learning journey comparison
        ax3 = fig.add_subplot(gs[1, :2])
        journey_counts = [13, 77, 41, 25, 25]  # Manual, Twitter, YouTube, XLM-Var, XLM-Enh
        journey_models = ['Manual', 'Twitter-RoBERTa', 'YouTube-BERT', 'XLM-Var K', 'XLM-Enhanced']
        
        bars = ax3.bar(journey_models, journey_counts, color=['#333333'] + colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Learning Journeys Detected', fontweight='bold')
        ax3.set_title('Learning Journey Detection', fontweight='bold')
        ax3.axhline(y=13, color='red', linestyle='--', alpha=0.7, label='Manual Baseline')
        
        for bar, count in zip(bars, journey_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Key metrics table
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.axis('off')
        
        metrics_data = [
            ['Model', 'Accuracy', 'Training Domain', 'Aggregation'],
            ['Twitter-RoBERTa', '37%', 'Twitter/Social', 'Original'],
            ['YouTube-BERT', '73%', 'YouTube', 'Original'],
            ['XLM-RoBERTa Var K', '74.5%', 'YouTube', 'Original'],
            ['XLM-RoBERTa Enhanced', '74%', 'YouTube', 'Enhanced']
        ]
        
        table = ax4.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F5F5F5')
        
        ax4.set_title('Performance Summary Table', fontweight='bold', pad=20)
        
        # 5. Domain training impact
        ax5 = fig.add_subplot(gs[2, :])
        factors = ['Domain Training\n(Twitter → YouTube)', 'Model Architecture\n(BERT variants)', 
                  'Aggregation Method\n(Original → Enhanced)']
        impacts = [36, 1.5, -0.5]
        impact_colors = ['#FF6B6B', '#45B7D1', '#96CEB4']
        
        bars = ax5.bar(factors, impacts, color=impact_colors, alpha=0.8, edgecolor='black')
        ax5.set_ylabel('Performance Impact (percentage points)', fontweight='bold')
        ax5.set_title('Factor Impact Analysis: What Drives Performance?', fontweight='bold')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_ylim(-5, 40)  # Adjust y-limits to prevent label overlap
        
        for bar, imp in zip(bars, impacts):
            height = bar.get_height()
            label_y = height + 1 if height > 0 else height - 2
            ax5.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{imp:+.1f}pp', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=12)
        
        # Add key insight (moved down from border)
        ax5.text(0.98, 0.90, 'Key Insight: Domain training is 24x more impactful than model architecture',
                transform=ax5.transAxes, fontsize=12, fontweight='bold', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.savefig(self.output_dir / 'model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created comprehensive dashboard")
    
    def run_all_visualizations(self):
        """Execute all visualization functions."""
        print("="*60)
        print("MODEL COMPARISON VISUALIZATIONS")
        print("="*60)
        
        # Load all data
        data, twitter_summary = self.load_all_data()
        
        # Create all visualizations
        self.create_accuracy_comparison(data)
        self.create_sentiment_distribution_comparison(data)
        self.create_confusion_matrices(data)
        self.create_learning_journey_comparison(data)
        self.create_disagreement_analysis(data)
        self.create_domain_impact_analysis(data)
        self.create_summary_dashboard(data)
        
        print("="*60)
        print("VISUALIZATION SUMMARY")
        print("="*60)
        print(f"All visualizations saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  - {file.name}")
        
        print("\nKey findings visualized:")
        print("  • Domain training provides 36-37.5pp improvement")
        print("  • Model architecture differences: only 1.5pp")
        print("  • Aggregation improvements: minimal (1pp)")
        print("  • All models struggle with neutral educational content")

def main():
    """Main execution function."""
    visualizer = ModelComparisonVisualizer()
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main()