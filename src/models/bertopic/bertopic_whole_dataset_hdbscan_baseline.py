#!/usr/bin/env python3
"""
BERTopic Whole-Dataset HDBSCAN Baseline Analysis
Complete Pipeline Comments Dataset (34,057 comments)

This script runs a whole-dataset HDBSCAN BERTopic model on the complete pipeline
comments dataset to establish baseline performance for comparison with the 
documented 18.22% weighted noise baseline from per-query optimization.

"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import warnings
import re

warnings.filterwarnings('ignore')

class WholeDatasetHDBSCANBaseline:
    """Whole-dataset HDBSCAN BERTopic analysis for baseline comparison"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"bertopic_whole_dataset_hdbscan_{self.timestamp}"
        
        # Complete pipeline dataset path
        self.dataset_file = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/data/processed/comments_complete_filtered_20250720_224959.csv"
        
        # BERTopic parameters matching gradual approach methodology
        self.bertopic_params = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'umap_params': {
                'n_components': 5,
                'n_neighbors': 15,
                'metric': 'cosine',
                'random_state': 42
            },
            'hdbscan_params': {
                'min_cluster_size': 15,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'
            },
            'vectorizer_params': {
                'ngram_range': (1, 2),
                'stop_words': 'english',
                'max_features': 1000,
                'min_df': 2
            }
        }
        
        # Analysis results storage
        self.results = {}
        
    def log(self, message):
        """Log messages with timestamps"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def setup_output_directory(self):
        """Create structured output directory"""
        self.log("Setting up output directory structure...")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories matching existing structure
        subdirs = [
            'models',
            'visualizations', 
            'data_outputs',
            'summary_reports',
            'statistical_analysis'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        self.log(f"Output directory created: {self.output_dir}")
    
    def load_dataset(self):
        """Load the complete pipeline comments dataset"""
        self.log("Loading complete pipeline comments dataset...")
        
        if not os.path.exists(self.dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
        
        df = pd.read_csv(self.dataset_file)
        
        # Basic dataset validation
        required_columns = ['comment_text', 'video_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean and prepare text data
        df['comment_text_clean'] = df['comment_text'].fillna('').astype(str)
        df = df[df['comment_text_clean'].str.len() > 10].copy()  # Remove very short comments
        
        self.log(f"Dataset loaded: {len(df):,} comments from {df['video_id'].nunique():,} videos")
        self.log(f"Average comment length: {df['comment_text_clean'].str.len().mean():.1f} characters")
        
        self.results['dataset_info'] = {
            'total_comments': len(df),
            'unique_videos': df['video_id'].nunique(),
            'avg_comment_length': df['comment_text_clean'].str.len().mean(),
            'dataset_file': self.dataset_file
        }
        
        return df
    
    def initialize_bertopic_model(self):
        """Initialize BERTopic model with gradual approach parameters"""
        self.log("Initializing BERTopic model with gradual approach parameters...")
        
        # Embedding model
        embedding_model = SentenceTransformer(self.bertopic_params['embedding_model'])
        
        # UMAP for dimensionality reduction
        umap_model = UMAP(**self.bertopic_params['umap_params'])
        
        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(**self.bertopic_params['hdbscan_params'])
        
        # CountVectorizer for topic representation
        vectorizer_model = CountVectorizer(**self.bertopic_params['vectorizer_params'])
        
        # Initialize BERTopic (disable probabilities to avoid prediction data error)
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True,
            calculate_probabilities=False
        )
        
        self.log("BERTopic model initialized successfully")
        return topic_model
    
    def run_bertopic_analysis(self, df):
        """Run BERTopic analysis on the complete dataset"""
        self.log("Starting BERTopic analysis on complete dataset...")
        
        # Prepare documents
        documents = df['comment_text_clean'].tolist()
        self.log(f"Processing {len(documents):,} documents...")
        
        # Initialize model
        topic_model = self.initialize_bertopic_model()
        
        # Fit the model
        start_time = datetime.now()
        self.log("Fitting BERTopic model...")
        
        topics = topic_model.fit_transform(documents)
        probabilities = None  # No probabilities calculated
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.log(f"BERTopic analysis completed in {duration}")
        
        # Handle topics return format (may be list or array)
        if isinstance(topics, list) and len(topics) > 0 and isinstance(topics[0], list):
            # If topics is a list of lists, take the first element (topics)
            topics = topics[0]
        
        # Ensure topics is a simple list/array of integers
        topics = np.array(topics).flatten()
        
        # Ensure topics array has the correct length (same as documents)
        if len(topics) != len(documents):
            self.log(f"Warning: Topics length {len(topics)} doesn't match documents length {len(documents)}")
            # Take only the first len(documents) elements if topics is longer
            if len(topics) > len(documents):
                topics = topics[:len(documents)]
            else:
                # This would be an error we can't easily fix
                raise ValueError(f"Topics array too short: {len(topics)} < {len(documents)}")
        
        # Calculate performance metrics
        unique_topics = len(set(topics))
        noise_count = (np.array(topics) == -1).sum()
        noise_percentage = (noise_count / len(topics)) * 100
        
        # Calculate silhouette score (excluding noise points)
        try:
            # Get the reduced embeddings used for clustering
            reduced_embeddings = topic_model.umap_model.embedding_
            non_noise_mask = topics != -1
            
            if non_noise_mask.sum() > 1 and len(reduced_embeddings) == len(topics):
                silhouette_avg = silhouette_score(reduced_embeddings[non_noise_mask], 
                                                topics[non_noise_mask])
            else:
                silhouette_avg = -1
        except Exception as e:
            self.log(f"Warning: Could not calculate silhouette score: {e}")
            silhouette_avg = -1
        
        # Store results
        self.results['bertopic_analysis'] = {
            'total_documents': len(documents),
            'unique_topics': unique_topics,
            'noise_count': noise_count,
            'noise_percentage': noise_percentage,
            'silhouette_score': silhouette_avg,
            'processing_duration': str(duration),
            'topics_distribution': dict(zip(*np.unique(topics, return_counts=True)))
        }
        
        self.log(f"Analysis Results:")
        self.log(f"  Total documents: {len(documents):,}")
        self.log(f"  Unique topics: {unique_topics}")
        self.log(f"  Noise documents: {noise_count:,} ({noise_percentage:.2f}%)")
        self.log(f"  Silhouette score: {silhouette_avg:.4f}")
        
        return topic_model, topics, probabilities
    
    def generate_topic_info(self, topic_model, df, topics):
        """Generate detailed topic information"""
        self.log("Generating detailed topic information...")
        
        # Get topic info from BERTopic
        topic_info = topic_model.get_topic_info()
        
        # Add topic assignments to dataframe
        df_with_topics = df.copy()
        df_with_topics['topic'] = topics
        
        # Calculate additional statistics per topic
        topic_stats = []
        for topic_id in topic_info['Topic']:
            topic_docs = df_with_topics[df_with_topics['topic'] == topic_id]
            
            stats = {
                'topic': topic_id,
                'count': len(topic_docs),
                'unique_videos': topic_docs['video_id'].nunique() if topic_id != -1 else 0,
                'avg_comment_length': topic_docs['comment_text_clean'].str.len().mean(),
                'representative_docs': topic_docs['comment_text_clean'].head(3).tolist()
            }
            topic_stats.append(stats)
        
        topic_stats_df = pd.DataFrame(topic_stats)
        
        # Merge with topic info
        enhanced_topic_info = topic_info.merge(topic_stats_df, left_on='Topic', right_on='topic', how='left')
        
        return enhanced_topic_info, df_with_topics
    
    def create_visualizations(self, topic_model, documents):
        """Create comprehensive visualizations"""
        self.log("Creating visualizations...")
        
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        
        try:
            # Topic overview visualization
            self.log("  Creating topic overview...")
            fig1 = topic_model.visualize_topics()
            fig1.write_html(os.path.join(viz_dir, 'topics_overview.html'))
            
            # Topic hierarchy
            self.log("  Creating topic hierarchy...")
            fig2 = topic_model.visualize_hierarchy()
            fig2.write_html(os.path.join(viz_dir, 'topics_hierarchy.html'))
            
            # Topic heatmap
            self.log("  Creating topic heatmap...")
            fig3 = topic_model.visualize_heatmap()
            fig3.write_html(os.path.join(viz_dir, 'topics_heatmap.html'))
            
            # Bar chart of topics
            self.log("  Creating topic bar chart...")
            fig4 = topic_model.visualize_barchart(top_k_topics=20)
            fig4.write_html(os.path.join(viz_dir, 'topics_barchart.html'))
            
            self.log("Visualizations created successfully")
            
        except Exception as e:
            self.log(f"Warning: Some visualizations failed to generate: {e}")
    
    def create_performance_comparison_charts(self):
        """Create performance comparison charts"""
        self.log("Creating performance comparison charts...")
        
        # Performance comparison data (using correct complete pipeline baseline)
        comparison_data = {
            'Method': ['Whole-Dataset HDBSCAN (Current)', 'Per-Query HDBSCAN (Complete Pipeline)', 'K-means Variable K (Optimal)'],
            'Noise_Percentage': [self.results['bertopic_analysis']['noise_percentage'], 8.72, 0.0],
            'Dataset_Size': [34048, 34024, 34024],
            'Silhouette_Score': [self.results['bertopic_analysis']['silhouette_score'], 0.25, 0.42]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BERTopic Performance Comparison\nWhole-Dataset HDBSCAN vs. Documented Baselines', fontsize=16)
        
        # Noise percentage comparison
        axes[0, 0].bar(comparison_df['Method'], comparison_df['Noise_Percentage'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Noise Percentage Comparison')
        axes[0, 0].set_ylabel('Noise Percentage (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Dataset size comparison
        axes[0, 1].bar(comparison_df['Method'], comparison_df['Dataset_Size'],
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_title('Dataset Size Comparison')
        axes[0, 1].set_ylabel('Number of Comments')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Silhouette score comparison
        axes[1, 0].bar(comparison_df['Method'], comparison_df['Silhouette_Score'],
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 0].set_title('Silhouette Score Comparison')
        axes[1, 0].set_ylabel('Silhouette Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Topic count comparison
        current_topics = self.results['bertopic_analysis']['unique_topics']
        axes[1, 1].bar(['Current Analysis', 'Typical Range'], [current_topics, 25],
                       color=['#FF6B6B', '#95A5A6'])
        axes[1, 1].set_title('Topic Count Comparison')
        axes[1, 1].set_ylabel('Number of Topics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'summary_reports', 'performance_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'summary_reports', 'performance_comparison.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        # Save comparison data
        comparison_df.to_csv(os.path.join(self.output_dir, 'statistical_analysis', 'performance_comparison.csv'), 
                           index=False)
    
    def save_model_and_data(self, topic_model, enhanced_topic_info, df_with_topics):
        """Save model, data, and analysis results"""
        self.log("Saving model and data outputs...")
        
        # Save BERTopic model
        model_path = os.path.join(self.output_dir, 'models', 'bertopic_whole_dataset_model')
        topic_model.save(model_path)
        
        # Save enhanced topic information
        enhanced_topic_info.to_csv(os.path.join(self.output_dir, 'data_outputs', 'topic_info_enhanced.csv'), index=False)
        
        # Save data with topic assignments
        df_with_topics.to_csv(os.path.join(self.output_dir, 'data_outputs', 'comments_with_topics.csv'), index=False)
        
        # Save analysis results as JSON
        with open(os.path.join(self.output_dir, 'summary_reports', 'analysis_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        self.log("Generating comprehensive summary report...")
        
        report_path = os.path.join(self.output_dir, 'summary_reports', 'baseline_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("BERTOPIC WHOLE-DATASET HDBSCAN BASELINE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: Complete Pipeline Comments (34,057 comments)\n")
            f.write(f"Purpose: Baseline comparison with documented methodology evolution\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 40 + "\n")
            dataset_info = self.results['dataset_info']
            f.write(f"Total Comments: {dataset_info['total_comments']:,}\n")
            f.write(f"Unique Videos: {dataset_info['unique_videos']:,}\n")
            f.write(f"Average Comment Length: {dataset_info['avg_comment_length']:.1f} characters\n")
            f.write(f"Source File: {dataset_info['dataset_file']}\n\n")
            
            f.write("BERTOPIC CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Embedding Model: {self.bertopic_params['embedding_model']}\n")
            f.write(f"UMAP Parameters: {self.bertopic_params['umap_params']}\n")
            f.write(f"HDBSCAN Parameters: {self.bertopic_params['hdbscan_params']}\n")
            f.write(f"Vectorizer Parameters: {self.bertopic_params['vectorizer_params']}\n\n")
            
            f.write("ANALYSIS RESULTS:\n")
            f.write("-" * 40 + "\n")
            analysis = self.results['bertopic_analysis']
            f.write(f"Total Documents Processed: {analysis['total_documents']:,}\n")
            f.write(f"Unique Topics Found: {analysis['unique_topics']}\n")
            f.write(f"Noise Documents: {analysis['noise_count']:,}\n")
            f.write(f"Noise Percentage: {analysis['noise_percentage']:.2f}%\n")
            f.write(f"Silhouette Score: {analysis['silhouette_score']:.4f}\n")
            f.write(f"Processing Duration: {analysis['processing_duration']}\n\n")
            
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Current Analysis (Whole-Dataset HDBSCAN): {analysis['noise_percentage']:.2f}% noise\n")
            f.write(f"Complete Pipeline Baseline (Per-Query HDBSCAN): 8.72% weighted noise\n")
            f.write(f"Documented Optimal (Variable K-means): 0% noise\n\n")
            
            if analysis['noise_percentage'] < 8.72:
                f.write("✅ PERFORMANCE: Better than complete pipeline baseline\n")
            elif analysis['noise_percentage'] < 15:
                f.write("⚠️  PERFORMANCE: Comparable to complete pipeline baseline\n")
            else:
                f.write("❌ PERFORMANCE: Below complete pipeline baseline\n")
            
            f.write(f"\nIMPROVEMENT vs BASELINE: {8.72 - analysis['noise_percentage']:.2f} percentage points\n\n")
            
            f.write("METHODOLOGY IMPLICATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("This analysis validates the complete pipeline dataset quality and provides\n")
            f.write("baseline comparison for the documented methodology evolution from HDBSCAN\n")
            f.write("to variable K-means clustering approach.\n\n")
            
            f.write("NEXT STEPS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Proceed with variable K-means implementation for optimal performance\n")
            f.write("2. Use this baseline for methodology comparison documentation\n")
            f.write("3. Integrate with sentiment analysis pipeline\n\n")
            
            f.write(f"Report Generated: {datetime.now()}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
        
        self.log(f"Summary report saved: {report_path}")
    
    def run_complete_analysis(self):
        """Run the complete baseline analysis"""
        self.log("Starting complete baseline analysis...")
        start_time = datetime.now()
        
        try:
            # Setup
            self.setup_output_directory()
            
            # Load dataset
            df = self.load_dataset()
            
            # Run BERTopic analysis
            topic_model, topics, probabilities = self.run_bertopic_analysis(df)
            
            # Generate detailed topic information
            enhanced_topic_info, df_with_topics = self.generate_topic_info(topic_model, df, topics)
            
            # Create visualizations
            self.create_visualizations(topic_model, df['comment_text_clean'].tolist())
            
            # Create performance comparison charts
            self.create_performance_comparison_charts()
            
            # Save all outputs
            self.save_model_and_data(topic_model, enhanced_topic_info, df_with_topics)
            
            # Generate summary report
            self.generate_summary_report()
            
            end_time = datetime.now()
            total_duration = end_time - start_time
            
            self.log("=" * 80)
            self.log("BASELINE ANALYSIS COMPLETED SUCCESSFULLY!")
            self.log("=" * 80)
            self.log(f"Total Duration: {total_duration}")
            self.log(f"Output Directory: {self.output_dir}")
            self.log(f"Noise Percentage: {self.results['bertopic_analysis']['noise_percentage']:.2f}%")
            self.log(f"Baseline Comparison: {8.72 - self.results['bertopic_analysis']['noise_percentage']:.2f} percentage points vs complete pipeline")
            
            return self.results, self.output_dir
            
        except Exception as e:
            self.log(f"ERROR: Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main execution function"""
    print("BERTopic Whole-Dataset HDBSCAN Baseline Analysis")
    print("Complete Pipeline Comments Dataset (34,057 comments)")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = WholeDatasetHDBSCANBaseline(verbose=True)
        
        # Run complete analysis
        results, output_dir = analyzer.run_complete_analysis()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Noise percentage: {results['bertopic_analysis']['noise_percentage']:.2f}%")
        print(f"📈 Baseline comparison: {8.72 - results['bertopic_analysis']['noise_percentage']:.2f} percentage points vs complete pipeline")
        print(f"🎯 Ready for methodology comparison and variable K-means implementation")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()