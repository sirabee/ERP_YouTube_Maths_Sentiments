#!/usr/bin/env python3
"""
BERTopic Analysis by Search Query - Complete Pipeline Version

This is the original script that achieved 18.22% weighted average noise,
modified only to use our complete comment pipeline dataset (34,057 comments)
instead of the original 78,102 comment dataset.

All parameters, models, and processing logic remain identical to the original.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import re

warnings.filterwarnings('ignore')

class BERTopicQueryAnalyzer:
    def __init__(self):
        """Initialize the analyzer with our complete pipeline dataset."""
        # MODIFIED: Use our complete pipeline output instead of original dataset
        self.dataset_file = "/Users/siradbihi/Desktop/MScDataScience/ERP Maths Sentiments/Video Datasets/Complete Pipeline/comments_complete_filtered_20250720_224959.csv"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"bertopic_complete_pipeline_analysis_{self.timestamp}"
        self.all_topic_assignments = []
        
        # Initialize embedding model once to avoid repeated downloads
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")

    def load_dataset(self):
        """Load and analyze the consolidated dataset."""
        print(f"\nLoading dataset: {self.dataset_file}")
        
        if not os.path.exists(self.dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
        
        df = pd.read_csv(self.dataset_file)
        print(f"Loaded {len(df):,} comments from {df['video_id'].nunique():,} videos")
        
        text_col = 'comment_text'
        query_col = 'search_query'
        
        if text_col not in df.columns or query_col not in df.columns:
            raise ValueError(f"Required columns '{text_col}' and '{query_col}' not found. Available: {list(df.columns)}")
        
        # Ensure text column is string and handle potential NaN values
        df[text_col] = df[text_col].astype(str).fillna('')

        query_counts = df[query_col].value_counts()
        print(f"Found {len(query_counts)} unique search queries")
        
        return df, text_col, query_col, query_counts

    def setup_directories(self):
        """Create output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in ['_combined_analysis', '_summary_reports']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        print(f"Created output directory: {self.output_dir}")

    def create_bertopic_model(self, n_docs):
        """Create BERTopic model with adaptive parameters - EXACT ORIGINAL LOGIC."""
        if n_docs < 100:
            min_cluster_size, n_neighbors = max(5, int(n_docs * 0.1)), max(10, int(n_docs * 0.2))
        elif n_docs < 1000:
            min_cluster_size, n_neighbors = 15, 15
        else:
            min_cluster_size, n_neighbors = 25, 20

        # Ensure minimum values
        min_cluster_size = max(min_cluster_size, 5)
        n_neighbors = max(n_neighbors, 5)
        min_samples = max(3, min_cluster_size // 3)

        print(f"  Adaptive parameters: min_cluster_size={min_cluster_size}, n_neighbors={n_neighbors}, min_samples={min_samples}")

        # Use the pre-loaded embedding model
        umap_model = UMAP(n_components=5, n_neighbors=n_neighbors, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_features=1000)
        
        return BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            verbose=False
        )

    def process_query(self, df, query, text_col, query_col):
        """Process a single search query - EXACT ORIGINAL LOGIC."""
        print(f"\nProcessing query: '{query}'")
        
        subset = df[df[query_col] == query].copy()
        docs = subset[text_col].tolist()
        
        print(f"  Found {len(docs):,} documents")
        
        if len(docs) < 20:
            print(f"  Skipping '{query}' - insufficient documents for meaningful analysis.")
            return None
        
        safe_query = re.sub(r'[^a-zA-Z0-9_]+', '', query.replace(' ', '_'))
        query_dir = os.path.join(self.output_dir, safe_query)
        os.makedirs(query_dir, exist_ok=True)
        
        try:
            topic_model = self.create_bertopic_model(len(docs))
            topics, probabilities = topic_model.fit_transform(docs)
            
            # Convert topics to numpy array if it's a list
            if isinstance(topics, list):
                topics = np.array(topics)
            elif not isinstance(topics, np.ndarray):
                print(f"  ERROR: topics returned as {type(topics)}, expected numpy array or list")
                return {'query': query, 'success': False, 'error': f'Invalid topics type: {type(topics)}'}
            
            # Convert probabilities to numpy array if needed
            if probabilities is not None and isinstance(probabilities, list):
                probabilities = np.array(probabilities)
            
            topic_info = topic_model.get_topic_info()
            n_topics = len(topic_info[topic_info.Topic != -1])
            noise_count = np.sum(topics == -1)
            noise_pct = (noise_count / len(topics)) * 100 if len(topics) > 0 else 0
            
            print(f"  Discovered {n_topics} topics with {noise_pct:.1f}% noise.")
            
            # Store assigned topics for combined CSV
            assigned_df = subset.copy()
            assigned_df['topic'] = topics
            
            # Handle probabilities safely
            if probabilities is not None and hasattr(probabilities, 'max'):
                if probabilities.ndim > 1:
                    assigned_df['topic_probability'] = probabilities.max(axis=1)
                else:
                    assigned_df['topic_probability'] = probabilities
            else:
                assigned_df['topic_probability'] = 0
                
            self.all_topic_assignments.append(assigned_df)

            self.save_query_results(topic_model, assigned_df, topic_info, query_dir, safe_query, query)
            
            return {
                'query': query, 'safe_query': safe_query, 'n_docs': len(docs),
                'n_topics': n_topics, 'noise_count': noise_count, 'noise_pct': noise_pct, 'success': True
            }
            
        except Exception as e:
            print(f"  ERROR processing '{query}': {str(e)}")
            import traceback
            traceback.print_exc()
            return {'query': query, 'success': False, 'error': str(e)}

    def save_query_results(self, model, assigned_df, topic_info, query_dir, safe_query, query):
        """Save all results for a single query analysis."""
        # Save model
        model.save(os.path.join(query_dir, f"model_{safe_query}"), serialization="safetensors")
        
        # Save data
        assigned_df.to_csv(os.path.join(query_dir, f"data_with_topics_{safe_query}.csv"), index=False)
        topic_info.to_csv(os.path.join(query_dir, f"topic_info_{safe_query}.csv"), index=False)
        
        # Save standard visualizations
        self.save_standard_visualizations(model, query_dir, safe_query)
        
        # Save detailed report and publication-ready visuals
        self.create_detailed_report(model, assigned_df, topic_info, query_dir, safe_query, query)

    def save_standard_visualizations(self, model, query_dir, safe_query):
        """Generate and save standard BERTopic visualizations."""
        viz_path = os.path.join(query_dir, 'visualizations')
        os.makedirs(viz_path, exist_ok=True)
        
        try:
            fig = model.visualize_topics()
            fig.write_html(os.path.join(viz_path, f"topics_{safe_query}.html"))
        except Exception as e:
            print(f"    Warning: Could not generate 'visualize_topics': {e}")

        try:
            fig = model.visualize_hierarchy()
            fig.write_html(os.path.join(viz_path, f"hierarchy_{safe_query}.html"))
        except Exception as e:
            print(f"    Warning: Could not generate 'visualize_hierarchy': {e}")

        try:
            fig = model.visualize_barchart(top_n_topics=20)
            fig.write_html(os.path.join(viz_path, f"barchart_{safe_query}.html"))
        except Exception as e:
            print(f"    Warning: Could not generate 'visualize_barchart': {e}")

    def create_detailed_report(self, model, assigned_df, topic_info, query_dir, safe_query, query):
        """Generate a detailed text report and publication-ready visualizations."""
        report_path = os.path.join(query_dir, 'reports')
        os.makedirs(report_path, exist_ok=True)
        
        # Generate and save publication-ready bar chart
        self.create_publication_barchart(topic_info, report_path, safe_query, query)

        # Generate detailed text report
        report_content = f"--- Detailed Analysis Report for Query: '{query}' ---\n\n"
        report_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"Total documents analyzed: {len(assigned_df)}\n"
        report_content += f"Topics discovered: {len(topic_info[topic_info.Topic != -1])}\n"
        report_content += f"Noise (unassigned documents): {topic_info.iloc[0]['Count']} ({topic_info.iloc[0]['Count']*100/len(assigned_df):.2f}%)\n\n"
        
        report_content += "--- Top Topics Summary ---\n"
        for index, row in topic_info[topic_info.Topic != -1].head(15).iterrows():
            topic_num = row['Topic']
            report_content += f"\nTopic {topic_num}: {row['Name']} ({row['Count']} comments)\n"
            
            # Get representative documents
            try:
                rep_docs = model.get_representative_docs(topic_num)
                report_content += "  Representative Comments:\n"
                for doc in rep_docs:
                    report_content += f"    - {doc[:150]}...\n"
            except Exception:
                report_content += "  (Could not retrieve representative comments)\n"
        
        with open(os.path.join(report_path, f"detailed_report_{safe_query}.txt"), 'w', encoding='utf-8') as f:
            f.write(report_content)

    def create_publication_barchart(self, topic_info, path, safe_query, query):
        """Creates a high-quality, publication-ready bar chart of topic sizes."""
        try:
            display_topics = topic_info[topic_info.Topic != -1].head(15)
            if display_topics.empty:
                return

            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(display_topics['Name'], display_topics['Count'], color='skyblue')
            
            ax.set_title(f'Top {len(display_topics)} Topics for Query: "{query}"', fontsize=16, pad=20)
            ax.set_xlabel('Number of Comments', fontsize=12)
            ax.set_ylabel('Topic', fontsize=12)
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            
            # Add counts on bars
            for bar in bars:
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width()}', va='center', ha='left', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(path, f"barchart_publication_{safe_query}.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate publication bar chart: {e}")

    def create_final_summary(self, results):
        """Create a final summary report and consolidated data files."""
        summary_path = os.path.join(self.output_dir, '_summary_reports')
        combined_path = os.path.join(self.output_dir, '_combined_analysis')

        # Save consolidated topic assignments
        if self.all_topic_assignments:
            consolidated_df = pd.concat(self.all_topic_assignments, ignore_index=True)
            consolidated_df.to_csv(os.path.join(combined_path, 'all_comments_with_topics.csv'), index=False)
            print(f"\nSaved consolidated data for {len(consolidated_df)} comments.")

        # Create summary dataframe
        successful_results = [r for r in results if r is not None and r['success']]
        summary_df = pd.DataFrame(successful_results)
        
        if not summary_df.empty:
            summary_df.to_csv(os.path.join(summary_path, 'analysis_summary_stats.csv'), index=False)
            
            # Calculate weighted average noise
            total_docs = summary_df['n_docs'].sum()
            total_noise = (summary_df['n_docs'] * summary_df['noise_pct'] / 100).sum()
            weighted_avg_noise = (total_noise / total_docs) * 100 if total_docs > 0 else 0
            
            print(f"\n" + "=" * 70)
            print(f"PERFORMANCE COMPARISON TO ORIGINAL METHODOLOGY")
            print(f"=" * 70)
            print(f"Original methodology (78,102 docs): 18.22% weighted average noise")
            print(f"Our pipeline ({total_docs:,} docs): {weighted_avg_noise:.2f}% weighted average noise")
            
            if weighted_avg_noise <= 18.22:
                improvement = 18.22 - weighted_avg_noise
                print(f"✅ IMPROVEMENT: {improvement:.2f} percentage points better!")
                print(f"Our more focused filtering achieves superior performance")
            else:
                difference = weighted_avg_noise - 18.22
                print(f"❌ HIGHER NOISE: {difference:.2f} percentage points worse")
                print(f"Consider adjusting filtering parameters")

        # Generate final text report
        successful_analyses = [r for r in results if r is not None and r['success']]
        report = f"--- BERTopic Complete Pipeline Analysis Final Report ---\n\n"
        report += f"Analysis completed on: {self.timestamp}\n"
        report += f"Dataset: Complete comment filtering pipeline output\n"
        report += f"Processed {len(results)} queries. Successful analyses: {len(successful_analyses)}.\n\n"
        
        if not summary_df.empty:
            report += "--- Analysis Summary by Query ---\n"
            report += summary_df[['query', 'n_docs', 'n_topics', 'noise_pct']].to_string(index=False)
            report += "\n\n"
            
            report += f"--- Performance Comparison ---\n"
            report += f"Original methodology: 18.22% weighted average noise (78,102 docs)\n"
            report += f"Complete pipeline: {weighted_avg_noise:.2f}% weighted average noise ({total_docs:,} docs)\n"

        report += f"\nAll outputs saved in: {os.path.abspath(self.output_dir)}"
        
        with open(os.path.join(summary_path, 'final_analysis_report_complete_pipeline.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Final summary report created at: {summary_path}")

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("BERTopic Complete Pipeline Analysis (Original Script)")
        print("=" * 70)
        print("Using EXACT original script with our complete pipeline dataset")
        print("Expected: Better performance than 18.22% due to higher quality filtering")
        print("=" * 70)
        
        try:
            df, text_col, query_col, query_counts = self.load_dataset()
            self.setup_directories()
            
            results = []
            for i, (query, count) in enumerate(query_counts.items(), 1):
                print(f"\n[{i}/{len(query_counts)}] Query: '{query}' ({count:,} comments)")
                result = self.process_query(df, query, text_col, query_col)
                if result:
                    results.append(result)
            
            self.create_final_summary(results)
            
            print(f"\n{'='*70}")
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"Output directory: {self.output_dir}/")
            
        except Exception as e:
            print(f"\n{'!'*70}")
            print(f"A critical error occurred during the analysis pipeline: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'!'*70}")

def main():
    """Main execution function."""
    print("Starting BERTopic analysis using original script with complete pipeline dataset...")
    analyzer = BERTopicQueryAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()