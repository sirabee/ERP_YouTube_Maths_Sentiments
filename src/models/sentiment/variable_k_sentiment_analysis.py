#!/usr/bin/env python3
"""
Sentiment Analysis Integration with Variable K-means BERTopic Results
Processing Variable K-means topic assignments with sentiment analysis
"""

import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import gc
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

class VariableKMeansSentimentIntegration:
    def __init__(self, comments_file=None):
        """Initialize sentiment analysis pipeline for Variable K-means BERTopic results."""
        if comments_file is None:
            comments_file = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/optimised_variable_k_phase_4_20250722_224755/results/all_comments_with_topics.csv"
        self.comments_file = comments_file
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/Sentiment_Analysis/Variable_K_Means/sentiment_analysis_{self.timestamp}"
        
        # Setup device for hardware optimization
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize Twitter RoBERTa sentiment pipeline
        print("Loading Twitter RoBERTa sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,  # Use CPU for pipeline compatibility
            batch_size=16,  # Reduced for better memory management
            truncation=True,
            max_length=512
        )
        print("Sentiment model loaded successfully.")
        
    def load_variable_k_results(self):
        """Load the Variable K-means BERTopic analysis results."""
        print(f"Loading Variable K-means BERTopic results from: {self.comments_file}")
        
        if not os.path.exists(self.comments_file):
            raise FileNotFoundError(f"Variable K-means results file not found: {self.comments_file}")
        
        df = pd.read_csv(self.comments_file)
        print(f"Loaded {len(df):,} comments with Variable K-means topic assignments")
        print(f"Unique topics: {df['topic'].nunique()}")
        print(f"Unique queries: {df['search_query'].nunique()}")
        print(f"Algorithm: {df['algorithm'].iloc[0] if 'algorithm' in df.columns else 'Not specified'}")
        
        # Ensure text column is string and handle NaN values
        df['comment_text'] = df['comment_text'].astype(str).fillna('')
        
        # Filter out empty comments and validate data
        initial_count = len(df)
        df = df[df['comment_text'].str.len() > 0]
        filtered_count = len(df)
        print(f"After filtering empty comments: {filtered_count:,} comments (removed {initial_count - filtered_count:,})")
        
        # Validate required columns
        required_cols = ['comment_text', 'topic', 'search_query']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def process_sentiment_batch(self, texts, batch_size=16):
        """Process sentiment analysis in memory-efficient batches."""
        sentiments = []
        
        # Process in batches to manage memory
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing sentiment for Variable K-means"):
            batch = texts[i:i + batch_size]
            
            try:
                # Get sentiment predictions
                results = self.sentiment_pipeline(batch)
                
                # Extract sentiment information
                for result in results:
                    sentiments.append({
                        'sentiment_label': result['label'],
                        'sentiment_score': result['score']
                    })
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add placeholder results for failed batch
                for _ in batch:
                    sentiments.append({
                        'sentiment_label': 'NEUTRAL',
                        'sentiment_score': 0.5
                    })
            
            # Memory cleanup every 5 batches for better performance
            if i % (batch_size * 5) == 0:
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
        
        return sentiments
    
    def analyze_sentiment_by_group(self, df, group_col, description):
        """Generic sentiment analysis by grouping column."""
        print(f"Analyzing sentiment by {description}...")
        
        # Calculate sentiment distribution by group
        sentiment_counts = df.groupby([group_col, 'sentiment_label']).size().unstack(fill_value=0)
        
        # Calculate percentages
        sentiment_pct = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
        
        # Calculate average sentiment scores by group
        sentiment_scores = df.groupby(group_col).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'comment_text': 'count'
        }).round(3)
        
        return sentiment_counts, sentiment_pct, sentiment_scores
    
    def analyze_sentiment_by_topic(self, df):
        """Analyze sentiment distribution across Variable K-means topics."""
        return self.analyze_sentiment_by_group(df, 'topic', 'Variable K-means topic')
    
    def analyze_sentiment_by_query(self, df):
        """Analyze sentiment distribution across search queries."""
        return self.analyze_sentiment_by_group(df, 'search_query', 'search query')
    
    def create_topic_sentiment_mapping(self, df):
        """Create topic-sentiment mapping for Variable K-means results."""
        print("Creating Variable K-means topic-sentiment aspect mapping...")
        
        # Group by topic and analyze patterns
        topic_analysis = []
        
        for topic_id in df['topic'].unique():
            if topic_id == -1:  # Skip noise
                continue
                
            topic_data = df[df['topic'] == topic_id]
            
            # Get dominant sentiment
            sentiment_dist = topic_data['sentiment_label'].value_counts(normalize=True)
            dominant_sentiment = sentiment_dist.index[0]
            dominant_pct = sentiment_dist.iloc[0] * 100
            
            # Calculate average sentiment score
            avg_score = topic_data['sentiment_score'].mean()
            
            # Get most common query for context
            common_query = topic_data['search_query'].value_counts().index[0]
            
            topic_analysis.append({
                'topic': topic_id,
                'document_count': len(topic_data),
                'dominant_sentiment': dominant_sentiment,
                'dominant_sentiment_pct': dominant_pct,
                'avg_sentiment_score': avg_score,
                'most_common_query': common_query,
                'sentiment_diversity': len(sentiment_dist),
                'algorithm': 'Variable_K'
            })
        
        return pd.DataFrame(topic_analysis)
    
    def create_visualizations(self, df, topic_sentiment_pct, query_sentiment_pct):
        """Create comprehensive visualizations for Variable K-means results."""
        print("Creating Variable K-means sentiment visualizations...")
        
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Overall sentiment distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts = df['sentiment_label'].value_counts()
        sentiment_labels = sentiment_counts.index
        if len(sentiment_labels) == 3:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(sentiment_labels)))
        
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Variable K-means: Overall Sentiment Distribution in YouTube Math Education Comments')
        plt.savefig(os.path.join(viz_dir, "variable_k_overall_sentiment_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Topic-level sentiment heatmap
        if len(topic_sentiment_pct) > 0:
            plt.figure(figsize=(12, 8))
            # Sort by topic ID for better visualization
            sorted_topics = topic_sentiment_pct.sort_index()
            sns.heatmap(sorted_topics, annot=True, fmt='.1f', cmap='RdYlBu_r', center=50)
            plt.title('Variable K-means: Sentiment Distribution by Topic')
            plt.xlabel('Sentiment')
            plt.ylabel('Topic ID')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "variable_k_topic_sentiment_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Query-level sentiment analysis (top 15 queries)
        if len(query_sentiment_pct) > 0:
            plt.figure(figsize=(15, 10))
            top_queries = query_sentiment_pct.head(15)
            
            # Create stacked bar chart
            top_queries.plot(kind='bar', stacked=True, figsize=(15, 8), 
                           color=['#d62728', '#ff7f0e', '#2ca02c'])
            plt.title('Variable K-means: Sentiment Distribution by Search Query (Top 15 Queries)', fontsize=14)
            plt.xlabel('Search Query', fontsize=12)
            plt.ylabel('Percentage', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "variable_k_query_sentiment_stacked.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Variable K-means visualizations saved to: {viz_dir}")
    
    def generate_academic_report(self, df, topic_analysis):
        """Generate comprehensive academic report for Variable K-means sentiment analysis."""
        print("Generating Variable K-means academic report...")
        
        report_path = os.path.join(self.output_dir, "variable_k_academic_report.txt")
        
        report = f"""
=== VARIABLE K-MEANS BERTOPIC SENTIMENT ANALYSIS INTEGRATION ===
Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: YouTube Mathematical Education Comments with Variable K-means Topic Assignments
Algorithm: Variable K-means Clustering (optimised_variable_k_phase_4)

=== OVERALL ANALYSIS SUMMARY ===
Total comments analyzed: {len(df):,}
Unique topics identified: {df['topic'].nunique()}
Unique search queries: {df['search_query'].nunique()}
Processing source: {df['processing_source'].iloc[0] if 'processing_source' in df.columns else 'Variable K-means'}

=== SENTIMENT DISTRIBUTION ===
"""
        
        # Overall sentiment distribution
        sentiment_dist = df['sentiment_label'].value_counts(normalize=True) * 100
        for sentiment, pct in sentiment_dist.items():
            report += f"{sentiment}: {pct:.2f}%\n"
        
        report += f"\nAverage sentiment confidence: {df['sentiment_score'].mean():.3f}\n"
        
        # Topic-level analysis
        report += f"\n=== VARIABLE K-MEANS TOPIC-LEVEL SENTIMENT ANALYSIS ===\n"
        if len(topic_analysis) > 0:
            report += f"Topics with positive majority sentiment: {len(topic_analysis[topic_analysis['dominant_sentiment'] == 'POSITIVE'])}\n"
            report += f"Topics with negative majority sentiment: {len(topic_analysis[topic_analysis['dominant_sentiment'] == 'NEGATIVE'])}\n"
            report += f"Topics with neutral majority sentiment: {len(topic_analysis[topic_analysis['dominant_sentiment'] == 'NEUTRAL'])}\n"
            
            # Most positive/negative topics
            if not topic_analysis.empty:
                most_positive = topic_analysis.loc[topic_analysis['avg_sentiment_score'].idxmax()]
                most_negative = topic_analysis.loc[topic_analysis['avg_sentiment_score'].idxmin()]
                
                report += f"\nMost positive Variable K-means topic: Topic {most_positive['topic']} (score: {most_positive['avg_sentiment_score']:.3f})\n"
                report += f"Most negative Variable K-means topic: Topic {most_negative['topic']} (score: {most_negative['avg_sentiment_score']:.3f})\n"
        
        # Query-level insights
        report += f"\n=== SEARCH QUERY SENTIMENT INSIGHTS ===\n"
        query_avg_sentiment = df.groupby('search_query')['sentiment_score'].mean().sort_values(ascending=False)
        
        report += f"\nMost positive queries:\n"
        for query, score in query_avg_sentiment.head(5).items():
            report += f"  - '{query}': {score:.3f}\n"
        
        report += f"\nMost negative queries:\n"
        for query, score in query_avg_sentiment.tail(5).items():
            report += f"  - '{query}': {score:.3f}\n"
        
        # Algorithm-specific insights
        report += f"\n=== VARIABLE K-MEANS SPECIFIC INSIGHTS ===\n"
        report += f"This analysis applies sentiment analysis to Variable K-means topic assignments.\n"
        report += f"Variable K-means clustering identified {df['topic'].nunique()} distinct topics from\n"
        report += f"mathematical education discourse, providing a different perspective from HDBSCAN.\n"
        report += f"The sentiment patterns can be compared with HDBSCAN results for algorithmic\n"
        report += f"comparison and validation in mathematical education research.\n"
        
        # Educational insights
        report += f"\n=== EDUCATIONAL INSIGHTS ===\n"
        report += f"Variable K-means topic modeling with sentiment analysis reveals specific\n"
        report += f"emotional responses to mathematical education content. This approach enables\n"
        report += f"comparison between different clustering algorithms (Variable K-means vs HDBSCAN)\n"
        report += f"in identifying sentiment patterns within mathematical learning discourse.\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Variable K-means academic report saved to: {report_path}")
    
    def run_complete_analysis(self):
        """Execute the complete Variable K-means sentiment analysis integration."""
        print("=" * 70)
        print("VARIABLE K-MEANS BERTOPIC SENTIMENT ANALYSIS INTEGRATION")
        print("=" * 70)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # Step 1: Load Variable K-means BERTopic results
            df = self.load_variable_k_results()
            
            # Step 2: Process sentiment analysis
            print("\nProcessing sentiment analysis for Variable K-means results...")
            texts = df['comment_text'].tolist()
            sentiments = self.process_sentiment_batch(texts, batch_size=16)
            
            # Step 3: Integrate sentiment results
            sentiment_df = pd.DataFrame(sentiments)
            df = pd.concat([df, sentiment_df], axis=1)
            
            # Step 4: Save enhanced dataset
            enhanced_file = os.path.join(self.output_dir, "variable_k_comments_with_topics_and_sentiment.csv")
            df.to_csv(enhanced_file, index=False)
            print(f"Variable K-means enhanced dataset saved: {enhanced_file}")
            
            # Step 5: Analyze patterns
            _, topic_sentiment_pct, _ = self.analyze_sentiment_by_topic(df)
            _, query_sentiment_pct, _ = self.analyze_sentiment_by_query(df)
            topic_analysis = self.create_topic_sentiment_mapping(df)
            
            # Step 6: Save analysis results
            topic_sentiment_pct.to_csv(os.path.join(self.output_dir, "variable_k_topic_sentiment_percentages.csv"))
            query_sentiment_pct.to_csv(os.path.join(self.output_dir, "variable_k_query_sentiment_percentages.csv"))
            topic_analysis.to_csv(os.path.join(self.output_dir, "variable_k_topic_sentiment_analysis.csv"), index=False)
            
            # Step 7: Create visualizations
            self.create_visualizations(df, topic_sentiment_pct, query_sentiment_pct)
            
            # Step 8: Generate academic report
            self.generate_academic_report(df, topic_analysis)
            
            print("\n" + "=" * 70)
            print("VARIABLE K-MEANS SENTIMENT ANALYSIS INTEGRATION COMPLETED")
            print("=" * 70)
            print(f"Results saved in: {self.output_dir}/")
            print(f"Enhanced dataset: variable_k_comments_with_topics_and_sentiment.csv")
            print(f"Academic report: variable_k_academic_report.txt")
            print(f"Visualizations: visualizations/")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    # Initialize and run Variable K-means sentiment analysis
    analyzer = VariableKMeansSentimentIntegration()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()