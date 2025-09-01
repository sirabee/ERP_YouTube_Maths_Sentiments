#!/usr/bin/env python3
"""
Optimized Phase 4: BERTopic Implementation Using Refined Variable K Strategy

This script runs the final BERTopic models using the statistically validated
optimal K values from the refined strategy analysis.

Key differences from original Phase 4:
1. Uses pre-validated optimal K values (no K optimization needed)
2. Focuses on generating final topic models for analysis
3. Includes statistical validation from refined strategy
4. Ready for sentiment analysis integration
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

class OptimizedPhase4BERTopic:
    def __init__(self, refined_k_mapping_path, output_dir=None):
        """
        Phase 4 implementation using refined strategy validated K values.
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"optimized_phase4_final_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load validated K values from refined strategy
        print("Loading validated K values from refined strategy...")
        with open(refined_k_mapping_path, 'r') as f:
            self.refined_mapping = json.load(f)
        
        self.k_mapping = self.refined_mapping['final_k_mapping']
        self.baseline_k = self.refined_mapping['baseline_k']
        self.improvement_threshold = self.refined_mapping['improvement_threshold']
        
        print(f"Loaded K values for {len(self.k_mapping)} queries")
        print(f"Baseline K: {self.baseline_k}")
        print(f"Statistical threshold: {self.improvement_threshold:.1%}")
        print(f"Queries using baseline: {self.refined_mapping['queries_using_baseline']}")
        print(f"Queries using custom K: {self.refined_mapping['queries_using_custom']}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Results storage
        self.final_models = {}
        self.final_results = []
        
    def load_complete_pipeline_data(self, dataset_path):
        """Load complete pipeline data for final topic modeling."""
        print(f"Loading complete pipeline dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        query_data = {}
        query_counts = df['search_query'].value_counts()
        
        for query in query_counts.index:
            if query in self.k_mapping and query_counts[query] >= 50:
                query_comments = df[df['search_query'] == query]['comment_text'].dropna().tolist()
                query_data[query] = query_comments
        
        print(f"Queries for final modeling: {len(query_data)}")
        return query_data
    
    def create_final_bertopic_model(self, documents, query_name, optimal_k):
        """
        Create final BERTopic model using validated optimal K.
        """
        print(f"Creating final model for '{query_name}' with K={optimal_k}")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
        
        # UMAP dimensionality reduction
        umap_model = UMAP(n_components=5, metric='cosine', random_state=42)
        
        # K-means with validated optimal K
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        
        # Vectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), 
            stop_words="english", 
            max_features=500
        )
        
        # Create final BERTopic model
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=kmeans,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,
            verbose=False
        )
        
        # Fit final model
        topics, probabilities = topic_model.fit_transform(documents)
        
        # Calculate final metrics
        reduced_embeddings = umap_model.fit_transform(embeddings)
        cluster_labels = kmeans.fit_predict(reduced_embeddings)
        silhouette = silhouette_score(reduced_embeddings, cluster_labels)
        
        n_actual_topics = len(set(topics)) - (1 if -1 in topics else 0)
        noise_rate = (np.array(topics) == -1).sum() / len(topics) * 100
        
        # Topic quality assessment
        topic_quality = self.assess_topic_quality(topic_model, optimal_k)
        
        result = {
            'query': query_name,
            'optimal_k': optimal_k,
            'k_source': 'baseline' if optimal_k == self.baseline_k else 'custom',
            'n_documents': len(documents),
            'n_actual_topics': n_actual_topics,
            'noise_rate': noise_rate,
            'silhouette_score': silhouette,
            'topic_quality_score': topic_quality,
            'model_status': 'success'
        }
        
        print(f"  Final metrics: topics={n_actual_topics}, noise={noise_rate:.1f}%, silhouette={silhouette:.3f}")
        
        return topic_model, result
    
    def assess_topic_quality(self, topic_model, k):
        """Assess quality of final topics for educational content."""
        try:
            quality_scores = []
            
            for topic_id in range(k):
                try:
                    topic_words = topic_model.get_topic(topic_id)
                    if not topic_words:
                        continue
                    
                    words = [word for word, _ in topic_words[:10]]
                    
                    # Educational relevance keywords
                    math_keywords = {
                        'math', 'equation', 'solve', 'calculate', 'formula', 
                        'algebra', 'geometry', 'calculus', 'trigonometry', 'statistics',
                        'number', 'problem', 'solution', 'theorem', 'proof', 'learn',
                        'teach', 'understand', 'explain', 'tutorial', 'lesson'
                    }
                    
                    relevance = len(set(words) & math_keywords) / len(words)
                    diversity = len(set(words)) / len(words)
                    
                    topic_score = (relevance * 0.7) + (diversity * 0.3)
                    quality_scores.append(topic_score)
                    
                except Exception:
                    continue
            
            return np.mean(quality_scores) if quality_scores else 0.0
            
        except Exception:
            return 0.0
    
    def run_final_bertopic_modeling(self, dataset_path, max_queries=None):
        """
        Run final BERTopic modeling using validated K values.
        """
        print("="*80)
        print("OPTIMIZED PHASE 4: FINAL BERTOPIC MODELING")
        print("Using Refined Variable K Strategy Validated Results")
        print("="*80)
        print(f"Strategy: Use validated optimal K per query")
        print(f"Statistical validation: {self.improvement_threshold:.1%} improvement threshold")
        print(f"Expected: Superior performance with validated K values")
        print("="*80)
        
        # Load data
        query_data = self.load_complete_pipeline_data(dataset_path)
        
        if not query_data:
            print("Error: No data available for modeling.")
            return None
        
        # Limit queries if specified for testing
        if max_queries:
            query_items = list(query_data.items())[:max_queries]
            query_data = dict(query_items)
            print(f"Limited to first {max_queries} queries for initial run")
        
        print(f"\nProcessing {len(query_data)} queries with validated K values...")
        
        all_results = []
        checkpoint_interval = 5  # Save progress every 5 queries
        
        # Process each query with its validated optimal K
        for i, (query_name, documents) in enumerate(query_data.items(), 1):
            optimal_k = self.k_mapping[query_name]
            print(f"\n[{i}/{len(query_data)}] {query_name} → K={optimal_k} ({len(documents)} docs)")
            
            try:
                # Create final model
                topic_model, result = self.create_final_bertopic_model(
                    documents, query_name, optimal_k
                )
                
                # Store model and results
                self.final_models[query_name] = topic_model
                all_results.append(result)
                
                print(f"  ✓ Model created successfully")
                
            except Exception as e:
                print(f"  ✗ Error creating model for {query_name}: {e}")
                all_results.append({
                    'query': query_name,
                    'optimal_k': optimal_k,
                    'model_status': 'failed',
                    'error': str(e)
                })
            
            # Checkpoint save every N queries
            if i % checkpoint_interval == 0:
                print(f"\n  → Checkpoint: Processed {i}/{len(query_data)} queries")
                self.final_results = all_results
                self.save_checkpoint_results(i)
        
        self.final_results = all_results
        
        # Save final results
        print(f"\nFinalizing results for {len(all_results)} queries...")
        self.save_final_results()
        self.generate_final_analysis()
        
        return all_results, self.final_models
    
    def save_final_results(self):
        """Save final BERTopic modeling results."""
        print("\nSaving final modeling results...")
        
        # Save final results
        results_df = pd.DataFrame(self.final_results)
        results_path = f"{self.output_dir}/final_bertopic_results_{self.timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save topic models (metadata only - models are large)
        model_metadata = {
            'timestamp': self.timestamp,
            'total_models': len(self.final_models),
            'successful_models': len([r for r in self.final_results if r.get('model_status') == 'success']),
            'failed_models': len([r for r in self.final_results if r.get('model_status') == 'failed']),
            'models_available': list(self.final_models.keys()),
            'refined_strategy_source': self.refined_mapping.get('timestamp', 'unknown')
        }
        
        metadata_path = f"{self.output_dir}/model_metadata_{self.timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save individual topic extractions for analysis
        topics_dir = f"{self.output_dir}/topic_extractions"
        os.makedirs(topics_dir, exist_ok=True)
        
        for query_name, model in self.final_models.items():
            try:
                # Extract top topics
                topic_info = []
                for topic_id in range(self.k_mapping[query_name]):
                    topic_words = model.get_topic(topic_id)
                    if topic_words:
                        topic_info.append({
                            'topic_id': topic_id,
                            'words': [word for word, score in topic_words[:10]],
                            'scores': [score for word, score in topic_words[:10]]
                        })
                
                topic_path = f"{topics_dir}/{query_name.replace(' ', '_')}_topics.json"
                with open(topic_path, 'w') as f:
                    json.dump(topic_info, f, indent=2)
                    
            except Exception as e:
                print(f"  Warning: Could not extract topics for {query_name}: {e}")
        
        print(f"Results saved in: {self.output_dir}")
    
    def save_checkpoint_results(self, checkpoint_num):
        """Save checkpoint results during processing."""
        if not self.final_results:
            return
        
        checkpoint_path = f"{self.output_dir}/checkpoint_{checkpoint_num}_results.csv"
        results_df = pd.DataFrame(self.final_results)
        results_df.to_csv(checkpoint_path, index=False)
        print(f"    Checkpoint saved: {checkpoint_path}")
    
    def generate_final_analysis(self):
        """Generate comprehensive final analysis report."""
        if not self.final_results:
            return
        
        successful_results = [r for r in self.final_results if r.get('model_status') == 'success']
        
        report_path = f"{self.output_dir}/optimized_phase4_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("OPTIMIZED PHASE 4: FINAL BERTOPIC MODELING REPORT\n")
            f.write("Using Refined Variable K Strategy Validated Results\n")
            f.write("MSc Data Science Thesis - Perceptions of Maths on YouTube\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Strategy validation
            f.write("REFINED STRATEGY INTEGRATION\n")
            f.write("-"*40 + "\n")
            f.write(f"✓ Used validated K values from refined strategy\n")
            f.write(f"✓ Statistical threshold: {self.improvement_threshold:.1%}\n")
            f.write(f"✓ Baseline K: {self.baseline_k}\n")
            f.write(f"✓ Total queries processed: {len(self.final_results)}\n")
            f.write(f"✓ Successful models: {len(successful_results)}\n")
            f.write(f"✓ Failed models: {len(self.final_results) - len(successful_results)}\n\n")
            
            if successful_results:
                # Performance metrics
                f.write("FINAL PERFORMANCE METRICS\n")
                f.write("-"*40 + "\n")
                
                results_df = pd.DataFrame(successful_results)
                avg_noise = results_df['noise_rate'].mean()
                avg_silhouette = results_df['silhouette_score'].mean()
                avg_quality = results_df['topic_quality_score'].mean()
                
                f.write(f"Average noise rate: {avg_noise:.2f}%\n")
                f.write(f"Average silhouette score: {avg_silhouette:.3f}\n")
                f.write(f"Average topic quality: {avg_quality:.3f}\n")
                
                # Zero noise achievement
                zero_noise = (results_df['noise_rate'] == 0.0).sum()
                zero_noise_pct = (zero_noise / len(results_df)) * 100
                f.write(f"Queries with 0% noise: {zero_noise}/{len(results_df)} ({zero_noise_pct:.1f}%)\n\n")
                
                # K value usage analysis
                f.write("K VALUE USAGE ANALYSIS\n")
                f.write("-"*40 + "\n")
                baseline_count = (results_df['k_source'] == 'baseline').sum()
                custom_count = (results_df['k_source'] == 'custom').sum()
                
                f.write(f"Using baseline K={self.baseline_k}: {baseline_count} queries\n")
                f.write(f"Using custom K: {custom_count} queries\n")
                f.write(f"Custom K benefit: {custom_count}/{len(results_df)} ({custom_count/len(results_df)*100:.1f}%)\n\n")
                
                # Top performing models
                f.write("TOP PERFORMING MODELS\n")
                f.write("-"*40 + "\n")
                top_models = results_df.nlargest(5, 'silhouette_score')
                
                for idx, row in top_models.iterrows():
                    f.write(f"Query: {row['query']}\n")
                    f.write(f"  Optimal K: {row['optimal_k']} ({row['k_source']})\n")
                    f.write(f"  Silhouette: {row['silhouette_score']:.3f}\n")
                    f.write(f"  Noise: {row['noise_rate']:.1f}%\n")
                    f.write(f"  Quality: {row['topic_quality_score']:.3f}\n\n")
            
            f.write("THESIS INTEGRATION STATUS\n")
            f.write("-"*40 + "\n")
            f.write("✓ Final BERTopic models generated with validated K values\n")
            f.write("✓ Topics extracted and saved for analysis\n")
            f.write("✓ Ready for sentiment analysis integration\n")
            f.write("✓ Statistical validation completed via refined strategy\n")
            f.write("✓ Phase 4 methodology evolution complete\n\n")
            
            f.write("NEXT STEPS FOR THESIS\n")
            f.write("-"*40 + "\n")
            f.write("1. Use topic_extractions/ for topic analysis\n")
            f.write("2. Integrate with sentiment analysis pipeline\n")
            f.write("3. Analyze educational content themes\n")
            f.write("4. Compare with Phase 2 HDBSCAN baseline\n")
            f.write("5. Document methodology evolution benefits\n")
        
        print(f"Final analysis report saved: {report_path}")

def main():
    """Execute optimized Phase 4 using refined strategy results."""
    
    # Paths
    refined_mapping_path = "/Users/siradbihi/Desktop/MScDataScience/ERP Maths Sentiments/Complete_Pipeline_Methodology_Evolution_2/refined_variable_k_20250722_213352/refined_k_mapping_20250722_213352.json"
    dataset_path = "/Users/siradbihi/Desktop/MScDataScience/ERP Maths Sentiments/Complete_Pipeline_Organized/comments_pipeline/comments_complete_filtered_20250720_224959.csv"
    
    # Initialize optimized Phase 4
    optimizer = OptimizedPhase4BERTopic(refined_mapping_path)
    
    # Run final modeling for all 80 queries
    print("Running optimized Phase 4 using refined strategy validated K values...")
    results, models = optimizer.run_final_bertopic_modeling(dataset_path)
    
    if results and models:
        print("\n" + "="*80)
        print("OPTIMIZED PHASE 4 COMPLETE")
        print("="*80)
        print(f"Results directory: {optimizer.output_dir}")
        print(f"Final models created: {len(models)}")
        print(f"Topics extracted for {len(models)} queries")
        print("Ready for sentiment analysis and thesis integration")
        
        # Success summary
        successful = len([r for r in results if r.get('model_status') == 'success'])
        print(f"\nSuccess Summary:")
        print(f"  Successful models: {successful}/{len(results)}")
        print(f"  Success rate: {successful/len(results)*100:.1f}%")
        print(f"  Using refined strategy validated K values")
        print("="*80)
    else:
        print("Optimized Phase 4 failed. Please check paths and requirements.")

if __name__ == "__main__":
    main()