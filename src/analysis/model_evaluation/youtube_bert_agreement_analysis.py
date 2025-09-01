#!/usr/bin/env python3
"""
YouTube-BERT Model Agreement Analysis
Evaluates the YouTube-BERT model's agreement with manual annotations.
Adapted from xlm_roberta_clean_agreement_analysis.py for YouTube-BERT results.

MSc Data Science Thesis - Perceptions of Maths on YouTube
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

class YouTubeBERTAgreementAnalyzer:
    def __init__(self):
        """Initialize agreement analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/analysis/model_evaluation"
        
        print("YouTube-BERT Model Agreement Analysis")
        print("Evaluating agreement with manual annotations")
    
    def load_manual_annotations(self):
        """Load the consensus manual annotations."""
        annotations_path = f"{self.output_dir}/consensus_annotations_20250813_222930.csv"
        
        print(f"Loading manual annotations...")
        df_manual = pd.read_csv(annotations_path)
        
        print(f"Loaded {len(df_manual):,} manually annotated samples")
        return df_manual
    
    def load_youtube_bert_results(self):
        """Load YouTube-BERT model results."""
        results_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/models/sentiment_analysis/youtube_bert_sentence_sentiment_analysis_20250813_234226/youtube_bert_comments_with_sentence_sentiment.csv"
        
        print(f"Loading YouTube-BERT results...")
        df_model = pd.read_csv(results_path)
        
        print(f"Loaded {len(df_model):,} YouTube-BERT predictions")
        return df_model
    
    def map_predictions_to_annotations(self, df_manual, df_model):
        """
        Map YouTube-BERT predictions to manual annotations by matching comment text.
        """
        print("Mapping predictions to manual annotations...")
        
        # Create mapping based on comment text matching
        manual_comments = df_manual[['comment_id', 'comment_text', 'consensus_sentiment', 'consensus_journey']].copy()
        
        # Clean whitespace and normalize for matching
        manual_comments['comment_text_clean'] = manual_comments['comment_text'].str.strip().str.lower()
        df_model_clean = df_model.copy()
        df_model_clean['comment_text_clean'] = df_model_clean['comment_text'].str.strip().str.lower()
        
        # Find matches
        matched_data = []
        unmatched_count = 0
        
        for _, manual_row in manual_comments.iterrows():
            manual_text = manual_row['comment_text_clean']
            
            # Find exact text matches in model results
            model_matches = df_model_clean[df_model_clean['comment_text_clean'] == manual_text]
            
            if len(model_matches) > 0:
                # Take first match if multiple found
                model_row = model_matches.iloc[0]
                
                matched_data.append({
                    'manual_comment_id': manual_row['comment_id'],
                    'model_comment_id': model_row['comment_id'],
                    'comment_text': manual_row['comment_text'],
                    'manual_sentiment': manual_row['consensus_sentiment'],
                    'manual_journey': manual_row['consensus_journey'],
                    'model_sentiment': model_row['sentence_level_sentiment'],
                    'model_confidence': model_row.get('final_sentiment_weight', 0.5),
                    'model_journey': model_row['has_transition'],
                    'progression_type': model_row['sentence_progression'],
                    'sentence_count': model_row['sentence_count']
                })
            else:
                unmatched_count += 1
                print(f"No match found for comment {manual_row['comment_id']}: {manual_row['comment_text'][:50]}...")
        
        matched_df = pd.DataFrame(matched_data)
        
        print(f"Successfully matched {len(matched_df):,} comments")
        print(f"Unmatched annotations: {unmatched_count}")
        
        if len(matched_df) == 0:
            raise ValueError("No matching comments found between manual annotations and model results")
        
        return matched_df
    
    def calculate_agreement_metrics(self, matched_df):
        """Calculate comprehensive agreement metrics."""
        print("Calculating agreement metrics...")
        
        # Sentiment agreement
        manual_sentiments = matched_df['manual_sentiment'].values
        model_sentiments = matched_df['model_sentiment'].values
        
        # Basic agreement metrics
        sentiment_accuracy = accuracy_score(manual_sentiments, model_sentiments)
        sentiment_matches = (manual_sentiments == model_sentiments).sum()
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(manual_sentiments, model_sentiments)
        
        # Classification report for detailed metrics
        class_report = classification_report(manual_sentiments, model_sentiments, 
                                           output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(manual_sentiments, model_sentiments, 
                            labels=['positive', 'neutral', 'negative'])
        
        # Learning journey agreement (convert to binary)
        manual_journeys = (matched_df['manual_journey'] == 'yes').astype(int)
        model_journeys = matched_df['model_journey'].astype(int)
        
        journey_accuracy = accuracy_score(manual_journeys, model_journeys)
        journey_matches = (manual_journeys == model_journeys).sum()
        journey_kappa = cohen_kappa_score(manual_journeys, model_journeys)
        
        # Distribution analysis
        manual_dist = matched_df['manual_sentiment'].value_counts().to_dict()
        model_dist = matched_df['model_sentiment'].value_counts().to_dict()
        
        # Journey distribution
        manual_journey_dist = matched_df['manual_journey'].value_counts().to_dict()
        model_journey_dist = matched_df['model_journey'].value_counts().to_dict()
        
        # Detailed sentiment analysis by category
        sentiment_details = {}
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in class_report:
                sentiment_details[sentiment] = {
                    'precision': class_report[sentiment]['precision'],
                    'recall': class_report[sentiment]['recall'],
                    'f1_score': class_report[sentiment]['f1-score'],
                    'support': class_report[sentiment]['support']
                }
        
        metrics = {
            'total_samples': len(matched_df),
            'timestamp': self.timestamp,
            'model_name': 'youtube_bert',
            'model_details': {
                'model_path': 'rahulk98/bert-finetuned-youtube_sentiment_analysis',
                'aggregation_method': 'final_third_weighting_with_learning_journey_detection',
                'features': [
                    'spacy_sentence_segmentation',
                    'learning_journey_detection',
                    'final_third_weighting',
                    'negative_to_positive_progression'
                ]
            },
            'sentiment_agreement': {
                'accuracy': round(sentiment_accuracy, 4),
                'matches': int(sentiment_matches),
                'match_rate': round(sentiment_accuracy, 4),
                'cohens_kappa': round(kappa, 4),
                'macro_f1': round(class_report['macro avg']['f1-score'], 4),
                'weighted_f1': round(class_report['weighted avg']['f1-score'], 4)
            },
            'learning_journey_agreement': {
                'accuracy': round(journey_accuracy, 4),
                'matches': int(journey_matches),
                'match_rate': round(journey_accuracy, 4),
                'cohens_kappa': round(journey_kappa, 4)
            },
            'sentiment_distribution': {
                'manual_annotations': manual_dist,
                'model_predictions': model_dist
            },
            'learning_journey_distribution': {
                'manual_annotations': manual_journey_dist,
                'model_predictions': model_journey_dist
            },
            'detailed_sentiment_metrics': sentiment_details,
            'confusion_matrix': {
                'labels': ['positive', 'neutral', 'negative'],
                'matrix': cm.tolist()
            },
            'model_confidence_stats': {
                'mean': round(matched_df['model_confidence'].mean(), 4),
                'std': round(matched_df['model_confidence'].std(), 4),
                'min': round(matched_df['model_confidence'].min(), 4),
                'max': round(matched_df['model_confidence'].max(), 4)
            },
            'progression_type_distribution': matched_df['progression_type'].value_counts().to_dict()
        }
        
        return metrics, matched_df
    
    def analyze_disagreements(self, matched_df):
        """Analyze cases where model and manual annotations disagree."""
        disagreements = matched_df[matched_df['manual_sentiment'] != matched_df['model_sentiment']].copy()
        
        if len(disagreements) == 0:
            return {}
        
        # Group disagreements by type
        disagreement_patterns_raw = disagreements.groupby(['manual_sentiment', 'model_sentiment']).size().to_dict()
        
        # Convert tuple keys to strings for JSON serialization
        disagreement_patterns = {f"{manual}_to_{model}": count for (manual, model), count in disagreement_patterns_raw.items()}
        
        # Examples of each disagreement type
        disagreement_examples = {}
        for (manual, model), count in disagreement_patterns_raw.items():
            pattern_key = f"{manual}_to_{model}"
            examples = disagreements[
                (disagreements['manual_sentiment'] == manual) & 
                (disagreements['model_sentiment'] == model)
            ]['comment_text'].head(3).tolist()
            
            disagreement_examples[pattern_key] = {
                'count': int(count),
                'percentage': round(count / len(matched_df) * 100, 2),
                'examples': examples
            }
        
        return {
            'total_disagreements': len(disagreements),
            'disagreement_rate': round(len(disagreements) / len(matched_df) * 100, 2),
            'disagreement_patterns': disagreement_patterns,
            'disagreement_examples': disagreement_examples
        }
    
    def generate_comprehensive_report(self, metrics, matched_df):
        """Generate comprehensive evaluation report."""
        disagreement_analysis = self.analyze_disagreements(matched_df)
        
        # Create final comprehensive metrics
        comprehensive_metrics = {
            **metrics,
            'disagreement_analysis': disagreement_analysis,
            'performance_summary': {
                'primary_metric': 'sentiment_accuracy',
                'primary_score': metrics['sentiment_agreement']['accuracy'],
                'benchmark_comparison': {
                    'twitter_roberta_accuracy': 0.37,  # From original analysis
                    'improvement_over_twitter_roberta': round(
                        metrics['sentiment_agreement']['accuracy'] - 0.37, 4
                    )
                }
            },
            'methodology_notes': {
                'matching_method': 'exact_text_matching',
                'preprocessing': 'whitespace_normalization_and_lowercase',
                'evaluation_date': datetime.now().strftime('%Y-%m-%d'),
                'code_version': 'youtube_bert_sentence_analysis',
                'aggregation_confidence': 0.9  # For learning journeys
            }
        }
        
        return comprehensive_metrics, matched_df
    
    def save_results(self, comprehensive_metrics, matched_df):
        """Save evaluation results."""
        output_dir = Path(self.output_dir)
        
        # Save comprehensive metrics JSON
        metrics_file = output_dir / f"youtube_bert_original_agreement_metrics_{self.timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_metrics, f, indent=2, ensure_ascii=False)
        
        # Save detailed matched data
        matched_file = output_dir / f"youtube_bert_original_matched_annotations_{self.timestamp}.csv"
        matched_df.to_csv(matched_file, index=False)
        
        print(f"Results saved:")
        print(f"  Metrics: {metrics_file}")
        print(f"  Matched data: {matched_file}")
        
        return metrics_file, matched_file
    
    def run_agreement_analysis(self):
        """Execute complete agreement analysis."""
        print("=" * 70)
        print("YouTube-BERT Original Model Agreement Analysis")
        print("=" * 70)
        
        try:
            # Load data
            df_manual = self.load_manual_annotations()
            df_model = self.load_youtube_bert_results()
            
            # Map predictions to annotations
            matched_df = self.map_predictions_to_annotations(df_manual, df_model)
            
            # Calculate metrics
            metrics, matched_df = self.calculate_agreement_metrics(matched_df)
            
            # Generate comprehensive report
            comprehensive_metrics, matched_df = self.generate_comprehensive_report(metrics, matched_df)
            
            # Save results
            metrics_file, matched_file = self.save_results(comprehensive_metrics, matched_df)
            
            print("\n" + "=" * 70)
            print("AGREEMENT ANALYSIS COMPLETED")
            print("=" * 70)
            
            # Print summary
            print(f"\nKEY RESULTS:")
            print(f"Total samples evaluated: {comprehensive_metrics['total_samples']}")
            print(f"Sentiment accuracy: {comprehensive_metrics['sentiment_agreement']['accuracy']:.1%}")
            print(f"Cohen's Kappa: {comprehensive_metrics['sentiment_agreement']['cohens_kappa']:.3f}")
            print(f"Learning journey accuracy: {comprehensive_metrics['learning_journey_agreement']['accuracy']:.1%}")
            print(f"Macro F1-score: {comprehensive_metrics['sentiment_agreement']['macro_f1']:.3f}")
            
            # Performance comparison
            perf = comprehensive_metrics['performance_summary']['benchmark_comparison']
            print(f"\nBENCHMARK COMPARISON:")
            print(f"vs Twitter RoBERTa: {perf['improvement_over_twitter_roberta']:+.3f}")
            
            return comprehensive_metrics, matched_df
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main execution function."""
    analyzer = YouTubeBERTAgreementAnalyzer()
    analyzer.run_agreement_analysis()

if __name__ == "__main__":
    main()