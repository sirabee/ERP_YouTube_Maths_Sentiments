#!/usr/bin/env python3
"""
Enhanced Sentence-Level Sentiment Aggregation Methods
Implements hierarchical, aspect-based, and ensemble approaches for educational comments.
Compares performance against original aggregation and manual annotations.

MSc Data Science Thesis - Perceptions of Maths on YouTube
"""

import pandas as pd
import numpy as np
import re
from transformers import pipeline
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedSentimentAggregator:
    def __init__(self):
        """Initialize enhanced sentiment aggregator with Twitter RoBERTa."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize Twitter RoBERTa sentiment pipeline
        print("Loading Twitter RoBERTa sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,
            truncation=True,
            max_length=512
        )
        
        # Define educational aspects and keywords
        self.educational_aspects = {
            'difficulty': ['hard', 'difficult', 'confus', 'struggle', 'trouble', 'don\'t understand', 'stuck'],
            'understanding': ['understand', 'clear', 'makes sense', 'get it', 'see', 'obvious', 'simple'],
            'gratitude': ['thank', 'thanks', 'appreciate', 'helpful', 'great', 'amazing', 'awesome'],
            'content_quality': ['good', 'excellent', 'perfect', 'brilliant', 'bad', 'poor', 'terrible', 'useless'],
            'learning_progress': ['learn', 'finally', 'now i', 'got it', 'figured', 'breakthrough', 'click']
        }
        
        print("Enhanced sentiment aggregator initialized successfully.")
    
    def analyze_sentence_sentiment(self, sentence):
        """Analyze sentiment of a single sentence using Twitter RoBERTa."""
        try:
            result = self.sentiment_pipeline(sentence)[0]
            
            # Normalize labels
            label_mapping = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative', 
                'NEUTRAL': 'neutral',
                'LABEL_2': 'positive',
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral'
            }
            
            normalized_label = label_mapping.get(result['label'], result['label'].lower())
            
            return {
                'sentence': sentence,
                'sentiment': normalized_label,
                'confidence': result['score']
            }
        except Exception as e:
            print(f"Error analyzing sentence: {e}")
            return {
                'sentence': sentence,
                'sentiment': 'neutral',
                'confidence': 0.5
            }
    
    def extract_phrases(self, sentence):
        """Extract meaningful phrases from a sentence for hierarchical analysis."""
        # Split on common punctuation and conjunctions
        phrase_separators = r'[,.;!?]|\band\b|\bbut\b|\bhowever\b|\balthough\b|\byet\b'
        phrases = re.split(phrase_separators, sentence, flags=re.IGNORECASE)
        
        # Clean and filter phrases
        phrases = [phrase.strip() for phrase in phrases if len(phrase.strip()) > 5]
        return phrases if phrases else [sentence]
    
    def hierarchical_sentiment_aggregation(self, comment_text):
        """
        Method 1: Hierarchical Sentiment Analysis
        Analyzes phrases within sentences, then aggregates hierarchically.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', comment_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 'neutral', 0.5, {'method': 'hierarchical', 'sentences': 0, 'phrases': 0}
        
        sentence_results = []
        total_phrases = 0
        
        for sentence in sentences:
            # Extract phrases from sentence
            phrases = self.extract_phrases(sentence)
            total_phrases += len(phrases)
            
            # Analyze each phrase
            phrase_sentiments = []
            phrase_confidences = []
            
            for phrase in phrases:
                phrase_result = self.analyze_sentence_sentiment(phrase)
                phrase_sentiments.append(phrase_result['sentiment'])
                phrase_confidences.append(phrase_result['confidence'])
            
            # Aggregate phrase sentiments to sentence level (confidence-weighted)
            if phrase_sentiments:
                # Weight by confidence
                sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
                total_weight = sum(phrase_confidences)
                
                for sentiment, confidence in zip(phrase_sentiments, phrase_confidences):
                    sentiment_scores[sentiment] += confidence
                
                sentence_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                sentence_confidence = sentiment_scores[sentence_sentiment] / total_weight if total_weight > 0 else 0.5
            else:
                sentence_sentiment = 'neutral'
                sentence_confidence = 0.5
            
            sentence_results.append({
                'sentence': sentence,
                'sentiment': sentence_sentiment,
                'confidence': sentence_confidence,
                'phrases': phrases
            })
        
        # Aggregate sentences to comment level (position + confidence weighted)
        if len(sentence_results) == 1:
            final_sentiment = sentence_results[0]['sentiment']
            final_confidence = sentence_results[0]['confidence']
        else:
            # Apply position weighting (later sentences more important for learning journeys)
            weighted_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
            total_weight = 0
            
            for i, result in enumerate(sentence_results):
                position_weight = (i + 1) / len(sentence_results)  # Linear increase
                confidence_weight = result['confidence']
                combined_weight = position_weight * confidence_weight
                
                weighted_scores[result['sentiment']] += combined_weight
                total_weight += combined_weight
            
            final_sentiment = max(weighted_scores, key=weighted_scores.get)
            final_confidence = weighted_scores[final_sentiment] / total_weight if total_weight > 0 else 0.5
        
        metadata = {
            'method': 'hierarchical',
            'sentences': len(sentence_results),
            'phrases': total_phrases
        }
        
        return final_sentiment, final_confidence, metadata
    
    def aspect_based_sentiment_aggregation(self, comment_text):
        """
        Method 2: Aspect-Based Sentiment Analysis (ABSA)
        Identifies educational aspects and weights sentiment accordingly.
        """
        # Analyze overall sentiment first
        overall_result = self.analyze_sentence_sentiment(comment_text)
        
        # Detect educational aspects
        text_lower = comment_text.lower()
        detected_aspects = {}
        aspect_weights = {
            'gratitude': 0.35,        # Highest weight - clear positive indicator
            'learning_progress': 0.25, # Learning journey indicator
            'understanding': 0.20,     # Comprehension indicator
            'content_quality': 0.15,   # Content evaluation
            'difficulty': 0.05         # Lowest weight - often negative but part of learning
        }
        
        # Check for each aspect
        for aspect, keywords in self.educational_aspects.items():
            aspect_score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    aspect_score += 1
            
            if aspect_score > 0:
                detected_aspects[aspect] = aspect_score
        
        if not detected_aspects:
            # No educational aspects detected, use overall sentiment
            return overall_result['sentiment'], overall_result['confidence'], {
                'method': 'absa',
                'aspects_detected': [],
                'aspect_override': False
            }
        
        # Calculate aspect-weighted sentiment
        total_weight = 0
        positive_weight = 0
        negative_weight = 0
        
        for aspect, score in detected_aspects.items():
            weight = aspect_weights[aspect] * score
            total_weight += weight
            
            # Determine aspect sentiment bias
            if aspect in ['gratitude', 'understanding', 'learning_progress']:
                positive_weight += weight
            elif aspect == 'content_quality':
                # Neutral aspect - depends on context
                if any(pos_word in text_lower for pos_word in ['good', 'excellent', 'perfect', 'brilliant']):
                    positive_weight += weight
                elif any(neg_word in text_lower for neg_word in ['bad', 'poor', 'terrible', 'useless']):
                    negative_weight += weight
            elif aspect == 'difficulty':
                # Usually indicates struggle but part of learning process
                negative_weight += weight * 0.5  # Reduced impact
        
        # Determine final sentiment based on aspect analysis
        if positive_weight > negative_weight * 1.5:  # Bias toward positive for educational content
            final_sentiment = 'positive'
            final_confidence = min(0.9, 0.6 + (positive_weight / total_weight) * 0.3)
        elif negative_weight > positive_weight:
            final_sentiment = 'negative'
            final_confidence = min(0.8, 0.6 + (negative_weight / total_weight) * 0.2)
        else:
            final_sentiment = 'neutral'
            final_confidence = 0.6
        
        # If original confidence was very high, blend with aspect-based result
        if overall_result['confidence'] > 0.8:
            if overall_result['sentiment'] == final_sentiment:
                final_confidence = max(final_confidence, overall_result['confidence'])
            else:
                # Conflict between overall and aspect-based - use ensemble
                final_confidence = min(final_confidence, 0.7)
        
        metadata = {
            'method': 'absa',
            'aspects_detected': list(detected_aspects.keys()),
            'aspect_override': final_sentiment != overall_result['sentiment']
        }
        
        return final_sentiment, final_confidence, metadata
    
    def original_sentence_aggregation(self, comment_text):
        """
        Original Method: Simple sentence-level analysis with basic aggregation
        (Mimics the original approach for comparison)
        """
        sentences = re.split(r'[.!?]+', comment_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 'neutral', 0.5, {'method': 'original', 'sentences': 0}
        
        # Analyze each sentence
        sentence_results = []
        for sentence in sentences:
            result = self.analyze_sentence_sentiment(sentence)
            sentence_results.append(result)
        
        # Simple majority voting with basic confidence weighting
        sentiments = [r['sentiment'] for r in sentence_results]
        confidences = [r['confidence'] for r in sentence_results]
        
        # Check for learning journey pattern (basic)
        has_negative_start = any(s == 'negative' for s in sentiments[:len(sentiments)//2]) if len(sentiments) > 1 else False
        has_positive_end = any(s == 'positive' for s in sentiments[len(sentiments)//2:]) if len(sentiments) > 1 else False
        
        if has_negative_start and has_positive_end:
            # Learning journey detected - favor positive
            final_sentiment = 'positive'
            final_confidence = 0.7
        else:
            # Simple majority voting
            sentiment_counts = {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            }
            final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            final_confidence = np.mean(confidences) if confidences else 0.5
        
        metadata = {
            'method': 'original',
            'sentences': len(sentence_results),
            'learning_journey': has_negative_start and has_positive_end
        }
        
        return final_sentiment, final_confidence, metadata
    
    def ensemble_aggregation(self, comment_text):
        """
        Method 3: Ensemble Learning Approach
        Combines hierarchical and aspect-based methods with original approach.
        """
        # Get predictions from all three methods
        hierarchical_result = self.hierarchical_sentiment_aggregation(comment_text)
        absa_result = self.aspect_based_sentiment_aggregation(comment_text)
        original_result = self.original_sentence_aggregation(comment_text)
        
        methods = [
            {'sentiment': hierarchical_result[0], 'confidence': hierarchical_result[1], 'name': 'hierarchical'},
            {'sentiment': absa_result[0], 'confidence': absa_result[1], 'name': 'absa'},
            {'sentiment': original_result[0], 'confidence': original_result[1], 'name': 'original'}
        ]
        
        # Weighted voting based on confidence and method reliability
        method_weights = {
            'hierarchical': 0.3,  # Good for complex comments
            'absa': 0.4,          # Best for educational content
            'original': 0.3       # Baseline comparison
        }
        
        # If ABSA detected educational aspects, increase its weight
        if absa_result[2].get('aspects_detected'):
            method_weights['absa'] = 0.5
            method_weights['hierarchical'] = 0.3
            method_weights['original'] = 0.2
        
        # Calculate weighted votes
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_weight = 0
        
        for method in methods:
            base_weight = method_weights[method['name']]
            confidence_weight = method['confidence']
            combined_weight = base_weight * confidence_weight
            
            sentiment_scores[method['sentiment']] += combined_weight
            total_weight += combined_weight
        
        # Determine final sentiment
        final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        final_confidence = sentiment_scores[final_sentiment] / total_weight if total_weight > 0 else 0.5
        
        # Consensus boost - if all methods agree, increase confidence
        unique_sentiments = set(method['sentiment'] for method in methods)
        if len(unique_sentiments) == 1:
            final_confidence = min(0.95, final_confidence * 1.2)
        
        metadata = {
            'method': 'ensemble',
            'component_results': {
                'hierarchical': hierarchical_result[0],
                'absa': absa_result[0],
                'original': original_result[0]
            },
            'consensus': len(unique_sentiments) == 1,
            'absa_aspects': absa_result[2].get('aspects_detected', [])
        }
        
        return final_sentiment, final_confidence, metadata

def load_sample_data():
    """Load 200 sample comments with manual annotations."""
    base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
    
    # Load the detailed predictions file with manual annotations
    detailed_file = base_path / "results" / "analysis" / "model_evaluation" / "detailed_model_predictions_20250813_233026.csv"
    
    if not detailed_file.exists():
        print(f"Error: Required file not found: {detailed_file}")
        return None
    
    df = pd.read_csv(detailed_file)
    print(f"Loaded {len(df)} samples from detailed predictions file")
    
    # Take first 200 samples
    sample_df = df.head(200).copy()
    
    # Ensure we have the required columns
    required_columns = ['comment_text', 'consensus_sentiment', 'baseline_prediction']
    missing_columns = [col for col in required_columns if col not in sample_df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    print(f"Using {len(sample_df)} samples for analysis")
    print(f"Manual annotation distribution:")
    print(sample_df['consensus_sentiment'].value_counts())
    
    return sample_df

def analyze_samples(aggregator, sample_df):
    """Analyze all samples using different aggregation methods."""
    
    methods = ['original', 'hierarchical', 'absa', 'ensemble']
    results = {method: {'predictions': [], 'confidences': [], 'metadata': []} for method in methods}
    
    print("\nAnalyzing samples with enhanced aggregation methods...")
    
    for idx, row in sample_df.iterrows():
        comment_text = row['comment_text']
        
        # Apply each method
        original_result = aggregator.original_sentence_aggregation(comment_text)
        hierarchical_result = aggregator.hierarchical_sentiment_aggregation(comment_text)
        absa_result = aggregator.aspect_based_sentiment_aggregation(comment_text)
        ensemble_result = aggregator.ensemble_aggregation(comment_text)
        
        # Store results
        method_results = [original_result, hierarchical_result, absa_result, ensemble_result]
        
        for method, result in zip(methods, method_results):
            results[method]['predictions'].append(result[0])
            results[method]['confidences'].append(result[1])
            results[method]['metadata'].append(result[2])
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/200 samples...")
    
    return results

def calculate_performance_metrics(true_labels, predictions, method_name):
    """Calculate performance metrics for a given method."""
    
    accuracy = accuracy_score(true_labels, predictions)
    kappa = cohen_kappa_score(true_labels, predictions)
    
    # Classification report
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'kappa': kappa,
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1_score': report['macro avg']['f1-score']
    }

def create_comparison_report(sample_df, results, output_dir):
    """Create comprehensive comparison report."""
    
    true_labels = sample_df['consensus_sentiment'].tolist()
    baseline_predictions = sample_df['baseline_prediction'].tolist()
    
    print("\n" + "="*80)
    print("ENHANCED AGGREGATION METHODS - PERFORMANCE COMPARISON")
    print("="*80)
    
    # Calculate performance for baseline (original sentence-level from file)
    baseline_metrics = calculate_performance_metrics(true_labels, baseline_predictions, 'Baseline (Original File)')
    
    # Calculate performance for each new method
    method_metrics = []
    for method in ['original', 'hierarchical', 'absa', 'ensemble']:
        predictions = results[method]['predictions']
        metrics = calculate_performance_metrics(true_labels, predictions, method.title())
        method_metrics.append(metrics)
    
    # Display results
    all_metrics = [baseline_metrics] + method_metrics
    
    print(f"\n{'Method':<25} | {'Accuracy':<8} | {'Kappa':<6} | {'F1-Score':<8} | {'Precision':<9} | {'Recall':<6}")
    print("-" * 80)
    
    for metrics in all_metrics:
        print(f"{metrics['method']:<25} | {metrics['accuracy']:<8.3f} | {metrics['kappa']:<6.3f} | "
              f"{metrics['f1_score']:<8.3f} | {metrics['precision']:<9.3f} | {metrics['recall']:<6.3f}")
    
    # Find best performing method
    best_method = max(method_metrics, key=lambda x: x['accuracy'])
    improvement = best_method['accuracy'] - baseline_metrics['accuracy']
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print(f"{'='*80}")
    print(f"Best performing method: {best_method['method']}")
    print(f"Improvement over baseline: {improvement:+.3f} ({improvement*100:+.1f} percentage points)")
    print(f"Best method Kappa score: {best_method['kappa']:.3f}")
    
    # Analyze method-specific insights
    print(f"\nMETHOD-SPECIFIC INSIGHTS:")
    print("-" * 40)
    
    # ABSA insights
    absa_metadata = results['absa']['metadata']
    aspect_detections = sum(1 for meta in absa_metadata if meta.get('aspects_detected'))
    print(f"ABSA detected educational aspects in {aspect_detections}/200 comments ({aspect_detections/2:.1f}%)")
    
    # Ensemble insights
    ensemble_metadata = results['ensemble']['metadata']
    consensus_count = sum(1 for meta in ensemble_metadata if meta.get('consensus', False))
    print(f"Ensemble achieved consensus in {consensus_count}/200 comments ({consensus_count/2:.1f}%)")
    
    # Hierarchical insights
    hierarchical_metadata = results['hierarchical']['metadata']
    avg_phrases = np.mean([meta.get('phrases', 0) for meta in hierarchical_metadata])
    print(f"Hierarchical analysis averaged {avg_phrases:.1f} phrases per comment")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"enhanced_aggregation_comparison_{timestamp}.csv"
    
    # Create detailed comparison DataFrame
    comparison_data = []
    for idx, row in sample_df.iterrows():
        data_row = {
            'comment_id': idx,
            'comment_text': row['comment_text'],
            'manual_annotation': true_labels[idx],
            'baseline_prediction': baseline_predictions[idx],
            'original_prediction': results['original']['predictions'][idx],
            'hierarchical_prediction': results['hierarchical']['predictions'][idx],
            'absa_prediction': results['absa']['predictions'][idx],
            'ensemble_prediction': results['ensemble']['predictions'][idx],
            'baseline_correct': baseline_predictions[idx] == true_labels[idx],
            'original_correct': results['original']['predictions'][idx] == true_labels[idx],
            'hierarchical_correct': results['hierarchical']['predictions'][idx] == true_labels[idx],
            'absa_correct': results['absa']['predictions'][idx] == true_labels[idx],
            'ensemble_correct': results['ensemble']['predictions'][idx] == true_labels[idx]
        }
        comparison_data.append(data_row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved: {output_file}")
    
    return all_metrics, best_method

def main():
    """Main execution function."""
    print("Enhanced Sentence-Level Sentiment Aggregation Comparison")
    print("=" * 60)
    
    # Load sample data
    sample_df = load_sample_data()
    if sample_df is None:
        return
    
    # Initialize aggregator
    aggregator = EnhancedSentimentAggregator()
    
    # Analyze samples
    results = analyze_samples(aggregator, sample_df)
    
    # Create output directory
    output_dir = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/analysis/enhanced_aggregation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison report
    all_metrics, best_method = create_comparison_report(sample_df, results, output_dir)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()