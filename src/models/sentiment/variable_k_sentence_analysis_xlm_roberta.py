#!/usr/bin/env python3
"""
Variable K-means Per-Sentence Sentiment Analysis for Educational Comments - XLM-RoBERTa Version
Implements sentence-level analysis for Variable K-means BERTopic results
Uses XLM-RoBERTa YouTube model instead of Twitter-RoBERTa
"""

import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import spacy
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

class VariableKMeansSentenceLevelAnalyzer:
    def __init__(self, comments_file=None):
        """Initialize Variable K-means sentence-level sentiment analyzer with XLM-RoBERTa."""
        if comments_file is None:
            comments_file = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/models/bertopic_outputs/Optimised_Variable_K/merged_topic_info/optimised_variable_k_phase_4_20250722_224755/results/all_comments_with_topics.csv"
        self.comments_file = comments_file
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/Sentiment_Analysis/Variable_K_Means/xlm_roberta_variable_k_sentence_sentiment_analysis_{self.timestamp}"
        
        # Setup device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("Variable K-means Per-Sentence Analysis with XLM-RoBERTa YouTube Model Initialization")
        
        # Initialize sentiment pipeline with XLM-RoBERTa YouTube model
        print("Loading XLM-RoBERTa YouTube sentiment model for Variable K-means analysis...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual",
            device=-1,  # Use CPU for compatibility
            batch_size=16,  # Reduced for sentence-level processing
            truncation=True,
            max_length=512
        )
        
        # Load spaCy for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("SpaCy model loaded successfully for Variable K-means sentence segmentation.")
        except OSError:
            print("Warning: SpaCy English model not found.")
            print("Install with: python -m spacy download en_core_web_sm")
            print("Falling back to regex-based sentence splitting.")
            self.nlp = None
    
    def segment_sentences(self, text):
        """Segment text into sentences using spaCy or regex fallback."""
        if self.nlp:
            # Use spaCy for accurate sentence segmentation
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to regex-based splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out very short sentences (less than 5 characters)
        sentences = [s for s in sentences if len(s) >= 5]
        return sentences
    
    def analyze_sentence_sentiment(self, sentence):
        """Analyze sentiment of a single sentence."""
        try:
            result = self.sentiment_pipeline(sentence)[0]
            
            # Normalize labels to consistent format
            label_mapping = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative', 
                'NEUTRAL': 'neutral',
                'POS': 'positive',
                'NEG': 'negative',
                'NEU': 'neutral',
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'LABEL_0': 'negative',  # XLM-RoBERTa specific mapping
                'LABEL_1': 'neutral',   # XLM-RoBERTa specific mapping
                'LABEL_2': 'positive'   # XLM-RoBERTa specific mapping
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
    
    def analyze_comment_progression(self, comment_text):
        """Analyze sentiment progression through sentences in a comment."""
        sentences = self.segment_sentences(comment_text)
        
        if len(sentences) == 0:
            return {
                'original_text': comment_text,
                'sentence_count': 0,
                'sentence_sentiments': [],
                'overall_sentiment': 'neutral',
                'sentiment_progression': 'no_sentences',
                'has_transition': False,
                'transition_type': None,
                'final_sentiment_weight': 1.0
            }
        
        # Analyze each sentence
        sentence_results = []
        for sentence in sentences:
            result = self.analyze_sentence_sentiment(sentence)
            sentence_results.append(result)
        
        # Determine overall sentiment patterns
        sentiments = [r['sentiment'] for r in sentence_results]
        confidences = [r['confidence'] for r in sentence_results]
        
        # Check for sentiment transitions
        transition_analysis = self.detect_sentiment_transitions(sentiments)
        
        # Calculate weighted final sentiment
        final_sentiment, final_weight = self.calculate_weighted_sentiment(
            sentence_results, transition_analysis
        )
        
        return {
            'original_text': comment_text,
            'sentence_count': len(sentences),
            'sentence_sentiments': sentence_results,
            'overall_sentiment': final_sentiment,
            'sentiment_progression': transition_analysis['progression_type'],
            'has_transition': transition_analysis['has_transition'],
            'transition_type': transition_analysis['transition_type'],
            'final_sentiment_weight': final_weight,
            'avg_confidence': np.mean(confidences) if confidences else 0.5
        }
    
    def detect_sentiment_transitions(self, sentiments):
        """Detect sentiment transitions within a comment."""
        if len(sentiments) <= 1:
            return {
                'progression_type': 'single_sentence',
                'has_transition': False,
                'transition_type': None
            }
        
        # Check for negative-to-positive transitions (learning journeys)
        first_half = sentiments[:len(sentiments)//2]
        second_half = sentiments[len(sentiments)//2:]
        
        has_negative_start = 'negative' in first_half
        has_positive_end = 'positive' in second_half
        
        # Check for consecutive transitions
        consecutive_transitions = []
        for i in range(len(sentiments) - 1):
            if sentiments[i] != sentiments[i + 1]:
                consecutive_transitions.append((sentiments[i], sentiments[i + 1]))
        
        # Determine transition type
        if has_negative_start and has_positive_end:
            transition_type = 'negative_to_positive'
            has_transition = True
            progression_type = 'learning_journey'
        elif len(consecutive_transitions) >= 2:
            transition_type = 'multiple_transitions'
            has_transition = True
            progression_type = 'complex_progression'
        elif len(consecutive_transitions) == 1:
            transition_type = f"{consecutive_transitions[0][0]}_to_{consecutive_transitions[0][1]}"
            has_transition = True
            progression_type = 'simple_transition'
        else:
            transition_type = None
            has_transition = False
            progression_type = 'stable'
        
        return {
            'progression_type': progression_type,
            'has_transition': has_transition,
            'transition_type': transition_type,
            'consecutive_transitions': consecutive_transitions
        }
    
    def calculate_weighted_sentiment(self, sentence_results, transition_analysis):
        """Calculate weighted final sentiment based on progression patterns."""
        if len(sentence_results) == 0:
            return 'neutral', 1.0
        
        sentiments = [r['sentiment'] for r in sentence_results]
        
        # For learning journeys, weight the final sentences more heavily
        if transition_analysis['progression_type'] == 'learning_journey':
            # Give 70% weight to the final third of sentences
            final_third_size = max(1, len(sentence_results) // 3)
            final_third_start = max(0, len(sentence_results) - final_third_size)
            final_sentiments = sentiments[final_third_start:]
            
            if 'positive' in final_sentiments:
                return 'positive', 0.9  # High confidence for learning journey
            elif 'negative' in final_sentiments:
                return 'negative', 0.7
            else:
                return 'neutral', 0.6
        
        # For stable progression, use simple majority voting
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        # Get the most common sentiment
        final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        # Calculate confidence based on distribution
        total_sentences = len(sentiments)
        majority_count = sentiment_counts[final_sentiment]
        confidence = majority_count / total_sentences
        
        return final_sentiment, confidence
    
    def process_dataset(self):
        """Process the Variable K-means dataset with sentence-level analysis."""
        print(f"Loading Variable K-means dataset: {self.comments_file}")
        df = pd.read_csv(self.comments_file)
        
        print(f"Loaded {len(df):,} comments with Variable K-means topic assignments")
        print(f"Unique Variable K-means topics: {df['topic'].nunique() if 'topic' in df.columns else 'N/A'}")
        print(f"Unique search queries: {df['search_query'].nunique() if 'search_query' in df.columns else 'N/A'}")
        print(f"Algorithm: Variable K-means clustering")
        print(f"Model: XLM-RoBERTa YouTube (AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual)")
        df['comment_text'] = df['comment_text'].astype(str).fillna('')
        df = df[df['comment_text'].str.len() > 0]
        
        print(f"Processing {len(df):,} Variable K-means comments with XLM-RoBERTa sentence-level analysis...")
        
        results = []
        for idx, comment in tqdm(enumerate(df['comment_text']), total=len(df), desc="Analyzing Variable K-means comments with XLM-RoBERTa"):
            result = self.analyze_comment_progression(comment)
            results.append(result)
            
            # Memory cleanup every 1000 comments
            if idx % 1000 == 0 and idx > 0:
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
        
        # Create results DataFrame
        sentence_results = []
        for i, result in enumerate(results):
            base_data = {
                'comment_id': i,
                'original_text': result['original_text'],
                'sentence_count': result['sentence_count'],
                'overall_sentiment': result['overall_sentiment'],
                'sentiment_progression': result['sentiment_progression'],
                'has_transition': result['has_transition'],
                'transition_type': result['transition_type'],
                'final_sentiment_weight': result['final_sentiment_weight'],
                'avg_confidence': result.get('avg_confidence', 0.5)
            }
            
            # Add sentence-level details
            for j, sent_result in enumerate(result['sentence_sentiments']):
                sentence_data = base_data.copy()
                sentence_data.update({
                    'sentence_id': j,
                    'sentence_text': sent_result['sentence'],
                    'sentence_sentiment': sent_result['sentiment'],
                    'sentence_confidence': sent_result['confidence']
                })
                sentence_results.append(sentence_data)
        
        sentence_df = pd.DataFrame(sentence_results)
        
        # Merge with original dataset
        comment_level_data = []
        for i, result in enumerate(results):
            comment_data = {
                'comment_id': i,
                'sentence_level_sentiment': result['overall_sentiment'],
                'sentence_progression': result['sentiment_progression'],
                'has_transition': result['has_transition'],
                'transition_type': result['transition_type'],
                'sentence_count': result['sentence_count'],
                'final_sentiment_weight': result['final_sentiment_weight']
            }
            comment_level_data.append(comment_data)
        
        comment_df = pd.DataFrame(comment_level_data)
        enhanced_df = pd.concat([df.reset_index(drop=True), comment_df], axis=1)
        
        return enhanced_df, sentence_df
    
    def create_analysis_report(self, enhanced_df, sentence_df):
        """Create comprehensive Variable K-means analysis report."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        report_path = f"{self.output_dir}/xlm_roberta_variable_k_sentence_analysis_report.txt"
        
        # Calculate statistics
        total_comments = len(enhanced_df)
        learning_journeys = len(enhanced_df[enhanced_df['sentence_progression'] == 'learning_journey'])
        transitions = len(enhanced_df[enhanced_df['has_transition'] == True])
        
        sentiment_dist = enhanced_df['sentence_level_sentiment'].value_counts()
        progression_dist = enhanced_df['sentence_progression'].value_counts()
        
        report = f"""
=== XLM-ROBERTA VARIABLE K-MEANS SENTENCE-LEVEL SENTIMENT ANALYSIS REPORT ===
Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Algorithm: Variable K-means clustering with BERTopic
Model: XLM-RoBERTa YouTube (AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual)
Method: Per-sentence sentiment analysis with progression detection
Dataset: YouTube Mathematical Education Comments

=== SUMMARY STATISTICS ===
Variable K-means Topics Analyzed: {enhanced_df['topic'].nunique() if 'topic' in enhanced_df.columns else 'N/A'}
Search Queries Covered: {enhanced_df['search_query'].nunique() if 'search_query' in enhanced_df.columns else 'N/A'}
Total comments analyzed: {total_comments:,}
Comments with sentiment transitions: {transitions:,} ({transitions/total_comments*100:.2f}%)
Learning journeys detected: {learning_journeys:,} ({learning_journeys/total_comments*100:.2f}%)

=== SENTENCE-LEVEL SENTIMENT DISTRIBUTION (XLM-RoBERTa) ===
"""
        
        for sentiment, count in sentiment_dist.items():
            percentage = count / total_comments * 100
            report += f"{sentiment.upper()}: {count:,} ({percentage:.2f}%)\n"
        
        report += f"\n=== SENTIMENT PROGRESSION PATTERNS ===\n"
        for progression, count in progression_dist.items():
            percentage = count / total_comments * 100
            report += f"{progression}: {count:,} ({percentage:.2f}%)\n"
        
        # Examples of learning journeys
        journey_examples = enhanced_df[
            enhanced_df['sentence_progression'] == 'learning_journey'
        ]['comment_text'].head(3)
        
        if not journey_examples.empty:
            report += f"\n=== LEARNING JOURNEY EXAMPLES ===\n"
            for i, example in enumerate(journey_examples, 1):
                report += f"\nExample {i}:\n{example[:300]}...\n"
        
        report += f"""
=== XLM-ROBERTA VS TWITTER-ROBERTA COMPARISON ===
This analysis uses XLM-RoBERTa (YouTube-trained) instead of Twitter-RoBERTa (Twitter-trained)
with the SAME Variable K-means aggregation method (final-third weighting).
This allows direct comparison of model performance with identical aggregation.

Key differences from Twitter-RoBERTa:
- Model training: YouTube comments vs Twitter social media
- Expected improvement in educational context understanding
- Better handling of mathematical discourse patterns

Average sentences per comment: {enhanced_df['sentence_count'].mean():.1f}
Comments with multiple sentences: {len(enhanced_df[enhanced_df['sentence_count'] > 1]):,}

=== METHODOLOGY ===
- Topic Modeling: Variable K-means clustering with BERTopic
- Sentence segmentation: spaCy English model
- Sentiment analysis: XLM-RoBERTa YouTube (AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual)
- Learning journey detection: Negative-to-positive progression patterns
- Final sentiment weighting: Educational context-aware algorithms (same as Twitter-RoBERTa)
- Comparative Analysis: Direct comparison with Twitter-RoBERTa using identical aggregation
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"XLM-RoBERTa Variable K-means analysis report saved: {report_path}")
    
    def create_visualizations(self, enhanced_df, sentence_df):
        """Create visualizations for Variable K-means sentence-level analysis."""
        viz_dir = f"{self.output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Sentiment progression distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        progression_counts = enhanced_df['sentiment_progression'].value_counts()
        plt.pie(progression_counts.values, labels=progression_counts.index, autopct='%1.1f%%')
        plt.title('XLM-RoBERTa Variable K-means: Sentiment Progression Patterns')
        
        plt.subplot(1, 2, 2)
        sentiment_counts = enhanced_df['sentence_level_sentiment'].value_counts()
        # Safe color assignment
        colors = plt.cm.viridis(np.linspace(0, 1, len(sentiment_counts)))
        plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        plt.title('XLM-RoBERTa Variable K-means: Final Sentiment Distribution')
        plt.ylabel('Number of Comments')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/xlm_roberta_variable_k_sentiment_progression_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Learning journey analysis
        if len(enhanced_df[enhanced_df['sentence_progression'] == 'learning_journey']) > 0:
            plt.figure(figsize=(10, 6))
            
            journey_data = enhanced_df[enhanced_df['sentence_progression'] == 'learning_journey']
            
            plt.hist(journey_data['sentence_count'], bins=range(1, 15), alpha=0.7, color='skyblue')
            plt.xlabel('Number of Sentences')
            plt.ylabel('Number of Learning Journey Comments')
            plt.title('XLM-RoBERTa Variable K-means: Sentence Count in Learning Journey Comments')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/xlm_roberta_variable_k_learning_journey_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"XLM-RoBERTa Variable K-means visualizations saved to: {viz_dir}")
    
    def run_complete_analysis(self):
        """Execute complete Variable K-means sentence-level sentiment analysis with XLM-RoBERTa."""
        print("=" * 80)
        print("XLM-ROBERTA VARIABLE K-MEANS SENTENCE-LEVEL SENTIMENT ANALYSIS")
        print("Educational Comments - YouTube Mathematics Content")
        print("Model: AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual")
        print("=" * 80)
        
        try:
            # Create output directory first
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Process dataset
            enhanced_df, sentence_df = self.process_dataset()
            
            # Save Variable K-means results
            enhanced_df.to_csv(f"{self.output_dir}/xlm_roberta_variable_k_comments_with_sentence_sentiment.csv", index=False)
            sentence_df.to_csv(f"{self.output_dir}/xlm_roberta_variable_k_sentence_level_details.csv", index=False)
            
            # Create analysis report
            self.create_analysis_report(enhanced_df, sentence_df)
            
            # Create visualizations
            self.create_visualizations(enhanced_df, sentence_df)
            
            print("\n" + "=" * 80)
            print("XLM-ROBERTA VARIABLE K-MEANS ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Results saved in: {self.output_dir}/")
            print("Key files:")
            print("- xlm_roberta_variable_k_comments_with_sentence_sentiment.csv (enhanced dataset)")
            print("- xlm_roberta_variable_k_sentence_level_details.csv (sentence-by-sentence breakdown)")
            print("- xlm_roberta_variable_k_sentence_analysis_report.txt (comprehensive analysis)")
            print("- visualizations/ (XLM-RoBERTa Variable K-means charts and graphs)")
            print("\n" + "=" * 80)
            print("COMPARISON NOTE:")
            print("This analysis uses the SAME aggregation method as Twitter-RoBERTa")
            print("but with XLM-RoBERTa YouTube model to isolate model training impact.")
            print("Expected improvement: ~35 percentage points in human agreement")
            print("=" * 80)
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function for XLM-RoBERTa Variable K-means sentence-level analysis."""
    analyzer = VariableKMeansSentenceLevelAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()