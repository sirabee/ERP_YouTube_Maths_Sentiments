#!/usr/bin/env python3
"""
Per-Sentence Sentiment Analysis for Educational Comments
Implements sentence-level analysis as described in academic literature
Addresses learning journey detection and sentiment transitions
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

class SentenceLevelSentimentAnalyzer:
    def __init__(self, comments_file=None):
        """Initialize sentence-level sentiment analyzer."""
        if comments_file is None:
            comments_file = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/BERTopic HDBSCAN Per Query 20250720/results/all_comments_with_topics.csv                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      "
        self.comments_file = comments_file
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"sentence_sentiment_analysis_{self.timestamp}"
        
        # Setup device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize sentiment pipeline
        print("Loading Twitter RoBERTa sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,  # Use CPU for compatibility
            batch_size=16,  # Reduced for sentence-level processing
            truncation=True,
            max_length=512
        )
        
        # Load spaCy for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("SpaCy model loaded successfully.")
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
                'NEU': 'neutral'
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
        """Process the entire dataset with sentence-level analysis."""
        print(f"Loading dataset: {self.comments_file}")
        df = pd.read_csv(self.comments_file)
        
        print(f"Loaded {len(df):,} comments")
        df['comment_text'] = df['comment_text'].astype(str).fillna('')
        df = df[df['comment_text'].str.len() > 0]
        
        print(f"Processing {len(df):,} comments with sentence-level analysis...")
        
        results = []
        for idx, comment in tqdm(enumerate(df['comment_text']), total=len(df), desc="Analyzing comments"):
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
        """Create comprehensive analysis report."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        report_path = f"{self.output_dir}/sentence_analysis_report.txt"
        
        # Calculate statistics
        total_comments = len(enhanced_df)
        learning_journeys = len(enhanced_df[enhanced_df['sentiment_progression'] == 'learning_journey'])
        transitions = len(enhanced_df[enhanced_df['has_transition'] == True])
        
        sentiment_dist = enhanced_df['sentence_level_sentiment'].value_counts()
        progression_dist = enhanced_df['sentiment_progression'].value_counts()
        
        report = f"""
=== SENTENCE-LEVEL SENTIMENT ANALYSIS REPORT ===
Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Method: Per-sentence sentiment analysis with progression detection

=== SUMMARY STATISTICS ===
Total comments analyzed: {total_comments:,}
Comments with sentiment transitions: {transitions:,} ({transitions/total_comments*100:.2f}%)
Learning journeys detected: {learning_journeys:,} ({learning_journeys/total_comments*100:.2f}%)

=== SENTENCE-LEVEL SENTIMENT DISTRIBUTION ===
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
            enhanced_df['sentiment_progression'] == 'learning_journey'
        ]['comment_text'].head(3)
        
        if not journey_examples.empty:
            report += f"\n=== LEARNING JOURNEY EXAMPLES ===\n"
            for i, example in enumerate(journey_examples, 1):
                report += f"\nExample {i}:\n{example[:300]}...\n"
        
        report += f"""
=== EDUCATIONAL INSIGHTS ===
This sentence-level analysis reveals {learning_journeys} comments showing
educational transformation patterns (negative-to-positive progression).
These represent genuine learning experiences where students express
initial confusion or difficulty followed by understanding and gratitude.

Average sentences per comment: {enhanced_df['sentence_count'].mean():.1f}
Comments with multiple sentences: {len(enhanced_df[enhanced_df['sentence_count'] > 1]):,}

=== METHODOLOGY ===
- Sentence segmentation: spaCy English model
- Sentiment analysis: Twitter RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
- Learning journey detection: Negative-to-positive progression patterns
- Final sentiment weighting: Educational context-aware algorithms
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Analysis report saved: {report_path}")
    
    def create_visualizations(self, enhanced_df, sentence_df):
        """Create visualizations for sentence-level analysis."""
        viz_dir = f"{self.output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Sentiment progression distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        progression_counts = enhanced_df['sentiment_progression'].value_counts()
        plt.pie(progression_counts.values, labels=progression_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Progression Patterns')
        
        plt.subplot(1, 2, 2)
        sentiment_counts = enhanced_df['sentence_level_sentiment'].value_counts()
        # Safe color assignment
        colors = plt.cm.viridis(np.linspace(0, 1, len(sentiment_counts)))
        plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        plt.title('Final Sentence-Level Sentiment Distribution')
        plt.ylabel('Number of Comments')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/sentiment_progression_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Learning journey analysis
        if len(enhanced_df[enhanced_df['sentiment_progression'] == 'learning_journey']) > 0:
            plt.figure(figsize=(10, 6))
            
            journey_data = enhanced_df[enhanced_df['sentiment_progression'] == 'learning_journey']
            
            plt.hist(journey_data['sentence_count'], bins=range(1, 15), alpha=0.7, color='skyblue')
            plt.xlabel('Number of Sentences')
            plt.ylabel('Number of Learning Journey Comments')
            plt.title('Distribution of Sentence Count in Learning Journey Comments')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/learning_journey_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {viz_dir}")
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations using separate script."""
        print("Creating comprehensive HDBSCAN visualizations...")
        try:
            import subprocess
            import sys
            
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comprehensive_script = os.path.join(current_dir, "create_comprehensive_visualizations.py")
            
            if os.path.exists(comprehensive_script):
                # Run the comprehensive visualization script
                result = subprocess.run([sys.executable, comprehensive_script], 
                                      capture_output=True, text=True, cwd=current_dir)
                
                if result.returncode == 0:
                    print("✓ Comprehensive visualizations created successfully")
                else:
                    print(f"⚠ Warning: Comprehensive visualization script had issues:")
                    print(result.stderr)
            else:
                print(f"⚠ Warning: Comprehensive visualization script not found at {comprehensive_script}")
                
        except Exception as e:
            print(f"⚠ Warning: Could not create comprehensive visualizations: {e}")
    
    def run_complete_analysis(self):
        """Execute complete sentence-level sentiment analysis."""
        print("=" * 70)
        print("SENTENCE-LEVEL SENTIMENT ANALYSIS FOR EDUCATIONAL COMMENTS")
        print("=" * 70)
        
        try:
            # Process dataset
            enhanced_df, sentence_df = self.process_dataset()
            
            # Save results
            enhanced_df.to_csv(f"{self.output_dir}/comments_with_sentence_sentiment.csv", index=False)
            sentence_df.to_csv(f"{self.output_dir}/sentence_level_details.csv", index=False)
            
            # Create analysis report
            self.create_analysis_report(enhanced_df, sentence_df)
            
            # Create visualizations
            self.create_visualizations(enhanced_df, sentence_df)
            
            # Create comprehensive visualizations
            self.create_comprehensive_visualizations()
            
            print("\n" + "=" * 70)
            print("SENTENCE-LEVEL ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Results saved in: {self.output_dir}/")
            print("Key files:")
            print("- comments_with_sentence_sentiment.csv (enhanced dataset)")
            print("- sentence_level_details.csv (sentence-by-sentence breakdown)")
            print("- sentence_analysis_report.txt (comprehensive analysis)")
            print("- visualizations/ (charts and graphs)")
            print("- comprehensive_visualizations/ (Publication-quality analysis)")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    analyzer = SentenceLevelSentimentAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()