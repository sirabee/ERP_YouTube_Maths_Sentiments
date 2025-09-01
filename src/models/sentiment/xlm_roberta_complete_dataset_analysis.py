#!/usr/bin/env python3
"""
Complete Dataset Analysis with XLM-RoBERTa using Original Sentence-Level Method
Applies the exact same methodology that achieved 73% agreement with manual annotations
to the entire dataset of educational YouTube comments.

Processes all comments including duplicates to preserve natural sentiment frequency
distribution, consistent with other sentiment analysis models in the pipeline.

"""

import pandas as pd
import numpy as np
import re
from transformers import pipeline
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gc
import torch

warnings.filterwarnings('ignore')

class XLMRoBERTaCompleteAnalyzer:
    def __init__(self):
        """Initialize XLM-RoBERTa analyzer with the exact validated methodology."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/models/sentiment_analysis/xlm_roberta_complete_analysis_{self.timestamp}"
        
        # Setup device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("XLM-RoBERTa Complete Dataset Analysis Initialization")
        print("Using EXACT methodology that achieved 73% manual annotation agreement")
        print("Processing all comments including duplicates for natural frequency distribution")
        
        # Initialize XLM-RoBERTa sentiment pipeline (best performing model)
        print("Loading XLM-RoBERTa sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual",
            device=-1,  # Use CPU for compatibility and stability
            truncation=True,
            max_length=512
        )
        print("XLM-RoBERTa model loaded successfully")
        
    def standardize_sentiment_label(self, raw_label):
        """Standardize sentiment labels exactly as in the validated method."""
        if pd.isna(raw_label):
            return 'neutral'
            
        label = str(raw_label).lower().strip()
        
        # Standard mapping used in validation
        if label in ['positive', 'pos', 'label_2', '2', 'POSITIVE']:
            return 'positive'
        elif label in ['negative', 'neg', 'label_0', '0', 'NEGATIVE']:
            return 'negative'
        elif label in ['neutral', 'neu', 'label_1', '1', 'NEUTRAL']:
            return 'neutral'
        else:
            return 'neutral'
    
    def analyze_sentence_sentiment(self, sentence):
        """Analyze sentiment of a single sentence - exact method from validation."""
        try:
            result = self.sentiment_pipeline(sentence)[0]
            normalized_label = self.standardize_sentiment_label(result['label'])
            
            return {
                'sentence': sentence,
                'sentiment': normalized_label,
                'confidence': result['score'],
                'raw_label': result['label']
            }
        except Exception as e:
            print(f"Error analyzing sentence: {e}")
            return {
                'sentence': sentence,
                'sentiment': 'neutral',
                'confidence': 0.5,
                'raw_label': 'error'
            }
    
    def original_sentence_aggregation(self, comment_text, comment_id):
        """
        Apply the EXACT original sentence-level aggregation method that achieved 73% accuracy.
        This is the validated approach from model_comparison_original_method.py
        """
        # Split into sentences using exact same method
        sentences = re.split(r'[.!?]+', comment_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return {
                'comment_id': comment_id,
                'final_sentiment': 'neutral',
                'confidence': 0.5,
                'sentences': 0,
                'learning_journey': False,
                'sentence_sentiments': [],
                'method': 'original_validated'
            }
        
        # Analyze each sentence
        sentence_results = []
        for sentence in sentences:
            result = self.analyze_sentence_sentiment(sentence)
            sentence_results.append(result)
        
        # Extract sentiments and confidences
        sentiments = [r['sentiment'] for r in sentence_results]
        confidences = [r['confidence'] for r in sentence_results]
        
        # Apply EXACT learning journey detection logic from validation
        has_negative_start = False
        has_positive_end = False
        
        if len(sentiments) > 1:
            # Check first half for negative sentiment
            first_half = sentiments[:len(sentiments)//2]
            has_negative_start = 'negative' in first_half
            
            # Check second half for positive sentiment
            second_half = sentiments[len(sentiments)//2:]
            has_positive_end = 'positive' in second_half
        
        learning_journey_detected = has_negative_start and has_positive_end
        
        # Apply EXACT aggregation logic that achieved 73% accuracy
        if learning_journey_detected:
            # Learning journey detected - favor positive (validated approach)
            final_sentiment = 'positive'
            final_confidence = 0.7
        else:
            # Simple majority voting (validated approach)
            sentiment_counts = {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            }
            final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            final_confidence = np.mean(confidences) if confidences else 0.5
        
        return {
            'comment_id': comment_id,
            'final_sentiment': final_sentiment,
            'confidence': final_confidence,
            'sentences': len(sentence_results),
            'learning_journey': learning_journey_detected,
            'sentence_sentiments': sentiments,
            'sentence_details': sentence_results,
            'method': 'original_validated'
        }
    
    def load_complete_dataset(self):
        """Load the complete dataset of 35,438 comments."""
        
        # Load the Variable K dataset (most recent complete dataset)
        dataset_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/models/bertopic_outputs/Optimised_Variable_K/merged_topic_info/optimised_variable_k_phase_4_20250722_224755/results/all_comments_with_topics.csv"
        
        print(f"Loading complete dataset from: {Path(dataset_path).name}")
        df = pd.read_csv(dataset_path)
        
        print(f"Loaded {len(df):,} comments from complete dataset")
        print(f"Dataset columns: {list(df.columns)}")
        
        # Ensure comment_text exists and is clean
        df['comment_text'] = df['comment_text'].astype(str).fillna('')
        df = df[df['comment_text'].str.len() > 0]
        
        print(f"After filtering empty comments: {len(df):,} comments")
        
        # NOTE: Keeping duplicates to maintain consistency with other sentiment models
        # and preserve natural frequency distribution of sentiment expressions
        # Each duplicate represents a genuine user expression and should be counted
        # Reset index for consistent processing
        df = df.reset_index(drop=True)
        
        # Add unique analysis IDs (use index to ensure uniqueness)
        df['analysis_comment_id'] = range(len(df))
        
        # Sample data characteristics
        print(f"\nDataset characteristics:")
        print(f"Average comment length: {df['comment_text'].str.len().mean():.1f} characters")
        print(f"Search queries: {df['search_query'].nunique() if 'search_query' in df.columns else 'N/A'}")
        print(f"Topics: {df['topic'].nunique() if 'topic' in df.columns else 'N/A'}")
        
        return df
    
    def process_complete_dataset(self, df):
        """Process the complete dataset using validated methodology."""
        
        print(f"\nProcessing {len(df):,} comments with XLM-RoBERTa...")
        print("Using exact original sentence-level method (73% validation accuracy)")
        
        results = []
        batch_size = 1000  # Process in batches for memory management
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(df))
            batch = df.iloc[i:batch_end]
            
            batch_results = []
            for idx, row in batch.iterrows():
                comment_text = row['comment_text']
                comment_id = row['analysis_comment_id']
                
                try:
                    result = self.original_sentence_aggregation(comment_text, comment_id)
                    batch_results.append(result)
                    
                except Exception as e:
                    print(f"Error processing comment {comment_id}: {e}")
                    # Add error result
                    batch_results.append({
                        'comment_id': comment_id,
                        'final_sentiment': 'neutral',
                        'confidence': 0.5,
                        'sentences': 0,
                        'learning_journey': False,
                        'sentence_sentiments': [],
                        'method': 'error'
                    })
                    continue
            
            results.extend(batch_results)
            
            # Memory cleanup every batch
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            
            # Progress update
            if (i // batch_size + 1) % 5 == 0:
                processed = len(results)
                print(f"    Processed {processed:,}/{len(df):,} comments ({processed/len(df)*100:.1f}%)")
        
        print(f"Processing complete: {len(results):,} comments analyzed")
        return results
    
    def create_enhanced_dataset(self, df, results):
        """Create enhanced dataset with sentiment analysis results."""
        
        print("Creating enhanced dataset with XLM-RoBERTa sentiment analysis...")
        
        # Create comment-level results DataFrame
        comment_results = []
        for result in results:
            comment_data = {
                'comment_id': result['comment_id'],
                'xlm_roberta_sentiment': result['final_sentiment'],
                'xlm_roberta_confidence': result['confidence'],
                'sentence_count': result['sentences'],
                'learning_journey_detected': result['learning_journey'],
                'analysis_method': result['method']
            }
            comment_results.append(comment_data)
        
        comment_df = pd.DataFrame(comment_results)
        
        # Merge with original dataset (should be 1:1 match)
        enhanced_df = df.merge(comment_df, left_on='analysis_comment_id', right_on='comment_id', how='left')
        
        # Verify merge was successful
        if len(enhanced_df) != len(df):
            print(f"WARNING: Unexpected merge result! Original: {len(df)}, After merge: {len(enhanced_df)}")
        
        # Create sentence-level details
        sentence_results = []
        for result in results:
            if 'sentence_details' in result and result['sentence_details']:
                for i, sentence_detail in enumerate(result['sentence_details']):
                    sentence_row = {
                        'comment_id': result['comment_id'],
                        'sentence_id': i,
                        'sentence_text': sentence_detail['sentence'],
                        'sentence_sentiment': sentence_detail['sentiment'],
                        'sentence_confidence': sentence_detail['confidence'],
                        'sentence_raw_label': sentence_detail['raw_label']
                    }
                    sentence_results.append(sentence_row)
        
        sentence_df = pd.DataFrame(sentence_results)
        
        return enhanced_df, sentence_df
    
    def generate_analysis_report(self, enhanced_df, sentence_df):
        """Generate comprehensive analysis report."""
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        total_comments = len(enhanced_df)
        learning_journeys = enhanced_df['learning_journey_detected'].sum()
        
        sentiment_dist = enhanced_df['xlm_roberta_sentiment'].value_counts()
        avg_confidence = enhanced_df['xlm_roberta_confidence'].mean()
        avg_sentences = enhanced_df['sentence_count'].mean()
        
        # Multi-sentence comments
        multi_sentence = enhanced_df[enhanced_df['sentence_count'] > 1]
        
        report = f"""
=== XLM-RoBERTa COMPLETE DATASET SENTIMENT ANALYSIS REPORT ===
Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: XLM-RoBERTa (AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual)
Method: Original sentence-level aggregation (73% manual annotation agreement)
Dataset: Complete YouTube Mathematical Education Comments
Processing timestamp: {self.timestamp}

=== DATASET SUMMARY ===
Total comments analyzed: {total_comments:,}
Total sentences analyzed: {len(sentence_df):,}
Learning journeys detected: {learning_journeys:,} ({learning_journeys/total_comments*100:.2f}%)
Average sentences per comment: {avg_sentences:.2f}
Comments with multiple sentences: {len(multi_sentence):,} ({len(multi_sentence)/total_comments*100:.1f}%)
Average confidence score: {avg_confidence:.3f}

=== XLM-RoBERTa SENTIMENT DISTRIBUTION ===
"""
        
        for sentiment, count in sentiment_dist.items():
            percentage = count / total_comments * 100
            report += f"{sentiment.upper()}: {count:,} ({percentage:.2f}%)\\n"
        
        # Learning journey analysis
        if learning_journeys > 0:
            journey_examples = enhanced_df[enhanced_df['learning_journey_detected']]['comment_text'].head(5)
            report += f"""
=== LEARNING JOURNEY EXAMPLES (XLM-RoBERTa) ===
Detected {learning_journeys} learning journeys using validated methodology:
"""
            for i, example in enumerate(journey_examples, 1):
                report += f"\\nExample {i}:\\n{example[:200]}...\\n"
        
        # Comparison with previous analyses
        report += f"""
=== METHODOLOGY VALIDATION ===
This analysis uses the EXACT methodology that achieved:
- 73.0% accuracy vs manual annotations
- Cohen's Kappa: 0.540
- F1-Score: 0.616
- Best performance on positive sentiment detection (95.9%)
- Superior educational content analysis (85.2% accuracy)

Key advantages of XLM-RoBERTa:
1. Multilingual training on YouTube data
2. Superior positive sentiment detection
3. Better educational context understanding
4. Validated learning journey detection

=== PROCESSING DETAILS ===
Processing method: Batch processing (1000 comments per batch)
Memory management: Automatic garbage collection
Error handling: Robust fallback to neutral sentiment
Sentence segmentation: Regex-based splitting on [.!?]+
Aggregation: Simple majority voting with learning journey detection
Duplicate handling: Preserves all comments including duplicates for natural frequency distribution

=== COMPARISON WITH PREVIOUS MODELS ===
This XLM-RoBERTa analysis represents the optimal approach based on:
- Validation testing on 200 manually annotated samples
- Comparison with Twitter-RoBERTa and YouTube-BERT
- Focus on educational sentiment analysis accuracy
- Proven learning journey detection capability

Expected improvements over previous analyses:
- Higher positive sentiment detection accuracy
- Better learning journey identification
- More accurate educational content classification
- Improved confidence calibration
"""
        
        # Write report
        report_path = Path(self.output_dir) / "xlm_roberta_complete_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Analysis report saved: {report_path}")
        return report
    
    def create_visualizations(self, enhanced_df, sentence_df):
        """Create comprehensive visualizations."""
        
        viz_dir = Path(self.output_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Sentiment distribution
        plt.figure(figsize=(15, 10))
        
        # Overall sentiment distribution
        plt.subplot(2, 3, 1)
        sentiment_counts = enhanced_df['xlm_roberta_sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('XLM-RoBERTa: Overall Sentiment Distribution')
        
        # Confidence distribution
        plt.subplot(2, 3, 2)
        plt.hist(enhanced_df['xlm_roberta_confidence'], bins=30, alpha=0.7, color='skyblue')
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Comments')
        plt.title('XLM-RoBERTa: Confidence Distribution')
        plt.axvline(enhanced_df['xlm_roberta_confidence'].mean(), color='red', linestyle='--', label=f'Mean: {enhanced_df["xlm_roberta_confidence"].mean():.3f}')
        plt.legend()
        
        # Sentence count distribution
        plt.subplot(2, 3, 3)
        sentence_counts = enhanced_df['sentence_count'].value_counts().sort_index()
        plt.bar(sentence_counts.index[:10], sentence_counts.values[:10])
        plt.xlabel('Number of Sentences')
        plt.ylabel('Number of Comments')
        plt.title('Sentence Count Distribution')
        
        # Learning journey analysis
        plt.subplot(2, 3, 4)
        journey_data = enhanced_df['learning_journey_detected'].value_counts()
        plt.pie(journey_data.values, labels=['No Journey', 'Learning Journey'], autopct='%1.1f%%')
        plt.title('Learning Journey Detection')
        
        # Sentiment by sentence count
        plt.subplot(2, 3, 5)
        multi_sentence = enhanced_df[enhanced_df['sentence_count'] > 1]
        if len(multi_sentence) > 0:
            sentiment_by_length = multi_sentence.groupby('sentence_count')['xlm_roberta_sentiment'].value_counts().unstack(fill_value=0)
            sentiment_by_length.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.xlabel('Number of Sentences')
            plt.ylabel('Number of Comments')
            plt.title('Sentiment by Comment Length')
            plt.legend(title='Sentiment')
            plt.xticks(rotation=45)
        
        # Search query analysis (if available)
        plt.subplot(2, 3, 6)
        if 'search_query' in enhanced_df.columns:
            top_queries = enhanced_df['search_query'].value_counts().head(10)
            plt.barh(range(len(top_queries)), top_queries.values)
            plt.yticks(range(len(top_queries)), top_queries.index)
            plt.xlabel('Number of Comments')
            plt.title('Top 10 Search Queries')
        else:
            plt.text(0.5, 0.5, 'Search query data\nnot available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Search Query Analysis')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "xlm_roberta_complete_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Learning journey specific analysis
        if enhanced_df['learning_journey_detected'].sum() > 0:
            plt.figure(figsize=(12, 8))
            
            journey_comments = enhanced_df[enhanced_df['learning_journey_detected']]
            
            # Journey sentiment distribution
            plt.subplot(2, 2, 1)
            journey_sentiment = journey_comments['xlm_roberta_sentiment'].value_counts()
            plt.pie(journey_sentiment.values, labels=journey_sentiment.index, autopct='%1.1f%%')
            plt.title('Learning Journey: Sentiment Distribution')
            
            # Journey confidence
            plt.subplot(2, 2, 2)
            plt.hist(journey_comments['xlm_roberta_confidence'], bins=20, alpha=0.7, color='orange')
            plt.xlabel('Confidence Score')
            plt.ylabel('Number of Learning Journeys')
            plt.title('Learning Journey: Confidence Distribution')
            
            # Journey sentence count
            plt.subplot(2, 2, 3)
            journey_sentences = journey_comments['sentence_count'].value_counts().sort_index()
            plt.bar(journey_sentences.index, journey_sentences.values)
            plt.xlabel('Number of Sentences')
            plt.ylabel('Number of Learning Journeys')
            plt.title('Learning Journey: Sentence Count')
            
            # Comparison with non-journeys
            plt.subplot(2, 2, 4)
            non_journey = enhanced_df[~enhanced_df['learning_journey_detected']]
            
            journey_avg_conf = journey_comments['xlm_roberta_confidence'].mean()
            non_journey_avg_conf = non_journey['xlm_roberta_confidence'].mean()
            
            categories = ['Learning Journey', 'Non-Journey']
            confidences = [journey_avg_conf, non_journey_avg_conf]
            
            plt.bar(categories, confidences, color=['orange', 'lightblue'])
            plt.ylabel('Average Confidence')
            plt.title('Confidence: Journey vs Non-Journey')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "learning_journey_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {viz_dir}")
    
    def save_results(self, enhanced_df, sentence_df):
        """Save analysis results."""
        
        output_dir = Path(self.output_dir)
        
        # Save enhanced dataset
        enhanced_file = output_dir / "xlm_roberta_comments_with_sentiment.csv"
        enhanced_df.to_csv(enhanced_file, index=False)
        
        # Save sentence-level details
        sentence_file = output_dir / "xlm_roberta_sentence_level_details.csv"
        sentence_df.to_csv(sentence_file, index=False)
        
        print(f"Results saved:")
        print(f"  Enhanced dataset: {enhanced_file}")
        print(f"  Sentence details: {sentence_file}")
        
        return enhanced_file, sentence_file
    
    def run_complete_analysis(self):
        """Execute complete XLM-RoBERTa analysis on full dataset."""
        
        print("=" * 80)
        print("XLM-RoBERTa COMPLETE DATASET SENTIMENT ANALYSIS")
        print("Using VALIDATED Original Sentence-Level Method (73% Accuracy)")
        print("=" * 80)
        
        try:
            # Load complete dataset
            df = self.load_complete_dataset()
            
            # Process all comments
            results = self.process_complete_dataset(df)
            
            # Create enhanced datasets
            enhanced_df, sentence_df = self.create_enhanced_dataset(df, results)
            
            # Generate analysis report
            report = self.generate_analysis_report(enhanced_df, sentence_df)
            
            # Create visualizations
            self.create_visualizations(enhanced_df, sentence_df)
            
            # Save results
            enhanced_file, sentence_file = self.save_results(enhanced_df, sentence_df)
            
            print("\\n" + "=" * 80)
            print("XLM-RoBERTa COMPLETE ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Results saved in: {self.output_dir}")
            print("Key files:")
            print("- xlm_roberta_comments_with_sentiment.csv (enhanced dataset)")
            print("- xlm_roberta_sentence_level_details.csv (sentence-by-sentence breakdown)")
            print("- xlm_roberta_complete_analysis_report.txt (comprehensive analysis)")
            print("- visualizations/ (analysis charts and graphs)")
            
            # Summary statistics
            total_comments = len(enhanced_df)
            learning_journeys = enhanced_df['learning_journey_detected'].sum()
            avg_confidence = enhanced_df['xlm_roberta_confidence'].mean()
            
            sentiment_dist = enhanced_df['xlm_roberta_sentiment'].value_counts()
            
            print(f"\\nANALYSIS SUMMARY:")
            print(f"Total comments processed: {total_comments:,}")
            print(f"Learning journeys detected: {learning_journeys:,} ({learning_journeys/total_comments*100:.2f}%)")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Sentiment distribution:")
            for sentiment, count in sentiment_dist.items():
                print(f"  {sentiment}: {count:,} ({count/total_comments*100:.1f}%)")
            
        except Exception as e:
            print(f"\\nERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    analyzer = XLMRoBERTaCompleteAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()