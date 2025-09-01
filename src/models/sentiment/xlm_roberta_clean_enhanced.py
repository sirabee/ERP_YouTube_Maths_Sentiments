R#!/usr/bin/env python3
"""
Clean Enhanced XLM-RoBERTa Sentiment Analysis
Simplified, focused implementation with contradiction-aware learning journey detection.

Key Features:
- SpaCy segmentation with regex fallback
- Contradiction marker detection for learning journeys
- Simple, maintainable aggregation logic
- Comprehensive visualizations

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
import spacy

warnings.filterwarnings('ignore')

class CleanXLMRoBERTaAnalyzer:
    def __init__(self):
        """Initialize clean XLM-RoBERTa analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/models/sentiment_analysis/xlm_roberta_clean_{self.timestamp}"
        
        # Setup device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("Clean Enhanced XLM-RoBERTa Analysis")
        
        # Load XLM-RoBERTa model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual",
            device=-1,
            truncation=True,
            max_length=512
        )
        
        # Load SpaCy if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("SpaCy loaded successfully")
        except OSError:
            print("SpaCy not available, using regex fallback")
            self.nlp = None
        
        # Contradiction markers for learning journey detection
        self.contradiction_markers = {
            'but', 'however', 'although', 'though', 'yet', 'nevertheless',
            'nonetheless', 'still', 'despite', 'whereas', 'while', 'now',
            'actually', 'instead', 'even though'
        }
    
    def segment_sentences(self, text):
        """Segment text into sentences using SpaCy or regex fallback."""
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 5]
        else:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) >= 5]
        
        return sentences
    
    def standardize_sentiment(self, label):
        """Convert model output to standard sentiment."""
        label = str(label).lower()
        if label in ['positive', 'pos', 'label_2', '2']:
            return 'positive'
        elif label in ['negative', 'neg', 'label_0', '0']:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentence(self, sentence):
        """Analyze sentiment of a single sentence."""
        try:
            result = self.sentiment_pipeline(sentence)[0]
            return {
                'text': sentence,
                'sentiment': self.standardize_sentiment(result['label']),
                'confidence': result['score']
            }
        except Exception:
            return {
                'text': sentence,
                'sentiment': 'neutral',
                'confidence': 0.5
            }
    
    def detect_progression_type(self, sentiments):
        """Detect progression type based on sentiment sequence."""
        if len(sentiments) == 1:
            return 'single_sentence'
        
        # Count transitions
        transitions = 0
        for i in range(len(sentiments) - 1):
            if sentiments[i] != sentiments[i + 1]:
                transitions += 1
        
        if transitions == 0:
            return 'stable'
        elif transitions == 1:
            return 'simple_transition'
        else:
            # Check for learning journey pattern
            if self._is_learning_journey(sentiments):
                return 'learning_journey'
            else:
                return 'complex_progression'
    
    def _is_learning_journey(self, sentiments):
        """Simple learning journey detection: negative to positive trend."""
        if len(sentiments) < 2:
            return False
        
        # Check if there's a negative to positive transition
        for i in range(len(sentiments) - 1):
            if sentiments[i] == 'negative' and sentiments[i + 1] == 'positive':
                return True
        
        # Check overall pattern: more negative at start, more positive at end
        if len(sentiments) >= 3:
            first_half = sentiments[:len(sentiments)//2]
            second_half = sentiments[len(sentiments)//2:]
            
            neg_start = first_half.count('negative')
            pos_end = second_half.count('positive')
            
            return neg_start > 0 and pos_end > 0
        
        return False
    
    def has_contradiction_pattern(self, comment_text, sentence_results):
        """Check for contradiction-based learning patterns."""
        text_lower = comment_text.lower()
        
        # Simple check: does comment contain contradiction marker and positive sentiment?
        has_contradiction = any(marker in text_lower for marker in self.contradiction_markers)
        has_positive = any(s['sentiment'] == 'positive' for s in sentence_results)
        
        return has_contradiction and has_positive
    
    def aggregate_sentiment(self, comment_text, comment_id):
        """Main aggregation function - clean and simple."""
        sentences = self.segment_sentences(comment_text)
        
        if not sentences:
            return self._empty_result(comment_id)
        
        # Analyze each sentence
        sentence_results = [self.analyze_sentence(s) for s in sentences]
        sentiments = [r['sentiment'] for r in sentence_results]
        confidences = [r['confidence'] for r in sentence_results]
        
        # Detect progression type
        progression_type = self.detect_progression_type(sentiments)
        
        # Enhanced learning journey detection
        is_learning_journey = (
            progression_type == 'learning_journey' or 
            self.has_contradiction_pattern(comment_text, sentence_results)
        )
        
        # Final sentiment decision (maintain 73% accuracy approach)
        if is_learning_journey:
            final_sentiment = 'positive'
            final_confidence = 0.7
        else:
            # Simple majority voting
            sentiment_counts = {s: sentiments.count(s) for s in ['positive', 'negative', 'neutral']}
            final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            final_confidence = np.mean(confidences)
        
        return {
            'comment_id': comment_id,
            'final_sentiment': final_sentiment,
            'confidence': final_confidence,
            'sentence_count': len(sentences),
            'progression_type': progression_type,
            'learning_journey': is_learning_journey,
            'sentence_sentiments': sentiments,
            'sentence_details': sentence_results
        }
    
    def _empty_result(self, comment_id):
        """Return result for empty comments."""
        return {
            'comment_id': comment_id,
            'final_sentiment': 'neutral',
            'confidence': 0.5,
            'sentence_count': 0,
            'progression_type': 'no_sentences',
            'learning_journey': False,
            'sentence_sentiments': [],
            'sentence_details': []
        }
    
    def load_dataset(self):
        """Load the complete dataset."""
        dataset_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/models/bertopic_outputs/Optimised_Variable_K/merged_topic_info/optimised_variable_k_phase_4_20250722_224755/results/all_comments_with_topics.csv"
        
        print(f"Loading dataset...")
        df = pd.read_csv(dataset_path)
        
        # Clean data
        df['comment_text'] = df['comment_text'].astype(str).fillna('')
        df = df[df['comment_text'].str.len() > 0].reset_index(drop=True)
        df['analysis_id'] = range(len(df))
        
        print(f"Loaded {len(df):,} comments")
        return df
    
    def process_dataset(self, df):
        """Process all comments in the dataset."""
        print(f"Processing {len(df):,} comments...")
        
        results = []
        batch_size = 1000
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing"):
            batch = df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                try:
                    result = self.aggregate_sentiment(row['comment_text'], row['analysis_id'])
                    results.append(result)
                except Exception as e:
                    print(f"Error processing comment {row['analysis_id']}: {e}")
                    results.append(self._empty_result(row['analysis_id']))
            
            # Memory cleanup
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        return results
    
    def create_enhanced_dataframe(self, df, results):
        """Create enhanced dataframe with results."""
        # Convert results to dataframe
        results_df = pd.DataFrame([
            {
                'analysis_id': r['comment_id'],
                'xlm_sentiment': r['final_sentiment'],
                'xlm_confidence': r['confidence'],
                'sentence_count': r['sentence_count'],
                'progression_type': r['progression_type'],
                'learning_journey': r['learning_journey']
            }
            for r in results
        ])
        
        # Merge with original data
        enhanced_df = df.merge(results_df, on='analysis_id', how='left')
        
        # Create sentence-level dataframe
        sentence_data = []
        for result in results:
            for i, sentence_detail in enumerate(result['sentence_details']):
                sentence_data.append({
                    'comment_id': result['comment_id'],
                    'sentence_id': i,
                    'sentence_text': sentence_detail['text'],
                    'sentence_sentiment': sentence_detail['sentiment'],
                    'sentence_confidence': sentence_detail['confidence']
                })
        
        sentence_df = pd.DataFrame(sentence_data)
        return enhanced_df, sentence_df
    
    def create_visualizations(self, enhanced_df):
        """Create clean, focused visualizations."""
        viz_dir = Path(self.output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set clean style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 10})
        
        # 1. Overview visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Enhanced XLM-RoBERTa: Sentiment Analysis Overview', fontsize=14, fontweight='bold')
        
        # Progression patterns
        progression_counts = enhanced_df['progression_type'].value_counts()
        colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c', '#9b59b6']
        
        ax1.pie(progression_counts.values, labels=progression_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Progression Patterns')
        
        # Sentiment distribution
        sentiment_counts = enhanced_df['xlm_sentiment'].value_counts()
        sentiment_order = ['positive', 'neutral', 'negative']
        bar_data = [sentiment_counts.get(s, 0) for s in sentiment_order]
        bar_colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        
        bars = ax2.bar(sentiment_order, bar_data, color=bar_colors)
        ax2.set_ylabel('Number of Comments')
        ax2.set_title('Final Sentiment Distribution')
        
        # Add value labels
        for bar, value in zip(bars, bar_data):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Query-level analysis (if available)
        if 'search_query' in enhanced_df.columns:
            self._create_query_visualization(enhanced_df, viz_dir)
        
        # 3. Learning journey analysis
        self._create_journey_analysis(enhanced_df, viz_dir)
        
        print(f"Visualizations saved to: {viz_dir}")
        return viz_dir
    
    def _create_query_visualization(self, enhanced_df, viz_dir):
        """Create query-level sentiment visualization."""
        fig, ax = plt.subplots(figsize=(18, 8))
        
        # Top 15 queries
        top_queries = enhanced_df['search_query'].value_counts().head(15).index
        
        # Calculate percentages for each query
        query_data = []
        for query in top_queries:
            query_df = enhanced_df[enhanced_df['search_query'] == query]
            sentiment_pcts = query_df['xlm_sentiment'].value_counts(normalize=True) * 100
            
            query_data.append({
                'query': query,
                'positive': sentiment_pcts.get('positive', 0),
                'neutral': sentiment_pcts.get('neutral', 0),
                'negative': sentiment_pcts.get('negative', 0)
            })
        
        query_df = pd.DataFrame(query_data)
        
        # Stacked bar chart
        x = np.arange(len(query_df))
        
        ax.bar(x, query_df['negative'], label='negative', color='#e74c3c')
        ax.bar(x, query_df['neutral'], bottom=query_df['negative'], 
               label='neutral', color='#e67e22')
        ax.bar(x, query_df['positive'], 
               bottom=query_df['negative'] + query_df['neutral'],
               label='positive', color='#2ecc71')
        
        ax.set_xticks(x)
        ax.set_xticklabels(query_df['query'], rotation=45, ha='right')
        ax.set_ylabel('Percentage')
        ax.set_title('Sentiment Distribution by Search Query (Top 15)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "query_sentiment.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_journey_analysis(self, enhanced_df, viz_dir):
        """Create learning journey specific analysis."""
        journey_comments = enhanced_df[enhanced_df['learning_journey']]
        
        if len(journey_comments) == 0:
            print("No learning journeys detected for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Learning Journey Analysis', fontsize=14, fontweight='bold')
        
        # Journey sentiment distribution
        journey_sentiment = journey_comments['xlm_sentiment'].value_counts()
        axes[0, 0].pie(journey_sentiment.values, labels=journey_sentiment.index, 
                       autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#95a5a6'])
        axes[0, 0].set_title('Journey Sentiment Distribution')
        
        # Journey confidence
        axes[0, 1].hist(journey_comments['xlm_confidence'], bins=15, 
                        color='orange', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Journey Confidence Distribution')
        
        # Progression types for journeys
        journey_progression = journey_comments['progression_type'].value_counts()
        axes[1, 0].bar(range(len(journey_progression)), journey_progression.values)
        axes[1, 0].set_xticks(range(len(journey_progression)))
        axes[1, 0].set_xticklabels(journey_progression.index, rotation=45)
        axes[1, 0].set_title('Journey Progression Types')
        
        # Summary stats
        axes[1, 1].axis('off')
        stats_text = f"""Learning Journey Summary:

Total Journeys: {len(journey_comments):,}
Percentage: {len(journey_comments)/len(enhanced_df)*100:.1f}%

Average Confidence: {journey_comments['xlm_confidence'].mean():.3f}
Average Sentences: {journey_comments['sentence_count'].mean():.1f}

Most Common Type:
{journey_progression.index[0] if len(journey_progression) > 0 else 'None'}"""
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "learning_journey.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, enhanced_df, sentence_df):
        """Generate clean, focused analysis report."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        total_comments = len(enhanced_df)
        learning_journeys = enhanced_df['learning_journey'].sum()
        progression_dist = enhanced_df['progression_type'].value_counts()
        sentiment_dist = enhanced_df['xlm_sentiment'].value_counts()
        
        report = f"""
=== CLEAN ENHANCED XLM-RoBERTa SENTIMENT ANALYSIS REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: XLM-RoBERTa (AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual)

=== SUMMARY ===
Total comments: {total_comments:,}
Total sentences: {len(sentence_df):,}
Learning journeys: {learning_journeys:,} ({learning_journeys/total_comments*100:.1f}%)
Average sentences/comment: {enhanced_df['sentence_count'].mean():.2f}

=== SENTIMENT DISTRIBUTION ===
"""
        for sentiment, count in sentiment_dist.items():
            percentage = count / total_comments * 100
            report += f"{sentiment.upper()}: {count:,} ({percentage:.1f}%)\n"
        
        report += f"""
=== PROGRESSION TYPES ===
"""
        for prog_type, count in progression_dist.items():
            percentage = count / total_comments * 100
            report += f"{prog_type}: {count:,} ({percentage:.1f}%)\n"
        
        report += f"""
=== METHODOLOGY ===
1. SpaCy sentence segmentation (regex fallback)
2. XLM-RoBERTa sentiment classification
3. Contradiction-aware learning journey detection
4. Simple majority voting aggregation
5. 0.7 confidence for learning journeys (validated approach)

=== ENHANCEMENTS ===
- Contradiction marker detection for learning patterns
- Five-type progression classification
- Clean, maintainable code structure
- Comprehensive visualizations
"""
        
        report_path = Path(self.output_dir) / "analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report_path
    
    def save_results(self, enhanced_df, sentence_df):
        """Save analysis results."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        enhanced_file = output_dir / "enhanced_comments.csv"
        sentence_file = output_dir / "sentence_details.csv"
        
        enhanced_df.to_csv(enhanced_file, index=False)
        sentence_df.to_csv(sentence_file, index=False)
        
        return enhanced_file, sentence_file
    
    def run_analysis(self):
        """Execute complete analysis pipeline."""
        print("=" * 60)
        print("CLEAN ENHANCED XLM-RoBERTa SENTIMENT ANALYSIS")
        print("=" * 60)
        
        try:
            # Load and process data
            df = self.load_dataset()
            results = self.process_dataset(df)
            enhanced_df, sentence_df = self.create_enhanced_dataframe(df, results)
            
            # Generate outputs
            report_path = self.generate_report(enhanced_df, sentence_df)
            viz_dir = self.create_visualizations(enhanced_df)
            enhanced_file, sentence_file = self.save_results(enhanced_df, sentence_df)
            
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Results: {self.output_dir}")
            print(f"Report: {report_path.name}")
            print(f"Visualizations: {viz_dir.name}")
            
            # Summary
            total = len(enhanced_df)
            journeys = enhanced_df['learning_journey'].sum()
            print(f"\nSUMMARY:")
            print(f"Comments analyzed: {total:,}")
            print(f"Learning journeys: {journeys:,} ({journeys/total*100:.1f}%)")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    analyzer = CleanXLMRoBERTaAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()