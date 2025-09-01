#!/usr/bin/env python3
"""
Keyword Frequency Analyzer for BERTopic Variable K Results
Calculates actual keyword frequencies within topic documents for data transparency
MSc Data Science Thesis - Perceptions of Maths on YouTube
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re
from collections import Counter
from tqdm import tqdm

class KeywordFrequencyAnalyzer:
    def __init__(self):
        """Initialize keyword frequency analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "analysis" / "keyword_frequencies"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Keyword Frequency Analyzer Initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_topic_documents(self):
        """Load the complete dataset with topic assignments."""
        # Load the all_comments_with_topics file
        comments_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "merged_topic_info" / "optimised_variable_k_phase_4_20250722_224755" / "results" / "all_comments_with_topics.csv"
        
        if not comments_file.exists():
            # Try alternative path
            comments_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "all_comments_with_topics.csv"
        
        print(f"Loading comments with topic assignments...")
        df = pd.read_csv(comments_file)
        print(f"Loaded {len(df):,} comments")
        
        return df
    
    def calculate_keyword_frequencies(self, topic_documents, keywords_str):
        """Calculate frequency of each keyword within the topic's documents."""
        if pd.isna(keywords_str):
            return {}
        
        # Parse keywords from string representation
        keywords = keywords_str.replace('[', '').replace(']', '').replace("'", '').split(',')
        keywords = [k.strip().lower() for k in keywords]
        
        # Combine all documents into one text
        combined_text = ' '.join(topic_documents).lower()
        
        # Count each keyword's frequency
        keyword_freqs = {}
        for keyword in keywords:
            # Handle multi-word keywords
            if ' ' in keyword:
                # For phrases, count exact matches
                count = combined_text.count(keyword)
            else:
                # For single words, use word boundaries
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, combined_text)
                count = len(matches)
            
            keyword_freqs[keyword] = count
        
        return keyword_freqs
    
    def process_topics(self):
        """Process all topics and calculate keyword frequencies."""
        # Load topic info
        topic_info_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "merged_topic_info" / "merged_topic_info_sorted.csv"
        print(f"Loading topic info from: {topic_info_file}")
        topic_info_df = pd.read_csv(topic_info_file, nrows=500)  # Process first 500 topics
        
        # Load comments with topics
        comments_df = self.load_topic_documents()
        
        print(f"\nProcessing {len(topic_info_df)} topics...")
        
        enhanced_topics = []
        
        for idx, row in tqdm(topic_info_df.iterrows(), total=len(topic_info_df), desc="Analyzing keyword frequencies"):
            topic_num = row['Topic']
            query = row['query'] if 'query' in row else ''
            keywords = row['Representation'] if 'Representation' in row else ''
            doc_count = row['Count'] if 'Count' in row else 0
            
            # Get documents for this topic and query
            topic_docs = comments_df[
                (comments_df['topic'] == topic_num) & 
                (comments_df['search_query'] == query)
            ]['comment_text'].tolist() if 'search_query' in comments_df.columns else []
            
            # If no documents found, try without query filter
            if len(topic_docs) == 0:
                topic_docs = comments_df[comments_df['topic'] == topic_num]['comment_text'].tolist()
            
            # Calculate keyword frequencies
            keyword_freqs = self.calculate_keyword_frequencies(topic_docs, keywords)
            
            # Create weighted representation
            weighted_keywords = []
            total_keyword_occurrences = sum(keyword_freqs.values()) if keyword_freqs else 1
            
            for keyword, freq in keyword_freqs.items():
                if freq > 0:
                    # Calculate percentage of total keyword occurrences
                    percentage = (freq / total_keyword_occurrences) * 100 if total_keyword_occurrences > 0 else 0
                    weighted_keywords.append(f"{keyword} ({freq})")
            
            # Sort by frequency
            sorted_keywords = sorted(keyword_freqs.items(), key=lambda x: x[1], reverse=True)
            weighted_repr = ', '.join([f"{k} ({f})" for k, f in sorted_keywords[:10]])  # Top 10 keywords
            
            # Calculate keyword diversity metrics
            unique_keywords = len(keyword_freqs)
            total_occurrences = sum(keyword_freqs.values())
            avg_freq = total_occurrences / unique_keywords if unique_keywords > 0 else 0
            
            # Identify dominant keywords (those with >10% of total occurrences)
            dominant_keywords = [k for k, f in keyword_freqs.items() 
                               if (f / total_occurrences * 100) > 10] if total_occurrences > 0 else []
            
            enhanced_topics.append({
                'topic_number': topic_num,
                'query': query,
                'document_count': doc_count,
                'actual_doc_count': len(topic_docs),
                'original_keywords': keywords,
                'weighted_keywords': weighted_repr,
                'unique_keywords': unique_keywords,
                'total_keyword_occurrences': total_occurrences,
                'avg_keyword_frequency': round(avg_freq, 2),
                'dominant_keywords': ', '.join(dominant_keywords),
                'keyword_frequencies': json.dumps(keyword_freqs)  # Store full frequency dict as JSON
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(enhanced_topics)
        
        # Save detailed results
        output_file = self.output_dir / f"keyword_frequencies_detailed_{self.timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved detailed frequencies to: {output_file}")
        
        return results_df
    
    def create_readable_summary(self, results_df):
        """Create a more readable summary of keyword frequencies."""
        print("\nGenerating readable summary...")
        
        # Create simplified version for presentation
        summary_data = []
        
        for _, row in results_df.iterrows():
            # Parse keyword frequencies
            freqs = json.loads(row['keyword_frequencies'])
            
            # Get top 5 keywords with percentages
            total = row['total_keyword_occurrences']
            if total > 0:
                top_keywords = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:5]
                keyword_display = []
                for keyword, freq in top_keywords:
                    pct = (freq / total) * 100
                    keyword_display.append(f"{keyword} ({freq}, {pct:.1f}%)")
                
                summary_data.append({
                    'Query': row['query'],
                    'Topic': row['topic_number'],
                    'Documents': row['document_count'],
                    'Top Keywords (freq, %)': ' | '.join(keyword_display)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = self.output_dir / f"keyword_frequency_summary_{self.timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to: {summary_file}")
        
        return summary_df
    
    def create_analysis_report(self, results_df):
        """Create analysis report with insights."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("KEYWORD FREQUENCY ANALYSIS - BERTOPIC VARIABLE K")
        report_lines.append("=" * 80)
        report_lines.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Topics Analyzed: {len(results_df)}")
        report_lines.append(f"Total Documents: {results_df['document_count'].sum():,}")
        
        # Overall statistics
        report_lines.append("\n" + "=" * 40)
        report_lines.append("KEYWORD STATISTICS")
        report_lines.append("=" * 40)
        report_lines.append(f"Average keywords per topic: {results_df['unique_keywords'].mean():.1f}")
        report_lines.append(f"Average keyword frequency: {results_df['avg_keyword_frequency'].mean():.1f}")
        report_lines.append(f"Total keyword occurrences: {results_df['total_keyword_occurrences'].sum():,}")
        
        # Find most frequent keywords across all topics
        all_freqs = Counter()
        for _, row in results_df.iterrows():
            freqs = json.loads(row['keyword_frequencies'])
            all_freqs.update(freqs)
        
        report_lines.append("\n" + "=" * 40)
        report_lines.append("TOP 20 MOST FREQUENT KEYWORDS (CORPUS-WIDE)")
        report_lines.append("=" * 40)
        for keyword, freq in all_freqs.most_common(20):
            report_lines.append(f"{keyword}: {freq:,} occurrences")
        
        # Topics with highest keyword concentration
        results_df['keyword_concentration'] = results_df['total_keyword_occurrences'] / results_df['document_count']
        top_concentrated = results_df.nlargest(10, 'keyword_concentration')
        
        report_lines.append("\n" + "=" * 40)
        report_lines.append("TOPICS WITH HIGHEST KEYWORD CONCENTRATION")
        report_lines.append("=" * 40)
        for _, row in top_concentrated.iterrows():
            report_lines.append(f"{row['query']} Topic {row['topic_number']}: "
                              f"{row['keyword_concentration']:.1f} keywords/document")
            report_lines.append(f"  Dominant: {row['dominant_keywords'][:50]}...")
        
        # Save report
        report_file = self.output_dir / f"frequency_analysis_report_{self.timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved report to: {report_file}")
        
        return report_file
    
    def create_visualization_data(self, results_df):
        """Create data formatted for visualization."""
        print("\nPreparing visualization data...")
        
        # Create topic-keyword matrix for heatmap
        viz_data = []
        
        for _, row in results_df.head(50).iterrows():  # Top 50 topics for visibility
            freqs = json.loads(row['keyword_frequencies'])
            sorted_keywords = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for keyword, freq in sorted_keywords:
                viz_data.append({
                    'topic': f"{row['query'][:20]}_T{row['topic_number']}",
                    'keyword': keyword,
                    'frequency': freq,
                    'documents': row['document_count']
                })
        
        viz_df = pd.DataFrame(viz_data)
        
        # Save visualization data
        viz_file = self.output_dir / f"visualization_data_{self.timestamp}.csv"
        viz_df.to_csv(viz_file, index=False)
        print(f"Saved visualization data to: {viz_file}")
        
        return viz_df
    
    def run_analysis(self):
        """Execute complete keyword frequency analysis."""
        print("=" * 60)
        print("KEYWORD FREQUENCY ANALYSIS")
        print("=" * 60)
        
        # Process topics and calculate frequencies
        results_df = self.process_topics()
        
        # Create readable summary
        summary_df = self.create_readable_summary(results_df)
        
        # Create analysis report
        report_file = self.create_analysis_report(results_df)
        
        # Create visualization data
        viz_df = self.create_visualization_data(results_df)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"\nKey Statistics:")
        print(f"  • Analyzed {len(results_df)} topics")
        print(f"  • Processed {results_df['document_count'].sum():,} documents")
        print(f"  • Calculated frequencies for {results_df['unique_keywords'].sum()} keyword instances")
        
        print(f"\nOutput files in: {self.output_dir}")
        print("  • keyword_frequencies_detailed_*.csv - Full frequency data")
        print("  • keyword_frequency_summary_*.csv - Readable summary")
        print("  • frequency_analysis_report_*.txt - Analysis insights")
        print("  • visualization_data_*.csv - Data for charts")
        
        # Display sample output
        print("\nSAMPLE OUTPUT (First 3 topics):")
        print("-" * 60)
        for _, row in summary_df.head(3).iterrows():
            print(f"\n{row['Query']} - Topic {row['Topic']} ({row['Documents']} docs)")
            print(f"Keywords: {row['Top Keywords (freq, %)']}")
        
        return results_df, summary_df

def main():
    """Main execution function."""
    analyzer = KeywordFrequencyAnalyzer()
    results_df, summary_df = analyzer.run_analysis()

if __name__ == "__main__":
    main()