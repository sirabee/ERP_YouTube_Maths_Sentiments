#!/usr/bin/env python3
"""
Generate word clouds for BERTopic Variable K topics using gensim and wordcloud.
Creates individual word clouds for each topic based on BERTopic's topic words and probabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
from collections import defaultdict, Counter
from gensim import corpora, models
from gensim.models import LdaModel
import re

# Configuration
BASE_DIR = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
BERTOPIC_DIR = BASE_DIR / "results/models/bertopic_outputs/Optimised_Variable_K"
OUTPUT_DIR = BASE_DIR / "results/visualizations/topic_wordclouds"

def normalize_singular_plural(word_frequencies):
    """
    Normalize word frequencies by combining singular/plural forms.
    Returns dictionary with combined frequencies using the more frequent form as the key.
    """
    # Create mapping from normalized form to actual words
    normalized_groups = defaultdict(list)
    
    # First pass: group words by their potential base forms
    for word, freq in word_frequencies.items():
        word_lower = word.lower()
        
        # Simple rule-based approach for common patterns
        potential_singular = word_lower
        if word_lower.endswith('s') and len(word_lower) > 1:
            # Check if removing 's' gives us a reasonable singular form
            candidate = word_lower[:-1]
            # Don't modify if it would create very short words or known exceptions
            if len(candidate) >= 3 and candidate not in ['thi', 'wa', 'i', 'clas', 'glas']:
                potential_singular = candidate
        elif word_lower.endswith('es') and len(word_lower) > 2:
            # Handle -es endings (classes -> class, boxes -> box)
            candidate = word_lower[:-2]
            if len(candidate) >= 3:
                potential_singular = candidate
        elif word_lower.endswith('ies') and len(word_lower) > 3:
            # Handle -ies endings (studies -> study)
            potential_singular = word_lower[:-3] + 'y'
        
        # Group by the potential singular form
        normalized_groups[potential_singular].append((word, freq))
    
    # Second pass: for each group, combine frequencies and choose representative word
    combined_frequencies = {}
    
    for base_form, word_list in normalized_groups.items():
        if len(word_list) == 1:
            # No variants, keep original
            word, freq = word_list[0]
            combined_frequencies[word] = freq
        else:
            # Multiple variants - combine frequencies and pick most frequent form as representative
            total_freq = sum(freq for _, freq in word_list)
            most_frequent_word = max(word_list, key=lambda x: x[1])[0]
            combined_frequencies[most_frequent_word] = total_freq
            
            # Print combination info for verification
            variants = [f"{word}({freq})" for word, freq in word_list]
            print(f"  Combined: {' + '.join(variants)} → {most_frequent_word}({total_freq})")
    
    return combined_frequencies

class BERTopicWordCloudGenerator:
    def __init__(self):
        """Initialize the word cloud generator."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = OUTPUT_DIR / f"wordclouds_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Word cloud settings
        self.wordcloud_params = {
            'width': 800,
            'height': 600,
            'background_color': 'white',
            'max_words': 50,
            'colormap': 'viridis',
            'relative_scaling': 0.5,
            'min_font_size': 12
        }
        
        print(f"BERTopic Word Cloud Generator initialized")
        print(f"Output directory: {self.output_dir}")
    
    def find_bertopic_results(self):
        """Find the most recent BERTopic Variable K results."""
        # Look for merged topic info directories
        merged_dirs = list(BERTOPIC_DIR.glob("merged_topic_info/optimised_variable_k_*"))
        
        if not merged_dirs:
            print("No merged topic info found, looking for individual query results...")
            # Look for individual query directories
            query_dirs = [d for d in BERTOPIC_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if query_dirs:
                return self.load_individual_queries(query_dirs)
            else:
                raise FileNotFoundError("No BERTopic results found")
        
        # Use the most recent merged results
        latest_dir = max(merged_dirs, key=lambda x: x.stat().st_mtime)
        print(f"Using merged results from: {latest_dir}")
        
        return self.load_merged_results(latest_dir)
    
    def load_merged_results(self, results_dir):
        """Load merged BERTopic results."""
        # Check for direct merged files first
        direct_merged = BERTOPIC_DIR / "merged_topic_info" / "merged_topic_info.csv"
        
        if direct_merged.exists():
            print(f"Loading merged topic info from: {direct_merged}")
            topic_info = pd.read_csv(direct_merged)
            return topic_info, None
        
        # Look for topic info and document info in results subdirectory
        topic_info_file = results_dir / "results" / "merged_topic_info.csv"
        doc_info_file = results_dir / "results" / "all_comments_with_topics.csv"
        
        if not topic_info_file.exists():
            # Try the parent directory structure
            topic_info_file = results_dir.parent / "merged_topic_info.csv"
        
        if not topic_info_file.exists():
            raise FileNotFoundError(f"Topic info not found. Tried: {topic_info_file}")
        
        print(f"Loading topic info from: {topic_info_file}")
        topic_info = pd.read_csv(topic_info_file)
        
        if doc_info_file.exists():
            print(f"Loading document info from: {doc_info_file}")
            doc_info = pd.read_csv(doc_info_file)
        else:
            doc_info = None
        
        return topic_info, doc_info
    
    def load_individual_queries(self, query_dirs):
        """Load and merge individual query results."""
        print("Loading individual query results...")
        
        all_topic_info = []
        all_doc_info = []
        
        for query_dir in query_dirs[:5]:  # Limit to first 5 for speed
            topic_file = query_dir / "topic_info.csv"
            if topic_file.exists():
                topic_df = pd.read_csv(topic_file)
                topic_df['source_query'] = query_dir.name
                all_topic_info.append(topic_df)
        
        if not all_topic_info:
            raise FileNotFoundError("No topic_info.csv files found")
        
        merged_topics = pd.concat(all_topic_info, ignore_index=True)
        return merged_topics, None
    
    def prepare_topic_word_frequencies(self, topic_info):
        """Extract word frequencies for each topic from BERTopic results."""
        topic_words = {}
        
        print(f"Processing {len(topic_info)} topic entries...")
        
        # Group by topic to get all words for each topic
        for topic_id in topic_info['Topic'].unique():
            if topic_id == -1:  # Skip outlier topic
                continue
            
            topic_data = topic_info[topic_info['Topic'] == topic_id]
            
            # Extract words and their frequencies/probabilities
            word_freq = {}
            
            # Check if we have representation column with word probabilities
            if 'Representation' in topic_data.columns:
                # Parse representation strings like: ['word1', 'word2', ...]
                for _, row in topic_data.iterrows():
                    rep_str = row['Representation']
                    if pd.isna(rep_str) or rep_str == '[]':
                        continue
                    
                    # Try to parse the representation
                    try:
                        # Clean the string and extract words
                        words = self.extract_words_from_representation(rep_str)
                        for i, word in enumerate(words):
                            # Give higher weight to words appearing earlier (more important)
                            weight = max(10 - i, 1)  # Decreasing weight
                            word_freq[word] = word_freq.get(word, 0) + weight
                    except Exception as e:
                        print(f"Error parsing representation for topic {topic_id}: {e}")
                        continue
            
            # Check for individual word columns
            word_columns = [col for col in topic_data.columns if col.startswith('Word_')]
            prob_columns = [col for col in topic_data.columns if col.startswith('Prob_')]
            
            if word_columns and prob_columns:
                for word_col, prob_col in zip(word_columns, prob_columns):
                    words = topic_data[word_col].dropna().values
                    probs = topic_data[prob_col].dropna().values
                    
                    for word, prob in zip(words, probs):
                        if pd.notna(word) and pd.notna(prob):
                            word_freq[str(word)] = float(prob) * 100  # Scale probabilities
            
            if word_freq:
                topic_words[topic_id] = word_freq
                print(f"Topic {topic_id}: {len(word_freq)} words extracted")
        
        return topic_words
    
    def extract_words_from_representation(self, rep_str):
        """Extract words from BERTopic representation string."""
        # Handle different formats of representation strings
        rep_str = str(rep_str).strip()
        
        # Remove brackets and quotes, split by comma
        words = []
        
        # Try JSON-like parsing first
        try:
            # Remove outer brackets if present
            if rep_str.startswith('[') and rep_str.endswith(']'):
                rep_str = rep_str[1:-1]
            
            # Split by comma and clean each word
            parts = rep_str.split(',')
            for part in parts:
                # Remove quotes, whitespace, and extract the word
                word = part.strip().strip('\'"').strip()
                if word and len(word) > 1 and word.isalpha():
                    words.append(word.lower())
        
        except:
            # Fallback: simple regex to extract words
            words = re.findall(r'\b[a-zA-Z]{2,}\b', rep_str.lower())
        
        return words[:10]  # Return top 10 words
    
    def create_wordcloud_for_topic(self, topic_id, word_frequencies, topic_info=None):
        """Generate a word cloud for a specific topic."""
        
        if not word_frequencies:
            print(f"No words found for topic {topic_id}")
            return None
        
        # Get topic name/description if available
        topic_name = f"Topic {topic_id}"
        topic_size = 0
        
        if topic_info is not None:
            topic_data = topic_info[topic_info['Topic'] == topic_id]
            if not topic_data.empty:
                if 'Name' in topic_data.columns:
                    names = topic_data['Name'].dropna().unique()
                    if len(names) > 0 and str(names[0]) != 'nan':
                        topic_name = f"Topic {topic_id}: {names[0]}"
                
                if 'Count' in topic_data.columns:
                    counts = topic_data['Count'].dropna()
                    if len(counts) > 0:
                        topic_size = int(counts.iloc[0])
        
        # Create word cloud
        wordcloud = WordCloud(**self.wordcloud_params).generate_from_frequencies(word_frequencies)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Add title with topic info
        title = f"{topic_name}"
        if topic_size > 0:
            title += f" ({topic_size:,} documents)"
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Save the word cloud
        filename = f"topic_{topic_id}_wordcloud.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Generated word cloud for {topic_name}: {filepath}")
        return filepath
    
    def create_aggregate_wordcloud(self, topic_words):
        """Create an aggregate word cloud combining all topics."""
        print("Creating aggregate word cloud...")
        
        # Aggregate all word frequencies across topics
        aggregate_frequencies = {}
        word_topic_counts = {}  # Track how many topics each word appears in
        
        for topic_id, words in topic_words.items():
            print(f"  Aggregating Topic {topic_id}: {len(words)} words")
            for word, freq in words.items():
                # Add frequency to aggregate
                aggregate_frequencies[word] = aggregate_frequencies.get(word, 0) + freq
                # Track topic appearances
                word_topic_counts[word] = word_topic_counts.get(word, 0) + 1
        
        print(f"  Total unique words before normalization: {len(aggregate_frequencies)}")
        print(f"  Most frequent words before normalization: {sorted(aggregate_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]}")
        
        # Normalize singular/plural forms
        print("  Normalizing singular/plural word forms...")
        normalized_frequencies = normalize_singular_plural(aggregate_frequencies)
        
        print(f"  Total unique words after normalization: {len(normalized_frequencies)}")
        print(f"  Most frequent words after normalization: {sorted(normalized_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]}")
        
        # Create large aggregate word cloud
        plt.figure(figsize=(16, 12))
        
        wordcloud = WordCloud(
            width=1200, height=800,
            background_color='white',
            max_words=100,
            colormap='plasma',
            relative_scaling=0.6,
            min_font_size=14,
            prefer_horizontal=0.7
        ).generate_from_frequencies(normalized_frequencies)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('BERTopic Variable K: Aggregate Word Cloud\n(All Topics Combined)', 
                 fontsize=20, fontweight='bold', pad=30)
        
        # Calculate statistics for summary file (no longer displayed on image)
        total_words_normalized = sum(normalized_frequencies.values())
        unique_words_normalized = len(normalized_frequencies)
        # Note: topic appearances calculation becomes complex after normalization, so we'll use original count
        avg_topic_appearances = np.mean(list(word_topic_counts.values()))
        
        plt.tight_layout()
        
        # Save aggregate word cloud
        aggregate_path = self.output_dir / "aggregate_wordcloud.png"
        plt.savefig(aggregate_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Generated aggregate word cloud: {aggregate_path}")
        
        # Save the normalized frequencies for analysis
        freq_path = self.output_dir / "aggregate_word_frequencies.csv"
        freq_df = pd.DataFrame([
            {'word': word, 'total_frequency': freq}
            for word, freq in normalized_frequencies.items()
        ])
        freq_df = freq_df.sort_values('total_frequency', ascending=False)
        freq_df.to_csv(freq_path, index=False)
        print(f"✓ Saved aggregate frequencies: {freq_path}")
        
        return aggregate_path, normalized_frequencies, {
            'total_words': total_words_normalized,
            'unique_words': unique_words_normalized,
            'avg_topics_per_word': avg_topic_appearances,
            'topics_analyzed': len(topic_words)
        }

    def create_combined_wordcloud(self, topic_words, max_topics=9):
        """Create a combined view of multiple topics."""
        # Sort topics by total word frequency (proxy for importance)
        topic_importance = {tid: sum(words.values()) for tid, words in topic_words.items()}
        top_topics = sorted(topic_importance.items(), key=lambda x: x[1], reverse=True)[:max_topics]
        
        # Create subplots
        n_rows = 3
        n_cols = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
        fig.suptitle('BERTopic Variable K: Individual Topic Word Clouds', fontsize=20, fontweight='bold')
        
        for i, (topic_id, _) in enumerate(top_topics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Generate small word cloud
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                max_words=30,
                colormap='viridis'
            ).generate_from_frequencies(topic_words[topic_id])
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Topic {topic_id}', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Hide any unused subplots
        for i in range(len(top_topics), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save combined view
        combined_path = self.output_dir / "individual_topics_grid.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Generated individual topics grid: {combined_path}")
        return combined_path
    
    def generate_topic_summary(self, topic_words, topic_info, aggregate_stats=None):
        """Generate a summary report of topics and their key words."""
        summary_lines = [
            "BERTopic Variable K - Topic Word Cloud Summary",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total topics processed: {len(topic_words)}",
            ""
        ]
        
        # Add aggregate statistics if provided
        if aggregate_stats:
            summary_lines.extend([
                "AGGREGATE WORD CLOUD STATISTICS",
                "-" * 35,
                f"Total word occurrences: {aggregate_stats['total_words']:,}",
                f"Unique words: {aggregate_stats['unique_words']:,}",
                f"Average topics per word: {aggregate_stats['avg_topics_per_word']:.1f}",
                f"Topics analyzed: {aggregate_stats['topics_analyzed']}",
                ""
            ])
        
        # Sort topics by word count (proxy for richness)
        sorted_topics = sorted(topic_words.items(), 
                             key=lambda x: len(x[1]), reverse=True)
        
        for topic_id, words in sorted_topics:
            # Get top words
            top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:10]
            word_list = [word for word, _ in top_words]
            
            summary_lines.append(f"Topic {topic_id}:")
            summary_lines.append(f"  Words: {len(words)}")
            summary_lines.append(f"  Top words: {', '.join(word_list)}")
            summary_lines.append("")
        
        # Save summary
        summary_path = self.output_dir / "topic_summary.txt"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"✓ Generated topic summary: {summary_path}")
        return summary_path
    
    def generate_wordclouds(self, max_individual_topics=15):
        """Main function to generate all word clouds."""
        print("\n" + "="*60)
        print("BERTOPIC VARIABLE K WORD CLOUD GENERATION")
        print("="*60)
        
        try:
            # Load BERTopic results
            topic_info, doc_info = self.find_bertopic_results()
            
            # Prepare word frequencies for each topic
            topic_words = self.prepare_topic_word_frequencies(topic_info)
            
            if not topic_words:
                print("No topics with words found!")
                return
            
            print(f"\nGenerating word clouds for {len(topic_words)} topics...")
            
            # Generate individual word clouds (limit for performance)
            generated_files = []
            topic_ids = sorted(topic_words.keys())
            
            for topic_id in topic_ids[:max_individual_topics]:
                filepath = self.create_wordcloud_for_topic(
                    topic_id, topic_words[topic_id], topic_info
                )
                if filepath:
                    generated_files.append(filepath)
            
            # Generate aggregate word cloud (all topics combined)
            aggregate_path, aggregate_freqs, aggregate_stats = self.create_aggregate_wordcloud(topic_words)
            generated_files.append(aggregate_path)
            
            # Generate individual topics grid
            grid_path = self.create_combined_wordcloud(topic_words)
            generated_files.append(grid_path)
            
            # Generate summary with aggregate statistics
            summary_path = self.generate_topic_summary(topic_words, topic_info, aggregate_stats)
            generated_files.append(summary_path)
            
            print("\n" + "="*60)
            print("WORD CLOUD GENERATION COMPLETED")
            print("="*60)
            print(f"Output directory: {self.output_dir}")
            print(f"Generated {len(generated_files)} files")
            
            return generated_files
            
        except Exception as e:
            print(f"Error generating word clouds: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function."""
    generator = BERTopicWordCloudGenerator()
    results = generator.generate_wordclouds()
    
    if results:
        print(f"\nWord clouds generated successfully!")
        print(f"Check the output directory: {generator.output_dir}")
    else:
        print("Failed to generate word clouds")

if __name__ == "__main__":
    main()