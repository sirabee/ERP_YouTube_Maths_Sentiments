#!/usr/bin/env python3
"""
Keyword Frequency Bar Charts Visualization
Creates BERTopic-style bar charts for keyword frequencies by topic
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re

class KeywordFrequencyVisualizer:
    def __init__(self):
        """Initialize keyword frequency visualizer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "visualizations" / "keyword_frequency_charts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for professional charts
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print("Keyword Frequency Bar Chart Visualizer Initialized")
        print(f"Output directory: {self.output_dir}")
    
    def parse_summary_file(self, summary_path):
        """Parse the keyword frequency summary text file."""
        with open(summary_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        topics_data = {}
        current_topic = None
        
        for line in lines:
            line = line.strip()
            
            # Match topic header
            if line.startswith("TOPIC"):
                topic_match = re.match(r"TOPIC (\d+):", line)
                if topic_match:
                    current_topic = int(topic_match.group(1))
                    topics_data[current_topic] = {
                        'queries': 0,
                        'documents': 0,
                        'keywords': []
                    }
            
            # Match queries count
            elif "Queries:" in line and current_topic is not None:
                queries_match = re.search(r"Queries:\s*(\d+)", line)
                if queries_match:
                    topics_data[current_topic]['queries'] = int(queries_match.group(1))
            
            # Match documents count
            elif "Documents:" in line and current_topic is not None:
                docs_match = re.search(r"Documents:\s*([\d,]+)", line)
                if docs_match:
                    doc_count = docs_match.group(1).replace(',', '')
                    topics_data[current_topic]['documents'] = int(doc_count)
            
            # Match keyword entries
            elif line.startswith("•") and current_topic is not None:
                # Parse: • keyword: count occurrences in X queries (Y%)
                keyword_match = re.match(r"•\s*([^:]+):\s*(\d+)\s*occurrences", line)
                if keyword_match:
                    keyword = keyword_match.group(1).strip()
                    count = int(keyword_match.group(2))
                    topics_data[current_topic]['keywords'].append({
                        'word': keyword,
                        'frequency': count
                    })
        
        return topics_data
    
    def create_single_topic_barchart(self, topic_num, topic_data, n_words=10):
        """Create a bar chart for a single topic."""
        if not topic_data['keywords']:
            return None
        
        # Get top n keywords
        keywords = topic_data['keywords'][:n_words]
        words = [kw['word'] for kw in keywords]
        frequencies = [kw['frequency'] for kw in keywords]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart (BERTopic style)
        y_pos = np.arange(len(words))
        bars = ax.barh(y_pos, frequencies, color='#4CAF50', alpha=0.8)
        
        # Customize appearance
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()  # Top word at the top
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Topic {topic_num} - Top Keywords\n({topic_data["queries"]} queries, {topic_data["documents"]:,} documents)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            ax.text(bar.get_width() + max(frequencies)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{freq:,}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Add grid for readability
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = f'topic_{topic_num}_keywords_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath
    
    def create_multi_topic_comparison(self, topics_data, topics_to_show=6, n_words=5):
        """Create a multi-panel comparison chart for multiple topics."""
        # Select topics to show
        topic_nums = sorted(list(topics_data.keys()))[:topics_to_show]
        
        # Create subplots
        n_cols = 3
        n_rows = (len(topic_nums) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        fig.suptitle('Keyword Frequencies Across Topics', fontsize=16, fontweight='bold', y=1.02)
        
        # Flatten axes array for easier iteration
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten()
        
        # Create bar chart for each topic
        for idx, topic_num in enumerate(topic_nums):
            ax = axes_flat[idx]
            topic_data = topics_data[topic_num]
            
            if topic_data['keywords']:
                # Get top n keywords
                keywords = topic_data['keywords'][:n_words]
                words = [kw['word'] for kw in keywords]
                frequencies = [kw['frequency'] for kw in keywords]
                
                # Create horizontal bars
                y_pos = np.arange(len(words))
                bars = ax.barh(y_pos, frequencies, color=f'C{topic_num % 10}', alpha=0.8)
                
                # Customize
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words, fontsize=10)
                ax.invert_yaxis()
                ax.set_xlabel('Frequency', fontsize=10)
                ax.set_title(f'Topic {topic_num}\n({topic_data["documents"]:,} docs)', 
                           fontsize=11, fontweight='bold')
                
                # Add value labels
                for bar, freq in zip(bars, frequencies):
                    ax.text(bar.get_width() + max(frequencies)*0.02, 
                          bar.get_y() + bar.get_height()/2,
                          f'{freq}', ha='left', va='center', fontsize=9)
                
                ax.grid(True, axis='x', alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
        
        # Hide unused subplots
        for idx in range(len(topic_nums), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'multi_topic_comparison_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath
    
    def create_top_keywords_overall(self, topics_data, n_words=20):
        """Create a bar chart of top keywords across all topics."""
        # Aggregate all keywords
        keyword_totals = {}
        
        for topic_num, topic_data in topics_data.items():
            for kw_data in topic_data['keywords']:
                keyword = kw_data['word']
                freq = kw_data['frequency']
                if keyword in keyword_totals:
                    keyword_totals[keyword] += freq
                else:
                    keyword_totals[keyword] = freq
        
        # Sort and get top keywords
        sorted_keywords = sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)[:n_words]
        words = [kw[0] for kw in sorted_keywords]
        frequencies = [kw[1] for kw in sorted_keywords]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(words))
        bars = ax.barh(y_pos, frequencies, color='#2196F3', alpha=0.8)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Total Frequency Across All Topics', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {n_words} Keywords Overall\n(Aggregated across {len(topics_data)} topics)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for bar, freq in zip(bars, frequencies):
            ax.text(bar.get_width() + max(frequencies)*0.01, 
                   bar.get_y() + bar.get_height()/2,
                   f'{freq:,}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'top_keywords_overall_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath
    
    def create_keyword_distribution_matrix(self, topics_data, n_keywords=10):
        """Create a heatmap showing keyword distribution across topics."""
        # Get all unique keywords
        all_keywords = set()
        for topic_data in topics_data.values():
            for kw_data in topic_data['keywords'][:n_keywords]:
                all_keywords.add(kw_data['word'])
        
        # Sort keywords
        keywords_list = sorted(list(all_keywords))
        topics_list = sorted(topics_data.keys())
        
        # Create matrix
        matrix = np.zeros((len(keywords_list), len(topics_list)))
        
        for j, topic_num in enumerate(topics_list):
            topic_keywords = {kw['word']: kw['frequency'] 
                            for kw in topics_data[topic_num]['keywords']}
            for i, keyword in enumerate(keywords_list):
                if keyword in topic_keywords:
                    matrix[i, j] = topic_keywords[keyword]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(8, len(keywords_list) * 0.3)))
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(topics_list)))
        ax.set_yticks(np.arange(len(keywords_list)))
        ax.set_xticklabels([f'Topic {t}' for t in topics_list])
        ax.set_yticklabels(keywords_list)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency', rotation=270, labelpad=20)
        
        # Add title
        ax.set_title('Keyword Distribution Across Topics', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'keyword_distribution_matrix_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath
    
    def create_split_topic_comparisons(self, topics_data, n_words=8):
        """Create two separate multi-panel charts: Topics 0-5 and Topics 6-11."""
        generated_files = []
        
        # Get all topic numbers and sort them
        all_topics = sorted(list(topics_data.keys()))
        
        # Split into two groups
        topics_groups = [
            (all_topics[:6], "Topics 0-5", "topics_0_5"),
            (all_topics[6:12], "Topics 6-11", "topics_6_11")
        ]
        
        for topic_nums, title_suffix, file_suffix in topics_groups:
            if not topic_nums:
                continue
                
            # Create figure with 2 rows x 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(f'Keyword Frequencies - {title_suffix}', 
                        fontsize=18, fontweight='bold', y=1.02)
            
            # Flatten axes for easier iteration
            axes_flat = axes.flatten()
            
            # Create bar chart for each topic
            for idx, topic_num in enumerate(topic_nums):
                if topic_num not in topics_data:
                    continue
                    
                ax = axes_flat[idx]
                topic_data = topics_data[topic_num]
                
                if topic_data['keywords']:
                    # Get top n keywords
                    keywords = topic_data['keywords'][:n_words]
                    words = [kw['word'] for kw in keywords]
                    frequencies = [kw['frequency'] for kw in keywords]
                    
                    # Create horizontal bars with distinct colors for each topic
                    y_pos = np.arange(len(words))
                    color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
                                   '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788']
                    bars = ax.barh(y_pos, frequencies, color=color_palette[topic_num % 12], alpha=0.8)
                    
                    # Customize appearance
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(words, fontsize=11)
                    ax.invert_yaxis()
                    ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
                    
                    # Enhanced title with more info
                    ax.set_title(f'Topic {topic_num}\n{topic_data["queries"]} queries | {topic_data["documents"]:,} documents', 
                               fontsize=12, fontweight='bold')
                    
                    # Add value labels on bars
                    max_freq = max(frequencies) if frequencies else 1
                    for bar, freq in zip(bars, frequencies):
                        ax.text(bar.get_width() + max_freq*0.02, 
                              bar.get_y() + bar.get_height()/2,
                              f'{freq:,}', ha='left', va='center', 
                              fontsize=10, fontweight='bold')
                    
                    # Add subtle grid
                    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
                    ax.set_axisbelow(True)
                    
                    # Set x-axis limits for consistency
                    ax.set_xlim(0, max_freq * 1.15)
            
            # Hide unused subplots
            for idx in range(len(topic_nums), len(axes_flat)):
                axes_flat[idx].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            filename = f'keyword_frequencies_{file_suffix}_{self.timestamp}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            generated_files.append(filepath)
            print(f"  ✓ {title_suffix} chart saved: {filename}")
        
        return generated_files
    
    def generate_all_visualizations(self, summary_path):
        """Generate split bar chart visualizations for all topics."""
        print("\n" + "=" * 60)
        print("GENERATING KEYWORD FREQUENCY BAR CHARTS")
        print("=" * 60)
        
        # Parse summary file
        print(f"\nParsing summary file: {summary_path}")
        topics_data = self.parse_summary_file(summary_path)
        print(f"Found {len(topics_data)} topics")
        
        # Generate split multi-topic comparison charts
        print("\nGenerating split topic comparison charts...")
        generated_files = self.create_split_topic_comparisons(topics_data)
        
        print("\n" + "=" * 60)
        print("VISUALIZATION GENERATION COMPLETE")
        print("=" * 60)
        print(f"\nGenerated {len(generated_files)} visualizations")
        print(f"Output directory: {self.output_dir}")
        
        return generated_files

def main():
    """Main execution function."""
    visualizer = KeywordFrequencyVisualizer()
    
    # Default summary file path
    summary_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/analysis/formatted_summaries/keyword_frequency_summary_20250829_210917.txt"
    
    # Generate all visualizations
    generated_files = visualizer.generate_all_visualizations(summary_path)
    
    print("\nGenerated files:")
    for filepath in generated_files:
        print(f"  • {filepath.name}")

if __name__ == "__main__":
    main()