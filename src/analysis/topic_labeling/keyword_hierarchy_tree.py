#!/usr/bin/env python3
"""
Keyword Hierarchy Tree Analysis
Creates dendrogram of keywords showing semantic relationships
MSc Data Science Thesis - Perceptions of Maths on YouTube
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

class KeywordHierarchyAnalyzer:
    def __init__(self):
        """Initialize keyword hierarchy analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "analysis" / "keyword_hierarchy"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Keyword Hierarchy Tree Analyzer")
        print(f"Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load topic data and extract all keywords."""
        topic_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "merged_topic_info" / "merged_topic_info_sorted.csv"
        
        print(f"Loading data...")
        df = pd.read_csv(topic_file)
        print(f"Loaded {len(df)} topics from {df['query'].nunique()} queries")
        
        return df
    
    def extract_all_keywords(self, df):
        """Extract and count all keywords across all topics."""
        print("Extracting keywords...")
        
        keyword_counter = Counter()
        keyword_contexts = {}
        
        for _, row in df.iterrows():
            if pd.notna(row['Representation']):
                # Clean and split keywords
                keywords = row['Representation'].replace('[', '').replace(']', '').replace("'", '').split(',')
                keywords = [k.strip().lower() for k in keywords[:10]]
                
                for keyword in keywords:
                    keyword_counter[keyword] += 1
                    
                    # Store context for each keyword
                    if keyword not in keyword_contexts:
                        keyword_contexts[keyword] = []
                    keyword_contexts[keyword].append({
                        'query': row['query'],
                        'topic': row['Topic'],
                        'count': row['Count']
                    })
        
        print(f"Found {len(keyword_counter)} unique keywords")
        return keyword_counter, keyword_contexts
    
    def create_keyword_cooccurrence_matrix(self, df, top_keywords):
        """Create co-occurrence matrix for keywords."""
        print("Creating keyword co-occurrence matrix...")
        
        # Create binary matrix: each row is a topic, each column is a keyword
        keyword_presence = []
        
        for _, row in df.iterrows():
            if pd.notna(row['Representation']):
                keywords = row['Representation'].replace('[', '').replace(']', '').replace("'", '').split(',')
                keywords = [k.strip().lower() for k in keywords[:10]]
                
                # Binary vector for this topic
                vector = [1 if kw in keywords else 0 for kw in top_keywords]
                keyword_presence.append(vector)
        
        keyword_matrix = np.array(keyword_presence)
        
        # Calculate co-occurrence (how often keywords appear together)
        cooccurrence = np.dot(keyword_matrix.T, keyword_matrix)
        
        # Normalize by keyword frequency to get association strength
        keyword_frequencies = np.diag(cooccurrence)
        normalized_cooccurrence = cooccurrence / np.sqrt(np.outer(keyword_frequencies, keyword_frequencies))
        
        return normalized_cooccurrence
    
    def create_semantic_groups(self, keyword_counter):
        """Group keywords by semantic categories."""
        print("Creating semantic groups...")
        
        semantic_groups = {
            'mathematics_core': ['math', 'maths', 'mathematics', 'mathematical', 'calculus', 'algebra', 'geometry'],
            'appreciation': ['thank', 'thanks', 'appreciate', 'grateful', 'love', 'amazing', 'awesome', 'great'],
            'learning': ['learn', 'learning', 'understand', 'explanation', 'tutorial', 'lesson', 'teach', 'teacher'],
            'education': ['school', 'student', 'class', 'exam', 'test', 'homework', 'study', 'course'],
            'content': ['video', 'videos', 'channel', 'watch', 'content', 'series'],
            'difficulty': ['hard', 'easy', 'difficult', 'simple', 'complex', 'basic', 'advanced'],
            'help': ['help', 'helped', 'helpful', 'support', 'assist', 'guide'],
            'question': ['question', 'questions', 'ask', 'answer', 'problem', 'problems', 'solve']
        }
        
        # Map keywords to groups
        keyword_groups = {}
        ungrouped_keywords = []
        
        for keyword in keyword_counter.keys():
            grouped = False
            for group_name, group_keywords in semantic_groups.items():
                if keyword in group_keywords:
                    keyword_groups[keyword] = group_name
                    grouped = True
                    break
            
            if not grouped:
                ungrouped_keywords.append(keyword)
                keyword_groups[keyword] = 'other'
        
        print(f"Grouped {len(keyword_counter) - len(ungrouped_keywords)} keywords into semantic categories")
        print(f"Ungrouped keywords: {len(ungrouped_keywords)}")
        
        return keyword_groups, semantic_groups
    
    def create_keyword_dendrogram(self, top_keywords, cooccurrence_matrix):
        """Create hierarchical clustering dendrogram of keywords."""
        print("Creating keyword dendrogram...")
        
        # Convert similarity to distance
        distance_matrix = 1 - cooccurrence_matrix
        
        # Ensure diagonal is zero and matrix is symmetric
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Convert to condensed distance matrix format
        distances = pdist(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')
        
        # Create dendrogram
        plt.figure(figsize=(16, 10))
        
        dendrogram(
            linkage_matrix,
            labels=top_keywords,
            leaf_rotation=90,
            leaf_font_size=10
        )
        
        plt.title('Keyword Hierarchy Tree\n(Keywords clustered by co-occurrence patterns)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Keywords', fontweight='bold')
        plt.ylabel('Distance (1 - Normalized Co-occurrence)', fontweight='bold')
        plt.tight_layout()
        
        # Save dendrogram
        dendro_file = self.output_dir / f'keyword_hierarchy_{self.timestamp}.png'
        plt.savefig(dendro_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved keyword dendrogram to: {dendro_file}")
        return dendro_file
    
    def create_semantic_hierarchy(self, keyword_groups, keyword_counter, semantic_groups):
        """Create semantic group hierarchy."""
        print("Creating semantic hierarchy...")
        
        # Count keywords per semantic group
        group_counts = {}
        group_keywords = {}
        
        for group_name in semantic_groups.keys():
            group_counts[group_name] = 0
            group_keywords[group_name] = []
            
            for keyword, count in keyword_counter.most_common():
                if keyword_groups.get(keyword) == group_name:
                    group_counts[group_name] += count
                    group_keywords[group_name].append((keyword, count))
        
        # Create hierarchical visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left plot: Group totals
        groups = list(group_counts.keys())
        counts = [group_counts[g] for g in groups]
        
        bars = ax1.barh(groups, counts, color='lightblue')
        ax1.set_xlabel('Total Keyword Occurrences', fontweight='bold')
        ax1.set_title('Semantic Groups by Total Frequency', fontweight='bold')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax1.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{count}', ha='left', va='center', fontweight='bold')
        
        # Right plot: Top keywords per group
        y_pos = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for i, group in enumerate(groups):
            top_keywords_in_group = group_keywords[group][:5]  # Top 5 per group
            
            if top_keywords_in_group:
                for j, (keyword, count) in enumerate(top_keywords_in_group):
                    ax2.barh(y_pos, count, color=colors[i], alpha=0.7)
                    ax2.text(count + max([kc[1] for kc in keyword_counter.most_common(50)])*0.01, 
                            y_pos, f'{keyword}', ha='left', va='center', fontsize=9)
                    y_pos += 1
                
                # Add group separator
                y_pos += 0.5
        
        ax2.set_xlabel('Keyword Frequency', fontweight='bold')
        ax2.set_title('Top Keywords by Semantic Group', fontweight='bold')
        ax2.set_yticks([])
        
        plt.tight_layout()
        
        # Save semantic hierarchy
        semantic_file = self.output_dir / f'semantic_hierarchy_{self.timestamp}.png'
        plt.savefig(semantic_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved semantic hierarchy to: {semantic_file}")
        return semantic_file
    
    def create_keyword_tree_summary(self, keyword_counter, keyword_groups, semantic_groups):
        """Create text summary of keyword hierarchy."""
        print("Creating hierarchy summary...")
        
        summary_lines = []
        summary_lines.append("KEYWORD HIERARCHY ANALYSIS")
        summary_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        summary_lines.append("="*60)
        summary_lines.append("")
        
        # Overall statistics
        summary_lines.append("OVERALL STATISTICS:")
        summary_lines.append(f"  Total unique keywords: {len(keyword_counter)}")
        summary_lines.append(f"  Total keyword occurrences: {sum(keyword_counter.values())}")
        summary_lines.append("")
        
        # Top keywords overall
        summary_lines.append("TOP 15 KEYWORDS (All Topics):")
        for i, (keyword, count) in enumerate(keyword_counter.most_common(15), 1):
            group = keyword_groups.get(keyword, 'other')
            summary_lines.append(f"  {i:2d}. {keyword:<15} ({count:3d} occurrences) [{group}]")
        summary_lines.append("")
        
        # Semantic groups breakdown
        summary_lines.append("SEMANTIC GROUPS BREAKDOWN:")
        
        for group_name, group_keywords_list in semantic_groups.items():
            group_total = 0
            group_keywords_found = []
            
            for keyword, count in keyword_counter.most_common():
                if keyword_groups.get(keyword) == group_name:
                    group_total += count
                    group_keywords_found.append((keyword, count))
            
            if group_keywords_found:
                summary_lines.append(f"\n  {group_name.upper().replace('_', ' ')} ({group_total} total):")
                for keyword, count in group_keywords_found[:8]:  # Top 8 per group
                    percentage = (count / sum(keyword_counter.values())) * 100
                    summary_lines.append(f"    • {keyword:<12} {count:3d} ({percentage:.1f}%)")
        
        # Hierarchy interpretation
        summary_lines.append("\n" + "="*60)
        summary_lines.append("HIERARCHY INTERPRETATION:")
        summary_lines.append("="*60)
        
        # Find parent keywords (most frequent in each group)
        parent_keywords = {}
        for group_name in semantic_groups.keys():
            for keyword, count in keyword_counter.most_common():
                if keyword_groups.get(keyword) == group_name:
                    parent_keywords[group_name] = (keyword, count)
                    break
        
        summary_lines.append("\nPARENT KEYWORDS (Most frequent per semantic group):")
        for group, (keyword, count) in parent_keywords.items():
            summary_lines.append(f"  {group:<20} → {keyword} ({count} occurrences)")
        
        # Save summary
        summary_file = self.output_dir / f'hierarchy_summary_{self.timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"Saved hierarchy summary to: {summary_file}")
        return summary_file
    
    def run(self):
        """Run complete keyword hierarchy analysis."""
        print("="*60)
        print("KEYWORD HIERARCHY TREE ANALYSIS")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Extract keywords
        keyword_counter, keyword_contexts = self.extract_all_keywords(df)
        
        # Get top 50 keywords for dendrogram (for clarity)
        top_keywords = [kw for kw, count in keyword_counter.most_common(50)]
        
        # Create co-occurrence matrix
        cooccurrence_matrix = self.create_keyword_cooccurrence_matrix(df, top_keywords)
        
        # Create semantic groups
        keyword_groups, semantic_groups = self.create_semantic_groups(keyword_counter)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        files = []
        
        files.append(self.create_keyword_dendrogram(top_keywords, cooccurrence_matrix))
        files.append(self.create_semantic_hierarchy(keyword_groups, keyword_counter, semantic_groups))
        files.append(self.create_keyword_tree_summary(keyword_counter, keyword_groups, semantic_groups))
        
        print("\n" + "="*60)
        print("KEYWORD HIERARCHY ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved in: {self.output_dir}")
        
        return files

def main():
    """Main function."""
    analyzer = KeywordHierarchyAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()