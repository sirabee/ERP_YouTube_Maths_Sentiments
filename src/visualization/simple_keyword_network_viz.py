#!/usr/bin/env python3
"""
Simple Keyword Network Visualizations
Basic network visualizations without complex algorithms
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from collections import defaultdict

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

class SimpleKeywordVisualizer:
    def __init__(self):
        """Initialize simple keyword visualizer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "visualizations" / "simple_keyword_networks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Simple Keyword Network Visualizer Initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load co-occurrence data."""
        # Find the most recent co-occurrence file
        cooccur_files = list(self.base_path.glob("results/analysis/**/keyword_cooccurrences_*.csv"))
        
        if not cooccur_files:
            raise FileNotFoundError("No co-occurrence files found. Run keyword_cooccurrence_analyzer.py first.")
        
        latest_file = max(cooccur_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading data from: {latest_file}")
        
        cooccur_df = pd.read_csv(latest_file)
        print(f"Loaded {len(cooccur_df)} keyword pairs")
        
        return cooccur_df
    
    def normalize_cooccurrence_data(self, cooccur_df):
        """Apply singular/plural normalization to co-occurrence data."""
        print("Normalizing singular/plural forms in keyword co-occurrences...")
        
        # Get all unique keywords from the co-occurrence data
        all_keywords = set(cooccur_df['keyword1'].tolist() + cooccur_df['keyword2'].tolist())
        
        print(f"  Original keywords: {len(all_keywords)}")
        
        # Create a simple mapping based on direct normalization rules
        keyword_mapping = {}
        
        for keyword in all_keywords:
            normalized = self._normalize_single_keyword(keyword)
            keyword_mapping[keyword] = normalized
        
        # Print combinations for verification
        reverse_mapping = defaultdict(list)
        for orig, norm in keyword_mapping.items():
            reverse_mapping[norm].append(orig)
        
        for norm, origs in reverse_mapping.items():
            if len(origs) > 1:
                print(f"  Combined: {' + '.join(origs)} → {norm}")
        
        # Apply mapping to co-occurrence data
        combined_pairs = {}
        
        for _, row in cooccur_df.iterrows():
            kw1_norm = keyword_mapping[row['keyword1']]
            kw2_norm = keyword_mapping[row['keyword2']]
            
            # Skip self-loops (same keyword with itself)
            if kw1_norm == kw2_norm:
                continue
                
            # Ensure consistent ordering for pairs
            if kw1_norm > kw2_norm:
                kw1_norm, kw2_norm = kw2_norm, kw1_norm
            
            pair_key = (kw1_norm, kw2_norm)
            if pair_key in combined_pairs:
                combined_pairs[pair_key] += row['cooccurrence_count']
            else:
                combined_pairs[pair_key] = row['cooccurrence_count']
        
        # Create normalized DataFrame
        normalized_data = []
        for (kw1, kw2), count in combined_pairs.items():
            normalized_data.append({
                'keyword1': kw1,
                'keyword2': kw2,
                'cooccurrence_count': count
            })
        
        normalized_df = pd.DataFrame(normalized_data)
        normalized_df = normalized_df.sort_values('cooccurrence_count', ascending=False).reset_index(drop=True)
        
        unique_normalized = len(set(keyword_mapping.values()))
        print(f"  Normalized to {unique_normalized} unique keywords")
        print(f"  Reduced keyword pairs from {len(cooccur_df)} to {len(normalized_df)}")
        
        return normalized_df
    
    def _normalize_single_keyword(self, keyword):
        """Normalize a single keyword using simple rules."""
        word_lower = keyword.lower()
        
        # Simple rule-based approach for common patterns
        if word_lower.endswith('s') and len(word_lower) > 1:
            # Check if removing 's' gives us a reasonable singular form
            candidate = word_lower[:-1]
            # Don't modify if it would create very short words or known exceptions
            if len(candidate) >= 3 and candidate not in ['thi', 'wa', 'i', 'clas', 'glas']:
                return candidate
        elif word_lower.endswith('es') and len(word_lower) > 2:
            # Handle -es endings (classes -> class, boxes -> box)
            candidate = word_lower[:-2]
            if len(candidate) >= 3:
                return candidate
        elif word_lower.endswith('ies') and len(word_lower) > 3:
            # Handle -ies endings (studies -> study)
            return word_lower[:-3] + 'y'
        
        return word_lower
    
    def create_top_connections_chart(self, cooccur_df):
        """Create simple bar chart of top keyword connections."""
        print("Creating top connections chart...")
        
        # Get top 20 connections
        top_connections = cooccur_df.head(20)
        
        # Create connection labels
        top_connections['connection'] = top_connections['keyword1'] + ' + ' + top_connections['keyword2']
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_connections['connection'], top_connections['cooccurrence_count'], color='skyblue')
        
        # Add value labels
        for bar, count in zip(bars, top_connections['cooccurrence_count']):
            width = bar.get_width()
            plt.text(width + 5, bar.get_y() + bar.get_height()/2, f'{count}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.xlabel('Co-occurrence Count', fontweight='bold')
        plt.title('Top 20 Keyword Connections\n(How often keywords appear together)', fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save chart
        chart_file = self.output_dir / f'top_connections_{self.timestamp}.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved top connections chart to: {chart_file}")
        return chart_file
    
    def create_keyword_network_simple(self, cooccur_df):
        """Create simple network visualization using only connection counts."""
        print("Creating simple network visualization...")
        
        # Filter to strong connections for clarity
        strong_connections = cooccur_df[cooccur_df['cooccurrence_count'] >= 20]
        
        # Get all unique keywords
        all_keywords = set(strong_connections['keyword1'].tolist() + strong_connections['keyword2'].tolist())
        
        # Count total connections per keyword (simple centrality)
        keyword_connections = {}
        for keyword in all_keywords:
            connections = len(strong_connections[
                (strong_connections['keyword1'] == keyword) | 
                (strong_connections['keyword2'] == keyword)
            ])
            keyword_connections[keyword] = connections
        
        # Get top 30 most connected keywords
        top_keywords = sorted(keyword_connections.items(), key=lambda x: x[1], reverse=True)[:30]
        top_keyword_names = [kw[0] for kw in top_keywords]
        
        # Filter connections to only include top keywords
        filtered_connections = strong_connections[
            (strong_connections['keyword1'].isin(top_keyword_names)) & 
            (strong_connections['keyword2'].isin(top_keyword_names))
        ]
        
        # Create network graph
        G = nx.Graph()
        
        # Add edges
        for _, row in filtered_connections.iterrows():
            G.add_edge(row['keyword1'], row['keyword2'], weight=row['cooccurrence_count'])
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create visualization
        plt.figure(figsize=(16, 12))
        
        # Draw edges with varying thickness
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 5 + 0.5 for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
        
        # Draw nodes with sizes based on connections
        node_sizes = [keyword_connections.get(node, 1) * 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Keyword Connection Network\n(Node size = number of connections, Line thickness = connection strength)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Save network
        network_file = self.output_dir / f'simple_network_{self.timestamp}.png'
        plt.savefig(network_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved simple network to: {network_file}")
        return network_file
    
    def create_connection_heatmap(self, cooccur_df):
        """Create heatmap of keyword connections."""
        print("Creating connection heatmap...")
        
        # Get top 20 most frequent keywords
        all_keywords = list(set(cooccur_df['keyword1'].tolist() + cooccur_df['keyword2'].tolist()))
        
        # Count frequency for each keyword
        keyword_freq = {}
        for keyword in all_keywords:
            freq = cooccur_df[
                (cooccur_df['keyword1'] == keyword) | 
                (cooccur_df['keyword2'] == keyword)
            ]['cooccurrence_count'].sum()
            keyword_freq[keyword] = freq
        
        # Get top 20 keywords
        top_20_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        top_20_names = [kw[0] for kw in top_20_keywords]
        
        # Create connection matrix
        matrix = np.zeros((20, 20))
        
        for i, kw1 in enumerate(top_20_names):
            for j, kw2 in enumerate(top_20_names):
                if i != j:
                    # Find connection strength
                    connection = cooccur_df[
                        ((cooccur_df['keyword1'] == kw1) & (cooccur_df['keyword2'] == kw2)) |
                        ((cooccur_df['keyword1'] == kw2) & (cooccur_df['keyword2'] == kw1))
                    ]
                    if not connection.empty:
                        matrix[i, j] = connection['cooccurrence_count'].iloc[0]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, 
                   xticklabels=top_20_names, 
                   yticklabels=top_20_names,
                   annot=True, 
                   fmt='g',
                   cmap='viridis',
                   cbar_kws={'label': 'Co-occurrence Count'})
        
        plt.title('Keyword Co-occurrence Heatmap\n(Top 20 Most Connected Keywords)', 
                 fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save heatmap
        heatmap_file = self.output_dir / f'connection_heatmap_{self.timestamp}.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved connection heatmap to: {heatmap_file}")
        return heatmap_file
    
    def create_interactive_network(self, cooccur_df):
        """Create simple interactive network using plotly."""
        print("Creating interactive network...")
        
        # Filter to top connections
        top_connections = cooccur_df.head(50)
        
        # Get unique keywords
        all_keywords = list(set(top_connections['keyword1'].tolist() + top_connections['keyword2'].tolist()))
        
        # Count connections per keyword
        keyword_counts = {}
        for keyword in all_keywords:
            count = len(top_connections[
                (top_connections['keyword1'] == keyword) | 
                (top_connections['keyword2'] == keyword)
            ])
            keyword_counts[keyword] = count
        
        # Create simple layout (circular)
        n_keywords = len(all_keywords)
        angles = np.linspace(0, 2*np.pi, n_keywords, endpoint=False)
        
        # Position keywords in circle
        keyword_pos = {}
        for i, keyword in enumerate(all_keywords):
            keyword_pos[keyword] = {
                'x': np.cos(angles[i]),
                'y': np.sin(angles[i])
            }
        
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for _, row in top_connections.iterrows():
            kw1, kw2 = row['keyword1'], row['keyword2']
            if kw1 in keyword_pos and kw2 in keyword_pos:
                edge_x.extend([keyword_pos[kw1]['x'], keyword_pos[kw2]['x'], None])
                edge_y.extend([keyword_pos[kw1]['y'], keyword_pos[kw2]['y'], None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = [keyword_pos[kw]['x'] for kw in all_keywords]
        node_y = [keyword_pos[kw]['y'] for kw in all_keywords]
        node_size = [keyword_counts[kw] * 5 + 10 for kw in all_keywords]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=all_keywords,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(width=2, color='black')
            ),
            hovertext=[f'{kw}: {keyword_counts[kw]} connections' for kw in all_keywords]
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                        title='Interactive Keyword Network<br><sub>Click and drag to explore connections</sub>',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Node size = number of connections",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(color="black", size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                      )
        
        # Save interactive network
        interactive_file = self.output_dir / f'interactive_network_{self.timestamp}.html'
        pyo.plot(fig, filename=str(interactive_file), auto_open=False)
        
        print(f"Saved interactive network to: {interactive_file}")
        return interactive_file
    
    def run_all_visualizations(self):
        """Run all simple visualizations."""
        print("=" * 60)
        print("SIMPLE KEYWORD NETWORK VISUALIZATIONS")
        print("=" * 60)
        
        # Load data
        cooccur_df = self.load_data()
        
        # Apply singular/plural normalization
        normalized_df = self.normalize_cooccurrence_data(cooccur_df)
        
        # Create visualizations using normalized data
        viz_files = {}
        
        print("\nGenerating visualizations...")
        viz_files['top_connections'] = self.create_top_connections_chart(normalized_df)
        viz_files['simple_network'] = self.create_keyword_network_simple(normalized_df)
        viz_files['heatmap'] = self.create_connection_heatmap(normalized_df)
        viz_files['interactive'] = self.create_interactive_network(normalized_df)
        
        print("\n" + "=" * 60)
        print("SIMPLE VISUALIZATION COMPLETE")
        print("=" * 60)
        
        print(f"\nGenerated files in: {self.output_dir}")
        for viz_type, file_path in viz_files.items():
            print(f"  • {viz_type}: {file_path.name}")
        
        print(f"\nData processed:")
        print(f"  • Original: {len(cooccur_df)} keyword connections")
        print(f"  • Normalized: {len(normalized_df)} keyword connections")
        print(f"  • Top connections range from {normalized_df['cooccurrence_count'].max()} to {normalized_df['cooccurrence_count'].min()} co-occurrences")
        
        return viz_files

def main():
    """Main execution function."""
    visualizer = SimpleKeywordVisualizer()
    viz_files = visualizer.run_all_visualizations()

if __name__ == "__main__":
    main()