#!/usr/bin/env python3
"""
Keyword Network Analyzer for BERTopic Co-occurrence Data
Creates hierarchical network analysis to identify most connected keywords
MSc Data Science Thesis - Perceptions of Maths on YouTube
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import community as community_louvain

class KeywordNetworkAnalyzer:
    def __init__(self):
        """Initialize keyword network analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "analysis" / "keyword_networks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Keyword Network Analyzer Initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_cooccurrence_data(self):
        """Load the latest co-occurrence analysis results."""
        # Find the most recent co-occurrence files
        cooccur_files = list(self.base_path.glob("results/analysis/**/keyword_cooccurrences_*.csv"))
        if not cooccur_files:
            raise FileNotFoundError("No co-occurrence files found. Run keyword_cooccurrence_analyzer.py first.")
        
        # Get the most recent file
        latest_file = max(cooccur_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading co-occurrence data from: {latest_file}")
        
        cooccur_df = pd.read_csv(latest_file)
        print(f"Loaded {len(cooccur_df)} keyword co-occurrence pairs")
        
        return cooccur_df
    
    def build_keyword_network(self, cooccur_df, min_cooccurrence=5):
        """Build network graph from co-occurrence data."""
        print(f"\nBuilding keyword network (minimum {min_cooccurrence} co-occurrences)...")
        
        # Filter for meaningful relationships
        filtered_df = cooccur_df[cooccur_df['cooccurrence_count'] >= min_cooccurrence]
        print(f"Filtered to {len(filtered_df)} strong relationships")
        
        # Create network graph
        G = nx.Graph()
        
        # Add edges with weights
        for _, row in filtered_df.iterrows():
            G.add_edge(
                row['keyword1'], 
                row['keyword2'], 
                weight=row['cooccurrence_count']
            )
        
        print(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def calculate_centrality_metrics(self, G):
        """Calculate various centrality measures for keywords."""
        print("\nCalculating centrality metrics...")
        
        # Different types of centrality
        centrality_metrics = {}
        
        # Degree centrality - how many connections a keyword has
        centrality_metrics['degree'] = nx.degree_centrality(G)
        
        # Betweenness centrality - how often a keyword acts as a bridge
        centrality_metrics['betweenness'] = nx.betweenness_centrality(G, weight='weight')
        
        # Closeness centrality - how close a keyword is to all others
        centrality_metrics['closeness'] = nx.closeness_centrality(G, distance='weight')
        
        # Eigenvector centrality - importance based on connections to important nodes
        try:
            centrality_metrics['eigenvector'] = nx.eigenvector_centrality(G, weight='weight')
        except:
            print("Warning: Could not calculate eigenvector centrality")
            centrality_metrics['eigenvector'] = {node: 0 for node in G.nodes()}
        
        # PageRank - Google's algorithm for importance
        centrality_metrics['pagerank'] = nx.pagerank(G, weight='weight')
        
        # Weighted degree - sum of edge weights
        centrality_metrics['weighted_degree'] = dict(G.degree(weight='weight'))
        
        return centrality_metrics
    
    def create_hierarchical_ranking(self, centrality_metrics):
        """Create hierarchical ranking of keywords based on multiple centrality measures."""
        print("\nCreating hierarchical keyword ranking...")
        
        # Get all keywords
        all_keywords = set()
        for metric_dict in centrality_metrics.values():
            all_keywords.update(metric_dict.keys())
        
        # Create comprehensive ranking dataframe
        ranking_data = []
        
        for keyword in all_keywords:
            keyword_data = {'keyword': keyword}
            
            # Add each centrality measure
            for metric_name, metric_dict in centrality_metrics.items():
                keyword_data[f'{metric_name}_centrality'] = metric_dict.get(keyword, 0)
            
            ranking_data.append(keyword_data)
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Normalize centrality scores to 0-1 scale for comparison
        centrality_cols = [col for col in ranking_df.columns if col.endswith('_centrality')]
        for col in centrality_cols:
            ranking_df[f'{col}_normalized'] = (ranking_df[col] - ranking_df[col].min()) / (ranking_df[col].max() - ranking_df[col].min())
        
        # Calculate composite centrality score
        normalized_cols = [col for col in ranking_df.columns if col.endswith('_normalized')]
        ranking_df['composite_centrality'] = ranking_df[normalized_cols].mean(axis=1)
        
        # Sort by composite centrality
        ranking_df = ranking_df.sort_values('composite_centrality', ascending=False)
        
        return ranking_df
    
    def detect_keyword_communities(self, G):
        """Detect communities/clusters of related keywords."""
        print("\nDetecting keyword communities...")
        
        # Use Louvain algorithm for community detection
        try:
            partition = community_louvain.best_partition(G, weight='weight')
            
            # Organize communities
            communities = defaultdict(list)
            for keyword, community_id in partition.items():
                communities[community_id].append(keyword)
            
            # Sort communities by size
            sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
            
            print(f"Found {len(sorted_communities)} keyword communities")
            
            return dict(sorted_communities), partition
            
        except Exception as e:
            print(f"Warning: Community detection failed: {e}")
            return {}, {}
    
    def create_network_analysis_report(self, ranking_df, communities, G):
        """Create comprehensive network analysis report."""
        print("\nGenerating network analysis report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("KEYWORD NETWORK ANALYSIS - HIERARCHICAL CENTRALITY")
        report_lines.append("=" * 80)
        report_lines.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Network Size: {G.number_of_nodes()} keywords, {G.number_of_edges()} connections")
        report_lines.append(f"Communities Detected: {len(communities)}")
        
        # Network density and connectivity
        density = nx.density(G)
        report_lines.append(f"Network Density: {density:.4f}")
        
        # Top keywords by centrality
        report_lines.append("\n" + "=" * 50)
        report_lines.append("TOP 20 MOST CENTRAL KEYWORDS (HIERARCHICAL RANKING)")
        report_lines.append("=" * 50)
        report_lines.append("Rank | Keyword | Composite Score | Degree | PageRank | Betweenness")
        report_lines.append("-" * 75)
        
        for i, row in ranking_df.head(20).iterrows():
            rank = ranking_df.index.get_loc(i) + 1
            pagerank_val = row.get('pagerank_centrality', row.get('pagerank', 0))
            report_lines.append(
                f"{rank:4d} | {row['keyword']:15s} | {row['composite_centrality']:13.4f} | "
                f"{row['degree_centrality']:6.3f} | {pagerank_val:8.4f} | {row['betweenness_centrality']:11.4f}"
            )
        
        # Centrality leaders by specific metrics
        report_lines.append("\n" + "=" * 50)
        report_lines.append("CENTRALITY LEADERS BY METRIC")
        report_lines.append("=" * 50)
        
        metrics = ['degree_centrality', 'pagerank_centrality', 'betweenness_centrality', 'weighted_degree']
        for metric in metrics:
            if metric in ranking_df.columns:
                top_keyword = ranking_df.loc[ranking_df[metric].idxmax()]
                report_lines.append(f"{metric.replace('_', ' ').title()}: {top_keyword['keyword']} ({top_keyword[metric]:.4f})")
        
        # Keyword communities
        if communities:
            report_lines.append("\n" + "=" * 50)
            report_lines.append("KEYWORD COMMUNITIES (TOP 5 LARGEST)")
            report_lines.append("=" * 50)
            
            for i, (community_id, keywords) in enumerate(list(communities.items())[:5]):
                report_lines.append(f"\nCommunity {community_id + 1} ({len(keywords)} keywords):")
                
                # Show top keywords by centrality within this community
                community_keywords = set(keywords)
                community_ranking = ranking_df[ranking_df['keyword'].isin(community_keywords)]
                top_in_community = community_ranking.head(8)
                
                keyword_list = []
                for _, row in top_in_community.iterrows():
                    keyword_list.append(f"{row['keyword']} ({row['composite_centrality']:.3f})")
                
                report_lines.append(f"  Top keywords: {', '.join(keyword_list)}")
        
        # Network insights
        report_lines.append("\n" + "=" * 50)
        report_lines.append("NETWORK INSIGHTS")
        report_lines.append("=" * 50)
        
        # Identify hub keywords (high degree, high betweenness)
        hub_threshold = ranking_df['degree_centrality'].quantile(0.9)
        bridge_threshold = ranking_df['betweenness_centrality'].quantile(0.9)
        
        hubs = ranking_df[ranking_df['degree_centrality'] >= hub_threshold]['keyword'].tolist()
        bridges = ranking_df[ranking_df['betweenness_centrality'] >= bridge_threshold]['keyword'].tolist()
        
        report_lines.append(f"\nHub Keywords (highly connected): {', '.join(hubs[:10])}")
        report_lines.append(f"Bridge Keywords (connect communities): {', '.join(bridges[:10])}")
        
        # Most influential overall
        top_influencer = ranking_df.iloc[0]
        report_lines.append(f"\nMost Influential Keyword: '{top_influencer['keyword']}'")
        report_lines.append(f"  - Composite Centrality: {top_influencer['composite_centrality']:.4f}")
        report_lines.append(f"  - Direct Connections: {int(top_influencer['degree_centrality'] * (G.number_of_nodes() - 1))}")
        pagerank_score = top_influencer.get('pagerank_centrality', top_influencer.get('pagerank', 0))
        report_lines.append(f"  - PageRank Score: {pagerank_score:.4f}")
        
        # Save report
        report_file = self.output_dir / f"keyword_network_analysis_{self.timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved network analysis report to: {report_file}")
        
        return report_file
    
    def create_network_visualization(self, G, ranking_df, communities, partition):
        """Create network visualization with hierarchical layout."""
        print("\nCreating network visualization...")
        
        # Filter to top nodes for clarity
        top_keywords = set(ranking_df.head(50)['keyword'])
        G_viz = G.subgraph(top_keywords).copy()
        
        # Create visualization
        plt.figure(figsize=(20, 16))
        
        # Use spring layout with clustering
        pos = nx.spring_layout(G_viz, k=3, iterations=50, weight='weight')
        
        # Node sizes based on composite centrality
        node_sizes = []
        for node in G_viz.nodes():
            centrality = ranking_df[ranking_df['keyword'] == node]['composite_centrality'].iloc[0]
            node_sizes.append(centrality * 3000 + 100)
        
        # Node colors based on communities
        node_colors = []
        color_map = plt.cm.Set3(np.linspace(0, 1, len(set(partition.values()))))
        
        for node in G_viz.nodes():
            if node in partition:
                community_id = partition[node]
                node_colors.append(color_map[community_id % len(color_map)])
            else:
                node_colors.append('lightgray')
        
        # Draw edges with varying thickness
        edges = G_viz.edges()
        weights = [G_viz[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [w / max_weight * 3 + 0.5 for w in weights]
        
        nx.draw_networkx_edges(G_viz, pos, width=edge_widths, alpha=0.6, edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(G_viz, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        
        # Add labels for top keywords only
        top_10_keywords = set(ranking_df.head(10)['keyword'])
        labels = {node: node if node in top_10_keywords else '' for node in G_viz.nodes()}
        nx.draw_networkx_labels(G_viz, pos, labels, font_size=10, font_weight='bold')
        
        plt.title('Keyword Co-occurrence Network\n(Top 50 Keywords by Centrality, Colored by Community)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / f"keyword_network_visualization_{self.timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved network visualization to: {viz_file}")
        
        return viz_file
    
    def save_analysis_data(self, ranking_df, communities):
        """Save analysis data for further use."""
        print("\nSaving analysis datasets...")
        
        # Save ranking data
        ranking_file = self.output_dir / f"keyword_centrality_ranking_{self.timestamp}.csv"
        ranking_df.to_csv(ranking_file, index=False)
        
        # Save community data
        community_data = []
        for community_id, keywords in communities.items():
            for keyword in keywords:
                community_data.append({
                    'keyword': keyword,
                    'community_id': community_id,
                    'community_size': len(keywords)
                })
        
        community_df = pd.DataFrame(community_data)
        community_file = self.output_dir / f"keyword_communities_{self.timestamp}.csv"
        community_df.to_csv(community_file, index=False)
        
        print(f"Saved analysis data:")
        print(f"  • {ranking_file}")
        print(f"  • {community_file}")
        
        return ranking_file, community_file
    
    def run_analysis(self, min_cooccurrence=5):
        """Execute complete keyword network analysis."""
        print("=" * 60)
        print("KEYWORD NETWORK ANALYSIS - HIERARCHICAL CENTRALITY")
        print("=" * 60)
        
        # Load co-occurrence data
        cooccur_df = self.load_cooccurrence_data()
        
        # Build network
        G = self.build_keyword_network(cooccur_df, min_cooccurrence)
        
        # Calculate centrality metrics
        centrality_metrics = self.calculate_centrality_metrics(G)
        
        # Create hierarchical ranking
        ranking_df = self.create_hierarchical_ranking(centrality_metrics)
        
        # Detect communities
        communities, partition = self.detect_keyword_communities(G)
        
        # Create analysis report
        report_file = self.create_network_analysis_report(ranking_df, communities, G)
        
        # Create visualization
        viz_file = self.create_network_visualization(G, ranking_df, communities, partition)
        
        # Save analysis data
        ranking_file, community_file = self.save_analysis_data(ranking_df, communities)
        
        print("\n" + "=" * 60)
        print("NETWORK ANALYSIS COMPLETE")
        print("=" * 60)
        
        print(f"\nKey Network Statistics:")
        print(f"  • {G.number_of_nodes()} keywords in network")
        print(f"  • {G.number_of_edges()} connections")
        print(f"  • {len(communities)} communities detected")
        print(f"  • Network density: {nx.density(G):.4f}")
        
        print(f"\nTop 5 Most Central Keywords:")
        for i, row in ranking_df.head(5).iterrows():
            rank = ranking_df.index.get_loc(i) + 1
            print(f"  {rank}. {row['keyword']} (centrality: {row['composite_centrality']:.4f})")
        
        print(f"\nOutput saved to: {self.output_dir}")
        
        return ranking_df, communities, G

def main():
    """Main execution function."""
    analyzer = KeywordNetworkAnalyzer()
    ranking_df, communities, G = analyzer.run_analysis()

if __name__ == "__main__":
    main()