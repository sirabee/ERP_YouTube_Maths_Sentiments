#!/usr/bin/env python3
"""
Simple Variable K BERTopic Intertopic Distance Map
Creates basic visualization showing topic relationships
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVariableKDistance:
    def __init__(self):
        """Initialize simple intertopic distance visualizer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "visualizations" / "simple_variable_k"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Simple Variable K Distance Map Visualizer")
        print(f"Output directory: {self.output_dir}")
    
    def load_topic_data(self):
        """Load Variable K topic data."""
        topic_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "merged_topic_info" / "merged_topic_info_sorted.csv"
        
        print(f"Loading topics from: {topic_file}")
        df = pd.read_csv(topic_file)
        
        # Limit to first 100 topics for clarity
        df = df.head(100)
        
        print(f"Loaded {len(df)} topics from {df['query'].nunique()} queries")
        return df
    
    def create_topic_vectors(self, df):
        """Create simple topic vectors from keywords."""
        print("Creating topic vectors...")
        
        # Get all unique keywords
        all_keywords = set()
        for rep in df['Representation']:
            if pd.notna(rep):
                # Clean and split keywords
                keywords = rep.replace('[', '').replace(']', '').replace("'", '').split(',')
                keywords = [k.strip().lower() for k in keywords[:5]]  # Use top 5
                all_keywords.update(keywords)
        
        keyword_list = sorted(list(all_keywords))
        print(f"Using {len(keyword_list)} unique keywords")
        
        # Create topic-keyword matrix
        vectors = []
        for _, row in df.iterrows():
            vector = np.zeros(len(keyword_list))
            
            if pd.notna(row['Representation']):
                keywords = row['Representation'].replace('[', '').replace(']', '').replace("'", '').split(',')
                keywords = [k.strip().lower() for k in keywords[:5]]
                
                for i, kw in enumerate(keywords):
                    if kw in keyword_list:
                        idx = keyword_list.index(kw)
                        vector[idx] = 5 - i  # Weight: 5 for first keyword, 4 for second, etc.
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def create_distance_map(self, df, vectors):
        """Create interactive distance map."""
        print("Creating distance map...")
        
        # Reduce to 2D using PCA (simpler than t-SNE)
        pca = PCA(n_components=2, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)
        
        # Apply PCA
        coords = pca.fit_transform(vectors_scaled)
        
        # Add coordinates to dataframe
        df = df.copy()
        df['x'] = coords[:, 0]
        df['y'] = coords[:, 1]
        
        # Extract top keywords for display
        df['keywords'] = df['Representation'].apply(lambda x: 
            ', '.join(x.replace('[', '').replace(']', '').replace("'", '').split(',')[:3]) 
            if pd.notna(x) else ''
        )
        
        # Create custom color sequence with more distinct colors for queries
        import plotly.colors as pc
        import colorsys
        
        # Get unique queries
        unique_queries = df['query'].unique()
        n_queries = len(unique_queries)
        
        print(f"Creating colors for {n_queries} unique queries...")
        
        # Use a combination of color scales for better distinction
        if n_queries <= 10:
            colors = px.colors.qualitative.Plotly
        elif n_queries <= 24:
            colors = px.colors.qualitative.Light24
        else:
            # Create extended color palette by combining multiple scales
            colors = (
                px.colors.qualitative.Light24 +
                px.colors.qualitative.Dark24 +
                px.colors.qualitative.Alphabet
            )
            # Generate additional colors if needed using HSV color space
            if len(colors) < n_queries:
                additional_colors = []
                n_additional = n_queries - len(colors)
                for i in range(n_additional):
                    # Distribute hues evenly across color wheel
                    hue = (i * 360 / n_additional) / 360
                    # Vary saturation and value for better distinction
                    saturation = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9
                    value = 0.8 + (i % 2) * 0.1  # 0.8, 0.9
                    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                    hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
                    additional_colors.append(hex_color)
                colors = colors + additional_colors
        
        # Create plot with queries for coloring
        fig = px.scatter(
            df,
            x='x', 
            y='y',
            color='query',
            size='Count',
            hover_data=['Topic', 'Count', 'keywords'],
            title='Variable K Topics - Distance Map<br><sub>Topics positioned by keyword similarity (PCA), colored by query</sub>',
            labels={'x': 'Component 1', 'y': 'Component 2', 'query': 'Query'},
            color_discrete_sequence=colors[:n_queries]
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate='<b>Topic %{customdata[0]}</b><br>' +
                         'Query: %{fullData.name}<br>' +
                         'Documents: %{customdata[1]}<br>' +
                         'Keywords: %{customdata[2]}<br>' +
                         '<extra></extra>'
        )
        
        fig.update_layout(
            width=1000,
            height=700,
            plot_bgcolor='white',
            legend=dict(
                title="Queries",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Save
        output_file = self.output_dir / f'distance_map_{self.timestamp}.html'
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"Saved distance map to: {output_file}")
        
        return output_file
    
    def create_similarity_heatmap(self, df, vectors):
        """Create topic similarity heatmap."""
        print("Creating similarity heatmap...")
        
        # Calculate cosine similarity
        similarities = cosine_similarity(vectors)
        
        # Get labels for first 20 topics
        n_topics = min(20, len(df))
        labels = []
        for i in range(n_topics):
            row = df.iloc[i]
            label = f"{row['query'][:8]}_T{row['Topic']}"
            labels.append(label)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarities[:n_topics, :n_topics],
            x=labels,
            y=labels,
            colorscale='viridis',
            text=np.round(similarities[:n_topics, :n_topics], 2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title='Topic Similarity Matrix<br><sub>Cosine similarity between topics</sub>',
            width=800,
            height=800,
            xaxis=dict(tickangle=45)
        )
        
        # Save
        output_file = self.output_dir / f'similarity_heatmap_{self.timestamp}.html'
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"Saved heatmap to: {output_file}")
        
        return output_file
    
    def create_topic_overview(self, df):
        """Create simple topic overview."""
        print("Creating topic overview...")
        
        # Topics per query
        query_counts = df.groupby('query').agg({
            'Topic': 'count',
            'Count': 'sum'
        }).reset_index()
        query_counts.columns = ['Query', 'Topics', 'Documents']
        
        # Top 15 queries
        top_queries = query_counts.nlargest(15, 'Documents')
        
        # Create bar chart
        fig = go.Figure([
            go.Bar(
                x=top_queries['Query'],
                y=top_queries['Documents'],
                marker_color='skyblue',
                text=top_queries['Topics'],
                textposition='outside',
                texttemplate='%{text} topics',
                hovertemplate='Query: %{x}<br>Documents: %{y}<br>Topics: %{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Topics by Query<br><sub>Document count and topic diversity</sub>',
            xaxis_title="Query",
            yaxis_title="Number of Documents",
            width=1000,
            height=500,
            xaxis=dict(tickangle=45),
            showlegend=False
        )
        
        # Save
        output_file = self.output_dir / f'topic_overview_{self.timestamp}.html'
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        print(f"Saved overview to: {output_file}")
        
        return output_file
    
    def run(self):
        """Run all visualizations."""
        print("=" * 60)
        print("SIMPLE VARIABLE K DISTANCE MAP")
        print("=" * 60)
        
        try:
            # Load data
            df = self.load_topic_data()
            
            # Create vectors
            vectors = self.create_topic_vectors(df)
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            files = []
            
            files.append(self.create_distance_map(df, vectors))
            files.append(self.create_similarity_heatmap(df, vectors))
            files.append(self.create_topic_overview(df))
            
            print("\n" + "=" * 60)
            print("SUCCESS")
            print("=" * 60)
            print(f"\nCreated {len(files)} visualizations in:")
            print(f"{self.output_dir}")
            
            print(f"\nSummary:")
            print(f"  • Topics analyzed: {len(df)}")
            print(f"  • Unique queries: {df['query'].nunique()}")
            print(f"  • Total documents: {df['Count'].sum():,}")
            
            return files
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return []

def main():
    """Main function."""
    visualizer = SimpleVariableKDistance()
    visualizer.run()

if __name__ == "__main__":
    main()