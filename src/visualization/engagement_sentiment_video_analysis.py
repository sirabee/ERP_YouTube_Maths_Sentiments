#!/usr/bin/env python3
"""
Video Engagement Score and Sentiment Distribution Analysis
Maps videos with highest engagement scores to their sentiment distributions
and analyzes specific identity-related queries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

class EngagementSentimentAnalyzer:
    def __init__(self):
        """Initialize engagement sentiment analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "analysis" / "engagement_sentiment"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define actual identity-related search queries from the dataset
        self.identity_queries = [
            'women_in_maths',
            'girls_and_math', 
            'diversity_mathematics',
            'maths_culture',
            'math_stereotypes',
            'mathematics_society',
            'maths_for_everyone'
        ]
        
        print("Engagement Sentiment Analyzer Initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_enhanced_comments(self):
        """Load the enhanced comments dataset."""
        data_path = self.base_path / "results" / "models" / "sentiment_analysis" / "xlm_roberta_clean_20250816_132139" / "enhanced_comments.csv"
        
        print(f"Loading enhanced comments from: {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"Total comments loaded: {len(df):,}")
        print(f"Unique videos: {df['video_id'].nunique():,}")
        print(f"Unique search queries: {df['search_query'].nunique():,}")
        
        return df
    
    def calculate_engagement_scores(self, df):
        """Calculate weighted engagement scores where comments are valued 10x more than likes."""
        
        # Apply minimum view threshold (consistent with other analysis scripts)
        MIN_VIEWS = 1000
        
        # Group by video to calculate metrics
        video_metrics = df.groupby(['video_id', 'video_title', 'search_query']).agg({
            'video_views': 'first',
            'video_likes': 'first',
            'comment_id': 'count'
        }).reset_index()
        
        video_metrics.rename(columns={'comment_id': 'comment_count'}, inplace=True)
        
        # Filter videos with minimum view threshold
        video_metrics_filtered = video_metrics[video_metrics['video_views'] >= MIN_VIEWS].copy()
        
        print(f"Applied minimum view threshold of {MIN_VIEWS:,} views")
        print(f"Videos before filtering: {len(video_metrics):,}")
        print(f"Videos after filtering: {len(video_metrics_filtered):,}")
        
        # Calculate rates as percentages (matching the source script exactly)
        video_metrics_filtered['like_rate'] = np.where(video_metrics_filtered['video_views'] > 0, 
                                            (video_metrics_filtered['video_likes'] / video_metrics_filtered['video_views'] * 100), 0)
        video_metrics_filtered['comment_rate'] = np.where(video_metrics_filtered['video_views'] > 0, 
                                               (video_metrics_filtered['comment_count'] / video_metrics_filtered['video_views'] * 100), 0)
        
        # Weighted engagement score: comments valued 10x more than likes
        COMMENT_WEIGHT = 10
        video_metrics_filtered['engagement_score'] = video_metrics_filtered['like_rate'] + (video_metrics_filtered['comment_rate'] * COMMENT_WEIGHT)
        
        return video_metrics_filtered.sort_values('engagement_score', ascending=False)
    
    def analyze_sentiment_by_video(self, df, top_videos):
        """Analyze sentiment distribution for top engagement videos."""
        
        sentiment_analysis = {}
        
        for _, video in top_videos.iterrows():
            video_id = video['video_id']
            video_comments = df[df['video_id'] == video_id]
            
            sentiment_counts = video_comments['xlm_sentiment'].value_counts()
            sentiment_percentages = video_comments['xlm_sentiment'].value_counts(normalize=True) * 100
            
            sentiment_analysis[video_id] = {
                'video_title': video['video_title'],
                'search_query': video['search_query'],
                'engagement_score': video['engagement_score'],
                'total_comments': len(video_comments),
                'sentiment_counts': sentiment_counts.to_dict(),
                'sentiment_percentages': sentiment_percentages.to_dict(),
                'video_views': video['video_views']
            }
        
        return sentiment_analysis
    
    def analyze_identity_queries(self, df):
        """Analyze sentiment for specific identity-related queries."""
        
        print(f"\nAnalyzing identity-related queries:")
        
        identity_analysis = {}
        found_queries = []
        
        for query in self.identity_queries:
            query_comments = df[df['search_query'] == query]
            
            if len(query_comments) > 0:
                found_queries.append(query)
                
                sentiment_counts = query_comments['xlm_sentiment'].value_counts()
                sentiment_percentages = query_comments['xlm_sentiment'].value_counts(normalize=True) * 100
                
                identity_analysis[query] = {
                    'total_comments': len(query_comments),
                    'unique_videos': query_comments['video_id'].nunique(),
                    'sentiment_counts': sentiment_counts.to_dict(),
                    'sentiment_percentages': sentiment_percentages.to_dict(),
                    'learning_journeys': query_comments['learning_journey'].sum(),
                    'learning_journey_rate': query_comments['learning_journey'].mean() * 100 if len(query_comments) > 0 else 0
                }
                
                print(f"  ✓ {query}: {len(query_comments):,} comments")
            else:
                print(f"  ✗ {query}: No data found")
        
        return identity_analysis, found_queries
    
    def create_visualization(self, sentiment_analysis, identity_analysis):
        """Create focused visualization for engagement-sentiment analysis."""
        
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Video Engagement and Identity Query Sentiment Analysis', fontsize=14, fontweight='bold')
        
        # 1. Top 10 Videos by Engagement Score with Sentiment
        top_videos = list(sentiment_analysis.keys())[:10]
        engagement_scores = [sentiment_analysis[vid]['engagement_score'] for vid in top_videos]
        video_titles = [sentiment_analysis[vid]['video_title'][:40] + "..." if len(sentiment_analysis[vid]['video_title']) > 40 
                       else sentiment_analysis[vid]['video_title'] for vid in top_videos]
        
        # Create color based on dominant sentiment
        colors = []
        for vid in top_videos:
            sentiments = sentiment_analysis[vid]['sentiment_percentages']
            if sentiments.get('positive', 0) > 50:
                colors.append('#6BCF7F')  # Green for positive
            elif sentiments.get('negative', 0) > 30:
                colors.append('#FF6B6B')  # Red for negative
            else:
                colors.append('#FFD93D')  # Yellow for neutral
        
        bars1 = ax1.barh(range(len(top_videos)), engagement_scores, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(top_videos)))
        ax1.set_yticklabels(video_titles, fontsize=8)
        ax1.set_xlabel('Weighted Engagement Score (Like Rate + Comment Rate × 10)')
        ax1.set_title('Top 10 Videos by Engagement Score\n(Color = Dominant Sentiment)')
        ax1.invert_yaxis()
        
        # Add engagement score labels
        for i, bar in enumerate(bars1):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{engagement_scores[i]:.1f}', va='center', fontsize=8)
        
        # 2. Identity-Related Queries Sentiment Distribution
        if identity_analysis:
            queries = list(identity_analysis.keys())
            sentiment_data = []
            
            for query in queries:
                data = identity_analysis[query]
                negative = data['sentiment_percentages'].get('negative', 0)
                neutral = data['sentiment_percentages'].get('neutral', 0)
                positive = data['sentiment_percentages'].get('positive', 0)
                sentiment_data.append([negative, neutral, positive])
            
            sentiment_data = np.array(sentiment_data)
            query_labels = [q.replace('_', ' ').title() for q in queries]
            
            # Create stacked bar chart
            colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']  # Red, Yellow, Green
            bottom = np.zeros(len(query_labels))
            
            for i, (sentiment, color) in enumerate(zip(['Negative', 'Neutral', 'Positive'], colors)):
                bars = ax2.bar(query_labels, sentiment_data[:, i], bottom=bottom, 
                       color=color, alpha=0.8, label=sentiment)
                
                # Add percentage labels on each segment
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 5:  # Only show labels for segments > 5%
                        ax2.text(bar.get_x() + bar.get_width()/2, 
                                bottom[j] + height/2,
                                f'{height:.1f}%',
                                ha='center', va='center', fontsize=8)
                
                bottom += sentiment_data[:, i]
            
            ax2.set_ylabel('Sentiment Percentage')
            ax2.set_title('Sentiment Distribution - Identity-Related Queries')
            ax2.set_xticklabels(query_labels, rotation=45, ha='right', fontsize=9)
            ax2.legend()
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f'engagement_sentiment_analysis_{self.timestamp}.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Visualization saved: {viz_path.name}")
        return viz_path
    
    def generate_report(self, video_metrics, sentiment_analysis, identity_analysis, found_queries):
        """Generate comprehensive report."""
        
        report_lines = [
            "VIDEO ENGAGEMENT AND IDENTITY SENTIMENT ANALYSIS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "METHODOLOGY:",
            "• Weighted Engagement Score = Like Rate + (Comment Rate × 10)",
            "• Minimum view threshold: 1000 views (consistent with other analysis scripts)",
            "• Sentiment analysis using XLM-RoBERTa Enhanced (74% accuracy)",
            "• Identity queries based on actual dataset search terms",
            "",
            f"DATASET OVERVIEW:",
            f"• Total videos analyzed: {len(video_metrics):,}",
            f"• Identity queries found: {len(found_queries)} of {len(self.identity_queries)} searched",
            "",
            "TOP 15 VIDEOS BY ENGAGEMENT SCORE:",
            "=" * 80
        ]
        
        # Top videos analysis
        for i, (video_id, data) in enumerate(list(sentiment_analysis.items())[:15], 1):
            pos = data['sentiment_percentages'].get('positive', 0)
            neu = data['sentiment_percentages'].get('neutral', 0)
            neg = data['sentiment_percentages'].get('negative', 0)
            
            report_lines.extend([
                f"{i}. {data['video_title']}",
                f"   Video ID: {video_id}",
                f"   Search Query: {data['search_query']}",
                f"   Weighted Engagement Score: {data['engagement_score']:.1f}",
                f"   Total Comments: {data['total_comments']:,} | Views: {data['video_views']:,}",
                f"   Sentiment: Positive {pos:.1f}% | Neutral {neu:.1f}% | Negative {neg:.1f}%",
                ""
            ])
        
        # Identity queries analysis
        report_lines.extend([
            "",
            "IDENTITY-RELATED QUERIES ANALYSIS:",
            "=" * 80,
            ""
        ])
        
        if identity_analysis:
            for query in found_queries:
                data = identity_analysis[query]
                pos = data['sentiment_percentages'].get('positive', 0)
                neu = data['sentiment_percentages'].get('neutral', 0)
                neg = data['sentiment_percentages'].get('negative', 0)
                
                report_lines.extend([
                    f"Query: {query.replace('_', ' ').title()}",
                    f"  • Total Comments: {data['total_comments']:,}",
                    f"  • Unique Videos: {data['unique_videos']:,}",
                    f"  • Learning Journeys: {data['learning_journeys']} ({data['learning_journey_rate']:.1f}%)",
                    f"  • Sentiment: Positive {pos:.1f}% | Neutral {neu:.1f}% | Negative {neg:.1f}%",
                    ""
                ])
        
        # Not found queries
        not_found = [q for q in self.identity_queries if q not in found_queries]
        if not_found:
            report_lines.extend([
                "QUERIES NOT FOUND IN DATASET:",
                ", ".join(not_found),
                ""
            ])
        
        # Key findings
        if identity_analysis:
            avg_pos = np.mean([data['sentiment_percentages'].get('positive', 0) for data in identity_analysis.values()])
            avg_neg = np.mean([data['sentiment_percentages'].get('negative', 0) for data in identity_analysis.values()])
            avg_lj = np.mean([data['learning_journey_rate'] for data in identity_analysis.values()])
            
            report_lines.extend([
                "",
                "KEY FINDINGS:",
                "=" * 80,
                f"• Average positive sentiment in identity queries: {avg_pos:.1f}%",
                f"• Average negative sentiment in identity queries: {avg_neg:.1f}%",
                f"• Average learning journey rate in identity queries: {avg_lj:.1f}%",
            ])
        
        report_lines.extend([
            "",
            "DATA SOURCES:",
            "• Enhanced comments: results/models/sentiment_analysis/xlm_roberta_clean_20250816_132139/enhanced_comments.csv",
            "• Analysis method: XLM-RoBERTa Enhanced with learning journey detection",
            "• Engagement metric: Weighted score (Like Rate + Comment Rate × 10) where comments valued 10x more than likes"
        ])
        
        # Save report
        report_path = self.output_dir / f'engagement_sentiment_report_{self.timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Report saved: {report_path.name}")
        return report_path
    
    def run_analysis(self):
        """Main analysis function."""
        print("\n" + "=" * 80)
        print("VIDEO ENGAGEMENT AND SENTIMENT ANALYSIS")
        print("=" * 80)
        
        # Load data
        df = self.load_enhanced_comments()
        
        # Calculate engagement scores
        print("\nCalculating engagement scores...")
        video_metrics = self.calculate_engagement_scores(df)
        
        # Analyze sentiment for top engagement videos
        print("\nAnalyzing sentiment for top engagement videos...")
        top_videos = video_metrics.head(15)
        sentiment_analysis = self.analyze_sentiment_by_video(df, top_videos)
        
        # Analyze identity-related queries
        identity_analysis, found_queries = self.analyze_identity_queries(df)
        
        # Create visualization
        print("\nCreating visualization...")
        viz_path = self.create_visualization(sentiment_analysis, identity_analysis)
        
        # Generate report
        print("\nGenerating report...")
        report_path = self.generate_report(video_metrics, sentiment_analysis, identity_analysis, found_queries)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Found {len(found_queries)} of {len(self.identity_queries)} identity queries in dataset")
        
        return viz_path, report_path

def main():
    """Main execution function."""
    analyzer = EngagementSentimentAnalyzer()
    viz_path, report_path = analyzer.run_analysis()
    
    print(f"\nGenerated files:")
    print(f"  • Visualization: {viz_path.name}")
    print(f"  • Report: {report_path.name}")

if __name__ == "__main__":
    main()