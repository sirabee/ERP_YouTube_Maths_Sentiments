#!/usr/bin/env python3
"""
Learning Journey Video Analysis
Identifies videos with the most learning journeys and extracts representative comments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

class LearningJourneyVideoAnalyzer:
    def __init__(self):
        """Initialize learning journey video analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "analysis" / "learning_journey_videos"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Learning Journey Video Analyzer Initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_enhanced_comments(self):
        """Load the enhanced comments dataset with learning journey annotations."""
        data_path = self.base_path / "results" / "models" / "sentiment_analysis" / "xlm_roberta_clean_20250816_132139" / "enhanced_comments.csv"
        
        print(f"Loading enhanced comments from: {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"Total comments loaded: {len(df):,}")
        print(f"Learning journey comments: {df['learning_journey'].sum():,} ({df['learning_journey'].mean()*100:.1f}%)")
        
        return df
    
    def analyze_videos_by_learning_journeys(self, df, top_n=15):
        """Analyze videos with the most learning journeys."""
        
        # Filter for learning journey comments only
        lj_comments = df[df['learning_journey'] == True].copy()
        
        # Group by video and count learning journeys
        video_stats = lj_comments.groupby(['video_id', 'video_title']).agg({
            'learning_journey': 'count',
            'search_query': 'first',
            'video_views': 'first',
            'video_likes': 'first'
        }).reset_index()
        
        video_stats.rename(columns={'learning_journey': 'learning_journey_count'}, inplace=True)
        
        # Sort by learning journey count
        video_stats = video_stats.sort_values('learning_journey_count', ascending=False)
        
        print(f"\nTop {top_n} videos with most learning journeys:")
        print("=" * 80)
        
        for i, row in video_stats.head(top_n).iterrows():
            print(f"{row.name + 1}. {row['video_title']}")
            print(f"   Video ID: {row['video_id']}")
            print(f"   Learning Journeys: {row['learning_journey_count']}")
            print(f"   Search Query: {row['search_query']}")
            print(f"   Views: {row['video_views']:,} | Likes: {row['video_likes']:,}")
            print()
        
        return video_stats.head(top_n)
    
    def extract_representative_comments(self, df, top_videos):
        """Extract representative learning journey comments for each top video."""
        
        representative_comments = {}
        
        for _, video in top_videos.iterrows():
            video_id = video['video_id']
            video_title = video['video_title']
            
            # Get learning journey comments for this video
            video_lj_comments = df[
                (df['video_id'] == video_id) & 
                (df['learning_journey'] == True)
            ].copy()
            
            # Sort by like_count and xlm_confidence to get representative examples
            video_lj_comments = video_lj_comments.sort_values(
                ['like_count', 'xlm_confidence'], 
                ascending=[False, False]
            )
            
            # Select top comments (up to 3 representative examples)
            top_comments = video_lj_comments.head(3)
            
            representative_comments[video_id] = {
                'video_title': video_title,
                'learning_journey_count': video['learning_journey_count'],
                'search_query': video['search_query'],
                'comments': []
            }
            
            for _, comment in top_comments.iterrows():
                comment_data = {
                    'text': comment['comment_text'],
                    'likes': comment['like_count'],
                    'confidence': comment['xlm_confidence'],
                    'progression_type': comment['progression_type'],
                    'published_at': comment['published_at']
                }
                representative_comments[video_id]['comments'].append(comment_data)
        
        return representative_comments
    
    def generate_report(self, top_videos, representative_comments):
        """Generate a comprehensive report of learning journey videos and comments."""
        
        report_lines = [
            "LEARNING JOURNEY VIDEO ANALYSIS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "OVERVIEW:",
            f"• Analysis based on XLM-RoBERTa Enhanced sentiment analysis (74% accuracy)",
            f"• Total videos analyzed: {len(top_videos)}",
            f"• Learning journey detection method: Negative-to-positive sentiment progression",
            f"• Comments analyzed using domain-trained YouTube sentiment model",
            "",
            "VIDEOS WITH MOST LEARNING JOURNEYS:",
            "=" * 80,
            ""
        ]
        
        for i, (video_id, video_data) in enumerate(representative_comments.items(), 1):
            report_lines.extend([
                f"{i}. {video_data['video_title']}",
                f"   Video ID: {video_id}",
                f"   Search Query: {video_data['search_query']}",
                f"   Learning Journeys Detected: {video_data['learning_journey_count']}",
                "",
                "   REPRESENTATIVE LEARNING JOURNEY COMMENTS:",
                "   " + "-" * 50
            ])
            
            for j, comment in enumerate(video_data['comments'], 1):
                # Truncate very long comments for readability
                comment_text = comment['text']
                if len(comment_text) > 300:
                    comment_text = comment_text[:297] + "..."
                
                report_lines.extend([
                    f"   Comment {j}:",
                    f"   \"{comment_text}\"",
                    f"   • Likes: {comment['likes']} | Confidence: {comment['confidence']:.3f}",
                    f"   • Progression Type: {comment['progression_type']}",
                    f"   • Published: {comment['published_at'][:10]}",
                    ""
                ])
            
            report_lines.append("")
        
        # Add methodology section
        report_lines.extend([
            "",
            "METHODOLOGY NOTES:",
            "=" * 80,
            "• Learning Journey Detection: Comments showing negative-to-positive sentiment progression",
            "• XLM-RoBERTa Model: AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual",
            "• Validation: 74% agreement with manual annotations (200 sample validation)",
            "• Domain Training Impact: +36 percentage points over Twitter-trained models",
            "• Progression Types:",
            "  - simple_transition: Basic negative-to-positive change",
            "  - complex_progression: Multiple sentiment transitions within comment",
            "  - learning_journey: Explicit educational breakthrough patterns",
            "",
            "• Representative Comment Selection:",
            "  - Sorted by like_count (community validation) and model confidence",
            "  - Up to 3 examples per video for readability",
            "  - Comments truncated at 300 characters for report formatting",
            "",
            "SOURCE DATA:",
            "• File: results/models/sentiment_analysis/xlm_roberta_clean_20250816_132139/enhanced_comments.csv",
            "• Total Comments: 35,438",
            "• Learning Journey Comments: 5,620 (15.86%)",
            "• Analysis Date: August 16, 2025"
        ])
        
        # Save report
        report_path = self.output_dir / f"learning_journey_videos_report_{self.timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✓ Learning journey video report saved: {report_path.name}")
        return report_path
    
    def run_analysis(self):
        """Main analysis function."""
        print("\n" + "=" * 80)
        print("LEARNING JOURNEY VIDEO ANALYSIS")
        print("=" * 80)
        
        # Load data
        df = self.load_enhanced_comments()
        
        # Analyze videos by learning journey count
        print("\nAnalyzing videos with most learning journeys...")
        top_videos = self.analyze_videos_by_learning_journeys(df, top_n=15)
        
        # Extract representative comments
        print("\nExtracting representative learning journey comments...")
        representative_comments = self.extract_representative_comments(df, top_videos)
        
        # Generate comprehensive report
        print("\nGenerating comprehensive report...")
        report_path = self.generate_report(top_videos, representative_comments)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Report saved: {report_path}")
        
        return report_path

def main():
    """Main execution function."""
    analyzer = LearningJourneyVideoAnalyzer()
    report_path = analyzer.run_analysis()
    
    print(f"\nGenerated report: {report_path.name}")

if __name__ == "__main__":
    main()