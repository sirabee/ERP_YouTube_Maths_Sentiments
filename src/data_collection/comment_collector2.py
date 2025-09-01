#!/usr/bin/env python3

import pandas as pd
import time
import os
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YouTubeCommentCollector:
    """Efficiently collect YouTube comments with quota management"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.quota_used = 0
        self.daily_quota_limit = 10000
        self.processed_video_ids = set()
        
    def load_processed_video_ids(self):
        """Load already processed video IDs from existing comment files"""
        comment_files = [f for f in os.listdir('.') if f.startswith('maths_video_comments_') and f.endswith('.csv')]
        
        for file in comment_files:
            try:
                # Read just the video_id column to save memory
                df = pd.read_csv(file, usecols=['video_id'])
                self.processed_video_ids.update(df['video_id'].unique())
                logger.info(f"Loaded {len(df['video_id'].unique())} processed videos from {file}")
            except Exception as e:
                logger.warning(f"Could not read {file}: {e}")
        
        logger.info(f"Total processed videos to skip: {len(self.processed_video_ids)}")
        
    def get_video_comments(self, video_id, max_comments=200):
        """Get comments for a single video with quota tracking"""
        comments = []
        next_page_token = None
        requests_made = 0
        
        try:
            while len(comments) < max_comments and requests_made < 5:
                self.quota_used += 1
                requests_made += 1
                
                response = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=next_page_token,
                    order='relevance'
                ).execute()
                
                for item in response['items']:
                    # Add top-level comment
                    snippet = item['snippet']['topLevelComment']['snippet']
                    comments.append(self._create_comment_dict(
                        video_id, item['id'], snippet, False, None,
                        item['snippet'].get('totalReplyCount', 0)
                    ))
                    
                    # Add replies (limit to 5 per comment)
                    if 'replies' in item and len(comments) < max_comments:
                        for reply in item['replies']['comments'][:5]:
                            if len(comments) >= max_comments:
                                break
                            comments.append(self._create_comment_dict(
                                video_id, reply['id'], reply['snippet'], True, item['id'], 0
                            ))
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
        except HttpError as e:
            if e.resp.status == 403:
                if 'commentsDisabled' in str(e):
                    logger.info(f"Comments disabled for video {video_id}")
                else:
                    logger.warning(f"Quota exceeded: {e}")
                    return None
            elif e.resp.status == 404:
                logger.info(f"Video {video_id} not found or private")
            else:
                logger.error(f"Error getting comments for {video_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {video_id}: {e}")
        
        return comments
    
    def _create_comment_dict(self, video_id, comment_id, snippet, is_reply, parent_id, reply_count):
        """Helper to create comment dictionary"""
        return {
            'video_id': video_id,
            'comment_id': comment_id,
            'author': snippet['authorDisplayName'],
            'text': snippet['textDisplay'],
            'like_count': snippet.get('likeCount', 0),
            'published_at': snippet['publishedAt'],
            'is_reply': is_reply,
            'parent_id': parent_id,
            'reply_count': reply_count
        }
    
    def process_video_batch(self, video_df, comments_per_video=200):
        """Process a batch of videos with quota management"""
        all_comments = []
        failed_videos = []
        processed_count = 0
        skipped_count = 0
        
        logger.info(f"Processing {len(video_df)} videos (max {comments_per_video} comments each)")
        
        for _, video_row in video_df.iterrows():
            # Check quota limit
            if self.quota_used >= self.daily_quota_limit - 100:
                logger.warning(f"Approaching quota limit. Processed {processed_count} videos.")
                break
            
            video_id = video_row['video_id']
            
            # Skip if already processed
            if video_id in self.processed_video_ids:
                skipped_count += 1
                continue
            
            comments = self.get_video_comments(video_id, comments_per_video)
            
            if comments is None:  # Quota exceeded
                logger.warning("Quota exceeded, stopping collection")
                break
            elif comments:
                # Add video metadata to each comment
                for comment in comments:
                    comment.update({
                        'video_title': video_row.get('title', ''),
                        'video_views': video_row.get('view_count', 0),
                        'video_likes': video_row.get('like_count', 0),
                        'video_duration': video_row.get('duration_minutes', 0),
                        'search_query': video_row.get('search_query', ''),
                        'collection_date': datetime.now().isoformat()
                    })
                
                all_comments.extend(comments)
                processed_count += 1
                
                # Progress logging and saving
                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count}/{len(video_df)} videos. "
                               f"Comments: {len(all_comments)}. Quota: {self.quota_used}/{self.daily_quota_limit}")
                    
                    if processed_count % 100 == 0 and all_comments:
                        temp_filename = self.save_progress(all_comments, "temp_progress")
                        logger.info(f"Progress saved to {temp_filename}")
            else:
                failed_videos.append(video_id)
            
            time.sleep(0.1)  # Rate limiting
        
        logger.info(f"Batch complete: {processed_count} videos, {skipped_count} skipped, {len(all_comments)} comments, {len(failed_videos)} failed")
        return all_comments, failed_videos, processed_count
    
    def save_progress(self, comments, filename_prefix="youtube_comments"):
        """Save comments to CSV with timestamp"""
        if not comments:
            logger.warning("No comments to save")
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        pd.DataFrame(comments).to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Saved {len(comments)} comments to {filename}")
        return filename

def estimate_daily_capacity():
    """Estimate how many videos can be processed daily for comprehensive collection"""
    print("\n=== YOUTUBE COMMENT COLLECTION CAPACITY ===")
    print("Daily API quota: 10,000 units | Comment request cost: 1 unit per request")
    print("Max results per request: 100 comment threads\n")
    
    # Comprehensive collection: 200 comments per video
    comments_per_video = 200
    avg_requests = max(1, (comments_per_video + 50) / 100)  # Account for replies and pagination
    videos_per_day = int(9500 / avg_requests)  # Leave 500 unit buffer
    
    print(f"Comprehensive collection (200 comments/video):")
    print(f"  - Cost per video: ~{avg_requests:.1f} units")
    print(f"  - Videos per day: ~{videos_per_day:,}")
    print(f"  - Days for 8,996 videos: {(8996 / videos_per_day):.1f}")
    print()

def get_user_input(df_length):
    """Get user preferences with input validation"""
    # Fixed at 200 comments per video for comprehensive collection
    comments_per_video = 200
    print(f"Using comprehensive collection: {comments_per_video} comments per video")
    
    # Batch size
    try:
        batch_input = input("Videos to process today (number or 'max' for all): ") or "1000"
        if batch_input.lower() == 'max':
            batch_size = df_length
        else:
            batch_size = min(int(batch_input), df_length)
    except ValueError:
        print("Invalid input, using default: 1000 videos")
        batch_size = min(1000, df_length)
    
    return comments_per_video, batch_size


def main():
    """Main execution function"""
    estimate_daily_capacity()
    
    # Get API key from environment
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("Error: YOUTUBE_API_KEY environment variable not set")
        return
    
    # Load video dataset
    video_file = 'youtube_maths_cleaned_final.csv'
    if not os.path.exists(video_file):
        print(f"Error: {video_file} not found")
        return
    
    df = pd.read_csv(video_file)
    print(f"\nLoaded {len(df)} videos from {video_file}")
    
    # Initialize collector
    collector = YouTubeCommentCollector(api_key)
    
    # Load already processed video IDs
    collector.load_processed_video_ids()
    
    # Get user preferences
    comments_per_video, batch_size = get_user_input(len(df))
    
    # Process videos
    batch_df = df.head(batch_size)
    comments, failed, processed = collector.process_video_batch(
        batch_df, 
        comments_per_video=comments_per_video
    )
    
    # Save results
    if comments:
        filename = collector.save_progress(comments, "maths_video_comments")
        print(f"\nCollection complete!")
        print(f"   Videos processed: {processed}")
        print(f"   Comments collected: {len(comments)}")
        print(f"   Failed videos: {len(failed)}")
        print(f"   Quota used: {collector.quota_used}/10,000")
        print(f"   Saved to: {filename}")
    else:
        print("\nNo comments collected")

if __name__ == "__main__":
    main()
