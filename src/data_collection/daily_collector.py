import pandas as pd
from datetime import datetime
import time
import glob
import os
from googleapiclient.discovery import build


class DailyYouTubeCollector:
    """Collect YouTube data in manageable daily chunks"""
    
    START_DATE = datetime(2020, 1, 1)
    END_DATE = datetime(2024, 12, 31)
    REGION_CODE = "GB"
    EDUCATIONAL_CATEGORIES = ['27', '28', '26']  # Education, Science & Tech, How-to & Style
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.existing_video_ids = set()
        
    def load_existing_video_ids(self):
        """Load existing video IDs to avoid duplicates"""
        csv_files = glob.glob("youtube_maths_data_*.csv")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'video_id' in df.columns:
                    self.existing_video_ids.update(df['video_id'].tolist())
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
        
        print(f"Loaded {len(self.existing_video_ids)} existing video IDs")
    
    def get_daily_keywords(self, day_number=1):
        """Get different keyword sets for each day to maximize variety"""
        
        # Base keywords
        base_keywords = {
            1: [  # Day 1: Core educational terms
                "maths GCSE", "math A Level", "mathematics tutorial", "maths lesson",
                "algebra explained", "geometry help", "calculus tutorial", "statistics lesson",
                "trigonometry explained", "arithmetic tutorial", "probability help",
                "maths revision", "math exam", "mathematics study", "maths tips"
            ],
            2: [  # Day 2: Sentiment and difficulty
                "maths difficult", "math hard", "mathematics easy", "maths anxiety",
                "math phobia", "maths boring", "math fun", "mathematics challenging",
                "maths frustrating", "math confusing", "maths useful", "math important",
                "why learn maths", "maths hate", "math love"
            ],
            3: [  # Day 3: UK education specific
                "year 11 maths", "year 10 math", "sixth form mathematics", "Key Stage maths",
                "AQA maths", "Edexcel mathematics", "OCR math",
                "university maths", "college mathematics", "maths degree"
            ],
            4: [  # Day 4: Teachers and careers
                "maths teacher", "math teacher", "mathematics instructor", "teaching maths",
                "math career", "maths job", "mathematician", "STEM education", "math for data science", "maths for data science",
                "maths pedagogy", "math classroom", "mathematics education"
            ],
            5: [  # Day 5: Real world and practical
                "real world maths", "practical mathematics", "everyday math", "maths in life",
                "applied mathematics", "math skills", "financial maths", "business math",
                "engineering maths", "physics mathematics"
            ],
            6: [  # Day 6: Specific topics
                "linear equations", "quadratic equations", "fractions tutorial", "decimals help",
                "percentages explained", "ratios lesson", "differentiation tutorial",
                "integration help", "matrices explained", "vectors tutorial"
            ],
            7: [  # Day 7: Social and cultural
                "women in maths", "girls and math", "diversity mathematics", "maths culture",
                "math stereotypes", "mathematics society", "maths history", "famous mathematicians",
                "math discoveries", "mathematics breakthroughs", "discalculia", "maths dyslexia", "maths accessibility", "maths for everyone"
            ]
        }
        
        # Cycle through days if day_number > 7
        day_key = ((day_number - 1) % 7) + 1
        return base_keywords.get(day_key, base_keywords[1])
    
    def search_videos_smart(self, query, max_results=50):
        """Smart search that only uses categories 26, 27, 28"""
        published_after = self.START_DATE.strftime('%Y-%m-%dT%H:%M:%SZ')
        published_before = self.END_DATE.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        base_params = {
            'part': 'id,snippet',
            'q': query,
            'type': 'video',
            'maxResults': min(max_results, 50),
            'regionCode': self.REGION_CODE,
            'relevanceLanguage': 'en',
            'publishedAfter': published_after,
            'publishedBefore': published_before
        }
        
        all_videos = []
        
        # Only search within categories 26, 27, 28
        for category_id in self.EDUCATIONAL_CATEGORIES:
            try:
                params = base_params.copy()
                params['videoCategoryId'] = category_id
                params['order'] = 'relevance'
                
                request = self.youtube.search().list(**params)
                response = request.execute()
                
                if response and 'items' in response:
                    all_videos.extend(response['items'])
            except Exception as e:
                print(f"Error searching category {category_id}: {e}")
        
        
        # Remove duplicates and filter existing videos
        seen_ids = set()
        new_videos = []
        for video in all_videos:
            # Check if video has proper structure
            if 'id' not in video or not isinstance(video['id'], dict) or 'videoId' not in video['id']:
                continue
                
            video_id = video['id']['videoId']
            if video_id not in seen_ids and video_id not in self.existing_video_ids:
                seen_ids.add(video_id)
                new_videos.append(video)
                if len(new_videos) >= max_results:
                    break
        
        return {'items': new_videos} if new_videos else None
    
    def enhance_videos(self, videos):
        """Add statistics to videos"""
        if not videos:
            return videos
            
        video_ids = [video['video_id'] for video in videos]
        
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i + 50]
            try:
                request = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(batch_ids)
                )
                details = request.execute()
                
                if details and 'items' in details:
                    details_lookup = {item['id']: item for item in details['items']}
                    
                    for video in videos[i:i + 50]:
                        video_id = video['video_id']
                        if video_id in details_lookup:
                            item = details_lookup[video_id]
                            stats = item.get('statistics', {})
                            
                            video.update({
                                'view_count': int(stats.get('viewCount', 0)),
                                'like_count': int(stats.get('likeCount', 0)),
                                'comment_count': int(stats.get('commentCount', 0)),
                                'duration': item.get('contentDetails', {}).get('duration', ''),
                                'category_id': item.get('snippet', {}).get('categoryId', ''),
                                'tags': item.get('snippet', {}).get('tags', [])
                            })
                            
            except Exception as e:
                print(f"Error enhancing videos: {e}")
            
            time.sleep(0.1)
        
        return videos
    
    def process_videos(self, search_results, query):
        """Extract video data from search results"""
        if not search_results or 'items' not in search_results:
            return []
            
        videos = []
        for item in search_results['items']:
            # Check if item has proper structure
            if 'id' not in item or not isinstance(item['id'], dict) or 'videoId' not in item['id']:
                continue
                
            video_data = {
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_title': item['snippet']['channelTitle'],
                'channel_id': item['snippet']['channelId'],
                'published_at': item['snippet']['publishedAt'],
                'thumbnail_url': item['snippet']['thumbnails']['default']['url'],
                'search_query': query,
                'collected_at': datetime.now().isoformat()
            }
            videos.append(video_data)
        
        return self.enhance_videos(videos)
    
    def collect_daily_batch(self, day_number=1, target_videos=1500):
        """Collect a daily batch of videos"""
        self.load_existing_video_ids()
        
        keywords = self.get_daily_keywords(day_number)
        
        print(f"\n=== DAY {day_number} COLLECTION ===")
        print(f"Target: {target_videos} videos")
        print(f"Keywords: {len(keywords)}")
        print(f"Existing videos to avoid: {len(self.existing_video_ids)}")
        
        all_videos = []
        videos_per_keyword = target_videos // len(keywords)
        
        for i, keyword in enumerate(keywords):
            if len(all_videos) >= target_videos:
                break
                
            print(f"Keyword {i+1}/{len(keywords)}: '{keyword}' (Collected: {len(all_videos)}/{target_videos})")
            
            search_results = self.search_videos_smart(keyword, max_results=videos_per_keyword)
            
            if search_results:
                videos = self.process_videos(search_results, keyword)
                if videos:
                    all_videos.extend(videos)
                    # Add to existing IDs to avoid duplicates in same session
                    for video in videos:
                        self.existing_video_ids.add(video['video_id'])
                    print(f"  → Found {len(videos)} new videos")
                else:
                    print(f"  → No new videos found")
            else:
                print(f"  → No results")
            
            time.sleep(1)  # Rate limiting
        
        # Save the daily batch
        if all_videos:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"youtube_maths_data_day_{day_number:02d}_{timestamp}.csv"
            
            df = pd.DataFrame(all_videos)
            df.to_csv(filename, index=False)
            
            print(f"\nDay {day_number} Complete!")
            print(f"   Collected: {len(all_videos)} videos")
            print(f"   Saved to: {filename}")
            print(f"   Unique videos: {df['video_id'].nunique()}")
            
            return len(all_videos)
        else:
            print(f"\nDay {day_number} - No videos collected")
            return 0


def estimate_daily_quota():
    """Estimate quota for daily collection"""
    keywords_per_day = 15
    searches_per_keyword = 3  # Only 3 categories now (26, 27, 28)
    videos_per_day = 1500
    
    search_quota = keywords_per_day * searches_per_keyword * 100
    detail_quota = videos_per_day / 50 * 1
    total_quota = search_quota + detail_quota
    
    print(f"\n=== DAILY QUOTA ESTIMATION ===")
    print(f"Keywords per day: {keywords_per_day}")
    print(f"Categories searched: 26, 27, 28 only")
    print(f"Search requests: {keywords_per_day * searches_per_keyword} (cost: {search_quota} units)")
    print(f"Detail requests: {videos_per_day / 50:.0f} (cost: {detail_quota:.0f} units)")
    print(f"Total daily quota: {total_quota:.0f} units")
    print(f"Daily quota limit: 10,000 units")
    print(f"Quota usage: {total_quota/10000*100:.1f}%")


def main():
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    if not API_KEY:
        print("Error: YOUTUBE_API_KEY environment variable not set")
        return
    
    collector = DailyYouTubeCollector(API_KEY)
    
    estimate_daily_quota()
    
    print(f"\n" + "="*50)
    print("DAILY COLLECTION STRATEGY")
    print("This will collect ~1,500 videos per day over 7 days = ~10,500 videos")
    print("Each day uses different keywords to maximize variety")
    print("="*50)
    
    day = input("\nWhich day would you like to collect? (1-7, or 'auto' for automatic): ").strip()
    
    if day.lower() == 'auto':
        # Collect multiple days automatically
        total_collected = 0
        for day_num in range(1, 8):
            print(f"\n{'='*20} STARTING DAY {day_num} {'='*20}")
            collected = collector.collect_daily_batch(day_num, target_videos=1500)
            total_collected += collected
            
            if day_num < 7:
                print(f"\nDay {day_num} complete. Total so far: {total_collected}")
                continue_choice = input("Continue to next day? (y/n): ").lower().strip()
                if continue_choice != 'y':
                    break
        
        print(f"\nCOLLECTION COMPLETE! Total videos: {total_collected}")
    
    elif day.isdigit() and 1 <= int(day) <= 7:
        day_num = int(day)
        collected = collector.collect_daily_batch(day_num, target_videos=1500)
        print(f"\nDay {day_num} complete! Collected {collected} videos")
    
    else:
        print("Invalid input. Please enter a number 1-7 or 'auto'")


if __name__ == "__main__":
    main()
