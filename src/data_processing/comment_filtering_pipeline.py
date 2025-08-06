#!/usr/bin/env python3
"""
Complete Comment Dataset Filtering Pipeline

This script creates a complete comment filtering pipeline that uses the output
from the complete video filtering pipeline as input. Follows the same 
methodology and structure as the video pipeline.

Processing Steps:
1. Load video-filtered comments dataset
2. Apply video ID filtering (restrict to complete pipeline videos)
3. Comment quality filtering (length, spam, language)
4. Content relevance filtering (educational mathematics focus)
5. Final anonymization and formatting

Expected Output: Clean comment dataset for topic modeling

"""

import pandas as pd
import re
import os
import hashlib
import html
import unicodedata
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class NonEnglishCommentFilter:
    """Filter for detecting and removing non-English comments"""
    
    def __init__(self):
        # Common non-English patterns
        self.non_english_patterns = [
            # Hindi/Devanagari scripts
            r'[\u0900-\u097F]+',
            # Arabic script  
            r'[\u0600-\u06FF]+',
            # Chinese/Japanese/Korean
            r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]+',
            # Cyrillic
            r'[\u0400-\u04FF]+',
            # Thai
            r'[\u0E00-\u0E7F]+',
            # Other major non-Latin scripts
            r'[\u0590-\u05FF\u0370-\u03FF\u1F00-\u1FFF]+',
        ]
        
        # Common transliterated patterns
        self.transliterated_patterns = [
            r'\b(aap|hai|hain|kar|karo|kya|nahin|nahi|matlab|samjh|dekh|bhi|toh|agar|lekin)\b',
            r'\b(bahut|accha|achha|theek|sahi|galat|bura|khush|padhiye|samajhiye)\b',
        ]
    
    def should_filter_comment(self, comment_text):
        """Check if comment should be filtered as non-English"""
        if pd.isna(comment_text) or not comment_text.strip():
            return True, "Empty comment"
        
        text = str(comment_text).strip()
        
        # Check for non-Latin scripts
        for pattern in self.non_english_patterns:
            if re.search(pattern, text):
                return True, "Non-English script detected"
        
        # Check for transliterated patterns
        text_lower = text.lower()
        for pattern in self.transliterated_patterns:
            if re.search(pattern, text_lower):
                return True, "Transliterated non-English detected"
        
        return False, "English content"

class CompleteCommentFilteringPipeline:
    """Complete comment filtering pipeline following video pipeline methodology"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stats = {}
        self.non_english_filter = NonEnglishCommentFilter()
        
        # Comment quality parameters
        self.MIN_COMMENT_LENGTH = 3      # Minimum meaningful comment length
        self.MAX_COMMENT_LENGTH = 2000   # Maximum comment length (remove extremely long)
        
        # Mathematics education keywords (focused on learning/teaching)
        self.educational_keywords = [
            # Core learning terms
            'learn', 'understand', 'explain', 'help', 'teach', 'tutorial', 'lesson', 'study',
            'practice', 'exercise', 'homework', 'assignment', 'exam', 'test', 'quiz',
            'revision', 'review', 'prepare', 'preparation',
            
            # Mathematics terms
            'math', 'maths', 'mathematics', 'mathematical', 'solve', 'solution', 'answer',
            'equation', 'formula', 'problem', 'question', 'method', 'technique', 'approach',
            'concept', 'theory', 'principle', 'rule', 'theorem', 'proof',
            
            # Educational levels
            'gcse', 'a level', 'a-level', 'university', 'college', 'school', 'grade',
            'year 7', 'year 8', 'year 9', 'year 10', 'year 11', 'year 12',
            'ks3', 'ks4', 'key stage', 'sixth form',
            
            # Subject areas
            'algebra', 'geometry', 'calculus', 'trigonometry', 'statistics', 'probability',
            'arithmetic', 'fractions', 'decimals', 'percentages', 'ratios',
            
            # Educational feedback
            'thanks', 'helpful', 'useful', 'great', 'excellent', 'amazing', 'brilliant',
            'clear', 'easy', 'difficult', 'hard', 'confusing', 'confused', 'stuck'
        ]
        
        # Spam/low-quality patterns
        self.spam_patterns = [
            r'\b(first|1st|second|2nd|early)\b.*\b(comment|here)\b',  # "First comment"
            r'^(nice|good|great|wow|amazing|cool)\.?\s*$',  # Single word reactions
            r'\b(subscribe|like|follow|share)\b',  # Self-promotion
            r'\b(fake|scam|bot|spam)\b',  # Spam accusations
            r'^[0-9\s\.\!\?]*$',  # Only numbers/punctuation
            r'^[a-zA-Z]\s*$',  # Single character
            r'(.)\1{4,}',  # Repeated characters (aaaaa, 11111)
        ]
    
    def log(self, message):
        if self.verbose:
            print(message)
    
    def _clean_comment_text(self, text):
        """Clean and normalize comment text"""
        if pd.isna(text):
            return ""
        
        # Unescape HTML entities
        text = html.unescape(str(text))
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters but keep basic punctuation
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        return text
    
    def _is_spam_comment(self, comment_text):
        """Check if comment is spam or low quality"""
        if pd.isna(comment_text) or not comment_text.strip():
            return True, "Empty comment"
        
        text = str(comment_text).strip().lower()
        
        # Check spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, f"Spam pattern: {pattern}"
        
        return False, "Not spam"
    
    def _has_educational_content(self, comment_text):
        """Check if comment contains educational content"""
        if pd.isna(comment_text) or not comment_text.strip():
            return False, []
        
        text = str(comment_text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation for keyword matching
        
        found_keywords = []
        for keyword in self.educational_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                found_keywords.append(keyword)
        
        return len(found_keywords) > 0, found_keywords
    
    def step1_load_comments_and_filter_by_videos(self, comments_file, videos_file):
        """Step 1: Load comments and filter by video pipeline output"""
        self.log("=" * 60)
        self.log("STEP 1: LOAD COMMENTS AND FILTER BY VIDEOS")
        self.log("=" * 60)
        
        # Load video pipeline output (approved videos)
        df_videos = pd.read_csv(videos_file)
        approved_video_ids = set(df_videos['video_id'])
        self.log(f"Approved videos from pipeline: {len(approved_video_ids):,}")
        
        # Load comments dataset
        df_comments = pd.read_csv(comments_file)
        self.stats['step1_original_comments'] = len(df_comments)
        self.log(f"Original comments: {len(df_comments):,}")
        
        # Filter comments to only include those from approved videos
        df_filtered = df_comments[df_comments['video_id'].isin(approved_video_ids)].copy()
        self.stats['step1_after_video_filter'] = len(df_filtered)
        
        removed_comments = len(df_comments) - len(df_filtered)
        unique_videos_before = df_comments['video_id'].nunique()
        unique_videos_after = df_filtered['video_id'].nunique()
        
        self.log(f"After video filtering: {len(df_filtered):,} comments")
        self.log(f"Removed: {removed_comments:,} comments from excluded videos")
        self.log(f"Videos: {unique_videos_after:,} (was {unique_videos_before:,})")
        
        retention = (len(df_filtered) / len(df_comments)) * 100 if len(df_comments) > 0 else 0
        self.log(f"Retention: {retention:.1f}%")
        
        self.log(f"Step 1 Complete: {len(df_filtered):,} comments")
        return df_filtered
    
    def step2_comment_quality_filter(self, df):
        """Step 2: Comment Quality Filtering"""
        self.log("\\n" + "=" * 60)
        self.log("STEP 2: COMMENT QUALITY FILTERING")
        self.log("=" * 60)
        
        before_count = len(df)
        df = df.reset_index(drop=True)
        self.log(f"Input comments: {before_count:,}")
        
        # Clean comment text
        self.log("🧼 Cleaning comment text...")
        df['comment_text_clean'] = df['comment_text'].apply(self._clean_comment_text)
        
        # Length filtering
        self.log(f"📏 Length filtering ({self.MIN_COMMENT_LENGTH}-{self.MAX_COMMENT_LENGTH} characters)...")
        df['comment_length'] = df['comment_text_clean'].str.len()
        
        length_mask = ((df['comment_length'] >= self.MIN_COMMENT_LENGTH) & 
                      (df['comment_length'] <= self.MAX_COMMENT_LENGTH))
        
        too_short = (df['comment_length'] < self.MIN_COMMENT_LENGTH).sum()
        too_long = (df['comment_length'] > self.MAX_COMMENT_LENGTH).sum()
        
        self.log(f"   Removing {too_short:,} comments < {self.MIN_COMMENT_LENGTH} chars")
        self.log(f"   Removing {too_long:,} comments > {self.MAX_COMMENT_LENGTH} chars")
        
        df = df[length_mask].copy()
        
        # Spam filtering
        self.log("Spam filtering...")
        spam_results = []
        for _, row in df.iterrows():
            is_spam, reason = self._is_spam_comment(row['comment_text_clean'])
            spam_results.append(is_spam)
        
        spam_mask = pd.Series(spam_results, index=df.index)
        spam_count = spam_mask.sum()
        
        self.log(f"   Removing {spam_count:,} spam/low-quality comments")
        df = df[~spam_mask].copy()
        
        self.stats['step2_after_quality'] = len(df)
        total_removed = before_count - len(df)
        retention = (len(df) / before_count) * 100 if before_count > 0 else 0
        
        self.log(f"📈 Quality filter results:")
        self.log(f"   Before: {before_count:,} comments")
        self.log(f"   After: {len(df):,} comments")
        self.log(f"   Removed: {total_removed:,} comments")
        self.log(f"   Retention: {retention:.1f}%")
        
        self.log(f"Step 2 Complete: {len(df):,} comments")
        return df
    
    def step3_language_filter(self, df):
        """Step 3: Language Filtering (English only)"""
        self.log("\\n" + "=" * 60)
        self.log("STEP 3: LANGUAGE FILTERING")
        self.log("=" * 60)
        
        before_count = len(df)
        df = df.reset_index(drop=True)
        self.log(f"Input comments: {before_count:,}")
        
        # Apply language filtering
        self.log("🌐 Filtering non-English comments...")
        
        filter_results = []
        for i, row in df.iterrows():
            if i % 5000 == 0:
                self.log(f"   Processed {i:,}/{len(df):,} comments...")
            
            should_filter, reason = self.non_english_filter.should_filter_comment(row['comment_text_clean'])
            filter_results.append(should_filter)
        
        should_filter_mask = pd.Series(filter_results, index=df.index)
        df_kept = df[~should_filter_mask].copy()
        df_removed = df[should_filter_mask].copy()
        
        self.stats['step3_after_language'] = len(df_kept)
        
        self.log(f"📈 Language filter results:")
        self.log(f"   Before: {before_count:,} comments")
        self.log(f"   English comments kept: {len(df_kept):,}")
        self.log(f"   Non-English removed: {len(df_removed):,}")
        
        retention = (len(df_kept) / before_count) * 100 if before_count > 0 else 0
        self.log(f"   Retention: {retention:.1f}%")
        
        self.log(f"Step 3 Complete: {len(df_kept):,} comments")
        return df_kept
    
    def step4_educational_content_filter(self, df):
        """Step 4: Educational Content Filtering"""
        self.log("\\n" + "=" * 60)
        self.log("STEP 4: EDUCATIONAL CONTENT FILTERING")
        self.log("=" * 60)
        
        before_count = len(df)
        df = df.reset_index(drop=True)
        self.log(f"Input comments: {before_count:,}")
        self.log(f"Filtering for educational mathematics content...")
        
        # Apply educational content filter
        educational_results = []
        keyword_matches = []
        
        for _, row in df.iterrows():
            has_educational, keywords = self._has_educational_content(row['comment_text_clean'])
            educational_results.append(has_educational)
            keyword_matches.append(keywords)
        
        educational_mask = pd.Series(educational_results, index=df.index)
        df_educational = df[educational_mask].copy()
        df_non_educational = df[~educational_mask].copy()
        
        # Add keyword information to educational comments
        df_educational['educational_keywords'] = [kw for kw, edu in zip(keyword_matches, educational_results) if edu]
        
        self.stats['step4_after_educational'] = len(df_educational)
        
        # Analyze keywords
        if len(df_educational) > 0:
            all_keywords = []
            for keywords in df_educational['educational_keywords']:
                all_keywords.extend(keywords)
            
            if all_keywords:
                top_keywords = Counter(all_keywords).most_common(5)
                self.log("Top educational keywords:")
                for keyword, count in top_keywords:
                    self.log(f"   {keyword}: {count:,} comments")
        
        self.log(f"📈 Educational content filter results:")
        self.log(f"   Before: {before_count:,} comments")
        self.log(f"   Educational comments: {len(df_educational):,}")
        self.log(f"   Non-educational removed: {len(df_non_educational):,}")
        
        retention = (len(df_educational) / before_count) * 100 if before_count > 0 else 0
        self.log(f"   Retention: {retention:.1f}%")
        
        self.log(f"Step 4 Complete: {len(df_educational):,} comments")
        return df_educational
    
    def step5_final_processing_and_anonymization(self, df):
        """Step 5: Final Processing and Anonymization"""
        self.log("\\n" + "=" * 60)
        self.log("STEP 5: FINAL PROCESSING AND ANONYMIZATION")
        self.log("=" * 60)
        
        before_count = len(df)
        self.log(f"Input comments: {before_count:,}")
        
        # Final text cleaning
        self.log("🧼 Final text cleaning...")
        df['comment_text_final'] = df['comment_text_clean'].apply(self._clean_comment_text)
        
        # Enhanced anonymization
        self.log("🔒 Enhanced anonymization...")
        
        # Update author anonymization to be more secure
        if 'author_anon' not in df.columns:
            self.log("   Creating author anonymization...")
            df['author_anon'] = df.apply(lambda row: 
                f"COMMENTER_{hashlib.sha256((str(row.get('comment_id', '')) + str(row.get('video_id', ''))).encode()).hexdigest()[:8].upper()}", 
                axis=1)
        
        # Add processing metadata
        df['processing_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['processing_version'] = 'complete_pipeline_v1.0'
        df['filtered_by_video_pipeline'] = True
        
        # Select final columns for output
        final_columns = [
            'video_id', 'comment_id', 'comment_text_final', 'like_count', 
            'published_at', 'is_reply', 'parent_id', 'video_title', 
            'video_views', 'video_likes', 'search_query', 'collection_date',
            'author_anon', 'processing_timestamp', 'processing_version',
            'filtered_by_video_pipeline', 'comment_length'
        ]
        
        # Keep only columns that exist
        available_columns = [col for col in final_columns if col in df.columns]
        df_final = df[available_columns].copy()
        
        # Rename for consistency
        df_final = df_final.rename(columns={'comment_text_final': 'comment_text'})
        
        self.stats['step5_final'] = len(df_final)
        
        self.log(f"Final processing results:")
        self.log(f"   Input comments: {before_count:,}")
        self.log(f"   Final dataset: {len(df_final):,} comments")
        self.log(f"   Final columns: {len(df_final.columns)}")
        
        # Dataset statistics
        video_count = df_final['video_id'].nunique()
        avg_comments_per_video = len(df_final) / video_count if video_count > 0 else 0
        
        self.log(f"Dataset statistics:")
        self.log(f"   Unique videos: {video_count:,}")
        self.log(f"   Average comments per video: {avg_comments_per_video:.1f}")
        
        if 'comment_length' in df_final.columns:
            length_stats = df_final['comment_length'].describe()
            self.log(f"   Comment length stats:")
            self.log(f"      Mean: {length_stats['mean']:.1f} characters")
            self.log(f"      Median: {length_stats['50%']:.1f} characters")
        
        self.log(f"Step 5 Complete: {len(df_final):,} comments")
        return df_final
    
    def run_complete_pipeline(self, comments_file, videos_file):
        """Run the complete comment filtering pipeline"""
        self.log("STARTING COMPLETE COMMENT FILTERING PIPELINE")
        self.log("=" * 80)
        
        start_time = datetime.now()
        
        # Execute all steps
        df = self.step1_load_comments_and_filter_by_videos(comments_file, videos_file)
        df = self.step2_comment_quality_filter(df)
        df = self.step3_language_filter(df)
        df = self.step4_educational_content_filter(df)
        df = self.step5_final_processing_and_anonymization(df)
        
        # Generate final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.log("\\n" + "=" * 80)
        self.log("COMPLETE COMMENT FILTERING PIPELINE COMPLETE!")
        self.log("=" * 80)
        
        self.log("PROCESSING STATISTICS:")
        self.log(f"   Original comments: {self.stats['step1_original_comments']:,}")
        self.log(f"   After video filtering: {self.stats['step1_after_video_filter']:,}")
        self.log(f"   After quality filtering: {self.stats['step2_after_quality']:,}")
        self.log(f"   After language filtering: {self.stats['step3_after_language']:,}")
        self.log(f"   After educational filtering: {self.stats['step4_after_educational']:,}")
        self.log(f"   Final dataset: {self.stats['step5_final']:,}")
        
        original = self.stats['step1_original_comments']
        final = self.stats['step5_final']
        self.log(f"\\nRETENTION ANALYSIS:")
        self.log(f"   Overall retention: {(final/original)*100:.1f}%")
        self.log(f"   Total removed: {original - final:,}")
        self.log(f"   Processing time: {duration}")
        
        # Save final dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"comments_complete_filtered_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        self.log(f"\\nFinal dataset saved: {output_file}")
        self.log(f"Final comment count: {len(df):,}")
        self.log(f"Ready for topic modeling analysis!")
        
        return df


def main():
    """Main execution function"""
    
    # File paths
    videos_file = "/Users/siradbihi/Desktop/MScDataScience/ERP Maths Sentiments/Video Datasets/Complete Pipeline/videos_gradual_complete_filtered_20250720_223652.csv"
    comments_file = "/Users/siradbihi/Desktop/MScDataScience/ERP Maths Sentiments/Comments Datasets/Consolidated script/No Reply with Final Language Filter/consolidated_comments_no_replies_20250708_214416_video_filtered_20250710_001758_minimal_filtered_20250717_232559.csv"
    
    # Verify input files exist
    if not os.path.exists(videos_file):
        print(f"ERROR: Video file not found: {videos_file}")
        return
    
    if not os.path.exists(comments_file):
        print(f"ERROR: Comments file not found: {comments_file}")
        return
    
    # Initialize and run pipeline
    pipeline = CompleteCommentFilteringPipeline(verbose=True)
    
    try:
        final_df = pipeline.run_complete_pipeline(comments_file, videos_file)
        
        # Generate report file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"comment_pipeline_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("COMPLETE COMMENT FILTERING PIPELINE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing completed: {datetime.now()}\n")
            f.write(f"Input files:\n")
            f.write(f"  Videos: {videos_file}\n")
            f.write(f"  Comments: {comments_file}\n\n")
            
            f.write("PROCESSING STATISTICS:\n")
            for key, value in pipeline.stats.items():
                f.write(f"  {key}: {value:,}\n")
            
            original = pipeline.stats['step1_original_comments']
            final = pipeline.stats['step5_final']
            f.write(f"\nRETENTION: {(final/original)*100:.1f}%\n")
            f.write(f"REMOVED: {original - final:,} comments\n")
            
            f.write(f"\nOUTPUT: comments_complete_filtered_{timestamp}.csv\n")
            f.write(f"FINAL COUNT: {len(final_df):,} comments\n")
        
        print(f"\nReport saved: {report_file}")
        print(f"Pipeline completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()