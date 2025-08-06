#!/usr/bin/env python3
"""
Complete Video Dataset Filtering Pipeline - Optimized Implementation

This script implements the complete video filtering approach:

1. Initial Preprocessing: Category filtering (26,27,28) + quality filtering
2. Math Keywords Filter: Required mathematical terminology validation  
3. BYPASSED: Targeted Filter (not used in actual pipeline)
4. Consolidated Processing: Category 26 removal + duration/quality filters
5. Non-Latin Script Filter: Removed videos with non-Latin script

Expected Output: ~3,897 videos (optimized pipeline approach)

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

class NonLatinScriptFilter:
    """Filter for detecting and removing videos with non-Latin script"""
    
    def __init__(self):
        # Define major non-Latin script blocks
        self.non_latin_script_blocks = {
            'Arabic': [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
            'Chinese': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x30000, 0x3134F)],
            'Japanese': [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0x1B000, 0x1B0FF), (0x1B100, 0x1B12F)],
            'Korean': [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xD7B0, 0xD7FF)],
            'Thai': [(0x0E00, 0x0E7F)], 'Vietnamese': [(0x1EA0, 0x1EFF)],
            'Hindi/Devanagari': [(0x0900, 0x097F), (0xA8E0, 0xA8FF)], 'Bengali': [(0x0980, 0x09FF)],
            'Tamil': [(0x0B80, 0x0BFF)], 'Telugu': [(0x0C00, 0x0C7F)], 'Kannada': [(0x0C80, 0x0CFF)],
            'Malayalam': [(0x0D00, 0x0D7F)], 'Gujarati': [(0x0A80, 0x0AFF)], 'Punjabi': [(0x0A00, 0x0A7F)],
            'Marathi': [(0x0900, 0x097F)], 'Urdu': [(0x0600, 0x06FF)],
            'Greek': [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
            'Cyrillic': [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
            'Hebrew': [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)], 'Armenian': [(0x0530, 0x058F)],
            'Georgian': [(0x10A0, 0x10FF)], 'Ethiopic': [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF)]
        }
        
        # Create Unicode range mapping
        self.unicode_range_to_script = {}
        for script_name, ranges in self.non_latin_script_blocks.items():
            for start, end in ranges:
                for code_point in range(start, end + 1):
                    self.unicode_range_to_script[code_point] = script_name
        
        # Mathematical symbols and currency that should be allowed
        self.allowed_non_latin_chars = {
            '±', '×', '÷', '√', '∑', '∫', '∞', '∂', '∆', '∇', '∏', '∪', '∩', '∈', '∉', '⊂', '⊃',
            '≤', '≥', '≠', '≈', '≡', '∝', '→', '←', '↑', '↓', '↔', '⇒', '⇔', '∀', '∃', '∧', '∨', '¬',
            '€', '£', '¥', '₹', '₽', '₨', '₩', '₪', '₫', '°', '′', '″', '‰', '‱', '℃', '℉',
            '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁺', '⁻', '⁼', '⁽', '⁾',
            '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉', '₊', '₋', '₌', '₍', '₎',
            '½', '⅓', '⅔', '¼', '¾', '⅕', '⅖', '⅗', '⅘', '⅙', '⅚', '⅛', '⅜', '⅝', '⅞'
        }
        
        # Extended Latin characters (accented) that should be allowed
        self.allowed_extended_latin = set()
        for i in range(0x0080, 0x024F):
            char = chr(i)
            if unicodedata.category(char)[0] == 'L':
                self.allowed_extended_latin.add(char)
    
    def detect_non_latin_script(self, text):
        """Detect non-Latin script in text"""
        if not text or pd.isna(text):
            return {'has_non_latin': False, 'non_latin_chars': [], 'scripts_detected': set(), 'non_latin_ratio': 0.0, 'char_count': 0}
        
        text = str(text).strip()
        if not text:
            return {'has_non_latin': False, 'non_latin_chars': [], 'scripts_detected': set(), 'non_latin_ratio': 0.0, 'char_count': 0}
        
        non_latin_chars, scripts_detected, total_chars = [], set(), 0
        
        for char in text:
            if char.isspace() or char in '.,;:!?()[]{}"\'-_=+*/<>|\\@#$%^&`~' or char.isdigit():
                continue
            total_chars += 1
            
            if 'A' <= char <= 'Z' or 'a' <= char <= 'z':
                continue
            if char in self.allowed_extended_latin or char in self.allowed_non_latin_chars:
                continue
                
            char_code = ord(char)
            if char_code in self.unicode_range_to_script:
                script_name = self.unicode_range_to_script[char_code]
                non_latin_chars.append({'char': char, 'script': script_name, 'unicode': f'U+{char_code:04X}'})
                scripts_detected.add(script_name)
        
        return {
            'has_non_latin': len(non_latin_chars) > 0,
            'non_latin_chars': non_latin_chars,
            'scripts_detected': scripts_detected,
            'non_latin_ratio': len(non_latin_chars) / max(total_chars, 1),
            'char_count': total_chars
        }
    
    def analyze_video_metadata(self, title, description, tags):
        """Analyze video metadata for non-Latin script"""
        results = {
            'title': self.detect_non_latin_script(title),
            'description': self.detect_non_latin_script(description),
            'tags': self.detect_non_latin_script(tags)
        }
        
        all_scripts, total_non_latin_chars, total_chars, has_any_non_latin = set(), 0, 0, False
        
        for field_result in results.values():
            if field_result['has_non_latin']:
                has_any_non_latin = True
                all_scripts.update(field_result['scripts_detected'])
                total_non_latin_chars += len(field_result['non_latin_chars'])
            total_chars += field_result['char_count']
        
        results['combined'] = {
            'has_non_latin': has_any_non_latin,
            'scripts_detected': all_scripts,
            'non_latin_ratio': total_non_latin_chars / max(total_chars, 1),
            'total_non_latin_chars': total_non_latin_chars,
            'total_chars': total_chars
        }
        
        return results
    
    def should_filter_video(self, title, description, tags, min_non_latin_chars=1, max_non_latin_ratio=0.05, strict_mode=False):
        """Determine if video should be filtered based on non-Latin script"""
        analysis = self.analyze_video_metadata(title, description, tags)
        
        if strict_mode:
            min_non_latin_chars = max(1, min_non_latin_chars // 2)
            max_non_latin_ratio = max(0.01, max_non_latin_ratio / 2)
        
        combined = analysis['combined']
        should_filter, reason = False, None
        
        if combined['has_non_latin']:
            if combined['total_non_latin_chars'] >= min_non_latin_chars:
                should_filter = True
                reason = f"Contains {combined['total_non_latin_chars']} non-Latin characters"
            if combined['non_latin_ratio'] > max_non_latin_ratio:
                should_filter = True
                reason = f"Non-Latin character ratio ({combined['non_latin_ratio']:.1%}) exceeds threshold ({max_non_latin_ratio:.1%})"
        
        if should_filter and combined['scripts_detected']:
            scripts_str = ', '.join(sorted(combined['scripts_detected']))
            reason += f" (Scripts: {scripts_str})"
        
        return {
            'should_filter': should_filter,
            'reason': reason or 'No non-Latin script detected',
            'analysis': analysis
        }


class CompleteVideoFilteringPipeline:
    """Complete video filtering pipeline implementation"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stats = {
            'step1_original': 0, 'step1_after_category': 0, 'step1_after_quality': 0,
            'step2_after_keywords': 0, 'step4_after_consolidated': 0, 'step5_final': 0
        }
        
        # Math keywords for educational content validation
        self.math_keywords = [
            'math', 'maths', 'mathematics', 'mathematical', 'mathematician',
            'algebra', 'geometry', 'calculus', 'trigonometry', 'statistics', 'arithmetic', 'probability', 'equation', 'formula',
            'gcse', 'a level', 'a-level', 'tutorial', 'lesson', 'learn', 'study', 'revision', 'exam', 'explained', 'help', 'teaching',
            'fractions', 'decimals', 'percentages', 'ratios', 'linear', 'quadratic', 'dyscalculia', 'discalculia',
            'differentiation', 'integration', 'derivative', 'matrix', 'vector',
            'year 7', 'year 8', 'year 9', 'year 10', 'year 11', 'year 12', 'sixth form', 'university', 'ks3', 'ks4', 'key stage',
            'aqa', 'edexcel', 'ocr', 'cambridge', 'stem', 'engineering', 'physics', 'data science'
        ]
        
        # Educational categories (27=Education, 28=Science & Technology)  
        # Category 26 will be removed in consolidation step
        self.educational_categories = [27, 28]
        self.non_latin_filter = NonLatinScriptFilter()
        
    def log(self, message):
        if self.verbose:
            print(message)
    
    def _clean_and_process_text(self, text):
        """Clean HTML and normalize text"""
        if pd.isna(text):
            return ""
        text = html.unescape(str(text))
        if text.startswith('[') and text.endswith(']'):
            text = text.strip('[]').replace("'", "").replace('"', '')
        return text
    
    def _duration_to_minutes_format(self, duration_str):
        """Convert PT47M10S to 47.10 format"""
        if pd.isna(duration_str) or not str(duration_str).startswith('PT'):
            return 0.00
        
        try:
            match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', str(duration_str))
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                total_minutes = hours * 60 + minutes
                return float(f"{total_minutes}.{seconds:02d}")
        except:
            pass
        return 0.00
    
    def _duration_to_actual_minutes(self, duration_minutes):
        """Convert 47.10 format to actual minutes (47 + 10/60)"""
        if pd.isna(duration_minutes) or duration_minutes == 0:
            return 0.0
        try:
            minutes_part = int(duration_minutes)
            seconds_part = int((duration_minutes - minutes_part) * 100)
            return minutes_part + (seconds_part / 60.0)
        except:
            return 0.0
    
    def _contains_math_keywords(self, title, tags, description):
        """Check if video contains math keywords"""
        text = f"{str(title).lower()} {str(tags).lower()} {str(description).lower()}"
        text = re.sub(r'[^\w\s]', ' ', text)
        
        matched = []
        for keyword in self.math_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                matched.append(keyword)
        
        return len(matched) > 0, matched
    
    def step1_initial_preprocessing(self, input_file):
        """Step 1: Initial Preprocessing"""
        self.log("=" * 60)
        self.log("STEP 1: INITIAL PREPROCESSING")
        self.log("=" * 60)
        
        df = pd.read_csv(input_file)
        self.stats['step1_original'] = len(df)
        self.log(f"Original videos: {len(df):,}")
        
        # Category filtering: Keep 26, 27, 28
        target_categories = [26, 27, 28]
        videos_to_remove = df[~df['category_id'].isin(target_categories)]
        if len(videos_to_remove) > 0:
            self.log(f"Removing {len(videos_to_remove):,} videos from non-target categories")
        
        df = df[df['category_id'].isin(target_categories)].copy()
        self.stats['step1_after_category'] = len(df)
        
        # Quality filtering: Remove spam
        spam_patterns = [r'\b(cheat|generator|free.*download)\b', r'\$\$\$', r'\b(click.*link|subscribe.*now)\b']
        spam_title = df['title'].str.contains('|'.join(spam_patterns), case=False, na=False)
        spam_count = spam_title.sum()
        
        if spam_count > 0:
            self.log(f"Removing {spam_count} spam videos")
            df = df[~spam_title].copy()
        
        self.stats['step1_after_quality'] = len(df)
        
        # Add duration and anonymize
        df['duration_minutes'] = df['duration'].apply(self._duration_to_minutes_format)
        df['channel_title_anon'] = df['channel_title'].apply(lambda x: f"CHANNEL_{hashlib.sha256(str(x).encode()).hexdigest()[:8].upper()}" if pd.notna(x) else "CHANNEL_UNKNOWN")
        df['channel_id_anon'] = df['channel_id'].apply(lambda x: f"ID_{hashlib.sha256(str(x).encode()).hexdigest()[:8].upper()}" if pd.notna(x) else "ID_UNKNOWN")
        df = df.drop(['channel_title', 'channel_id'], axis=1)
        
        self.log(f"Step 1 Complete: {len(df):,} videos")
        return df
    
    def step2_math_keywords_filter(self, df):
        """Step 2: Math Keywords Filter"""
        self.log("\n" + "=" * 60)
        self.log("STEP 2: MATH KEYWORDS FILTER")
        self.log("=" * 60)
        
        # Reset index to avoid indexing issues
        df = df.reset_index(drop=True)
        
        has_keywords_mask = []
        all_keywords = []
        
        for idx, row in df.iterrows():
            has_keywords, matched = self._contains_math_keywords(row['title'], row.get('tags', ''), row.get('description', ''))
            has_keywords_mask.append(has_keywords)
            if has_keywords:
                all_keywords.extend(matched)
        
        relevant_videos = df[has_keywords_mask].copy()
        
        self.stats['step2_after_keywords'] = len(relevant_videos)
        self.log(f"Relevant videos: {len(relevant_videos):,} ({len(relevant_videos)/len(df)*100:.1f}%)")
        
        if len(relevant_videos) > 0:
            top_keywords = Counter(all_keywords).most_common(5)
            self.log("Most common keywords:")
            for keyword, count in top_keywords:
                self.log(f"  {keyword}: {count:,}")
        
        self.log(f"Step 2 Complete: {len(relevant_videos):,} videos")
        return relevant_videos
    
    def step4_integrated_notebook_processing(self, df):
        """Step 4: Integrated Processing (matches notebook's exact approach)"""
        self.log("\n" + "=" * 60)
        self.log("STEP 4: INTEGRATED PROCESSING (NOTEBOOK APPROACH)")
        self.log("=" * 60)
        
        before_count = len(df)
        self.log(f"Input videos: {before_count:,}")
        
        # 1. Category filtering (exclude Category 26) - matches notebook
        self.log("Applying category filter (exclude Category 26)...")
        df['category_id'] = df['category_id'].astype(str)
        
        category_counts = df['category_id'].value_counts()
        self.log("Current category distribution:")
        for cat, count in category_counts.head(10).items():
            status = "KEEP" if cat in ['27', '28'] else "EXCLUDE"
            if cat == "26":
                status = "EXCLUDE (How-to)"
            self.log(f"   {status} Category {cat}: {count:,} videos")
        
        category_26_count = (df['category_id'] == '26').sum()
        df = df[df['category_id'].isin(['27', '28'])].copy()
        self.log(f"Removed Category 26: {category_26_count:,} videos")
        
        # 2. Duration conversion and filtering - matches notebook
        self.log("Converting duration format...")
        df['duration_actual_minutes'] = df['duration_minutes'].apply(self._duration_to_actual_minutes)
        
        MIN_DURATION, MAX_DURATION = 0.5, 120
        self.log(f"Duration filter: {MIN_DURATION}-{MAX_DURATION} minutes")
        
        duration_mask = (df['duration_actual_minutes'] >= MIN_DURATION) & (df['duration_actual_minutes'] <= MAX_DURATION)
        too_short = (df['duration_actual_minutes'] < MIN_DURATION).sum()
        too_long = (df['duration_actual_minutes'] > MAX_DURATION).sum()
        
        self.log(f"Removing {too_short:,} videos < {MIN_DURATION} min")
        self.log(f"Removing {too_long:,} videos > {MAX_DURATION} min")
        df = df[duration_mask].copy()
        
        # 3. Spam filtering - matches notebook
        self.log("Applying spam detection...")
        spam_pattern = r'\b(cheat|generator|free.*download|hack|\$\$\$|click.*link)\b'
        spam_mask = df['title'].str.contains(spam_pattern, case=False, na=False, regex=True)
        spam_count = spam_mask.sum()
        self.log(f"Removing {spam_count:,} spam videos")
        df = df[~spam_mask].copy()
        
        # 4. Mathematics content filtering - from notebook
        self.log("Applying mathematics content filter...")
        self.log(f"Checking for {len(self.math_keywords)} mathematics keywords")
        
        def has_math_content(row):
            """Check for mathematics keywords (notebook approach)"""
            fields = ['title', 'description', 'tags', 'search_query']
            texts = []
            for field in fields:
                if field in row and pd.notna(row[field]):
                    texts.append(str(row[field]))
            
            combined_text = ' '.join(texts).lower()
            
            for keyword in self.math_keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', combined_text):
                    return True
            return False
        
        # Apply math content filter
        math_mask = df.apply(lambda row: has_math_content(row), axis=1)
        content_excluded = (~math_mask).sum()
        self.log(f"Removing {content_excluded:,} non-mathematical videos")
        df = df[math_mask].copy()
        
        # 5. Text cleaning - matches notebook
        self.log("Cleaning text fields...")
        for col in ['title', 'description', 'tags']:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_and_process_text)
        
        # 6. Channel anonymization - matches notebook
        self.log("Anonymizing channel information...")
        if 'channel_title' in df.columns:
            df['channel_title_anon'] = df['channel_title'].apply(lambda x: f"CHANNEL_{hashlib.sha256(str(x).encode()).hexdigest()[:8].upper()}" if pd.notna(x) else "UNKNOWN")
            df = df.drop(columns=['channel_title'], errors='ignore')
        
        if 'channel_id' in df.columns:
            df['channel_id_anon'] = df['channel_id'].apply(lambda x: f"CHANNEL_{hashlib.sha256(str(x).encode()).hexdigest()[:8].upper()}" if pd.notna(x) else "UNKNOWN")
            df = df.drop(columns=['channel_id'], errors='ignore')
        
        self.stats['step4_after_consolidated'] = len(df)
        total_removed = before_count - len(df)
        
        self.log(f"\nIntegrated processing results:")
        self.log(f"   Before: {before_count:,} videos")
        self.log(f"   After: {len(df):,} videos")
        self.log(f"   Total removed: {total_removed:,} videos")
        self.log(f"   Retention: {(len(df)/before_count)*100:.1f}%")
        
        self.log(f"Step 4 Complete: {len(df):,} videos")
        return df
    
    def step5_non_latin_script_filter(self, df):
        """Step 5: Non-Latin Script Filter"""
        self.log("\n" + "=" * 60)
        self.log("STEP 5: NON-LATIN SCRIPT FILTER")
        self.log("=" * 60)
        
        # Reset index to avoid indexing issues
        df = df.reset_index(drop=True)
        
        filter_results = []
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 1000 == 0:
                self.log(f"Processed {i}/{len(df)} videos...")
            
            result = self.non_latin_filter.should_filter_video(
                row.get('title', ''), row.get('description', ''), row.get('tags', ''),
                min_non_latin_chars=1, max_non_latin_ratio=0.05, strict_mode=False
            )
            filter_results.append(result)
        
        should_filter_mask = [r['should_filter'] for r in filter_results]
        df_kept = df[~pd.Series(should_filter_mask, index=df.index)].copy()
        df_removed = df[pd.Series(should_filter_mask, index=df.index)].copy()
        
        self.stats['step5_final'] = len(df_kept)
        
        self.log(f"Processed: {len(df):,}, Removed: {len(df_removed):,}, Retained: {len(df_kept):,}")
        
        if len(df_removed) > 0:
            reasons = [r['reason'] for r in filter_results if r['should_filter']]
            reason_counts = Counter(reasons)
            self.log("Top removal reasons:")
            for reason, count in reason_counts.most_common(3):
                self.log(f"  {reason}: {count} videos")
        
        self.log(f"Step 5 Complete: {len(df_kept):,} videos")
        return df_kept
    
    def run_complete_pipeline(self, input_file):
        """Run the complete filtering pipeline"""
        self.log("STARTING COMPLETE VIDEO FILTERING PIPELINE")
        self.log("=" * 80)
        
        start_time = datetime.now()
        
        # Execute all steps (step 3 is bypassed in this implementation)
        df = self.step1_initial_preprocessing(input_file)
        df = self.step2_math_keywords_filter(df)
        # BYPASSED: step3_targeted_filter - not used in actual pipeline
        df = self.step4_integrated_notebook_processing(df)
        df = self.step5_non_latin_script_filter(df)
        
        # Generate final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.log("\n" + "=" * 80)
        self.log("COMPLETE FILTERING PIPELINE FINISHED!")
        self.log("=" * 80)
        
        self.log("PROCESSING STATISTICS:")
        self.log(f"   Original videos: {self.stats['step1_original']:,}")
        self.log(f"   After preprocessing: {self.stats['step1_after_quality']:,}")
        self.log(f"   After keywords: {self.stats['step2_after_keywords']:,}")
        self.log(f"   BYPASSED: Targeted filter (not used in pipeline)")
        self.log(f"   After consolidated: {self.stats['step4_after_consolidated']:,}")
        self.log(f"   Final dataset: {self.stats['step5_final']:,}")
        
        original = self.stats['step1_original']
        final = self.stats['step5_final']
        self.log(f"\nRETENTION ANALYSIS:")
        self.log(f"   Overall retention: {(final/original)*100:.1f}%")
        self.log(f"   Total removed: {original - final:,}")
        self.log(f"   Processing time: {duration}")
        
        # Save final dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"videos_complete_filtered_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        self.log(f"\nFinal dataset saved: {output_file}")
        self.log(f"Final video count: {len(df):,}")
        
        return df


def main():
    """Main execution function"""
    input_file = "/Users/siradbihi/Desktop/MScDataScience/ERP Maths Sentiments/Video Datasets/youtube_maths_data_merged_20250623_172638.csv"
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    pipeline = CompleteVideoFilteringPipeline(verbose=True)
    
    try:
        final_df = pipeline.run_complete_pipeline(input_file)
        
        # Generate report file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"complete_pipeline_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("COMPLETE VIDEO DATASET FILTERING PIPELINE REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Author: Graduate Student\n")
            f.write(f"Thesis: \"Perceptions of Maths on YouTube: Analysis using BERT-based Topic Modelling and Sentiment Analysis\"\n\n")
            
            f.write("FILTERING PIPELINE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for key, value in pipeline.stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
            
            original = pipeline.stats['step1_original']
            final = pipeline.stats['step5_final']
            f.write(f"\nOverall retention rate: {(final/original)*100:.1f}%\n")
            f.write(f"Total videos removed: {original - final:,}\n")
            
            f.write("\nFILTERING METHODOLOGY:\n")
            f.write("-" * 22 + "\n")
            f.write("1. Initial Preprocessing: Category filter (26,27,28) + quality filter\n")
            f.write("2. Math Keywords Filter: Required mathematical terminology validation\n")
            f.write("3. Targeted Filter: BYPASSED (not used in actual pipeline)\n")
            f.write("4. Consolidated Processing: Category 26 removal + duration/quality filters\n")
            f.write("5. Non-Latin Script Filter: Removed videos with non-Latin script\n")
        
        print(f"\nReport saved: {report_file}")
        print(f"Expected result: ~3,897 videos (complete pipeline target)")
        print(f"Actual result: {len(final_df):,} videos")
        
        if len(final_df) == 3897:
            print("SUCCESS: Target achieved!")
        else:
            difference = abs(len(final_df) - 3897)
            print(f"Note: Difference from target: {difference:,} videos")
        
    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()