# Data Directory

## Privacy and Ethics Notice

**Important**: Raw YouTube data containing user information has been **intentionally excluded** from this repository. However, for full transparency and reproducibility, **processed datasets with anonymized data are included**. This approach ensures:

1. **Protect User Privacy**: Comply with GDPR and data protection regulations
2. **Respect Content Creators**: Protect channel and creator information
3. **Maintain Ethics Standards**: Follow institutional research ethics guidelines
4. **YouTube ToS Compliance**: Adhere to YouTube API Terms of Service

## Directory Structure

```
data/
├── raw/                    # Original API data (gitignored)
│   ├── videos/            # Video metadata
│   └── comments/          # Comment data
├── processed/             # Filtered datasets (INCLUDED for transparency)
│   ├── comments_complete_filtered_20250720_224959.csv
│   └── videos_gradual_complete_filtered_20250720_223652.csv
├── annotations/           # Manual annotations
│   ├── guidelines/        # Annotation protocols
│   └── samples/          # Anonymized samples
└── sample/               # Small demo datasets
    ├── sample_videos.csv  # 10 anonymized videos
    └── sample_comments.csv # 100 anonymized comments
```

## Data Collection Pipeline

Data is collected using the YouTube Data API v3:

1. **Video Collection**: `src/data_collection/daily_collector.py`
2. **Comment Collection**: `src/data_collection/comment_collector.py`

## Data Processing Pipeline

1. **Video Filtering**: `src/data_processing/video_filtering_pipeline.py`

   - Educational content validation
   - Mathematical keyword matching
   - Non-Latin script removal
   - Quality filters

2. **Comment Filtering**: `src/data_processing/comment_filtering_pipeline.py`
   - Language detection (English)
   - Length validation
   - Spam removal
   - Educational relevance

## Accessing Research Data

### Included Processed Data

The processed datasets are included in this repository for transparency:

- **Comments**: `data/processed/comments_complete_filtered_20250720_224959.csv`
- **Videos**: `data/processed/videos_gradual_complete_filtered_20250720_223652.csv`

These datasets have been fully anonymized with all user identifiers removed or hashed.

### For Additional Raw Data

1. **Contact Author** with:

   - Institutional affiliation
   - Research purpose
   - Ethics approval documentation

2. **Sign Data Use Agreement** including:

   - No re-identification attempts
   - No commercial use
   - Data security measures
   - Deletion after project completion

3. **Receive Anonymized Dataset** with:
   - Hashed user IDs
   - Removed channel information
   - Aggregated metrics only

## Data Statistics

### Original Dataset

- Videos collected: 9,394
- Comments collected: 422,258
- Collection period: 17-30 June 2025

### Processed Dataset (Included)

- Videos retained: 3,897 (41.5%)
- Comments retained: 34,057 (8.1%)
- Topics identified: 82
- **Files provided**:
  - `comments_complete_filtered_20250720_224959.csv`: Filtered and anonymized comments
  - `videos_gradual_complete_filtered_20250720_223652.csv`: Filtered video metadata

## File Formats

- **CSV Files**: UTF-8 encoded, comma-separated
- **JSON Files**: For hierarchical data
- **Pickle Files**: For Python objects (models, preprocessed data)

## Privacy Measures

- User IDs: SHA256 hashed
- Channel names: Removed
- Video IDs: Anonymized
- Timestamps: Generalized to day-level
- Personal information: Completely removed

## Contact

For data access requests:

- Author: 14151162
- Institution: The University of Manchester
- Purpose: MSc Data Science Thesis
