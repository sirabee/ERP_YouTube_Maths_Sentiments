# Reproducibility Guide

## System Requirements

### Minimum Requirements

- CPU: 4 cores
- RAM: 16GB
- Storage: 50GB free space
- Python: 3.8+

## Environment Setup

### Option 1: pip (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: conda

```bash
# Create conda environment
conda create -n youtube-math python=3.8

# Activate environment
conda activate youtube-math

# Install dependencies
pip install -r requirements.txt
```

## Data Collection

### 1. Configure API Access

```bash
# Copy template
cp config/config_template.py config/config.py

# Edit config.py and add your YouTube API key
```

### 2. Collect Video Data

```bash
python src/data_collection/daily_collector.py
```

This will:

- Query YouTube API for mathematics videos
- Save video metadata
- Respect API quotas

### 3. Collect Comments

```bash
python src/data_collection/comment_collector.py
# or for enhanced collection:
python src/data_collection/comment_collector2.py
```

## Data Processing

### 1. Filter Videos

```bash
python src/data_processing/video_filtering_pipeline.py
```

Expected output:

- ~41.5% retention rate
- ~3,897 videos from initial 9,394

### 2. Filter Comments

```bash
python src/data_processing/comment_filtering_pipeline.py
```

Expected output:

- ~8.1% retention rate
- ~34,057 comments from initial 422,258

## Analysis

### 1. Topic Modeling

```bash
python src/models/bertopic_model.py
```

Expected performance:

- 8.72% average noise (per-query approach)
- 82 successful query models

### 2. Sentiment Analysis

```bash
python src/analysis/sentiment_analysis.py
```

### 3. Generate Visualizations

```bash
python scripts/generate_figures.py
```

## Verification

### Check Results

1. Verify file outputs in `data/processed/`
2. Check figures in `results/figures/`
3. Review metrics in `results/tables/`

### Expected Metrics

| Metric            | Expected Value |
| ----------------- | -------------- |
| Videos Retained   | ~8.1%          |
| Comments Retained | ~41.5%         |
| BERTopic Noise    | ~8.72%         |
| Processing Time   | 4-7 hours      |

## Troubleshooting

### Common Issues

1. **API Quota Exceeded**

   - Wait 24 hours for quota reset
   - Reduce MAX_VIDEOS_PER_QUERY in config

2. **Memory Error**

   - Reduce batch_size in scripts
   - Process data in chunks
   - Use GPU if available

3. **Missing Dependencies**

   ```bash
   pip install --upgrade -r requirements.txt
   ```

4. **Path Errors**
   - Ensure running from repository root
   - Check file paths in config.py
