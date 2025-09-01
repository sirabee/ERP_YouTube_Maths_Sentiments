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

- ~96.7% retention rate
- ~3,897 videos from initial 4,026

### 2. Filter Comments

```bash
python src/data_processing/comment_filtering_pipeline.py
```

Expected output:

- ~44.2% retention rate
- ~34,057 comments from initial 77,067

## Analysis Pipeline

### 1. Topic Modeling (BERTopic)

#### Per-Query HDBSCAN Approach

```bash
python src/models/bertopic/per_query_hdbscan.py
```

#### Variable K-Means Approach (Recommended)

```bash
python src/models/bertopic/variable_k_optimized.py
```

Expected performance:

- 8.72% weighted average noise (Variable K)
- 12 distinct topics across 378 query-topic combinations
- 82 successful query models

### 2. Sentiment Analysis

#### XLM-RoBERTa Enhanced (74% accuracy)

```bash
python src/models/sentiment/xlm_roberta_clean_enhanced.py
```

#### Alternative Models

```bash
# YouTube-BERT (73% accuracy)
python src/models/sentiment/per-sentence_analysis-variable-k-youtube.py

# Variable K XLM-RoBERTa (74.5% accuracy)
python src/models/sentiment/variable_k_sentence_analysis_xlm_roberta.py
```

### 3. Advanced Analysis

#### Coherence Analysis

```bash
python src/analysis/coherence/coherence_analysis.py
```

#### Model Performance Comparison

```bash
python src/analysis/performance/bertopic_model_comparison.py
```

#### Keyword Analysis

```bash
# Frequency analysis
python src/analysis/topic_labeling/keyword_frequency_analyzer.py

# Co-occurrence patterns
python src/analysis/topic_labeling/keyword_cooccurrence_analyzer.py

# Hierarchical structure
python src/analysis/keyword_hierarchy_tree.py
```

### 4. Generate Visualizations

#### Core Analysis Plots

```bash
python src/visualization/create_analysis_plots.py
```

#### Model Comparison Dashboard

```bash
python src/visualization/model_comparison_visualizations.py
```

#### BERTopic Comparison Charts

```bash
python src/visualization/bertopic_comparison_charts.py
```

#### Learning Journey Analysis

```bash
python src/analysis/learning_journey_video_analysis.py
```

#### Engagement Analysis

```bash
python src/analysis/engagement_sentiment_video_analysis.py
```

## Verification

### Check Results

1. Verify file outputs in `data/processed/`
2. Check figures in `results/figures/`
3. Review metrics in `results/tables/`
4. Examine model outputs in `results/models/`

### Expected Metrics

| Metric                           | Expected Value |
| -------------------------------- | -------------- |
| Videos Retained                  | ~96.7%         |
| Comments Retained                | ~44.2%         |
| BERTopic Noise (Variable K)      | ~8.72%         |
| Sentiment Accuracy (XLM-RoBERTa) | ~74%           |
| Learning Journey Detection       | ~88%           |
| Processing Time                  | 4-7 hours      |

### Key Performance Indicators

#### Topic Modeling

- **Variable K-Means**: 0.425 mean silhouette score
- **HDBSCAN Standard**: 0.380 mean silhouette score
- **Topics Generated**: 12 distinct topics
- **Query Coverage**: 82/82 queries successfully modeled

#### Sentiment Analysis Comparison

- **XLM-RoBERTa Enhanced**: 74% accuracy
- **Variable K XLM-RoBERTa**: 74.5% accuracy
- **YouTube-BERT**: 73% accuracy
- **Twitter-RoBERTa**: 37% accuracy (domain mismatch)

## Included Processed Data

The repository includes anonymized processed datasets for reproducibility:

- `data/processed/comments_complete_filtered_20250720_224959.csv`
- `data/processed/videos_gradual_complete_filtered_20250720_223652.csv`

These files contain:

- Anonymized user IDs (hashed)
- Filtered educational content
- Pre-processed text ready for analysis

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
   - Check file paths match your system
   - Note: Scripts use absolute paths specific to development environment

5. **Model Download Issues**
   - First run may take time to download pre-trained models
   - Ensure stable internet connection
   - Models cached in `~/.cache/huggingface/`

## Complete Reproduction Timeline

Estimated time for full pipeline reproduction:

1. Data Collection: 2-4 hours (API dependent)
2. Data Processing: 30-45 minutes
3. BERTopic Modeling: 2-3 hours
4. Sentiment Analysis: 1-2 hours
5. Visualization Generation: 15-30 minutes

**Total: 6-10 hours** (excluding API quota waiting times)

## Citation

If using this code or methodology, please cite:

```
14151162 (2025). Perceptions of Mathematics on YouTube:
BERT-based Topic Modelling and Sentiment Analysis.
MSc Thesis, University of Manchester.
```
