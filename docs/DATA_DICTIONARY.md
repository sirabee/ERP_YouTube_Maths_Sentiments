# Data Dictionary

## Video Data Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| video_id_hash | string | Anonymized video identifier | "a3f5b2c1" |
| published_date | date | Video publication date | "2024-03-15" |
| duration_seconds | integer | Video length in seconds | 425 |
| view_count_log | float | Log-transformed view count | 5.32 |
| engagement_rate | float | (likes + comments) / views | 0.045 |
| category_id | integer | YouTube category code | 27 |
| has_math_keywords | boolean | Mathematical terms present | true |
| language_detected | string | Primary language | "en" |
| topic_assigned | integer | BERTopic cluster ID | 3 |

## Comment Data Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| comment_id_hash | string | Anonymized comment ID | "b4e6c3d2" |
| video_id_hash | string | Associated video ID | "a3f5b2c1" |
| text_processed | string | Cleaned comment text | "great explanation of derivatives" |
| text_length | integer | Character count | 34 |
| word_count | integer | Number of words | 5 |
| sentiment_label | string | Sentiment classification | "positive" |
| sentiment_score | float | Confidence score | 0.89 |
| is_reply | boolean | Reply to another comment | false |
| has_math_terms | boolean | Contains math vocabulary | true |

## Topic Model Output

| Field | Type | Description |
|-------|------|-------------|
| topic_id | integer | Topic identifier |
| topic_keywords | list[string] | Top representative words |
| topic_size | integer | Number of documents |
| coherence_score | float | Topic coherence metric |
| representative_docs | list[string] | Example documents |

## Sentiment Analysis Output

| Field | Type | Description |
|-------|------|-------------|
| document_id | string | Document identifier |
| positive_prob | float | Positive sentiment probability |
| neutral_prob | float | Neutral sentiment probability |
| negative_prob | float | Negative sentiment probability |
| predicted_label | string | Final classification |
| confidence | float | Prediction confidence |

## Processing Flags

| Field | Type | Description |
|-------|------|-------------|
| passed_language_filter | boolean | English detected |
| passed_length_filter | boolean | Within length bounds |
| passed_spam_filter | boolean | Not spam |
| passed_math_filter | boolean | Mathematical relevance |
| passed_script_filter | boolean | Latin script only |

## Aggregated Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| retention_rate | Percentage retained | (output / input) * 100 |
| noise_percentage | BERTopic outliers | (outliers / total) * 100 |
| avg_coherence | Mean topic coherence | mean(coherence_scores) |
| sentiment_distribution | Class proportions | count(class) / total |

## File Naming Conventions

| Pattern | Description | Example |
|---------|-------------|---------|
| {stage}_{date}.csv | Processing outputs | videos_filtered_20250105.csv |
| {model}_{version}.pkl | Saved models | bertopic_v1.0.pkl |
| {metric}_{analysis}.json | Results | coherence_perquery.json |
