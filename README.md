# Meetup POI Prediction
Predict where a user’s next **group meetup** will happen using check-ins and the friendship graph from Foursquare Location-Based Social Network (LBSN) data and friendship network. This repository implements a two-stage pipeline:
1) **Cluster prediction** with a Transformer-based classifier over spatio-temporal and social features, then
2) **POI ranking** within the predicted cluster to return Top-K meetup venues.

> **Associated paper**: *Where Will You Meet on the Next Sunday?: A Deep Learning-based Approach for a Meetup POI Prediction* (PDF included). Please cite if you use this work.

---

## Highlights
- **Two-stage** Cluster→POI architecture for sparse LBSN data
- **Companion-aware** features: favorite companion/day, companion lag cluster
- **Temporal & mobility priors**: weekday/period, user activeness per cluster
- **State-of-the-art (paper)**: Large gains over RNN/LSTM/GRU/BiLSTM baselines for cluster accuracy and Top-K POI accuracy
- **Reproducible** notebooks and baseline experiments

---

## Repository Layout
```
Meetup_prediction/
├── US_training_test_ratio/         # Train/test split experiments
├── baselines/                      # RNN/LSTM/GRU/BiLSTM baselines
├── fig_meetup-*/ fig_meetup/       # Generated figures
├── graphs_jun29.ipynb              # Plotting & analysis
├── test_dist_utility.py            # Distance helpers / smoke tests
└── (your data here)
```
> Tip: Most experiments are notebook-driven. Convert to scripts for production if desired.

---

## Data Model
- **Check-ins**: `user_id, poi_id, lat, lon, category, timestamp`
- **Friendship graph**: `user_id, friend_id` (undirected)
- **Meetup event**: co-location of a user with ≥1 friend within a **120-minute** window (paper setting)
- **Clusters**: Mean-Shift over meetup POIs; every POI maps to nearest cluster center

---

## Getting Started
### 1) Environment
```bash
conda create -n meetup-poi python=3.10 -y
conda activate meetup-poi
pip install -r requirements.txt
```

### 2) Prepare Data
Download the Foursquare check-in data with user friendship network from here: https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_7rmPjnwFGIx9. Place/point your raw data under `data/` (or your preferred folder):
```
data/
├── checkin data           # user_id, poi_id, lat, lon, category, timestamp
├── friendship network       # user_id, friend_id
└── (optional) metadata/
```
Ensure timestamps are in UTC or consistently localized. 
Clean dataset and generate meetup events. This part requires large data joins, so consider using AWS S3 and Athena if you struggle to accommodate everything in your local machine.

### 3) Build Clusters
Use a preprocessing notebook/script to run **Mean-Shift** on meetup POIs and map all POIs to cluster IDs. Persist:
```
data/
├── clusters.parquet       # cluster_id, center_lat, center_lon, radius, n_pois
└── poi_to_cluster.parquet # poi_id -> cluster_id
```

### 4) Feature Engineering
Generate feature tables per (user, time_slot):
- User activeness per cluster, recent cluster history
- Favorite companion/day; companion lag cluster
- Temporal context (weekday/period)

### 5) Train the Cluster Classifier
- **Model**: Embeddings + simplified **Transformer encoder** → softmax over clusters
- **Loss**: Cross-entropy; **metrics**: accuracy@1 (cluster), macro-F1

Example (pseudo-CLI):
```bash
python train_cluster.py \
  --features data/features.parquet \
  --clusters data/clusters.parquet \
  --out runs/cluster_model
```

### 6) Rank POIs within Predicted Cluster
- Candidate selection by category & user/companion frequency
- Probability-weighted cluster center → **distance-based** ranking
- Report **Top-K** accuracy (K=5/10/20)

Example (pseudo-CLI):
```bash
python rank_poi.py \
  --pred_clusters runs/cluster_model/preds.parquet \
  --poi_map data/poi_to_cluster.parquet \
  --k_list 5 10 20 \
  --out runs/poi_ranking
```

---

## Benchmarks (from paper)
- **Cluster accuracy** (US dataset): ~**0.81** (Transformer) vs baselines ≤ ~0.44
- **POI prediction**: Acc@5 ≈ **0.234**, Acc@10 ≈ **0.352**, Acc@20 ≈ **0.464**
- Saturation around ~120 epochs; strong gains with 80/20 train/test split

> Numbers depend on dataset preprocessing & coverage; reproduce using the provided notebooks.

---

## Configuration
Use environment variables or a simple YAML (`config.yaml`) to centralize paths:
```yaml
paths:
  data_dir: data/
  clusters: data/clusters.parquet
  poi_map: data/poi_to_cluster.parquet
  features: data/features.parquet
train:
  batch_size: 256
  lr: 0.001
  epochs: 120
  seed: 42
model:
  d_model: 128
  n_heads: 4
  n_layers: 2
  dropout: 0.1
```

---

## Development
- Format/lint: `black`, `ruff`
- Type hints: `mypy` (optional)
- Tests: add unit tests under `tests/` (e.g., distance utilities, feature builders)

---

## Contributing
1. Create a feature branch: `git checkout -b feat/your-feature`
2. Commit with conventional commits (e.g., `feat: add transformer layer`)
3. Open a Pull Request with context and evaluation results

---

## Ethics & Privacy
Handle LBSN data responsibly:
- Remove/obfuscate user identifiers where possible
- Respect platform ToS and rate limits
- Avoid sharing raw trajectories publicly

---

## Citation
If you use this code or ideas, please cite:
```
Arabi, S., Shaira, A., & Hashem, T.
"Where Will You Meet on the Next Sunday?: A Deep Learning-based Approach for a Meetup POI Prediction."
(See included PDF for details.)
```

---

## License
This code is made available for research and educational purposes. See `LICENSE` if present; otherwise, please contact the authors for terms.
