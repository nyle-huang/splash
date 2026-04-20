# Data Contracts

## Source Datasets

- CreativeRanking-style corpus under `/workspace/data`
- Raw input product photos supplied by the user at inference time
- Synthetic bootstrap views generated from CreativeRanking items for training the input-photo encoder
- Human-review seed photos sampled from publicly viewable customer review photos in e-commerce comment sections, curated locally with source metadata

## CreativeRanking Manifest Schema

Each manifest row is expected to contain:

1. `item_id`
2. `image_name`
3. `ds`
4. `pv`
5. `clk`

## Derived Assets

- Typed parquet manifests
- Localized product crops
- Segmentation masks
- Synthetic everyday-photo inputs
- Embeddings for retrieval and evaluation

## Integrity Requirements

- No split leakage across `item_id`
- Every manifest row references an existing image
- Every synthetic view retains its source `item_id`
- Every generated artifact is traceable back to the source row or input photo
- Every human-review seed photo must retain its source URL, source platform, capture date, and local file path
