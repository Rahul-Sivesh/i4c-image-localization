# i4C Hackathon – Image Localization

This repository follows the latest repository requirements provided for the hackathon.

## Required files

- `README.md`
- `scripts/dataset_generator.py`
- `scripts/localization_inference.py`
- `requirements.txt`
- `references/README.md`
- `outputs/`

## 1. Installation

```bash
git clone <https://github.com/Rahul-Sivesh/i4c-image-localization>
cd i4c_localization_repo

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Generate a sample image pair

The generator accepts the required architecture type, number of pairs and output directory.

```bash
python scripts/dataset_generator.py --architecture FinFET --num_pairs 1 --output_dir sample_data
```

For DRAM:

```bash
python scripts/dataset_generator.py --architecture DRAM --num_pairs 1 --output_dir sample_data
```

The generated directory contains:

```text
reference_0000.png
search_0000.png
labels.csv
```

`labels.csv` records the true center coordinates `(x, y)` of the reference pattern in every generated search image.

## 3. Run localization inference

The inference script accepts exactly the two image paths required by the submission:

```bash
python scripts/localization_inference.py \
    --reference sample_data/reference_0000.png \
    --search sample_data/search_0000.png
```

It prints:

```text
Predicted center: (x, y)
```

and writes a JSON result to `prediction.json`.

No source-code editing or hard-coded test paths are required.

## 4. Algorithm

This repository contains a classical image-localization baseline using normalized template matching:

1. Load the reference image.
2. Load the search image.
3. Match the reference against the search image.
4. Find the highest-scoring location.
5. Convert the best-match bounding box to its center coordinate.
6. Output one `(x, y)` coordinate.

This baseline does not require DL weights.

## 5. Reproducibility test

Run all of the following on a fresh environment:

```bash
pip install -r requirements.txt

python scripts/dataset_generator.py \
    --architecture FinFET \
    --num_pairs 1 \
    --output_dir sample_data

python scripts/localization_inference.py \
    --reference sample_data/reference_0000.png \
    --search sample_data/search_0000.png
```

The complete workflow should run without modifying either Python script.

## 6. Repository requirement checklist

| Hackathon requirement | Repository item |
|---|---|
| README.md | `README.md` |
| Dataset Generator Script | `scripts/dataset_generator.py` |
| Localization Inference Script | `scripts/localization_inference.py` |
| DL Model Weights | Not applicable to this template-matching implementation |
| Training Script/Notebook | Not applicable |
| requirements.txt | `requirements.txt` |
| Citation Documents / Supporting References | `references/README.md` |

## Important before submission

Replace `<YOUR_PUBLIC_GITHUB_REPOSITORY_URL>` with your actual public GitHub URL.

Also add the exact research papers, datasets, augmentation methods, or model references actually used by your final project to `references/README.md` and your PPT.

If your final solution is a deep-learning localization model rather than template matching, replace `scripts/localization_inference.py` with the final model inference code and include its downloadable weights. The command-line interface should remain:

```bash
python scripts/localization_inference.py --reference <reference_path> --search <search_path>
```

The script must output one predicted `(x, y)` coordinate.
