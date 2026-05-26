# update_asign_4

This folder contains a self-contained medium-level EDA and classification
pipeline for Assignment 4. It's designed so you can run the analyses from
this folder without touching other parts of the repository. It will attempt
to locate parsed transformed CSV files under `outputs_transformed_data` in
this folder or the repository root; if found it will create a local
`outputs_regression_dataset/regression_dataset.csv` and run the analyses.

Quick start:

1. Install dependencies (recommended in a venv):

```
python -m pip install -r requirements.txt
```

2. Run the EDA:

```
python Assignment_4_Script/1_A4_eda.py
```

3. Run the classification pipeline:

```
python Assignment_4_Script/2_A4_classification.py
```

Outputs will be saved under `output/` inside this folder.
