# Data Dictionary

This document describes every column produced across the Pok\u00e9mon ML
Assessment pipeline. Columns are grouped by the stage in which they are
created.

## `data/pokemon_dataset.csv` (Part 1 \u2014 Data Collection & Preprocessing)

| Column | Data Type | Description |
|---|---|---|
| `name` | string | Pok\u00e9mon name (lowercase, as returned by PokeAPI). |
| `height` | int | Height of the Pok\u00e9mon, in decimetres. |
| `weight` | int | Weight of the Pok\u00e9mon, in hectograms. |
| `base_experience` | int | Base experience yielded for defeating this Pok\u00e9mon. |
| `hp` | int | Base HP (hit points) stat. |
| `attack` | int | Base physical Attack stat. |
| `defense` | int | Base physical Defense stat. |
| `special_attack` | int | Base Special Attack stat. |
| `special_defense` | int | Base Special Defense stat. |
| `speed` | int | Base Speed stat. |
| `primary_type` | string | Primary elemental type (e.g. `grass`, `fire`, `water`). |
| `primary_type_encoded` | int | Label-encoded integer representation of `primary_type` (`sklearn.preprocessing.LabelEncoder`, alphabetical mapping). |

**Row count:** 200 Pok\u00e9mon (Pok\u00e9dex IDs 1\u2013200). **Nulls:** 0. **Duplicates:** 0.

## `data/pokemon_feature_engineered.csv` (Part 2 \u2014 Feature Engineering)

Contains every column above, plus:

| Column | Data Type | Description |
|---|---|---|
| `total_power` | int | Engineered feature: `hp + attack + defense + special_attack + special_defense + speed`. |
| `is_high_power` | int (0/1) | Engineered binary label: `1` if `total_power` is strictly greater than the dataset median `total_power` (405.0), else `0`. |

**Row count:** 200. **Class balance:** 105 `is_high_power = 0`, 95 `is_high_power = 1`.

## `report/regression_metrics.csv` (Part 3 \u2014 Regression)

| Column | Description |
|---|---|
| `Metric` | Name of the evaluation metric (`MAE`, `MSE`, `RMSE`, `R2_Score`, `Mean_CV_R2`). |
| `Value` | Computed value of that metric on the held-out test set (or 5-fold CV mean). |

## `report/model_comparison.csv` (Part 4 \u2014 Classification)

| Column | Description |
|---|---|
| `Model` | Model name: Logistic Regression, Decision Tree (Default), or Decision Tree (Tuned - GridSearchCV). |
| `Accuracy` | Test-set accuracy. |
| `Precision` | Test-set precision (positive class = `is_high_power = 1`). |
| `Recall` | Test-set recall. |
| `F1_Score` | Test-set F1 score. |

## Primary Type Encoding Map

`primary_type_encoded` is generated via alphabetical label encoding of the
16 distinct `primary_type` values present in the 200-Pok\u00e9mon sample:

| Code | Type | Code | Type |
|---|---|---|---|
| 0 | bug | 8 | grass |
| 1 | dark | 9 | ground |
| 2 | dragon | 10 | ice |
| 3 | electric | 11 | normal |
| 4 | fairy | 12 | poison |
| 5 | fighting | 13 | psychic |
| 6 | fire | 14 | rock |
| 7 | ghost | 15 | water |
