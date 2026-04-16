# Animated Movies: Standalone vs Sequel Analysis

A data mining project investigating whether standalone animated films differ from sequels and remakes in audience ratings and commercial success.

## Research Question

Do standalone animated films receive different audience ratings and engagement compared to sequels or remakes, and how does this affect their success?

## Dataset

- **182 animated movies** from major studios: Disney, Pixar, DreamWorks, Illumination, Sony
- Features: budget, runtime, release date, studio, audience engagement (votes), ratings, revenue
- Engineered labels: `is_standalone`, `is_sequel_or_spinoff`, `is_remake`

## Key Findings

| Movie Type    | Avg Rating | Avg Revenue | Avg Votes |
|---------------|------------|-------------|-----------|
| Standalone    | 7.04       | $307M       | 264,429   |
| Sequel/Remake | 6.67       | $617M       | 205,705   |

- Standalone films score **higher on ratings** but earn **lower revenue** than sequels/remakes
- Audience engagement (vote count) shows a weak positive correlation with ratings

## Models

All models use XGBoost (regressor and classifier variants).

| Model                       | Metric (R² or Accuracy) |
|-----------------------------|-------------------------|
| Rating Regressor            | R² = 0.556              |
| Rating Regressor (Fixed*)   | R² = 0.485              |
| Rating Classifier           | Accuracy = 80.0%        |
| Revenue Regressor           | R² = 0.523              |
| Revenue Regressor (Fixed*)  | R² = 0.485              |
| Revenue Classifier          | Accuracy = 83.6%        |

*Fixed models remove leaky features (revenue removed for rating prediction; averageRating removed for revenue prediction).

Classification targets:
- **Rating**: Low (< 7.0) vs High (≥ 7.0)
- **Revenue**: Low (< $500M) vs High (≥ $500M)

## Feature Importance Insight

Removing `is_standalone` from the rating classifier slightly *improved* accuracy (80.0% → 81.8%), suggesting that standalone vs sequel status is not a strong predictor compared to budget, vote count, and studio.

## Requirements

```
pandas
scikit-learn
xgboost
shap
matplotlib
numpy
```

## Usage

Open `Final_Project.ipynb` in Google Colab. Place `Animated_Movies_Dataset.csv` in your Google Drive under `Social_Media_Mining_Project/` and update the `%cd` path accordingly.
