# Predicting the Carbon Intensity of France's Electricity Mix

Machine learning from scratch: a **Naive Bayes classifier** and a **k-nearest neighbors (k-NN)** algorithm, implemented in Python without any machine-learning library, and evaluated with **k-fold cross-validation** that we also coded ourselves.

Built for the Kaggle class competition [ARE SD 2024 – Émissions CO2 du mix électrique](https://kaggle.com/competitions/are-sd-2024-emissions-co2-du-mix-electrique) (Sorbonne Université).

**Jury grade: 84.48 / 100**

## The problem

Weather drives electricity production. Wind powers wind turbines, temperature drives heating demand, and cloud cover affects solar panels. When renewables can't keep up, the grid turns to fossil-fuel plants, which emit far more CO₂ (a coal plant emits about 175 times more CO₂ per kWh than nuclear or hydro, according to ADEME's Base Carbone).

**Goal:** from weather data, predict the state of France's electricity mix (`MixProdElec`), which has three classes:

- **Décarboné** (low-carbon)
- **Normal**
- **Carbonné** (carbon-intensive)

## Data

Nearly 100,000 observations, with three classes that are fairly balanced (about 35,000 Carbonné, 32,500 Normal, 28,000 Décarboné). A model that always predicts the most common class would only be right about **37%** of the time: that's the baseline to beat.

Features explored:

| Feature | What it is | What we found |
|---|---|---|
| `Temperature` | °C | Two peaks (around 9 °C and 17 °C): winter/summer seasonality |
| `WindSpeed` | m/s | Mostly 3–4.5 m/s, with a long tail of windy episodes |
| `Nebulosity` | cloud cover, % | Heavily concentrated around 90%, so it separates the classes poorly |

The dataset is not included in this repository. Download it from the [Kaggle competition page](https://kaggle.com/competitions/are-sd-2024-emissions-co2-du-mix-electrique).

## Approach

### 1. Checking the Naive Bayes assumption
Naive Bayes assumes features are independent given the class. We measured the correlation between temperature and wind speed: **−0.05**, almost zero, so the assumption is reasonable for these two features.

### 2. Naive Bayes classifier
For each observation, the model picks the class *c* that maximizes
P(c) × P(temperature | c) × P(wind speed | c),
with every probability estimated from frequencies in the training data.

### 3. k-nearest neighbors
For each new observation, find the *k* closest training points (Euclidean distance, vectorized with NumPy) and predict the most common class among them.

### 4. k-fold cross-validation
Shuffle the data, split it into *n* folds, and use each fold in turn as the test set while training on the others. We record the accuracy on each fold and the total run time.

## Key findings

- Naive Bayes is simple and fast, but estimating probabilities value by value works poorly for continuous variables: many temperatures in the test set never appear in training, which gives a probability of zero.
- k-NN handles continuous variables naturally, but it's expensive (it computes a distance to every training point), so we ran cross-validation on a subset of the data.
- Cross-validation gives a more reliable accuracy estimate than a single train/test split.

## Possible improvements

- Bin continuous variables, or switch to Gaussian Naive Bayes
- Normalize features before computing k-NN distances (°C and m/s have different scales)
- Compare several values of *k* with cross-validation
- Add more features (hour, season, cloud cover) and compare both models on the same folds

## Repository structure

```
├── README.md
├── classifiers.py     # Naive Bayes, k-NN and cross-validation, from scratch
├── requirements.txt
└── report.pdf         # Full project report (in French)
```

## How to run

1. Download the data from Kaggle and put the CSV files next to `classifiers.py`.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Check the file names at the bottom of `classifiers.py`, then run:
   ```
   python classifiers.py
   ```

## Built with

- Python 3
- NumPy and pandas (for data handling only; both models and the cross-validation are written by hand)

## Team

- Amine Kariouta
- Lina Nour
- Tassadit Aksas
- Mohamed Nassim Ferroukh

## References

- Brégère, M. (2024). *ARE SD 2024 – Émissions CO2 du mix électrique*. Kaggle.
- ADEME, *Base Carbone*.
- Dhkissi, Y. (2023). *Quelles sont les émissions de CO2 par source d'énergie ?* Selectra.
