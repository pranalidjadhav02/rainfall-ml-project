# Monsoon Rainfall Analysis and Prediction for Indian Subdivisions

A final-year machine learning project that analyses 121 years (1901-2021) of monsoon rainfall data from 36 meteorological subdivisions of India. It builds **classification** models (Low / Medium / High rainfall) and **regression** models (total rainfall in mm) using Python and Scikit-Learn.

## Project Highlights

- Found and fixed a hidden data quality problem: the value `-99.9` was used as a placeholder for missing data in 10 rows, so `isnull()` showed no missing values.
- Exploratory analysis of rainfall by year, month and subdivision with histograms, boxplots, trend lines and a correlation heatmap.
- Compared 3 classifiers and 3 regressors against simple baselines.
- Used Pipelines (one-hot encoding + scaling), a stratified train-test split and GridSearchCV.
- Saved the best models with Pickle.

## Results

**Classification (test set, 865 records)**

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Baseline (most frequent) | 0.333 | 0.111 | 0.333 | 0.167 |
| **Logistic Regression** | **0.849** | **0.850** | **0.849** | **0.849** |
| Decision Tree | 0.812 | 0.824 | 0.812 | 0.815 |
| Random Forest | 0.822 | 0.831 | 0.822 | 0.825 |

**Regression (test set, 865 records)**

| Model | MAE (mm) | RMSE (mm) | R² |
|---|---|---|---|
| Baseline (mean) | 538.5 | 739.7 | -0.001 |
| **Linear Regression** | **111.8** | **156.8** | **0.955** |
| Decision Tree | 129.9 | 174.8 | 0.944 |
| Random Forest | 115.6 | 162.4 | 0.952 |

**Best models:** Logistic Regression (classification) and Linear Regression (regression). The simple linear models won because the seasonal total grows almost in a straight line with June + July rainfall.

## Key Findings

- July is the wettest month (32.6% of the seasonal rainfall).
- Rainfall differs hugely by region: Coastal Karnataka averages 2977 mm, West Rajasthan only 264 mm.
- There is no strong long-term trend (about -1.7 mm per decade).
- June + July rainfall is by far the most important model input. The year has almost no effect.

## Problem Setup

| Item | Details |
|---|---|
| Inputs | `subdivision`, `YEAR`, `JUN`, `JUL`, `JUN_JUL` (June + July) |
| Classification target | `CLASS`: Low (up to 664.8 mm), Medium (664.8 to 1126.3 mm), High (above 1126.3 mm) |
| Regression target | Total monsoon rainfall `JUN-SEP` (mm) |
| Split | 80% train / 20% test, `random_state=42`, stratified by class |

August and September are **not** used as inputs because together with June and July they add up to the target (data leakage). The models therefore work as a **mid-season estimate**.

## Project Structure

```
rainfall_project/
├── Rainfall_Prediction_Project.ipynb   # Complete notebook (code, outputs, observations)
├── Project_Report.docx                 # Project report
├── Project_Presentation.pptx           # 12-slide presentation
├── README.md
├── requirements.txt
├── data/
│   └── rainfaLLIndia.csv               # Dataset
├── images/                             # Charts saved from the notebook
└── models/
    ├── best_classifier.pkl             # Logistic Regression pipeline
    ├── best_regressor.pkl              # Linear Regression pipeline
    ├── class_thresholds.json
    └── results.json                    # All metrics
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Rainfall_Prediction_Project.ipynb
```

Run all cells from top to bottom. The notebook expects the dataset at `data/rainfaLLIndia.csv`. Training takes about two minutes because of the grid search.

## Using the Saved Models

```python
import pickle
import pandas as pd

with open('models/best_regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)
with open('models/best_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Subdivision names must match the cleaned names used in the notebook
sample = pd.DataFrame({'subdivision': ['Madhya Maharashtra'], 'YEAR': [2020],
                       'JUN': [174.7], 'JUL': [344.3], 'JUN_JUL': [519.0]})

print('Predicted rainfall (mm):', regressor.predict(sample)[0])
print('Predicted class        :', classifier.predict(sample)[0])
```

## Limitations

- Uses the actual June and July rainfall of the same year, so it is a mid-season estimate, not a pre-monsoon forecast.
- Only rainfall data is used (no temperature, humidity or El Niño data).
- The split is random, not by year. A time-based split would be a stricter test.
- The rainfall class boundaries were calculated from the whole dataset.

## Future Scope

- Add climate variables (El Niño / La Niña, Indian Ocean Dipole, temperature, humidity).
- Evaluate with a time-based split.
- Build separate models for each region.
- Try Gradient Boosting and K-Nearest Neighbours.
- Create a small Streamlit app for predictions.

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Pickle, Jupyter Notebook.

## Dataset

Monsoon (June to September) rainfall of 36 meteorological subdivisions of India, 1901-2021, in millimetres. The source file is `rainfaLLIndia.csv`.

## Author

Pranali Jadhav| SOCSE , Sandip University , Nashik 
