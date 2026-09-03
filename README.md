# EDA and Gradient Descent

Two standalone ML engineering portfolio projects: exploratory data analysis
on a real-world churn dataset, and a from-scratch implementation of
gradient descent validated against scikit-learn.

## Projects

### 📊 [EDA Dashboard](./eda-dashboard)

Exploratory data analysis on the Telco Customer Churn dataset (7043 rows,
Kaggle) across three visualization libraries: Matplotlib, Seaborn, and
Plotly. 10 questions covering churn rate, contract type, tenure, monthly
charges, correlation, service type, payment method, and demographic factors.

* `01_eda_matplotlib.py` — static distribution and comparison charts
* `02_eda_seaborn.py` — statistical plots, correlation heatmap, outlier detection
* `03_eda_plotly.py` — interactive scatter and bar charts with hover context

**Key finding:** contract type, tenure, and monthly charges are the
strongest churn signals in the dataset; month-to-month, low-tenure,
high-paying customers are the highest-risk segment.

### 📈 [Gradient Descent from Scratch](./gradient-descent-from-scratch)

A manual implementation of multi-feature linear regression trained via
gradient descent, no `model.fit()` shortcuts. Validated against
scikit-learn's `LinearRegression` as a correctness check.

**Key finding:** the from-scratch implementation converges to within
~0.0003 (weights) and ~0.005 (bias) of scikit-learn's closed-form solution.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn

## Structure

```
eda-and-gradient-descent/
├── eda-dashboard/
│   ├── data/
│   ├── visuals/
│   │   ├── matplotlib/
│   │   ├── seaborn/
│   │   └── plotly/
│   ├── 01_eda_matplotlib.py
│   ├── 02_eda_seaborn.py
│   ├── 03_eda_plotly.py
│   ├── notes.md
│   └── README.md
├── gradient-descent-from-scratch/
│   ├── gradient_descent_from_scratch.py
│   ├── convergence_plot.png
│   ├── notes.md
│   └── README.md
└── README.md
```