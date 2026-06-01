# data-analysis-tool

A robust Python toolkit for data cleaning, exploratory data analysis (EDA), preprocessing, feature engineering, and interactive visualization. The package is designed to simplify common data science workflows by providing an easy-to-use interface for inspecting, cleaning, transforming, and visualizing datasets.

---

## Features

### Data Loading & Inspection
- Automatic detection of common null values (`?`, `N/A`, `NULL`, etc.)
- Automatic numeric type conversion where possible
- Dataset shape and structure inspection
- Statistical summaries for numerical and categorical features

### Data Cleaning
- Missing value detection and imputation
  - Mean
  - Median
  - Mode
  - Constant value
- Duplicate row removal
- Outlier detection and removal using IQR method
- Interactive row and column deletion

### Data Transformation
#### Numerical Data
- Min-Max Scaling
- Standard (Z-Score) Scaling
- Robust Scaling

#### Categorical Data
- One-Hot Encoding
- Ordinal Encoding
- Uniform Encoding

### Data Visualization
Interactive visualizations powered by Plotly:
- Histograms
- Scatter Plots
- Violin Plots
- Grouped Bar Charts
- Pie Charts
- Relationship Plots

### Statistical Analysis
- Pearson Correlation Heatmaps
- Cramér’s V Association Heatmaps
- Unified Mixed-Type Association Heatmaps
- Point-Biserial Correlation
- Eta/ANOVA Based Associations

---

## Installation

### Install from GitHub

```bash
pip install "git+https://github.com/Nipun-Pasindu/data-analysis-tool.git"
```

### Install with Plotting Support

```bash
pip install data-analysis-tool[plotting]
```

---

## Quick Start

### Import the Package

```python
from data_analysis import DataInspector

inspector = DataInspector()
```

### Upload and Clean Data

```python
inspector.upload_data()

inspector.handle_missing_values(strategy="median")

inspector.remove_duplicates()
```

### Visualize Data

```python
inspector.plot_numerical(["Age", "Salary"])

inspector.plot_relationship("Department", "Salary")
```

### Feature Engineering

```python
normalized_numeric = inspector.extract_normalized_numeric_data(
    method="robust"
)

encoded_cat = inspector.extract_normalized_categorical_data(
    method="onehot"
)

final_df = inspector.create_normalized_data_df()
```

### Statistical Analysis

```python
inspector.plot_all_associations_heatmap()
```

---

## Custom Visualization Examples

### Bar Chart

```python
result = plotter.plot_bar_chart(
    x="Department",
    y="Salary",
    color="Gender",
    barmode="group",
    data=my_json_data
)

plotter.display_image(result)
```

### Pie Chart

```python
result = plotter.plot_pie_chart(
    names="Category",
    values="Total",
    hole=0.4,
    title="Revenue Split"
)

plotter.display_image(result)
```

### Histogram

```python
result = plotter.plot_histogram(
    x="Age",
    bins=[0,18,35,60,100],
    title="Age Demographics"
)

plotter.display_image(result)
```

---

## Project Structure

```text
data-analysis-tool/
│
├── data_analysis/
│   ├── __init__.py
│   └── core.py
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## GitHub Repository

Repository:
https://github.com/Nipun-Pasindu/data-analysis-tool.git

---

## Author

**Nipun Pasindu**

- Email: nipunpasindu276@gmail.com
- Index Number: E/23/210

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

This package was developed as part of academic and practical work in data analysis, preprocessing, visualization, and statistical exploration using Python.
