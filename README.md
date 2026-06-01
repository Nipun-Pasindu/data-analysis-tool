# Data Analysis Tool

A robust Python toolkit designed for data cleaning, exploration, and interactive visualization within Google Colab environments. This tool automates common preprocessing tasks such as data sanitization, missing value imputation, outlier detection, and advanced statistical association mapping.

## Installation

```bash
# Basic installation containing core mathematical engines
pip install "git+[https://github.com/Nipun-Pasindu/data-analysis-tool.git](https://github.com/Nipun-Pasindu/data-analysis-tool.git)"

# Install with Plotly plotting support (required for interactive visualizations)
pip install "git+[https://github.com/Nipun-Pasindu/data-analysis-tool.git#egg=data-analysis-tool](https://github.com/Nipun-Pasindu/data-analysis-tool.git#egg=data-analysis-tool)[plotting]"
from data_analysis import DataInspector

inspector = DataInspector()

# 1. Interactive Loader in Colab
# inspector.upload_data()

# 2. Programmatic processing mock setup
import pandas as pd
inspector.df = pd.DataFrame({
    "Age": [24, 45, 31, 120, None, 31], # 120 acts as an IQR outlier
    "Salary": [52000, 98000, 74000, 110000, 64000, 74000],
    "Dept": ["HR", "Eng", "Mkt", "Eng", "HR", "Mkt"]
})

# Run clean automation features
inspector.handle_missing_values(strategy="median")
inspector.remove_duplicates()
inspector.remove_outliers_iqr(columns=["Age"])

# 3. Create machine learning datasets
final_ml_df = inspector.create_normalized_data_df(numeric_method="robust", categorical_method="onehot")
print(final_ml_df.head())

# 4. View statistical distributions and mixed analytics maps
inspector.plot_all_associations_heatmap()
Authors
e/23/210 - nipunpasindu276@gmail.com

License
This project is licensed under the MIT License.

---

### How to push this directly to your GitHub Destination Repo
Open your local terminal inside your workspace directory and execute the following Git initialization and push routines:

```bash
# Initialize repository
git init

# Add remote configuration targeting your specific repo link
git remote add origin https://github.com/Nipun-Pasindu/data-analysis-tool.git

# Stage and commit your new package layouts
git add .
git commit -m "Initial commit: Production-ready data analysis tool package architecture"

# Branch configuration and pushing to GitHub
git branch -M main
git push -u origin main
