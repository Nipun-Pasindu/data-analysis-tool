from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List, Union
import json
import uuid
import sys
import io

import pandas as pd
import numpy as np
import scipy
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from pydantic import BaseModel, Field, ValidationError

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

try:
    from google.colab import files
    from IPython.display import HTML, display
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False


# =====================================================================
# Pydantic Structural Validation Schemas
# =====================================================================

class ImputationConfig(BaseModel):
    strategy: str = Field(..., description="Imputation logic: 'mean', 'median', 'mode', or 'constant'")
    fill_value: Optional[Any] = Field(None, description="Custom value used if strategy is set to 'constant'")

    @classmethod
    def validate_strategy(cls, strategy: str) -> str:
        valid = ['mean', 'median', 'mode', 'constant']
        if strategy not in valid:
            raise ValueError(f"Strategy must be one of {valid}")
        return strategy


class ScalingConfig(BaseModel):
    method: str = Field(..., description="Scaling choice: 'minmax', 'standard', or 'robust'")

    @classmethod
    def validate_method(cls, method: str) -> str:
        valid = ['minmax', 'standard', 'robust']
        if method not in valid:
            raise ValueError(f"Scaling method must be one of {valid}")
        return method


class EncodingConfig(BaseModel):
    method: str = Field(..., description="Categorical encoding option: 'onehot', 'ordinal', or 'uniform'")

    @classmethod
    def validate_method(cls, method: str) -> str:
        valid = ['onehot', 'ordinal', 'uniform']
        if method not in valid:
            raise ValueError(f"Encoding method must be one of {valid}")
        return method


# =====================================================================
# Main Component: PlottingMethods Base Class
# =====================================================================

class PlottingMethods:
    """
    Component generating Plotly markup outputs wrapped in standard communication dictionary blocks.
    Compatible with direct instantiation or class inheritance.
    """

    def _generate_response_packet(self, fig: go.Figure, success_msg: str) -> Dict[str, Any]:
        """Converts a Plotly figure to a JSON-safe response wrapper containing CDN-backed HTML."""
        fig_id = str(uuid.uuid4())[:8]
        fig_html = to_html(fig, include_plotlyjs='cdn', full_html=False, div_id=fig_id)
        
        meta_data = {"message": success_msg}
        return {
            "status": "success",
            "response": {
                "meta_data": meta_data,
                "data": json.dumps({"figure": fig_html}),
                "message": json.dumps(meta_data)
            }
        }

    def _generate_error_packet(self, error_msg: str) -> Dict[str, Any]:
        """Generates a failure communication layout packet if runtime graphics crash."""
        meta_data = {"message": f"Error: {error_msg}"}
        return {
            "status": "error",
            "response": {
                "meta_data": meta_data,
                "data": json.dumps({"figure": ""}),
                "message": json.dumps(meta_data)
            }
        }

    def plot_bar_chart(self, x: str, y: str, color: Optional[str] = None, barmode: str = 'group', data: Any = None) -> Dict[str, Any]:
        """Generates a responsive stacked or grouped bar chart layout from an absolute sequence array or JSON string."""
        try:
            if isinstance(data, str):
                df = pd.read_json(io.StringIO(data))
            elif isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict) or isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("Valid target data input matrix parameter sequence not supplied.")

            fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, template="plotly_white")
            return self._generate_response_packet(fig, "Bar chart successfully generated")
        except Exception as e:
            return self._generate_error_packet(str(e))

    def plot_pie_chart(self, names: str, values: str, hole: float = 0.4, title: str = 'Pie Chart', data: Any = None) -> Dict[str, Any]:
        """Generates an explicit interactive pie chart containing structured inner radius parameters."""
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif data is not None:
                df = pd.DataFrame(data)
            else:
                raise ValueError("Data frame workspace environment instance reference unavailable.")

            fig = px.pie(df, names=names, values=values, hole=hole, title=title, template="plotly_white")
            return self._generate_response_packet(fig, "Pie chart successfully generated")
        except Exception as e:
            return self._generate_error_packet(str(e))

    def plot_histogram(self, x: str, bins: Optional[List[Union[int, float]]] = None, title: str = 'Histogram', data: Any = None) -> Dict[str, Any]:
        """Plots a localized data frequency structure containing uniform or discrete continuous bucketing definitions."""
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif data is not None:
                df = pd.DataFrame(data)
            else:
                raise ValueError("Target contextual dataframe matrix array is missing or unassigned.")

            if bins is not None:
                series_data = df[x].dropna()
                counts, edges = np.histogram(series_data, bins=bins)
                bin_labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)]
                fig = px.bar(x=bin_labels, y=counts, labels={'x': x, 'y': 'count'}, title=title, template="plotly_white")
            else:
                fig = px.histogram(df, x=x, title=title, template="plotly_white")

            return self._generate_response_packet(fig, "Histogram successfully generated")
        except Exception as e:
            return self._generate_error_packet(str(e))

    def display_image(self, result: Dict[str, Any]) -> None:
        """Renders raw standard communication markup string divs inside Google Colab environments."""
        if result.get('status') == 'success':
            try:
                response_data = json.loads(result['response']['data'])
                plot_html = response_data['figure']
                if COLAB_AVAILABLE:
                    display(HTML(plot_html))
                else:
                    print("Google Colab display interface context absent. HTML generation verified.")
            except Exception as e:
                print(f"Failed to unpack execution visualization graphics stream safely: {e}")
        else:
            err_msg = result.get('response', {}).get('message', 'Unknown package error occurred')
            print(f"Failed to plot: {err_msg}")


# =====================================================================
# Primary Toolkit Module: DataInspector Workflow Class
# =====================================================================

class DataInspector(PlottingMethods):
    """
    Core functional framework for running end-to-end processing operations,
    exploratory analysis structures, and statistical profiling mappings inside notebook workflows.
    """

    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.numeric_df: Optional[pd.DataFrame] = None
        self.categorical_df: Optional[pd.DataFrame] = None
        
        self.numeric_normalized_df: Optional[pd.DataFrame] = None
        self.categorical_normalized_df: Optional[pd.DataFrame] = None
        self.normalized_data_df: Optional[pd.DataFrame] = None
        
        self._null_indicators = ['?', 'N/A', 'NULL', 'null', 'nan', 'NaN', ' ', '']

    # --- 1. Intelligent Data Loading ---
    def upload_data(self) -> None:
        """Prompts user to upload a CSV file in Colab, sanitizes text null flags, and runs data-type inference."""
        if not COLAB_AVAILABLE:
            print("Operation aborted: Interactive standard local imports require active Google Colab workspaces.")
            return

        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return

        file_name = list(uploaded.keys())[0]
        self.df = pd.read_csv(io.BytesIO(uploaded[file_name]), na_values=self._null_indicators)
        self._auto_convert_types()
        self._slice_internal_subsets()
        print(f"\nSuccessfully loaded '{file_name}' with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    def _auto_convert_types(self) -> None:
        """Scans string objects and converts numerical targets safely without breaking true text values."""
        if self.df is None: 
            return
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                attempt = pd.to_numeric(self.df[col], errors='coerce')
                if not attempt.isna().all():
                    self.df[col] = attempt

    def _slice_internal_subsets(self) -> None:
        """Refreshes separated numeric and categorical sub-frames across active processing steps."""
        if self.df is None: 
            return
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])

    # --- 2. Comprehensive Inspection ---
    def inspect_data(self) -> None:
        """Prints global profile shapes, structural missing value sums, and broken down feature types."""
        if self.df is None:
            print("Operation standard warning: Primary workspace active frame has not been set yet.")
            return

        self._slice_internal_subsets()
        print("=== Data Dimensions ===")
        print(f"Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}\n")

        print("=== Column Types Summary ===")
        print(self.df.dtypes.value_counts())
        print(f"\nMissing values global total: {self.df.isna().sum().sum()}")

        if self.numeric_df is not None and not self.numeric_df.empty:
            print("\n=== Numerical Features Profile ===")
            print(self.numeric_df.describe().T)

        if self.categorical_df is not None and not self.categorical_df.empty:
            print("\n=== Categorical Features Profile ===")
            print(self.categorical_df.describe().T)

    # --- 3. Automated Cleaning ---
    def handle_missing_values(self, strategy: str = 'median', fill_value: Optional[Any] = None) -> None:
        """Imputes missing data fields across runtime attributes using Pydantic validated strategy choices."""
        if self.df is None: 
            raise ValueError("Data workspace uninitialized.")
        
        try:
            config = ImputationConfig(strategy=strategy, fill_value=fill_value)
            ImputationConfig.validate_strategy(config.strategy)
        except ValidationError as e:
            raise ValueError(f"Configuration structural strategy mismatch parameters error: {e}")

        for col in self.df.columns:
            if self.df[col].isna().sum() == 0:
                continue
            
            if self.df[col].dtype in [np.float64, np.int64]:
                if config.strategy == 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif config.strategy == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif config.strategy == 'mode':
                    if not self.df[col].mode().empty:
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif config.strategy == 'constant':
                    self.df[col] = self.df[col].fillna(config.fill_value if config.fill_value is not None else 0)
            else:
                if config.strategy == 'constant' and config.fill_value is not None:
                    self.df[col] = self.df[col].fillna(str(config.fill_value))
                else:
                    fallback_mode = self.df[col].mode()
                    self.df[col] = self.df[col].fillna(fallback_mode[0] if not fallback_mode.empty else 'Missing')

        self._slice_internal_subsets()
        print(f"Automated imputation pass executed using strategy parameter logic: '{config.strategy}'.")

    def remove_duplicates(self) -> None:
        """Purges absolute row duplicates from the active dataframe environment workspace."""
        if self.df is None: 
            return
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self._slice_internal_subsets()
        print(f"Removed {initial_count - len(self.df)} exact duplicate rows from current workspace matrix.")

    def remove_outliers_iqr(self, columns: Union[str, List[str]]) -> None:
        """Filters extreme values falling outside 1.5x IQR boundaries on specified features."""
        if self.df is None: 
            return
        target_cols = [columns] if isinstance(columns, str) else columns
        row_mask = pd.Series(True, index=self.df.index)

        for col in target_cols:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                q25 = self.df[col].quantile(0.25)
                q75 = self.df[col].quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                row_mask = row_mask & (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)

        initial_count = len(self.df)
        self.df = self.df[row_mask]
        self.df.reset_index(drop=True, inplace=True)
        self._slice_internal_subsets()
        print(f"IQR processing successfully dropped {initial_count - len(self.df)} rows from array workspace.")

    def interactive_row_deletion(self, indices: List[int]) -> None:
        """Drops rows dynamically based on positional reference labels."""
        if self.df is None: 
            return
        self.df.drop(index=indices, errors='ignore', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self._slice_internal_subsets()

    def drop_columns(self, columns: Union[str, List[str]]) -> None:
        """Discards specific structural attribute dimensions permanently from the processing pool."""
        if self.df is None: 
            return
        target_cols = [columns] if isinstance(columns, str) else columns
        self.df.drop(columns=target_cols, errors='ignore', inplace=True)
        self._slice_internal_subsets()

    # --- 4. Advanced Scaling & Encoding ---
    def extract_normalized_numeric_data(self, method: str = 'robust') -> pd.DataFrame:
        """Transforms structural numeric data features based on MinMax, Standard, or Robust scaling rules."""
        if self.df is None: 
            return pd.DataFrame()
        self._slice_internal_subsets()
        if self.numeric_df is None or self.numeric_df.empty:
            return pd.DataFrame()

        try:
            config = ScalingConfig(method=method)
            ScalingConfig.validate_method(config.method)
        except ValidationError as e:
            raise ValueError(f"Invalid scaling configuration strategy input: {e}")

        if config.method == 'minmax':
            scaler = MinMaxScaler()
        elif config.method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()

        clean_numeric = self.numeric_df.fillna(0)
        scaled_array = scaler.fit_transform(clean_numeric)
        
        self.numeric_normalized_df = pd.DataFrame(scaled_array, columns=self.numeric_df.columns, index=self.df.index)
        return self.numeric_normalized_df

    def extract_normalized_categorical_data(self, method: str = 'onehot') -> pd.DataFrame:
        """Encodes object categories using One-Hot, Ordinal, or Uniform conversion routines."""
        if self.df is None: 
            return pd.DataFrame()
        self._slice_internal_subsets()
        if self.categorical_df is None or self.categorical_df.empty:
            return pd.DataFrame()

        try:
            config = EncodingConfig(method=method)
            EncodingConfig.validate_method(config.method)
        except ValidationError as e:
            raise ValueError(f"Encoding parameters requirements validation fault mismatch: {e}")

        clean_cat = self.categorical_df.fillna('Missing').astype(str)

        if config.method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded_matrix = encoder.fit_transform(clean_cat)
            feature_names = encoder.get_feature_names_out(self.categorical_df.columns)
            self.categorical_normalized_df = pd.DataFrame(encoded_matrix, columns=feature_names, index=self.df.index)
        else:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoded_matrix = encoder.fit_transform(clean_cat)
            self.categorical_normalized_df = pd.DataFrame(encoded_matrix, columns=self.categorical_df.columns, index=self.df.index)

        return self.categorical_normalized_df

    def create_normalized_data_df(self, numeric_method: str = 'robust', categorical_method: str = 'onehot') -> pd.DataFrame:
        """Assembles normalized numerical and categorical fields into a single machine-learning-ready dataframe."""
        num_part = self.extract_normalized_numeric_data(method=numeric_method)
        cat_part = self.extract_normalized_categorical_data(method=categorical_method)
        
        if num_part.empty and cat_part.empty:
            self.normalized_data_df = pd.DataFrame()
        elif num_part.empty:
            self.normalized_data_df = cat_part
        elif cat_part.empty:
            self.normalized_data_df = num_part
        else:
            self.normalized_data_df = pd.concat([num_part, cat_part], axis=1)
            
        return self.normalized_data_df

    # --- 5. Interactive Visualizations ---
    def plot_numerical(self, columns: Union[str, List[str]]) -> None:
        """Generates a complete three-pane plot (Horizontal Violin, Index Scatter, and Histogram) for specific numeric columns."""
        if self.df is None: 
            return
        target_cols = [columns] if isinstance(columns, str) else columns

        for col in target_cols:
            if col not in self.df.columns or self.df[col].dtype not in [np.float64, np.int64]:
                continue

            fig = make_subplots(
                rows=1, cols=3, 
                subplot_titles=('Horizontal Violin Plot', 'Index Distribution Scatter', 'Distribution Histogram')
            )

            fig.add_trace(go.Violin(x=self.df[col], name=col, box_visible=True, points='all', marker_color='#636EFA'), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.df[col], mode='markers', marker=dict(opacity=0.6, color='#EF553B')), row=1, col=2)
            fig.add_trace(go.Histogram(x=self.df[col], marker_color='#00CC96'), row=1, col=3)

            fig.update_layout(
                title_text=f"Multi-Chart Structural Insight Diagnostic: Space Array Vector '{col}'",
                showlegend=False, 
                template="plotly_white"
            )
            
            packet = self._generate_response_packet(fig, f"Rendered analysis grid metrics for: {col}")
            self.display_image(packet)

    def plot_relationship(self, col1: str, col2: str) -> None:
        """Dynamically matches data types between two properties to plot the ideal chart layout."""
        if self.df is None or col1 not in self.df.columns or col2 not in self.df.columns:
            return

        t1, t2 = self.df[col1].dtype, self.df[col2].dtype
        is_num1 = t1 in [np.float64, np.int64]
        is_num2 = t2 in [np.float64, np.int64]

        try:
            if is_num1 and is_num2:
                has_statsmodels = 'statsmodels' in sys.modules
                fig = px.scatter(self.df, x=col1, y=col2, trendline="ols" if has_statsmodels else None, template="plotly_white")
            elif not is_num1 and not is_num2:
                counts_frame = self.df.groupby([col1, col2]).size().reset_index(name='Occurrences Count')
                fig = px.bar(counts_frame, x=col1, y='Occurrences Count', color=col2, barmode='group', template="plotly_white")
            else:
                numeric_target = col2 if is_num2 else col1
                categorical_target = col1 if is_num2 else col2
                fig = px.box(self.df, x=categorical_target, y=numeric_target, points="all", template="plotly_white")

            fig.update_layout(title=f"Context-Aware Relationship Blueprint: '{col1}' mapped against '{col2}'")
            packet = self._generate_response_packet(fig, f"Relationship layout plotted for {col1} and {col2}")
            self.display_image(packet)
        except Exception as e:
            self.display_image(self._generate_error_packet(str(e)))

    # --- 6. Deep Statistical Insights ---
    def _calculate_cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculates Cramér's V correlation metric for nominal feature associations."""
        contingency_table = pd.crosstab(x, y)
        if contingency_table.sum().sum() == 0:
            return 0.0
        chi2 = chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        r, k = contingency_table.shape
        denominator = n * min(k - 1, r - 1)
        return np.sqrt(chi2 / denominator) if denominator > 0 else 0.0

    def _calculate_eta_squared(self, categories: pd.Series, values: pd.Series) -> float:
        """Computes Eta-squared via ANOVA to capture the association between nominal and continuous variables."""
        valid_indices = ~(categories.isna() | values.isna())
        c_clean = categories[valid_indices]
        v_clean = values[valid_indices]
        
        unique_groups = c_clean.unique()
        if len(unique_groups) <= 1 or len(v_clean) == 0:
            return 0.0
            
        grouped_data = [v_clean[c_clean == group] for group in unique_groups]
        try:
            f_stat, _ = f_oneway(*grouped_data)
            ss_between_approx = f_stat * (len(unique_groups) - 1)
            ss_total_approx = ss_between_approx + (len(v_clean) - len(unique_groups))
            return np.sqrt(ss_between_approx / ss_total_approx) if ss_total_approx > 0 else 0.0
        except Exception:
            return 0.0

    def plot_all_associations_heatmap(self) -> None:
        """Computes an all-inclusive association matrix (Pearson, Cramér's V, and Eta) across mixed data types."""
        if self.df is None: 
            return
        
        columns_list = self.df.columns.tolist()
        matrix_dim = len(columns_list)
        association_matrix = np.zeros((matrix_dim, matrix_dim))

        for i in range(matrix_dim):
            for j in range(matrix_dim):
                if i == j:
                    association_matrix[i, j] = 1.0
                    continue

                col1, col2 = columns_list[i], columns_list[j]
                is_num1 = self.df[col1].dtype in [np.float64, np.int64]
                is_num2 = self.df[col2].dtype in [np.float64, np.int64]

                try:
                    if is_num1 and is_num2:
                        val = self.df[col1].corr(self.df[col2], method='pearson')
                        association_matrix[i, j] = val if not np.isnan(val) else 0.0
                    elif not is_num1 and not is_num2:
                        association_matrix[i, j] = self._calculate_cramers_v(self.df[col1], self.df[col2])
                    else:
                        num_col = col1 if is_num1 else col2
                        cat_col = col2 if is_num1 else col1
                        association_matrix[i, j] = self._calculate_eta_squared(self.df[cat_col], self.df[num_col])
                except Exception:
                    association_matrix[i, j] = 0.0

        association_matrix = np.nan_to_num(association_matrix)

        fig = go.Figure(data=go.Heatmap(
            z=association_matrix,
            x=columns_list,
            y=columns_list,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(association_matrix, 2),
            texttemplate="%{text}",
            hoverinfo="z"
        ))

        fig.update_layout(
            title='Unified Structural Attribute Association Matrix (Pearson, Cramér\'s V, & Eta-squared)',
            template='plotly_white'
        )
        
        packet = self._generate_response_packet(fig, "Unified multi-type association matrix mapped successfully")
        self.display_image(packet)