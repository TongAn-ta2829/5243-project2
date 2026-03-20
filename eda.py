from __future__ import annotations
from random import choices

from htmltools.tags import col
from shiny import ui, reactive, render
from shiny.express import output
from shiny.render import DataGrid
import pandas as pd


def register_eda_server(input, output, session, *, raw_data, cleaned_data, featured_data):
    @reactive.calc
    def eda_data():
        source = input.eda_source()
        if source == "raw":
            return raw_data()
        if source == "featured":
            return featured_data()
        return cleaned_data()

    def _eda_numeric_cols(df: pd.DataFrame):
        return df.select_dtypes(include="number").columns.tolist()

    def _eda_categorical_cols(df: pd.DataFrame):
        return df.select_dtypes(exclude="number").columns.tolist()

    @output
    @render.ui
    def eda_col_ui():
        df = eda_data()
        plot_type = input.eda_plot_type()

        if df is None:
            return ui.tags.div()

        if plot_type in {"hist", "box"}:
            choices = _eda_numeric_cols(df)
            return ui.input_select("eda_col", "Numeric column", choices=choices)

        if plot_type == "bar":
            choices = _eda_categorical_cols(df)
            return ui.input_select("eda_cat_col", "Categorical column", choices=choices)

        return ui.tags.div()

    @output
    @render.ui
    def eda_x_ui():
        df = eda_data()
        if df is None or input.eda_plot_type() != "scatter":
            return ui.tags.div()
        choices = _eda_numeric_cols(df)
        return ui.input_select("eda_x", "X (numeric)", choices=choices)

    @output
    @render.ui
    def eda_y_ui():
        df = eda_data()
        if df is None or input.eda_plot_type() != "scatter":
            return ui.tags.div()
        choices = _eda_numeric_cols(df)
        return ui.input_select("eda_y", "Y (numeric)", choices=choices)

    @output
    @render.data_frame
    def eda_summary():
        df = eda_data()
        if df is None:
            return DataGrid(pd.DataFrame({"Info": ["No dataset available for EDA"]}))

        rows = []
        for col in df.columns:
            s = df[col]
            row = {
                "Column": col,
                "Dtype": str(s.dtype),
                "Missing": int(s.isna().sum()),
                "Unique": int(s.nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(s):
                s_num = pd.to_numeric(s, errors="coerce")
                row.update(
                    {
                        "Mean": float(s_num.mean()) if s_num.notna().any() else None,
                        "Std": float(s_num.std()) if s_num.notna().any() else None,
                        "Min": float(s_num.min()) if s_num.notna().any() else None,
                        "Median": float(s_num.median()) if s_num.notna().any() else None,
                        "Max": float(s_num.max()) if s_num.notna().any() else None,
                    }
                )
            else:
                vc = s.astype(str).value_counts(dropna=False)
                row.update(
                    {
                        "Top": (vc.index[0] if len(vc) else None),
                        "Top Count": int(vc.iloc[0]) if len(vc) else None,
                    }
                )

            rows.append(row)

        summary_df = pd.DataFrame(rows)
        return DataGrid(summary_df, width="100%")

    @output
    @render.text
    def eda_plot_note():
        try:
            import matplotlib.pyplot as _  # type: ignore[import-not-found]  # noqa: F401
            return ""
        except Exception:
            return (
                "Plotting requires the 'matplotlib' package, which isn't installed in this environment.\n"
                "To enable EDA plots, install it (e.g., `python3 -m pip install matplotlib`)."
            )

    @output
    @render.plot
    def eda_plot():
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        except Exception:
            return None

        df = eda_data()
        plot_type = input.eda_plot_type()

        fig, ax = plt.subplots(figsize=(8, 4.5))

        if df is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "No dataset available for EDA.", ha="center", va="center")
            return fig

        dropna = bool(input.eda_dropna())

        def _finish(title: str):
            ax.set_title(title)
            fig.tight_layout()
            return fig

        if plot_type == "hist":
            col = input.eda_col()
            if not col or col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                ax.axis("off")
                ax.text(0.5, 0.5, "Select a numeric column for the histogram.", ha="center", va="center")
                return fig
            s = df[col]
            if dropna:
                s = s.dropna()
            ax.hist(s, bins=int(input.eda_bins()), edgecolor="white")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            return _finish(f"Histogram of {col}")

        if plot_type == "box":
            col = input.eda_col()
            if not col or col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                ax.axis("off")
                ax.text(0.5, 0.5, "Select a numeric column for the boxplot.", ha="center", va="center")
                return fig
            s = df[col]
            if dropna:
                s = s.dropna()
            ax.boxplot(s, vert=True)
            ax.set_xticks([1])
            ax.set_xticklabels([col], rotation=0)
            ax.set_ylabel(col)
            return _finish(f"Boxplot of {col}")

        if plot_type == "scatter":
            x = input.eda_x()
            y = input.eda_y()
            if not x or not y or x not in df.columns or y not in df.columns:
                ax.axis("off")
                ax.text(0.5, 0.5, "Select X and Y numeric columns for the scatter plot.", ha="center", va="center")
                return fig
            if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
                ax.axis("off")
                ax.text(0.5, 0.5, "X and Y must both be numeric.", ha="center", va="center")
                return fig
            plot_df = df[[x, y]]
            if dropna:
                plot_df = plot_df.dropna()
            ax.scatter(plot_df[x], plot_df[y], alpha=0.7)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            return _finish(f"Scatter: {y} vs {x}")

        if plot_type == "bar":
            col = input.eda_cat_col()
            if not col or col not in df.columns:
                ax.axis("off")
                ax.text(0.5, 0.5, "Select a categorical column for the bar chart.", ha="center", va="center")
                return fig
            s = df[col]
            if dropna:
                s = s.dropna()
            counts = s.astype(str).value_counts().head(int(input.eda_top_n()))
            ax.barh(counts.index[::-1], counts.values[::-1])
            ax.set_xlabel("Count")
            ax.set_ylabel(col)
            return _finish(f"Top {len(counts)} categories in {col}")

        ax.axis("off")
        ax.text(0.5, 0.5, "Unknown plot type.", ha="center", va="center")
        return fig

    @output
    @render.ui
    def eda_filter_col_ui():
        df = eda_data()
        if df is None:
            return ui.tags.div()

        choices = ["None"] + df.columns.tolist()
        return ui.input_select("eda_filter_col", "Filter column", choices=choices, selected="None")
    
    @output
    @render.ui
    def eda_filter_value_ui():
        df = eda_data()
        if df is None:
            return ui.tags.div()

        col = input.eda_filter_col()
        if not col or col == "None" or col not in df.columns:
            return ui.tags.div()

        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if len(s_num) == 0:
                return ui.tags.div("No numeric values available for filtering.")
            return ui.TagList(
                ui.input_numeric("eda_filter_min", f"{col} min", float(s_num.min())),
                ui.input_numeric("eda_filter_max", f"{col} max", float(s_num.max())),
            )

        vals = sorted(s.dropna().astype(str).unique().tolist())
        if len(vals) == 0:
            return ui.tags.div("No category values available for filtering.")

        return ui.input_selectize(
            "eda_filter_levels",
            f"{col} values",
            choices=vals,
            selected=vals[: min(5, len(vals))],
            multiple=True,
        )