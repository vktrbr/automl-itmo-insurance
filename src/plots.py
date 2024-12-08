import os
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


VS_1500_500 = {"width": 1500, "height": 500, "scale": 3}


def feature_analysis(
    df: pd.DataFrame,
    feature_column: str,
    target_column: str,
    date_column: str,
    output_path: Path,
) -> dict:
    """
    Perform feature analysis on a feature column for regression task.

    :param df: Input DataFrame
    :param feature_column: Feature column to analyze
    :param target_column: Target column for regression
    :param date_column: Date column for time-based analysis
    :param output_path: Path to save the figures
    :return: Dictionary with paths to the saved HTML and PNG files
    """

    # Create a layout for the figure
    fig = get_layout()

    # Perform one-dimensional analysis
    fig = one_dimensional_analysis(df.copy(), feature_column, target_column, fig)

    # Perform feature stability analysis with MAPE
    fig = feature_stability_analysis_with_mape(
        df.copy(), feature_column, target_column, date_column, fig
    )

    # Save the chart
    result = save_chart(fig, target_column, feature_column, output_path)

    return {"fig": fig, "paths": result}


def save_chart(
    fig: go.Figure, feature_column: str, output_path: Path, vs: dict | None = None
):
    """
    Save a chart to files.

    :param fig: Plotly figure
    :param feature_column: Feature column name
    :param output_path: Output directory path
    :param vs: Visual settings
    :return: Paths to saved files
    """
    os.makedirs(output_path / "html", exist_ok=True)
    os.makedirs(output_path / "png", exist_ok=True)

    # Save the figure
    output_path_html = output_path / f"html/{feature_column}_analysis.html"
    output_path_png = output_path / f"png/{feature_column}_analysis.png"

    fig.write_html(str(output_path_html))
    vs = vs or VS_1500_500
    vs.pop("renderer", None)
    fig.write_image(str(output_path_png), **vs)

    return {"html_path": output_path_html, "png_path": output_path_png}


def get_layout():
    """
    Create a layout for the figure with two subplots for one-dimensional
    analysis and feature stability.

    :return: Plotly Figure
    """
    return make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=("1D Analysis: Mean by Buckets", "Feature Stability: MAPE"),
        horizontal_spacing=0.12,
    )


def one_dimensional_analysis(
    df: pd.DataFrame, feature_column: str, target_column: str, fig: go.Figure
):
    """
    Perform one-dimensional analysis on a feature column.

    :param df: Input DataFrame
    :param feature_column: Feature column name
    :param target_column: Target column name
    :param fig: Plotly Figure
    :return: Updated figure
    """
    legend_group = "legend1"

    # Binning the feature
    if df[feature_column].dtype == "object" or df[feature_column].nunique() < 50:
        df["Bucket"] = df[feature_column]
    else:
        df["Bucket"] = pd.qcut(df[feature_column], 10, duplicates="drop")

    bucket_means = df.groupby("Bucket")[target_column].mean().reset_index()

    # Add bar plot for bucket means
    fig.add_trace(
        go.Bar(
            x=bucket_means["Bucket"].astype(str),
            y=bucket_means[target_column],
            name=f"Mean {target_column}",
            opacity=0.7,
            legendgroup=legend_group,
        ),
        row=1,
        col=1,
    )

    return fig


def feature_stability_analysis_with_mape(
    df: pd.DataFrame,
    feature_column: str,
    target_column: str,
    date_column: str,
    fig: go.Figure,
    buckets: int = 3,
):
    """
    Perform feature stability analysis on a feature column with MAPE.

    :param df: Input DataFrame
    :param feature_column: Feature column name
    :param target_column: Target column name
    :param date_column: Date column name
    :param fig: Plotly Figure
    :param buckets: Number of buckets
    :return: Updated figure
    """
    legend_group = "legend2"

    # Extract the month from the date column
    df["Month"] = pd.to_datetime(df[date_column]).dt.to_period("M").astype(str)

    # Calculate MAPE per month
    mape_data = []
    for month, group in df.groupby("Month"):
        X = group[[feature_column]].dropna().values.reshape(-1, 1)
        y = group[target_column].dropna().values

        if len(X) > 1 and len(y) > 1:
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            mape = mean_absolute_percentage_error(y, predictions)
            mape_data.append({"Month": month, "MAPE": mape})

    mape_df = pd.DataFrame(mape_data)

    # Add MAPE line plot
    fig.add_trace(
        go.Scatter(
            x=mape_df["Month"],
            y=mape_df["MAPE"],
            name="MAPE",
            mode="lines+markers",
            line=dict(color="blue", width=2),
            legendgroup=legend_group,
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig["layout"]["yaxis3"].update(
        title="MAPE",
        tickformat=".1%",
    )

    return fig
