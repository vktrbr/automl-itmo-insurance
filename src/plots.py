import os
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_log_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import plotly.express as px
from loguru import logger

import numpy as np

VS_1500_500 = {"width": 1500, "height": 500, "scale": 3}


def feature_analysis(
    df: pd.DataFrame,
    feature_column: str,
    target_column: str,
    date_column: str,
) -> dict:
    """
    Perform feature analysis on a feature column for regression task.

    :param df: Input DataFrame
    :param feature_column: Feature column to analyze
    :param target_column: Target column for regression
    :param date_column: Date column for time-based analysis
    :return: Dictionary with paths to the saved HTML and PNG files
    """

    # Create a layout for the figure
    fig = get_layout()

    # Perform one-dimensional analysis
    fig = one_dimensional_analysis(df.copy(), feature_column, target_column, fig)

    # Perform feature stability analysis with RMSLE
    res = feature_stability_analysis_with_rmsle(
        df.copy(), feature_column, target_column, date_column, fig
    )

    fig = res["fig"]

    fig.update_layout(
        legend={"title": "1D Analysis", "y": 1},
        legend2={"title": "Feature Stability", "y": 1},
    )

    fig.update_layout(title=f"<b>{feature_column} analysis. RMSLE: {res['rmsle']:.2%}")

    return {"fig": fig}


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
        subplot_titles=("1D Analysis: Mean by Buckets", "Feature Stability: RMSLE"),
        horizontal_spacing=0.12,
    )


def one_dimensional_analysis(
    df: pd.DataFrame,
    feature_column: str,
    target_column: str,
    fig: go.Figure,
    n_buckets: int = 10,
):
    """
    Perform one-dimensional analysis on a feature column.

    :param df: Input DataFrame
    :param feature_column: Feature column name
    :param target_column: Target column name
    :param fig: Plotly Figure
    :param n_buckets: Number of buckets
    :return: Updated figure
    """
    legend_group = "legend1"

    # Binning the feature
    if df[feature_column].dtype == "object" or df[feature_column].nunique() < n_buckets:
        df["Bucket"] = df[feature_column]
    else:
        df["Bucket"] = pd.qcut(df[feature_column], n_buckets, duplicates="drop")
        df["Bucket"] = df["Bucket"].apply(lambda x: round(x.left, 4))

    bucket_means = (
        df.groupby("Bucket", observed=False)[target_column]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Add bar plot for bucket means
    fig.add_trace(
        go.Scatter(
            x=bucket_means["Bucket"].astype(str),
            y=bucket_means["mean"],
            name=f"Mean {target_column}",
            legendgroup=legend_group,
            mode="lines+markers",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=bucket_means["Bucket"].astype(str),
            y=bucket_means["count"],
            name="Count",
            opacity=0.7,
            legendgroup=legend_group,
        ),
        row=1,
        col=1,
    )

    return fig


def feature_stability_analysis_with_rmsle(
    df: pd.DataFrame,
    feature_column: str,
    target_column: str,
    date_column: str,
    fig: go.Figure,
):
    """
    Perform feature stability analysis on a feature column with RMSLE.

    :param df: Input DataFrame
    :param feature_column: Feature column name
    :param target_column: Target column name
    :param date_column: Date column name
    :param fig: Plotly Figure
    :return: Updated figure
    """
    legend_group = "legend2"

    # Extract the month from the date column
    df: pd.DataFrame = df.copy()
    df["Month"] = pd.to_datetime(df[date_column]).dt.to_period("M").astype(str)

    pipeline = Pipeline(
        [
            ("encoder", TargetEncoder()),
            ("imputer", SimpleImputer(strategy="mean")),
            ("regressor", LinearRegression()),
        ]
    )
    x = df[[feature_column]]
    y = df[target_column]
    pipeline.fit(x, y)
    y_pred = pipeline.predict(x)
    df["Prediction"] = y_pred

    # Calculate RMSLE over months
    rmsle_df = df.groupby(["Month"], as_index=False).apply(
        lambda x_: root_mean_squared_log_error(x_[target_column], x_["Prediction"]),
        include_groups=False,
    )
    rmsle_df.columns = ["Month", "RMSLE"]

    # Add RMSLE line plot
    fig.add_trace(
        go.Scatter(
            x=rmsle_df["Month"],
            y=rmsle_df["RMSLE"],
            name="RMSLE",
            mode="lines+markers",
            line=dict(color="blue", width=2),
            legendgroup=legend_group,
        ),
        row=1,
        col=2,
    )

    rmsle = root_mean_squared_log_error(df[target_column], df["Prediction"])

    # Update layout
    fig["layout"]["yaxis3"].update(
        title="RMSLE",
        tickformat=".3f",
    )

    return {"fig": fig, "rmsle": rmsle}


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


def numerical_groupped_data(data, score, n_buckets=5, values=None, rnd=2):
    df_null = data[data[score].isna()].copy()
    df = data[data[score].notna()].copy()

    if values is None:
        if df[score].nunique(
            dropna=False
        ) < n_buckets * 2 or not pd.api.types.is_numeric_dtype(df[score]):
            df[f"BUCKET_{score}"] = "bin:  " + df[score].astype(str).fillna("None")
        else:
            df[f"BUCKET_{score}"] = pd.qcut(
                df[score], n_buckets, duplicates="drop"
            ).apply(
                lambda interval: f"({interval.left:.{rnd}f}, {interval.right:.{rnd}f}]"
            )
    else:
        n_buckets = len(values) + 1
        values = np.array(values)
        if values[-1] < df[score].max():
            values = np.append(values, df[score].max())
        if values[0] > df[score].min():
            values = np.append(values, df[score].min())
        values = [np.round(x, rnd) for x in sorted(values)]
        if df[score].nunique(
            dropna=False
        ) < n_buckets * 2 or not pd.api.types.is_numeric_dtype(data[score]):
            df[f"BUCKET_{score}"] = "bin:  " + df[score].astype(str).fillna("None")
        else:
            df[f"BUCKET_{score}"] = pd.cut(df[score], bins=values, duplicates="drop")

    df_null[f"BUCKET_{score}"] = "None"
    df = pd.concat([df, df_null], ignore_index=True)
    return df


def categorical_groupped_data(data, score, weight=None, n_buckets=5):
    df = data.copy()
    df[score] = df[score].fillna("None")
    if weight is None:
        weight = "cnt"
        df[weight] = 1

    categorical_df = (
        pd.pivot_table(data=df, index=score, values=weight, aggfunc="sum")
        .reset_index()
        .sort_values([weight], ascending=False)
    )
    categorical_df = categorical_df.head(n_buckets)
    columns = categorical_df[score].values
    df[f"BUCKET_{score}"] = df[score]
    df.loc[~df[score].isin(columns), f"BUCKET_{score}"] = "Others"
    return df


def groupped_data(data, score, weight=None, n_buckets=5, values=None, rnd=2):
    df = data[[score]].dropna()
    try:
        df[score] = df[score].astype(float)
        return numerical_groupped_data(
            data=data, score=score, n_buckets=n_buckets, values=values, rnd=rnd
        )
    except Exception as e:
        logger.error(e)
        return categorical_groupped_data(
            data=data, score=score, weight=weight, n_buckets=n_buckets
        )


def crosstab(
    data, score_1, score_2, target, weight=None, n_buckets: int = 5, values=None, rnd=2
):
    """

    :param data:
    :param score_1:
    :param score_2:
    :param target:
    :param n_buckets:
    :return:
    """
    data = data[data[target].notna()].copy()

    data = groupped_data(
        data=data,
        score=score_1,
        weight=weight,
        n_buckets=n_buckets,
        values=values,
        rnd=rnd,
    )

    data = groupped_data(
        data=data,
        score=score_2,
        weight=weight,
        n_buckets=n_buckets,
        values=values,
        rnd=rnd,
    )

    data.columns = [col.upper() for col in data.columns]
    score_1, score_2, target = score_1.upper(), score_2.upper(), target.upper()

    data = data.sort_values([f"BUCKET_{score_2}", f"BUCKET_{score_1}"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[target.upper(), "COUNTS"],
        horizontal_spacing=0.3,
    )

    data_risk = data.pivot_table(
        index=f"BUCKET_{score_1}",
        columns=f"BUCKET_{score_2}",
        values=target,
        aggfunc="mean",
    )
    data_cnt = data.pivot_table(
        index=f"BUCKET_{score_1}",
        columns=f"BUCKET_{score_2}",
        values=target,
        aggfunc="count",
    )

    data_risk.sort_index(axis=0, inplace=True)
    data_risk.sort_index(axis=1, inplace=True)

    data_cnt.sort_index(axis=0, inplace=True)
    data_cnt.sort_index(axis=1, inplace=True)

    fig.add_trace(px.imshow(data_risk, text_auto=".3f").data[0], 1, 1)
    fig.add_trace(px.imshow(data_cnt, text_auto=True).data[0], 1, 2)

    fig.update_traces(coloraxis="coloraxis1", selector=dict(xaxis="x"))
    fig.update_traces(coloraxis="coloraxis2", selector=dict(xaxis="x2"))

    cmax = data_risk.max().max()
    cmin = data_risk.min().min()
    cmid = np.median(data_risk.values.flatten())

    fig.update_layout(
        title_text=f"Crosstab {score_1} x {score_2}. {target.upper()}",
        coloraxis=dict(
            showscale=True,
            colorscale="Reds",
            colorbar_x=0.35,
            cmin=cmin,
            cmax=cmax,
            cmid=cmid,
        ),
        yaxis_autorange="reversed",
        yaxis2_autorange="reversed",
        coloraxis2=dict(showscale=True, colorscale="Greys"),
    )
    fig.update_xaxes(title_text=score_2, row=1, col=1)
    fig.update_xaxes(title_text=score_2, row=1, col=2)
    fig.update_yaxes(title_text=score_1, row=1, col=1)

    return fig
