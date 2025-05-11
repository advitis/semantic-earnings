import plotly.express as px
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_plot_columns(df):
    df["label"] = df.apply(lambda r: f"{r['company'].title()} {r['year']} Q{r['quarter']}", axis=1)
    df["mention_size"] = MinMaxScaler((10, 60)).fit_transform(df[["mention_count"]])
    return df


def plot_semantic_map(df):
    df = add_plot_columns(df)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="company",
        size="mention_size",
        hover_name="label",
        hover_data={"text": True, "mention_count": True, "x": False, "y": False},
        title="Semantic Map of AI Discussion Across Earnings Calls"
    )
    fig.update_layout(legend_title_text='Company', height=700)
    fig.show()
