import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


def add_plot_columns(df):
    df["label"] = df.apply(lambda r: f"{r['company'].title()} {r['year']} Q{r['quarter']}", axis=1)
    scaler = MinMaxScaler((10, 60))
    df["point_size"] = scaler.fit_transform(df[["sentence_count"]])
    df["year_q"] = df["year"].astype(str) + "‑Q" + df["quarter"].astype(str)
    return df


def plot_semantic_map(df):
    df = add_plot_columns(df)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="company",
        size="point_size",
        text="label",
        hover_data={
            "label": False,
            "cluster_name": True,
            "sentence_count": True,
            "x": False,
            "y": False
        },
        title="Semantic Map of AI Discussion Across Earnings Calls",
        custom_data=["cluster_name"]
    )

    fig.update_traces(
        textposition="top center",
        textfont_size=8
    )

    for name in df["cluster_name"].unique():
        sub = df[df["cluster_name"] == name]
        fig.add_annotation(
            x=sub["x"].mean(),
            y=sub["y"].mean(),
            text=name.title(),
            showarrow=False,
            font_size=10,
            bgcolor="rgba(255,255,255,0.7)"
        )
    fig.update_layout(legend_title_text='Company', height=700)
    fig.show()
