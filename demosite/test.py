import plotly.graph_objects as go

fig = go.Figure()

config = {'displaylogo': False}

fig.add_trace(
    go.Scatter(
        x=[1, 2, 3],
        y=[1, 3, 1]))

fig.show(config=config)
