"""Plotly-Visualisierungen: Ähnlichkeitsgraph (Punkte + Kanten - in dieser Reihe neu),
Eigenwert-Scree-Plot (ebenfalls neu), Spektral-Einbettung, finales Ergebnis,
Kleinmultiples und Methoden-Vergleichsdiagramm."""

import numpy as np

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _cluster_color(index, n_clusters):
    if n_clusters <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
    hue = (index * 360.0 / n_clusters) % 360
    return f"hsl({hue:.1f}, 65%, 50%)"


def _axis_range(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.1 or 1.0
    pady = (ymax - ymin) * 0.1 or 1.0
    return [xmin - padx, xmax + padx], [ymin - pady, ymax + pady]


def _cluster_traces(data, labels, legend):
    import plotly.graph_objects as go

    traces = []
    cluster_ids = sorted(set(labels.tolist()))
    n_clusters = len(cluster_ids)
    show_legend = legend and n_clusters <= 12
    for index, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = _cluster_color(index, n_clusters)
        traces.append(
            go.Scatter(
                x=data[mask, 0], y=data[mask, 1], mode="markers", name=f"Cluster {cid + 1}",
                showlegend=show_legend,
                marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
                hoverinfo="skip",
            )
        )
    return traces


def _build_scatter(data, labels, height, legend):
    import plotly.graph_objects as go

    fig = go.Figure()
    for trace in _cluster_traces(data, np.asarray(labels), legend):
        fig.add_trace(trace)

    xr, yr = _axis_range(data)
    layout_kwargs = dict(
        template="plotly_white", height=height,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        showlegend=legend,
        margin=dict(t=40 if legend else 5, l=10 if legend else 5, r=10 if legend else 5, b=10 if legend else 5),
    )
    if legend:
        layout_kwargs["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(**layout_kwargs)
    return fig


def build_scatter_figure(data, labels):
    return _build_scatter(data, labels, height=460, legend=True)


def build_mini_scatter_figure(data, labels):
    return _build_scatter(data, labels, height=220, legend=False)


def build_graph_figure(data, similarity):
    """Zeichnet den Ähnlichkeitsgraphen: eine Linie je Kante (Deckkraft nach Kantengewicht,
    schwächere Kanten heller), Punkte obendrauf - macht die tatsächliche Graphstruktur
    sichtbar, die die Eigenzerlegung als Eingabe bekommt."""
    import plotly.graph_objects as go

    n = len(data)
    edge_x, edge_y, edge_opacity = [], [], []
    max_weight = similarity.max() if similarity.max() > 0 else 1.0
    for i in range(n):
        for j in range(i + 1, n):
            w = similarity[i, j]
            if w > 0:
                edge_x += [data[i, 0], data[j, 0], None]
                edge_y += [data[i, 1], data[j, 1], None]
                edge_opacity.append(w / max_weight)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines", line=dict(color="#9aa6ba", width=1),
            opacity=0.5, hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[:, 0], y=data[:, 1], mode="markers",
            marker=dict(color="#1f77b4", size=7, line=dict(width=0.5, color="white")),
            hoverinfo="skip", showlegend=False,
        )
    )
    xr, yr = _axis_range(data)
    fig.update_layout(
        template="plotly_white", height=380,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        margin=dict(t=10, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_scree_plot(eigenvalues, k, max_shown=15):
    """Sortierte Eigenwerte des Laplace (die ersten `max_shown`), mit einer markierten
    Linie bei Index k - die Standard-Heuristik zur k-Wahl bei Spectral Clustering: eine
    grosse Luecke nach dem k-ten Eigenwert deutet auf k gut getrennte Cluster hin
    (Analogon zu dbscan-demos k-Distanz-Knick)."""
    import plotly.graph_objects as go

    shown = list(eigenvalues[:max_shown])
    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(shown) + 1)), y=shown, mode="lines+markers",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_vline(x=k + 0.5, line=dict(color="#d62728", dash="dash"), annotation_text=f"k={k}")
    fig.update_layout(
        template="plotly_white", height=320,
        xaxis=dict(title="Eigenwert-Index (aufsteigend)", fixedrange=True, dtick=1),
        yaxis=dict(title="Eigenwert", fixedrange=True),
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_embedding_figure(embedding, labels):
    """Streudiagramm der ersten zwei Koordinaten der Spektral-Einbettung, eingefärbt
    nach der finalen Clusterzuordnung - macht sichtbar, warum k-Means AUF DIESER
    Einbettung leicht funktioniert, obwohl es auf den rohen Koordinaten scheitern würde:
    die Einbettung macht aus nicht-konvexen Formen konvexe, gut trennbare Gruppen."""
    import plotly.graph_objects as go

    labels = np.asarray(labels)
    fig = go.Figure()
    cluster_ids = sorted(set(labels.tolist()))
    for index, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = _cluster_color(index, len(cluster_ids))
        fig.add_trace(
            go.Scatter(
                x=embedding[mask, 0], y=embedding[mask, 1], mode="markers",
                marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
                hoverinfo="skip", showlegend=False,
            )
        )
    fig.update_layout(
        template="plotly_white", height=380,
        xaxis=dict(title="Eigenvektor 1", fixedrange=True),
        yaxis=dict(title="Eigenvektor 2", fixedrange=True),
        margin=dict(t=10, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_method_comparison_chart(scores):
    import plotly.graph_objects as go

    labels = ["Spectral Clustering\n(aktuelles k)", "k-Means-Baseline\n(aktuelles k)", "Spectral Clustering\n(wahres k)"]
    values = [scores["spectral"], scores["kmeans_baseline"], scores["spectral_at_true_k"]]
    fig = go.Figure(
        go.Bar(x=labels, y=values, marker_color=["#1f77b4", "#d68a2e", "#2ca02c"])
    )
    fig.update_layout(
        template="plotly_white", height=280,
        yaxis=dict(title="Rand-Index", range=[0, 1.05], fixedrange=True),
        xaxis=dict(fixedrange=True), margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig
