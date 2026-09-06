"""Spectral Clustering für Sammel-Routen bei nicht-konvexen Formen - interaktive
Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo EIN Verfahren - Spectral Clustering, über Graphentheorie und
lineare Algebra statt Dichte. Achtes Stück der "Konzepte"-Reihe, dritter unabhängiger
Zweig ab kmeans-demo (siehe README für die Einordnung: löst k-Means' Nicht-Konvexitäts-
Annahme, aber NICHT dessen "k muss feststehen"-Schwäche - ein bewusster Kontrast zu
DBSCAN/HDBSCAN).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sp_constants as C
from sp_algorithm import run
from sp_evaluation import eigenvalue_gap, method_comparison
from sp_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from sp_scenario import generate_instance
from sp_visualization import (
    build_embedding_figure,
    build_graph_figure,
    build_method_comparison_chart,
    build_mini_scatter_figure,
    build_scatter_figure,
    build_scree_plot,
)

st.set_page_config(page_title="Spectral Clustering – Sebastian Hanisch", layout="wide")

PHASES = ["1. Ähnlichkeitsgraph", "2. Eigenwerte", "3. Spektral-Einbettung", "4. Ergebnis"]


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, target_k, spread, shape, seed, n_neighbors):
    """`k` erzeugt die wahre Anzahl Gruppen im Szenario, `target_k` ist die Clusterzahl,
    die Spectral Clustering tatsächlich verwenden soll - bewusst ENTKOPPELT (wie
    agglomerative-demos `k`/`target_k`), sonst könnte "falsches k" nie von der wahren
    Clusterzahl abweichen."""
    instance = generate_instance(n_points, k, spread, shape, seed)
    result = run(instance.as_array(), target_k, n_neighbors, seed)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_mini_run(instance, target_k, n_neighbors):
    return run(instance.as_array(), target_k, n_neighbors, C.COMPARISON_SEED)


@st.cache_data(show_spinner=False)
def _compute_method_comparison(instance, target_k, n_neighbors):
    return method_comparison(instance.as_array(), instance.true_labels, target_k, n_neighbors, C.COMPARISON_SEED)


st.title("🕸️ Spectral Clustering: nicht-konvexe Sammel-Routen ohne Dichtebegriff")
st.markdown(
    """
dbscan-demo hat gezeigt, wie **DBSCAN** k-Means' Annahme konvexer/kugelförmiger Cluster
über den lokalen Dichtebegriff behebt. Diese Demo zeigt einen völlig anderen Weg zum
**selben** Ziel: **Spectral Clustering** baut aus den Daten einen **Ähnlichkeitsgraphen**,
zerlegt dessen **Graph-Laplace-Matrix** in Eigenvektoren und clustert die daraus
resultierende Einbettung - reine **Graphentheorie und lineare Algebra**, keine Dichte.
Wichtiger, ehrlicher Unterschied zu DBSCAN: die Clusterzahl k muss weiterhin vorab
feststehen - Spectral Clustering behebt genau EINE der beiden k-Means-Schwächen, nicht
beide auf einmal. Genau **wie** das funktioniert, erklärt der aufgeklappte Abschnitt direkt
darunter - bevor weiter unten live geprüft wird, was das Verfahren kann und was nicht.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren "
    "vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - **ein** "
    "Verfahren an einem wachsenden Beispiel: die Schwierigkeitsachse ist hier die Wahl des "
    "Ähnlichkeitsgraphen (`n_neighbors`) - ein Parameter des einen gezeigten Verfahrens, "
    "kein Methodenvergleich."
)

with st.expander("So funktioniert Spectral Clustering", expanded=True):
    st.markdown(
        """
Spectral Clustering (Ng, Jordan & Weiss, 2002) durchläuft vier Schritte - **einmalig**,
nicht iterativ wie k-Means/EM/Gibbs-Sampling:

1. **Ähnlichkeitsgraph**: jeder Punkt wird mit seinen `n_neighbors` nächsten Nachbarn
   verbunden (symmetrisiert), Kanten werden nach Distanz gewichtet (Gauß-Kernel) - nahe
   Punkte bekommen ein starkes, entfernte ein schwaches oder gar kein Gewicht.
2. **Graph-Laplace-Matrix**: eine Matrix, deren Eigenvektoren die Struktur des Graphen
   codieren - insbesondere gilt ein klassischer Satz der Spektralgraphentheorie: die
   Anzahl der (nahezu) Null-Eigenwerte entspricht der Anzahl der Zusammenhangskomponenten
   des Graphen.
3. **Eigenzerlegung**: die k Eigenvektoren zu den k kleinsten Eigenwerten bilden eine neue,
   niedrigdimensionale **Einbettung** der Punkte - nicht-konvexe Formen im Original werden
   darin oft zu klar trennbaren, konvexen Gruppen.
4. **k-Means auf der Einbettung** (nicht auf den Rohdaten!) liefert die finale
   Clusterzuordnung.

Die Grafiken weiter unten zeigen alle vier Phasen nacheinander - ein Schritt in der
Animation ist eine Phase, keine Iteration.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel": "Konvexe Gruppen, richtiges k - funktioniert, wie k-Means auch funktionieren würde.",
    "Nicht-konvexe Formen (Halbmonde)": "k-Means scheitert grundsätzlich, Spectral Clustering trennt sauber.",
    "Falsches k": "Richtige Graph-Konstruktion, aber k falsch gewählt - zeigt, dass Spectral Clustering diese k-Means-Schwäche NICHT behebt.",
    "Schwierige Graph-Konstruktion": "n_neighbors schlecht gewählt - eigene Parameterempfindlichkeit, analog zu DBSCANs eps.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Adressen", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider(
        "Anzahl wahrer Gruppen", *bounds("k_slider"), key="k_slider",
        help="Wie viele Gruppen im Szenario tatsächlich erzeugt werden.",
    )
    spread = st.slider(
        "Streuung / Rauschen", *bounds("spread_slider"), key="spread_slider", step=0.05,
        help="Bei „Gruppen“: Streuung je Gruppe. Bei „Halbmonde“: Rauschen um die ideale Kurve.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Punktwolken-Form**")
    shape = st.radio(
        "Form", options=C.SHAPES, key="shape_radio", format_func=lambda s: C.SHAPE_LABELS[s],
        help="„Gruppen“: konvexe Cluster. „Halbmonde“: nicht-konvexe Bögen - k wirkt auf "
        "beide Formen.",
    )

    st.markdown("**Spectral-Parameter**")
    target_k = st.slider(
        "Ziel-Clusteranzahl (für Spectral Clustering)", *bounds("target_k_slider"), key="target_k_slider",
        help="Wie viele Cluster Spectral Clustering tatsächlich suchen soll - unabhängig "
        "von der wahren Gruppenzahl oben. Weichen beide voneinander ab, zeigt das genau "
        "die 'k muss feststehen'-Schwäche, die Spectral Clustering von k-Means erbt.",
    )
    n_neighbors = st.slider(
        "n_neighbors (Ähnlichkeitsgraph)", *bounds("n_neighbors_slider"), key="n_neighbors_slider",
        help="Wie viele nächste Nachbarn im Ähnlichkeitsgraphen verbunden werden. Zu klein: "
        "der Graph zerfällt in mehr Teile als die Ziel-Clusteranzahl. Zu groß: die "
        "Graphstruktur verwischt.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Adressen.",
    )

sync_query_params(n_points, k, target_k, spread, seed, shape, n_neighbors)

with st.spinner("Führe Spectral Clustering aus..."):
    instance, result = _compute_run(
        int(n_points), int(k), int(target_k), spread, shape, int(seed), int(n_neighbors)
    )

data = instance.as_array()

st.markdown("## 🎯 Spectral Clustering in Aktion")

phase = st.select_slider("Phase", options=range(len(PHASES)), format_func=lambda i: PHASES[i], key="sp_phase")

phase_slot = st.empty()
if phase == 0:
    with phase_slot.container():
        st.plotly_chart(build_graph_figure(data, result.similarity), width="stretch", key="graph_phase")
        st.caption(f"Rohdaten mit Ähnlichkeitsgraph - {int(n_neighbors)} nächste Nachbarn je Punkt, symmetrisiert.")
elif phase == 1:
    with phase_slot.container():
        st.plotly_chart(build_scree_plot(result.eigenvalues, int(target_k)), width="stretch", key="scree_phase")
        st.caption(
            "Sortierte Eigenwerte des Graph-Laplace - eine große Lücke nach dem k-ten Eigenwert "
            "deutet auf k gut getrennte Cluster hin (Analogon zu dbscan-demos k-Distanz-Knick)."
        )
elif phase == 2:
    with phase_slot.container():
        st.plotly_chart(build_embedding_figure(result.embedding, result.labels), width="stretch", key="embedding_phase")
        st.caption(
            "Die ersten zwei Koordinaten der Spektral-Einbettung - nicht-konvexe Formen im "
            "Original werden hier oft zu klar trennbaren, konvexen Gruppen."
        )
else:
    with phase_slot.container():
        st.plotly_chart(build_scatter_figure(data, result.labels), width="stretch", key="result_phase")
        st.caption("Finales Ergebnis: k-Means auf der Spektral-Einbettung (nicht auf den Rohdaten).")

st.markdown("**Und mit anderen n_neighbors-Werten?**")
st.caption(
    "Gleiche Adressen wie oben, jeweils mit derselben k-Means-Startkonfiguration - nur "
    "n_neighbors unterscheidet sich, jeweils das finale Ergebnis gezeigt."
)
example_nn_values = sorted({
    max(C.N_NEIGHBORS_MIN, min(C.N_NEIGHBORS_MAX, v))
    for v in (max(2, n_neighbors // 4), max(2, n_neighbors // 2), n_neighbors * 2, n_neighbors * 4)
} - {int(n_neighbors)})
example_cols = st.columns(len(example_nn_values)) if example_nn_values else []
for col, example_nn in zip(example_cols, example_nn_values):
    with col:
        example_result = _compute_mini_run(instance, int(target_k), example_nn)
        st.plotly_chart(
            build_mini_scatter_figure(data, example_result.labels), width="stretch", key=f"mini_{example_nn}",
        )
        st.caption(f"n_neighbors={example_nn}")

st.markdown("---")

st.subheader("📐 Was kann Spectral Clustering, was k-Means nicht kann - und was nicht?")
st.markdown(
    """
Live für Ihr aktuelles Szenario berechnet: der **Rand-Index** von Spectral Clustering
gegen eine reine **k-Means-Baseline** auf denselben Rohdaten, beide beim aktuell
eingestellten k - macht sichtbar, wo Spectral Clustering etwas bringt (Nicht-Konvexität).
Die dritte Säule zeigt Spectral Clustering beim **wahren** k als Referenz - der direkte
Abfall gegenüber der aktuellen Einstellung macht sichtbar, dass ein falsches k auch
Spectral Clustering schadet, unabhängig davon, wie die Baseline gerade abschneidet.
"""
)

scores = _compute_method_comparison(instance, int(target_k), int(n_neighbors))
st.plotly_chart(build_method_comparison_chart(scores), width="stretch", key="method_comparison")

wrong_k_gap = scores["spectral_at_true_k"] - scores["spectral"]
convexity_gap = scores["spectral"] - scores["kmeans_baseline"]

if int(target_k) != instance.k and wrong_k_gap > 0.15:
    st.info(
        f"📉 Bei der eingestellten Ziel-Clusteranzahl {int(target_k)} erreicht Spectral "
        f"Clustering nur {scores['spectral']:.2f} - bei der wahren Gruppenzahl {instance.k} "
        f"wären es {scores['spectral_at_true_k']:.2f}. Genau das ist die ehrliche Botschaft "
        f"dieser Demo: Spectral Clustering hat k-Means' 'k muss feststehen'-Schwäche NICHT "
        f"abgeschafft."
    )
elif convexity_gap > 0.2:
    st.success(
        f"✅ Spectral Clustering erreicht hier einen Rand-Index von {scores['spectral']:.2f}, "
        f"die k-Means-Baseline nur {scores['kmeans_baseline']:.2f} - derselbe Datensatz, "
        f"nur der Zugang zur Nicht-Konvexität unterscheidet sich."
    )
else:
    st.info(
        "Bei diesem Szenario liegen alle drei Werte nah beieinander - probieren Sie „Nicht-"
        "konvexe Formen“ oder „Falsches k“ (Regler links), um die beiden Effekte einzeln zu sehen."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Ähnlichkeitsgraph**: $w_{ij} = \exp\left(-\frac{\lVert x_i-x_j\rVert^2}{2\sigma^2}\right)$
für Kanten des symmetrisierten `n_neighbors`-nächste-Nachbarn-Graphen, $\sigma$ = Median der
erhaltenen Kantenlängen (datengetrieben, kein zusätzlicher Regler nötig).

**Symmetrisch normalisierter Graph-Laplace**:

$$
L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}, \qquad D_{ii} = \sum_j w_{ij}
$$

**Zusammenhangskomponenten-Satz**: die Vielfachheit des Eigenwerts 0 von $L_{\text{sym}}$
entspricht der Anzahl der Zusammenhangskomponenten des Graphen (vorausgesetzt jeder Knoten
hat Grad > 0) - exakt nachgerechnet in `tests/test_algorithm.py` auf künstlich konstruierten
Graphen mit bekannter Komponentenzahl.

**Einbettung**: die Eigenvektoren $u_1,\dots,u_k$ zu den $k$ kleinsten Eigenwerten von
$L_{\text{sym}}$ bilden $U \in \mathbb{R}^{n\times k}$; die Zeilen von $U$ werden auf
Einheitslänge normiert (Ng-Jordan-Weiss, 2002) - das Ergebnis minimiert (approximativ) den
**Normalized Cut** des Graphen, eine Zielfunktion, die Cluster mit wenigen, schwachen
Kanten dazwischen und vielen, starken Kanten darin bevorzugt.

**Letzter Schritt**: k-Means auf den Zeilen der Einbettung, nicht auf den Rohdaten - genau
hier steckt weiterhin k-Means' "k muss feststehen"-Schwäche, unverändert gegenüber
k-Means selbst.

Implementiert in `sp_algorithm.py` (Ähnlichkeitsgraph, Laplace, Eigenzerlegung, finales
Lloyd's) und `sp_evaluation.py` (Rand-Index, Methoden-Vergleich).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
