# Spectral Clustering für nicht-konvexe Sammel-Routen – Streamlit-Demo

Achtes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **dritter unabhängiger Zweig ab [kmeans-demo](../kmeans-demo)**
(neben dem Dichte-Zweig dbscan-demo+agglomerative-demo→hdbscan-demo und dem
parametrisch-bayesianischen Zweig gmm-demo→dpmm-demo). Löst k-Means'
**Nicht-Konvexitäts-Annahme** über einen im Portfolio bisher ungenutzten
mathematischen Werkzeugkasten: **Graphentheorie + lineare Algebra**
(Graph-Laplace-Matrix, Eigenzerlegung) statt Dichte oder Bayesianischer Inferenz.

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo   (löst BEIDE: über Dichte)
              agglomerative-demo ──────────┘
kmeans-demo → gmm-demo → dpmm-demo             (löst NUR "kein k": über Bayesianische Nichtparametrik)
kmeans-demo → spectral-demo                     (löst NUR Nicht-Konvexität: über Graphentheorie)
```

**Bewusst gewählter, ehrlicher Kontrast zu DBSCAN, nicht Redundanz**: klassisches
Spectral Clustering (Ng, Jordan & Weiss, 2002) braucht am Ende trotzdem **k-Means auf
der Spektral-Einbettung** - die Clusterzahl k muss also weiterhin vorab feststehen,
anders als bei DBSCAN/HDBSCAN. Die Demo zeigt explizit: zwei unabhängige
k-Means-Schwächen (Nicht-Konvexität, festes k) lassen sich getrennt beheben, und kein
neues Verfahren muss zwangsläufig beide auf einmal lösen.

## Warum diese Demo anders aufgebaut ist

Anders als k-Means/EM/Gibbs-Sampling ist Spectral Clustering **kein iteratives
Verfahren**, sondern ein **einmaliger vierstufiger Pipeline-Durchlauf**:
Ähnlichkeitsgraph → Graph-Laplace-Matrix → Eigenzerlegung → k-Means auf der
Einbettung. Statt eines künstlichen Iterations-Sliders zeigt die Primäransicht deshalb
einen **4-Phasen-Regler**, der genau diese vier Schritte nacheinander sichtbar macht -
inklusive zweier für diese Reihe neuer Diagrammtypen: dem **Ähnlichkeitsgraphen**
(Punkte + Kanten) und dem **Eigenwert-Scree-Plot** (die Standard-Heuristik zur
k-Wahl: eine große Lücke nach dem k-ten Eigenwert deutet auf k gut getrennte Cluster
hin, das Analogon zu dbscan-demos k-Distanz-Knick).

Die Presets legen zwei getrennte Achsen offen:

- **Einfaches Beispiel**: konvexe Gruppen, richtiges k - funktioniert, wie k-Means auch
  funktionieren würde (Baseline, kein Alleinstellungsmerkmal nötig).
- **Nicht-konvexe Formen (Halbmonde)**: der Kernnachweis - eine k-Means-Baseline auf
  denselben Rohdaten scheitert an der Konvexitätsannahme, Spectral Clustering trennt
  sauber.
- **Falsches k**: richtige Graph-Konstruktion, aber eine andere Ziel-Clusteranzahl als
  die wahre Gruppenzahl - zeigt die ehrliche Botschaft der Demo: Spectral Clustering
  behebt k-Means' "k muss vorab feststehen"-Schwäche **nicht**. Die
  📐-Vergleichssektion zeigt dafür eine dritte Referenzsäule "Spectral Clustering beim
  wahren k", weil der Rand-Index eine Aufspaltung eines wahren Clusters nur milde
  bestraft und ein reiner Spectral-vs-Baseline-Vergleich das Leiden sonst verdecken
  könnte.
- **Schwierige Graph-Konstruktion**: `n_neighbors` zu klein gewählt - der Graph zerfällt
  stärker, als die Eigenwert-Lücke bei k vermuten lässt - eine eigene
  Parameterempfindlichkeit, analog zu DBSCANs eps.

## Visualisierung

Der 4-Phasen-Regler zeigt (1) Rohdaten + Ähnlichkeitsgraph, (2) den Eigenwert-Scree-Plot
mit markierter Lücke bei der Ziel-Clusteranzahl, (3) die Spektral-Einbettung (Scatter der
ersten zwei Eigenvektor-Koordinaten - macht sichtbar, warum k-Means darauf leicht
funktioniert, obwohl es auf den rohen Koordinaten scheitern würde), (4) das finale
Cluster-Ergebnis. Kleinmultiples zeigen dieselben Daten mit unterschiedlichen
`n_neighbors`-Werten nebeneinander. Punktfarben nutzen das portfolioweite dynamische
HSL-Farbrad-Schema für Cluster jenseits der festen 8-Farben-Palette.

## Sicherheitsgrenzen

`N_POINTS_HARD_MAX` (300) begrenzt die naive O(n²)-Ähnlichkeitsberechnung.

## Verifikation

- **Handgerechnetes Beispiel**: eine 4-Punkte-Instanz (zwei klar getrennte Paare) mit
  `n_neighbors=1` von Hand durchgerechnet - Ähnlichkeitsgewichte, Laplace-Block-Matrix
  und deren Eigenwerte exakt nachgerechnet.
- **Zusammenhangskomponenten-Verifikation** (das mathematische Kernstück, exakt und
  algorithmusunabhängig): die Anzahl der exakt-Null-Eigenwerte der normalisierten
  Graph-Laplace-Matrix entspricht einem klassischen Satz der Spektralgraphentheorie
  zufolge exakt der Anzahl der Zusammenhangskomponenten des Graphen (bei
  ausschließlich Knoten mit Grad > 0) - auf künstlich in mehrere disjunkte Komponenten
  konstruierten Graphen über mehrere Größen/Komponentenzahlen direkt nachgerechnet.
  Dokumentierte Ausnahme: ein isolierter Knoten (Grad 0) trägt bei der
  **normalisierten** Laplace-Matrix immer den Eigenwert **1** bei, nicht 0 - eine
  echte mathematische Randbedingung des Satzes, kein Bug, mit eigenem Test explizit
  nachgewiesen.
- **Struktur-Invarianten**: Ähnlichkeitsmatrix symmetrisch/nicht-negativ mit
  Null-Diagonale, Laplace-Eigenwerte innerhalb der theoretischen Schranken [0, 2],
  Zeilen der Einbettung auf Einheitslänge normiert.
- **Kreuzvergleich** gegen `sklearn.cluster.SpectralClustering` (Partitions-
  Übereinstimmung, toleranzbasiert wie bei hdbscan-demo, kein exaktes Zahlen-Match
  nötig, da Implementierungsdetails wie Eigenvektor-Vorzeichen frei variieren dürfen).
- **Kern-Behauptungen der Demo direkt getestet**: Spectral Clustering trennt Halbmonde
  zuverlässig, wo eine k-Means-Baseline auf denselben Rohdaten nachweislich scheitert
  (`test_spectral_clustering_solves_moons_where_kmeans_baseline_fails`); ein falsches
  Ziel-k schadet Spectral Clustering ebenso
  (`test_wrong_k_hurts_spectral_clustering_too`,
  `test_spectral_at_true_k_reference_reveals_wrong_k_degradation`).
- **Alle vier Presets direkt gegen das tatsächliche App-Verhalten getestet** (`k` erzeugt
  die Szenario-Wahrheit, `target_k` ist die tatsächlich an den Algorithmus übergebene
  Clusterzahl - exakt wie `app.py` es macht). Dieses Muster deckte während der
  Entwicklung einen echten Design-Fehler auf: das "Falsches k"-Preset nutzte anfangs
  denselben Parameter für Szenario-Wahrheit UND Algorithmus-Ziel, wodurch das "falsche"
  k zufällig mit der wahren Gruppenzahl übereinstimmte - erst live im Browser bemerkt,
  nicht von einem Test, was zur Trennung in `k`/`target_k` und den entsprechenden
  Regressionstests führte (siehe project-memory für den konkreten Vorfall).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, 4-Phasen-Regler, Methoden-Vergleich, Formulierungs-Expander |
| `sp_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `sp_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `sp_scenario.py` | Ring-Blobs (wie kmeans-demo) und Halbmonde/Bögen (wie dbscan-demo, k>2 als "Blütenblätter auf einem Ring" verallgemeinert) |
| `sp_algorithm.py` | KNN-Ähnlichkeitsgraph mit Gauß-Gewichtung, normalisierte Graph-Laplace-Matrix, Eigenzerlegung, zeilennormierte Einbettung, finales from-scratch Lloyd's-k-Means auf der Einbettung |
| `sp_evaluation.py` | Rand-Index (from scratch), Methoden-Vergleich (Spectral vs. k-Means-Baseline vs. Spectral beim wahren k) |
| `sp_visualization.py` | Ähnlichkeitsgraph-Diagramm, Eigenwert-Scree-Plot, Spektral-Einbettungs-Scatter, finales Cluster-Ergebnis, Kleinmultiples, Methoden-Vergleichsdiagramm (Plotly) |
| `tests/` | Handinstanz, Zusammenhangskomponenten-Verifikation, Struktur-Invarianten, sklearn-Kreuzvergleich, Kern-Nachweise, Preset-gegen-App-Verhalten-Tests, AppTest-Smoke-Test |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
