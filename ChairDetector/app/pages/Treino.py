import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Dashboard de Treino – IA Chair Detector",
    layout="wide",
)

st.title("Resultados de Treino")
st.caption("Métricas, curvas de aprendizagem e imagens de validação para cada modelo YOLOv8 treinado.")

st.divider()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELOS_DIR = BASE_DIR / "modelos"

def discover_models(modelos_dir: Path) -> dict:
    """Percorre modelos/ e devolve info de cada modelo encontrado."""
    models = {}
    if not modelos_dir.exists():
        return models
    for model_dir in sorted(modelos_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        train_dir = model_dir / "train"
        csv_path  = train_dir / "results.csv"
        models[model_dir.name] = {
            "dir":       model_dir,
            "train_dir": train_dir if train_dir.exists() else None,
            "csv":       csv_path  if csv_path.exists()  else None,
        }
    return models

MODELS = discover_models(MODELOS_DIR)

# ── Cores para os gráficos ───────────────────────────────────────────────────
COLORS = ["#6378ff", "#ff6b6b", "#00d4aa", "#ffd166", "#a259ff", "#ff79c6"]

def plot_lines(ax, epochs, series: list, title: str, ylabel=""):
    """Desenha linhas num eixo matplotlib."""
    for i, (label, values) in enumerate(series):
        color = COLORS[i % len(COLORS)]
        ax.plot(epochs, values, label=label, color=color, linewidth=1.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Época", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(prop={'size': 7})
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))

if not MODELS:
    st.error(f"Nenhum modelo encontrado em `{MODELOS_DIR}`.")
    st.stop()

# ── Tabs por modelo ────────────────────────────────────────────────────────────
tab_labels = [f" {name.upper()}" for name in MODELS]
tabs = st.tabs(tab_labels)

for tab, (model_name, info) in zip(tabs, MODELS.items()):
    with tab:
        train_dir = info["train_dir"]
        csv_path  = info["csv"]

        if train_dir is None:
            st.info(f"Pasta `train/` não encontrada para o modelo **{model_name}**.")
            continue

        # ── Métricas via CSV ───────────────────────────────────────────────────
        if csv_path:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            if "epoch" in df.columns:
                df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
            epochs = df["epoch"].tolist()

            last = df.iloc[-1]
            best_row = df.loc[df["metrics/mAP50(B)"].idxmax()]

            st.subheader("Métricas Finais")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Épocas", int(df["epoch"].max()))
            c2.metric("mAP50 (Best)", f'{best_row["metrics/mAP50(B)"]:.3f}')
            c3.metric("mAP50-95", f'{last["metrics/mAP50-95(B)"]:.3f}')
            c4.metric("Precision", f'{last["metrics/precision(B)"]:.3f}')
            c5.metric("Recall", f'{last["metrics/recall(B)"]:.3f}')

            st.divider()

            st.subheader("Gráficos de Treino")
            
            charts = [
                ("Perda de Treino", [
                    ("Box Loss", df["train/box_loss"].tolist()),
                    ("Cls Loss", df["train/cls_loss"].tolist()),
                    ("DFL Loss", df["train/dfl_loss"].tolist()),
                ]),
                ("Perda de Validação", [
                    ("Box Loss", df["val/box_loss"].tolist()),
                    ("Cls Loss", df["val/cls_loss"].tolist()),
                    ("DFL Loss", df["val/dfl_loss"].tolist()),
                ]),
                ("Métricas de Precisão", [
                    ("mAP50", df["metrics/mAP50(B)"].tolist()),
                    ("mAP50-95", df["metrics/mAP50-95(B)"].tolist()),
                ]),
                ("Precision & Recall", [
                    ("Precision", df["metrics/precision(B)"].tolist()),
                    ("Recall", df["metrics/recall(B)"].tolist()),
                ])
            ]

            for i in range(0, len(charts), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(charts):
                        title, series = charts[i + j]
                        fig, ax = plt.subplots(figsize=(5, 3))
                        plot_lines(ax, epochs, series, title)
                        cols[j].pyplot(fig)
                        plt.close(fig)

            with st.expander("Ver tabela de dados"):
                st.dataframe(df, width='stretch')

            st.divider()
        else:
            st.warning("Ficheiro `results.csv` não encontrado.")

        # ── Galeria de imagens ─────────────────────────────────────────────────
        st.subheader("Galeria de Resultados")
        
        # Procura todas as imagens .png e .jpg na pasta de treino
        available_imgs = sorted(list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")))
        
        if available_imgs:
            cols = st.columns(3)
            for idx, img_path in enumerate(available_imgs):
                cols[idx % 3].image(str(img_path), caption=img_path.name, width='stretch')
        else:
            st.write("Nenhuma imagem disponível.")
