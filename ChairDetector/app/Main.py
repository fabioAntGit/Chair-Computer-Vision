import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import av
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import pandas as pd
import json
from datetime import datetime
import io
import time
from pathlib import Path

@st.cache_data
def img_para_bytes(img_rgb: np.ndarray) -> bytes:
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

AVAILABLE_DEVICES = ["cpu"]
if torch.cuda.is_available(): AVAILABLE_DEVICES.append("cuda")
if torch.backends.mps.is_available(): AVAILABLE_DEVICES.append("mps")

st.set_page_config(page_title="IA Chair Detector", layout="wide")


MODELS_DIR = Path("modelos")


@st.cache_data
def discover_models() -> dict[str, str]:
    found: dict[str, str] = {}
    if not MODELS_DIR.is_dir():
        return found
    for pt_file in sorted(MODELS_DIR.glob("*/weights/best.pt")):
        found[pt_file.parts[-3]] = str(pt_file)
    return found


ALL_MODEL_PATHS: dict[str, str] = discover_models()
AVAILABLE_MODELS: list[str] = list(ALL_MODEL_PATHS.keys())


def load_model(nome: str):
    key = f"model_{nome}"
    if key not in st.session_state or st.session_state[key] is None:
        path = ALL_MODEL_PATHS.get(nome)
        if path is None:
            st.error(f"Modelo `{nome}` não encontrado no diretório de modelos.")
            return None
        with st.spinner(f"A carregar modelo `{nome}`..."):
            try:
                st.session_state[key] = YOLO(path)
            except Exception:
                st.session_state[key] = None
                st.error(f"Erro ao carregar o modelo `{nome}`.")
    return st.session_state[key]


def _build_detections(r) -> list[dict]:
    names = r.names
    return [
        {
            "classe": names[int(cls)],
            "confianca": round(float(cf), 4),
            "bbox_xyxy": {
                "x1": round(float(b[0]), 2), "y1": round(float(b[1]), 2),
                "x2": round(float(b[2]), 2), "y2": round(float(b[3]), 2),
            },
        }
        for cls, cf, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
    ]


def run_inference(mdl, image, conf: float, device: str, show_labels: bool, show_scores: bool) -> dict:
    t0 = time.perf_counter()
    results = mdl.predict(image, conf=conf, device=device)
    tempo_ms = (time.perf_counter() - t0) * 1000

    img_rgb = cv2.cvtColor(
        results[0].plot(labels=show_labels, conf=show_scores), cv2.COLOR_BGR2RGB
    )
    boxes = results[0].boxes
    detections = _build_detections(results[0]) if len(boxes) > 0 else []

    return {
        "img_rgb": img_rgb,
        "tempo_ms": tempo_ms,
        "detections": detections,
        "confs": boxes.conf.tolist(),
        "n_boxes": len(boxes),
    }


def render_model_col(nome: str, data: dict, *, show_image: bool = True) -> None:
    st.markdown(f"### {nome}")
    if show_image:
        st.image(data["img_rgb"], use_container_width=True)

    if data["n_boxes"] > 0:
        df = pd.Series([d["classe"] for d in data["detections"]]).value_counts().reset_index()
        df.columns = ["Peça", "Qtd"]
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(
            f"Total: **{data['n_boxes']}** | "
            f"Conf. média: **{sum(data['confs']) / len(data['confs']):.2%}**"
        )
    else:
        st.warning("Nenhuma deteção.")

    st.metric("Tempo de Inferência", f"{data['tempo_ms']:.1f} ms")


st.sidebar.title("Configurações")

st.sidebar.markdown("### Seleção do Modelo")
if not AVAILABLE_MODELS:
    st.sidebar.error("Nenhum modelo .pt encontrado em `modelos/*/weights/best.pt`.")
    st.stop()

modelo_ver = st.sidebar.selectbox("Escolha o Modelo", options=AVAILABLE_MODELS, index=0,
                                  disabled=st.session_state.get("page") == "compare",
                                  help="Desativado na página de Comparação (os dois modelos são escolhidos lá).")

model = load_model(modelo_ver)

st.sidebar.divider()

st.sidebar.markdown("### Processamento")
if len(AVAILABLE_DEVICES) > 1:
    DEVICE = st.sidebar.selectbox(
        "Hardware",
        options=AVAILABLE_DEVICES,
        index=len(AVAILABLE_DEVICES) - 1,
        format_func=lambda x: "GPU (NVIDIA)" if x == "cuda" else "GPU (Apple Silicon)" if x == "mps" else "CPU"
    )
else:
    DEVICE = "cpu"
    st.sidebar.write("Hardware: **CPU** (Nenhuma GPU detetada)")
st.sidebar.divider()

st.sidebar.markdown("### Ajustes do Modelo")
confianca = st.sidebar.slider("Confiança (Threshold)", min_value=0.0, max_value=1.0, value=0.5,
                               help="Nível mínimo de certeza para mostrar uma deteção.")

st.sidebar.divider()

st.sidebar.markdown("### Interface Visual")
mostrar_labels = st.sidebar.checkbox("Mostrar Nome da Peça", value=True)
mostrar_scores = st.sidebar.checkbox("Mostrar % de Certeza", value=True)

if "page" not in st.session_state:
    st.session_state.page = "image"
if "historico" not in st.session_state:
    st.session_state.historico = []

_, col_nav1, col_nav2, col_nav3, col_nav4, _ = st.columns([2, 1, 1, 1, 1, 2])
with col_nav1:
    if st.button("Imagem Estática", use_container_width=True):
        st.session_state.page = "image"
with col_nav2:
    if st.button("Webcam", use_container_width=True):
        st.session_state.page = "webcam"
with col_nav3:
    if st.button("Comparação", use_container_width=True):
        st.session_state.page = "compare"
with col_nav4:
    hist_label = f"Histórico ({len(st.session_state.historico)})"
    if st.button(hist_label, use_container_width=True):
        st.session_state.page = "historico"

st.divider()


# ─── PÁGINA: IMAGEM ESTÁTICA ───
if st.session_state.page == "image":
    st.markdown("<h1 style='text-align: center;'>Deteção em Imagem</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Escolhe uma imagem de uma cadeira...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if model:
            with st.spinner("A analisar..."):
                data = run_inference(model, image, confianca, DEVICE, mostrar_labels, mostrar_scores)

                col_res, col_stats = st.columns([2, 1])

                with col_res:
                    st.markdown("### Resultado da Deteção")
                    st.image(data["img_rgb"], width=350)

                with col_stats:
                    render_model_col(modelo_ver, data, show_image=False)

                    # ─── Exportar JSON (sempre visível) ───
                    export_data = {
                        "threshold_confianca": confianca,
                        modelo_ver: {
                            "total_detetado": len(data["detections"]),
                            "detecoes": data["detections"],
                        },
                    }
                    with st.expander("Pré-visualizar JSON"):
                        st.json(export_data)
                    st.download_button(
                        label="Exportar JSON",
                        data=json.dumps(export_data, indent=2, ensure_ascii=False),
                        file_name="detecoes.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    st.download_button(
                        label="Descarregar Imagem",
                        data=img_para_bytes(data["img_rgb"]),
                        file_name=f"detecao_{modelo_ver}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_img_static",
                    )

                    # ─── Guardar no Histórico (1x por ficheiro) ───
                    hist_key = f"img_{uploaded_file.file_id}_{modelo_ver}"
                    if hist_key not in st.session_state:
                        st.session_state[hist_key] = True
                        conf_media = sum(data["confs"]) / len(data["confs"]) if data["confs"] else 0.0
                        st.session_state.historico.append({
                            "imagem": data["img_rgb"],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "modelo": modelo_ver,
                            "total": len(data["detections"]),
                            "conf_media": conf_media,
                            "detecoes": data["detections"],
                            "fonte": "Imagem Estática",
                        })
        else:
            st.error("Modelo não carregado.")


# ─── PÁGINA: WEBCAM ───
elif st.session_state.page == "webcam":
    st.markdown("<h1 style='text-align: center;'>Deteção por Webcam</h1>", unsafe_allow_html=True)

    if model:
        class VideoProcessor:
            def __init__(self):
                self.confianca = confianca
                self.mostrar_labels = mostrar_labels
                self.mostrar_scores = mostrar_scores
                self._frame_count = 0
                self._last_annotated = None  # Cache do último frame anotado
                self.last_detections = []

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self._frame_count += 1

                # Correr YOLO apenas em 1 de cada 3 frames para ganhar FPS
                if self._frame_count % 3 == 0 or self._last_annotated is None:
                    results = model.predict(img, conf=self.confianca, verbose=False, imgsz=320, device=DEVICE)
                    r = results[0]
                    self._last_annotated = r.plot(labels=self.mostrar_labels, conf=self.mostrar_scores)

                    self.last_detections = _build_detections(r)

                return av.VideoFrame.from_ndarray(self._last_annotated, format="bgr24")

        ctx = webrtc_streamer(
            key="chair-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.confianca = confianca
            ctx.video_processor.mostrar_labels = mostrar_labels
            ctx.video_processor.mostrar_scores = mostrar_scores

            # ─── Exportar JSON do último frame ───
            st.divider()
            detections = ctx.video_processor.last_detections
            export_data = {
                "modelo": modelo_ver,
                "threshold_confianca": confianca,
                "total_detetado": len(detections),
                "detecoes": detections,
            }
            
            with st.expander("Pré-visualizar JSON do frame"):
                st.json(export_data)

            col_info, col_btn = st.columns([2, 1])
            with col_info:
                st.caption(f"📸 Último frame: **{len(detections)}** deteção(ões) capturada(s)")
            with col_btn:
                st.download_button(
                    label="Exportar JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name="detecoes_webcam.json",
                    mime="application/json",
                    use_container_width=True,
                )
    else:
        st.error("Modelo não carregado.")

# ─── PÁGINA: COMPARAÇÃO ───
elif st.session_state.page == "compare":
    st.markdown("<h1 style='text-align: center;'>Comparação de Modelos</h1>", unsafe_allow_html=True)
    st.caption("Escolhe dois modelos para comparar os resultados lado a lado na mesma imagem.")

    col_sel_a, col_sel_b = st.columns(2)
    with col_sel_a:
        modelo_a = st.selectbox("Modelo A", options=AVAILABLE_MODELS, index=0, key="cmp_modelo_a")
    with col_sel_b:
        # Índice por defeito: segundo modelo disponível (evitar duplicado)
        default_b = 1 if len(AVAILABLE_MODELS) > 1 else 0
        modelo_b = st.selectbox("Modelo B", options=AVAILABLE_MODELS, index=default_b, key="cmp_modelo_b")

    if modelo_a == modelo_b:
        st.warning("Seleciona dois modelos diferentes para comparar.")

    uploaded_cmp = st.file_uploader("Escolhe uma imagem...", type=["jpg", "jpeg", "png"], key="cmp_upload")

    if uploaded_cmp is not None and modelo_a != modelo_b:
        img_cmp = Image.open(uploaded_cmp)

        mdl_a = load_model(modelo_a)
        mdl_b = load_model(modelo_b)

        if mdl_a and mdl_b:
            with st.spinner(f"A correr {modelo_a} e {modelo_b}..."):
                data_a = run_inference(mdl_a, img_cmp, confianca, DEVICE, mostrar_labels, mostrar_scores)
                data_b = run_inference(mdl_b, img_cmp, confianca, DEVICE, mostrar_labels, mostrar_scores)

            col_a, col_b = st.columns(2)
            with col_a:
                render_model_col(modelo_a, data_a)
            with col_b:
                render_model_col(modelo_b, data_b)

            # ── Exportar JSON combinado ──
            st.divider()
            export_cmp = {
                "threshold_confianca": confianca,
                modelo_a: {"total_detetado": len(data_a["detections"]), "detecoes": data_a["detections"]},
                modelo_b: {"total_detetado": len(data_b["detections"]), "detecoes": data_b["detections"]},
            }
            with st.expander("Pré-visualizar JSON comparativo"):
                st.json(export_cmp)
            st.download_button(
                label="Exportar JSON comparativo",
                data=json.dumps(export_cmp, indent=2, ensure_ascii=False),
                file_name="comparacao_modelos.json",
                mime="application/json",
                use_container_width=True,
            )

            col_dl_a, col_dl_b = st.columns(2)
            with col_dl_a:
                st.download_button(
                    label=f"Descarregar Imagem {modelo_a}",
                    data=img_para_bytes(data_a["img_rgb"]),
                    file_name=f"detecao_{modelo_a}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_img_cmp_a",
                )
            with col_dl_b:
                st.download_button(
                    label=f"Descarregar Imagem {modelo_b}",
                    data=img_para_bytes(data_b["img_rgb"]),
                    file_name=f"detecao_{modelo_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_img_cmp_b",
                )

            # ── Guardar no Histórico (1x por ficheiro) ──
            hist_key_cmp = f"cmp_{uploaded_cmp.file_id}_{modelo_a}_{modelo_b}"
            if hist_key_cmp not in st.session_state:
                st.session_state[hist_key_cmp] = True
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for nome, data in [(modelo_a, data_a), (modelo_b, data_b)]:
                    st.session_state.historico.append({
                        "imagem": data["img_rgb"],
                        "timestamp": ts,
                        "modelo": nome,
                        "total": len(data["detections"]),
                        "conf_media": sum(data["confs"]) / len(data["confs"]) if data["confs"] else 0.0,
                        "detecoes": data["detections"],
                        "fonte": "Comparação",
                    })
        else:
            st.error("Erro ao carregar um ou ambos os modelos selecionados.")


# ─── PÁGINA: HISTÓRICO ───
if st.session_state.page == "historico":
    st.markdown("<h1 style='text-align: center;'>Histórico de Inferências</h1>", unsafe_allow_html=True)

    hist = st.session_state.historico
    if not hist:
        st.info("Ainda não existem inferências guardadas. Faz upload de uma imagem na página de Imagem Estática ou Comparação.")
    else:
        if st.button("Limpar Histórico", type="secondary"):
            st.session_state.historico = []
            st.rerun()

        st.caption(f"{len(hist)} inferência(s) guardada(s)")
        st.divider()

        COLS = 3
        for row_start in range(0, len(hist), COLS):
            cols = st.columns(COLS)
            for col_idx, entry in enumerate(hist[row_start:row_start + COLS]):
                idx = row_start + col_idx
                with cols[col_idx]:
                    st.image(entry["imagem"], use_container_width=True)

                    st.markdown(
                        f"**Modelo:** `{entry['modelo']}`  \n"
                        f"**Fonte:** {entry['fonte']}  \n"
                        f"**Data/Hora:** {entry['timestamp']}  \n"
                        f"**Deteções:** {entry['total']}  \n"
                        f"**Conf. Média:** {entry['conf_media']:.2%}"
                    )

                    if entry["detecoes"]:
                        with st.expander("Ver detalhes"):
                            for i, det in enumerate(entry["detecoes"], 1):
                                b = det["bbox_xyxy"]
                                st.caption(
                                    f"{i}. **{det['classe']}** ({det['confianca']:.2%}) — "
                                    f"x1={b['x1']} y1={b['y1']} x2={b['x2']} y2={b['y2']}"
                                )

                    entry_json = {
                        "timestamp": entry["timestamp"],
                        "fonte": entry["fonte"],
                        "threshold_confianca": confianca,
                        entry["modelo"]: {
                            "total_detetado": entry["total"],
                            "detecoes": entry["detecoes"],
                        },
                    }
                    st.download_button(
                        label="Exportar JSON",
                        data=json.dumps(entry_json, indent=2, ensure_ascii=False),
                        file_name=f"historico_{idx+1}_{entry['modelo']}_{entry['timestamp'][:10]}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"dl_hist_{idx}",
                    )
                    st.download_button(
                        label="Descarregar Imagem",
                        data=img_para_bytes(entry["imagem"]),
                        file_name=f"historico_{idx+1}_{entry['modelo']}_{entry['timestamp'][:10]}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_img_hist_{idx}",
                    )
                    st.divider()