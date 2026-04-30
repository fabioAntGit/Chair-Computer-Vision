import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import av
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import pandas as pd

# Deteção de hardware disponível
AVAILABLE_DEVICES = ["cpu"]
if torch.cuda.is_available(): AVAILABLE_DEVICES.append("cuda")
if torch.backends.mps.is_available(): AVAILABLE_DEVICES.append("mps")

st.set_page_config(page_title="IA Chair Detector", layout="wide")


@st.cache_resource
def get_model(path):
    try:
        return YOLO(path)
    except Exception:
        st.error(f"Erro ao carregar o modelo em {path}. Verifica se os pesos existem!")
        return None


# Sidebar
st.sidebar.title("Configurações")

st.sidebar.markdown("### Seleção do Modelo")
modelo_ver = st.sidebar.selectbox("Escolha o Modelo", options=["YOLOv8m", "YOLOv8l"], index=0)

MODEL_PATH = "modelos/yolov8m/weights/best.pt" if modelo_ver == "YOLOv8m" else "modelos/yolov8l/weights/best.pt"
model = get_model(MODEL_PATH)

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

st.sidebar.divider()
st.sidebar.info("Projeto IA - 2026\n\nAlunos: 8230365 | 8230196")

# Navegação por session_state
if "page" not in st.session_state:
    st.session_state.page = "image"

col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("🖼️ Imagem Estática"):
        st.session_state.page = "image"
with col_nav2:
    if st.button("📷 Câmara em Tempo Real"):
        st.session_state.page = "webcam"

st.divider()


# ─── PÁGINA: IMAGEM ESTÁTICA ───
if st.session_state.page == "image":
    st.markdown("<h1 style='text-align: center;'>Deteção em Imagem</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Escolhe uma imagem de uma cadeira...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if model:
            with st.spinner("A analisar..."):
                results = model.predict(image, conf=confianca, device=DEVICE)
                # plot() devolve BGR; converter para RGB para o Streamlit
                res_plotted_rgb = cv2.cvtColor(
                    results[0].plot(labels=mostrar_labels, conf=mostrar_scores),
                    cv2.COLOR_BGR2RGB
                )

                col_res, col_stats = st.columns([2, 1])

                with col_res:
                    st.markdown("### Resultado da Deteção")
                    st.image(res_plotted_rgb, width=350)

                with col_stats:
                    if len(results[0].boxes) > 0:
                        st.markdown("### Estatísticas")
                        classes = results[0].boxes.cls.tolist()
                        names = results[0].names
                        detected_names = [names[int(c)] for c in classes]
                        df_counts = pd.Series(detected_names).value_counts().reset_index()
                        df_counts.columns = ['Peça', 'Qtd']
                        st.dataframe(df_counts, hide_index=True, use_container_width=True)
                        st.markdown(f"**Total detetado:** {len(classes)}")
                    else:
                        st.warning("Nenhuma peça detetada.")
        else:
            st.error("Modelo não carregado.")


# ─── PÁGINA: WEBCAM ───
else:
    st.markdown("<h1 style='text-align: center;'>Deteção por Webcam</h1>", unsafe_allow_html=True)

    if model:
        class VideoProcessor:
            def __init__(self):
                self.confianca = confianca
                self.mostrar_labels = mostrar_labels
                self.mostrar_scores = mostrar_scores
                self._frame_count = 0
                self._last_annotated = None  # Cache do último frame anotado

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self._frame_count += 1

                # Correr YOLO apenas em 1 de cada 3 frames para ganhar FPS
                if self._frame_count % 3 == 0 or self._last_annotated is None:
                    results = model.predict(img, conf=self.confianca, verbose=False, imgsz=320, device=DEVICE)
                    self._last_annotated = results[0].plot(labels=self.mostrar_labels, conf=self.mostrar_scores)

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

        # Atualizar parâmetros em tempo real sem reiniciar a stream
        if ctx.video_processor:
            ctx.video_processor.confianca = confianca
            ctx.video_processor.mostrar_labels = mostrar_labels
            ctx.video_processor.mostrar_scores = mostrar_scores
    else:
        st.error("Modelo não carregado.")