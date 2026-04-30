import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import pandas as pd

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
                results = model.predict(image, conf=confianca)
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
        # Forçar pedido de permissão da câmara no browser
        if st.button("Solicitar Acesso à Câmara (Browser)"):
            st.components.v1.html("""
                <script>
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function(stream) {
                        alert("Acesso concedido! Por favor, faz refresh à página.");
                        window.parent.location.reload();
                    })
                    .catch(function(err) {
                        alert("Não foi possível aceder à câmara. Verifica as definições do browser ou o ícone do cadeado no URL.");
                    });
                </script>
            """, height=0)

        class VideoProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = model.predict(img, conf=confianca, verbose=False)
                annotated_frame = results[0].plot(labels=mostrar_labels, conf=mostrar_scores)
                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        webrtc_streamer(
            key="chair-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True},
            async_processing=True,
        )
    else:
        st.error("Modelo não carregado.")