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


# Pré-carregar ambos os modelos (sempre em cache)
model_m = get_model("modelos/yolov8m/weights/best.pt")
model_l = get_model("modelos/yolov8l/weights/best.pt")

# Sidebar
st.sidebar.title("Configurações")

st.sidebar.markdown("### Seleção do Modelo")
modelo_ver = st.sidebar.selectbox("Escolha o Modelo", options=["YOLOv8m", "YOLOv8l"], index=0,
                                  disabled=st.session_state.get("page") == "compare",
                                  help="Desativado na página de Comparação (ambos os modelos são usados).")

MODEL_PATH = "modelos/yolov8m/weights/best.pt" if modelo_ver == "YOLOv8m" else "modelos/yolov8l/weights/best.pt"
model = model_m if modelo_ver == "YOLOv8m" else model_l

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
                    st.markdown("### Estatísticas")
                    boxes = results[0].boxes
                    names = results[0].names
                    if len(boxes) > 0:
                        classes = boxes.cls.tolist()
                        confs   = boxes.conf.tolist()
                        bboxes  = boxes.xyxy.tolist()
                        detected_names = [names[int(c)] for c in classes]
                        df_counts = pd.Series(detected_names).value_counts().reset_index()
                        df_counts.columns = ['Peça', 'Qtd']
                        st.dataframe(df_counts, hide_index=True, use_container_width=True)
                        st.caption(f"Total: **{len(classes)}** | Conf. média: **{sum(confs)/len(confs):.2%}**")
                        detections = [
                            {
                                "classe": names[int(cls)],
                                "confianca": round(float(conf), 4),
                                "bbox_xyxy": {
                                    "x1": round(float(bbox[0]), 2), "y1": round(float(bbox[1]), 2),
                                    "x2": round(float(bbox[2]), 2), "y2": round(float(bbox[3]), 2),
                                },
                            }
                            for cls, conf, bbox in zip(classes, confs, bboxes)
                        ]
                    else:
                        detections = []
                        confs = []
                        st.warning("Nenhuma deteção.")

                    # ─── Exportar JSON (sempre visível) ───
                    export_data = {
                        "threshold_confianca": confianca,
                        modelo_ver: {
                            "total_detetado": len(detections),
                            "detecoes": detections,
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

                    # ─── Guardar no Histórico (1x por ficheiro) ───
                    hist_key = f"img_{uploaded_file.file_id}_{modelo_ver}"
                    if hist_key not in st.session_state:
                        st.session_state[hist_key] = True
                        conf_media = sum(confs) / len(confs) if confs else 0.0
                        st.session_state.historico.append({
                            "imagem": res_plotted_rgb,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "modelo": modelo_ver,
                            "total": len(detections),
                            "conf_media": conf_media,
                            "detecoes": detections,
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
                self.last_detections = []    # Última lista de deteções para exportar

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self._frame_count += 1

                # Correr YOLO apenas em 1 de cada 3 frames para ganhar FPS
                if self._frame_count % 3 == 0 or self._last_annotated is None:
                    results = model.predict(img, conf=self.confianca, verbose=False, imgsz=320, device=DEVICE)
                    r = results[0]
                    self._last_annotated = r.plot(labels=self.mostrar_labels, conf=self.mostrar_scores)

                    # Guardar deteções do frame atual
                    names = r.names
                    self.last_detections = [
                        {
                            "classe": names[int(cls)],
                            "confianca": round(float(conf), 4),
                            "bbox_xyxy": {
                                "x1": round(float(bbox[0]), 2),
                                "y1": round(float(bbox[1]), 2),
                                "x2": round(float(bbox[2]), 2),
                                "y2": round(float(bbox[3]), 2),
                            },
                        }
                        for cls, conf, bbox in zip(
                            r.boxes.cls.tolist(),
                            r.boxes.conf.tolist(),
                            r.boxes.xyxy.tolist(),
                        )
                    ]

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
    st.caption("Corre ambos os modelos (YOLOv8m e YOLOv8l) na mesma imagem e compara os resultados lado a lado.")

    uploaded_cmp = st.file_uploader("Escolhe uma imagem...", type=["jpg", "jpeg", "png"], key="cmp_upload")

    if uploaded_cmp is not None:
        img_cmp = Image.open(uploaded_cmp)

        if model_m and model_l:
            with st.spinner("A correr ambos os modelos..."):
                res_m = model_m.predict(img_cmp, conf=confianca, device=DEVICE)
                res_l = model_l.predict(img_cmp, conf=confianca, device=DEVICE)

            img_m_rgb = cv2.cvtColor(
                res_m[0].plot(labels=mostrar_labels, conf=mostrar_scores), cv2.COLOR_BGR2RGB
            )
            img_l_rgb = cv2.cvtColor(
                res_l[0].plot(labels=mostrar_labels, conf=mostrar_scores), cv2.COLOR_BGR2RGB
            )

            col_m, col_l = st.columns(2)

            # ── YOLOv8m ──
            with col_m:
                st.markdown("### YOLOv8m")
                st.image(img_m_rgb, use_container_width=True)

                boxes_m = res_m[0].boxes
                names_m = res_m[0].names
                if len(boxes_m) > 0:
                    classes_m   = boxes_m.cls.tolist()
                    confs_m     = boxes_m.conf.tolist()
                    bboxes_m    = boxes_m.xyxy.tolist()
                    det_names_m = [names_m[int(c)] for c in classes_m]
                    df_m = pd.Series(det_names_m).value_counts().reset_index()
                    df_m.columns = ["Peça", "Qtd"]
                    st.dataframe(df_m, hide_index=True, use_container_width=True)
                    st.caption(f"Total: **{len(classes_m)}** | Conf. média: **{sum(confs_m)/len(confs_m):.2%}**")
                    detections_m = [
                        {
                            "classe": names_m[int(cls)],
                            "confianca": round(float(conf), 4),
                            "bbox_xyxy": {"x1": round(float(b[0]),2), "y1": round(float(b[1]),2),
                                          "x2": round(float(b[2]),2), "y2": round(float(b[3]),2)},
                        }
                        for cls, conf, b in zip(classes_m, confs_m, bboxes_m)
                    ]
                else:
                    detections_m = []
                    st.warning("Nenhuma deteção.")

            # ── YOLOv8l ──
            with col_l:
                st.markdown("### YOLOv8l")
                st.image(img_l_rgb, use_container_width=True)

                boxes_l = res_l[0].boxes
                names_l = res_l[0].names
                if len(boxes_l) > 0:
                    classes_l   = boxes_l.cls.tolist()
                    confs_l     = boxes_l.conf.tolist()
                    bboxes_l    = boxes_l.xyxy.tolist()
                    det_names_l = [names_l[int(c)] for c in classes_l]
                    df_l = pd.Series(det_names_l).value_counts().reset_index()
                    df_l.columns = ["Peça", "Qtd"]
                    st.dataframe(df_l, hide_index=True, use_container_width=True)
                    st.caption(f"Total: **{len(classes_l)}** | Conf. média: **{sum(confs_l)/len(confs_l):.2%}**")
                    detections_l = [
                        {
                            "classe": names_l[int(cls)],
                            "confianca": round(float(conf), 4),
                            "bbox_xyxy": {"x1": round(float(b[0]),2), "y1": round(float(b[1]),2),
                                          "x2": round(float(b[2]),2), "y2": round(float(b[3]),2)},
                        }
                        for cls, conf, b in zip(classes_l, confs_l, bboxes_l)
                    ]
                else:
                    detections_l = []
                    st.warning("Nenhuma deteção.")

            # ── Exportar JSON combinado ──
            st.divider()
            export_cmp = {
                "threshold_confianca": confianca,
                "YOLOv8m": {"total_detetado": len(detections_m), "detecoes": detections_m},
                "YOLOv8l": {"total_detetado": len(detections_l), "detecoes": detections_l},
            }
            with st.expander("Pré-visualizar JSON comparativo"):
                st.json(export_cmp)
            st.download_button(
                label="Exportar JSON comparativo",
                data=json.dumps(export_cmp, indent=2, ensure_ascii=False),
                file_name="comparacao_modelos.json",
                mime="application/json",
            )

            # ── Guardar no Histórico (1x por ficheiro) ──
            hist_key_cmp = f"cmp_{uploaded_cmp.file_id}"
            if hist_key_cmp not in st.session_state:
                st.session_state[hist_key_cmp] = True
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for img_rgb, modelo_nome, dets, confs_list in [
                    (img_m_rgb, "YOLOv8m", detections_m, confs_m if len(boxes_m) > 0 else []),
                    (img_l_rgb, "YOLOv8l", detections_l, confs_l if len(boxes_l) > 0 else []),
                ]:
                    st.session_state.historico.append({
                        "imagem": img_rgb,
                        "timestamp": ts,
                        "modelo": modelo_nome,
                        "total": len(dets),
                        "conf_media": sum(confs_list) / len(confs_list) if confs_list else 0.0,
                        "detecoes": dets,
                        "fonte": "Comparação",
                    })
        else:
            st.error("Um ou ambos os modelos não foram carregados.")


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

        # Grid de 3 colunas
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
                    st.divider()