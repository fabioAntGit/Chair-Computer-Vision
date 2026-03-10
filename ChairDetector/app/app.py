import streamlit as st

st.sidebar.title("Configurações")
st.sidebar.markdown("Usa este menu para ajustar os parâmetros do modelo.")

st.sidebar.divider()

st.sidebar.markdown("### Ajustes do Modelo")
confianca = st.sidebar.slider(
    "Confiança (Threshold)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
)
st.sidebar.divider()

st.sidebar.markdown("### Interface Visual")
mostrar_labels = st.sidebar.checkbox("Mostrar Nome da Peça", value=True)
mostrar_scores = st.sidebar.checkbox("Mostrar % de Certeza", value=True)

st.sidebar.divider()

st.sidebar.badge("Projeto IA - 2026 | 8230365 | 8230196", color="violet")


st.markdown("<h1 style='text-align: center;'>Interface IA</h1>", unsafe_allow_html=True)

st.divider()
st.badge(f"A confiança atual: **{confianca * 100:.0f}%**", color="grey")
st.badge(f"Mostrar Nome da Peça: **{mostrar_labels}**", color="grey")
st.badge(f"Mostrar % de Certeza: **{mostrar_scores}**", color="grey")