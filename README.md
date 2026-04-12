# IA_CHAIR_DETECTOR
# Interface de Deteção Industrial
Esta aplicação demonstra o modelo de IA desenvolvido para a UC de Inteligência Artificial.

## Como correr a aplicação:
1. Criar ambiente virtual: `python -m venv venv` (Python 3.13)
2. Ativar: `source venv/bin/activate` (ou `venv\Scripts\activate` em Windows)
3. Mover: `cd ChairDetector`
4. Instalar dependências: `pip install -r app/requirements.txt`
5. Executar: `streamlit run app/app.py`

## Como descarregar o dataset e treinar o modelo:

### Pré-requisitos
- Ter o ambiente virtual ativo (passos 1-3 acima)
- Criar um ficheiro `.env` na raiz do projeto com a tua API key do Roboflow: ROBOFLOW_API_KEY=a_tua_api_key

### 1. Descarregar o dataset
```bash
python modelos/download_dataset.py
```
Isto vai descarregar o dataset do Roboflow para a pasta `ChairDetector/IA-8230365-8230196-numero_da_versão/`.

> **Nota:** Se o dataset for atualizado no Roboflow, alterar o número da versão em `download_dataset.py` e em `yolo_model_train.py`.

### 2. Treinar o modelo
```bash
python modelos/yolo_model_train.py
```
O treino usa YOLOv8m com 100 épocas e resolução 640px (por enquanto). Os resultados são guardados em `modelos/yolov8/v{n}/` — cada versão tem a sua própria pasta (`v1`, `v2`, ...). Para treinar uma nova versão, altera o parâmetro `name` em `yolo_model_train.py`.

> **Nota:** O parâmetro `device='mps'` está otimizado para Mac com Apple Silicon. Em Windows/Linux mudar para `device='cuda'` (GPU NVIDIA) ou `device='cpu'`.