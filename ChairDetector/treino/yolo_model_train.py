from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model.train(
    data='IA-8230365-8230196-6/data.yaml',
    epochs=100,
    imgsz=640,
    device='mps', #em windows mudar para 'cuda' (para usar GPU) ou 'cpu'
    project='modelos/yolov8',
    name='v1' #pasta dentro do project onde ficam guardados os resultados do treino. Incrementar para cada nova versão do treino
) #o 6 do data é porque no roboflow é a versão 6 do dataset, quando aumentarmos a versão do dataset temos de alterar aqui tbm