import roboflow
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

rf = roboflow.Roboflow(api_key)

project = rf.workspace("8230365-8230196").project("ia-8230365-8230196")
dataset = project.version(7).download("yolov8")