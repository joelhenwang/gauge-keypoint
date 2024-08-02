from roboflow import Roboflow

rf = Roboflow(api_key="nZYucBVWio1FiOLBEOJz")
project = rf.workspace("personal-4rtkf").project("gauge-mark-detection")
version = project.version(6)
dataset = version.download("yolov8")

