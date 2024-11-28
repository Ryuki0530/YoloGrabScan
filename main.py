import cv2
import numpy as np
from ultralytics import YOLO
import TkEasyGUI as gui

def main():
    selectedModel = selectModels()

def selectModels():
    window = gui.Window("モデル選択",width = 500,height = 300)
    options = ["yolov8n","yolov8m","yolov8l","yolov11m"]


main()