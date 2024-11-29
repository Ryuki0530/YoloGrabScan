import cv2
import tkinter as gui
import numpy as np


def main():
    selectedModel = selectModel()
    print("model:"+selectedModel)
    cam = connCam()

    if cam is None or not cam.isOpened():
        print("Camera connection failed")
        gui.messagebox.showerror('エラー','カメラの接続に失敗しました。') 
        return

    handPosition = []
    objectPosition = []

    #MainLoop
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:#27はESCキー
            break


def selectModel():
    result = None
    window = gui.Tk()
    window.title("モデル選択")
    window.geometry("300x200")

    selectedModel = gui.StringVar(value="yolov8m")
    options = ["yolov8n", "yolov8m", "yolov8l", "yolo11m"]

    for option in options:
        gui.Radiobutton(window, text=option, value=option, variable=selectedModel).pack(anchor=gui.W)

    def onSubmit():
        nonlocal result
        result = selectedModel.get()
        window.destroy()  # GUIを閉じる

    submitButton = gui.Button(window, text="決定", command=onSubmit)
    submitButton.pack(pady=10)

    window.mainloop()
    return result


def connCam():
    result1 = None
    window1 = gui.Tk()
    window1.title("カメラ選択")
    window1.geometry("300x200")

    selectedCamera = gui.StringVar(value="OnDeviceCamera")
    options1 = ["OnDeviceCamera", "IPCamera"]

    for option in options1:
        gui.Radiobutton(window1, text=option, value=option, variable=selectedCamera).pack(anchor=gui.W)

    def onSubmit1():
        nonlocal result1
        result1 = selectedCamera.get()
        window1.destroy()  # GUIを閉じる

    submitButton1 = gui.Button(window1, text="決定", command=onSubmit1)
    submitButton1.pack(pady=10)

    window1.mainloop()

    if result1 == "OnDeviceCamera":
        print("Camera:On This Device Camera")
        return cv2.VideoCapture(0)

    result2 = None
    window2 = gui.Tk()
    window2.title("IPカメラ選択")
    window2.geometry("300x200")

    label2 = gui.Label(window2, text="IPアドレスを入力してください")
    label2.pack(pady=10)

    ipEntry = gui.Entry(window2, width=30)
    ipEntry.pack(pady=5)

    def onSubmit2():
        nonlocal result2
        result2 = ipEntry.get()
        window2.destroy()

    submitButton2 = gui.Button(window2, text="決定", command=onSubmit2)
    submitButton2.pack(pady=10)

    window2.mainloop()

    if result2:
        print("Camera:IPcam ["+result2+"]")
        return cv2.VideoCapture(result2)

    return None

def calculateMovement(positions,newPositions):
    positions.append(newPositions)
    if len(positions) > 2:
        positions.pop(0)
    if len(positions) == 2:
        return np.array(positions[1]) - np.array(positions[0])
    return np.array([0, 0])

if __name__ == "__main__":
    main()
