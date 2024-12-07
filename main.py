import cv2
import tkinter as gui
import numpy as np
from ultralytics import YOLO

ON_DEVICE_CAMARA_ID = 1
HAND_CLASS_ID = 0  # 手のクラスID
# TARGET_CLASS_IDS = [28, 29, 32 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,54 ,64 ,67 ,73 ,79 ]
TARGET_CLASS_IDS = [39]
DISTANCE_THRESHOLD = 150  # 距離のしきい値
MOVEMENT_THRESHOLD = 5   # 動きの類似性のしきい値

def main(): 
    print("Start main function")
    selectedModel = selectModel()
    print("model:"+selectedModel)
    model = YOLO(selectedModel)
    cam = connCam()

    if cam is None or not cam.isOpened():
        print("Camera connection failed")
        gui.messagebox.showerror('エラー','カメラの接続に失敗しました。') 
        return

    handPosition = []
    objPosition = []

    #MainLoop
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        
        #Yoloによる推論実行
        resultsTensor = model(frame)
        #テンソルをNumPy配列に変換
        results = resultsTensor[0].boxes.data.cpu().numpy()
        
        hand = []
        objs = []

        #検出された物体の分類と中心の算出
        for result in results:
            x1,y1,x2,y2,conf,classId = result
            center = [(x1+x2)/2,(y1+y2)/2]
            if classId == HAND_CLASS_ID:
                hand.append(center)
            elif classId in TARGET_CLASS_IDS:
                objs.append(center)

        #動きの算出
        handMove = np.array([0,0])
        objMoves = []
        
        for handf in hand:
            handMove = calculateMovement(handPosition,handf)
        for obj in objs:
            objMove = calculateMovement(objPosition,obj)
            objMoves.append(objMove)


        holding = False
        for handf, hMove in zip(hand,[handMove]):
            for obj, oMove in zip(objs, objMoves):
                #状態算出
                distance = np.linalg.norm(np.array(handf) - np.array(obj))
                movementSimilarity = np.linalg.norm(hMove -oMove)

                #判定
                if distance < DISTANCE_THRESHOLD and movementSimilarity < MOVEMENT_THRESHOLD:
                    holding = True
                    break

       
        #結果を表示
        for handf in hand:
            cv2.circle(frame,(int(handf[0]), int(handf[1])),5,(255,0,0),-1)
        for obj in objs:
            cv2.circle(frame, (int(obj[0]),int(obj[1])),5,(0,225,0),-1)
        if holding:
            cv2.putText(frame, "Holding Object",(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)       


        cv2.imshow("Yolo Grab", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:#27はESCキー
            break




def selectModel():
    result = None
    window = gui.Tk()
    window.title("モデル選択")
    window.geometry("300x200")

    selectedModel = gui.StringVar(value="yolov8m")
    options = ["yolov8n","yoloV8n", "yolov8m", "yolov8l", "yolo11m"]

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
        return cv2.VideoCapture(ON_DEVICE_CAMARA_ID)

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


print("Set uped sub functions.")
if __name__ == "__main__":
    main()