import cv2
import tkinter as gui


def main():
    selectedModel = selectModel()
    print(selectedModel + "を使用します。")
    cam = connCam()

    if cam is None or not cam.isOpened():
        print("カメラの接続に失敗しました。")
        return

    # 映像を表示
    while True:
        ret, frame = cam.read()
        if not ret:
            print("フレームの取得に失敗しました。")
            break

        cv2.imshow("Camera Feed", frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


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
        return cv2.VideoCapture(result2)

    return None


if __name__ == "__main__":
    main()
