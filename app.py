from flask import Flask, Response
from ultralytics import YOLO
from utils import plot_bboxes
import cv2

app = Flask(__name__)

model = YOLO("yolov8n.pt")
# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture('https://5d84fe297ee2b.streamlock.net/plawo/plawo.stream/chunklist_w971591422.m3u8')
camera.set(cv2.CAP_PROP_FOURCC, 0x47504A4D ); # MJPG codec
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, 30)
print("Resolution: (" + str(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; " + str(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")")
print("FPS: " + str(camera.get(cv2.CAP_PROP_FPS)))
print("FOURCC: " + str(camera.get(cv2.CAP_PROP_FOURCC)))

def gen_frames():
    while True:
            _, img = camera.read()

            output = model.predict(img) # device=0 for GPU

            img = plot_bboxes(img, output[0].boxes.data, score=True, conf=0.2)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'


@app.route('/')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)