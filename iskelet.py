import cv2
import numpy as np
from ultralytics import YOLO
import time

class PlakaTanimaSystemi:
    def __init__(self, plaka_tespit_model_path, plaka_okuma_model_path):
        self.plaka_tespit_model = YOLO(plaka_tespit_model_path)
        self.plaka_okuma_model = YOLO(plaka_okuma_model_path)
        self.karakter_listesi = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def tespit_et(self, frame):
        results = self.plaka_tespit_model(frame)
        return results[0].boxes.data.cpu().numpy()

    def oku(self, plaka_goruntu):
        results = self.plaka_okuma_model(plaka_goruntu)
        if results and len(results[0].boxes) > 0:
            # Karakterleri soldan sağa sırala
            chars = sorted(results[0].boxes.data.cpu().numpy(), key=lambda x: x[0])
            plaka_metni = ""
            for char in chars:
                class_id = int(char[5])
                confidence = char[4]
                if confidence > 0.5:  # Güven eşiği
                    plaka_metni += self.karakter_listesi[class_id]
            return plaka_metni
        return ""

def main():
    plaka_tespit_model_path = 'detection.pt'
    plaka_okuma_model_path = 'reading.pt'
    sistem = PlakaTanimaSystemi(plaka_tespit_model_path, plaka_okuma_model_path)

    cap = cv2.VideoCapture(0)  # 0 yerine video dosyası yolu da kullanılabilir

    # FPS hesaplama için değişkenler
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        plakalar = sistem.tespit_et(frame)

        for plaka in plakalar:
            x1, y1, x2, y2, conf, cls = plaka
            if conf > 0.5:  # Güven eşiği
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                plaka_goruntu = frame[y1:y2, x1:x2]
                plaka_metni = sistem.oku(plaka_goruntu)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plaka_metni, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # FPS hesaplama
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:  # Her saniyede bir FPS'i güncelle
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # FPS'i ekranda göster
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Plaka Tanima', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()