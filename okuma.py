from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2
import time
import psutil
import numpy as np

# Model yükleme
model = YOLO("reading.pt")  # veya "detection_updated.pt" eğer önceki adımda yeni bir model oluşturduysanız

# Performans ölçüm değişkenleri
fps_list = []
processing_times = []
memory_usages = []
cpu_usages = []

frame_count = 0
start_time = time.time()

# Video capture başlatma
cap = cv2.VideoCapture(0)  # 0 yerine video dosyası yolu da kullanabilirsiniz

while True:
    frame_start_time = time.time()

    # Kameradan frame okuma
    ret, frame = cap.read()
    if not ret:
        break

    # Tahmin yapma
    results = model(frame)

    # Sonuçları işleme ve görselleştirme
    annotated_frame = results[0].plot()

    # FPS hesaplama
    processing_time = time.time() - frame_start_time
    fps = 1 / processing_time
    fps_list.append(fps)
    processing_times.append(processing_time)

    # Bellek kullanımını ölçme
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB cinsinden
    memory_usages.append(memory_usage)

    # CPU kullanımını ölçme
    cpu_usage = psutil.cpu_percent()
    cpu_usages.append(cpu_usage)

    # Performans metriklerini ekrana yazdırma
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Processing Time: {processing_time*1000:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Memory Usage: {memory_usage:.2f} MB", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"CPU Usage: {cpu_usage:.2f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonucu gösterme
    cv2.imshow("YOLO Tespiti", annotated_frame)

    frame_count += 1

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()

# Toplam çalışma süresini hesapla
total_time = time.time() - start_time

# Ortalama performans metriklerini hesapla ve yazdır
print(f"\nToplam çalışma süresi: {total_time:.2f} saniye")
print(f"Toplam işlenen kare sayısı: {frame_count}")
print(f"Ortalama FPS: {np.mean(fps_list):.2f}")
print(f"Ortalama işlem süresi: {np.mean(processing_times)*1000:.2f} ms")
print(f"Ortalama bellek kullanımı: {np.mean(memory_usages):.2f} MB")
print(f"Ortalama CPU kullanımı: {np.mean(cpu_usages):.2f}%")
print(f"Minimum FPS: {np.min(fps_list):.2f}")
print(f"Maksimum FPS: {np.max(fps_list):.2f}")
print(f"Minimum işlem süresi: {np.min(processing_times)*1000:.2f} ms")
print(f"Maksimum işlem süresi: {np.max(processing_times)*1000:.2f} ms")