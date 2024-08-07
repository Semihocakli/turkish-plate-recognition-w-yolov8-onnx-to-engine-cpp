from ultralytics import YOLO
import cv2
import time
import psutil
import numpy as np

# Model yükleme
model = YOLO("detection.pt")

# Video capture başlatma
cap = cv2.VideoCapture(0)  # 0 yerine video dosyası yolu da kullanabilirsiniz

# Performans ölçüm değişkenleri
prev_frame_time = 0
fps_list = []
processing_times = []
memory_usages = []
cpu_usages = []

frame_count = 0
start_time = time.time()

while True:
    # Kameradan frame okuma
    ret, frame = cap.read()
    if not ret:
        break

    frame_start_time = time.time()

    # Tahmin yapma
    results = model(frame)

    # Sonuçları işleme
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Sınırlayıcı kutu koordinatları
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Güven skoru
            conf = round(float(box.conf[0]) * 100, 2)  # Yüzde olarak güven skoru

            # Sınırlayıcı kutu çizme
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # "Plaka" yazısını ve güven skorunu ekleme
            label = f"Plaka {conf}%"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # FPS hesaplama
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_list.append(fps)

    # İşlem süresini hesaplama
    processing_time = time.time() - frame_start_time
    processing_times.append(processing_time)

    # Bellek kullanımını ölçme
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB cinsinden
    memory_usages.append(memory_usage)

    # CPU kullanımını ölçme
    cpu_usage = psutil.cpu_percent()
    cpu_usages.append(cpu_usage)

    # Performans metriklerini ekrana yazdırma
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Processing Time: {processing_time * 1000:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.putText(frame, f"Memory Usage: {memory_usage:.2f} MB", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"CPU Usage: {cpu_usage:.2f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonucu gösterme
    cv2.imshow("Plaka Tespiti", frame)

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
print(f"Ortalama işlem süresi: {np.mean(processing_times) * 1000:.2f} ms")
print(f"Ortalama bellek kullanımı: {np.mean(memory_usages):.2f} MB")
print(f"Ortalama CPU kullanımı: {np.mean(cpu_usages):.2f}%")
print(f"Minimum FPS: {np.min(fps_list):.2f}")
print(f"Maksimum FPS: {np.max(fps_list):.2f}")
print(f"Minimum işlem süresi: {np.min(processing_times) * 1000:.2f} ms")
print(f"Maksimum işlem süresi: {np.max(processing_times) * 1000:.2f} ms")