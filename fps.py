import cv2
import numpy as np
import time
import onnxruntime as ort
import psutil

# Model yolu
model_path = 'best.onnx'

# ONNX Runtime oturumu oluşturma
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Sınıf isimleri (modelinize göre güncelleyin)
class_names = ['plate']  # Eğer birden fazla sınıfınız varsa, buraya ekleyin

# Performans ölçüm değişkenleri
prev_frame_time = 0
fps_list = []
processing_times = []
memory_usages = []
cpu_usages = []
inference_times = []

frame_count = 0
start_time = time.time()

while True:
    frame_start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü yeniden boyutlandırma ve normalize etme
    input_image = cv2.resize(frame, (640, 640))
    input_image = input_image.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Çıktıyı alma ve süreyi ölçme
    inference_start_time = time.time()
    inputs = {session.get_inputs()[0].name: input_image}
    outputs = session.run(None, inputs)
    inference_time = time.time() - inference_start_time
    inference_times.append(inference_time)

    # Çıktıları işleme
    results = outputs[0][0]
    boxes = results[:4, :].T
    scores = results[4, :]
    class_ids = np.zeros(scores.shape)  # Varsayılan olarak sınıf id'sini 0 olarak alıyoruz

    # (x, y, w, h) -> (x1, y1, x2, y2)
    boxes[:, 2:4] += boxes[:, 0:2]

    # Non-maximum suppression uygulama
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=0.5)

    # İndekslerin işlenmesi
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            score = scores[i]
            class_id = int(class_ids[i])
            x1, y1, x2, y2 = box
            x1 = int(x1 * frame.shape[1] / 640)
            y1 = int(y1 * frame.shape[0] / 640)
            x2 = int(x2 * frame.shape[1] / 640)
            y2 = int(y2 * frame.shape[0] / 640)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Class {class_id}"
            label = f"{class_name} {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Performans metriklerini hesaplama
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_list.append(fps)

    processing_time = time.time() - frame_start_time
    processing_times.append(processing_time)

    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB cinsinden
    memory_usages.append(memory_usage)

    cpu_usage = psutil.cpu_percent()
    cpu_usages.append(cpu_usage)

    # Performans metriklerini ekrana yazdırma
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference Time: {inference_time*1000:.2f} ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Processing Time: {processing_time*1000:.2f} ms", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Memory Usage: {memory_usage:.2f} MB", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"CPU Usage: {cpu_usage:.2f}%", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonucu gösterme
    cv2.imshow("Object Detection", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Toplam çalışma süresini hesapla
total_time = time.time() - start_time

# Ortalama performans metriklerini hesapla ve yazdır
print(f"\nToplam çalışma süresi: {total_time:.2f} saniye")
print(f"Toplam işlenen kare sayısı: {frame_count}")
print(f"Ortalama FPS: {np.mean(fps_list):.2f}")
print(f"Ortalama çıkarım süresi: {np.mean(inference_times)*1000:.2f} ms")
print(f"Ortalama işlem süresi: {np.mean(processing_times)*1000:.2f} ms")
print(f"Ortalama bellek kullanımı: {np.mean(memory_usages):.2f} MB")
print(f"Ortalama CPU kullanımı: {np.mean(cpu_usages):.2f}%")
print(f"Minimum FPS: {np.min(fps_list):.2f}")
print(f"Maksimum FPS: {np.max(fps_list):.2f}")
print(f"Minimum çıkarım süresi: {np.min(inference_times)*1000:.2f} ms")
print(f"Maksimum çıkarım süresi: {np.max(inference_times)*1000:.2f} ms")
print(f"Minimum işlem süresi: {np.min(processing_times)*1000:.2f} ms")
print(f"Maksimum işlem süresi: {np.max(processing_times)*1000:.2f} ms")
