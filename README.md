# Türk Plaka Tanıma Sistemi 

## 1. Verisetini Belirleme

### 1.1 Plaka Tespiti İçin
- Dataset: [Plate Detection](https://universe.roboflow.com/guler-kandeger/plate-detection-vh2rk/dataset/2)

### 1.2 Plaka Okumak İçin
- Dataset: [Plaka Reading](https://universe.roboflow.com/metehan-yasar/plaka-cript/dataset/1)

## 2. Verileri Uygun Formata Getirme

### 2.1 Veri Ayrıştırma
- Yolov8 formatında indirdiğim verisetini Train (%70), Valid (%20), Test (%10) olarak ayrıştırdım.

### 2.2 İsimlendirme
- Ayrıştırdığım verileri yolo formatına uygun isimlere sahip olacak şekilde yeniden isimlendirdim (Eğitime hazır hale getirdim).

## 3. Model Seçimi ve Eğitimleri

### 3.1 Model Seçimi
#### 3.1.1 Tespit için
- **Model**: Yolov8n
- **Sebep**: Model optimizasyonu ve TensorRT dönüşümü için uygun.

#### 3.1.2 Okumak için
- **Model**: Yolov8n
- **Sebep**: Hızlı ve hafif, gerçek zamanlı uygulamalar için uygun.

### 3.2 Eğitimler

#### 3.2.1 Plaka Tespiti İçin
**Başarım Metrikleri:**
- Box(P): 0.998 (Precision, Doğruluk)
- R: 0.973 (Recall, Hatırlama)
- mAP50: 0.994 (Mean Average Precision at IoU 50%)
- mAP50-95: 0.888 (Mean Average Precision at IoU 50% to 95%)

**Hız Metrikleri:**
- Ön İşleme Süresi: 0.4 ms resim başına
- Çıkarım Süresi: 3.0 ms resim başına
- Kayıp Hesaplama Süresi: 0.0 ms resim başına
- Son İşleme Süresi: 4.3 ms resim başına

#### 3.2.2 Plaka Okuma İçin
**Performans Metrikleri:**
- Box(P): 0.971 (Precision, Doğruluk)
- R: 0.975 (Recall, Hatırlama)
- mAP50: 0.98 (Mean Average Precision at IoU 50%)
- mAP50-95: 0.785 (Mean Average Precision at IoU 50% to 95%)

**Hız Metrikleri:**
- Ön İşleme Süresi: 1.4 ms resim başına
- Çıkarım Süresi: 13.0 ms resim başına
- Son İşleme Süresi: 4.5 ms resim başına

## 4. Modellerin Optimizasyonu

### 4.1 İzlenilen Yol
1. Plaka tespiti ve okuma modeli -> Pruning İşlemi (Gereksiz bağlantıları kaldırma)
2. Quantization İşlemi (Sayısal hassasiyeti düşürme, 8-bit yerine 32-bit kullanma)
3. Modeli ONNX'e dönüştürme (modelleri birden fazla platformda ve araçta kullanabilmek için)
4. ONNX modelini basitleştirme (daha hızlı ve daha az kaynak kullanımı için)
5. Modeli TensorRT'ye dönüştürme (NVIDIA'nın derin öğrenme modellerini optimize eden kütüphanesi)

**Bu adımlar fps değerini ve sistemin işlem hızını maksimum seviyeye taşımaktadır.**

## 5. Modelleri Real-Time Kullanılabilir Hale Getirme

### 5.1 Kod Yazma
- Modelleri kullanabilmek için iskelet bir kod yazılması gerekmektedir.
- Modellerin hızını arttırdıktan sonra kullanılacak olan kamera sisteminde en hızlı çıktıyı alabilmek için programın yazıldığı dil çok önemlidir.
- Genelde Python bu konuda çok rahattır ama yüksek seviye bir dil olduğu için hem okuma hem de anlama açısından yavaştır.
- C++ kodu ile modelleri kullanacağımız programın iskeletini oluşturmak daha hızlı veriyi işleyebilmek için önemlidir.
- Kodu yazarken ve kameraya bağlamak için C++ kodu yazıldı.

## 6. Algoritmalarla Verimliliği En Yükseğe Çıkarma ( eksik ) 

### 6.1 Verimlilik Artışı
- Tracking algoritmaları kullanarak her karede tekrar tespit yapmayı önlemek.
- ROI (Region of Interest) tekniklerini uygulamak (Belli Alanlara Daha Fazla İlgi Gösterilmesi).

## 7. Kamera Sistemlerine Entegrasyon

- Algoritmalar tamamlandıktan sonra, kamera sistemlerine entegre hale getirmeye hazır bir program oluşturulması.
