#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

class PlakaTanimaSystemi {
private:
    nvinfer1::ICudaEngine* plakaDetektorEngine;
    nvinfer1::IExecutionContext* plakaDetektorContext;
    cv::dnn::Net plakaOkuyucuModel;

    // TensorRT engine'i yüklemek için yardımcı fonksiyon
    nvinfer1::ICudaEngine* loadTRTEngine(const std::string& engineFile) {
        std::ifstream file(engineFile, std::ios::binary);
        if (!file.good()) {
            std::cerr << "'" << engineFile << "' engine dosyası bulunamadı" << std::endl;
            return nullptr;
        }
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        return runtime->deserializeCudaEngine(engineData.data(), size);
    }

public:
    PlakaTanimaSystemi(const std::string& yoloEnginePath, const std::string& cnnModelPath) {
        // YOLOv8 TensorRT engine'ini yükle
        plakaDetektorEngine = loadTRTEngine(yoloEnginePath);
        if (plakaDetektorEngine == nullptr) {
            throw std::runtime_error("Plaka detektör engine'i yüklenemedi!");
        }
        plakaDetektorContext = plakaDetektorEngine->createExecutionContext();

        // CNN ONNX modelini yükle
        plakaOkuyucuModel = cv::dnn::readNetFromONNX(cnnModelPath);
        plakaOkuyucuModel.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        plakaOkuyucuModel.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    std::vector<cv::Rect> tespit_et(const cv::Mat& frame) {
        // TensorRT inference kodu buraya gelecek
        // Bu kısım, TensorRT API'sini kullanarak inference işlemini gerçekleştirecek
        std::vector<cv::Rect> plakalar;
        // ... (TensorRT inference ve çıktı işleme kodu)
        return plakalar;
    }

    std::string oku(const cv::Mat& plakaGoruntu) {
        cv::Mat blob;
        cv::dnn::blobFromImage(plakaGoruntu, blob, 1/255.0, cv::Size(200, 50));
        plakaOkuyucuModel.setInput(blob);

        cv::Mat cikti = plakaOkuyucuModel.forward();

        // CNN çıktısını işle ve plaka metnini döndür
        std::string plakaMetni = "";
        // ... (CNN çıktısını işleme kodu)
        return plakaMetni;
    }

    ~PlakaTanimaSystemi() {
        if (plakaDetektorContext) plakaDetektorContext->destroy();
        if (plakaDetektorEngine) plakaDetektorEngine->destroy();
    }
};

int main() {
    PlakaTanimaSystemi sistem("yolov8n_optimized.engine", "plate_reader_cnn_simplified.onnx");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Kamera açılamadı!" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;

        std::vector<cv::Rect> plakalar = sistem.tespit_et(frame);

        for (const auto& plaka : plakalar) {
            cv::Mat plakaGoruntu = frame(plaka);
            std::string plakaMetni = sistem.oku(plakaGoruntu);
            
            cv::rectangle(frame, plaka, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, plakaMetni, plaka.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Plaka Tanima", frame);
        if (cv::waitKey(1) == 27) break; // ESC tuşuna basılırsa çık
    }

    return 0;
}
