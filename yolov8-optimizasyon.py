import torch
from ultralytics import YOLO
from torch.onnx import export
import onnx
from onnxsim import simplify
import tensorrt as trt
import os

# Adım 1: Model Yükleme
model = YOLO('path/to/your/yolov8n.pt')  # Eğitilmiş modelinizin yolunu belirtin

# Adım 2: Model Pruning
# YOLOv8, otomatik olarak pruning uygular, ek bir işlem gerekmez

# Adım 3: Quantization
model.model = torch.quantization.quantize_dynamic(
    model.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)

# Adım 4: ONNX'e dönüştürme
input_names = ['images']
output_names = ['output0', 'output1']
dynamic_axes = {'images': {0: 'batch'}, 'output0': {0: 'batch'}, 'output1': {0: 'batch'}}

torch.onnx.export(model.model, 
                  torch.randn(1, 3, 640, 640),
                  'yolov8n_optimized.onnx',
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes=dynamic_axes,
                  opset_version=11)

# Adım 5: ONNX Model Simplification
onnx_model = onnx.load('yolov8n_optimized.onnx')
simplified_model, check = simplify(onnx_model)
onnx.save(simplified_model, 'yolov8n_simplified.onnx')


