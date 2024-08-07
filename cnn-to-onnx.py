import torch
import torch.nn as nn
import torch.onnx

class PlateReaderCNN(nn.Module):
    def __init__(self):
        super(PlateReaderCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 50, 128)
        self.fc2 = nn.Linear(128, 36)  # 26 harf + 10 rakam

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 50)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = PlateReaderCNN()
model.load_state_dict(torch.load('...'))
model.eval()

dummy_input = torch.randn(1, 3, 50, 200)

torch.onnx.export(model, 
                  dummy_input, 
                  "plate_reader_cnn.onnx", 
                  export_params=True, 
                  opset_version=11, 
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

print("CNN model ONNX formatında kaydedildi: plate_reader_cnn.onnx")

import onnx
onnx_model = onnx.load("plate_reader_cnn.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model doğrulaması başarılı.")

from onnxsim import simplify
simplified_model, check = simplify(onnx_model)
onnx.save(simplified_model, "plate_reader_cnn_simplified.onnx")
print("Basitleştirilmiş ONNX model kaydedildi: plate_reader_cnn_simplified.onnx")
