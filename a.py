import onnx

model_path = 'yolov8n_optimized.onnx'

model = onnx.load(model_path)

input_names = [input.name for input in model.graph.input]
print("Modelin giriş isimleri:", input_names)