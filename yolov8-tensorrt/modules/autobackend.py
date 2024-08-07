# Ultralytics YOLO 🚀, GPL-3.0 license
import contextlib
import json
from collections import OrderedDict, namedtuple
from pathlib import Path
from urllib.parse import urlparse

# import cv2
import numpy as np
import torch
import torch.nn as nn
# from PIL import Image

from ultralytics.utils import LOGGER, ROOT, yaml_load
from ultralytics.utils.checks import check_yaml

def check_class_names(names):
    # Check class names. Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        if not all(isinstance(k, int) for k in names.keys()):  # convert string keys to int, i.e. '0' to 0
            names = {int(k): v for k, v in names.items()}
        if isinstance(names[0], str) and names[0].startswith('n0'):  # imagenet class codes, i.e. 'n01440764'
            map = yaml_load(ROOT / 'datasets/ImageNet.yaml')['map']  # human-readable names
            names = {k: map[v] for k, v in names.items()}
    return names


class AutoBackend(nn.Module):

    def _apply_default_class_names(self, data):
        with contextlib.suppress(Exception):
            return yaml_load(check_yaml(data))['names']
        return {i: f'class{i}' for i in range(999)}  # return default if above errors

    def __init__(self, weights='yolov8n.pt', device=torch.device('cpu'), data=None, fp16=False):
        """
        MultiBackend class for python inference on various platforms using Ultralytics YOLO.

        Args:
            weights (str): The path to the weights file. Default: 'yolov8n.pt'
            device (torch.device): The device to run the model on.
            dnn (bool): Use OpenCV's DNN module for inference if True, defaults to False.
            data (str), (Path): Additional data.yaml file for class names, optional
            fp16 (bool): If True, use half precision. Default: False
            fuse (bool): Whether to fuse the model or not. Default: True

        Supported formats and their naming conventions:
            | Format                | Suffix           |
            |-----------------------|------------------|
            | PyTorch               | *.pt             |
            | TorchScript           | *.torchscript    |
            | ONNX Runtime          | *.onnx           |
            | ONNX OpenCV DNN       | *.onnx dnn=True  |
            | OpenVINO              | *.xml            |
            | CoreML                | *.mlmodel        |
            | TensorRT              | *.engine         |
            | TensorFlow SavedModel | *_saved_model    |
            | TensorFlow GraphDef   | *.pb             |
            | TensorFlow Lite       | *.tflite         |
            | TensorFlow Edge TPU   | *_edgetpu.tflite |
            | PaddlePaddle          | *_paddle_model   |
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        # nn_module = isinstance(weights, torch.nn.Module)
        # pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        # fp16 &= pt or jit or onnx or engine  # FP16
        fp16 = False
        # nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        # stride = 32  # default stride
        model = None  # TODO: resolves ONNX inference, verify effect on other backends
        # cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        # NOTE: special case: in-memory pytorch model
        
        # Tensorrt Model
        LOGGER.info(f'Loading {w} for TensorRT inference...')
        import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
        # check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
        if device.type == 'cpu':
            device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        # Read file
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
            meta = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
            model = runtime.deserialize_cuda_engine(f.read())  # read engine
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        stride, names = int(meta['stride']), meta['names']
        
        # Load external metadata YAML
        w = Path(w)        

        # Check names
        if 'names' not in locals():  # names missing
            names = self._apply_default_class_names(data)
        names = check_class_names(names)

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, visualize=False):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False

        Returns:
            (tuple): Tuple containing the raw output tensor, and the processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape  # batch, channel, height, width            

        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        # if self.nhwc:
        #     im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
        
        # TensorRT
        if self.dynamic and im.shape != self.bindings['images'].shape:
            i = self.model.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings['images'].shape        
        
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]        

        # for x in y:
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)

        Returns:
            (None): This method runs the forward pass and don't return any value
        """
        warmup_types = "engine"
        if any(warmup_types) and (self.device.type != 'cpu'):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(1):  #
                self.forward(im)  # warmup

    # @staticmethod
    # def _model_type(p='path/to/model.pt'):
    #     """
    #     This function takes a path to a model file and returns the model type

    #     Args:
    #         p: path to the model file. Defaults to path/to/model.pt
    #     """
    #     # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
    #     # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
    #     from ultralytics.yolo.engine.exporter import export_formats
    #     sf = list(export_formats().Suffix)  # export suffixes
    #     # if not is_url(p, check=False) and not isinstance(p, str):
    #     #     check_suffix(p, sf)  # checks
    #     url = urlparse(p)  # if url may be Triton inference server
    #     types = [s in Path(p).name for s in sf]
    #     types[8] &= not types[9]  # tflite &= not edgetpu
    #     triton = not any(types) and all([any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
    #     return types + [triton]
