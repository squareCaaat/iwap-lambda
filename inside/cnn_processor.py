import torch
import torch.nn as nn  
from torchvision import transforms, models
import numpy as np
import os
from typing import Optional, Dict, Any

class FeatureMapProcessor:
    def __init__(self):
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.fc: Dict[str, torch.Tensor] = {}
        self.last_result: Optional[Dict[str, Any]] = None
        self._model: Optional[torch.nn.Module] = None

    def load_model(self):
        if self._model is None:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 10)

            if not os.path.exists("resnet18_mnist.pth"):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: resnet18_mnist.pth")

            state_dict = torch.load("resnet18_mnist.pth", map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
                
            self._model = model
        return self._model

    def register_hooks(self, model: torch.nn.Module):
        def save_feature(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook

        def save_fc(name):
            def hook(module, input, output):
                self.fc[name] = output.detach()
            return hook

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(save_feature(name))

        model.fc.register_forward_hook(save_fc("fc"))

    def process_image(self, pil_image, model: torch.nn.Module):
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
        ])

        tensor = transform(pil_image).unsqueeze(0)
        _ = model(tensor)

    def normalization(self, arr: torch.Tensor):
        arr = arr.cpu().numpy()
        arr_min = arr.min()
        arr_max = arr.max()

        normalized = ((arr - arr_min) / (arr_max - arr_min)) * 255
        normalized = normalized.astype(np.uint8)
        return normalized

    def get_normalized_outputs(self, pil_image=None):
            if pil_image is not None:
                self.feature_maps.clear()
                self.fc.clear()
                
                model = self.load_model()
                self.register_hooks(model)
                self.process_image(pil_image, model)

                fmap_out = {
                    layer_name: self.normalization(fmap).tolist()
                    for layer_name, fmap in self.feature_maps.items()
                }
                fc_out = {
                    layer_name: self.normalization(fmap).tolist()
                    for layer_name, fmap in self.fc.items()
                }
                self.last_result = {"layers": {**fmap_out, **fc_out}}

            return self.last_result

processor = FeatureMapProcessor()

def get_normalized_outputs(pil_image=None):
    return processor.get_normalized_outputs(pil_image)
