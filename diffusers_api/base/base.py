import torch

class BaseModel:
    def __init__(self):
        self.model = None

    def set_device(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        return self.device

    def load_model(self, model_class, model_name: str, **kwargs):
        self.model = model_class.from_pretrained(model_name, **kwargs)
        device = self.set_device()
        self.model.to(device)

        return self.model

    def generate_image(self, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
