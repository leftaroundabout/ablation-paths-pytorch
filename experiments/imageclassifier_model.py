
import timm

class TrainedTimmModel:
    def __init__(self, timm_model_name):
        self.timm_model_name = timm_model_name
        self.model = timm.create_model(timm_model_name, pretrained=True)
        self.model.eval()
    def __call__(self, model_input):
        return self.model(model_input)
    def to(self, torchdevice):
        return self.model.to(torchdevice)
