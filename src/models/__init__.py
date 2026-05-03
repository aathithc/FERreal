from src.models.simple_cnn import SimpleCNN
from src.models.resnet import ResNetFER
from src.models.vit import ViTFER

# Add your model class here once implemented.
# The key is the --model CLI argument value.
MODEL_REGISTRY: dict = {
    "simple_cnn": SimpleCNN,
    "resnet": ResNetFER,
    "vit": ViTFER,
}
