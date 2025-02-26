import os
from torchvision import transforms
import torch.nn.functional as F
import torch
from PIL import Image
from io import BytesIO

from utils.mamba_out import MambaOut


def load_model():
    model = MambaOut().eval()
    state = torch.load(
        "checkpoints/best_model.pth",
        weights_only=True,
        map_location=torch.device("cpu"),
    )["model_state_dict"]
    model.load_state_dict(state)
    return model


def search_image(image_bytes, model):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    embedded = torch.load(
        "checkpoints/embedded.pth", weights_only=True, map_location=torch.device("cpu")
    )
    result_dict = {}
    with torch.inference_mode():
        _, embed = model(image)
        for i in embedded:
            result_dict[i] = (
                ((F.cosine_similarity(embed, embedded[i]) + 1) / 2 * 100).cpu().item()
            )

    return dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True)[:5])


def upload_pth_embedded(model):
    dict_embedded = {}
    img_fold = "train_data/test"
    img_list = os.listdir(img_fold)
    transform = transforms.ToTensor()
    for img_name in img_list:
        image = Image.open(os.path.join(img_fold, img_name)).convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.inference_mode():
            _, embedded = model(image)
        dict_embedded[img_name] = embedded.detach().cpu()
    torch.save(dict_embedded, "checkpoints/embedded.pth")
