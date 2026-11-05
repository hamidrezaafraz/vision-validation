import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import transforms

def similar_pairs(X, thr, chunk=2048, device="cpu"):
    """X: (N,D) L2-normalised. Returns (i,j), i<j, with cosine >= thr."""
    X = X.to(device)
    out = []
    if len(X) == 0:
        return np.empty((0, 2), dtype=int)
    for s in tqdm(range(0, len(X), chunk), desc="Finding similar pairs"):
        i, j = torch.where(X[s:s + chunk] @ X.T >= thr)
        i = i + s
        m = i < j
        out.append(torch.stack([i[m], j[m]], 1).cpu())
    return torch.cat(out).numpy()

def get_transform(model_name):
    """Get the appropriate transform for the specified model."""
    import torchvision.models as models
    if model_name == "alexnet":
        return models.AlexNet_Weights.DEFAULT.transforms()
    elif model_name == "resnet18":
        return models.ResNet18_Weights.DEFAULT.transforms()
    elif model_name in ["facenet-vggface2", "facenet-casia-webface"]:
        return transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    raise ValueError(f"Model {model_name} not supported.")
    

def get_model(model_name:str = "alexnet", device:str = "cpu"):
    """
    Get a pre-trained model from torchvision.

    Parameters:
    - model_name: str, name of the model (default: "alexnet")
    - device: str, device to load the model on (default: "cpu")

    Returns:
    - model: torch.nn.Module, the pre-trained model
    """
    import torchvision.models as models

    if model_name == "alexnet":
        w = models.AlexNet_Weights.DEFAULT
        model = models.alexnet(weights=w)
    elif model_name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=w)
    elif model_name == "facenet-vggface2":
        model = InceptionResnetV1(pretrained='vggface2', classify=False, device=device)
    elif model_name == "facenet-casia-webface":
        model = InceptionResnetV1(pretrained='casia-webface', classify=False, device=device)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # Remove the final classification layer
    if hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()
    elif hasattr(model, 'classifier'):
        model.classifier = torch.nn.Identity()

    return model.to(device).eval()

def get_embeddings(model, sub_ds, batch_size, device="cpu"):
    # Placeholder for the actual implementation of get_embeddings
    feats = []
    dataloader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Computing features"):
            feats.append(F.normalize(model(x.to(device)), dim=1).cpu())
            
    emb = torch.cat(feats)
    return emb