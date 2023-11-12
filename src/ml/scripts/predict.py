import torch
from ..datasets.transforms import test_val_transforms

def predict(model, images):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)

    with torch.no_grad():

        input_tensor = test_val_transforms(images)
        input_tensor = input_tensor.to(DEVICE)

        output = model(input_tensor)

        return (round(i.item()) for i in output)
