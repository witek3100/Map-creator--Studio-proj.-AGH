import matplotlib.pyplot as plt
import torch
from datasets.transforms import test_val_transforms
from config import BASE_DIR
from PIL import Image
import importlib
import os
import sys
import numpy as np
import math


def predict(model_name, images):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        module = importlib.import_module(f'models.{model_name}')
        model = module.Model()
        model.load_state_dict(torch.load(os.path.join(BASE_DIR, f'src/ml/models/{model_name}.pth'), map_location=torch.device('cpu')))
    except (FileNotFoundError, ModuleNotFoundError):
        print('Model not found')
        sys.exit()

    model = model.to(device)

    with torch.no_grad():

        input = [test_val_transforms(Image.fromarray(image)) for image in images]
        input_tensor = torch.stack(input, dim=0).to(device)

        predictions = []
        for i in range(0, input_tensor.shape[0], 100):
            begin = i
            end = i + 100 if i + 100 < input_tensor.shape[0] else input_tensor.shape[0]

            images_batch = input_tensor[begin:end]
            output = model(images_batch)
            predictions_batch = np.array([int(i[0]) for i in torch.round(output)])
            predictions.extend(predictions_batch)

        return predictions
