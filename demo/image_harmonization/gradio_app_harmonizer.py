import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf

from src import model


def harmonize_image(comp, mask):
    cuda = torch.cuda.is_available()
    
    # create/load the harmonizer model
    print('Create/load Harmonizer...')
    harmonizer = model.Harmonizer()
    if cuda:
        harmonizer = harmonizer.cuda()
    harmonizer.load_state_dict(torch.load(args.pretrained), strict=True)
    harmonizer.eval()

    if comp.size[0] != mask.size[0] or comp.size[1] != mask.size[1]:
        print('The size of the composite image and the mask are inconsistent')
        exit()

    comp = tf.to_tensor(comp)[None, ...]
    mask = tf.to_tensor(mask)[None, ...]

    if cuda:
        comp = comp.cuda()
        mask = mask.cuda()

    # harmonization
    with torch.no_grad():
        arguments = harmonizer.predict_arguments(comp, mask)
        harmonized = harmonizer.restore_image(comp, mask, arguments)[-1]

    # save the result
    harmonized = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
    harmonized = Image.fromarray(harmonized.astype(np.uint8))
    # harmonized.save(os.path.join(args.example_path, 'harmonized', example))

    print('Finished.')
    print('\n')
    return harmonized
