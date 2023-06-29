import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import (
    GlobalAlignmentNetwork,
    GazeRepresentationLearning,
    GazeRedirectionNetwork
)
from torch.utils.data import DataLoader
from dataloader import MpiigazeDataset
from typing import List, Optional

model_align = GlobalAlignmentNetwork()
model_gaze = GazeRepresentationLearning()
model_redirect = GazeRedirectionNetwork()

model_align.load_state_dict(torch.load('pretrained/model-align-[29-06-23_19-40].pth', map_location='cpu'))
model_gaze.load_state_dict(torch.load('pretrained/baseline_25_[25-06-23_20-02]_13.11.pth', map_location='cpu'))
model_redirect.load_state_dict(torch.load('pretrained/model-redirect-[29-06-23_19-40].pth', map_location='cpu'))
model_align.eval()
model_gaze.eval()
model_redirect.eval()

test_image = MpiigazeDataset(path='datasets/MPIIGaze.h5', person_id=(14,))

def to_readable(x: torch.Tensor) -> str:
    return str(round(float(x), 3))

def main():
    labels = []
    image_toshow_list = []
    for iter in range(5):
        src_index, tgt_index = np.random.randint(len(test_image)), np.random.randint(len(test_image))
        image_src, yaw_src, pitch_src = test_image.__getitem__(src_index)
        image_ref, yaw_ref, pitch_ref = test_image.__getitem__(tgt_index)
        image_src = torch.unsqueeze(image_src, 0)
        image_ref = torch.unsqueeze(image_ref, 0) 
        image_tgt = forward_model(image_src, image_ref)

        label = f'{iter} : ({to_readable(yaw_src)}, {to_readable(pitch_src)}) to ({to_readable(yaw_ref)}, {to_readable(pitch_ref)})'
        labels.extend([label + ' src', label + ' ref', label + 'tgt'])
        image_toshow_list.extend([image_src, image_ref, image_tgt])

    show_images(image_toshow_list, labels)

def show_images(
    images: List[torch.Tensor], 
    labels: Optional[List[str]] = None
) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(n//3, 3, i + 1)
        try:
            plt.imshow(images[i][0])
        except:
            plt.imshow(images[i][0][0].detach().numpy())
        plt.colorbar()
        if labels is not None:
            plt.title(labels[i])

    plt.show(block=True)

def forward_model(image_src: torch.Tensor, image_ref: torch.Tensor) -> torch.Tensor:
    image_src = image_src.view(image_src.size(0), 1, image_src.size(1), image_src.size(2)) 
    image_ref = image_src.view(image_ref.size(0), 1, image_ref.size(1), image_ref.size(2)) 
    image_aligned = model_align(image_src, image_ref) # fix later

    angle_src = model_gaze(image_src)
    angle_tgt = model_gaze(image_ref)

    angle_yaw = angle_src[:,0] - angle_tgt[:,0]
    angle_pitch = angle_src[:,1] - angle_tgt[:,1]

    grid_out = model_redirect(image_src, angle_yaw, angle_pitch)
    grid_out = torch.permute(grid_out, (0,2,3,1))

    return F.grid_sample(image_src, grid_out)

if __name__ == '__main__':
    main()
