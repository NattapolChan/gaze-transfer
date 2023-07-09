from model import (
    init_weights,
    VGG_3,
    VGG_8,
    VGG_13,
    GazeRedirectionNetwork,
    GazeRepresentationLearning,
    GlobalAlignmentNetwork,
)
from dataloader import MpiigazeDataset

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from loss import Loss
from typing import List
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
writer = SummaryWriter('runs/GazeUnsupervised-run-8')

n_epoch = 275 
debug = False
model_gaze = GazeRepresentationLearning()
model_align = GlobalAlignmentNetwork()
model_redirect = GazeRedirectionNetwork()
model_gaze.to(device)
model_redirect.to(device)
model_align.to(device)
criterion = Loss() 
params = list(model_redirect.parameters()) + list(model_align.parameters()) 
optimizer = torch.optim.Adam(
    params, lr=2e-6
)

model_gaze.load_state_dict(torch.load("pretrained/baseline_25_[25-06-23_20-02]_13.11.pth", map_location=device))
model_gaze.eval()
for param in model_gaze.parameters():
    param.requires_grad = False
VGG = [VGG_3.to(device), VGG_8.to(device), VGG_13.to(device)]

def show_images(images: List[torch.Tensor] | torch.Tensor) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i][0])

    plt.show(block=True)


train_dataset = MpiigazeDataset(path="datasets/MPIIGaze.h5", person_id=tuple(range(1,9)))
val_dataset = MpiigazeDataset(path="datasets/MPIIGaze.h5", person_id=tuple(range(9,12)))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

array_log = []
running_loss = 0.0
running_angle_error = 0.0
i=0

for epoch in range(n_epoch):
    for iter, data in enumerate((pbar:=tqdm(train_loader))):

        images, _, _ = data
        i+=1
        
        # random paring by rotaing src image tensor by 1
        images = images.view(images.size(0), 1, images.size(1), images.size(2)).to(device)
        images_ref = torch.cat(
            [images[1:], images[0].view(1,1,images.size(2), images.size(3))]
        ).view(images.size(0), 1, images.size(2), images.size(3)).to(device)

        if epoch == 0 and iter == 30 and debug:
            
            grid_images = torchvision.utils.make_grid(images.cpu(), nrow = 4)
            writer.add_image("images_test", grid_images)
            show_images(images.cpu())

        
        if epoch == 75:
            model_gaze.train(True)
            for param in model_gaze.parameters():
                param.requires_grad = True

        model_align.train(True)
        model_redirect.train(True)

        optimizer.zero_grad()
        images_aligned = model_align(images, images_ref)

        loss = 0.01 * criterion(images_aligned, images, [], [])

        angle_src = model_gaze(images)
        angle_tgt = model_gaze(images_ref)

        angle_yaw = angle_src[:,0] - angle_tgt[:,0]
        angle_pitch = angle_src[:,1] - angle_tgt[:,1]

        grid_out = model_redirect(images_aligned, angle_yaw, angle_pitch)
        grid_out = torch.permute(grid_out, (0,2,3,1))
        output = F.grid_sample(images, grid_out)

        feature_src, feature_tgt = [], []
        for each in VGG:
            images_rgb = torch.cat([images_ref, images_ref, images_ref], 1)
            outputs_rgb = torch.cat([output, output, output], 1)
            assert images_rgb.size(1) == 3
            feature_src.append(each(images_rgb))
            feature_tgt.append(each(outputs_rgb))
        loss += criterion(images_ref, output, feature_src, feature_tgt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10000 == 9999:

            writer.add_scalar('training_loss', running_loss / 10000, epoch * len(train_loader) + i)

            grid_images = torchvision.utils.make_grid(images, nrow=4)
            grid_ref = torchvision.utils.make_grid(images_ref, nrow=4)
            grid_aligned = torchvision.utils.make_grid(images_aligned, nrow=4)
            grid_output = torchvision.utils.make_grid(output, nrow=4)
            grid_grid = torchvision.utils.make_grid(grid_out[0, :, :, 0])
            writer.add_image("images_src", grid_images)
            writer.add_image("images_ref", grid_ref)
            writer.add_image("images_aligned", grid_aligned)
            writer.add_image("images_tgt", grid_output)
            writer.add_image("grid", grid_grid)

            array_log.append(running_loss/10000)

            pbar.set_description(
                f"EPOCH : {epoch} loss={round(array_log[-1],3)} s_loss={round(float(criterion.loss__style * criterion.coef_spatial),3)} p_loss={round(float(criterion.loss__pixel * criterion.coef_pixel),3)} f_loss={round(float(criterion.loss__feature) * criterion.coef_perceptual, 3)}"
            )
            running_loss = 0.0
            running_angle_error = 0.0
print(array_log)
now = datetime.now()
time_string = now.strftime("%d-%m-%y_%H-%M")
torch.save(model_redirect.state_dict(), f'pretrained/model-redirect-[{time_string}].pth')
torch.save(model_align.state_dict(), f'pretrained/model-align-[{time_string}].pth')
torch.save(model_gaze.state_dict(), f'pretrained/model-gaze-[{time_string}].pth')

# torch.save(
#     model.state_dict(), 
#     f"pretrained/baseline_{n_epoch}_[{time_string}]_{str(round(float(array_log[-1]), 3))}.pth"
# )

