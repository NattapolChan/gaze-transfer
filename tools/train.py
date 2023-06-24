from model import GazeRepresentationLearning, init_weights
from dataloader import MpiigazeDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils import find_abs_angle_difference

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
writer = SummaryWriter('runs/GazeBaseline')
n_epoch = 100 
debug = False
model = GazeRepresentationLearning()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

if __name__ == "__main__":
    train_dataset = MpiigazeDataset(path="datasets/MPIIGaze.h5", person_id=range(1,9))
    val_dataset = MpiigazeDataset(path="datasets/MPIIGaze.h5", person_id=range(9,12))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=2)

    for name, param in model.named_parameters():
        print(name)

    print(model)

    model.train()
    
    array_log = []
    running_loss = 0.0
    running_angle_error = 0.0
    i=0

    for epoch in range(n_epoch):
        for _, data in enumerate((pbar:=tqdm(train_loader))):
            images, yaws, pitchs = data
            i+=1
            if np.random.randint(1000) > 994 and debug:
                index = np.random.randint(32)
                plt.imshow(images[index])
                plt.title('test image: \n V: ' + str(yaws[index]) + '\n H: ' + str(pitchs[index]))
                plt.show()
            images = images.view(images.size(0), 1, images.size(1), images.size(2)).to(device)

            labels = torch.stack([yaws, pitchs]).to(device)
            labels = labels.permute((1,0))

            model.train(True)
            optimizer.zero_grad()
            outputs = model(images)
            assert outputs.size() == labels.size()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_angle_error += find_abs_angle_difference(
                torch.abs(outputs[:,0]-labels[:,0]), torch.abs(outputs[:,1]-labels[:,1])
            ).to('cpu')
            running_loss += loss.item()

            del images
            del labels
            del outputs

            if i % 100 == 99:

                writer.add_scalar('training_loss', running_loss / 100, epoch * len(train_loader) + i)
                writer.add_scalar('training_angle_error', running_angle_error / 100, epoch * len(train_loader) + i)
                array_log.append(running_loss/100)
                pbar.set_description(f"{epoch=} loss={array_log[-1]} error={running_angle_error/100}")

                model.train(False)
                val_error = 0.0
                val_loss = 0.0
    
                with torch.no_grad():
                    for _, data in enumerate(val_loader):
                        images, yaws, pitchs = data
                        images = images.view(images.size(0), 1, images.size(1), images.size(2)).to(device)
                        labels = torch.stack([yaws, pitchs]).to(device)
                        labels = labels.permute((1,0))
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        val_error += find_abs_angle_difference(
                            torch.abs(outputs[:,0]-labels[:,0]), torch.abs(outputs[:,1]-labels[:,1])
                        ).to('cpu')
                        val_error = float(val_error)
                running_angle_error = float(running_angle_error)

                pbar.set_description(
                    f"EPOCH : {epoch} loss={round(array_log[-1],3)} / {round(val_loss/len(val_loader),3)} error={round(running_angle_error/100,3)} / {round(val_error/len(val_loader),3)}"
                )

                writer.add_scalar('val_loss', val_loss/len(val_loader), epoch * len(train_loader) + i)
                writer.add_scalar('val_angle_error', val_error/len(val_loader), epoch * len(train_loader) + i)

                running_loss = 0.0
                running_angle_error = 0.0

    print(array_log)
    now = datetime.now()
    time_string = now.strftime("%d-%m-%y_%H-%M")
    torch.save(
        model.state_dict(), 
        f"pretrained/baseline_{n_epoch}_[{time_string}]_{str(round(float(array_log[-1]), 3))}.pth"
    )

