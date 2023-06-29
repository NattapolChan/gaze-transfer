import matplotlib.pyplot as plt
import h5py
import torch
from random import shuffle
import math
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

debug = False

preprocess = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

class MpiigazeDataset(Dataset):
    # MPIIFaceGaze Dataset (INPUT: PREPROCESSED CROPPED FACE IMAGE)
    def __init__(self, path: str, 
                 person_id: tuple[int, ...], gray=True,
                 augmentation=True, outSize=(36,60)
    ):
        self.path = path
        self.images, self.yaw, self.pitch = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
        self.augmentation = augmentation
        self.person_id = person_id
        self.outSize = outSize

        self._load_data()

        print(f"[FINISHED >> Dataset] person {str(person_id)} loaded")

    def _load_data(self):
        for idx in self.person_id:
            person_id_str = str(idx)
            person_id_str = 'p0' + person_id_str if len(person_id_str) == 1 else 'p' + person_id_str
            with h5py.File(self.path, 'r') as f:
                print(person_id_str)
                images = f.get(f'{person_id_str}/image')[()]
                labels = f.get(f'{person_id_str}/gaze')[()] * 180/ math.pi
            images = torch.from_numpy(images).to(torch.float32) 
            images /= 255.0
            images = preprocess(images)
            self.images = torch.concat([self.images, images], axis=0)
            yaw, pitch = labels[:,0] , labels[:,1]
            self.yaw = torch.concat([self.yaw, torch.Tensor(yaw)])
            self.pitch = torch.concat([self.pitch, torch.Tensor(pitch)])
                        

    def __getitem__(
            self,
            index: int
    ):
        return self.images[index], self.yaw[index], self.pitch[index]

    def __len__(self) -> int:
        return len(self.images)

if __name__ == "__main__":
    dataset = MpiigazeDataset("./datasets/MPIIGaze.h5", list(range(0,3)))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for (images, yaws, pitchs) in train_loader:
        if np.random.randint(1000) > 992 and debug:
            index = np.random.randint(32)
            plt.imshow(images[index])
            plt.colorbar()
            plt.title('test image: \n V: ' + str(yaws[index]) + '\n H: ' + str(pitchs[index]))
            plt.show()
    fig, axes = plt.subplots(5,5, figsize=(20,12))
    plt.tight_layout(h_pad=1.5)
    for i in range(25):
        randIdx = np.random.randint(len(dataset.images))
        randIdx = i
        im =axes[i//5, i%5].imshow(dataset.images[randIdx])
        yaw, pitch = dataset.yaw[randIdx], dataset.pitch[randIdx]
        fig.colorbar(im, ax=axes[i//5, i%5], orientation='vertical')
        axes[i//5, i%5].set_title(f"{round(float(yaw), 3)} {round(float(pitch), 3)}")
    plt.show()
