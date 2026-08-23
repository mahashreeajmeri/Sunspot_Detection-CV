from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from .dataset_utils import read_yolo_labels, yolo_to_bbox
import numpy as np 

class SunspotDataset(Dataset):

    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.image_files = sorted(self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        #  get image
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        # find label
        label_path = self.label_dir/f"{image_path.stem}.txt"
        boxes=[]

        # read annotations and convert into boxes
        if label_path.exists():
            annotations= read_yolo_labels(label_path)
            for annotation in annotations:
                box = yolo_to_bbox(annotation, image_width, image_height)
                boxes.append(box)

        # convert boxes to tensor
        boxes= torch.tensor(boxes, dtype= torch.float32).reshape(-1,4)

        # assign 1 for each sunspot and count
        labels= torch.ones(len(boxes), dtype=torch.int64)

        # convert image to tensor
        image= torch.tensor(np.array(image),dtype=torch.float32)

        # convert tensor form HxWxC to CxHxW 
        image= torch.permute(image,(2,0,1))

        # normalize RGB values to 0-1
        image= image/255.0

        target= {"boxes":boxes,"labels":labels}

        return image, target

def collate_fn(batch):
        images,targets=zip(*batch)
        return list(images), list(targets)
    