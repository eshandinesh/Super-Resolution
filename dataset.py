import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Train_Dataset_Prep(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(Train_Dataset_Prep, self).__init__()
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
        self.hr_preprocess = transforms.Compose([transforms.CenterCrop(384), transforms.RandomCrop(crop_size), transforms.ToTensor()])
        self.lr_preprocess = transforms.Compose([transforms.RandomCrop(crop_size // upscale_factor), transforms.ToTensor()])

    def __getitem__(self, index):
        hr_image = self.hr_preprocess(Image.open(self.image_filenames[index]))
        lr_image = self.lr_preprocess(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)