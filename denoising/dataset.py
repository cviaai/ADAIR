from torch.utils.data.dataset import Dataset
from PIL import Image


class DenoisingDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def read_tiff_file(self, path):
        image = Image.open(path)

        return image.convert("L")

    def do_transform(self):
        return self.transform is not None

    def __getitem__(self, index):
        image = self.read_tiff_file(self.dataset.iloc[index]['image'])
        mask = self.read_tiff_file(self.dataset.iloc[index]['image'])

        if self.do_transform():
            image, mask = self.transform((image, mask))

        return image, mask

    def __len__(self):
        return len(self.dataset)
