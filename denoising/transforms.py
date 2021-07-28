import random
import math
import torch
import torchvision


class Resize(object):
    def __init__(self, size):
        self.resize = torchvision.transforms.Resize(size, interpolation=2)

    def __call__(self, sample):
        image, mask = sample

        image = self.resize(image)
        mask = self.resize(mask)

        return image, mask


class ToTensor(object):
    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        image, mask = sample

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask


class BatchNoiseOverlay(object):
    def __init__(self):
        pass
    
    def __call__(self, batch):
        images, masks = batch
        return torch.stack([self.noise_overlay(images[i]) for i in range(images.shape[0])], dim=0), masks
    
    def noise_overlay(self, tensor):
        return tensor


class BatchRicianNoise(BatchNoiseOverlay):
    def __init__(self, variance=(0, 0.1)):
        super(BatchRicianNoise, self).__init__()
        
        self.variance = variance
    
    def noise_overlay(self, tensor):
        if type(self.variance) is tuple:
            variance = random.uniform(self.variance[0], self.variance[1])
        else:
            variance = self.variance
        
        return torch.clamp(torch.sqrt(torch.pow(tensor + torch.FloatTensor(tensor.shape).normal_(mean=0.0, std=variance), 2) +\
                                      torch.pow(torch.FloatTensor(tensor.shape).normal_(mean=0.0, std=variance), 2)), min=0.0, max=1.0)


class BatchRandomErasing(BatchNoiseOverlay):
    """
    Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, Yi Yang, Random Erasing Data Augmentation.
    """
    def __init__(self, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        super(BatchRandomErasing, self).__init__()
        
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def noise_overlay(self, tensor):
        for attempt in range(100):
            area = tensor.shape[1] * tensor.shape[2] 
        
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < tensor.shape[2] and h < tensor.shape[1]:
                x1 = random.randint(0, tensor.shape[1] - h)
                y1 = random.randint(0, tensor.shape[2] - w)
                if tensor.shape[0] == 3:
                    tensor[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    tensor[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    tensor[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    tensor[0, x1:x1+h, y1:y1+w] = self.mean[0]
                
                return tensor

        return tensor
