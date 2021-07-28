import torch
from pytorch_msssim import ssim, ms_ssim


class CombinedLoss(torch.nn.Module):
    def __init__(self, losses, coefficients=[0.4, 0.6]):
        super(CombinedLoss, self).__init__()
        
        self.losses = losses
        self.coefficients = coefficients
    
    def forward(self, predicted_masks, masks):
        loss = 0.0
        for loss_function, coefficient in zip(self.losses, self.coefficients):
            loss += coefficient * loss_function(predicted_masks, masks)
        
        return loss


class SIMMLoss(torch.nn.Module):
    def __init__(self, multiscale=False):
        super(SIMMLoss, self).__init__()
        
        if multiscale:
            self.metric = ms_ssim
        else:
            self.metric = ssim
        
    def forward(self, cleaned_images, images):
        return 1 - self.metric(cleaned_images, images, data_range=1, size_average=True)


class L1Loss(torch.nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        
        self.loss = torch.nn.L1Loss(reduction='mean')
    
    def forward(self, cleaned_images, images):
        return self.loss(cleaned_images, images)
