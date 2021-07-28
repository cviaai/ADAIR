import torch
from IQA_pytorch import FSIM as fsim
from pytorch_msssim import ssim, ms_ssim


class PSNR(object):
    def __init__(self):
        pass

    def __call__(self, cleaned_images, images):
        mse = torch.mean((cleaned_images - images) ** 2, dim=[1, 2, 3])
        psnr_coefficients = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        return torch.mean(psnr_coefficients)


class SSIM(object):
    """For multiscale param:
    
    https://www.researchgate.net/publication/3327793_Image_Quality_Assessment_From_Error_Visibility_to_Structural_Similarity
    https://www.researchgate.net/publication/4071876_Multiscale_structural_similarity_for_image_quality_assessment
    """
    def __init__(self, multiscale=False):
        self.multiscale = multiscale
    
    def __call__(self, cleaned_images, images):
        if self.multiscale:
            return ms_ssim(cleaned_images, images, data_range=1, size_average=True)
        else:
            return ssim(cleaned_images, images, data_range=1, size_average=True)


class FSIM(object):
    """
    https://www.researchgate.net/publication/224216147_FSIM_A_Feature_SIMilarity_index_for_image_quality_assessment
    """
    def __init__(self):
        pass
    
    def __call__(self, cleaned_images, images):
        return torch.mean(fsim()(cleaned_images, images, as_loss=False))
