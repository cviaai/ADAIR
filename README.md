[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)

# ADAIR
This is the official repository of the IEEE Control Systems Letters submission entitled 'Adaptive Denoising and Alignment Agents for Infrared Imaging'.

## Overview
Namely, the Denoiser learns the proper frequency decomposition of the acquired infrared data until the target segmentation metric is maximized; whereas, the Aligner learns the intensity fluctuations within the segmentation mask tuned by the Denoiser until its maximal overlap with the source infrared image.

## Code structure
- `./rl/`: folder where Q-window reinforcement learning codes are stored. 
    1) `./rl/Mask_Proj_Env.py`: projection-related routine. 
    2) `./rl/run.py`: run to start the alignment

## Citing 

If you use this package in your publications or in other work, please cite it as follows:

```
@misc{ADAIR,
  author = {Vito Leli and Viktor Shipitsin and Oleg Rogov and Aleksandr Sarachakov and Dmitry V. Dylov}
  title = {Adaptive Denoising and Alignment Agents for Infrared Imaging},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cviaai/ADAIR/}}
}
```
