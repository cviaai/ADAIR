import yaml
import os

import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from utils import get_root

from transforms import Resize, ToTensor, BatchRicianNoise, BatchRandomErasing
from dataset import DenoisingDataset

from models import DnCNN

from metrics import PSNR, SSIM, FSIM
from losses import SIMMLoss, L1Loss, CombinedLoss

# for printing
torch.set_printoptions(precision=2)

# for reproducibility
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params(root):
    with open(os.path.join(root, "configs.yaml"), "r") as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)
    params = {'train_data': configs['paths']['data']['train_data'],
              'dataset_table_path': configs['paths']['dataset_table'],
              'log_dir': configs['paths']['log_dir']}

    for param in params.keys():
        params[param] = os.path.join(root, params[param])

    params.update({'image_size': tuple(map(int, configs['data_parameters']['image_size'].split(', '))),
                   'batch_size': int(configs['data_parameters']['batch_size'])})

    params.update({'num_features': int(configs['model_parameters']['DnCNN']['num_features']),
                   'num_layers': int(configs['model_parameters']['DnCNN']['num_layers'])})

    params.update({'fourier_layer': configs['model_parameters']['Fourier']['fourier_layer']})

    params.update({'lr': float(configs['train_parameters']['lr']),
                   'n_epochs': int(configs['train_parameters']['epochs'])})

    return params


def make_dataset_table(data_path, csv_file_path):
    image_names = sorted([name for name in os.listdir(os.path.join(data_path, 'Cropped images'))])
    data = []

    print('dataset csv table creating...')
    for image_name in tqdm(image_names):
        image_path = os.path.join(data_path, 'Cropped images', image_name)
        data.append(np.array([image_path]))

    pd.DataFrame(np.vstack(data), columns=['image']).to_csv(csv_file_path, index=False)


def train_val_split(csv_file_path, val_size=0.2):
    dataset = pd.read_csv(csv_file_path)

    test_number = int(len(dataset) * val_size) + 1
    train_number = len(dataset) - test_number
    phase = ['train'] * train_number + ['val'] * test_number
    random.Random(1).shuffle(phase)

    pd.concat([dataset[['image']],
               pd.DataFrame(phase, columns=['phase'])],
              axis=1).to_csv(csv_file_path, index=False)


def setup_experiment(title, params, log_dir):
    model_name = [title, 'fourier', params['fourier_layer']]
    # model_name.extend(['lr', str(params['lr'])])
    model_name = '_'.join(model_name)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
    best_model_path = f"{model_name}.best.pth"

    return writer, model_name, best_model_path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_transform(batch_transform=None):
    def collate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        if batch_transform is not None:
            collated = batch_transform(collated)
        return collated
    return collate


def run_epoch(model, iterator,
              criterion, optimizer,
              metrics,
              phase='train', epoch=0,
              device='cpu', writer=None):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}

    with torch.set_grad_enabled(is_train):
        for i, (images, masks) in enumerate(tqdm(iterator)):
            images, masks = images.to(device), masks.to(device)

            cleaned_images = model(images)

            loss = criterion(cleaned_images, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            for metric_name in epoch_metrics.keys():
                epoch_metrics[metric_name] += metrics[metric_name](cleaned_images.detach(), masks)

        epoch_loss = epoch_loss / len(iterator)
        for metric_name in epoch_metrics.keys():
            epoch_metrics[metric_name] = epoch_metrics[metric_name] / len(iterator)
        
        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss, epoch)
            for metric_name in epoch_metrics.keys():
                writer.add_scalar(f"metric_epoch/{metric_name}/{phase}", epoch_metrics[metric_name], epoch)

        return epoch_loss, epoch_metrics
    
    
def train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metrics,
          n_epochs,
          device,
          writer,
          best_model_path):
    best_val_loss = float('+inf')
    for epoch in range(n_epochs):
        train_loss, train_metrics = run_epoch(model, train_dataloader,
                                              criterion, optimizer,
                                              metrics,
                                              phase='train', epoch=epoch,
                                              device=device, writer=writer)
        val_loss, val_metrics = run_epoch(model, val_dataloader,
                                          criterion, None,
                                          metrics,
                                          phase='val', epoch=epoch,
                                          device=device, writer=writer)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch: {epoch + 1:02}')
        
        metrics_output = ' | '.join([metric_name + ': ' +
                                     "{:.2f}".format(train_metrics[metric_name]) for metric_name in train_metrics.keys()])
        print(f'\tTrain Loss: {train_loss:.2f} | Train Metrics: ' + metrics_output)

        metrics_output = ' | '.join([metric_name + ': ' +
                                     "{:.2f}".format(val_metrics[metric_name]) for metric_name in val_metrics.keys()])
        print(f'\t  Val Loss: {val_loss:.2f} |   Val Metrics: ' + metrics_output)


def main():
    root = get_root()
    params = get_params(root)
    train_data_path, dataset_table_path, log_dir = (params['train_data'],
                                                    params['dataset_table_path'],
                                                    params['log_dir'])
    image_size, batch_size = (params['image_size'],
                              params['batch_size'])
    lr, n_epochs = (params['lr'],
                    params['n_epochs'])

    make_dataset_table(train_data_path, dataset_table_path)
    train_val_split(dataset_table_path)
    dataset = pd.read_csv(dataset_table_path)

    pre_transforms = torchvision.transforms.Compose([Resize(size=image_size), ToTensor()])
    train_batch_transforms = BatchRicianNoise()
    # train_batch_transforms = BatchRandomErasing()

    train_dataset = DenoisingDataset(dataset=dataset[dataset['phase'] == 'train'],
                                     transform=pre_transforms)

    train_collate = collate_transform(train_batch_transforms)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=train_collate)

    val_batch_transforms = BatchRicianNoise()
    # val_batch_transforms = BatchRandomErasing()

    val_dataset = DenoisingDataset(dataset=dataset[dataset['phase'] == 'val'],
                                   transform=pre_transforms)

    val_collate = collate_transform(val_batch_transforms)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 collate_fn=val_collate)

    fourier_params = None
    if params['fourier_layer'] != 'None':
        fourier_params = {'fourier_layer': params['fourier_layer']}

    num_features, num_layers = params['num_features'], params['num_layers']
    model = DnCNN(n_channels=1,
                  num_features=num_features, num_layers=num_layers,
                  image_size=image_size, fourier_params=fourier_params).to(device)

    writer, model_name, best_model_path = setup_experiment(model.__class__.__name__, params, log_dir)
    best_model_path = os.path.join(root, best_model_path)
    print(f"Model name: {model_name}")
    print(f"Model has {count_parameters(model):,} trainable parameters")
    print()

    criterion = CombinedLoss([SIMMLoss(multiscale=True), L1Loss()], [0.8, 0.2])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metrics = {'PSNR': PSNR(), 'SSIM': SSIM(multiscale=False), 'FSIM': FSIM()}

    print("To see the learning process, use command in the new terminal:\ntensorboard --logdir <path to log directory>")
    print()
    train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metrics,
          n_epochs,
          device,
          writer,
          best_model_path)


if __name__ == "__main__":
    main()
