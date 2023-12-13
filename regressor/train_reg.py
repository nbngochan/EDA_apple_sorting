import argparse
import os
from pprint import pprint
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning_fabric.utilities.seed import seed_everything
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision.datasets import ImageFolder
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# solver settings
OPT = 'adam'  # adam, sgd
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9  # only when OPT is sgd
BASE_LR = 0.1
LR_SCHEDULER = 'step'  # step, multistep, reduce_on_plateau
LR_DECAY_RATE = 0.1
LR_STEP_SIZE = 10  # only when LR_SCHEDULER is step
LR_STEP_MILESTONES = [10, 15]  # only when LR_SCHEDULER is multistep


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train regressor.')
    parser.add_argument(
        '--dataset', '-d', type=str, required=False, help='Root directory of dataset'
    )
    parser.add_argument(
        '--outdir', '-o', type=str, default='results', help='Output directory'
    )
    parser.add_argument(
        '--model-name', '-m', type=str, default='resnet18', help='Model name (timm)'
    )
    parser.add_argument(
        '--img-size', '-i', type=int, default=112, help='Input size of image'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=100, help='Number of training epochs'
    )
    parser.add_argument(
        '--save-interval', '-s', type=int, default=10, help='Save interval (epoch)'
    )
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument(
        '--num-workers', '-w', type=int, default=12, help='Number of workers'
    )
    parser.add_argument(
        '--use-image-folder', '-u', action='store_true', help='Use ImageFolder dataset'
    )
    parser.add_argument(
        '--csv-train', type=str, default=None, help='Csv training file'
    )
    parser.add_argument(
        '--csv-val', type=str, default=None, help='Csv validation file'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--gpu-ids', type=int, default=None, nargs='+', help='GPU IDs to use'
    )
    group.add_argument('--n-gpu', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=44, help='Seed')
    args = parser.parse_args()
    return args


def get_optimizer(parameters) -> torch.optim.Optimizer:
    if OPT == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    elif OPT == 'sgd':
        optimizer = torch.optim.SGD(
            parameters, lr=BASE_LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM
        )
    else:
        raise NotImplementedError()

    return optimizer


def get_lr_scheduler_config(optimizer: torch.optim.Optimizer) -> dict:
    if LR_SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif LR_SCHEDULER == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_STEP_MILESTONES, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif LR_SCHEDULER == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val/mse',
            'interval': 'epoch',
            'frequency': 1,
        }
    else:
        raise NotImplementedError

    return lr_scheduler_config


class ImageTransform:
    def __init__(self, is_train: bool, img_size: int | tuple = 224):
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class DatasetFromDataframe(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(self.df.image_path[index]).convert('RGB')
        weight = self.df.weight[index]

        if self.transform:
            image = self.transform(image)
        return image, weight


class SimpleData(LightningDataModule):
    def __init__(
            self,
            root_dir: str,
            img_size: int | tuple = 224,
            batch_size: int = 8,
            num_workers: int = 16,
            use_image_folder: bool = True,
            csv_file_train: str = None,
            csv_file_val: str = None,

    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_image_folder = use_image_folder
        self.csv_file_train = csv_file_train
        self.csv_file_val = csv_file_val


        if not self.use_image_folder:  # Check if use_image_folder is False
            self.train_dataset = DatasetFromDataframe(
                csv_file=self.csv_file_train,
                transform=ImageTransform(is_train=True, img_size=self.img_size)
            )
            self.val_dataset = DatasetFromDataframe(
                csv_file=self.csv_file_val,
                transform=ImageTransform(is_train=False, img_size=self.img_size)
            )
            
        else:  # use_image_folder is True
            self.train_dataset = ImageFolder(
                root=os.path.join(root_dir, 'train'),
                transform=ImageTransform(is_train=True, img_size=self.img_size),
            )
            self.val_dataset = ImageFolder(
                root=os.path.join(root_dir, 'val'),
                transform=ImageTransform(is_train=False, img_size=self.img_size),
            )
            

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return dataloader


class RegressorModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'resnet18',
            pretrained: bool = True,
    ):
        super().__init__()
        self.preds = []
        self.targets = []

        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=0
        )
        
        
        if model_name == 'resnet18':
            last_layer = self.model.layer4[-1]
            last_conv = last_layer.conv2
            in_features = last_conv.out_channels
            
        elif model_name == 'mobilenetv3_small_050.lamb_in1k':
            last_conv = self.model.conv_head
            in_features = last_conv.out_channels
       
        elif model_name == 'efficientnet_b0.ra_in1k':
            last_conv = self.model.conv_head
            in_features = last_conv.out_channels
        
        elif model_name == 'efficientnet_b3.ra2_in1k':
            last_conv = self.model.conv_head
            in_features = last_conv.out_channels
            
        elif model_name == 'regnetx_002.pycls_in1k':
            last_conv = self.model.s4.b7.conv3.conv
            in_features = last_conv.out_channels
        
        elif model_name == 'regnety_002.pycls_in1k':
            last_conv = self.model.s4.b7.conv3.conv
            in_features = last_conv.out_channels
        
        elif model_name == 'fbnetc_100.rmsp_in1k':
            last_conv = self.model.conv_head
            in_features = last_conv.out_channels
        
        
        # Change output layer for regression
        self.regression_head = nn.Linear(in_features, 1)
        
        self.train_loss = nn.MSELoss()
        self.val_loss = nn.MSELoss()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()


    def forward(self, x):
        features = self.model(x)
        
        return self.regression_head(features)


    def training_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        out = out.squeeze(-1)
        
        loss = self.train_loss(out, target.float())
        mse = self.mse(out, target)
        mae = self.mae(out, target)
        self.log_dict({'train/loss': loss, 'train/mse': mse, 'train/mae': mae}, prog_bar=True, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        out = out.squeeze(-1)
        
        loss = self.val_loss(out, target.float())
        mse = self.mse(out, target)
        mae = self.mae(out, target)
        
        self.log_dict({'val/loss': loss, 'val/mse': mse, 'val/mae': mae}, prog_bar=True, on_epoch=True)
        self.preds.append(out)
        self.targets.append(target)


    def on_validation_epoch_end(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        loss = self.val_loss(preds, targets)
        
        self.log('val/mse_loss', loss, on_epoch=True, prog_bar=True)

        self.preds.clear()
        self.targets.clear()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def get_basic_callbacks() -> list:
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(
        filename='epoch{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=1,
    )
    last_ckpt_callback = ModelCheckpoint(
        filename='last_model_{epoch:03d}-{val/mse:.2f}-{val/mae:02.0f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=None,
    )
    best_ckpt_calllback = ModelCheckpoint(
        filename='best_model_{epoch:03d}-{val/mse:.2f}-{val/mae:.2f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        verbose=True
    )
    early_stopping_callback = EarlyStopping(
        monitor='val/loss',  # Metric to monitor for improvement
        mode='min',  # Choose 'min' or 'max' depending on the metric (e.g., 'min' for loss, 'max' for accuracy)
        patience=10,  # Number of epochs with no improvement before stopping
    )
    return [last_ckpt_callback, best_ckpt_calllback, lr_callback, early_stopping_callback]




def get_gpu_settings(
        gpu_ids: list[int], n_gpu: int) -> tuple[str, int | list[int] | None, str | None]:
    """Get gpu settings for pytorch-lightning trainer:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags

    Args:
        gpu_ids (list[int])
        n_gpu (int)

    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    """
    if not torch.cuda.is_available():
        return "cpu", None, None

    if gpu_ids is not None:
        devices = gpu_ids
        strategy = "ddp" if len(gpu_ids) > 1 else None
    elif n_gpu is not None:
        # int
        devices = n_gpu
        strategy = "ddp" if n_gpu > 1 else None
    else:
        devices = 1
        strategy = 'auto'

    return "gpu", devices, strategy


def get_trainer(args: argparse.Namespace) -> Trainer:
    callbacks = get_basic_callbacks()
    accelerator, devices, strategy = get_gpu_settings(args.gpu_ids, args.n_gpu)

    logs_dir = args.outdir
    tb_logger = TensorBoardLogger(os.path.join(logs_dir, 'tb_logs'))

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=logs_dir,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=[tb_logger],
        deterministic=True,
    )
    return trainer


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed, workers=True)

    data = SimpleData(
        root_dir=args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_image_folder=args.use_image_folder,
        csv_file_train = args.csv_train,
        csv_file_val=args.csv_val,
    )
    model = RegressorModel(model_name=args.model_name, pretrained=True)

    trainer = get_trainer(args)
    print(data.img_size)
    print('Args:')
    pprint(args.__dict__)
    trainer.fit(model, data)
    
# Example of training command:
# python train_reg.py --img-size 512 --batch-size 16 --num-workers 4 --epochs 100 --model-name mobilenetv3_small_050.lamb_in1k --csv-train ./train_reg.csv --csv-val ./assets/test_reg.csv