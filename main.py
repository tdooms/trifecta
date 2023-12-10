import torch

from dataclasses import dataclass, field
from typing import Optional
from torch.nn.functional import softplus
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer, LightningDataModule
from torchmetrics import Accuracy
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10
from torchvision.transforms import *


class Data(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = {} if config.test else dict(persistent_workers=True, num_workers=4, pin_memory=True)
        self.train = self.test = self.val = None
        self.batch_size = config.batch_size
        self.name = config.dataset

        mean, std, dim, self.dataset, self.split = dict(
            mnist=((0.1307,), (0.3081,), 28, MNIST, [50_000, 10_000]),
            f_mnist=((0.2860,), (0.3530,), 28, FashionMNIST, [50_000, 10_000]),
            svhn=((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970), 32, SVHN, [63_257, 10_000]),
            cifar10=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 32, CIFAR10, [45_000, 5_000]),
        )[config.dataset]

        self.train_transform = Compose([
            RandomCrop(dim, padding=4),
            RandomRotation(15),
            ToTensor(),
            Normalize(mean, std),
        ])

        self.test_transform = Compose([
            ToTensor(),
            Normalize(mean, std),
        ])

    def setup(self, stage):
        # uncomment/comment below to use validation instead of test

        # full = self.dataset(root="../data", split="train") if self.name == 'svhn' \
        #     else self.dataset(root="../data", train=True)

        # self.train, self.val = random_split(full, self.split)
        #
        # self.train.dataset.transform = self.train_transform
        # self.val.dataset.transform = self.test_transform

        self.train = self.dataset(root="../data", split="train", transform=self.train_transform) if self.name == 'svhn' \
            else self.dataset(root="../data", train=True, transform=self.train_transform)

        self.test = self.dataset(root="../data", split="test", transform=self.test_transform) if self.name == 'svhn' \
            else self.dataset(root="../data", train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True, **self.config)

    def val_dataloader(self):
        # return DataLoader(self.val, batch_size=2 * self.batch_size, shuffle=False, **self.config)
        return DataLoader(self.test, batch_size=2 * self.batch_size, shuffle=False, **self.config)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=2 * self.batch_size, shuffle=False, **self.config)


###############################################################################


def large(params, block, dims, prev, channels):
    b0 = block(channels, 64, 0, prev, dims, **params)
    b1 = block(64, 128, 1, b0, dims, **params, maxpool=True)
    b2 = block(128, 128, 2, b1, dims // 2, **params)
    b3 = block(128, 128, 3, b2, dims // 2, **params)
    b4 = block(128, 128, 4, b3, dims // 2, **params)
    b5 = block(128, 256, 5, b4, dims // 2, **params, maxpool=True)
    b6 = block(256, 256, 6, b5, dims // 4, **params)
    b7 = block(256, 256, 7, b6, dims // 4, **params)
    b8 = block(256, 256, 8, b7, dims // 4, **params)
    b9 = block(256, 512, 9, b8, dims // 4, **params, maxpool=True)
    b10 = block(512, 512, 10, b9, dims // 8, **params)
    b11 = block(512, 512, 11, b10, dims // 8, **params)

    return nn.ModuleList([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11])


def small(params, block, dims, prev, channels):
    b0 = block(channels, 128, 0, prev, dims, **params, maxpool=True)
    b1 = block(128, 128, 1, b0, dims // 2, **params)
    b2 = block(128, 256, 2, b1, dims // 2, **params, maxpool=True)
    b3 = block(256, 256, 3, b2, dims // 4, **params)
    b4 = block(256, 256, 4, b3, dims // 4, **params)
    b5 = block(256, 512, 5, b4, dims // 4, **params, maxpool=True)

    return nn.ModuleList([b0, b1, b2, b3, b4, b5])


###############################################################################


def should_detach(olu, index, iteration):
    return (index % 2 == 0) ^ (iteration % 2 == 0) if olu else True


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, index, prev, dims, lr, scale, alpha, tau, olu, bn, maxpool=False):
        super().__init__()

        in_channels *= scale if in_channels != 4 else 1
        out_channels *= scale

        self.inner = nn.Sequential(
            nn.BatchNorm2d(in_channels) if bn else nn.LayerNorm([in_channels, dims, dims]),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) if maxpool else nn.Identity(),
        )

        params = list(prev.parameters()) if prev else list()
        self.optimizer = Adam(params + list(self.inner.parameters()), lr=lr)

        self.index = index
        self.alpha = alpha
        self.tau = tau
        self.olu = olu

    def forward(self, x, iteration):
        out = self.inner(x)
        activations = out.pow(2).flatten(start_dim=1)

        if not self.training:
            return out, activations.mean(1), None, None, None, None

        elif should_detach(self.olu, self.index, iteration):
            pos, neg = activations.chunk(2)
            pos, neg = pos.mean(1), neg.mean(1)

            delta = pos - neg

            if self.alpha is None:
                loss = torch.cat([softplus(self.tau - pos), softplus(neg - self.tau)]).mean()
            else:
                loss = softplus(-self.alpha * delta).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return out.detach(), activations.mean(1), delta.mean(0), loss.item(), pos.mean(0), neg.mean(0)

        else:
            return out, None, None, None, None, None


class Network(nn.Module):
    def __init__(self, config, num_classes, dims):
        super().__init__()

        self.embedding = nn.Embedding(num_classes, dims * dims)

        params = dict(lr=config.lr, scale=config.scale, alpha=config.alpha, bn=config.bn, olu=config.olu, tau=config.tau)
        channels = 2 if config.dataset in ["mnist", "f_mnist"] else 4

        embedding = self.embedding if config.learned else None

        self.layers = dict(small=small, large=large)[config.model](params, Block, dims, embedding, channels)
        self.num_classes, self.dims = num_classes, dims

    def forward(self, x, y, iteration):
        embedding = self.embedding(y).view(-1, 1, self.dims,  self.dims)
        x = torch.cat([x, embedding], dim=1)

        metrics = []
        for layer in self.layers:
            x, *rest = layer(x, iteration)
            metrics.append(rest)

        if self.training:
            _, delta, loss, pos, neg = zip(*metrics)
            return delta, loss, pos, neg
        else:
            goodness, _, _, _, _ = zip(*metrics)
            return torch.stack(goodness)


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False

        datasets = dict(mnist=(10, 28), f_mnist=(10, 28), svhn=(10, 32), cifar10=(10, 32))
        self.num_classes, self.dims = datasets[config.dataset]

        self.net = Network(config, self.num_classes, self.dims)
        self.num_layers = len(self.net.layers)

        self.ensemble = config.ensemble
        self.accuracy = Accuracy("multiclass", num_classes=self.num_classes)

    def forward(self, x, y, iteration=None):
        return self.net(x, y, iteration)

    def training_step(self, batch, iteration):
        x, y_pos = batch

        y_random = torch.randint_like(y_pos, 0, self.num_classes - 1).cuda()
        y_same = torch.eq(y_random, y_pos)
        y_neg = ((self.num_classes - 1) * y_same) + (~y_same * y_random)

        delta, loss, pos, neg = self(torch.cat([x, x]), torch.cat([y_pos, y_neg]), iteration)

        losses = {f"train/loss_{i}": l for i, l in enumerate(loss) if l is not None}
        deltas = {f"train/delta_{i}": a for i, a in enumerate(delta) if a is not None}
        pos = {f"train/pos_{i}": a for i, a in enumerate(pos) if a is not None}
        neg = {f"train/neg_{i}": a for i, a in enumerate(neg) if a is not None}

        metrics = {**losses, **deltas, **pos, **neg}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def evaluate(self, x, y, kind):
        per_label = [self(x, torch.full_like(y, label)) for label in range(self.num_classes)]
        per_label = torch.stack(per_label)

        per_layer = [self.accuracy(per_label[:, layer].argmax(0), y) for layer in range(self.num_layers)]
        per_layer = torch.stack(per_layer)

        y_hat = per_label[:, self.ensemble].mean(1).argmax(0)
        accuracy = self.accuracy(y_hat, y)
        accuracies = {f"{kind}/accuracy_{i}": a for i, a in enumerate(per_layer)}
        metrics = {**accuracies, f"{kind}/accuracy": accuracy}

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, _):
        self.evaluate(*batch, "val")

    def test_step(self, batch, _):
        self.evaluate(*batch, "test")

    def configure_optimizers(self):
        return None


@dataclass
class Config:
    lr: float = 1e-3
    scale: int = 1                # scales the channels by this factor
    model: str = 'small'          # can be small or large
    learned: bool = True          # indicates whether the embedding is learned or not

    alpha: Optional[float] = 4.0  # the scale parameter for SymBa, use None for the original loss
    tau: Optional[float] = 2.0    # the threshold parameter for the original loss

    olu: bool = True              # True or False for olu or layer-wise respectively
    bn: bool = True               # True or False for batch-norm and layer-norm respectively

    ensemble: slice = field(default_factory=lambda: slice(-1, None))  # determines the layers used for evaluation
    batch_size: int = 512

    test: bool = False
    epochs: int = 100
    dataset: str = 'cifar10'      # can be mnist, f_mnist, svhn, or cifar10

    finetune: Optional[str] = None


def run(config):
    torch.set_float32_matmul_precision('medium')
    data = Data(config)

    if config.finetune is None:
        model = Model(config)
    else:
        model = Model.load_from_checkpoint(f"models/{config.finetune}.ckpt", config=config)

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config.epochs,
        fast_dev_run=config.test,
        benchmark=True,
        check_val_every_n_epoch=5,
        precision="16-mixed",
        enable_checkpointing=True,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    run(Config(model='small', dataset='mnist'))
