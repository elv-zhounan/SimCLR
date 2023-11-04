import os
# Master address for distributed data parallel
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "7777"
# set avaible gpu cards
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3, 4'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from loguru import logger
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook


def init_dist(args):
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    args.rank = args.node_rank * args.gpus + args.gpu
    dist.init_process_group(
        init_method='env://', # 
        backend='nccl',
        world_size=args.world_size, 
        rank=args.rank)
    dist.barrier()
    print(f'rank:{dist.get_rank()}')
    print(f'world_size:{dist.get_world_size()}')
    print('process group initialized')

def prepare(args):
    init_dist(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()


        loss = loss.data.clone().detach()
        dist.barrier()
        dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.rank == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.rank == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    args.gpu = gpu
    prepare(args)

    train_dataset = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        download=False,
        transform=TransformsSimCLR(size=args.image_size),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    model = SimCLR(encoder, args.projection_dim, n_features)
    model = model.cuda()

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # DDP / DP
    # if args.dataparallel:
    #     # model = convert_model(model) #something abotut BN I personally did not do any BN sync thigs before 
    #     model = DataParallel(model)
    # else:
        # syncBN 其实只需要在world size > 1的时候用
        # 不过话说回来, world size == 1, 谁也不会用ddp啊

        # BN层需要对所有数据进行，但是分布式训练将数据分别给到了每个进程
        # 所以模型中的BN只是对各自进程中的数据进行BN处理，
        # 如果每个进程的batch_size较大，影响不大，如果batch_size很小，那么BN层作用就很小，这时希望对所有数据进行BN处理。

        # 实际使用时根据实际情况决定
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[gpu])

    writer = None
    if args.rank == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.rank == 0 and scheduler:
            scheduler.step()

        if args.rank == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.rank == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            logger.info(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t reported from gpu:{gpu}"
            )
            args.current_epoch += 1

    ## end training
    if args.rank == 0:
        save_model(args, model, optimizer)


def demo_main(gpu, args):
    args.gpu = gpu
    prepare(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1, help='number of all nodes in distributed training')
    parser.add_argument('--gpus', type=int, default=4, help='number of gpu per node')
    parser.add_argument('--node_rank', type=int, default=0, help='node rank in all nodes')
    parser.add_argument('--test_init', action="store_true", help='test init process group related things')
    parser.add_argument('--download_dataset', action="store_true", help='download_dataset')
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    assert args.gpus == torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    
    if args.test_init:
        mp.spawn(demo_main, args=(args,), nprocs=args.gpus)

    elif args.download_dataset:
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    else:
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)

    

