from torchvision import transforms as T
import torch

from .dataset import Polyp,RandomCrop_Polyp,ToTensor
from data.generate_data_loader import generate_data_loader

def prepare_fed_polyp(args):
    sites = args.source
    ood_site = args.target if args.target is not None else '2'
    client_num = len(sites)
    train_transform = T.Compose([RandomCrop_Polyp((384,384)), ToTensor()])
    test_transform=ToTensor()
    trainsets = []
    valsets = []
    testsets = []
    train_len=[]
    ood_set = torch.utils.data.ConcatDataset([Polyp(client_idx=ood_site, split='train', transform=train_transform),
                                            Polyp(client_idx=ood_site, split='val', transform=train_transform),
                                            Polyp(client_idx=ood_site, split='test', transform=train_transform)])
    for site in sites: 
        trainset = Polyp(client_idx=site, split='train', transform=train_transform)
        valset = Polyp(client_idx=site, split='val', transform=test_transform)
        testset = Polyp(client_idx=site, split='test', transform=test_transform)
        print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
        trainsets.append(trainset)
        valsets.append(valset)
        testsets.append(testset)
        train_len.append(len(trainset)) 

    dataset = generate_data_loader(args, client_num, trainsets, valsets, testsets, ood_set)

    return dataset,train_len
