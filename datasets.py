import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
#changed
import numpy as np
from collections import defaultdict
import random
import tqdm

from timm.data import create_transform

from continual_datasets.continual_datasets import *
 
import utils

#changed
def mixup_same_class(img1, img2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * img1 + (1 - lam) * img2
    return mixed_img


def augment_dataset_same_class_mixup(dataset, num_augs=1, alpha=0.4):
    """
    Returns a dataset-like object with .targets and __getitem__ support.
    All mixup done within the same class.
    """
    class_data = defaultdict(list)

    for img, label in dataset:
        class_data[label].append(img)

    images = []
    labels = []

    for label, imgs in tqdm.tqdm(class_data.items()):
        for img in imgs:
            images.append(img)
            labels.append(label)
            for _ in range(num_augs):
                img2 = random.choice(imgs)
                mixed_img = mixup_same_class(img, img2, alpha)
                images.append(mixed_img)
                labels.append(label)

    # Stack images into a tensor only if they are Tensors
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
        labels = torch.tensor(labels)

        # Return a TensorDataset with .targets
        dataset = torch.utils.data.TensorDataset(images, labels)
        dataset.targets = labels
        return dataset
    else:
        # If still in PIL or other format
        dataset = list(zip(images, labels))
        dataset.targets = labels  # manually attach
        return dataset
    
class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.task_inc:
        mode = 'til'
    elif args.domain_inc:
        mode = 'dil'
    elif args.versatile_inc:
        mode = 'vil'
    elif args.joint_train:
        mode = 'joint'
    else:
        mode = 'cil'

    if mode in ['til', 'cil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )

                splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
                mask.append(class_mask)

                for i in range(len(splited_dataset)):
                    train.append(splited_dataset[i][0])
                    val.append(splited_dataset[i][1])

            splited_dataset = list()
            for i in range(args.num_tasks):
                t = [train[i+args.num_tasks*j] for j in range(len(dataset_list))]
                v = [val[i+args.num_tasks*j] for j in range(len(dataset_list))]
                splited_dataset.append((torch.utils.data.ConcatDataset(t), torch.utils.data.ConcatDataset(v)))

            args.nb_classes = 5
            class_mask = np.unique(np.array(mask), axis=0).tolist()[0]
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset, class_mask, domain_list = split_single_dataset(dataset_train, dataset_val, args)
            args.nb_classes = 5

    elif mode in ['dil', 'vil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            splited_dataset = list()

            for i in range(len(dataset_list)):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset_list[i],
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                splited_dataset.append((dataset_train, dataset_val))
            
            args.nb_classes = 5
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            #print(f"Dataset_Train : {dataset_train.data[1].classes} \n Dataset_val : {dataset_val.data[0][2]}")

            # dataset_train.data = [
            #     augment_dataset_same_class_mixup(domain, num_augs=2, alpha=0.4)
            #     for domain in tqdm.tqdm(dataset_train.data)
            # ]

            if args.dataset in ['CORe50']:
                splited_dataset = [(dataset_train[i], dataset_val) for i in range(len(dataset_train))]
                args.nb_classes = len(dataset_val.classes)
            else:
                splited_dataset = [(dataset_train, dataset_val) for i in range(len(dataset_train))]
                args.nb_classes = dataset_val.classes
    
    elif mode in ['joint']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                train.append(dataset_train)
                val.append(dataset_val)
                args.nb_classes = len(dataset_val.classes)

            dataset_train = torch.utils.data.ConcatDataset(train)
            dataset_val = torch.utils.data.ConcatDataset(val)
            splited_dataset = [(dataset_train, dataset_val)]

            class_mask = None
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset = [(dataset_train, dataset_val)]

            args.nb_classes = len(dataset_val.classes)
            class_mask = None
            
    else:
        raise ValueError(f'Invalid mode: {mode}')
                

    if args.versatile_inc:
        splited_dataset, class_mask, domain_list, args = build_vil_scenario(dataset_train,dataset_val, args)
        for c, d in zip(class_mask, domain_list):
            print(c, d)
    for i in range(len(splited_dataset)):
        dataset_train, dataset_val = splited_dataset[i]

        sampler_train = torch.utils.data.RandomSampler(dataset_train) 
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask, domain_list

def get_dataset(dataset, transform_train, transform_val, mode, args,):
    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'CORe50':
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)
    
    #changed
    elif dataset == 'OfficeHome':
        dataset_train = OfficeHome(args.data_path, train=True, transform=transform_train, mode=mode).data
        dataset_val = OfficeHome(args.data_path, train=False, transform=transform_val, mode=mode).data
    
    elif dataset == 'Dataset':
        dataset_train = Dataset(args.data_path, train=True, transform=transform_train)
        dataset_val = Dataset(args.data_path, train=False, transform=transform_val)

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    assert isinstance(dataset_train.data, list), "Expected dataset_train.data to be a list of domain datasets"
    assert isinstance(dataset_val.data, list), "Expected dataset_val.data to be a list of domain datasets"
    assert len(dataset_train.data) == len(dataset_val.data), "Mismatch in number of domains"

    # Define your custom task-to-(domain_id, class_ids) mapping
    custom_tasks = [
        (0, [0, 1, 2]),          # Task 0 (Base): Domain 0 - Initial exposure to Cardiomegaly, Effusion, Infiltration.
                                 # This is crucial as D0 supports all problematic classes.
    
        (3, [1, 2, 3]),          # Task 1 (Reinforce & New Domain): Domain 3 - Focus on reinforcing Effusion and Infiltration.
                                 # Introduce Nodule (Class 3) in a new domain, providing a fresh context for C1 & C2.
    
        (2, [0, 1, 4]),          # Task 2 (New Domain & Class): Domain 2 - Introduce Pneumothorax (Class 4),
                                 # and revisit Cardiomegaly (Class 0) and Effusion (Class 1) in this new domain.
                                 # This helps generalize C1 across domains.
    
        (1, [0, 4]),            # Task 3 (Domain Shift & Reinforce): Domain 1 - Further reinforce Cardiomegaly (Class 0)
                                 # and Pneumothorax (Class 4) in a domain with limited class overlap.
    
        (0, [0, 1, 2, 3, 4])    # Task 4 (Consolidation & Full Coverage): Domain 0 - Final comprehensive task with ALL
                                 # available classes in Domain 0. This consolidates learning and ensures the model
                                 # has seen all classes together within a single domain, which is crucial for
                                 # establishing full class relationships and for spectral regularization to work on
                                 # a complete feature space.
    ]


 
    split_datasets = []
    class_masks = []
    domain_list = []
    i=0

    for domain_id, class_ids in custom_tasks:
        domain_train = dataset_train.data[domain_id]
        domain_val = dataset_val.data[domain_id]

        # Filter indices that match the class_ids
        train_indices = [i for i, y in enumerate(domain_train.targets) if y in class_ids]
        val_indices   = [i for i, y in enumerate(domain_val.targets) if y in class_ids]

        # Create Subsets
        task_train_subset = Subset(domain_train, train_indices)
        task_val_subset = Subset(domain_val, val_indices)

        split_datasets.append([task_train_subset, task_val_subset])
        class_masks.append(class_ids)
        domain_list.append(domain_id)

        print(f"Task {i} : Domain {domain_id} Classes {class_ids} → "
              f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        i=i+1

    return split_datasets, class_masks,domain_list


def build_vil_scenario(dataset_train, dataset_val, args):
    split_datasets, class_masks,domain_list = split_single_dataset(dataset_train, dataset_val, args)

    args.num_tasks = len(split_datasets)

    # print(f"Splitted datasets : {split_datasets}, class_masks : {class_masks}, domain_list : {domain_list}")

    return split_datasets, class_masks, domain_list, args


def build_transform(is_train, args):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # slight rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),  # convert grayscale to 3 channels if needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),  # same here
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    return transform

