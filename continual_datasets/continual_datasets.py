import os
import os.path

import pathlib
from pathlib import Path

from typing import Any, Tuple

import glob
from shutil import move, rmtree

import numpy as np

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive

import PIL
from PIL import Image

import tqdm
import zipfile
import tarfile

from .dataset_utils import read_image_file, read_label_file

class MNIST_RGB(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_RGB, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()
        self.classes = [i for i in range(10)]

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTM(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'MNIST-M')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.classes = [i for i in range(10)]

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

class SynDigit(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'SynDigit')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.classes = [i for i in range(10)]

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class CORe50(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mode='cil'):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train
        self.mode = mode

        self.url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
        self.filename = 'core50_128x128.zip'

        # self.fpath = os.path.join(root, 'VIL_CORe50')
        self.fpath = os.path.join(root, 'core50_128x128')
        
        if not os.path.isfile(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'core50_128x128')):
            with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        pass

        self.train_session_list = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']
        self.test_session_list = ['s3', 's7', 's10']
        self.label = [f'o{i}' for i in range(1, 51)]
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.split()

        if self.train:
            fpath = self.fpath + '/train'
            if self.mode not in ['cil', 'joint']:
                self.data = [datasets.ImageFolder(f'{fpath}/{s}', transform=transform) for s in self.train_session_list]
            else:
                self.data = datasets.ImageFolder(fpath, transform=transform)
        else:
            fpath = self.fpath + '/test'
            self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder): 
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        if self.mode not in ['cil', 'joint']:
            for s in tqdm.tqdm(self.train_session_list, desc='Preprocessing'):
                src = os.path.join(self.fpath, s)
                if os.path.exists(os.path.join(train_folder, s)):
                    continue
                move(src, train_folder)
            
            for s in tqdm.tqdm(self.test_session_list, desc='Preprocessing'):
                for l in self.label:
                    dst = os.path.join(test_folder, l)
                    if not os.path.exists(dst):
                        os.mkdir(os.path.join(test_folder, l))
                    
                    f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                    for src in f:
                        move(src, dst)
                rmtree(os.path.join(self.fpath, s))
        else:
            for s in tqdm.tqdm(self.train_session_list, desc='Preprocessing'):
                for l in self.label:
                    dst = os.path.join(train_folder, l)
                    if not os.path.exists(dst):
                        os.mkdir(os.path.join(train_folder, l))
                    
                    f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                    for src in f:
                        move(src, dst)
                rmtree(os.path.join(self.fpath, s))

            for s in tqdm.tqdm(self.test_session_list, desc='Preprocessing'):
                for l in self.label:
                    dst = os.path.join(test_folder, l)
                    if not os.path.exists(dst):
                        os.mkdir(os.path.join(test_folder, l))
                    
                    f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                    for src in f:
                        move(src, dst)
                rmtree(os.path.join(self.fpath, s))

class DomainNet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mode='cil'):
        root = os.path.join(root, 'VIL_DomainNet')   
        # root = os.path.join(root, 'DomainNet')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train
        self.mode = mode

        self.url = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip'
        ]

        self.filename = [
            'clipart.zip',
            'infograph.zip',
            'painting.zip',
            'quickdraw.zip',
            'real.zip',
            'sketch.zip'
        ]

        self.train_url_list = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_train.txt'
        ]

        for u in self.train_url_list:
            filename = u.split('/')[-1]
            if not os.path.isfile(os.path.join(self.root, filename)):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+filename)
                    download_url(u, root, filename=filename)
        
        self.test_url_list = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_test.txt'
        ]

        for u in self.test_url_list:
            filename = u.split('/')[-1]
            if not os.path.isfile(os.path.join(self.root, filename)):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+filename)
                    download_url(u, root, filename=filename)

        self.fpath = [os.path.join(self.root, f) for f in self.filename]

        for i in range(len(self.fpath)):
            if not os.path.isfile(self.fpath[i]):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+self.url[i])
                    download_url(self.url[i], root, filename=self.filename[i])

        if not os.path.exists(self.root + '/train') and not os.path.exists(self.root + '/test'):
            for i in range(len(self.fpath)):
                if not os.path.exists(os.path.join(self.root, self.filename[i][:-4])):
                    with zipfile.ZipFile(os.path.join(self.root, self.filename[i]), 'r') as zf:
                        for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename[i]}'):
                            try:
                                zf.extract(member, root)
                            except zipfile.error as e:
                                pass
            
            self.split()
        
        if self.train:
            fpath = self.root + '/train'
            if self.mode not in ['cil', 'joint']:
                self.data = [datasets.ImageFolder(f'{fpath}/{d}', transform=transform) for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']]
            else:
                self.data = datasets.ImageFolder(fpath, transform=transform)
        else:
            fpath = self.root + '/test'
            if self.mode not in ['cil', 'joint']:
                self.data = [datasets.ImageFolder(f'{fpath}/{d}', transform=transform) for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']]
            else:
                self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.root + '/train'
        test_folder = self.root + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        if self.mode not in ['cil', 'joint']:
            for i in tqdm.tqdm(range(len(self.train_url_list)), desc='Preprocessing'):
                train_list = self.train_url_list[i].split('/')[-1]
                
                with open(os.path.join(self.root, train_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[:2])
                        
                        if not os.path.exists(os.path.join(train_folder, dst)):
                            os.makedirs(os.path.join(train_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = os.path.join(train_folder, path)

                        move(src, dst)
            
            for i in tqdm.tqdm(range(len(self.test_url_list)), desc='Preprocessing'):
                test_list = self.test_url_list[i].split('/')[-1]

                with open(os.path.join(self.root, test_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[:2])

                        if not os.path.exists(os.path.join(test_folder, dst)):
                            os.makedirs(os.path.join(test_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = os.path.join(test_folder, path)

                        move(src, dst)
                rmtree(os.path.join(self.root, test_list.split('_')[0]))
        else:
            for i in tqdm.tqdm(range(len(self.train_url_list)), desc='Preprocessing'):
                train_list = self.train_url_list[i].split('/')[-1]
                
                with open(os.path.join(self.root, train_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[1:2])
                        
                        if not os.path.exists(os.path.join(train_folder, dst)):
                            os.makedirs(os.path.join(train_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = '/'.join(path.split('/')[1:])
                        dst = os.path.join(train_folder, dst)

                        move(src, dst)

            for i in tqdm.tqdm(range(len(self.test_url_list)), desc='Preprocessing'):
                test_list = self.test_url_list[i].split('/')[-1]

                with open(os.path.join(self.root, test_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[1:2])

                        if not os.path.exists(os.path.join(test_folder, dst)):
                            os.makedirs(os.path.join(test_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = '/'.join(path.split('/')[1:])
                        dst = os.path.join(test_folder, dst)

                        move(src, dst)
                rmtree(os.path.join(self.root, test_list.split('_')[0]))

class OfficeHome(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mode='cil'):
        root = os.path.join(root, 'VIL_OfficeHome')
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.mode = mode

        self.domains = ['Art', 'Clipart', 'Product', 'Real_World']
        self.urls = {
            'Art': 'https://www.hemanthdv.org/OfficeHomeDomainData/Art.zip',
            'Clipart': 'https://www.hemanthdv.org/OfficeHomeDomainData/Clipart.zip',
            'Product': 'https://www.hemanthdv.org/OfficeHomeDomainData/Product.zip',
            'Real_World': 'https://www.hemanthdv.org/OfficeHomeDomainData/Real_World.zip'
        }

        if download:
            for domain in self.domains:
                zip_path = os.path.join(self.root, f'{domain}.zip')
                if not os.path.exists(zip_path):
                    download_url(self.urls[domain], self.root, filename=f'{domain}.zip')

                extract_path = os.path.join(self.root, domain)
                if not os.path.exists(extract_path):
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(self.root)

        # Automatically create train/test folders if they don’t exist
        if not os.path.exists(os.path.join(self.root, 'train')) or not os.path.exists(os.path.join(self.root, 'test')):
            self.split()

        data_path = os.path.join(self.root, 'train' if train else 'test')
        if self.mode not in ['cil', 'joint']:
            self.data = [datasets.ImageFolder(os.path.join(data_path, d), transform=transform) for d in self.domains]
        else:
            self.data = datasets.ImageFolder(data_path, transform=transform)

    def split(self):
        from sklearn.model_selection import train_test_split
        for domain in self.domains:
            domain_path = os.path.join(self.root, domain)
            all_classes = os.listdir(domain_path)
            for cls in all_classes:
                full_path = os.path.join(domain_path, cls)
                if not os.path.isdir(full_path):
                    continue
                images = [os.path.join(full_path, img) for img in os.listdir(full_path)]
                train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

                for split, img_list in zip(['train', 'test'], [train_imgs, test_imgs]):
                    out_dir = os.path.join(self.root, split, domain, cls)
                    os.makedirs(out_dir, exist_ok=True)
                    for img in img_list:
                        move(img, os.path.join(out_dir, os.path.basename(img)))
    
    def __len__(self):
        if isinstance(self.data, datasets.ImageFolder):
            return len(self.data)
        return sum(len(d) for d in self.data)
    def __getitem__(self, index):
        # Flatten across domains
        domain_lengths = [len(d) for d in self.data]
        for i, l in enumerate(domain_lengths):
            if index < l:
                return self.data[i][index]
            index -= l


#CHANGED
class_name_to_index = {
    "Cardiomegaly": 0,
    "Effusion": 1,
    "Infiltration": 2,
    "Nodule": 3,
    "Pneumothorax": 4
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.mode = 'vil'
        self.classes = 5
        self.data_path = os.path.join(self.root, 'train' if self.train else 'test')

        if not os.path.exists(self.data_path):
            raise RuntimeError(f"{self.data_path} not found. Make sure train/test structure exists.")

        self.domains = sorted([d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))])

        def patched_imagefolder(path, transform=None):
            # Patch ImageFolder so that class_to_idx is always as per your mapping
            dataset = datasets.ImageFolder(path, transform=transform)
            # Force class_to_idx
            dataset.class_to_idx = class_name_to_index
            # Rebuild samples with new label mapping
            new_samples = []
            for path, _ in dataset.samples:
                class_name = os.path.basename(os.path.dirname(path))
                if class_name in class_name_to_index:
                    new_samples.append((path, class_name_to_index[class_name]))
            dataset.samples = new_samples
            dataset.targets = [label for _, label in new_samples]
            return dataset

        if self.mode in ['cil', 'joint']:
            # Combine all domains into one dataset (for CIL)
            self.data = patched_imagefolder(os.path.join(self.data_path, self.domains[0]), transform=self.transform)
            for domain in self.domains[1:]:
                domain_dataset = patched_imagefolder(os.path.join(self.data_path, domain))
                self.data.samples.extend(domain_dataset.samples)
                self.data.targets.extend(domain_dataset.targets)
        else:
            # Keep data domain-wise
            self.data = [patched_imagefolder(os.path.join(self.data_path, d), transform=self.transform) for d in self.domains]

    def __getitem__(self, index):
        if isinstance(self.data, list):  # domain-wise
            domain_lengths = [len(d) for d in self.data]
            for i, l in enumerate(domain_lengths):
                if index < l:
                    return self.data[i][index]
                index -= l
        else:  # combined for CIL
            return self.data[index]

    def __len__(self):
        if isinstance(self.data, list):
            return sum(len(d) for d in self.data)
        return len(self.data)
