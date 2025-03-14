﻿import os
import numpy as np
import PIL.Image
import json
import torch
import dnnlib
import dnnlib
import cv2
from icecream import ic
from . import mask_generator
import os.path as osp
import matplotlib.pyplot as plt
from icecream import ic
import matplotlib.cm as cm
import copy
import albumentations as A
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageDataset(Dataset):
    
    def __init__(self,
        img_path,                   # Path to images.
        mask_path,
        synth_path,
        resolution      = None,     # Ensure specific resolution, None = highest available.
        **super_kwargs,             # Additional arguments for the Dataset base class.
    ):
        self.sz = resolution
        self.img_path = img_path
        self.mask_path = mask_path
        self.synth_path = synth_path
        self._type = 'dir'

        self._all_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in os.walk(self.img_path) for fname in files]
        self._all_mask_fnames = [os.path.relpath(os.path.join(root, fname), start=self.mask_path) for root, _dirs, files in os.walk(self.mask_path) for fname in files]
        self._all_synthesis_fnames = [os.path.relpath(os.path.join(root, fname), start=self.synth_path) for root, _dirs, files in os.walk(self.synth_path) for fname in files]

        PIL.Image.init()

        self._image_fnames = sorted(os.path.join(self.img_path,fname) for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        self._mask_fnames = sorted(os.path.join(self.mask_path, fname) for fname in self._all_mask_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        self._synthesis_fnames = sorted(os.path.join(self.synth_path, fname) for fname in self._all_synthesis_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        if len(self._mask_fnames) == 0:
            raise IOError("No mask files found in the specified path")
        if len(self._synthesis_fnames) == 0:
            raise IOError("No synthesis files found in the specified path")
        
        self.img_files = []
        self.mask_files = []
        self.synthesis_files = []
        
        for f in self._image_fnames:
            if not '_mask' in f:
                self.img_files.append(f)

        for f in self._mask_fnames:
            self.mask_files.append(f)
        
        for f in self._synthesis_fnames:
            self.synthesis_files.append(f)

        self.img_files = sorted(self.img_files)
        self.mask_files = sorted(self.mask_files)
        self.synthesis_files = sorted(self.synthesis_files)

        self.transform = A.Compose([
            A.PadIfNeeded(min_height=self.sz, min_width=self.sz),
            A.OpticalDistortion(),
            A.RandomCrop(height=self.sz, width=self.sz),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.ToFloat()
        ], additional_targets={"synth": "image", "mask": "image"})  

        name = os.path.splitext(os.path.basename(self.img_path))[0]
        raw_shape = [len(self.img_files)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __len__(self):
        return len(self.img_files)

    def _load_image(self, fn):
        return PIL.Image.open(fn).convert('RGB')
    
    def _load_mask_image(self, fn):
        return PIL.Image.open(fn).convert('L')
    
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_image(self, raw_idx):
        fname = self.img_files[raw_idx]
        image = np.array(PIL.Image.open(fname).convert('RGB'))
        image = self.transform(image=image)['image']
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image
    
    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _get_image(self, idx):
        fname = self.img_files[idx]
        mask = mask_generator.generate_random_mask(s=self.sz, hole_range=[0.1,0.7])

        rgb = np.array(self._load_image(fname)) # uint8
        rgb = self.transform(image=rgb)['image']
        rgb = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)
        
        return rgb, mask

    def _get_image_with_mask(self, idx):
        fname = self.img_files[idx]
        mask_fname = self.mask_files[idx]
        sysnthesis_fname = self.synthesis_files[idx]

        if ((os.path.basename(fname) != os.path.basename(mask_fname))
            or (os.path.basename(fname) != os.path.basename(sysnthesis_fname))):
            raise ValueError("Image and mask file names do not match")

        rgb = np.array(self._load_image(fname))
        synth = np.array(self._load_image(sysnthesis_fname))
        mask = np.array(self._load_mask_image(mask_fname))

        transformed = self.transform(image=rgb, synth=synth, mask=mask)
        rgb = transformed['image']
        synth = transformed['synth']
        mask = transformed['mask']

        rgb = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)
        synth = np.rint(synth * 255).clip(0, 255).astype(np.uint8)

        # make mask to (0, 1)
        mask = np.where(mask > 0.5, 1.0, 0.0)
        mask = 1 - mask # 0: foreground, 1: background

        # make mask to (1, H, W)
        mask = mask[np.newaxis, :, :]

        return synth, rgb, mask
        
    def __getitem__(self, idx):
        synth, rgb, mask = self._get_image_with_mask(idx) # modal, uint8 {0, 1}
        rgb = rgb.transpose(2,0,1)
        synth = synth.transpose(2,0,1)

        return synth, rgb, mask, super().get_label(idx)
