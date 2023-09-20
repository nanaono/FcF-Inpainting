from tabnanny import filename_only
import numpy as np
import cv2
import os
import PIL
import torch
from .dataset import Dataset

class ImageDataset(Dataset):
    
    def __init__(self,
        img_path,                   # Path to images.
        synth_path,
        mask_path,   
        resolution      = None,     # Ensure specific resolution, None = highest available.
        **super_kwargs,             # Additional arguments for the Dataset base class.
    ):
        self.sz = resolution
        self.img_path = img_path
        self.synth_path = synth_path
        self.mask_path = mask_path
        self._type = 'dir'
        self.files = []
        self.idx = 0

        self._all_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in os.walk(self.img_path) for fname in files]
        self._all_synth_fnames = [os.path.relpath(os.path.join(root, fname), start=self.synth_path) for root, _dirs, files in os.walk(self.synth_path) for fname in files]
        self._all_mask_fnames = [os.path.relpath(os.path.join(root, fname), start=self.mask_path) for root, _dirs, files in os.walk(self.mask_path) for fname in files]

        PIL.Image.init()
        self._image_fnames = sorted(os.path.join(self.img_path,fname) for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        self._synth_fnames = sorted(os.path.join(self.synth_path,fname) for fname in self._all_synth_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        self._mask_fnames = sorted(os.path.join(self.mask_path,fname) for fname in self._all_mask_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self.files = []
        self.synth_files = []
        self.mask_files = []
        
        for f in self._image_fnames:
            if not '_mask' in f:
                self.files.append(f)
        
        for f in self._synth_fnames:
            if not '_mask' in f:
                self.synth_files.append(f)

        for f in self._mask_fnames:
            if not '_mask' in f:
                self.mask_files.append(f)
        
        self.files = sorted(self.files)
        self.synth_files = sorted(self.synth_files)
        self.mask_files = sorted(self.mask_files)

    def __len__(self):
        return len(self.files)
    
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_image(self, fn):
        return PIL.Image.open(fn).convert('RGB')
    
    def _get_image(self, idx):
        # imgfn, seg_map, img_id = self.data_reader.get_image(idx)
        
        fname = self.files[idx]
        synth_fname = self.synth_files[idx]
        mask_fname = self.mask_files[idx]

        ext = self._file_ext(fname)

        mask = np.array(self._load_image(mask_fname).convert('L')) / 255
        mask = 1 - mask # 0: foreground, 1: background
        synth = np.array(self._load_image(synth_fname)) # uint8
        rgb = np.array(self._load_image(fname)) # uint8

        return rgb, synth, fname.split('/')[-1].replace(ext, ''), mask
        
    def __getitem__(self, idx):
        rgb, synth, fname, mask = self._get_image(idx) # modal, uint8 {0, 1}
        rgb = rgb.transpose(2,0,1)
        synth = synth.transpose(2,0,1)

        mask_tensor = torch.from_numpy(mask).to(torch.float32)
        mask_tensor = mask_tensor.unsqueeze(0)
        synth = torch.from_numpy(synth.astype(np.float32))
        synth = (synth.to(torch.float32) / 127.5 - 1)
        
        rgb = torch.from_numpy(rgb.astype(np.float32))
        rgb = (rgb.to(torch.float32) / 127.5 - 1)
        
        synth_erased = synth.clone()
        synth_erased = synth_erased # * (1 - mask_tensor) # erase the background
        synth_erased = synth_erased.to(torch.float32)

        return rgb, synth_erased, mask_tensor, fname
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of images.
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list
            - image: torch tensor of shape (3, 256, 256).
            
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        
    """

    rgbs, rgbs_erased, mask_tensors, fnames = zip(*data)
    
    rgbs = list(rgbs)
    rgbs_erased = list(rgbs_erased)
    mask_tensors = list(mask_tensors)
    fnames = list(fnames)

    return torch.stack(rgbs, dim=0), torch.stack(rgbs_erased, dim=0), torch.stack(mask_tensors, dim=0), fnames
    
def get_loader(img_path, synth_path, mask_path, resolution):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    ds = ImageDataset(img_path=img_path, synth_path=synth_path, mask_path=mask_path, resolution=resolution)

    data_loader = torch.utils.data.DataLoader(dataset=ds, 
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader