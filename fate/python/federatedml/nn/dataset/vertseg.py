import torch
from federatedml.nn.dataset.base import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np

#from federatedml.nn.dataset.segmentation.horizutils import CamVidDataset, get_transform

from federatedml.nn.dataset.segmentation.vertutils import SegmentationLabel, get_transform
from federatedml.nn.dataset.segmentation import image_transforms

from federatedml.util import LOGGER


class VertSegLbl(Dataset):
    
    """

    A basic Image Dataset built on pytorch ImageFolder, supports simple image transform
    Given a folder path, ImageDataset will load images from this folder, images in this
    folder need to be organized in a Torch-ImageFolder format, see
    https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html for details.

    Image name will be automatically taken as the sample id.

    Parameters
    ----------
    center_crop : bool, use center crop transformer
    center_crop_shape: tuple or list
    generate_id_from_file_name: bool, whether to take image name as sample id
    file_suffix: str, default is '.jpg', if generate_id_from_file_name is True, will remove this suffix from file name,
                 result will be the sample id
    return_label: bool, return label or not, this option is for host dataset, when running hetero-NN
    float64: bool, returned image tensors will be transformed to double precision
    label_dtype: str, long, float, or double, the dtype of return label
    """

    def __init__(self, center_crop=False,
                 generate_id_from_file_name=True, file_suffix='.png',
                 return_label=True, float64=False, label_dtype='long'):

        super(VertSegLbl, self).__init__()
        self.segmentations: SegmentationLabel = None
        self.return_label = return_label
        self.generate_id_from_file_name = generate_id_from_file_name
        self.file_suffix = file_suffix
        self.float64 = float64
        self.dtype = torch.float32 if not self.float64 else torch.float64
        avail_label_type = ['float', 'long', 'double']
        self.sample_ids = None
        assert label_dtype in avail_label_type, 'available label dtype : {}'.format(
            avail_label_type)
        if label_dtype == 'double':
            self.label_dtype = torch.float64
        elif label_dtype == 'long':
            self.label_dtype = torch.int64
        else:
            self.label_dtype = torch.float32

    def load(self, path):

        # read image from folders
        self.segmentations = SegmentationLabel(seg_dir=path, map_file=path+"/../"+"class_dict.csv")
        self.sample_ids = self.segmentations.files

    def __getitem__(self, item):
        #need to do this because FATE requires a bottom model even if there is no data on the federate
        return torch.zeros(1).type(self.dtype), self.segmentations.__getitem__(item)

    def __len__(self):
        return self.segmentations.__len__()

    def __repr__(self):
        return self.segmentations.__repr__()

    def get_classes(self):
        return self.segmentations.__len__()

    def get_sample_ids(self):
        return self.sample_ids


if __name__ == '__main__':
    pass