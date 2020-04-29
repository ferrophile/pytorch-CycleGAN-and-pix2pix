import torch
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class AdvNetDataset(BaseDataset):
    """
    Child class for AdvNet datasets. Note the following terms:
    - box:          Bounding box with background
    - image:        Ground truth image with object
    - mask:         Object mask without background
    - object:       Object without background
    - scene:        Ground truth background only
    - structure:    Object mask with background
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        objects = ['cola', 'sprite', 'milk', 'tea']
        if self.opt.direction == 'AtoB':
            self.classes = {
                'A1': 'object',
                'A2': 'scene',
                'B': 'image'
            }
        else:
            raise ValueError('Not implemented for AdvNet')

        self.paths = {}
        for c in self.classes:
            self.paths[c] = []
            for obj in objects:
                class_path = '{}_{}'.format(obj, self.classes[c])
                dir = os.path.join(opt.dataroot, opt.phase, obj, class_path)

                obj_limit = opt.max_dataset_size // len(objects)
                self.paths[c] += sorted(make_dataset(dir))[:obj_limit]

        # validate dataset size
        self.size = len(self.paths['A1'])
        for c in self.classes:
            assert self.size == len(self.paths[c])

        self.transform = get_transform(self.opt, grayscale=False)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # Settings for AdvNet
        parser.set_defaults(input_nc=6)
        return parser

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        assert index < self.size
        imgs = {}

        for c in self.classes:
            path = self.paths[c][index]
            imgs[c] = Image.open(path).convert('RGB')

        transform_params = get_params(self.opt, imgs['B'].size)
        transform = get_transform(self.opt, transform_params, grayscale=False)

        # apply transformation
        for c in self.classes:
            imgs[c] = transform(imgs[c])

        return {
            'A': torch.cat((imgs['A1'], imgs['A2']), 0),
            'B': imgs['B'],
            'A_paths': self.paths['A1'][index], # probably not used
            'B_paths': self.paths['B'][index]
        }

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.size
