import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class ADE20KDataset(BaseDataset):
    """Child class for ADE20K dataset.
    Assumes A -> ground truth, B -> semantic labels.
    Default direction is B to A.
    
    Hong Wing PANG, 13/1/2019
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        first_char = opt.dataset_class[0]
        self.dir = os.path.join(opt.dataroot, 'images', opt.phase, first_char, opt.dataset_class)
        self.paths = sorted(make_dataset(self.dir, float("inf")))

        self.A_paths = [p for p in self.paths if p[-4:] == '.jpg']
        self.B_paths = [p for p in self.paths if p[-8:] == '_seg.png']
        
        assert len(self.A_paths) == len(self.B_paths)
        if opt.max_dataset_size != float("inf"):
            self.A_paths = self.A_paths[:opt.max_dataset_size]
            self.B_paths = self.B_paths[:opt.max_dataset_size]
        self.size = len(self.A_paths)
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
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
        parser.set_defaults(direction='BtoA')
        parser.add_argument('--dataset_class', type=str, required=True, help='Specify ADE20K class used for training')
        return parser
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        assert index < self.size
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size