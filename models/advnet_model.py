import torch
from .pix2pix_model import Pix2PixModel

class AdvNetModel(Pix2PixModel):
    """ This class implements the AdvNet model derived from Pix2PixModel.
    Hong Wing PANG, 10/1/2019
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        Pix2PixModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(netG='unet_128', netD='n_layers', dataset_mode='advnet', no_flip=True)
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        Pix2PixModel.__init__(self, opt)
        self.visual_names = ['real_A1', 'real_A2', 'fake_B', 'real_B']

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        Pix2PixModel.set_input(self, input)
        self.real_A1 = self.real_A[:, :3].to(self.device)
        self.real_A2 = self.real_A[:, 3:].to(self.device)
