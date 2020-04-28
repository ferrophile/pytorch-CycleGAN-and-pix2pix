import torch
from .cycle_gan_model import CycleGANModel

class AdvNetCycleGANModel(CycleGANModel):
    """ This class implements the AdvNet model derived from CycleGAN model.
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
        CycleGANModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='advnet', no_flip=True)
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        opt.input2_nc = opt.output_nc; # Background channels
        opt.netG_A = 'alpha_resnet_9blocks'
        opt.netG_B = 'resnet_9blocks'
        CycleGANModel.__init__(self, opt)

        visual_names_A = ['real_A1', 'real_A2', 'real_A3', 'fake_B', 'rec_A2']
        visual_names_B = ['real_B', 'fake_A2', 'rec_B']
        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        CycleGANModel.set_input(self, input)
        self.real_A1 = self.real_A[:, :3].to(self.device)
        self.real_A2 = self.real_A[:, 3:6].to(self.device)
        self.real_A3 = self.real_A[:, 6:].to(self.device)
        binary_mask, _ = torch.max(self.real_A1, 1, keepdim=True)
        self.real_A3 = (binary_mask <= 0).float() * self.real_A2 + (binary_mask > 0).float() * self.real_A3

        self.empty = -1 * torch.ones_like(self.real_A1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A2 = self.netG_B(self.fake_B)

        self.fake_A2 = self.netG_B(self.real_B)
        self.fake_A = torch.cat((self.real_A1, self.fake_A2, self.real_A3), 1)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A2 = self.fake_A_pool.query(self.fake_A2)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A2, fake_A2)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(torch.cat((self.empty, self.real_B, self.real_B), 1))
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A2)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A2) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A2), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A2, self.real_A2) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
