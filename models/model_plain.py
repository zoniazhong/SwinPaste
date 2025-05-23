'''
# --------------------------------------------------------
# References:
# https://github.com/VisionICLab/SwinFuSR
# --------------------------------------------------------
'''
from collections import OrderedDict
from functools import wraps
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss_sr import custom_loss
import os

from torch.utils.tensorboard import SummaryWriter
from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.contrastive = self.opt_train['weights']["contrast"]!=0.0
        if self.contrastive:
            self.f1_head = 0
            self.f2_head = 0
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        # ------------------------------------
        # Define Tensorboard 
        # ------------------------------------
        tensorboard_path = os.path.join(self.opt['path']['root'], 'Tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log
        

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        self.netG.eval()
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
        # if load_path_optimizerG is not None:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)
            print('ok')

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'mixed':
            weights = self.opt_train['weights']
            dico_weight = {"mse":weights['mse'],"ssim":weights['ssim'],"psnr":weights['psnr'],"contrast":weights['contrast'],"lpips":weights['lpips'],"adversarial":weights['adversarial'], "l1":weights['l1']}
            self.G_lossfn = custom_loss(self.opt_train['batch_size'],dico_weight).to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed under/over data
    # ----------------------------------------
    def feed_data(self, data, need_GT=False, phase='test'):
        self.lr = data['Lr'].to(self.device)
        self.guided = data['Guide'].to(self.device)
        if need_GT:
            self.GT = data['Hr'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, phase='test'):
        if self.contrastive:
            if phase == 'train':
                self.output,self.f1_head,self.f2_head = self.netG(self.lr, self.guided)
            else:   
                self.output,_,_ = self.netG(self.lr, self.guided)
        else:
            self.output = self.netG(self.lr, self.guided)
            
        # with torch.no_grad():
        #     self.E_contrast = self.netG(self.A_contrast.detach(), self.B_contrast.detach()).detach()
        # print(self.A.shape,self.A_contrast.shape,self.B_contrast.shape)
        

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step,phase):
        self.G_optimizer.zero_grad()
        self.netG_forward(phase)
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type=="mixed":
            if self.contrastive:
                G_loss = self.G_lossfn(self.output, self.GT,self.f1_head,self.f2_head)
            else:
                G_loss =self.G_lossfn(self.output, self.GT)
        ## constructe loss function
        elif G_lossfn_type in ['loe', 'gt']:
            loe_loss, loss_tv, loss_grad, loss_l1 = self.G_lossfn(self.A, self.B, self.E, self.GT)   
            G_loss = self.G_lossfn_weight * loe_loss   
        elif G_lossfn_type in ['mef', 'mff', 'vif', 'nir', 'med']:
            total_loss, loss_text, loss_int, loss_ssim = self.G_lossfn(self.A, self.B, self.E)
            G_loss = self.G_lossfn_weight * total_loss      
        else:
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.output, self.GT)
        G_loss.backward()
        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()
        if G_lossfn_type in ['loe', 'mef', 'vif', 'mff', 'gt', 'nir', 'med']:
            self.log_dict['Text_loss'] = loss_text.item()
            self.log_dict['Int_loss'] = loss_int.item()
            self.log_dict['SSIM_loss'] = loss_ssim.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])
        # ----------------------------------------
        # write tensorboard
        # ----------------------------------------
        if G_lossfn_type == 'mef':
            self.writer.add_image('under_image[0]', self.A[0])
            self.writer.add_image('under_image[1]', self.A[-1])
            self.writer.add_image('over_image[0]', self.B[0])
            self.writer.add_image('over_image[1]', self.B[-1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])
        elif G_lossfn_type == 'vif':
            self.writer.add_image('ir_image[0]', self.A[0])
            self.writer.add_image('ir_image[1]', self.A[1])
            self.writer.add_image('vi_image[0]', self.B[0])
            self.writer.add_image('vi_image[1]', self.B[1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])
        elif G_lossfn_type == 'mff':
            self.writer.add_image('near_image[0]', self.A[0])
            self.writer.add_image('near_image[1]', self.A[-1])
            self.writer.add_image('far_image[0]', self.B[0])
            self.writer.add_image('far_image[1]', self.B[-1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])
        elif G_lossfn_type == 'nir':
            self.writer.add_image('Nir_image[0]', self.A[0])
            self.writer.add_image('Nir_image[1]', self.A[-1])
            self.writer.add_image('RGB_image[0]', self.B[0])
            self.writer.add_image('RGB_image[1]', self.B[-1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])
        elif G_lossfn_type == 'med':
            self.writer.add_image('pet_image[0]', self.A[0])
            self.writer.add_image('pet_image[1]', self.A[-1])
            self.writer.add_image('MRI_image[0]', self.B[0])
            self.writer.add_image('MRI_image[1]', self.B[-1])
            self.writer.add_image('fused_image[0]', self.E[0])
            self.writer.add_image('fused_image[1]', self.E[-1])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward(phase='test')
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=False):
        out_dict = OrderedDict()
        out_dict['Lr'] = self.lr.detach()[0].float().cpu()
        out_dict['Guided'] = self.guided.detach()[0].float().cpu()
        out_dict['Output'] = self.output.detach()[0].float().cpu()
        if need_H:
            out_dict['GT'] = self.GT.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['A'] = self.A.detach().float().cpu()
        out_dict['BL'] = self.B.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['GT'] = self.GT.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg