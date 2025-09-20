import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
import torchvision.transforms as T
from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler, ATTMAP_conditioning
from collections import OrderedDict
import os
import pydicom
import wandb
from PIL import Image
def at_spatial_l1_loss(feat_s: torch.Tensor, feat_t: torch.Tensor, p: int = 2, eps: float = 1e-6) -> torch.Tensor:
    if feat_s.shape[-2:] != feat_t.shape[-2:]:
        feat_t = F.interpolate(feat_t, size=feat_s.shape[-2:], mode='bilinear', align_corners=False)

    As = feat_s.abs().pow(p).sum(dim=1, keepdim=True)
    At = feat_t.abs().pow(p).sum(dim=1, keepdim=True)

    B = As.size(0)
    As = As.view(B, -1)
    At = At.view(B, -1)
    As = As / (As.norm(p=2, dim=1, keepdim=True) + eps)
    At = At / (At.norm(p=2, dim=1, keepdim=True) + eps)

    return F.l1_loss(As, At, reduction='mean')
class BrownianBridgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_config = config.model
        # model hyperparameters
        model_params = config.model.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.attmap_path = config.data.dataset_config.attmap_path
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.image_size = model_params.UNetStudentParams.image_size
        self.channels = model_params.UNetStudentParams.in_channels
        self.condition_key = model_params.UNetStudentParams.condition_key
        self.condition_key_teacher = model_params.UNetTeacherParams.condition_key
        self.teacher_ckpt = self.model_config.teacher_model_ckpt


        ckpt = torch.load(self.teacher_ckpt, map_location="cuda:0")
        state_dict = ckpt["model"]

        denoise_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("denoise_fn."):
                new_k = k[len("denoise_fn."):]
                denoise_state_dict[new_k] = v
        self.denoise_fn_teacher = UNetModel(**vars(model_params.UNetTeacherParams))
        self.denoise_fn_teacher.load_state_dict(denoise_state_dict)

        cond_stage_model_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("cond_stage_model."):
                new_k = k[len("cond_stage_model."):]
                cond_stage_model_state_dict[new_k] = v
        self.cond_stage_teacher_model = ATTMAP_conditioning()
        self.cond_stage_teacher_model.load_state_dict(cond_stage_model_state_dict)
        
        for p in self.denoise_fn_teacher.parameters():
            p.requires_grad = False
        for p in self.cond_stage_teacher_model.parameters():
            p.requires_grad = False

        self.denoise_fn = UNetModel(**vars(model_params.UNetStudentParams))
    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, x_name, y, context=None):

        attmap = self.get_attmap(x_name, y).squeeze(1)
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        
        with torch.no_grad():
            context_teacher = None
            attn_weight = None
            if self.condition_key_teacher == 'ATTMAP_conditioning':
                context_teacher, attn_weight = self.get_cond_stage_teacher_context(y, attmap)

        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, context_teacher, t)

    def p_losses(self, x0, y, context, context_teacher, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)


        x0_recon, hs1, hs2, out_stu = self.denoise_fn(x_t, y, timesteps=t, context=context)

        objective_recon = x_t - x0_recon

        with torch.no_grad():
            teacher_recon, hs1_teach, hs2_teach, out_teach = self.denoise_fn_teacher(x_t, y, timesteps=t, context=context_teacher)
            copied_hs1_list = [t.detach().clone() for t in hs1_teach]
            copied_hs2_list = [t.detach().clone() for t in hs2_teach]
            out_teach = out_teach.detach().clone()

        distill_loss_weight = 0.01
        feat_3_input_loss = at_spatial_l1_loss(hs1[3], copied_hs1_list[3])
        feat_4_input_loss = at_spatial_l1_loss(hs1[4], copied_hs1_list[4])


        feat_3_output_loss = at_spatial_l1_loss(hs2[3], copied_hs2_list[3])
        feat_4_output_loss = at_spatial_l1_loss(hs1[4], copied_hs1_list[4])


        feat_middle_loss = at_spatial_l1_loss(hs1[-1], copied_hs1_list[-1])

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean() + distill_loss_weight * (feat_3_input_loss + feat_3_output_loss + feat_middle_loss + feat_4_input_loss + feat_4_output_loss) + 0.01 * (out_stu - out_teach).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        # x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict, feat_3_input_loss, feat_3_output_loss, feat_middle_loss

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon, _, _, _ = self.denoise_fn(x_t, y, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon, _, _, _ = self.denoise_fn(x_t, y, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)

    @torch.no_grad()
    def get_cond_stage_teacher_context(self, x_cond_latent, attmap):
        context, attn_weights = self.cond_stage_teacher_model(x_cond_latent, attmap)
        return context, attn_weights
    def get_attmap(self, x_name, x_cond_latent):
        def find_attmap_file(folder_path, name):
            for filename in os.listdir(folder_path):
                if name in filename:
                    return os.path.join(folder_path, filename)
            raise FileNotFoundError("Not found 'ATTMAPREST")

        root_path = self.attmap_path
        conditions = []

        for i in range(x_cond_latent.shape[0]):
            patient_dir = os.path.join(root_path, self.reconstruct_path_before_rest_or_stress(x_name[i]+".png"))
            if 'REST' in x_name[i]:
                attmap_path = find_attmap_file(patient_dir, "ATTMAPREST")
            else:
                attmap_path = find_attmap_file(patient_dir, "ATTMAPSTRESS")

            ds = pydicom.dcmread(attmap_path)

            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                pixel_array = ds.pixel_array.astype(np.float32)  # shape: (z, y, x)
            else:
                raise ValueError("File DICOM don't have 3D (multi-frame).")

            tensor = torch.tensor(pixel_array, dtype=torch.float32)

            min_val = tensor.min()
            max_val = tensor.max()
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)    
                tensor = tensor * 2 - 1                              
            else:
                tensor = torch.zeros_like(tensor)  

            conditions.append(tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0))

        return torch.cat(conditions, dim=0).to(x_cond_latent.device)
    
    def reconstruct_path_before_rest_or_stress(self, filename):
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        cut_index = None
        for i, part in enumerate(parts):
            if part in ('REST', 'STRESS'):
                cut_index = i
                break

        if cut_index is None:
            raise ValueError("Not found 'REST' or 'STRESS'")

        path_before = '/'.join(parts[:cut_index])
        return path_before