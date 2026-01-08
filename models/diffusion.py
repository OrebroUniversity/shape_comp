import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple

from diff_utils.helpers import linear_beta_schedule, cosine_beta_schedule, default, extract

from tqdm.auto import tqdm
import open3d as o3d

from typing import Optional
import os


# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


class DiffusionModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000, sampling_timesteps: int = 50, beta_schedule: str = 'cosine',
        sample_pc_size = 682, perturb_pc = None,  crop_percent=0.25,
        loss_type = 'l2', objective = 'pred_x0', 
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()

        self.model = model
        self.objective = objective

        betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc 
        self.crop_percent = crop_percent
        assert self.perturb_pc in [None, "partial", "noisy"]

        self.loss_fn = F.l1_loss if loss_type=='l1' else F.mse_loss

        assert sampling_timesteps <= timesteps, f"sampling_timesteps {sampling_timesteps} cannot be larger than timesteps {timesteps}"
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        print("DiffusionModel:")
        print(f" - objective: {objective}")
        print(f" - beta_schedule: {beta_schedule}")
        print(f" - loss_type: {loss_type}")
        print(f" - ddim_sampling_eta: {ddim_sampling_eta}")
        print(f" - num_timesteps: {self.num_timesteps}")
        print(f" - sampling_timesteps: {self.sampling_timesteps}")

        # helper function to register buffer from float64 to float32
        def register_buffer(name: str, val: torch.Tensor): return self.register_buffer(name, val.to(torch.float32))
        

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def ddim_sample(
        self,
        dim: int,
        batch_size: int,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        traj: bool = False,
        cond: Optional[torch.Tensor] = None,
    ):

        times = torch.linspace(0., self.num_timesteps, steps = self.sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        trajectory = []

        x_T = default(noise, torch.randn(batch_size, dim, device = self.betas.device)) 

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch_size,), time, device = self.betas.device, dtype = torch.long)

            model_input = (x_T, cond) if cond is not None else x_T
            pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(x_T) if time_next > 0 else 0.

            x_T = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            trajectory.append(x_T.clone())

        if traj:
            return x_T, trajectory
        else:
            return x_T

    @torch.no_grad()
    def ddpm_sample(
        self,
        dim: int,
        batch_size: int,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        traj: bool = False,
        cond: Optional[torch.Tensor] = None,
    ):

        trajectory = []

        x_T = default(noise, torch.randn(batch_size, dim, device = self.betas.device))

        for t in reversed(range(0, self.num_timesteps)):
            time_cond = torch.full((batch_size,), t, device = self.betas.device, dtype = torch.long)

            model_input = (x_T, cond) if cond is not None else x_T
            pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)
            if clip_denoised:
                x_start.clamp_(-1., 1.)

            model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)

            noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0

            x_T = model_mean + (0.5 * model_log_variance).exp() * noise

            trajectory.append(x_T.clone())

        if traj:
            return x_T, trajectory
        else:
            return x_T

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        #noise = torch.clamp(noise, min=-6.0, max=6.0)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # main function for calculating loss
    def forward(
            self,
            x_start: torch.Tensor,  # latent
            t: torch.Tensor,
            cond: torch.Tensor,     # partial PCD
            ret_pred_x: bool = False,
            noise: Optional[torch.Tensor] = None,
        ):
        '''
        x_start: [B, D]
        t: [B]
        '''

        noise = default(noise, lambda: torch.randn_like(x_start)) 

        # x_start : clean latent
        # x : noisy latent
        x = self.q_sample(x_start=x_start, t=t, noise=noise) 

        model_in = (x, cond)
        model_out = self.model(data=model_in, diffusion_timesteps=t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        #loss = reduce(loss, 'b ... -> b (...)', 'mean', b = x_start.shape[0]) # only one dim of latent so don't need this line 
        
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        unreduced_loss = loss.detach().clone().mean(dim=1)
        
        if ret_pred_x:
            return loss.mean(), x, target, model_out, unreduced_loss
        else:
            return loss.mean(), unreduced_loss

    def model_predictions(self, model_input, t):

        # https://github.com/princeton-computational-imaging/Diffusion-SDF/issues/27#issuecomment-1962225908
        model_output = self.model(model_input, t)

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    # a wrapper function that only takes x_start (clean modulation vector) and condition
    # does everything including sampling timestep and returns loss, loss_100, loss_1000, prediction
    # x_start: VAE latent
    def diffusion_model_from_latent(self, x_start: torch.Tensor, partial_pcds: torch.Tensor):
        # STEP 1: sample timestep 
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        # STEP 2: pass to forward function
        loss, x, target, model_out, unreduced_loss = self(x_start, t, cond=partial_pcds, ret_pred_x=True)
        loss_100 = unreduced_loss[t<100].mean().detach()
        loss_1000 = unreduced_loss[t>100].mean().detach()

        return loss, loss_100, loss_1000, model_out


    def generate_from_pc(self, pc: torch.Tensor, batch: int = 5, save_pc_path: Optional[str] = None):
        if save_pc_path:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy().squeeze())
            o3d.io.write_point_cloud(os.path.join(save_pc_path, "input_pc.ply"), pcd, write_ascii=True)

        self.eval()
        with torch.no_grad():
            input_pc = pc.repeat(batch, 1, 1)
            samp = self.ddim_sample(dim=self.model.dim_in_out, batch_size=batch, traj=False, cond=input_pc)

        return samp
