import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import reduce
import numpy as np

from models.sdf_model import SdfModel
from models.autoencoder import BetaVAE
from models.diffusion import DiffusionModel
from models.archs.diffusion_arch import DiffusionNet

from typing import Dict, Optional, Union, List

class CombinedModel(pl.LightningModule):
    def __init__(self, specs: Dict):
        super().__init__()
        self.specs = specs

        self.task = specs['training_task'] # 'combined' or 'modulation' or 'diffusion'

        self._rr_anns_set: bool = False
        self._vae_latents: Dict[str, Dict[str, np.ndarray]] = {}    # top level str for category, bottom level str for sample name
        self._vae_mu     : Dict[str, Dict[str, np.ndarray]] = {}    # top level str for category, bottom level str for sample name

        self.sdf_model: Optional[SdfModel] = None
        self.diffusion_model: Optional[DiffusionModel] = None

        if self.task in ('combined', 'modulation'):
            self.sdf_model = SdfModel(specs=specs) 

            feature_dim = specs["SdfModelSpecs"]["latent_dim"] # latent dim of pointnet 
            modulation_dim = feature_dim*3 # latent dim of modulation
            latent_std = specs.get("latent_std", 0.25) # std of target gaussian distribution of latent space
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            self.vae_model = BetaVAE(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

        if self.task in ('combined', 'diffusion'):
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"])


    def training_step(self, x: dict[str, torch.Tensor], idx):
        """
            x is a dictionary with keys 'xyz', 'gt_sdf', 'point_cloud', all being torch.Tensor
        """

        self.train()

        if self.task == 'combined':
            assert x['point_cloud'].shape[1] > 0, f"Point cloud has no points, shape is {x['point_cloud'].shape}"
            return self.train_combined(x)
        elif self.task == 'modulation':
            assert x['point_cloud'].shape[1] > 0, f"Point cloud has no points, shape is {x['point_cloud'].shape}"
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        

    def configure_optimizers(self):

        if self.task == 'combined':
            params_list = [
                    { 'params': list(self.sdf_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['sdf_lr'] },
                    { 'params': self.diffusion_model.parameters(), 'lr':self.specs['diff_lr'] }
                ]
        elif self.task == 'modulation':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['sdf_lr'] }
                ]
        elif self.task == 'diffusion':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['diff_lr'] }
                ]

        optimizer = torch.optim.Adam(params_list)
        return {
                "optimizer": optimizer,
        }

    def train_modulation(self, x: dict[str, Union[torch.Tensor, str]]):

        # xyz: 3D coordinates of the points where signed distances are calculated from
        # gt_sdf: signed distances as calculated from xyz
        # point_cloud: point cloud (sampled on the surface)

        xyz : torch.Tensor = x['xyz'] # (B, N, 3)
        gt  : torch.Tensor = x['gt_sdf'] # (B, N)
        pc  : torch.Tensor = x['point_cloud'] # (B, PCsize, 3) where PCsize is set in specs.json
        class_names   : List[str] = x['class_name']      # length = B
        instance_names: List[str] = x['instance_name']   # length = B

        # STEP 1: obtain reconstructed plane feature and latent code 
        original_features: torch.Tensor = self.sdf_model.pointnet.get_plane_features(pc)
        out: List[torch.Tensor] = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature = out[0]    # (B, 768, 64, 64)
        mu = out[2]                             # (B, 768)
        latent = out[-1]                        # (B, 768)

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)

        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        loss_dict =  {
            f"training/{self.task}/sdf": sdf_loss,
            f"training/{self.task}/vae": vae_loss,
            f"training/{self.task}/total": loss,    # tracked in ModelCheckpoint
        }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        # book-keep VAE outputs
        latent_np: np.ndarray = latent.detach().cpu().numpy()
        mu_np: np.ndarray = mu.detach().cpu().numpy()
        sample_latent: np.ndarray
        sample_mu: np.ndarray
        for (cl_name, sample_name, sample_latent, sample_mu) in zip(class_names, instance_names, latent_np, mu_np):
            if cl_name not in self._vae_latents.keys():
                self._vae_latents[cl_name] = {}
            self._vae_latents[cl_name][sample_name] = sample_latent
            if cl_name not in self._vae_mu.keys():
                self._vae_mu[cl_name] = {}
            self._vae_mu[cl_name][sample_name] = sample_mu

        return loss


    def train_diffusion(self, x: dict[str, torch.Tensor]):

        latent: torch.Tensor = x['latent'] # (B, D)
        # class_names: List[str] = x['class_name']
        # instance_names: List[str] = x['instance_name']
        partial_pcd: torch.Tensor = x['partial_point_cloud']
        # partial_rgb_fp: List[str] = x['partial_rgb_fp']

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, _ = self.diffusion_model.diffusion_model_from_latent(latent, partial_pcds=partial_pcd)

        loss_dict =  {
                        f"training/{self.task}/total": diff_loss,    # tracked in ModelCheckpoint
                        f"training/{self.task}/diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        f"training/{self.task}/diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    # the first half is the same as "train_sdf_modulation"
    # the reconstructed latent is used as input to the diffusion model, rather than loading latents from the dataloader as in "train_diffusion"
    def train_combined(self, x: dict[str, torch.Tensor]):
        # xyz: 3D coordinates of the points where signed distances are calculated from
        # gt_sdf: signed distances as calculated from xyz
        # point_cloud: point cloud (sampled on the surface)
        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, PCsize, 3) where PCsize is set in specs.json
        # class_names   : List[str] = x['class_name']      # length = B
        # instance_names: List[str] = x['instance_name']   # length = B
        partial_pcd: torch.Tensor = x['partial_point_cloud']

        # STEP 1: obtain reconstructed plane feature for SDF and latent code for diffusion
        original_features: torch.Tensor = self.sdf_model.pointnet.get_plane_features(pc)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1] # [B, D*3, resolution, resolution], [B, D*3]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF 
        vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        # STEP 4: use latent as input to diffusion model
        diff_loss, _, _, pred_latent = self.diffusion_model.diffusion_model_from_latent(latent, partial_pcds=partial_pcd)
        
        # STEP 5: use predicted / reconstructed latent to run SDF loss 
        generated_plane_feature = self.vae_model.decode(pred_latent)
        generated_sdf_pred = self.sdf_model.forward_with_plane_features(generated_plane_feature, xyz)
        generated_sdf_loss = F.l1_loss(generated_sdf_pred.squeeze(), gt.squeeze())

        loss = sdf_loss + vae_loss + diff_loss + generated_sdf_loss

        loss_dict =  {
                        f"training/{self.task}/total": loss,    # tracked in ModelCheckpoint
                        f"training/{self.task}/sdf": sdf_loss,
                        f"training/{self.task}/vae": vae_loss,
                        f"training/{self.task}/diff": diff_loss,
                        f"training/{self.task}/gensdf": generated_sdf_loss,
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss

    def training_epoch_end(self, outputs) -> None:
        self._vae_latents = {}
        self._vae_mu = {}

    def validation_step(self, x: dict[str, torch.Tensor], idx):
        """
        Mirrors training_step()
        """

        with torch.no_grad():
            if self.task == 'combined':
                assert x['point_cloud'].shape[1] > 0, f"Point cloud has no points, shape is {x['point_cloud'].shape}"
                return self.validate_combined(x)
            elif self.task == 'modulation':
                assert x['point_cloud'].shape[1] > 0, f"Point cloud has no points, shape is {x['point_cloud'].shape}"
                return self.validate_modulation(x)
            elif self.task == 'diffusion':
                return self.validate_diffusion(x)


    def validate_modulation(self, x: dict[str, Union[torch.Tensor, str]]):
        xyz : torch.Tensor = x['xyz'] # (B, N, 3)
        gt  : torch.Tensor = x['gt_sdf'] # (B, N)
        pc  : torch.Tensor = x['point_cloud'] # (B, PCsize, 3) where PCsize is set in specs.json
        # class_names   : List[str] = x['class_name']      # length = B
        # instance_names: List[str] = x['instance_name']   # length = B

        original_features: torch.Tensor = self.sdf_model.pointnet.get_plane_features(pc)
        out = self.vae_model(original_features)
        reconstructed_plane_feature, _ = out[0], out[-1]

        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)

        vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"])
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        loss_dict =  {
            f"val/{self.task}/sdf": sdf_loss,
            f"val/{self.task}/vae": vae_loss,
            f"val/{self.task}/total": loss,    # tracked in ModelCheckpoint
        }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss


    def validate_diffusion(self, x: dict[str, torch.Tensor]):
        latent: torch.Tensor = x['latent'] # (B, D)
        # class_names: List[str] = x['class_name']
        # instance_names: List[str] = x['instance_name']
        partial_pcd: torch.Tensor = x['partial_point_cloud']
        # partial_rgb_fp: List[str] = x['partial_rgb_fp']

        diff_loss, diff_100_loss, diff_1000_loss, _ = self.diffusion_model.diffusion_model_from_latent(latent, partial_pcds=partial_pcd)

        loss_dict =  {
                        f"val/{self.task}/total": diff_loss,    # tracked in ModelCheckpoint
                        f"val/{self.task}/diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        f"val/{self.task}/diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss


    def validate_combined(self, x: dict[str, torch.Tensor]):
        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, PCsize, 3) where PCsize is set in specs.json
        # class_names   : List[str] = x['class_name']      # length = B
        # instance_names: List[str] = x['instance_name']   # length = B
        partial_pcd: torch.Tensor = x['partial_point_cloud']

        original_features: torch.Tensor = self.sdf_model.pointnet.get_plane_features(pc)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1] # [B, D*3, resolution, resolution], [B, D*3]

        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"])
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        diff_loss, _, _, pred_latent = self.diffusion_model.diffusion_model_from_latent(latent, partial_pcds=partial_pcd)

        generated_plane_feature = self.vae_model.decode(pred_latent)
        generated_sdf_pred = self.sdf_model.forward_with_plane_features(generated_plane_feature, xyz)
        generated_sdf_loss = F.l1_loss(generated_sdf_pred.squeeze(), gt.squeeze())

        loss = sdf_loss + vae_loss + diff_loss + generated_sdf_loss

        loss_dict =  {
                        f"val/{self.task}/total": loss,    # tracked in ModelCheckpoint
                        f"val/{self.task}/sdf": sdf_loss,
                        f"val/{self.task}/vae": vae_loss,
                        f"val/{self.task}/diff": diff_loss,
                        f"val/{self.task}/gensdf": generated_sdf_loss,
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss
