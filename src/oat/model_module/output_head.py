import torch
import torch.nn as nn
from typing import Dict, List, Literal
from oat.model_module.distributions import _ALL_DISTRIBUTIONS
from oat.model_module.distributions import DistributionMap, OATDistribution
import einops

class OutputHead(nn.Module):
    
    '''
    OutputHead is a distributional output head that can handle multiple variables with different distributions.
    It takes a dictionary mapping variable names to distribution names, which must be among the supported 
    distributions in supported_distributions(). Each variable is modeled with a separate output head corresponding to its distribution.
    The loss can be scaled individually for each variable using the loss_scaling tensor, which should have the same length as the number of variables
    and follow the same order as the variable names in the variable_to_distribution dictionary.
    If one of the distributions does not have a closed form CRPS, crps_mc_samples is then the number of samples used to 
    estimate the CRPS value.
    observation_assimilation_as is a flag used to control the kind of assimilation used. If "innovation", the innovation is computed
    in a distribution-specific manner. If "target", the target observations are used and the fusion layer is responsible to 
    learn an update from them.
    
    To get the predicted distributions as objects, call forward() with the input features. 
    To get samples from the predicted distributions, call sample().
    '''

    
    def __init__(self,
                 in_features: int,
                 variable_to_distribution: Dict[str, str],
                 loss_scaling: torch.Tensor | None = None,
                 crps_mc_samples: int = 50,
                 observation_assimilation_as: Literal["innovation", "target"] = "innovation"
                 ):
        super().__init__()
        
        
        assert all(dist_name in self.supported_distributions() for dist_name in variable_to_distribution.values()), \
            f"Unsupported distributions: {set(variable_to_distribution.values()) - set(self.supported_distributions())}"

        self.distribution_mappers = nn.ModuleDict({
           var_name: DistributionMap(dist_class=self.name_to_dist_cls()[dist_name],
                           in_features=in_features)
           for var_name, dist_name in variable_to_distribution.items()
        })
        
        self.variable_to_distribution = dict(variable_to_distribution) # make a copy to avoid side effects
        self.target_features = len(variable_to_distribution)
        
        if loss_scaling is None:
            loss_scaling = torch.ones(len(variable_to_distribution))
        assert loss_scaling.shape == (len(variable_to_distribution),), "loss_scaling must have the same length as the number of variables"
        
        self.loss_scaling = loss_scaling
        self.crps_mc_samples = crps_mc_samples
        assert observation_assimilation_as in ["innovation", "target"]
        self.observation_assimilation_as = observation_assimilation_as

    @classmethod
    def supported_distributions(cls):
        return list(cls.name_to_dist_cls().keys())
    
    @classmethod
    def name_to_dist_cls(cls):
        return {
            dist_cls.name: dist_cls
            for dist_cls in _ALL_DISTRIBUTIONS
            if dist_cls.name is not None
        }
    
    
    def forward(self, x: torch.Tensor, as_dict: bool = True) -> Dict[str, OATDistribution] | List[OATDistribution]:
        if as_dict:
            return {
                var_name: self.distribution_mappers[var_name](x)
                for var_name in self.distribution_mappers.keys()
            }

        return [
            self.distribution_mappers[var_name](x)
            for var_name in self.distribution_mappers.keys()
        ]
        
    
    def _reshape_samples(self, samples: torch.Tensor) -> torch.Tensor:
        
        if samples.ndim == 2:  # [sample, station]
            samples = samples.unsqueeze(1)  # [sample, batch=1, station]
        
        return einops.rearrange(samples, 'sample batch station -> batch station sample 1')
        
    def sample(self, x: torch.Tensor, num_samples: int, differentiable: bool, pred_dist: Dict[str, OATDistribution] = None, as_dict: bool = False) -> Dict[str, torch.Tensor] | torch.Tensor:
        '''
        Return samples from the predicted distribututions given input features x. 
        If as_dict is True, returns a dictionary mapping each variable to samples of shape [batch, station, num_samples, 1].
        If as_dict is False, returns samples of shape [batch, station, num_samples, len(variable_to_distribution)].
        If differentiable is True, will try to use the reparameterized sampling method (rsample) 
        if the distribution supports it, otherwise will raise an error. 
        If the output distributions were already computed, those can be passed as pred_dist and x set to None.
        '''
        
        # TODO for now the only "covariance" we can model is given by the shared input features (not really a valid model).
        # In the future, we might need a better approach.
        
        if pred_dist is None:
            distributions = self.forward(x, as_dict=True)
        else:
            assert x is None
            distributions = pred_dist
        
        output = {} if as_dict else []

        for var_name, dist in distributions.items():
            if differentiable and not dist.has_rsample:
                raise ValueError(f"Distribution {dist.__class__.__name__} does not support reparameterized sampling, cannot sample differentiably.")
            
            samples = dist.rsample((num_samples,)) if differentiable else dist.sample((num_samples,))
            if as_dict:
                output[var_name] = self._reshape_samples(samples)
            else:
                output.append(self._reshape_samples(samples))
                
        if as_dict:
            return output
        return torch.cat(output, dim=-1)
    
    def loss_normalization_per_sample(self, loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute normalization factor for loss per sample based on mask.
        
        Args:
            loss: loss tensor [B, S, target_features]
            mask: boolean mask indicating valid entries [B, S, target_features]
            
        Returns:
            Normalization factor per sample [B]
        """
        loss_per_sample = torch.sum(loss * mask, dim=[-2, -1])  # [B]
        valid_counts = torch.sum(mask, dim=[-2, -1])  # [B]
        normalization = torch.clamp(valid_counts, min=1.0)  # Avoid division by zero
        norm_loss = torch.mean(loss_per_sample / normalization)  # [B]
        return norm_loss
    
    def loss(self, x, target, mask):
        '''
        Given an input of shape [batch, station, in_features], a target of shape [batch, station, num_vars] 
        and a mask of the same shape as the target, computes the scalar loss by summing the scalar losses of each variable
        which are individually scaled by self.loss_scaling.
        '''
        
        overall_loss = torch.tensor(0.0, device=x.device)
        
        batch_dim, station_dim, num_vars = target.shape
        
        assert num_vars == len(self.variable_to_distribution)

        out = self.forward(x, as_dict=True)
        for i, (var_name, dist) in enumerate(out.items()):
            
            # mask=mask[..., i:i+1]
            if dist.event_shape == ():
                target_i = target[..., i]
            else:
                raise NotImplementedError() # if event shape is (1,) one must use target[..., i:i+1] (to be checked)

            # crps_mc_samples only used if the distribution does not have a closed form CRPS, otherwise it is ignored.            
            loss_values = dist.crps(target=target_i, num_samples=self.crps_mc_samples).unsqueeze(-1) # [B, S, 1]
            assert loss_values.shape == (batch_dim, station_dim, 1)

            mask_i = mask[..., i:i+1]
            
            loss_scalar = self.loss_normalization_per_sample(loss_values, mask_i) * self.loss_scaling[i]
            
            overall_loss = overall_loss + loss_scalar
        
        return out, overall_loss

    
    # TODO innovations
    
    def compute_innovation(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        
        if self.observation_assimilation_as == "innovation":
            predicted_distributions = self.forward(x, as_dict=True) # predict the distribution
            innovations = []
            for i, (var_name, dist) in enumerate(predicted_distributions.items()): # for each variable, compute the innovation according to the distribution-specific method
                # extract the univariate target (assumes the order of the dictionary is the same as the order of the variables in the target tensor)
                target_i = target[..., i] if dist.event_shape == () else target[..., i:i+1]
                mask_i = mask[..., i] if dist.event_shape == () else mask[..., i:i+1]
                
                innovation_i = dist.innovation(target=target_i)
                innovations.append(innovation_i * mask_i)
                
            return torch.stack(innovations, dim=-1)

        # straight-through: return the raw observations and let the fusion layer learn how to use them
        return target

    