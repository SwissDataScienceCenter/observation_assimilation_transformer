import torch
from torch import distributions as dist
from torch.distributions import transform_to

def check_if_valid_distribution(dist_class):
    # Check if the distribution class is valid by ensuring that all parameters are univariate (event_dim == 0).
    if not all([constraint.event_dim == 0 for constraint in dist_class.arg_constraints.values()]):
        raise ValueError(f"{dist_class} has multivariate parameters, which are not supported.")
    
class DistributionMap(torch.nn.Module):
    
    def __init__(self, 
                 dist_class: type[dist.Distribution], 
                 in_features: int,
                 param_order: list[str] | None = None):
        """
        Initialize a distribution layer.
        Args:
            dist_class: A PyTorch distribution class that inherits from torch.distributions.Distribution.
            in_features: The number of input features for the linear layer that generates distribution parameters.
        Raises:
            AssertionError: If the distribution has multivariate parameters (event_dim != 0), which are not supported.
        Attributes:
            dist_class: The distribution class.
            support: The support of the distribution.
            num_params: The number of parameters required by the distribution.
            net: A linear layer that maps from in_features to num_params.
        """
        check_if_valid_distribution(dist_class)
        super().__init__()
        self.dist_class = dist_class
        self.support = dist_class.support
        self.num_params = len(dist_class.arg_constraints)
        self.net = torch.nn.Linear(in_features, self.num_params)
        if param_order is not None:
            unknown = set(param_order) - set(dist_class.arg_constraints.keys())
            if len(unknown) > 0:
                raise ValueError(f"Unknown parameters in param_order: {unknown}")
            self.param_names = param_order
        else:
            self.param_names = list(dist_class.arg_constraints.keys())
            
    def forward(self, input: torch.Tensor) -> dist.Distribution:
        """
        Generate a probability distribution from input data.
        This method processes input through a neural network and transforms the outputs
        into valid distribution parameters according to the constraints defined by the
        distribution class.
        Args:
            input (torch.Tensor): Input tensor to be processed by the network.
        Returns:
            dist.Distribution: A probability distribution object initialized with the
                transformed network outputs as parameters.
        Notes:
            - The network output is transformed for each distribution parameter using
              the appropriate constraint transformations.
            - The number of output dimensions must match the number of distribution
              parameters.
        """
        dist_params = {}
        out = self.net(input)
        for i, name in enumerate(self.param_names):
            constraint = self.dist_class.arg_constraints[name]
            dist_params[name] = transform_to(constraint)(out[..., i])
        return self.dist_class(**dist_params)