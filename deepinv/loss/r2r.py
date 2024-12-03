import torch
from deepinv.loss.loss import Loss


class R2RLoss(Loss):
    r"""
    Recorrupted-to-Recorrupted (R2R) Loss

    This loss can be used for unsupervised image denoising with unorganized noisy images.

    The loss is designed for the noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).

    The loss is computed as:

    .. math::

        \| y^- - AR(y^+) \|_2^2 \quad \text{s.t.} \quad y^+ = y + \alpha z, \quad y^- = y -  z / \alpha

    where :math:`R` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement,
    :math:`z` is the additional Gaussian noise of standard deviation :math:`\sigma`,
    and :math:`\alpha` is a scaling factor.


    This loss is statistically equivalent to the supervised loss function defined on noisy/clean image pairs
    according to authors in https://ieeexplore.ieee.org/document/9577798

    .. warning::

        The model should be adapted before training using the method :meth:`adapt_model` to include the additional noise at the input.

    .. note::

        :math:`\sigma` should be chosen equal or close to :math:`\sigma` to obtain the best performance.

    .. note::

        To obtain the best test performance, the trained model should be averaged at test time
        over multiple realizations of the added noise, i.e. :math:`\hat{x} = \frac{1}{N}\sum_{i=1}^N R(y+\alpha z_i)`
        where :math:`N>1`. This can be achieved using :meth:`adapt_model`.

    :param float sigma: standard deviation of the Gaussian noise used for the perturbation.
    :param float alpha: scaling factor of the perturbation.

    |sep|

    :Example:

        >>> import torch
        >>> import deepinv as dinv
        >>> sigma = 0.1
        >>> physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
        >>> model = dinv.models.MedianFilter()
        >>> loss = dinv.loss.R2RLoss(sigma=sigma)
        >>> model = loss.adapt_model(model, MC_samples=2) # important step!
        >>> x = torch.ones((1, 1, 8, 8))
        >>> y = physics(x)
        >>> x_net = model(y, physics, update_parameters=True) # save extra noise in forward pass
        >>> l = loss(x_net, y, physics, model)
        >>> print(l.item() > 0)
        True

    """

    def __init__(self, metric=torch.nn.MSELoss(), sigma=0.1, alpha=0.5):
        super(R2RLoss, self).__init__()
        self.name = "r2r"
        self.metric = metric
        self.sigma = sigma
        self.alpha = alpha

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the R2R Loss.

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction model.
        :return: (torch.Tensor) R2R loss.
        """

        pert = model.get_noise()
        y_minus = y - pert / self.alpha
        return self.metric(physics.A(x_net), y_minus)

    def adapt_model(self, model, MC_samples=5):
        r"""
        Adds noise to model input.


        This method modifies a reconstruction
        model :math:`R` to include the splitting mechanism at the input:

        .. math::

            \hat{R}(y) = \frac{1}{N}\sum_{i=1}^N R(y+\alpha z_i)

        where :math:`N\geq 1` are the number of samples used for the Monte Carlo approximation.
        During training (i.e. when ``model.train()``), we use only one sample, i.e. :math:`N=1`
        for computational efficiency, whereas at test time, we use multiple samples for better performance.

        :param torch.nn.Module model: Reconstruction model.
        :param float sigma: standard deviation of the Gaussian noise used for the perturbation.
        :param float alpha: scaling factor of the perturbation.
        :param int MC_samples: number of samples used for the Monte Carlo approximation.
        :return: (torch.nn.Module) Modified model.
        """
        if isinstance(model, R2RModel):
            return model
        else:
            return R2RModel(model, self.sigma, self.alpha, MC_samples)


class R2RModel(torch.nn.Module):
    def __init__(self, model, sigma, alpha, MC_samples=5):
        super(R2RModel, self).__init__()
        self.model = model
        self.extra_noise = 0
        self.sigma = sigma
        self.alpha = alpha
        self.MC_samples = MC_samples

    def forward(self, y, physics, update_parameters=False):
        MC = 1 if self.training else self.MC_samples

        out = 0
        with torch.set_grad_enabled(self.training):
            for i in range(MC):
                extra_noise = torch.randn_like(y) * self.sigma
                y_plus = y + extra_noise * self.alpha
                out += self.model(y_plus, physics)

            if self.training and update_parameters:  # save the noise
                self.extra_noise = extra_noise

            out = out / MC
        return out

    def get_noise(self):
        return self.extra_noise
    

class R2RPoissonModel(torch.nn.Module):
    def __init__(self, model, gain, p, MC_samples=5):
        super(R2RPoissonModel, self).__init__()
        self.model = model
        self.extra_noise = 0
        self.gain = gain
        self.p = p
        self.MC_samples = MC_samples

    def forward(self, y, physics, update_parameters=False):
        MC = 1 if self.training else self.MC_samples

        out = 0

        y_param  = y.detach() / self.gain
        sampling = torch.distributions.binomial.Binomial(torch.round(y_param), self.p)
        
        with torch.set_grad_enabled(self.training):
            for i in range(MC):

                

                extra_noise = sampling.sample()
                # y_plus = y + extra_noise * self.alpha
                y_plus = ( y_param - extra_noise ) / ( 1 - self.p )
                y_plus = y_plus * self.gain

                out += self.model(y_plus, physics)

            if self.training and update_parameters:  # save the noise
                self.extra_noise = extra_noise

            out = out / MC
        return out

    def get_noise(self):
        return self.extra_noise
    

class R2RPoissonLoss(Loss):
    r"""
    Recorrupted-to-Recorrupted (R2R) Loss for Poisson noise

    """


    def __init__(self, metric=torch.nn.MSELoss(), gain=1.0, p=0.5):
        super(R2RPoissonLoss, self).__init__()
        self.name = "r2r"
        self.metric = metric
        self.gain = gain
        self.p = p

    def forward(self, x_net, y, physics, model, **kwargs):
        pert = model.get_noise()
        y_minus = pert / self.p 
        y_minus = y_minus * self.gain
        return self.metric(physics.A(x_net), y_minus)

    def adapt_model(self, model, MC_samples=5):

        if isinstance(model, R2RModel):
            return model
        else:
            return R2RPoissonModel(model, self.gain, self.p, MC_samples)



class R2RGammaModel(torch.nn.Module):
    def __init__(self, model, l, alpha, MC_samples=5):
        super(R2RGammaModel, self).__init__()
        self.model = model
        self.extra_noise = 0
        self.l = l
        self.alpha = alpha
        self.MC_samples = MC_samples

    def forward(self, y, physics, update_parameters=False):
        MC = 1 if self.training else self.MC_samples

        out = 0


        tmp = torch.ones_like(y)

        sampling = torch.distributions.Beta(tmp * self.l * self.alpha, tmp * self.l * (1 - self.alpha))
        
        with torch.set_grad_enabled(self.training):
            for i in range(MC):

                extra_noise = sampling.sample()
                y_plus =  y * ( 1 - extra_noise) / (1 - self.alpha)
                out +=  self.model(y_plus, physics) 

            if self.training and update_parameters:  # save the noise
                self.extra_noise = extra_noise

            out = out / MC
        return out

    def get_noise(self):
        return self.extra_noise
    

class R2RGammaLoss(Loss):
    r"""
    Recorrupted-to-Recorrupted (R2R) Loss for Gamma noise

    """

    def __init__(self, metric=torch.nn.MSELoss(), l=1.0, alpha=0.5):
        super(R2RGammaLoss, self).__init__()
        self.name = "r2r"
        self.metric = metric
        self.l = l
        self.alpha = alpha

    def forward(self, x_net, y, physics, model, **kwargs):
        pert = model.get_noise()
        y_minus = y * pert /self.alpha
        return self.metric(physics.A(x_net), y_minus)

    def adapt_model(self, model, MC_samples=5):

        if isinstance(model, R2RModel):
            return model
        else:
            return R2RGammaModel(model, self.l, self.alpha, MC_samples)


class SureGammaModel(torch.nn.Module):
    def __init__(self, model, l):
        super(SureGammaModel, self).__init__()
        self.model = model
        self.extra_noise = 0
        self.l = l

    def forward(self, y, physics, update_parameters=False):
        return -self.l / self.model(y, physics)

class SureGammaLoss(Loss):
    r"""
        SURE Gamma Loss
    """

    def __init__(self, metric=torch.nn.MSELoss(), l=1.0, alpha=0.5):
        super(SureGammaLoss, self).__init__()
        self.name = "sure"
        self.metric = metric
        self.l = l
        self.alpha = alpha
    
    def forward(self, x_net, y, physics, model, **kwargs):
        pert = model.get_noise()
        y_minus = y * pert /self.alpha
        return self.metric(physics.A(x_net), y_minus)

    def adapt_model(self, model):

        if isinstance(model, R2RModel):
            return model
        else:
            return SureGammaModel(model, self.l)

