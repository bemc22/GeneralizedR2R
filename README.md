# Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise
In this paper, we propose Generalized R2R (GR2R), extending the R2R framework to handle a broader class of noise distribution as additive noise like log-Rayleigh and address the natural exponential family including Poisson and Gamma noise distributions, which play a key role in many applications including low-photon imaging and synthetic aperture radar. We show that the GR2R loss is an unbiased estimator of the supervised loss and that the popular Stein's unbiased risk estimator can be seen as a special case.

### Implementations
| Noise Type        | Dataset |   Link |  
| -----------   | ----------- | ----------- |
| Poisson/Gaussian/Gamma   | DIV2K| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/GeneralizedR2R/blob/main/demo_poisson.ipynb)      |

