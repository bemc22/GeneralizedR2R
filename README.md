# Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise

[Brayan Monroy](https://bemc22.github.io), [Jorge Bacca](https://scholar.google.com/citations?user=I5f1HjEAAAAJ&hl=es), [Julian Tachella](https://tachella.github.io)

---

[![arXiv](https://img.shields.io/badge/arXiv-2412.04648-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2412.04648)

In this paper, we propose Generalized R2R (GR2R), extending the R2R framework to handle a broader class of noise distribution as additive noise like log-Rayleigh and address the natural exponential family including Poisson and Gamma noise distributions, which play a key role in many applications including low-photon imaging and synthetic aperture radar. We show that the GR2R loss is an unbiased estimator of the supervised loss and that the popular Stein's unbiased risk estimator can be seen as a special case.

:loudspeaker: **News**
- **Jan. 04, 2025**: Fully integrated into [DeepInverse](https://github.com/deepinv/deepinv) [![GitHub Stars](https://img.shields.io/github/stars/deepinv/deepinv?style=social)](https://github.com/deepinv/deepinv), take a look at the [examples](https://deepinv.github.io/deepinv/api/stubs/deepinv.loss.R2RLoss.html#deepinv.loss.R2RLoss)!
### Method

We present GR2R, this loss can be used for unsupervised image denoising with unorganized noisy images where the observation model $`\mathbf{y}\sim p(\mathbf{y}|\mathbf{x})`$ belongs to the natural exponential family as
```math
 p(\mathbf{y}|\mathbf{x})= h(\mathbf{y}) \exp( \mathbf{y}^{\top} \eta(\mathbf{x}) - \phi(\mathbf{x})).
```

For this family of measurements distribution, we generalize the corruption strategy as

```math
\mathbf{y}_1 \sim  \; p(\mathbf{y}_1| \mathbf{y}, \alpha),
```
```math
\mathbf{y}_2 =   \frac{1}{\alpha} \mathbf{y} -  \frac{(1-\alpha)}{\alpha}\mathbf{y}_1,
```

then, the generalize MSE loss is computed as
```math
\mathcal{L}_{\text{GR2R-MSE}}^{\alpha}(\mathbf{y};f)=\mathbb{E}_{\mathbf{y}_1,\mathbf{y}_2|\mathbf{y},\alpha}  \Vert f(\mathbf{y}_1) - \mathbf{y}_2 \Vert_2^2.
```
### Implementations

We provide training and testing demonstrations for image denoising across popular noise distributions belonging to the natural exponential family, such as, Gamma, Poisson, Gaussian, and Binomial noise.

| Demo  | Noise Type        | Dataset |   Link |  
| ----------- | -----------   | ----------- | ----------- |
|Train| Poisson/Gaussian/Gamma| DIV2K| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/GeneralizedR2R/blob/main/demo_denoising.ipynb)  |
|Test|Gamma| DIV2K | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/GeneralizedR2R/blob/main/demo_test_gamma.ipynb)  |
|Test|Poisson| DIV2K | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/GeneralizedR2R/blob/main/demo_test_poisson.ipynb)  |
|Test| Gaussian| fastMRI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/GeneralizedR2R/blob/main/demo_test_gaussian.ipynb)  |

### How to cite
If this code is useful for your and you use it in an academic work, please consider citing this paper as



```bib
@misc{monroy2024gr2r,
  author={Brayan Monroy and Jorge Bacca and Julián Tachella},
  title={Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise},
  year={2024},
  eprint={2412.04648},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2412.04648}, 
}
```

