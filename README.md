# Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise
In this paper, we propose Generalized R2R (GR2R), extending the R2R framework to handle a broader class of noise distribution as additive noise like log-Rayleigh and address the natural exponential family including Poisson and Gamma noise distributions, which play a key role in many applications including low-photon imaging and synthetic aperture radar. We show that the GR2R loss is an unbiased estimator of the supervised loss and that the popular Stein's unbiased risk estimator can be seen as a special case.

### Method

We present GR2R, this loss can be used for unsupervised image denoising with unorganized noisy images where the observation model $`\boldsymbol{y}\sim p(\boldsymbol{y}|\boldsymbol{x})`$ belongs to the natural exponential family as
```math
 p(\boldsymbol{y}|\boldsymbol{x})= h(\boldsymbol{y}) \exp( \boldsymbol{y}^{\top} \eta(\boldsymbol{x}) - \phi(\boldsymbol{x}).
```

For this family of measurements distribution, we generalize the corruption strategy as

```math
\boldsymbol{y}_1 \sim  \; p(\boldsymbol{y}_1| \boldsymbol{y}, \alpha),
```
```math
\boldsymbol{y}_2 =   \frac{1}{\alpha} \boldsymbol{y} -  \frac{(1-\alpha)}{\alpha}\boldsymbol{y}_1,
```

then, the generalize MSE loss is computed as
```math
\mathcal{L}_{\text{GR2R-MSE}}^{\alpha}(\boldsymbol{y};f)=\mathbb{E}_{\boldsymbol{y}_1,\boldsymbol{y}_2|\boldsymbol{y},\alpha}  \Vert f(\boldsymbol{y}_1) - \boldsymbol{y}_2 \Vert_2^2.
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
  author={Monroy, Brayan and Bacca, Jorge and Tachella, Julian},
  title={Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise}, 
  year={2024}
}
```

