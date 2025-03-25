# Inference-Time Prior Adaptation in Simulation-Based Inference via Guided Diffusion Models - Appendix

---

#### Page 10

## A APPENDIX

## A. 1 GAUSSIAN InteGRATION

Here is the detailed derivation for Eq. (14) from the main text:

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}_{t}} \log \mathbb{E}\left[\boldsymbol{\rho}\left(\boldsymbol{\theta}_{0}\right)\right] & \approx \nabla_{\boldsymbol{\theta}_{t}} \log \int \sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \mathcal{N}\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \boldsymbol{\Sigma}_{0 \mid t}\right) d \boldsymbol{\theta}_{0} \\
& =\nabla_{\boldsymbol{\theta}_{t}} \log \sum_{i=1}^{K} \int \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\theta}_{0}, \boldsymbol{\Sigma}_{i}\right) \mathcal{N}\left(\boldsymbol{\theta}_{0} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \boldsymbol{\Sigma}_{0 \mid t}\right) d \boldsymbol{\theta}_{0}
\end{aligned}
$$

The step above uses the symmetry property of Gaussian distributions: if $\mathbf{a} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ then $\boldsymbol{\mu} \sim$ $\mathcal{N}(\mathbf{a}, \boldsymbol{\Sigma})$. This allows us to swap $\boldsymbol{\theta}_{0}$ and $\boldsymbol{\mu}_{i}$ in the first Gaussian. Furthermore,

$$
=\nabla_{\boldsymbol{\theta}_{t}} \log \sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \boldsymbol{\Sigma}_{i}+\boldsymbol{\Sigma}_{0 \mid t}\right)
$$

using the standard result for the convolution of two Gaussian distributions:

$$
\int \mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{1}, \boldsymbol{\Sigma}_{1}\right) \mathcal{N}\left(\boldsymbol{\mu}_{1} \mid \boldsymbol{\mu}_{2}, \boldsymbol{\Sigma}_{2}\right) d \boldsymbol{\mu}_{1}=\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{2}, \boldsymbol{\Sigma}_{1}+\boldsymbol{\Sigma}_{2}\right)
$$

For notational convenience, we define $\widetilde{\boldsymbol{\Sigma}}_{i}=\boldsymbol{\Sigma}_{i}+\boldsymbol{\Sigma}_{0 \mid t}$ continuing with the derivation:

$$
\begin{aligned}
& =\nabla_{\boldsymbol{\theta}_{t}} \log \sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right) \\
& =\frac{\nabla_{\boldsymbol{\theta}_{t}} \sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right)}{\sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right)} \quad \text { (chain rule) } \\
& =\frac{\sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right) \nabla_{\boldsymbol{\theta}_{t}} \log \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right)}{\sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right)} \quad \text { (since } \nabla f=f \nabla \log f) \\
& =\frac{\sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right) \nabla_{\boldsymbol{\theta}_{t}}\left(-\frac{1}{2}\left(\boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)-\boldsymbol{\mu}_{i}\right)^{\top} \widetilde{\boldsymbol{\Sigma}}_{i}^{-1}\left(\boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)-\boldsymbol{\mu}_{i}\right)\right)}{\sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right)} \\
& =\frac{\sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right)\left(\boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)\right)^{\mathbf{T}} \widetilde{\boldsymbol{\Sigma}}_{i}^{-1} \nabla_{\boldsymbol{\theta}_{t}} \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right)}{\sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{\mu}_{i} \mid \boldsymbol{\mu}_{0 \mid t}\left(\boldsymbol{\theta}_{t}\right), \widetilde{\boldsymbol{\Sigma}}_{i}\right)}
\end{aligned}
$$

## A. 2 EXPERIMENTAL DETAILS

Toy Gaussian Example. A Gaussian likelihood is chosen for tractability, where $x \mid \boldsymbol{\theta} \sim$ $\mathcal{N}\left(x ; \theta_{1}, \theta_{2}^{2}\right)$ so $\boldsymbol{\theta} \in \mathbb{R}^{2}$. The original prior $p(\boldsymbol{\theta})$ is uniform over $[0,1]^{2}$, while the new prior $q(\boldsymbol{\theta})$ is a multivariate Gaussian distribution:

$$
q(\boldsymbol{\theta})=\mathcal{N}\left(\boldsymbol{\theta} ;\left[\begin{array}{l}
0.3 \\
0.8
\end{array}\right],\left[\begin{array}{cc}
0.039 & 0.025 \\
0.025 & 0.04
\end{array}\right]\right)
$$

where $\theta_{1}$ represents the mean and $\theta_{2}$ the standard deviation of the likelihood. This choice of prior introduces correlation between the mean and standard deviation parameters while concentrating probability mass in a specific region of the parameter space. The $\mathbf{x}$ for likelihood calculations for training are 10 samples from a given $\boldsymbol{\theta}^{(1)}$ therefore $\mathbf{x}^{(1)} \in \mathbb{R}^{10}$. The base model was trained with 10,000 simulations. The network architecture and training scheme was taken from the base configuration in Gloeckler et al. (2024). In Fig. 1 a histogram plot shows the sample frequency as a comparison for the posterior density which can be computed exactly.

---

#### Page 11

> **Image description.** The image consists of three scatter plots arranged horizontally, each depicting data points forming two crescent-shaped clusters, resembling moons. The plots are labeled (a), (b), and (c) along the bottom.
>
> - **Panel (a): Prior v samples:** This plot displays two crescent-shaped clusters of orange data points. Superimposed on the data points are several gray contour lines, each labeled with a percentage (e.g., 95%, 90%, 80%, 60%). The contours suggest a probability density function with two peaks, corresponding to the locations of the crescent shapes.
>
> - **Panel (b): PriorGuide v samples:** This plot also shows two crescent-shaped clusters. However, the data points are now a mix of orange and light blue. The orange points appear to be concentrated in the upper portions of the crescents, while the light blue points are more prevalent in the lower portions. There are no contour lines in this plot.
>
> - **Panel (c): PriorGuide v retrained:** Similar to panel (b), this plot displays two crescent-shaped clusters. The data points are a mix of light blue and red. The red points are concentrated in the lower portion of the crescents, while light blue points are more prevalent in the upper portions.
>
> The overall impression is a comparison of different sampling methods or models, with the color of the data points indicating different sources or algorithms.

Figure A.1: Two moons with correlated prior. The points are samples from the diffusion model trained with uniform prior $p(\boldsymbol{\theta})$. Contours of the new prior $q(\boldsymbol{\theta})$ are shown in - . The $\boldsymbol{\Delta}$ points are PriorGuide samples using this new prior. Fig. A.1c compares these against samples from a model retrained with the new prior, showing comparable results without retraining.

Two Moons with Correlated Prior. We use the standard two moons example in the SBI package detailed in Greenberg et al. (2019), where $\boldsymbol{\theta} \in \mathbb{R}^{2}$ and $\mathbf{x} \in \mathbb{R}^{2}$. The original prior $p(\boldsymbol{\theta})$ is uniform over $[-1,1]^{2}$, while the new prior $q(\boldsymbol{\theta})$ is a multivariate mixture Gaussian distribution:

$$
q(\boldsymbol{\theta})=\frac{1}{2} \mathcal{N}\left(\boldsymbol{\theta} ;\left[\begin{array}{l}
0.2 \\
0.2
\end{array}\right],\left[\begin{array}{cc}
0.01 & 0.007 \\
0.007 & 0.01
\end{array}\right]\right)+\frac{1}{2} \mathcal{N}\left(\boldsymbol{\theta} ;\left[\begin{array}{c}
-0.2 \\
-0.2
\end{array}\right],\left[\begin{array}{cc}
0.01 & 0.007 \\
0.007 & 0.01
\end{array}\right]\right)
$$

where the mixture weights are equal so 0.5 , and each component shares the same covariance matrix with correlation coefficient. The base model was trained with 10,000 simulations and same network architecture as in the previous example.

Ornstein-Uhlenbeck Process (OUP). OUP is a well-established stochastic process frequently applied in financial mathematics and evolutionary biology for modeling mean-reverting dynamics (Uhlenbeck \& Ornstein, 1930). The model is defined as:

$$
y_{t+1}=y_{t}+\Delta y_{t}, \quad \Delta y_{t}=\theta_{1}\left[\exp \left(\theta_{2}\right)-y_{t}\right] \Delta t+0.5 w, \quad \text { for } t=1, \ldots, T
$$

where we set $T=25, \Delta t=0.2$, and initialize $x_{0}=10$. The noise term follows a Gaussian distribution, $w \sim \mathcal{N}(0, \Delta t)$. We define $p(\boldsymbol{\theta})$ as a uniform prior, $U([0,2] \times[-2,2])$, over the latent parameters $\boldsymbol{\theta}=\left(\theta_{1}, \theta_{2}\right)$.
For this OUP task, the base model is trained on 10,000 simulations. We evaluate the performance using Maximum Mean Discrepancy (MMD) with an exponentiated quadratic kernel with a lengthscale of 1 , and Root Mean Squared Error (RMSE). Each experiment is evaluated using 100 randomly sampled $\boldsymbol{\theta}$. For each $\boldsymbol{\theta}$, we generate 1,000 posterior samples, repeating this process over five runs.
We define two new prior distributions $q(\boldsymbol{\theta})$ for the OUP experiments: (i) The simple prior consists of Gaussian distributions with a standard deviation set to $5 \%$ of the parameter range. Each prior's mean is sampled from a Gaussian centered on the true parameter value, using the same standard deviation (similar to Chang et al., 2024). (ii) The complex prior, a mixture of two slightly correlated bivariate Gaussians with equal component weights $\left(\pi_{1}=\pi_{2}=0.5\right)$ :

$$
q(\boldsymbol{\theta})=\pi_{1} \mathcal{N}\left(\binom{0.5}{-1.0},\left(\begin{array}{cc}
0.06 & 0.01 \\
0.01 & 0.06
\end{array}\right)\right)+\pi_{2} \mathcal{N}\left(\binom{1.3}{0.5},\left(\begin{array}{cc}
0.06 & 0.01 \\
0.01 & 0.06
\end{array}\right)\right)
$$

Turin Model. Turin is a widely used time-series model for simulating radio wave propagation (Turin et al., 1972; Pedersen, 2019). This model generates high-dimensional, complex-valued timeseries data and is governed by four key parameters: $G_{0}$ determines the reverberation gain, $T$ controls the reverberation time, $\lambda_{0}$ defines the arrival rate of the point process, and $\sigma_{N}^{2}$ represents the noise variance.
The model assumes a frequency bandwidth of $B=0.5 \mathrm{GHz}$ and simulates the transfer function $H_{k}$ at $N_{s}=101$ evenly spaced frequency points. The observed transfer function at the $k$-th frequency point, $Y_{k}$, is defined as:

$$
Y_{k}=H_{k}+W_{k}, \quad k=0,1, \ldots, N_{s}-1
$$

---

#### Page 12

where $W_{k}$ represents additive zero-mean complex Gaussian noise with circular symmetry and variance $\sigma_{W}^{2}$. The transfer function $H_{k}$ is expressed as:

$$
H_{k}=\sum_{l=1}^{N_{\text {pairs }}} \alpha_{l} \exp \left(-j 2 \pi \Delta f k \tau_{l}\right)
$$

where the time delays $\tau_{l}$ are sampled from a homogeneous Poisson point process with rate $\lambda_{0}$, and the complex gains $\alpha_{l}$ are modeled as independent zero-mean complex Gaussian random variables. The conditional variance of the gains is given by:

$$
\mathbb{E}\left[\left|\alpha_{l}\right|^{2} \mid \tau_{l}\right]=\frac{G_{0} \exp \left(-\tau_{l} / T\right)}{\lambda_{0}}
$$

To obtain the time-domain signal $\tilde{y}(t)$, an inverse Fourier transform is applied:

$$
\tilde{y}(t)=\frac{1}{N_{s}} \sum_{k=0}^{N_{s}-1} Y_{k} \exp (j 2 \pi k \Delta f t)
$$

where $\Delta f=B /\left(N_{s}-1\right)$ represents the frequency spacing. Finally, the real-valued output is computed by taking the absolute square of the complex signal and applying a logarithmic transformation:

$$
y(t)=10 \log _{10}\left(|\tilde{y}(t)|^{2}\right)
$$

We follow the same training and experimental setup as in OUP. In this Turin case, all parameters are normalized to $[0,1]$ using the transformation: $\tilde{x}=\frac{x-x_{\text {urin }}}{x_{\text {max }}-x_{\text {min }}}$, where $\tilde{x}$ is the normalized value. The true parameter bounds are: $G_{0} \in\left[10^{-9}, 10^{-8}\right], \quad T \in\left[10^{-9}, 10^{-8}\right], \quad \lambda_{0} \in\left[10^{7}, 5 \times 10^{9}\right], \quad \sigma_{N}^{2} \in$ $\left[10^{-10}, 10^{-9}\right]$.

For this Turin problem, the simple prior follows the same specification as in OUP, while the complex prior is also a multivariate Gaussian mixture with equal component weights but with different component parameters, adjusted to match the Turin model's parameter dimension and normalized range, defined as:

$$
\begin{aligned}
q(\boldsymbol{\theta})= & \pi_{1} \mathcal{N}\left(\left(\begin{array}{c}
0.30 \\
0.30 \\
0.70 \\
0.70
\end{array}\right),\left(\begin{array}{cccc}
0.01 & 0.005 & 0.005 & 0.005 \\
0.005 & 0.01 & 0.005 & 0.005 \\
0.005 & 0.005 & 0.01 & 0.005 \\
0.005 & 0.005 & 0.005 & 0.01
\end{array}\right)\right) \\
& +\pi_{2} \mathcal{N}\left(\left(\begin{array}{c}
0.70 \\
0.70 \\
0.30 \\
0.30
\end{array}\right),\left(\begin{array}{cccc}
0.01 & 0.005 & 0.005 & 0.005 \\
0.005 & 0.01 & 0.005 & 0.005 \\
0.005 & 0.005 & 0.01 & 0.005 \\
0.005 & 0.005 & 0.005 & 0.01
\end{array}\right)\right)
\end{aligned}
$$

# A. 3 SBI Mixture Prior Corner Plots

As a representative visualization of the SBI experiments, we present example corner plots of posterior samples for the case where the sampling distribution of $\boldsymbol{\theta}$ follows a mixture distribution in both the OUP and Turin SBI tasks. These plots illustrate marginal pairwise relationships between sampled latent parameters and demonstrate that PriorGuide can handle complex priors, producing posterior results that are reasonable given the prior structure.

Fig. A. 2 presents the corner plots for the OUP case, comparing Simformer and PriorGuide. The higher-dimensional Turin task is shown in Fig. A. 3 and Fig. A. 4 for Simformer and PriorGuide, respectively.

---

#### Page 13

> **Image description.** This image contains two sets of plots, labeled (a) and (b), each showing the distribution of two parameters, theta1 and theta2. Each set consists of three plots: a histogram for theta1, a histogram for theta2, and a contour plot showing the joint distribution of theta1 and theta2.
>
> In plot (a):
>
> - The histogram for theta1 is centered around 0.85, with error bars of +0.10 and -0.09. The histogram is blue. A vertical blue line marks the center.
> - The histogram for theta2 is centered around 0.14, with error bars of +0.28 and -0.43. The histogram is blue. A vertical blue line marks the center.
> - The contour plot shows the joint distribution of theta1 and theta2. The x-axis represents theta1, and the y-axis represents theta2. The contours are blue and show a single cluster of points. Horizontal and vertical blue lines mark the means of the distributions.
>
> In plot (b):
>
> - The histogram for theta1 shows two distributions: a blue histogram and a red curve labeled "Prior." The histogram is centered around 0.89, with error bars of +0.11 and -0.17. A vertical blue line marks the center.
> - The histogram for theta2 also shows two distributions: a blue histogram and a red curve. The histogram is centered around 0.28, with error bars of +0.20 and -1.17. A vertical blue line marks the center.
> - The contour plot shows the joint distribution of theta1 and theta2. The x-axis represents theta1, and the y-axis represents theta2. The contours are blue and show two clusters of points. Red "x" marks are labeled "Prior mean" and indicate the means of the prior distributions. Horizontal and vertical blue lines mark the means of the distributions.
>
> The axes in all plots are labeled with theta1 and theta2. The y-axes of the histograms are not labeled. The x-axes of the histograms are labeled with values ranging from approximately -1.6 to 1.6. The x and y axes of the contour plots range from approximately 0.4 to 2.0 and -1.6 to 1.6, respectively.

Figure A.2: OUP model. Comparison of posterior samples between Simformer and PriorGuide. The light blue line is the true parameter value. The bottom left corner of (b) shows the sampling mixture distribution (and prior); see Eq. (A.12) for detail. (a) Simformer results (without prior guidance), where the model fails to capture the true mixture distribution of $\boldsymbol{\theta}$. (b) PriorGuide helps the base model generate posterior results that align well with the structure of the complex prior.

---

#### Page 14

> **Image description.** The image is a correlation plot, also known as a pair plot or scatter plot matrix, displaying the relationships between four variables: G₀, T, λ₀, and σ²N. It consists of a grid of subplots.
>
> - **Diagonal Subplots:** The diagonal subplots display histograms of each individual variable.
>
>   - Top-left: Histogram of G₀, labeled as "G₀ = 0.37 ± 0.31". The histogram shows a decreasing frequency from left to right. A vertical blue line is positioned near the center of the distribution.
>   - Middle-center: Histogram of T, labeled as "T = 0.42 ± 0.09". The histogram shows a peak around the center. A vertical blue line is positioned near the center of the distribution.
>   - Center-right: Histogram of λ₀, labeled as "λ₀ = 0.64 ± 0.25". The histogram shows an increasing frequency from left to right. A vertical blue line is positioned near the center of the distribution.
>   - Bottom-right: Histogram of σ²N, labeled as "σ²N = 0.33 ± 0.08". The histogram shows a peak around the center. A vertical blue line is positioned near the center of the distribution.
>
> - **Off-Diagonal Subplots:** The off-diagonal subplots display scatter plots of each pair of variables. The scatter plots are represented by density contours and scattered points in blue. Each subplot also includes horizontal and vertical blue lines that correspond to the mean of each variable.
>
>   - Bottom-left: Scatter plot of σ²N vs. G₀. The x-axis is labeled "G₀", and the y-axis is labeled "σ²N".
>   - Bottom-middle: Scatter plot of σ²N vs. T. The x-axis is labeled "T", and the y-axis is labeled "σ²N".
>   - Bottom-center: Scatter plot of σ²N vs. λ₀. The x-axis is labeled "λ₀", and the y-axis is labeled "σ²N".
>   - Middle-left: Scatter plot of λ₀ vs. G₀. The x-axis is labeled "G₀", and the y-axis is labeled "λ₀".
>   - Middle-bottom: Scatter plot of λ₀ vs. T. The x-axis is labeled "T", and the y-axis is labeled "λ₀".
>   - Top-left: Scatter plot of T vs. G₀. The x-axis is labeled "G₀", and the y-axis is labeled "T".
>
> - **Axes Labels:** The axes are labeled with values ranging from 0.2 to 1.0 in increments of 0.2.
>
> The entire plot is contained within a rectangular frame.

Figure A.3: Turin model (SimFormer). Posterior samples using Simformer, without prior guidance. The light blue line is the true parameter value. The sampling distribution is the mixture described in Eq. (A.13) (see bottom left corner of Fig. A. 4 for visualization). Since the model is trained on a uniform prior, it yields a wide posterior that fails to capture the multimodality of the true $\boldsymbol{\theta}$ distribution.

---

#### Page 15

> **Image description.** This image is a visualization of a corner plot, displaying the marginal and joint posterior distributions of several parameters. The plot consists of a grid of subplots.
>
> - **Diagonal Subplots:** Each diagonal subplot displays a histogram (blue) and a probability density function (PDF) curve (red) for a single parameter. The parameters are G0, T, λ0, and σN^2. Above each histogram, the mean value and the uncertainty range are displayed (e.g., "G0 = 0.46 +0.16 -0.16").
> - **Off-Diagonal Subplots:** The off-diagonal subplots show the joint distributions of pairs of parameters as contour plots. The contours are blue. Each subplot has axes labeled with the corresponding parameters (G0, T, λ0, σN^2). A horizontal and vertical line are drawn at the mean value of each parameter.
> - **Parameter Labels:** The parameters are labeled as follows:
>   - G0 (top left)
>   - T (second row, second column)
>   - λ0 (third row, third column)
>   - σN^2 (bottom right)
> - **Contour Plots:** The contour plots in the off-diagonal subplots show the relationships between the parameters. They indicate the density of the posterior distribution. The plots in the first column show the relationship between G0 and T, λ0, and σN^2 respectively. The plots in the second column show the relationship between T and λ0, and σN^2 respectively. The plot in the third column shows the relationship between λ0 and σN^2.
> - **Additional Elements:** In the plot showing the relationship between T and G0, there is a legend indicating "Prior comp1" (red x) and "Prior comp2" (blue x). These likely represent the means of two components of the prior distribution.

Figure A.4: Turin model (PriorGuide). Posterior samples from PriorGuide. Compared to the Simformer without prior guidance (Fig. A.3), PriorGuide significantly improves posterior estimation, aligning it more closely with the complex prior structure while using the same model as the Simformer, without retraining. Note that the contour plots represent the sampling distribution (prior).
