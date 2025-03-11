# Amortized Probabilistic Conditioning for Optimization, Simulation and Inference - Appendix

---

#### Page 14

# Supplementary Material

## Contents

A TABLE OF ACRONYMS ..... 15
B METHODS ..... 15
B. 1 Details and experiments with prior injection ..... 16
B. 2 Architecture ..... 20
B. 3 Training batch construction ..... 21
B. 4 Autoregressive predictions ..... 21
C EXPERIMENTAL DETAILS ..... 21
C. 1 Gaussian process (GP) experiments ..... 21
C. 2 Image completion and classification ..... 22
C. 3 Bayesian optimization ..... 24
C. 4 Simulation-based inference ..... 33
C. 5 Computational resources and software ..... 39

---

#### Page 15

# A TABLE OF ACRONYMS

For ease of reference, Table S1 reports a list of key acronyms and abbreviations used in the paper.

|                Acronym                 |                  Full Name                  |                                               Description                                               |
| :------------------------------------: | :-----------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
|             Architectures              |                                             |                                                                                                         |
|                 TPM-D                  |    Transformer Prediction Map - Diagonal    |   Family of transformer architectures for diagonal prediction maps, including all architectures below   |
|                  ACE                   |        Amortized Conditioning Engine        |    Our transformer-based meta-learning model for probabilistic tasks with explicit latent variables     |
|                  ACEP                  | Amortized Conditioning Engine (with Priors) |                 ACE variant allowing runtime injection of priors over latent variables                  |
|                  CNP                   |         Conditional Neural Process          |                          Context-to-target mapping with permutation invariance                          |
|                  PFN                   |            Prior-Fitted Network             | Meta-learning approach using transformers for inference and introducing Riemannian output distributions |
|                 TNP-D                  |    Transformer Neural Process - Diagonal    |                 Transformer neural process variant with independent target predictions                  |
|    Bayesian Optimization (BO) Terms    |                                             |                                                                                                         |
|                   BO                   |            Bayesian Optimization            |                         Black-box function optimization using surrogate models                          |
|                  MES                   |          Max-Value Entropy Search           |                      Acquisition function based on uncertainty over optimum value                       |
|                   TS                   |              Thompson Sampling              |                   Optimization via sampling from the posterior over optimum location                    |
|                $\pi$ BO                |            Prior-information BO             |                          BO incorporating prior knowledge on optimum location                           |
|              AR-TNP-D-TS               |   Autoregressive TNP-D Thompson Sampling    |                            TNP extension with autoregressive sampling for BO                            |
| Simulation-Based Inference (SBI) Terms |                                             |                                                                                                         |
|                  SBI                   |         Simulation-Based Inference          |                           Parameter posterior inference using synthetic data                            |
|                  NPE                   |         Neural Posterior Estimation         |                             Direct posterior modeling with neural networks                              |
|                  NRE                   |           Neural Ratio Estimation           |                               Likelihood-ratio-based posterior inference                                |
|                  OUP                   |         Ornstein-Uhlenbeck Process          |                                    Mean-reverting stochastic process                                    |
|                  SIR                   |      Susceptible-Infectious-Recovered       |                                  Epidemiological disease spread model                                   |

Table S1: Key acronyms used in the paper, grouped by category.

## B METHODS

This section details several technical aspects of our paper, such as the prior amortization techniques, neural network architecture and general training and inference details.

---

#### Page 16

# B. 1 Details and experiments with prior injection

Prior generative process. To expose ACE to a wide array of distinct priors during training, we generate priors following a hierarchical approach that generates smooth priors over a bounded range. The process is as follows, separately for each latent variable $\theta_{l}$, for $1 \leq l \leq L$ :

- We first sample the type of priors for the latent variable. Specifically, with $80 \%$ probability, we sample from a mixture of Gaussians to generate a smooth prior, otherwise, we create a flat prior with uniform distribution.
- If we sample from a mixture of Gaussians:
- We first sample the number of Gaussian components $K$ from a geometric distribution with $q=0.5$ :

$$
K \sim \text { Geometric }(0.5)
$$

- If $K>1$, we randomly choose among three configurations with equal probability:

1. Same means and different standard deviations.
2. Different means and same standard deviations.
3. All different means and standard deviations.

- Given the predefined global priors for mean and standard deviation (uniform distributions whose ranges are determined by the range of the corresponding latent variable), we sample the means and standard deviations for each component from the predefined uniform distributions.
- The weights of the mixture components are sampled from a Dirichlet distribution:

$$
\mathbf{w} \sim \operatorname{Dirichlet}\left(\alpha_{0}=1\right)
$$

- Finally, we convert the mixture of Gaussians into a normalized histogram over a grid $\mathcal{G}$ with $N_{\text {bins }}$ uniformly-spaced bins. For each bin $b$, we compute the probability mass $\mathbf{p}_{l}^{(b)}$ by calculating the difference between the cumulative distribution function values at the bin edges. This is done for each Gaussian component and then summed up, weighted by the mixture weights.
- We normalize the bin probabilities to ensure a valid probability distribution:

$$
\mathbf{p}_{l}=\frac{\mathbf{p}_{l}}{\sum_{b=1}^{N_{\text {bins }}} \mathbf{p}_{l}^{(b)}}
$$

- If we sample from a uniform distribution:
- We assign equal probability to each bin over the grid:

$$
\mathbf{p}_{l}=\frac{1}{N_{\text {bins }}} \mathbf{1}_{N_{\text {bins }}}
$$

where $\mathbf{1}_{N_{\text {bins }}}$ is a vector of ones of length $N_{\text {bins }}$.

For all experiments, we select $N_{\text {bins }}=100$ as the number of bins for the prior grid. See Fig. S1 for some examples of sampled priors.

Investigation of prior injection with a Gaussian toy model. To investigate the effect of the injected prior, we test our method with a simple 1D Gaussian model with two latent variables: mean $\mu$ and standard deviation $\sigma$. The data is the samples drawn from this distribution, $\mathcal{D}_{N}=\left\{y_{n}\right\}_{n=1}^{N} \sim \mathcal{N}\left(\mu, \sigma^{2}\right)$. We can numerically compute the exact Bayesian posterior on the predefined grid given the data and any prior, and subsequently compare the ground-truth posterior with the ACE's predicted posterior after injecting the same prior.

We first sample random priors using the generative process described above. Then we sample $\mu$ and $\sigma$ from the priors and generate the corresponding data $\mathcal{D}_{N}$. We pass the data along with the prior to ACE to get the predictive distributions $p\left(\mu \mid \mathcal{D}_{N}\right)$ and $p\left(\sigma \mid \mathcal{D}_{N}\right)$ as well as the autoregressive predictions $p\left(\mu \mid \sigma, \mathcal{D}_{N}\right)$ and $p\left(\sigma \mid \mu, \mathcal{D}_{N}\right)$. With these equations, we can autoregressively compute our model's prediction for $p\left(\mu, \sigma \mid \mathcal{D}_{N}\right)$ on the grid. The true posterior is calculated numerically via Bayes rule on the grid. See Fig. S2 for several examples.

---

#### Page 17

> **Image description.** The image is a grid of 25 plots, each displaying a curve on a graph. Each plot is labeled "Sample [number]" where the number ranges from 1 to 25.
>
> Each plot has the same basic structure:
>
> - A horizontal axis ranging from 0 to 2.
> - A vertical axis with varying scales, but all starting at 0 and reaching different maximum values (e.g., 0.015, 0.03, 0.100).
> - A blue curve plotted on the graph. The shapes of these curves vary significantly across the samples. Some curves are unimodal, resembling Gaussian distributions (e.g., Sample 2, Sample 3, Sample 7, Sample 25), while others are more complex, exhibiting multiple peaks (e.g., Sample 1, Sample 23) or skewed shapes (e.g., Sample 5, Sample 6, Sample 11, Sample 14). Some are nearly flat lines (e.g., Sample 17, Sample 20). Sample 9 has a very narrow peak.
> - The axes are black, and the curves are blue.

Figure S1: Examples of randomly sampled priors over the range $[0,2]$. Samples include mixtures of Gaussians and Uniform distributions.

To quantitatively assess the quality of our model's predicted posteriors, we compare the posterior mean and standard deviation (i.e., the first two moments ${ }^{3}$ ) for $\mu$ and $\sigma$ of predicted vs. true posteriors, visualized in Fig. S3. The scatter points are aligned along the diagonal line, indicating that the moments of the predicted posterior closely match the moments of true posterior. These results show that ACE is effectively incorporating the information provided by the prior and adjusts the final posterior accordingly. In Appendix C.4.4 we perform a more extensive analysis of posterior calibration in ACE with a complex simulator model.

[^0]
[^0]: ${ }^{3}$ We prefer standard deviation to variance as it has the same units as the quantity of interest, as opposed to squared units which are less interpretable.

---

#### Page 18

> **Image description.** The image is a figure composed of a 4x5 grid of plots. Each plot visualizes a probability distribution on a two-dimensional space. The horizontal axis of each plot is labeled "μ", and the vertical axis is labeled "σ". The color scheme uses a gradient from dark purple to yellow to represent increasing probability density. The plots are arranged in columns labeled "(a) Prior Distribution", "(b) Likelihood", "(c) True Posterior", and "(d) ACE Posterior". The rows show different examples or cases.
>
> In the first column, labeled "(a) Prior Distribution", the plots show:
>
> - A vertical band of high probability density.
> - A region of high probability density shaped like a bell curve.
> - A horizontal band of high probability density.
> - A narrow, horizontal ellipse of high probability density.
> - A narrow, vertical band of high probability density.
>
> In the second column, labeled "(b) Likelihood", the plots show:
>
> - A roughly elliptical region of high probability density.
> - A smaller, more compact elliptical region of high probability density.
> - A similar, but slightly more elongated elliptical region of high probability density.
> - A flattened elliptical region of high probability density.
> - A similar, but slightly more elongated elliptical region of high probability density.
>
> In the third column, labeled "(c) True Posterior", the plots show:
>
> - A roughly elliptical region of high probability density.
> - A smaller, more compact elliptical region of high probability density.
> - A similar, but slightly more elongated elliptical region of high probability density.
> - A flattened elliptical region of high probability density.
> - A narrow, vertical band of high probability density.
>
> In the fourth column, labeled "(d) ACE Posterior", the plots show:
>
> - A roughly elliptical region of high probability density.
> - A smaller, more compact elliptical region of high probability density.
> - A similar, but slightly more elongated elliptical region of high probability density.
> - A flattened elliptical region of high probability density.
> - A narrow, vertical band of high probability density.
>
> The plots in columns (c) and (d) are visually similar, suggesting that the "ACE Posterior" approximates the "True Posterior".

Figure S2: Examples of the true and predicted posterior distributions in the toy 1D Gaussian case. (a) Prior distribution over $\boldsymbol{\theta}=(\mu, \sigma)$ set at runtime. (b) Likelihood for the observed data (the data themselves are not shown). (c) Ground-truth Bayesian posterior. (d) ACE's predicted posterior, based on the user-set prior and observed data, approximates well the true posterior.

---

#### Page 19

> **Image description.** The image contains four scatter plots arranged in a 2x2 grid. Each plot compares predicted vs. true posterior values for either the mean (mu) or standard deviation (sigma).
>
> - **Plot 1 (Top Left):**
>
>   - Title: "Predicted vs True Posterior Mean (μ)"
>   - X-axis label: "Predicted Posterior Mean (μ)" with values ranging from -1 to 1.
>   - Y-axis label: "True Posterior Mean (μ)" with values ranging from -0.75 to 1.00.
>   - Data points: Blue dots scattered around a diagonal red dashed line.
>   - Text: "R2: 1.00" in the top left corner.
>
> - **Plot 2 (Top Right):**
>
>   - Title: "Predicted vs True Posterior Std (μ)"
>   - X-axis label: "Predicted Posterior Std (μ)" with values ranging from 0.00 to 0.30.
>   - Y-axis label: "True Posterior Std (μ)" with values ranging from 0.05 to 0.25.
>   - Data points: Blue dots scattered around a diagonal red dashed line.
>   - Text: "R2: 0.94" in the top left corner.
>
> - **Plot 3 (Bottom Left):**
>
>   - Title: "Predicted vs True Posterior Mean (σ)"
>   - X-axis label: "Predicted Posterior Mean (σ)" with values ranging from 0.0 to 1.0.
>   - Y-axis label: "True Posterior Mean (σ)" with values ranging from 0.2 to 0.8.
>   - Data points: Blue dots scattered around a diagonal red dashed line.
>   - Text: "R2: 1.00" in the top left corner.
>
> - **Plot 4 (Bottom Right):**
>   - Title: "Predicted vs True Posterior Std (σ)"
>   - X-axis label: "Predicted Posterior Std (σ)" with values ranging from 0.00 to 0.18.
>   - Y-axis label: "True Posterior Std (σ)" with values ranging from 0.025 to 0.175.
>   - Data points: Blue dots scattered around a diagonal red dashed line.
>   - Text: "R2: 0.89" in the top left corner.
>
> In all four plots, the diagonal red dashed line represents a perfect prediction where the predicted value equals the true value. The R-squared values indicate the goodness of fit, with values closer to 1 indicating a better fit.

Figure S3: The scatter plots compare the predicted and true posterior mean and standard deviation values for both $\mu$ and $\sigma$ across 100 examples. We can see that the points lie closely along the diagonal red dashed line, indicating that the moments (mean and standard deviation) of the predicted posteriors closely match the true posteriors.

---

#### Page 20

# B. 2 Architecture

Here we give an overview of two key architectures used in our paper. First, we show the TNP-D (Nguyen and Grover, 2022) architecture in Fig. S4, which our method extends. Fig. S5 shows the ACE architecture introduced in this paper.

> **Image description.** This is a diagram illustrating a conceptual architecture, likely for a neural network or machine learning model.
>
> The diagram is structured as a series of connected blocks, with arrows indicating the flow of data. The diagram contains the following elements:
>
> - **Input Blocks:** On the left side, there are six input blocks. The top three blocks are labeled "(x1, y1)", "(x3, y3)", and "(x5, y5)". These blocks are gray. The bottom three blocks are labeled "(x2)", "(x4)", and "(x6)". These blocks are also gray, but the labels are in red. Arrows point from each of these blocks to the "Embedder" block.
>
> - **Embedder Block:** A large, light purple rectangular block labeled "Embedder" is located in the center-left of the diagram.
>
> - **MHSA Block:** To the right of the Embedder, there is a block consisting of three stacked gray rectangular blocks labeled "(z1)", "(z3)", and "(z5)". This block is enclosed in a rounded purple rectangle labeled "MHSA" at the top.
>
> - **CA Block:** Below the MHSA block and aligned with the bottom three inputs, there is a light purple rectangular block labeled "CA" in red. The MHSA and CA blocks are enclosed in a larger purple rounded rectangle labeled "k-blocks" at the bottom.
>
> - **Head Block:** To the right of the MHSA/CA block, there is a light purple rectangular block labeled "Head".
>
> - **Output/Loss Block:** On the right side, there is a block enclosed in an orange rounded rectangle labeled "Loss" at the top. This block contains six smaller blocks. The top three blocks are white rectangles labeled "(ŷ2)", "(ŷ4)", and "(ŷ6)". The bottom three blocks are gray rectangles labeled "(y2)", "(y4)", and "(y6)" in red.
>
> Arrows connect the blocks in the following way:
>
> - Arrows from the input blocks to the Embedder block.
> - Arrows from the Embedder block to the MHSA block and the CA block.
> - An arrow from the MHSA block to the CA block.
> - Arrows from the MHSA block and CA block to the Head block.
> - Arrows from the Head block to the output/loss block.

Figure S4: A conceptual figure of TNP-D architecture. The TNP-D architecture can be summarized in the embedding layer, attention layers and output head. The $x$ denotes locations where the output is unknown (target inputs). The $z$ is the embedded data, while MHSA stands for multi head cross attention and CA for cross attention. The head for TNP-D is Gaussian, so it outputs a mean and variance for each target point.

> **Image description.** A diagram illustrates the ACE architecture. The diagram is composed of several blocks and arrows, representing data flow and processing steps.
>
> - **Input Blocks (Left Side):** There are six rectangular blocks on the left side. From top to bottom, they contain the following text: "($\theta_1$)" in a green box, "(x3, y3)" in a gray box, "(x5, y5)" in a gray box, "(?$\theta_2$)" in a green box, "(x4)" in a red box, and "(x6)" in a dark-gray box. Each block is connected to a central block labeled "Embedder" by a black arrow pointing right.
>
> - **Embedder Block (Center-Left):** A large, light-red rectangular block is labeled "Embedder" in gray text. This block receives input from the blocks on the left and outputs to blocks on the right.
>
> - **MHSA and CA Blocks (Center-Right):** To the right of the "Embedder" block, there's a blue-outlined rounded rectangle labeled "MHSA" in purple. Inside this rounded rectangle are three gray rectangular blocks labeled "(z1)", "(z3)", and "(z5)". Below these is a purple arrow pointing down to another light-red rounded rectangle labeled "CA" in purple. Below the rounded rectangle is the text "k-blocks" in purple. To the left of the "CA" block are three gray rectangular blocks labeled "(z2)" in red, "(z4)" in dark-gray, and "(z6)" in dark-gray.
>
> - **Head Block (Center-Right):** To the right of the "CA" block is another light-red rectangular block labeled "Head (GMM or Cat)" in gray text.
>
> - **Output Blocks and Loss (Right Side):** To the right of the "Head" block, there's an orange-outlined rounded rectangle labeled "Loss" in orange. Inside this rounded rectangle are six blocks arranged in two columns. The left column contains white blocks with the text "($\hat{\theta}_2$)", "($\hat{y}_4$)", and "($\hat{y}_6$)". The right column contains a green block with the text "($\theta_2$)", and two gray blocks with the text "(y4)" and "(y6)".
>
> - **Arrows:** Black arrows connect each block, indicating the flow of data through the architecture.

Figure S5: A conceptual figure of ACE architecture. The diagram shows key differences between ACE and TNP-D. The differences boil down to the embedder layer that now incorporates latents $\theta_{l}$ (and possibly priors over these) and the output head that is now a Gaussian mixture model (GMM, for continuous variables) or categorical (Cat, for discrete variables). Both latent and data can be of either type.

---

#### Page 21

# B. 3 Training batch construction

ACE can condition on and predict data, latent variables, and combinations of both. Here, we outline the sampling process used to construct the training batch.

- First, we generate our dataset by following the steps outlined for the respective cases (GP, Image Completion, BO, SBI); see Appendix C. For example, in the GP emulation case, we draw $n_{\text {data }}$ points from a function sampled from a GP along with its respective latent variables.
- Next, we sample the number of context points, $n_{\text {context }}$, uniformly between the minimum and maximum context points, min*ctx and max_ctx. We then split our data based on this $n*{\text {context }}$ value; the remaining data points that are not in the context set are allocated to the target set.
- We then determine whether the context includes any latent variables at all with a $50 \%$ probability. If latent variables are to be included in the context set, we sample the number of latents residing in the context set, uniformly from 1 to $n_{\text {latents }}$. All latent variables not in the context set are assigned to the target set.

The above steps are applied for each element (dataset) in the training batch. In the implementation, we also ensure that, within each batch, the number of context points remains consistent across all elements, as does the number of target points, to facilitate batch training for our model. However, the number of latents in the context set may vary for each element, introducing variability that improves the model's training process.

## B. 4 Autoregressive predictions

While ACE predicts conditional marginals independently, we can still obtain joint predictions over both data and latents autoregressively (Nguyen and Grover, 2022; Bruinsma et al., 2023). Suppose we want to predict the joint target distribution $p\left(\mathbf{z}_{1: M}^{\star} \mid \boldsymbol{\xi}_{1: M}^{\star}, \mathfrak{D}_{N}\right)$, where we use compact indexing notation. We can write:

$$
p\left(\mathbf{z}_{1: M}^{\star} \mid \boldsymbol{\xi}_{1: M}^{\star}, \mathfrak{D}_{N}\right)=\prod_{m=1}^{M} p\left(z_{m}^{\star} \mid \mathbf{z}_{1: m-1}^{\star}, \boldsymbol{\xi}_{1: m}^{\star}, \mathfrak{D}_{N}\right)=\mathbb{E}_{\boldsymbol{\pi}}\left[\prod_{m=1}^{M} p\left(z_{\pi_{m}}^{\star} \mid \mathbf{z}_{\pi_{1}: \pi_{m-1}}^{\star}, \boldsymbol{\xi}_{\pi_{1}: \pi_{m}}^{\star}, \mathfrak{D}_{N}\right)\right]
$$

where $\boldsymbol{\pi}$ is a permutation of $(1, \ldots, M)$, i.e. an element of the symmetric group $\mathcal{S}_{M}$. The first passage follows from the standard rules of probability and the second passage follows from permutation invariance of the joint distribution with respect to the ordering of the variables $\boldsymbol{\xi}_{1: M}$. The last expression can be used to enforce permutation invariance and validity of our joint predictions even if sequential predictions of the model are not natively invariant (Murphy et al., 2019). In practice, for moderate to large $M(M \gtrsim 4)$ we approximate the expectation over permutations via Monte Carlo sampling.

## C EXPERIMENTAL DETAILS

In this section, we show additional experiments to validate our method and provide additional details about sampling, training, and model architecture.

## C. 1 Gaussian process (GP) experiments

We now demonstrate the use of ACE for performing amortized inference tasks in the Gaussian processes (GP) model class. GPs are a Bayesian non-parametric method used as priors over functions (see Rasmussen and Williams, 2006). To perform inference with a GP, one must first define a kernel function $\kappa_{\boldsymbol{\theta}}$ parameterized by hyperparameters $\boldsymbol{\theta}$ such as lengthscales and output scale. As a flexible model of distributions over functions used for regression and classification, GPs are a go-to generative model for meta-learning and feature heavily in the (conditional) neural process literature (CNP; Garnelo et al., 2018b). ACE can handle many of the challenges faced when applying GPs. Firstly, it can accurately amortize the GP predictive distribution as is usually shown in the CNP literature. In addition, ACE can perform other crucial tasks in the GP workflow, such as amortized learning of $\boldsymbol{\theta}$, usually found through optimization of the marginal likelihood (Gaussian likelihood) or via approximate inference for non-Gaussian likelihoods (e.g., Hensman et al. 2015). Furthermore, we can also do kernel selection by treating the kernel as a latent discrete variable, and incorporate prior knowledge about $\boldsymbol{\theta}$.

---

#### Page 22

> **Image description.** The image contains three line graphs, labeled (a), (b), and (c) respectively. Each graph has a horizontal axis labeled "Size of $\mathcal{D}_N$" ranging from approximately 0 to 20.
>
> Graph (a) displays "Log predictive density $p(y|\cdot)$" on the vertical axis, ranging from -1 to 2. Three lines are plotted: a dashed orange line labeled "$p(y|\mathcal{D}_N)$", a solid green line labeled "$p(y|\boldsymbol{\theta}, \mathcal{D}_N)$", and a solid blue line labeled "GP predictive". Each line has a shaded area around it, indicating a confidence interval.
>
> Graph (b) displays "Kernel identification accuracy" on the vertical axis, ranging from 0.4 to 1. A single solid orange line is plotted with a shaded area around it.
>
> Graph (c) displays "Log predictive density $p(\theta|\mathcal{D}_N)$" on the vertical axis, ranging from 0 to 0.6. A single solid orange line is plotted with a shaded area around it.
>
> All three graphs have a light gray grid in the background.

Figure S6: (a) Conditioning on the latent variable $\boldsymbol{\theta}$ (kernel hyperparameters and type) improves predictive performance, approaching the GP upper bound for the log predictive density. (b) ACE can identify the kernel $\kappa$. (c) ACE can learn kernel hyperparameters.

Results. The main results from our GP regression experiments are displayed in Fig. S6. We trained ACE on samples from four kernels, the RBF and Matérn- $(1 / 2,3 / 2,5 / 2)$, using the architecture described in Section 3; see below for details. In Fig. S6a, we show ACE's ability to condition on provided information: data only, or data and $\boldsymbol{\theta}$ (kernel hyperparameters and type). As expected, there is an improvement when conditioning on more information, specifically when the context set $\mathcal{D}_{N}$ is small. As an upper bound, we show the ground-truth GP predictive performance. The method can accurately predict the kernel, i.e. model selection (Fig. S6b), while at the same time learn the hyperparameters (Fig. S6c), both improving as a function of the context set size.

Sampling from a GP. Both the GP experiments and the Bayesian optimization experiments reported in Section 4.2 and further detailed in Appendix C. 3 use a similar sampling process to generate data.

- We first sample the latents. These are kernel hyperparameters, the output scale $\sigma_{f}$ and lengthscale $\ell$. Each input dimension of $\mathbf{x}$ is assigned its own lengthscale $\ell=\left(\ell^{(1)}, \ell^{(2)}, \ldots\right)$ and a corresponding kernel. For all GP examples the RBF and three Matérn- $(1 / 2,3 / 2,5 / 2)$ kernels were used with equal weights. The kernel output scale $\sigma_{f} \sim U(0.1,1)$ and each $k$-th lengthscale is $\ell^{(k)} \sim \mathcal{N}(1 / 3,0.75)$ truncated between $[0.05,2]$.
- Once all latent information is defined, we draw from a GP prior from a range $[-1,1]$ for each input dimension. The realizations from the prior form our context data $\mathcal{D}_{N}$ where the size of the context set $N$ is drawn from a discrete uniform distribution $3,4, \ldots .50$. The target data $\left(\mathbf{X}^{*}, \mathbf{y}^{*}\right)$ of size $200-N$ is then drawn from the predictive posterior of the GP conditioned on $\mathcal{D}_{N}$.

Architecture. The ACE model used in the GP experiments had embedding dimension 256 and 6 transformer layers. The attention blocks had 6 heads and the MLP block had hidden dimension 128. The output head had $K=20$ MLP components with hidden dimension 256. The model was trained for $5 \times 10^{4}$ steps with batch size 32 , using learning rate $1 \times 10^{-4}$ with cosine annealing. Following Nguyen and Grover (2022), and unlike the original transformer implementation (Vaswani et al., 2017), we do not use dropout in any of our experiments.

# C. 2 Image completion and classification

In this section, we detail the image experiments in Section 4.1 as well as report additional experiments. Image completion experiments have long been a benchmark in the neural process literature treating them as regression problems (Garnelo et al., 2018a; Kim et al., 2019). We use two standard datasets, MNIST (Deng, 2012) and CelebA (Liu et al., 2015). The MNIST results presented are with the full image size $28 \times 28$, while CelebA results were downsized to $32 \times 32$. However, as shown in Fig. S11, ACE can also handle the full image size. All image datasets were normalised based on the complete dataset average and standard deviation. The data input $\mathbf{x}$ for image experiments is the 2-D image pixel-coordinates and the data value $y$ for MNIST is one output dimension, while CelebA uses all three RGB channels and thus is a multi-output $\mathbf{y}$.

The experiments on images demonstrate the versatility of the ACE method and its advantages over conventional

---

#### Page 23

> **Image description.** This image shows a comparison of image completion results for the MNIST dataset, displayed in a 2x5 grid. Each row represents a different digit (9 and 7), and each column represents a different stage or method in the image completion process.
>
> - **Column 1 (a) Image:** Shows the original, complete images of the digits '9' and '7' on a black background. The digits are white.
> - **Column 2 (b) $\mathcal{D}_{N}$:** Shows the context images. These images have a blue background with a sparse scattering of small black and white squares, representing the observed pixels.
> - **Column 3 (c) TNPD:** Shows the image completion results using the TNPD method. The digits are blurry and grayscale, with a black background containing scattered blue dots. The observed pixels are marked with small blue squares.
> - **Column 4 (d) ACE:** Shows the image completion results using the ACE method. The digits are clearer than in the TNPD results, with a black background containing scattered blue dots. The observed pixels are marked with small blue squares.
> - **Column 5 (e) ACE- $\boldsymbol{\theta}$:** Shows the image completion results using the ACE method conditioned on the class label. The digits are the clearest and most similar to the original images, with a black background containing scattered blue dots. The observed pixels are marked with small blue squares.
>
> Below each column is a label indicating the stage or method: (a) Image, (b) $\mathcal{D}_{N}$, (c) TNPD, (d) ACE, (e) ACE- $\boldsymbol{\theta}$.

(a) Image
(b) $\mathcal{D}_{N}$
(c) TNPD
(d) ACE
(e) ACE- $\boldsymbol{\theta}$

> **Image description.** The image is a line graph comparing the performance of three different models: ACE, ACE-θ, and TNPD.
>
> - **Axes:** The x-axis ranges from 0 to 20, with tick marks at intervals of 10. The y-axis ranges from -1.2 to -0.6, with tick marks at intervals of 0.2.
> - **Data:**
>   - ACE is represented by a blue line with circular data points. A light blue shaded region surrounds the line, indicating a confidence interval.
>   - ACE-θ is represented by an orange line with circular data points. A light orange shaded region surrounds the line, indicating a confidence interval.
>   - TNPD is represented by a green line with circular data points. A light green shaded region surrounds the line, indicating a confidence interval.
> - **Legend:** A legend in the upper right corner identifies each model with its corresponding color and label: ACE (blue), ACE-θ (orange), and TNPD (green).
> - **Overall Trend:** All three models show a general downward trend, indicating improving performance (lower y-axis values) as the x-axis value increases. The TNPD model starts with a higher (worse) value but decreases more rapidly than the other two.

(f) NLPD v Context(\% of image)

Figure S7: Image regression (MNIST). Image (a) serves as the reference for the problem, while (b) is the context where $10 \%$ of the pixels are observed. Figures (c) - (e) are the respective model predictions, while (f) shows performance over varying context (mean and $95 \%$ confidence interval). In (e) the model is also conditioned on the class label, showing a clear improvement in performance.

CNPs. We outperform the current state-of-the-art TNP-D on the standard image completion task (Fig. 3). Given a random sample from the image space as context $\mathcal{D}_{N}$, the model predicts the remaining $M$ image pixel values at $\mathbf{x}^{*}$. The total number of points $N+M$ for MNIST is thus 784 points and 1024 for CelebA where the split is randomly sampled (see below for details). The model is then trained as detailed in Section 3.3. Thus, the final trained model can perform image completion, also sometimes known as in-painting.

In addition to image completion, our method can condition on and predict latent variables $\boldsymbol{\theta}$. For MNIST, we use the class labels as latents, so 0 , $1,2, \ldots$, which were encoded into a single discrete variable. Meanwhile, for CelebA we use as latents the 40 binary features that accompany the dataset, e.g. BrownHair, Man, Smiling, trained with the sampling procedure discussed below. We recall that in ACE the latents $\boldsymbol{\theta}$ can be both conditioned on and predicted. Thus, we can do conditional generation based on the class or features or, given a sample of an image, predict its class or features, as initially promised in Fig. 1a.

> **Image description.** The image is a line graph showing the relationship between "Context Size %" and "Classification Accuracy".
>
> The x-axis is labeled "Context Size %" and ranges from 0 to 100. The y-axis is labeled "Classification Accuracy" and ranges from 0 to 1. The graph contains a blue line with circular data points, illustrating the trend. A light blue shaded area surrounds the line, likely representing a confidence interval. The line starts at a low classification accuracy for small context sizes and rapidly increases until it reaches a context size of around 40%. After that, the classification accuracy plateaus near 1, with a slight dip towards the end. A grid of light gray lines is present in the background.

Figure S8: Classification accuracy for MNIST for varying context size.

Results. The main image completion results for the CelebA dataset are shown in Fig. 3, with the same experiment performed on MNIST and displayed in Fig. S7. In both figures, we display some example images and predictions and negative log-probability density for different context sizes (shaded area is $95 \%$ confidence interval). Our method demonstrates a clear improvement over the TNP-D method across all context sizes on both datasets (Fig. 3 and Fig. S7). Moreover, incorporating latent information for conditional generation further enhances the performance of our base method. A variation of the image completion experiment is shown in Fig. S9, where the context is no longer randomly sampled from within the image but instead selected according

> **Image description.** The image shows a blurry, low-resolution depiction of a person's head and shoulders, obscured by a large, bright green rectangle that covers the upper portion of the head. The area below the green rectangle shows the person's face, neck, and shoulders, although the details are indistinct due to the image quality. The person's skin appears to have a yellowish tint. The shoulders are dark, possibly indicating clothing. The background is white.

(a) Context

> **Image description.** The image shows a close-up photograph of a person's face. The person appears to be an older man with fair skin and a white beard. The image quality is somewhat low-resolution, resulting in a slightly blurred appearance. The background is plain white, which isolates the face as the primary subject. The man is facing forward, and his expression is neutral. He is wearing a dark-colored garment, possibly a shirt or jacket, but the details are not clear due to the image resolution.

(b) $\mathrm{BALD}=$ True

> **Image description.** The image shows a set of five images, arranged horizontally, each depicting a face.
>
> - **(a) Image:** The first image shows a blurry, low-resolution color image of a man's face. The man has light skin, dark hair, and is wearing a black jacket or shirt. The background is white.
>
> - **(b) $\mathcal{D}_{N}$:** The second image shows the same blurry face as in (a), but with a vertical green bar covering approximately 1/3 of the right side of the image. The rest of the image is visible.
>
> - **(c) TNPD:** The third image shows a blurry, low-resolution color image of a man's face. The face is similar to the one in (a) but slightly different, indicating a reconstruction or prediction.
>
> - **(d) ACE:** The fourth image shows another blurry, low-resolution color image of a man's face. This face is also similar to the one in (a) but has distinct differences compared to (c), suggesting a different reconstruction method.
>
> - **(e) ACE- $\boldsymbol{\theta}$:** The fifth image shows a blurry, low-resolution color image of a man's face. This face appears to be the clearest and most similar to the original image (a) compared to (c) and (d).

(c) $\mathrm{BALD}=\mathrm{False}$

> **Image description.** The image contains six panels, labeled (a) through (f), related to image regression on MNIST.
>
> Panel (a) shows a blurred image of what appears to be a handwritten digit.
>
> Panel (b) shows the same image as (a), but with the top portion covered by a solid green rectangle.
>
> Panels (c), (d), and (e) each show a blurred image, presumably reconstructions of the original digit. Panel (e) appears slightly clearer than (c) and (d). The colors in all three panels are predominantly blue and red.
>
> Panel (f) displays a line graph. The x-axis is labeled "Context (% of image)". The y-axis is not explicitly labeled but represents performance. A blue line with a shaded area around it represents the mean and 95% confidence interval. The line increases as the context increases.

(d) Context

> **Image description.** The image shows two blurred portraits of a person against a blue background.
>
> The left portrait shows a person with brown skin and a bald head. The right portrait shows a person with dark skin and dark hair. Both portraits are blurred, making it difficult to discern specific facial features. The background in both images is a gradient of blue shades, darker at the top and lighter at the bottom.

(e) $\mathrm{BALD}=\mathrm{True}$

> **Image description.** The image contains two blurry images of faces.
>
> The first image shows a person with dark hair and skin, set against a blue background. The image is blurry, making it difficult to discern specific facial features.
>
> The second image is also blurry, depicting a person with dark hair and skin against a blue background. A vertical black bar obscures a portion of the left side of the image. Like the first image, the blurriness makes it hard to identify distinct facial features.

(f) $\mathrm{BALD}=\mathrm{False}$

Figure S9: Example of ACE conditioning on the value of the BALD feature when the image is masked for the first 22 rows. (a) and (d) show the context points used for prediction, where (b) and (e) show predictions where the Bald feature is conditioned on True. Meanwhile, c and f are conditioned on False.

---

#### Page 24

to a top 22-row image mask. For this example, the latent information BALD is either conditioned on True or False. The results show that the model adjusts its generated output based on the provided latent information, highlighting the potential of conditional generation. Furthermore, in Fig. S10, we show examples of ACE's ability to perform image classification showing a subset of the 40 features in CelebA dataset. Despite only having $10 \%$ of the image available, ACE can predict most features successfully. Finally, in Fig. S8 the accuracy for predicting the correct class label for MNIST is reported.

> **Image description.** The image is a figure composed of three panels arranged horizontally. The figure shows the classification ability of an AI model.
>
> Panel (a), labeled "Context", shows a pixelated image with a predominantly green background. Scattered throughout are small squares of various colors including white, black, red, and gray. This represents the available context or partial information.
>
> Panel (b), labeled "Full image", presents a pixelated image of a face. The top image shows a light-skinned individual with short hair, while the bottom image shows a darker-haired individual. These are the full images that the model is trying to classify, given the context in panel (a).
>
> Panel (c), labeled "Classification probability for some features", displays two horizontal bar charts stacked vertically. Each chart represents the classification probabilities for a subset of features related to the face images. The y-axis lists features such as "Bald", "Gray_Hair", "Smiling", "Black_Hair", "Big_Lips", "Wearing_Necktie", "Male", "Bangs", "Young", and "No_Beard". The x-axis represents probability, ranging from 0 to 1. For each feature, there is a blue horizontal bar indicating the predicted probability, a black dot representing the average probability, and either a red asterisk (\*) or a black cross (x) indicating whether the true label for that feature is 1 or 0, respectively. A vertical dashed red line is at the 0.5 probability mark.

Figure S10: An example showing the classification ability of ACE. (a) is the context available of the full image displayed in the panel (b). The probabilities for a subset of features are in (c).

Sampling for Image experiments. For sampling, we use the full available dataset for both MNIST and CelebA, detailed in Appendix B.3. In the MNIST dataset there is one latent class label and for CelebA all 40 features were used. In Fig. S9, the sampling procedure was adjusted to represent features that would influence the top 22 rows of the images. Therefore, we selected a subset of seven features, which were BALD, BANGS, Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair and Eyeglasses. The same sampling procedure was performed again but, now on a smaller set of features.

Architecture and training. For the image experiments, we used the same embedder layer as in the other experiments. Through grid search, we found that a transformer architecture with 8 heads, 6 layers, and a hidden dimension of 128 performed best. For the MLP layer, we used a dimension of 256 . Finally, we reduced the number of components for the output head to $K=3$. We trained the model for 80,000 iterations using a cosine scheduler with Adam (Kingma and Ba, 2015), with a learning rate 0.0005 and a batch size of 64 .

# C. 3 Bayesian optimization

This section presents ACE's Bayesian Optimization (BO) experiments (Section 4.2 in the main paper) in more detail, including the training data generation, algorithms, benchmark functions, and baselines used in this paper.

## C.3.1 Bayesian Optimization dataset, architecture and training details

Dataset. The BO datasets are generated by sampling from a GP, following the approach described in Appendix C.1. The sampling procedure is adjusted to include the known optimum location and value of the function within the generative process. The detailed dataset generation procedure is outlined as follows:

1. Sampling GP hyper-parameters, to determine the base function shape:

---

#### Page 25

> **Image description.** The image presents a comparative visual analysis of image reconstruction, arranged in a 2x4 grid. The first row depicts a woman's face, while the second row shows a man wearing a baseball cap.
>
> - **Column (a) Context:** The first column displays a square image filled with a seemingly random arrangement of colored pixels. The dominant color is green, with scattered pixels of red, white, blue, and black. This likely represents the "context" data used for reconstruction.
>
> - **Column (b) Image:** The second column shows the original, low-resolution images. The top image is a woman in profile, facing right. She has dark hair and fair skin. The bottom image is a man wearing a red baseball cap and sunglasses. The background appears to be blurry and blue.
>
> - **Column (c) ACE:** The third column displays the images reconstructed using the "ACE" method. These images are slightly clearer than the original images in column (b), but still retain a pixelated appearance.
>
> - **Column (d) ACE-θ:** The fourth column shows the images reconstructed using the "ACE-θ" method. These images appear to be the clearest and most detailed of the three, with smoother edges and more defined features.
>
> Below each column is a label: "(a) Context", "(b) Image", "(c) ACE", and "(d) ACE-θ".

Figure S11: Examples of ACE on 64x64 image size.

- First, we randomly select a kernel from a set comprising the RBF kernel and three Matérn kernels (Matérn$1 / 2$, Matérn-3/2, and Matérn-5/2) based on predefined weights $[0.35,0.1,0.2,0.35]$, corresponding respectively to the RBF kernel and the Matérn kernels in the specified order.
- Then, we sample whether the kernel is isotropic or not with $p=0.5$.
- The output scale $\sigma_{f}$ and lengthscales $l^{(k)}$ are sampled following the procedure outlined in Appendix C.1.
- We assume the GP (constant) mean to be 0 for now.

2. Sampling the latent values, $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$ :

- We sample the optimum location $\mathbf{x}_{\text {opt }}$ uniformly inside $[-1,1]^{D}$.
- We sample the value of the global minimum $y_{\text {opt }}$ from a minimum-value distribution of a zero-mean Gaussian with variance equal to the output variance of the GP. The number of samples for the minimumvalue distribution, $N$, is approximated as the number of uncorrelated samples from the GP in the hypercube, determined based on the GP's length scale. This approach ensures that $y_{\text {opt }}$ roughly respects the statistics of optima for the GP hyperparameters.
- With $p=0.1$ probability, we add $\Delta y \sim \exp (1)$ to the mean function to represent an "unexpectedly low" optimum.

3. Sampling from GP posterior to get the context and target sets:

- We build a posterior GP with the above specification and a single observation at $\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }}\right)$.
- We sample a total of $100 \cdot D$ (context + target) locations where the number of context points is sampled similarly to the GP dataset generation. The maximum number of context points is 50 for the 1D case and 100 for both 2 D and 3 D cases.
- Then, the values of this context set are jointly sampled from a GP posterior conditioned on one observation at $\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }}\right)$.
- Instead, the target points are sampled independently from a GP posterior conditioned on $\mathcal{D}_{N}$ (the previously sampled context points) and $\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }}\right)$. Independent sampling of the targets speeds up GP data generation and is valid since during training we only predict 1D marginal distributions at the target points.

4. Further adjustment of $y$, and consequently $y_{\text {opt }}$ :

- To ensure that the global optimum is at $\left(\mathbf{x}_{\text {opt }}, y_{\text {opt }}\right)$ we add a convex envelope (a quadratic component). Specifically, we transform the $y$ values of the datasets as $y_{i}^{\prime}=\left|y_{i}\right|+\frac{1}{5}\left\|\mathbf{x}_{\text {opt }}-\mathbf{x}_{i}\right\|^{2}$ where $\mathbf{x}_{i}$ and $y_{i}$ are the input and output values of all sampled points.

---

#### Page 26

> **Image description.** This image presents a grid of 16 line graphs, each displaying a different sample function. Each graph is contained within its own subplot, arranged in a 4x4 grid.
>
> Each subplot features:
>
> - A line graph plotted with a teal/cyan colored line. The shapes of these lines vary across the subplots, showing different function behaviors. Some lines are smooth and curved, while others are jagged and erratic.
> - A red dot marking a specific point on the line, presumably indicating the global optimum.
> - X and Y axes. The X-axis ranges from -1.0 to 1.0 with tick marks at -1.0, -0.5, 0.0, 0.5, and 1.0. The Y-axis ranges vary across the subplots to accommodate the range of values for each sample function.
> - A title above each graph, labeled "Sample 1" through "Sample 16".
> - The X-axis is labeled 'x' at the bottom right of the entire grid. The Y-axis is labeled 'y' on the left side of the grid.
>
> The overall visual impression is that of a collection of diverse function samples, each with a marked global optimum.

Figure S12: One-dimensional Bayesian optimization dataset samples, with their global optimum (red dot).

- Lastly, we add an offset to the $y^{\prime}$ values of sampled points uniformly drawn from $[-5,5]$, meaning that $y_{\text {opt }} \in[-5,5]$.

One and two-dimensional examples of the sampled functions are illustrated in Fig. S12 and Fig. S13, respectively.
Architecture and training details. In the Bayesian Optimization (BO) experiments, the ACE model was configured differently depending on the dimensionality of the problem. For the 1-3D cases, the model used an embedding dimension of $D_{\text {emb }}=256$ with six transformer layers. Each attention block had 16 heads, while the MLP block had a hidden dimension of 128 . The output head consisted of $K=20 \mathrm{MLP}$ components, each with a hidden dimension of 128 . For the 4-6D cases, the model was configured with embedding dimension of $D_{\text {emb }}=128$ while still using six transformer layers. Each attention block had 8 heads, and the MLP block had a hidden dimension of 512 . The output head structure remained unchanged, consisting of $K=20 \mathrm{MLP}$ components, each with a hidden dimension of 128 . The model configuration varied with problem dimensionality to balance capacity and efficiency.

The model was trained for $5 \times 10^{5}$ steps with a batch size of 64 for 1-3D cases and $3.5 \times 10^{5}$ steps and 128 batch size for 4-6D cases, using learning rate $5 \times 10^{-4}$ with cosine annealing. We apply loss weighing to give more

---

#### Page 27

> **Image description.** The image presents a collection of nine contour plots, arranged in a 3x3 grid. Each plot is labeled "Sample [number]" from 1 to 9. The plots visualize a two-dimensional function, with the x-axis labeled as "X1" and the y-axis as "X2" on the left side of the middle row.
>
> Each subplot displays a contour plot with the x and y axes ranging from -1 to 1. The contours are colored according to a color scale shown on the right side of each plot, with yellow indicating higher values and purple indicating lower values. A red dot is present in each plot, indicating a specific point on the contour map. The location of the red dot varies across the different samples.
>
> The contour patterns differ significantly between the samples. Some samples (e.g., Sample 1, Sample 2, Sample 5, Sample 8, and Sample 9) show concentric or nested contours, suggesting a local minimum or maximum. Other samples (e.g., Sample 3, Sample 6, and Sample 7) exhibit more complex, elongated, or striped patterns. Sample 4 shows a more scattered contour pattern.

Figure S13: Two-dimensional Bayesian optimization dataset samples, with their optimum (red dot).
importance to the latent variables during training. This adjustment accounts for the fact that the number of latent variables, $n_{\text {latent }}$, is generally much smaller than the number of data points, $\left(n_{\text {total }}-n_{\text {latent }}\right)$. The weight assigned to the latent loss is calculated as $w_{\text {latent }}=\left(n_{\text {total }}-1 / 2\left(\max \_c t x+\min \_c t x\right) / n_{\text {latent }}\right)^{T}$ where $T$ is a tunable parameter, max_ctx and max_ctx are the maximum and minimum number of context points during the dataset generation. We conducted a grid search over $T=1,2 / 3,1 / 3,0$ to identify the best-performing model. In our experiments, the optimal $T$ values are $T=1$ for $1 \mathrm{D}, T=2 / 3$ for 2 D and 3 D , and $T=0$ for $4 \mathrm{D}-6 \mathrm{D}$. Note that ACE has different models trained with different datasets for each input dimensionality.

# C.3.2 ACE-BO Algorithm

Bayesian optimization with Thompson sampling (ACE-TS). For Thompson sampling, we sample the query point at each step from $p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$ where $\tau$ is a threshold lower than the minimum point seen so far. This encourages exploration to sample a new point that is lower than the current optimum. We set $\tau=y_{\min }-\alpha \max \left(1, y_{\max }-y_{\min }\right)$, where $y_{\max }$ and $y_{\min }$ are the maximum and minimum values currently observed so far, and $\alpha$ a parameter controlling the minimum improvement. We set $\alpha=0.01$ throughout all experiments. First, we sample $y_{\text {opt }}$ from a truncated mixture of Gaussian obtained from ACE's predictive distribution $p\left(y_{\text {opt }} \mid \mathcal{D}_{N}\right)$, truncated for $y_{\text {opt }}<\tau$. After that, we sample $\mathbf{x}_{\text {opt }}$ conditioned on that sampled $y_{\text {opt }}$ (i.e., sample from $p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$ ). For higher dimension $(D>1)$ we sample $\mathbf{x}_{\text {opt }}$ in an autoregressive manner, one dimension at a time. The order of the dimensions is randomly permuted to mitigate order bias among the dimensions. The detailed pseudocode for ACE-TS ( $\mathrm{D}>1$ ) is presented in Algorithm Algorithm 1. An example evolution of ACE-TS is reported in Fig. S14.

---

#### Page 28

# Algorithm 1 ACE-Thompson Sample ( $\mathrm{D}>1$ )

Input: observed data points $\mathcal{D}_{N}=\left\{\mathbf{x}_{1: N}, y_{1: N}\right\}$, improvement parameter $\alpha$, input dimensionality $D \in \mathbb{N}^{+}$, whether to condition on $y_{\text {opt }}$ or not flag $c \in\{$ True, False $\}$.
Initialization $y_{\min } \leftarrow \min y_{1: N}, y_{\max } \leftarrow \max y_{1: N}$.
if $c$ is True then
set threshold value: $\tau \leftarrow y_{\min }-\alpha \max \left(1, y_{\max }-y_{\min }\right)$.
sample $y_{\text {opt }}$ from mixture truncated at $\tau: y_{\text {opt }} \sim p\left(y_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$.
end if
randomly permute dimension indices: $(1, \ldots, D) \rightarrow\left(\pi_{1}, \ldots, \pi_{D}\right) . \quad \triangleright \pi$ is permutation of $(1, \ldots, D)$
for $i \leftarrow \pi_{1}, \ldots, \pi_{D}$ do
if $c$ is True then
sample $x_{\text {opt }}^{i}$ conditioned on $y_{\text {opt }}, \mathcal{D}_{N}$, and already sampled $\mathbf{x}_{\text {opt }}$ dimensions if any:
$x_{\text {opt }}^{i} \sim p\left(x_{\text {opt }}^{i} \mid \mathcal{D}_{N}, y_{\text {opt }}, x_{\text {opt }}^{l(i-1)}\right)$.
else
sample $x_{\text {opt }}^{i}$ conditioned on $\mathcal{D}_{N}$ and already sampled $\mathbf{x}_{\text {opt }}$ dimensions if any:
$x_{\text {opt }}^{i} \sim p\left(x_{\text {opt }}^{i} \mid \mathcal{D}_{N}, x_{\text {opt }}^{l(i-1)}\right)$.
end if
end for
get full value of $\mathbf{x}_{\text {opt }}$ using the true indices: $\mathbf{x}_{\text {opt }} \leftarrow\left(x_{\text {opt }}^{1}, \ldots, x_{\text {opt }}^{D}\right)$.
return $\mathbf{x}_{\text {opt }}$

## Algorithm 2 ACE-MES

Input: observed data points $\mathcal{D}_{N}=\left\{x_{1: N}, y_{1: N}\right\}$, number of candidate points $N_{\text {cand }}$, Thompson sampling ratio for candidate point $T S_{\text {ratio }}$.

1: Initialization $N_{T S 1} \leftarrow N_{\text {cand }} \times T S_{\text {ratio }}, N_{T S 2} \leftarrow N_{\text {cand }} \times\left(1-T S_{\text {ratio }}\right)$.
2: propose $N_{\text {cand }}$ candidate points $X_{1: N_{\text {cand }}}^{*}$ according to $T S_{\text {ratio }}$ :
3: sample $X_{1: N_{T S 1}}^{*}$ using ACE-TS with conditioning on $y_{\text {opt }}(c=$ True $)$.
4: sample $X_{N_{T S 1}+1: N_{T S 1}+N_{T S 2}}^{*}$ using ACE-TS without conditioning on $y_{\text {opt }}(c=$ True $)$.
5: for $i \leftarrow 1$ to $N_{\text {cand }}$ do:
6: $\quad$ sample $y_{\text {opt }}$ for conditioning: $y_{\text {opt }} \sim p\left(y_{\text {opt }} \mid \mathcal{D}_{N}\right)$.
7: $\quad \alpha_{(i)}\left(\mathbf{x}_{(i)}^{*}\right)=H\left[p\left(y_{(i)}^{*} \mid \mathbf{x}_{(i)}^{*}, \mathcal{D}_{N}\right)\right]-\mathbb{E}\left(H\left[p\left(y_{(i)}^{*} \mid \mathbf{x}_{(i)}^{*}, \mathcal{D}_{N}, y_{\text {opt }}\right)\right]\right) \quad \triangleright$ see Appendix C.3.2 for more detail
8: end for
9: $\mathbf{x}_{\text {opt }}=\arg \max \boldsymbol{\alpha}$.
10: return $\mathbf{x}_{\text {opt }}$

---

#### Page 29

> **Image description.** The image shows three panels, each depicting a Bayesian optimization process at different iterations. Each panel contains a plot with an x-axis labeled "x" ranging from -1.0 to 1.0 and a y-axis labeled "y".
>
> Each panel includes the following visual elements:
>
> - A dashed gray line represents an underlying function.
> - Black dots mark observed data points. In the second and third panels, there are blue dots as well.
> - A dotted line, surrounded by a shaded purple area, represents the model's prediction and uncertainty.
> - A red asterisk indicates the queried point at each iteration.
> - An orange probability density function (PDF) is displayed on the left side of the plot.
> - A horizontal dashed-dotted orange line intersects the orange PDF.
> - A red PDF is shown at the bottom of the plot.
> - A vertical dotted gray line extends from the red asterisk to the x-axis.
>
> The panels are labeled "Iteration 1", "Iteration 2", and "Iteration 3" respectively.

Figure S14: Bayesian optimization example. We show here an example evolution of ACE-TS on a 1D function. The orange pdf on the left of each panel is $p\left(y_{\text {opt }} \mid \mathcal{D}_{N}\right)$, the red pdf at the bottom of each panel is $p\left(x_{\text {opt }} \mid y_{\text {opt }}, \mathcal{D}_{N}\right)$, for a sampled $y_{\text {opt }}$ (orange dashed-dot line). The queried point at each iteration is marked with a red asterisk, while black and blue dots represent the observed points. Note how ACE is able to learn complex conditional predictive distributions for $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$.

Bayesian optimization with Minimum-value Entropy Search (ACE-MES). For Minimum-value Entropy Search (MES; Wang and Jegelka, 2017), the procedure is as follows:

1. First, we propose $N_{\text {candidate }}$ points. We generate these candidate points by sampling $80 \%$ of them using the conditional Thompson sampling approach described earlier, i.e., $p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}, y_{\text {opt }}<\tau\right)$, and the remaining $20 \%$ directly from $p\left(\mathbf{x}_{\text {opt }} \mid \mathcal{D}_{N}\right)$. In our experiments we use $N_{\text {candidate }}=20$.
2. For each candidate point $\mathbf{x}^{*}$, we evaluate the acquisition function, which in our case is the gain in mutual information between the maximum $y_{\text {opt }}$ and the candidate point $\mathbf{x}^{*}$ (Eq. (5)).
3. To compute the first term of the right-hand side of Eq. (5), for a candidate point $\mathbf{x}^{*}$, we calculate the predictive distribution $p\left(y^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{N}\right)$ represented in our model by a mixture of Gaussians. We compute its entropy via numerical integration over a grid.
4. For the second term of the right-hand side of Eq. (5), we perform Monte Carlo sampling to evaluate the expected entropy. For each candidate point $\mathbf{x}^{*}$, we draw $N_{\mathrm{mc}}$ samples of $y_{\mathrm{opt}}$ from the predictive distribution $p\left(y_{\text {opt }} \mid \mathcal{D}_{N}\right)$. We set $N_{\mathrm{mc}}=20$ to ensure the procedure remains efficient while maintaining accuracy.
5. For each sampled $y_{\text {opt }}$, we determine the predictive distribution $p\left(y^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{N}, y_{\text {opt }}\right)$. Then, for each mixture, we compute the entropy as in step 2 . We then average over samples to compute the expectation.
6. To compute the estimated MES value of candidate point $\mathbf{x}^{*}$ we subtract the computed first term to the second term of the equation Eq. (5).
7. We repeat this procedure for all candidate points $\mathbf{x}^{*}$ and select the point with the highest information gain. This point is expected to yield the lowest uncertainty about the value of the minimum, thus guiding our next query in the Bayesian optimization process.

To illustrate the implementation details of ACE-MES, we present its pseudocode in Algorithm 2.

# C.3.3 Bayesian optimization with prior over $\mathbf{x}_{\text {opt }}$

ACE is capable of injecting a prior over latents when predicting values and latents. In the context of BO, this prior could incorporate information about the location of the optimum, $\mathbf{x}_{\text {opt }}$. Several works, such as (Souza et al., 2021; Hvarfner et al., 2022; Müller et al., 2023), have explored the use of priors in BO to improve predictive performance. In our experiments, we evaluate two types of priors: strong and weak, to assess the robustness of the model under varying levels of prior knowledge. As a baseline, we utilize a $\pi$ BO-like procedure (Hvarfner et al., 2022), as described below, to perform Thompson sampling across all experiments.

---

#### Page 30

Training. For training, we generate a prior distribution similar to Appendix B.1, but with slight adjustments: when sampling the mixture distribution, we include a $50 \%$ chance of adding a uniform component. If present, the uniform distribution weight $w_{\text {unif }}$ is sampled uniformly from 0.0 to 0.2 (otherwise $w_{\text {unif }}=0$ ). The uniform component is then added as follows:

$$
\mathbf{p}=\left(w_{\text {unif }} \cdot \mathbf{p}_{\text {unif }}\right)+\left(1-w_{\text {unif }}\right) \cdot \mathbf{p}_{\text {mixture }}
$$

where $\mathbf{p}_{\text {unif }}$ represents the uniform component, and $\mathbf{p}_{\text {mixture }}$ is the sampled mixture. The inclusion of a uniform component during training means that the prior can be a mixture of an informative and a non-informative (flat) component, which will be useful later. Using this binned distribution, we then sample $\mathbf{x}_{\text {opt }}$ and $y_{\text {opt }}$, and use these two latent samples to construct our function, as outlined in Appendix C.3.1.

Testing. During the BO testing phase, we consider two scenarios:

1. Strong prior: We first sample a mean for the $\mathbf{x}_{\text {opt }}$ prior by drawing from a Gaussian distribution centered on the true $\mathbf{x}_{\text {opt }}$ with a standard deviation set to $10 \%$ of the domain (in our case $[-1,1]$ ), resulting in a standard deviation of 0.2 . We use this sampled prior mean and standard deviation to construct the binned prior.
2. Weak prior: The same steps are applied to generate the prior, but with a standard deviation of $25 \%$, which translates to 0.5 for our domain.

In both scenarios, we add a uniform prior component with $w_{\text {uniform }}=0.1$. The uniform component helps with model and prior misspecification, by allowing the model to explore outside the region indicated by the prior.

We compare ACE with Prior Thompson Sampling (ACEP-TS) to the no-prior ACE-TS and a baseline GP-TS. We also consider a state-of-the-art heuristic for prior injection in BO, $\pi$ BO (Hvarfner et al., 2022), with the TS acquisition procedure described below ( $\pi$ BO-TS). The procedure is repeated 10 times for each case, with different initial points sampled at random.
$\pi$ BO-TS. The main technique in $\pi \mathrm{BO}$ for injecting a prior in the BO procedure consists of rescaling the chosen acquisition function $\alpha(\mathbf{x})$ by the user-provided prior over the optimum location $\pi(\mathbf{x})$ (Eq. 6 in Hvarfner et al., 2022),

$$
\alpha_{\pi \mathrm{BO}}(\mathbf{x} ; \alpha) \propto \alpha(\mathbf{x}) \pi(\mathbf{x})^{\gamma_{n}}
$$

where $n$ is the BO iteration and $\gamma_{n}$ governs the relative influence of the prior with respect to the acquisition function, which is heuristically made to decay over iterations to reflect the increased role of the observed data. As in Hvarfner et al. (2022), we set $\gamma_{n}=\frac{\beta}{n}$ where $\beta$ is a hyperparameter reflecting the user confidence on the prior.

To implement Thompson sampling (TS) with $\pi \mathrm{BO}$, we first note that the TS acquisition function $\alpha_{\mathrm{TS}}(\mathbf{x})$ corresponds to the current posterior probability over the optimum location, and the TS procedure consists of drawing one sample from this acquisition function (as opposed to optimizing it). Thus, the $\pi \mathrm{BO}$ variant of TS ( $\pi$ BO-TS) corresponds to sampling from Eq. (S7), where the current posterior over the optimum takes the role of $\alpha(\mathbf{x})$. We sample from Eq. (S7) using a self-normalized importance sampling-resampling approach (Robert and Casella, 2004). Namely, we sample $N_{\mathrm{TS}}=100$ points from $\alpha_{\mathrm{TS}}$ using batch GP-TS, then resample one point from this batch using importance sampling weight $w \propto \frac{\alpha_{\mathrm{TS}}(\mathbf{x}) \pi(\mathbf{x})^{\beta / n}}{\alpha_{\mathrm{TS}}(\mathbf{x})}=\pi(\mathbf{x})^{\beta / n}$, where all weights are then normalized to sum to 1 . Following (Hvarfner et al., 2022), we set $\beta=10$, i.e., equal to their setting when running BO experiments with 100 iterations, as in our case.

# C.3.4 Benchmark functions and baselines

BO benchmarks. We use a diverse set of benchmark functions with input dimensions ranging from 1D to 6D to thoroughly evaluate ACE's performance on the BO task. These include (1) the non-convex Ackley function in both 1D and 2D, (2) the 1D Gramacy-Lee function, known for its multiple local minima, (3) the 1D Negative Easom function, characterized by a sharp, narrow global minimum and deceptive flat regions, (4) the non-convex 2D Branin Scaled function with multiple global minima, (5) the 2D Michalewicz function, which features a complex landscape with multiple local minima, (6) the 3D, 5D, and 6D Levy function, with numerous local minima due to its sinusoidal component, (7) the 5D and 6D Griewank function, which is highly multimodal and regularly spaced local minima, but a single smooth global minimum, (8) the 4D and 5D Rosenbrock function,

---

#### Page 31

> **Image description.** The image is a collection of eight line graphs arranged in a 2x4 grid. Each graph displays the "Regret" on the y-axis versus "Iteration" on the x-axis for different test functions.
>
> - **General Layout:** The graphs are organized in two rows with four graphs in each row. Each graph has labeled axes and a title indicating the test function and its dimensionality.
>
> - **Axes and Labels:**
>
>   - The y-axis is labeled "Regret" and ranges from 0 to varying maximum values depending on the graph (e.g., 3.8, 0.9, 1.2, 5.8, 1.8, 75, 90).
>   - The x-axis is labeled "Iteration" and ranges from 0 to varying maximum values depending on the graph (e.g., 25, 50, 75, 90).
>   - The x-axis has tick marks at approximately 10, 25, 50, 75, and 90.
>   - The y-axis has tick marks at 0 and the maximum value.
>
> - **Graph Titles (Test Functions):**
>
>   - Top Row: "Ackley 1D", "Easom 1D", "Michalewicz 2D", "Ackley 2D"
>   - Bottom Row: "Levy 3D", "Hartmann 4D", "Griewank 5D", "Griewank 6D"
>
> - **Lines and Shaded Regions:** Each graph contains multiple lines representing different algorithms, along with shaded regions around each line. The lines are colored and styled differently to distinguish them. The shaded regions likely represent the standard error or confidence intervals.
>
> - **Legend (Top of the image):** A legend at the top of the image identifies the different algorithms:
>
>   - "ACE-TS" (solid blue line)
>   - "ACE-MES" (dashed blue line)
>   - "AR-TNPD-TS" (solid green line)
>   - "GP-TS" (solid orange line)
>   - "GP-MES" (dashed orange line)
>   - "Random" (dotted pink line)
>
> - **Visual Patterns:** The lines generally start at a higher "Regret" value and decrease as the "Iteration" increases, indicating an improvement in performance over time. The rate of decrease and the final "Regret" value vary depending on the algorithm and the test function. The "Random" line generally stays higher than the other lines, suggesting poorer performance.

Figure S15: Bayesian optimization additional results. Regret comparison (mean $\pm$ standard error) on extended BO benchmark results on distinct test functions.
which has a narrow, curved valley containing the global minimum, and (9) the 3D, 4D, and 6D Hartmann function, a widely used standard benchmark. These functions present a range of challenges, allowing us to effectively test the robustness and accuracy of ACE across different scenarios.

BO baselines. For our baselines, we employ three methods: autoregressive Thompson Sampling with TNP-D (AR-TNPD-TS) (Nguyen and Grover, 2022), Gaussian Process-based Bayesian Optimization with the MES acquisition function (GP-MES) (Wang and Jegelka, 2017), and Gaussian Process-based Thompson Sampling (GP-TS) with 5000 candidate points. In addition, we use $\pi$ BO-TS for the prior injection case as the baseline Hvarfner et al. (2022) (using the same number of candidate points used in GP-TS). We optimize the acquisition function in GP-MES using the 'shotgun' procedure detailed later in this section (with 1000 candidate points for minimum value approximation via Gumbel sampling). Both GP-MES and GP-TS implementations are written using the BOTorch library (Balandat et al., 2020). For AR-TNPD-TS, we use the same network architecture configurations as ACE, but with a non-linear embedder and a single Gaussian head (Nguyen and Grover, 2022). Additionally, AR-TNPD-TS uses autoregressive sampling, as described in (Bruinsma et al., 2023).
We conducted our experiments with 100 BO iterations across all benchmark functions. The number of initial points was set to 3 for 1D experiments and 10 for 2D-6D experiments. These initial points were drawn uniformly randomly within the input domain. We evaluated the runtime performance of our methods and baseline algorithms on a local machine equipped with a 13th Gen Intel(R) Core(TM) i5-1335U processor and 15GB of RAM. On average, the runtime for 100 BO iterations was as follows: ACE-TS and ACEP-TS completed in approximately 5 seconds; ACE-MES required about 1.3 minutes; GP-TS and $\pi$ BO-TS took roughly 2 minutes; GP-MES took about 1.4 minutes; and AR-TNPD-TS was the slowest, requiring approximately 10 minutes, largely due to the computational cost of its autoregressive steps.

Shotgun optimizer. To perform fast optimization in parallel, we first sample 10000 points from a quasirandom grid using the Sobol sequence. Then we pick the point with the highest acquisition function value, referred to as $\mathbf{x}_{0}$. Subsequently, we sample 1000 points around $\mathbf{x}_{0}$ using a multivariate normal distribution with diagonal covariance $\sigma^{2} \mathbf{I}$, where the initial $\sigma$ is set to the median distance among the points. We re-evaluate the acquisition function over this neighborhood, including $\mathbf{x}_{0}$, and select the best point. After that, we reduce $\sigma$ by a factor of five and repeat the process, iterating from the current best point. This 'shotgun' approach allows us to zoom into a high-valued region of the acquisition function while exploiting large parallel evaluations.

---

#### Page 32

> **Image description.** The image contains six line graphs arranged in a 2x3 grid. Each graph displays the "Regret" on the y-axis versus "Iteration" on the x-axis. All graphs have the same four lines, each representing a different algorithm: ACE-TS (solid blue), ACEP-TS (dotted blue), GP-TS (solid orange), and πBO-TS (dotted-dashed orange). Shaded regions around the lines indicate the standard error.
>
> Each graph is titled with a function name and "(weak)". The function names are:
>
> - Top left: Ackley 1D (weak)
> - Top middle: Gramacy Lee 1D (weak)
> - Top right: Negeasom 1D (weak)
> - Bottom left: Branin 2D (weak)
> - Bottom middle: Ackley 2D (weak)
> - Bottom right: Hartmann 3D (weak)
>
> The y-axis scales vary between the graphs, with maximum values of 4.2, 0.7, 0.8, 0.13, 4.5, and 1.3 respectively. The x-axis scales also vary, with maximum values of 25, 75, 50, 75, 90, and 50 respectively.
>
> A legend is located at the top of the image, showing the line styles and colors corresponding to each algorithm.

Figure S16: Bayesian optimization with weak prior. Simple regret (mean $\pm$ standard error). Prior injection can improve the performance of ACE, making it perform competitively compared to $\pi$ BO-TS.

> **Image description.** The image presents a figure consisting of six line graphs arranged in a 2x3 grid. Each graph depicts the "Regret" on the y-axis versus "Iteration" on the x-axis for different benchmark functions. All graphs are labeled with the function name and "(strong)".
>
> The top row contains graphs for:
>
> 1.  "Ackley 1D (strong)" with the y-axis ranging from 0 to 4.2 and x-axis from 0 to 25.
> 2.  "Gramacy Lee 1D (strong)" with the y-axis ranging from 0 to 0.7 and x-axis from 0 to 75.
> 3.  "Negeasom 1D (strong)" with the y-axis ranging from 0 to 0.8 and x-axis from 0 to 50.
>
> The bottom row contains graphs for:
>
> 1.  "Branin 2D (strong)" with the y-axis ranging from 0 to 0.13 and x-axis from 0 to 75.
> 2.  "Ackley 2D (strong)" with the y-axis ranging from 0 to 4.5 and x-axis from 0 to 90.
> 3.  "Hartmann 3D (strong)" with the y-axis ranging from 0 to 1.3 and x-axis from 0 to 50.
>
> Each graph displays four lines representing different algorithms: "ACE-TS" (solid blue), "ACEP-TS" (dashed blue), "GP-TS" (solid orange), and "πBO-TS" (dashed orange). Shaded regions around the "ACE-TS" and "GP-TS" lines indicate the standard error. The plots show the regret decreasing with increasing iterations for each algorithm on the respective benchmark function.

Figure S17: Bayesian optimization with strong prior. Simple regret (mean $\pm$ standard error). When strong priors are used, the gap between ACE-TS and ACEP-TS is more evident compared to weak priors.

# C.3.5 Additional Bayesian optimization results.

Standard BO setting additional results. Additional results in Fig. S15 complement those in Fig. 5. While our method performs generally well across different benchmark functions, we find that it struggles on the Michalewicz function, likely because its sharp, narrow optima and highly irregular landscape differ significantly from the function classes used during training. Conversely, ACE performs competitively on Griewank, where the structured landscape aligns well with our approach. On the 2D Ackley function, the challenge may stem from its highly non-stationary nature, while our method was trained only on draws from stationary kernels. Addressing functions like Michalewicz and Ackley may require extending our relatively simple function generation process and incorporating specialized techniques like input and output warping (Müller et al., 2023) to better handle non-stationarity.

---

#### Page 33

BO with prior over $\mathbf{x}_{\text {opt }}$ additional results. Additional results on the weak prior scenario are presented in Fig. S16 and with strong prior in Fig. S17. The results indicate that ACEP-TS consistently outperforms ACE-TS, particularly when using a strong prior. In this case, the model benefits from the prior information, leading to a notable improvement in performance. Specifically, the strong prior allows the model to converge more rapidly toward the optimum.

# C. 4 Simulation-based inference

## C.4.1 Simulators

The experiments reported in Section 4.3 used three time-series models to simulate the training and test data. This section describes the simulators in more details.

Ornstein-Uhlenbeck Process (OUP) is widely used in financial mathematics and evolutionary biology due to its ability to model mean-reverting stochastic processes (Uhlenbeck and Ornstein, 1930). The model is defined as:

$$
y_{t+1}=y_{t}+\Delta y_{t}, \quad \Delta y_{t}=\theta_{1}\left[\exp \left(\theta_{2}\right)-y_{t}\right] \Delta t+0.5 w, \quad \text { for } t=1, \ldots, T
$$

where $T=25, \Delta t=0.2, x_{0}=10$, and $w \sim \mathcal{N}(0, \Delta t)$. We use a uniform prior $U([0,2] \times[-2,2])$ for the latent variables $\boldsymbol{\theta}=\left(\theta_{1}, \theta_{2}\right)$ to generate the simulated data.

Susceptible-Infectious-Recovered (SIR) is a simple compartmental model used to describe infectious disease outbreaks (Kermack and McKendrick, 1927). The model divides a population into susceptible (S), infectious (I), and recovered (R) individuals. Assuming population size $N$ and using $S_{t}, I_{t}$, and $R_{t}$ to denote the number of individuals in each compartment at time $t, t=1, \ldots, T$, the disease outbreak dynamics can be expressed as

$$
\Delta S_{t}=-\beta \frac{I_{t} S_{t}}{N}, \quad \Delta I_{t}=\beta \frac{I_{t} S_{t}}{N}-\gamma I_{t}, \quad \Delta R_{t}=\gamma I_{t}
$$

where the parameters $\beta$ and $\gamma$ denote the contact rate and the mean recovery rate. An observation model with parameters $\phi$ is used to convert the SIR model predictions to observations $\left(t, y_{t}\right)$. The experiments carried out in this work consider two observation models and simulator setups.

The setups considered in this work are as follows. First, we consider a SIR model with fixed initial condition and 10 observations $y_{t} \sim \operatorname{Bin}\left(1000, I_{t} / N\right)$ collected from $T=160$ time points at even interval, as proposed in (Lueckmann et al., 2021). Here the population size $N=10^{6}$ and the initial condition is fixed as $S_{0}=N-1$, $I_{0}=1, R_{0}=0$. We use uniform priors $\beta \sim U(0.01,1.5)$ and $\gamma \sim U(0.02,0.25)$. We used this model version in the main experiments presented in Section 4.3 and Appendix C.4.2.

In addition we consider a setup where $N$ and $I_{0}$ are unknown and we collect 25 observations $y_{t} \sim \operatorname{Poi}\left(\phi I_{t} / N\right)$ from $T=250$ time points at even interval. We use $\beta \sim U(0.5,3.5), \gamma \sim U(0.0001,1.5), \phi \sim U(50,5000)$, and $I_{0} / N \sim U(0.0001,0.01)$ with $S_{0} / N=1-I_{0} / N$ and $R_{0} / N=0$ to generate simulated samples. We used this model version in an additional experiment to test ACE on real world data, presented in Appendix C.4.5.

Turin model is a time-series model used to simulate radio propagation phenomena, making it useful for testing and designing wireless communication systems (Turin et al., 1972; Pedersen, 2019; Bharti et al., 2019). The model generates high-dimensional complex-valued time-series data and is characterized by four key parameters that control different aspects of the radio signal: $G_{0}$ controls the reverberation gain, $T$ determines the reverberation time, $\nu$ specifies the arrival rate of the point process, and $\sigma_{W}^{2}$ represents the noise variance.
The model starts with a frequency bandwidth $B=0.5 \mathrm{GHz}$ and simulates the transfer function $H_{k}$ over $N_{s}=101$ equidistant frequency points. The measured transfer function at the $k$-th point, $Y_{k}$, is given by:

$$
Y_{k}=H_{k}+W_{k}, \quad k=0,1, \ldots, N_{s}-1
$$

where $W_{k}$ denotes additive zero-mean complex circular symmetric Gaussian noise with variance $\sigma_{W}^{2}$. The transfer function $H_{k}$ is defined as:

$$
H_{k}=\sum_{l=1}^{N_{\text {points }}} \alpha_{l} \exp \left(-j 2 \pi \Delta f k \tau_{l}\right)
$$

---

#### Page 34

where $\tau_{l}$ are the time delays sampled from a one-dimensional homogeneous Poisson point process with rate $\nu$, and $\alpha_{l}$ are complex gains. The gains $\alpha_{l}$ are modeled as i.i.d. zero-mean complex Gaussian random variables conditioned on the delays, with a conditional variance:

$$
\mathbb{E}\left[\left|\alpha_{l}\right|^{2} \mid \tau_{l}\right]=\frac{G_{0} \exp \left(-\tau_{l} / T\right)}{\nu}
$$

The time-domain signal $\tilde{y}(t)$ can be obtained by taking the inverse Fourier transform:

$$
\tilde{y}(t)=\frac{1}{N_{s}} \sum_{k=0}^{N_{s}-1} Y_{k} \exp (j 2 \pi k \Delta f t)
$$

with $\Delta f=B /\left(N_{s}-1\right)$ being the frequency separation. Our final real-valued output is calculated by taking the absolute square of the complex-valued data and applying a logarithmic transformation $y(t)=10 \log _{10}\left(|\tilde{y}(t)|^{2}\right)$.
The four parameters of the model are sampled from the following uniform priors: $G_{0} \sim \mathcal{U}\left(10^{-9}, 10^{-8}\right), T \sim$ $\mathcal{U}\left(10^{-9}, 10^{-8}\right), \nu \sim \mathcal{U}\left(10^{7}, 5 \times 10^{9}\right), \sigma_{W}^{2} \sim \mathcal{U}\left(10^{-10}, 10^{-9}\right)$.

# C.4.2 Main experiments

ACE was trained on examples that included simulated time series data and model parameters divided between target and context. In these experiments, the time series data were divided into context and target data by sampling $N_{d}$ data points into the context set and including the rest in the target set. The context size $N_{d} \sim U(10,25)$ in the OUP experiments, $N_{d} \sim U(5,10)$ in the SIR experiments, and $N_{d} \sim U(50,101)$ in the Turin experiments. In addition, the model parameters were randomly assigned to either the context or target set. NPE and NRE cannot handle partial observations and was trained with the full time series data in both cases.

The ACE model used in these experiments had embedding dimension 64 and 6 transformer layers. The attention blocks had 4 heads and the MLP block had hidden dimension 128. The output head had $K=20$ MLP components with hidden dimension 128. The model was trained for $5 \times 10^{4}$ steps with batch size 32 , using learning rate $5 \times 10^{-4}$ with cosine annealing.

We used the sbi package (Tejero-Cantero et al., 2020) (https://sbi-dev.github.io/sbi/, Version: 0.22.0, License: Apache 2.0) to implement NPE and NRE. Specifically, we chose the NPE-C (Greenberg et al., 2019) and NRE-C (Miller et al., 2022) with Masked Autoregressive Flow (MAF) (Papamakarios et al., 2017) as the inference network. We used the default configuration with 50 hidden units and 5 transforms for MAF, and training with a fixed learning rate $5 \times 10^{-4}$. For Simformer (Gloeckler et al., 2024), we used their official package (https://github.com/mackelab/simformer, Version: 2, License: MIT). We used the same configuration as in our setup for the transformer, while we used their default configuration for the diffusion part. For a fair comparison, we pre-generated $10^{4}$ parameter-simulation pairs for all methods. We also normalized the parameters of the Turin model when feeding into the networks. For evaluation, we randomly generated 100 observations and assessed each method across 5 runs. For the RMSE evaluation, given $N_{\text {obs }}$ observations, with $N_{\text {post }}$ posterior samples generated for each observation, and $L$ latent parameters, our RMSE metric is calculated as:

$$
\operatorname{RMSE}=\frac{1}{N_{\mathrm{obs}}} \sum_{i=1}^{N_{\mathrm{obs}}} \sqrt{\frac{1}{L \cdot N_{\mathrm{post}}} \sum_{l=1}^{L} \sum_{j=1}^{N_{\mathrm{post}}}\left(\theta_{i, l}-\hat{\theta}_{i, l, j}\right)^{2}}
$$

where $\theta_{i, l}$ represents the true value of the $l$-th latent parameter for the $i$-th observation, and $\hat{\theta}_{i, l, j}$ represents the $j$-th posterior sample of the $l$-th latent parameter for the $i$-th observation. This approach first calculates the RMSE for each observation (averaging across all latent dimensions and posterior samples for that observation), and then averages these observation-specific RMSE values to obtain the final metric. For MMD, we use an exponentiated quadratic kernel with a lengthscale of 1 .

Statistical comparisons. We evaluate models based on their average results across multiple runs and perform pairwise comparisons to identify models with comparable performance. The results from pairwise comparisons are used in Table 1 to highlight in bold the models that are considered best in each experiment. The following procedure is used to determine the best models:

---

#### Page 35

- First, we identify the model (A) with the highest empirical mean and highlight it in bold.
- For each alternative model (B), we perform $10^{5}$ bootstrap iterations to resample the mean performance for both model A and model B.
- We then calculate the proportion of bootstrap iterations where model B outperforms model A.
- If this proportion is larger than the significance level $(\alpha=0.05)$, model B is considered statistically indistinguishable from model A.
- All models that are not statistically significantly different from the best model are highlighted in bold.

# C.4.3 Ablation study: Gaussian vs. mixture-of-Gaussians output heads

To assess the impact of using a Gaussian versus a mixture-of-Gaussians output head in ACE, we conduct an ablation study on the SBI tasks. In theory, a mixture-of-Gaussians head should improve performance when the predictive or posterior data distributions are non-Gaussian. Table S2 shows the results. As expected, we observe improvements in OUP and Turin when using a mixture-of-Gaussians head. This suggests that more flexible distributional families better capture complex distributions. However, for the SIR task, the performance difference is negligible as the posteriors are largely Gaussian. These findings align with our expectations.

|       |                                            | Gaussian (ablation) | Mixture-of-Gaussians (ACE) |
| :---: | :----------------------------------------: | :-----------------: | :------------------------: |
|       | $\log -\operatorname{probs}_{g}(\uparrow)$ |    $0.90(0.01)$     |  $\mathbf{1 . 0 3}(0.02)$  |
|  OUP  |   $\operatorname{RMSE}_{g}(\downarrow)$    |    $0.48(0.01)$     |        $0.48(0.00)$        |
|       |    $\operatorname{MMD}_{g}(\downarrow)$    |    $0.52(0.00)$     |  $\mathbf{0 . 5 1}(0.00)$  |
|       | $\log -\operatorname{probs}_{g}(\uparrow)$ |    $6.80(0.02)$     |        $6.78(0.02)$        |
|  SIR  |   $\operatorname{RMSE}_{g}(\downarrow)$    |    $0.02(0.00)$     |        $0.02(0.00)$        |
|       |    $\operatorname{MMD}_{g}(\downarrow)$    |    $0.02(0.00)$     |        $0.02(0.00)$        |
|       | $\log -\operatorname{probs}_{g}(\uparrow)$ |    $2.73(0.02)$     |  $\mathbf{3 . 1 4}(0.02)$  |
| Turin |   $\operatorname{RMSE}_{g}(\downarrow)$    |    $0.24(0.00)$     |        $0.24(0.00)$        |
|       |    $\operatorname{MMD}_{g}(\downarrow)$    |    $0.36(0.00)$     |  $\mathbf{0 . 3 5}(0.00)$  |

Table S2: Ablation study comparing single Gaussian versus mixture-of-Gaussians output heads across SBI tasks. Mean and standard deviation from 5 runs are reported. mixture-of-Gaussians heads benefit complex distributions (OUP and Turin), while maintaining similar performance on simpler tasks (SIR).

## C.4.4 Simulation-based calibration

To evaluate the calibration of the approximate posteriors obtained by ACE, we apply simulation-based calibration (SBC; Talts et al. 2018) on the Turin model to evaluate whether the approximate posteriors produced by ACE are calibrated. We recall that SBC checks if a Bayesian inference process is well-calibrated by repeatedly simulating data from parameters drawn from the prior and inferring posteriors under those priors and simulated datasets. If the inference is calibrated, the average posterior should match the prior. Equivalently, when ranking the true parameters within each posterior, the ranks should follow a uniform distribution (Talts et al., 2018).

We use the following procedure for SBC: for a given prior, we first sample 1000 samples from the prior distribution and generate corresponding simulated data. Then we use ACE to approximate the posteriors and subsequently compare the true parameter values with samples drawn from the inferred posterior distribution. To visualize the calibration, we plot the density of the posterior samples against the prior samples. If the model is well-calibrated, the posterior distribution should recover the true posterior, which results in a close match between the density of the posterior samples and the prior. We also present the fractional rank statistic against the ECDF difference (Săilynoja et al., 2022). Ideally, the ECDF difference between the rank statistics and the theoretical uniform distribution should remain close to zero, indicating well-calibrated posteriors.

Fig. S18 shows that our ACE is well-calibrated with pre-defined uniform priors across all four latents. Since ACEP allows conditioning on different priors at runtime, we also test the calibration of ACEP using randomly generated priors (following Appendix B.1). For comparison, we show what happens if we forego prior-injection, using vanilla ACE instead of ACEP. The visualization on one set of priors is shown in Fig. S19. As expected,

---

#### Page 36

> **Image description.** The image contains four pairs of plots, arranged in two rows and four columns. Each pair consists of a density plot in the top row and a fractional rank statistic plot in the bottom row.
>
> - **Top Row (Density Plots):** Each plot in the top row displays the density of posterior samples compared with prior samples. The y-axis is labeled "Density." Each plot contains two lines: one in gray, representing "prior samples," and one in purple, representing "ACE." The x-axis scales vary across the four plots. From left to right, the x-axis scales are x10^8, x10^8, x10^-10, and x10^9.
>
> - **Bottom Row (Fractional Rank Statistic Plots):** Each plot in the bottom row shows the fractional rank statistic against the ECDF difference. The y-axis is labeled "Δ ECDF." The x-axis is labeled "Fractional Rank." Each plot contains a purple line representing "ACE" and a gray shaded area resembling an oval. The x-axis ranges from 0.00 to 1.00 in all four plots. Below the x-axis label, each plot has a different label: "G0", "T", "v", and "σw^2" from left to right.

Figure S18: Simulation-based calibration of ACE on the Turin model. The top row shows the density of the posterior samples from ACE compared with the prior samples. The bottom row shows the fractional rank statistic against the ECDF difference with $95 \%$ confidence bands. ACE is well-calibrated.
vanilla ACE (without prior-injection) does not include the correct prior information and shows suboptimal calibration performance, whereas ACEP correctly leverages the provided prior information and shows closer alignment with the prior and lower ECDF deviations. We also calculate the average absolute deviation over 100 randomly sampled priors. In the prior-injection setting, ACEP demonstrates better calibration, with an average deviation of $0.03 \pm 0.01$ compared to $0.10 \pm 0.05$ for ACE without the correct prior.

# C.4.5 Extended SIR model on real-world data

We present here results obtained by considering an extended four-parameter version of the SIR model then applied to real-world data. We further include details on the training data and model configurations used in the real-data experiment as well as additional evaluation results from experiments carried out with simulated data. As our real-world data, we used a dataset that describes an influenza outbreak in a boarding school. The dataset is available in the R package outbreaks (https://cran.r-project.org/package=outbreaks, Version: 1.9.0, License: MIT).

Methods. The four-parameter SIR model we used is detailed in Appendix C.4.1 (last paragraph). The ACE models were trained with samples constructed based on simulated data as follows. The observations were divided into context and target points by sampling $N_{d} \sim U(2,20)$ data points into the context set and 2 data points into the target set. The examples included $50 \%$ interpolation tasks where the context and target points were sampled at random (without overlap) and $50 \%$ forecast tasks where the points were sampled in order. The model parameters were divided between the context and target set by sampling the number to include $N_{l} \sim U(0,4)$ and sampling the $N_{l}$ parameters from the parameter set at random. The parameters were normalized to range $[-1,1]$ and the observations were square-root compressed and scaled to the approximate range $[-1,1]$.

The ACE models had the same architecture as the models used in the main experiment, but the models were trained for $10^{5}$ steps with batch size 32 . In this experiment, we generated the data online during the training, which means that the models were trained with $3.2 \times 10^{6}$ samples. The NPE models used in this experiment had the same configuration as the model used in the main experiment, for fair comparison, the models were now trained with $3.2 \times 10^{6}$ samples. Each sample corresponded to a unique simulation and the full time series was used as the observation data.

To validate model predictions, we note that ground-truth parameter values are not available for real data. Instead, we examined whether running the simulator with parameters sampled from the posterior can replicate the observed data. For reference, we also included MCMC results. The MCMC posterior was sampled with Pyro (Bingham et al., 2018) (https://pyro.ai/, Version: 1.9.0, License: Apache 2.0) using the random walk kernel

---

#### Page 37

> **Image description.** The image presents a figure composed of eight subplots arranged in a 2x4 grid. The top row displays probability density plots, while the bottom row shows ECDF (Empirical Cumulative Distribution Function) difference plots.
>
> - **Top Row (Density Plots):** Each of the four subplots in the top row displays the probability density on the y-axis and a parameter value on the x-axis. The y-axis is labeled "Density." Each plot contains three overlapping curves representing different distributions: "prior samples" (gray), "ACE" (purple), and "ACEP" (green). The x-axes are labeled with parameter values and are scaled differently across the plots, with multipliers of x10^-8, x10^-8, x10^9, and x10^-9 respectively.
>
> - **Bottom Row (ECDF Difference Plots):** Each of the four subplots in the bottom row displays the difference in ECDF (Δ ECDF) on the y-axis and "Fractional Rank" on the x-axis. The y-axis is labeled "Δ ECDF." Each plot contains two curves representing "ACE" (purple) and "ACEP" (green). A gray shaded area, roughly elliptical in shape, is present in the background of each plot. The x-axis is labeled "Fractional Rank". The x-axis ranges from 0.00 to 1.00.
>
> - **Individual Subplots (Bottom Row):**
>   - The first subplot is labeled "G_0" below the x-axis.
>   - The second subplot is labeled "T" below the x-axis.
>   - The third subplot is labeled "v" below the x-axis.
>   - The fourth subplot is labeled "σ_w^2" below the x-axis.
>
> The overall layout suggests a comparison of the ACE and ACEP methods against a prior distribution, with the top row showing the density estimates and the bottom row showing the difference in their empirical cumulative distribution functions.

Figure S19: Simulation-based calibration of ACE and ACEP on the Turin model with an example custom prior. ACEP demonstrates improved calibration by closely following the prior distribution and showing lower deviations in the ECDF difference, highlighting its ability to condition on user-specified priors effectively.
and sampling 4 chains with $5 \times 10^{4}$ warm-up steps and $5 \times 10^{4}$ samples.

> **Image description.** The image consists of three line graphs arranged horizontally, each representing a different model: ACE, NPE, and MCMC. Each graph displays the relationship between "Time" on the x-axis and "Count" on the y-axis.
>
> - **General Layout:** The three graphs are identically formatted. The x-axis ranges from 0 to 10, and the y-axis ranges from 0 to 400. Each graph contains a blue line representing the "PPD mean," a shaded light blue area representing the "PPD 95% CI" (confidence interval), and black dots representing "observed" data points.
>
> - **Graph 1 (ACE):** The title "ACE" is above the graph. The text "log-prob ↑ -64.4" is positioned near the top of the graph. The blue line and shaded area form a curve that peaks around Time = 5. The black dots are clustered around the blue line.
>
> - **Graph 2 (NPE):** The title "NPE" is above the graph. The text "log-prob ↑ -64.6" is positioned near the top of the graph. The curve formed by the blue line and shaded area is similar to the ACE graph, peaking around Time = 5. The black dots are clustered around the blue line.
>
> - **Graph 3 (MCMC):** The title "MCMC" is above the graph. The text "log-prob ↑ -62.9" is positioned near the top of the graph. The curve formed by the blue line and shaded area is similar to the ACE and NPE graphs, peaking around Time = 5. The black dots are clustered around the blue line.
>
> - **Legend:** A legend is located to the right of the MCMC graph. It labels the blue line as "PPD mean," the shaded light blue area as "PPD 95% CI," and the black dots as "observed."

Figure S20: SIR model on a real dataset. Posterior predictive distributions based on the ACE, NPE, and MCMC posteriors. The dataset is mildly misspecified, in that even MCMC does not fully match the data.

Results. The posterior predictive distributions and log-probabilities for observed data calculated based on ACE, NPE, and MCMC results are shown in Fig. S20. For this visualization, ACE and NPE models were trained once, and simulations were carried out with 5000 parameters sampled from each posterior distribution. The log-probabilities observed in this experiment are -64.4 with ACE, -64.6 with NPE. Repeating ACE and NPE training and posterior estimation 10 times, the average log-probabilities across the 10 runs were -65.1 (standard deviation 0.4 ) with ACE and -65.5 (standard deviation 0.7 ) with NPE, showing a similar performance. The ACE predictions used in this experiment are sampled autoregressively (see Appendix B.4). These results show that ACE can handle inference with real data.

Validation on simulated data. For completeness, we performed a more extensive validation of ACE and other methods with the extended SIR model using simulated data. Specifically, we assessed the ACE and NPE models on simulated data and evaluated the same ACE models in a data completion task with the TNP-D baseline. All the training details remain the same as in the real-world experiment for ACE and NPE. The TNP-D models had the same overall architecture as ACE but used a different embedder and output head. The MLP block in the TNP-D embedder had hidden dimension 64 and the MLP block in the single-component output head hidden dimension 128. The TNP-D models were trained for $10^{5}$ steps with batch size 32 . The evaluation set used in these experiments included 1000 simulations sampled from the training distribution and the evaluation metrics included log-probabilities and coverage probabilities calculated based on $95 \%$ quantile intervals that were

---

#### Page 38

Table S3: Comparison between ACE and NPE in posterior estimation task in the extended SIR model. The ACE predictions were generated autoregressively so both methods target the joint posterior. The estimated posteriors are compared based on log-probabilities and $95 \%$ marginal coverage probabilities. The evaluation set includes 1000 examples and we report the mean and (standard deviation) from 10 runs. ACE log-probabilities are on average better than NPE log-probabilities and the coverage probabilities are close to the nominal level 0.95 .

|          | $\log$-probs $(\uparrow)$ | cover $\beta$ | cover $\gamma$ | cover $\phi$ | cover $I_{0}$ |  cover ave   |
| :------: | :-----------------------: | :-----------: | :------------: | :----------: | :-----------: | :----------: |
|   NPE    |       $6.63(0.16)$        | $0.92(0.01)$  |  $0.94(0.01)$  | $0.94(0.01)$ | $0.92(0.01)$  | $0.93(0.01)$ |
| ACE (AR) |       $7.38(0.04)$        | $0.96(0.00)$  |  $0.97(0.00)$  | $0.97(0.00)$ | $0.96(0.00)$  | $0.97(0.00)$ |

Table S4: ACE posterior estimation based on incomplete data with $M$ observation points using either independent or autoregressive predictions. The estimated posteriors are evaluated using (a) log-probabilities and (b) average $95 \%$ marginal coverage probabilities. We report the mean and (standard deviation) from 10 runs. The logprobabilities improve when the context size $M$ increases and when autoregressive predictions are used.

|     |  $M=25$  |    $M=20$    |    $M=15$    |    $M=10$    |    $M=5$     |
| :-: | :------: | :----------: | :----------: | :----------: | :----------: | ------------ |
| (a) |   ACE    | $4.94(0.04)$ | $4.55(0.03)$ | $3.87(0.02)$ | $2.82(0.03)$ | $0.88(0.03)$ |
|     | ACE (AR) | $7.38(0.04)$ | $6.93(0.04)$ | $6.21(0.04)$ | $5.11(0.04)$ | $2.91(0.05)$ |
|     |   ACE    | $0.97(0.00)$ | $0.96(0.00)$ | $0.95(0.00)$ | $0.95(0.00)$ | $0.96(0.00)$ |
|     | ACE (AR) | $0.97(0.00)$ | $0.97(0.00)$ | $0.96(0.00)$ | $0.96(0.00)$ | $0.97(0.00)$ |

estimated based on 5000 samples.
We start with the posterior estimation task where we used ACE and NPE to predict simulator parameters based on the simulated observations with 25 observation points. The results are reported in Table S3. We observe that the ACE log-probabilities are on average better than NPE log-probabilities and that both methods have marginal coverage probabilities close to the nominal level 0.95 .

The simulated observations used in the previous experiment were complete with 25 observation points. Next, we evaluate ACE posteriors estimated based on incomplete data with $5-20$ observation points. NPE is not included in this experiment since it cannot handle incomplete observations. Instead, we use this experiment to compare independent and autoregressive ACE predictions. The results are reported in Table S4. The log-probabilities indicate that both independent and autoregressive predictions improve when more observation points are available while the coverage probabilities are close to the nominal level in all conditions. That autoregressive predictions result in better log-probabilities than independent predictions indicates that ACE is able to use dependencies between simulator parameters.

Table S5: Comparison between ACE and TNP-D in data completion task in the extended SIR model. The estimated predictive distributions are compared based on (a) log-probabilities and (a) $95 \%$ coverage probabilities. We report the mean and (standard deviation) from 10 runs. ACE log-probabilities are on average better than TNP-D log-probabilities and improve both when the context size $M$ increases or when predictions are conditioned on the simulator parameters $\theta$.

|     |        $M=20$         |    $M=15$    |    $M=10$    |    $M=5$     |
| :-: | :-------------------: | :----------: | :----------: | :----------: | ------------ |
| (a) |         TNP-D         | $10.1(0.11)$ | $9.99(0.09)$ | $9.44(0.10)$ | $8.02(0.07)$ |
|     |          ACE          | $14.2(0.31)$ | $13.8(0.31)$ | $13.2(0.31)$ | $11.4(0.28)$ |
|     | $\mathrm{ACE}+\theta$ | $14.7(0.31)$ | $14.6(0.31)$ | $14.6(0.30)$ | $14.3(0.30)$ |
| (b) |         TNP-D         | $0.96(0.00)$ | $0.96(0.00)$ | $0.95(0.00)$ | $0.95(0.00)$ |
|     |          ACE          | $0.97(0.00)$ | $0.96(0.00)$ | $0.96(0.00)$ | $0.95(0.00)$ |
|     | $\mathrm{ACE}+\theta$ | $0.96(0.00)$ | $0.96(0.00)$ | $0.96(0.00)$ | $0.96(0.00)$ |

The same ACE models that have been evaluated in the posterior estimation (latent prediction) task can also make predictions about the unobserved values in incomplete data. To evaluate ACE in the data completion task,

---

#### Page 39

we selected 5 target observations from each evaluation sample and used 5-20 remaining observations as context. We used ACE to make target predictions either based on the context data alone or based on both context data and the simulator parameters $\theta$. For comparison, we also evaluated data completion with TNP-D. The results are reported in Table S5. We observe that ACE log-probabilities are on average better than TNP-D log-probabilities and improve when simulator parameters are available as context. In these experiments, both ACE and TNP-D were used to make independent predictions.

# C. 5 Computational resources and software

For the experiments and baselines, we used a GPU cluster containing AMD MI250X GPUs. All experiments can be run using a single GPU with a VRAM of 50 GB . Most of the experiments took under 6 hours, with the exception of a few BO experiments that took around 10 hours. The core code base was built using Pytorch (Paszke et al., 2019) (https://pytorch.org/ Version: 2.2.0, License: modified BSD license) and based on the Pytorch implementation for TNP (Nguyen and Grover, 2022) (https://github.com/tung-nd/TNP-pytorch, License: MIT). Botorch (Balandat et al., 2020) (https://github.com/pytorch/botorch Version: 0.10.0, License: MIT) was used for the implementation of GP-MES, GP-TS, and $\pi$ BO-TS.
