# Stacking Variational Bayesian Monte Carlo - Appendix

---

#### Page 10

# Appendix A. 

This appendix provides additional details and analyses to complement the main text, included in the following sections:

- A brief overview of relevant existing work, A. 1
- Model descriptions, A. 2
- Metrics description, A. 3
- Black-box variational inference implementation, A. 4
- Additional experiment results, A. 5
- Example posterior visualisations, A. 6
- Further discussion of the ELBO bias mentioned in Section 4, A. 7

## A.1. Related work

Our work addresses the challenge of building global posterior approximations by combining local solutions from the VBMC framework (Acerbi, 2018, 2019, 2020). While the idea of combining posterior distributions has been explored before, previous approaches differ substantially in their goals and methodology. Yao et al. (2022) propose a similar "stacking" approach, but focus on optimising predictive performance through a leave-one-out strategy, whereas S-VBMC optimises the ELBO on the full dataset, allowing treatment of the logjoint as a black box. Other relevant approaches include variational boosting (Guo et al., 2016; Miller et al., 2017; Campbell and Li, 2019), which sequentially builds a mixture posterior by running variational inference multiple times on the whole dataset, and embarrassingly parallel Markov Chain Monte Carlo (MCMC) (Neiswanger et al., 2013; Wang et al., 2015; Scott et al., 2022; De Souza et al., 2022), which combines parallel "sub-posteriors" obtained from data subsets. Our method differs from variational boosting through its inherent parallel and surrogate-based approach, offering significant computational advantages, and from embarrassingly parallel inference methods by using the complete dataset in each run, thus remaining robust to individual run failures.

## A.2. Model descriptions

GMM target. Our synthetic GMM target consists of a mixture of 20 bivariate Gaussian components arranged in four distinct clusters. The cluster centroids were positioned at $(-8,-8),(-7,7),(6,-6)$ and $(5,5)$. Around each centroid, we placed five Gaussian components with means drawn from $\mathcal{N}\left(\boldsymbol{\mu}_{c}, \mathbf{I}\right)$, where $\boldsymbol{\mu}_{c}$ is the respective cluster centroid and $\mathbf{I}$ is the $2 \times 2$ identity matrix. Each component was assigned unit marginal variances and a correlation coefficient of $\pm 0.5$ (randomly selected with equal probability). This configuration produces an irregular mixture structure that requires a substantial number of components to approximate accurately. All components were assigned equal mixing weights of $1 / 20$. The resulting distribution is illustrated in Figure A. 1 (top panels).

---

#### Page 11

Ring target. Our second synthetic target is a ring-shaped distribution defined by the probability density function

$$
p_{\text {ring }}\left(\theta_{1}, \theta_{2}\right) \propto \exp \left(-\frac{(r-R)^{2}}{2 \sigma^{2}}\right)
$$

where $r=\sqrt{\left(\theta_{1}-c_{1}\right)^{2}+\left(\theta_{2}-c_{2}\right)^{2}}$ represents the radial distance from centre $\left(c_{1}, c_{2}\right), R$ is the ring radius, and $\sigma$ controls the width of the annulus. We set $R=8, \sigma=0.1$, and centred the ring at $\left(c_{1}, c_{2}\right)=(1,-2)$. The small value of $\sigma$ produces a narrow annular distribution that challenges VBMC's exploration capabilities. The resulting distribution is shown in Figure A. 1 (bottom panels).

Neuronal model. Our first real-world problem involved fitting five biophysical parameters of a detailed compartmental model of a hippocampal CA1 pyramidal neuron. The model was constructed based on experimental data comprising a three-dimensional morphological reconstruction and electrophysiological recordings of neuronal responses to current injections. The deterministic neuronal responses were simulated using the NEURON simulation environment (Hines and Carnevale, 1997; Hines et al., 2009), applying current step inputs that matched the experimental protocol. The model's parameters characterise key biophysical properties: intracellular axial resistivity $\left(\theta_{1}\right)$, leak current reversal potential $\left(\theta_{2}\right)$, somatic leak conductance $\left(\theta_{3}\right)$, dendritic conductance gradient $\left(\theta_{4}\right.$, per $\left.\mu \mathrm{m}\right)$, and a dendritic surface scaling factor $\left(\theta_{5}\right)$. Based on independent measurements of membrane potential fluctuations, observation noise was modelled as a stationary Gaussian process with zero mean and a covariance function estimated from the data. The covariance structure was captured by the product of a cosine and an exponentially decaying function. For a similar approach applied to cerebellar Golgi cells, see Szoboszlay et al. (2016).

Multisensory causal inference model. Perceptual causal inference involves determining whether multiple sensory stimuli originate from a common source, a problem of particular interest in computational cognitive neuroscience (Körding et al., 2007). Our second real-world problem involved fitting a visuo-vestibular causal inference model to empirical data from a representative participant (S1 from Acerbi et al., 2018). In each trial, participants seated in a moving chair reported whether they perceived their movement direction ( $s_{\text {vest }}$ ) as congruent with an experimentally-manipulated looming visual field ( $s_{\text {vis }}$ ). The model assumes participants receive noisy sensory measurements, with vestibular information $z_{\text {vest }} \sim \mathcal{N}\left(s_{\text {vest }}, \sigma_{\text {vest }}^{2}\right)$ and visual information $z_{\text {vis }} \sim \mathcal{N}\left(s_{\text {vis }}, \sigma_{\text {vis }}^{2}(c)\right)$, where $\sigma_{\text {vest }}^{2}$ and $\sigma_{\text {vis }}^{2}$ represent sensory noise variances. The visual coherence level $c$ was experimentally manipulated across three levels $\left(c_{\text {low }}, c_{\text {med }}, c_{\text {high }}\right)$. The model assumes participants judge the stimuli as having a common cause when the absolute difference between sensory measurements falls below a threshold $\kappa$, with a lapse rate $\lambda$ accounting for random responses. The model parameters $\boldsymbol{\theta}$ comprise the visual noise parameters $\sigma_{\text {vis }}\left(c_{\text {low }}\right), \sigma_{\text {vis }}\left(c_{\text {med }}\right), \sigma_{\text {vis }}\left(c_{\text {high }}\right)$, vestibular noise $\sigma_{\text {vest }}$, lapse rate $\lambda$, and decision threshold $\kappa$ (Acerbi et al., 2018).

# A.3. Metrics description 

Following Acerbi (2020); Li et al. (2024), we evaluate our method using three metrics:

---

#### Page 12

1. The absolute difference between true and estimated log marginal likelihood ( $\Delta \mathrm{LML}$ ), where values $<1$ are considered negligible for model selection (Burnham and Anderson, 2003).
2. The mean marginal total variation distance (MMTV), which measures the average (lack of) overlap between true and approximate posterior marginals across dimensions:

$$
\operatorname{MMTV}(p, q)=\frac{1}{2 D} \sum_{d=1}^{D} \int_{-\infty}^{\infty}\left|p_{d}\left(x_{d}\right)-q_{d}\left(x_{d}\right)\right| d x_{d}
$$

where $p_{d}$ and $q_{d}$ denote the marginal distributions along the $d$-th dimension.
3. The "Gaussianised" symmetrised KL divergence (GsKL), which evaluates differences in means and covariances between the approximate and true posterior:

$$
\operatorname{GsKL}(p, q)=\frac{1}{2 D}\left[D_{\mathrm{KL}}(\mathcal{N}[p] \|\mathcal{N}[q])+D_{\mathrm{KL}}\left(\mathcal{N}[q] \|\mathcal{N}[p])\right]\right.
$$

where $\mathcal{N}[p]$ denotes a Gaussian with the same mean and covariance as $p$.
We consider MMTV $<0.2$ and $\operatorname{GsKL}<\frac{1}{8}$ as target thresholds for reasonable posterior approximation (Li et al., 2024). Ground-truth values are obtained through numerical integration, extensive MCMC sampling, or analytical methods as appropriate for each problem.

# A.4. Black-box variational inference implementation 

Our implementation of black-box variational inference (BBVI) follows Li et al. (2024). For gradient-free black-box models, we cannot use the reparameterisation trick (Kingma and Welling, 2013) to estimate ELBO gradients. Instead, we employ the score function estimator (REINFORCE; Ranganath et al., 2014) with control variates to reduce gradient variance.

The variational posterior is parameterised as a mixture of Gaussians (MoG) with either $K=50$ or $K=500$ components, matching the form used in VBMC. We initialise component means near the origin by adding Gaussian noise $(\sigma=0.1)$ and set all component variances to 0.01 . We optimise the ELBO using Adam (Kingma and Ba, 2014) with stochastic gradients, performing a grid search over Monte Carlo sample sizes $\{1,10,100\}$ and learning rates $\{0.01,0.001\}$. We select the best hyperparameters based on the estimated ELBO.

For fair comparison with VBMC, we set the target evaluation budget to $2000(D+2)$ and $3000(D+2)$ evaluations for noiseless and noisy problems respectively, matching the maximum evaluations used by 40 VBMC runs in total.

---

#### Page 13

# A.5. Additional experiment results 

We present a comprehensive comparison of S-VBMC against VBMC and BBVI in Tables A. 1 and A.2, complementing the visualisations in Figures 2 and 3. For both synthetic problems (Table A.1) and real-world problems (Table A.2), S-VBMC generally demonstrates consistently improved posterior approximation metrics compared to both baselines. However, we observe an increase in $\Delta$ LML error with larger numbers of stacked runs. This increase likely stems from the accumulation of ELBO estimation bias, a phenomenon we analyse in detail in Appendix A.7.

| Algorithm | Benchmarks |  |  |  |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | GMM |  |  | Ring |  |  |
|  | $\Delta$ LML | MMTV | GsKL | $\Delta$ LML | MMTV | GsKL |
|  |  |  |  | Noiseless |  |  |
| BBVI, MoG $(K=50)$ | 0.059 [0.030,0.072] | 0.059 [0.039,0.077] | 0.0083 [0.0015,0.014] | 8.0 [7.2,9.5] | 0.51 [0.48,0.53] | 0.72 [0.70,0.92] |
| BBVI, MoG $(K=500)$ | 0.053 [0.032,0.10] | 0.052 [0.044,0.069] | 0.0087 [0.0030,0.013] | 8.3 [7.2,10.] | 0.47 [0.46,0.49] | 0.67 [0.58,0.79] |
| VBMC | 0.71 [0.7,1.4] | 0.4 [0.36,0.56] | 8.1 [5.8,15] | 1.3 [1.1,1.6] | 0.54 [0.45,0.65] | 11 [4.3,39] |
| S-VBMC (10 runs) | 0.57 [0.016,0.83] | 0.047 [0.038,0.065] | 0.0043 [0.0024,0.0072] | 0.075 [0.028,0.14] | 0.16 [0.14,0.19] | 0.012 [0.0025,0.028] |
| S-VBMC (20 runs) | 0.74 [0.57,0.9] | 0.085 [0.034,0.11] | 0.013 [0.0029,0.026] | 0.2 [0.13,0.3] | 0.19 [0.16,0.19] | 0.016 [0.0097,0.025] |
|  | Noisy $(\sigma=3)$ |  |  |  |  |  |
| BBVI, MoG $(K=50)$ | 0.23 [0.13,0.34] | 0.13 [0.097,0.17] | 0.030 [0.010,0.095] | 4.3 [3.6,4.7] | 0.51 [0.48,0.54] | 1.1 [0.73,1.3] |
| BBVI, MoG $(K=500)$ | 0.27 [0.082,0.40] | 0.10 [0.097,0.12] | 0.019 [0.012,0.031] | 4.7 [4.1,5.4] | 0.93 [0.92,0.94] | 48. [32.,49.] |
| VBMC | 1 [0.7,1.4] | 0.44 [0.41,0.57] | 9.7 [7.2,17] | 1.3 [0.88,1.8] | 0.61 [0.5,0.68] | 36 [5.1,7s+02] |
| S-VBMC (10 runs) | 0.74 [0.57,0.92] | 0.17 [0.12,0.19] | 0.05 [0.021,0.084] | 0.77 [0.65,0.88] | 0.24 [0.21,0.26] | 0.043 [0.031,0.047] |
| S-VBMC (20 runs) | 1.1 [0.98,1.2] | 0.13 [0.12,0.14] | 0.025 [0.021,0.054] | 1.3 [1.3,1.4] | 0.18 [0.18,0.19] | 0.012 [0.0085,0.026] |

Table A.1: Comparison of S-VBMC, VBMC, and BBVI performance on synthetic benchmark problems. Values show median with interquartile ranges in brackets. Bold entries indicate best median performance; multiple entries are bolded when interquartile ranges overlap with the best median.

| Algorithm | Benchmarks |  |  |  |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | Multisensory model $(\sigma=3)$ |  |  | Neuronal model |  |  |
|  | $\Delta$ LML | MMTV | GsKL | $\Delta$ LML | MMTV | GsKL |
| BBVI, MoG $(K=50)$ | 1.7 [1.6,4.3] | 0.11 [0.098,0.13] | 0.17 [0.16,0.20] | 44. [35.,1.0s+02] | 0.60 [0.57,0.63] | 20. [18.,25.] |
| BBVI, MoG $(K=500)$ | 1.8 [1.6,2.3] | 0.31 [0.28,0.32] | 0.53 [0.50,0.54] | $1.7 \mathrm{e}+02$ [1.4e+02,2.4e+02] | 0.67 [0.65,0.69] | 21. [18.,25.] |
| VBMC | 0.27 [0.14,0.44] | 0.18 [0.13,0.21] | 0.19 [0.13,0.3] | 3 [2.9,3.1] | 0.32 [0.31,0.33] | $1.6 \mathrm{e}+02$ [ $73,5.2 \mathrm{e}+02]$ |
| S-VBMC (10 runs) | 2.2 [2.2,2.3] | 0.11 [0.086,0.12] | 0.049 [0.041,0.066] | 0.75 [0.71,0.81] | 0.17 [0.16,0.18] | 1.8 [1.1,2.9] |
| S-VBMC (20 runs) | 2.9 [2.9,3] | 0.095 [0.09,0.099] | 0.046 [0.044,0.051] | 0.087 [0.055,0.11] | 0.14 [0.14,0.15] | 0.46 [0.18,0.89] |

Table A.2: Comparison of S-VBMC, VBMC, and BBVI performance on neuronal and multisensory causal inference models.

## A.6. Example posterior visualisations

Figure A. 1 illustrates how S-VBMC significantly improves the result of a single VBMC run, capturing a larger portion of the target posterior mass as more runs are stacked together.

---

#### Page 14

> **Image description.** The image contains six scatter plots arranged in a 2x3 grid. Each plot has the same axes, labeled θ₁ on the x-axis and θ₂ on the y-axis, both ranging from approximately -12 to 12. The background of each plot is a dark purple color.
> 
> *   **Top Row:** The plots in the top row display a density estimation with contours and scattered red points.
>     *   The first plot is titled "VBMC" and shows four distinct, blob-like shapes with light blue/yellow centers and white contours. A cluster of red points is concentrated in the lower right quadrant.
>     *   The second plot is titled "S-VBMC (5 posteriors)" and also shows four blob-like shapes with light blue/yellow centers and white contours. More red points are scattered throughout the plot, particularly concentrated around the blob shapes.
>     *   The third plot is titled "S-VBMC (20 posteriors)" and shows similar blob shapes with light blue/yellow centers and white contours. The red points are more densely scattered and more closely aligned with the contours of the blob shapes.
> 
> *   **Bottom Row:** The plots in the bottom row display a circular shape.
>     *   The first plot is titled "VBMC". A gray circle is present, but the lower half of the circle is filled with red points.
>     *   The second plot is titled "S-VBMC (5 posteriors)". A gray circle is present, and red points are scattered around the circle, with some concentration along the circle's path.
>     *   The third plot is titled "S-VBMC (20 posteriors)". A gray circle is present, and the red points are densely concentrated along the circle's path, forming a more complete red circle.
> 
> In summary, the image visually compares the performance of VBMC and S-VBMC (with varying numbers of posteriors) in approximating a target density, with the red points representing samples from the posterior approximation.

Figure A.1: Examples of overlap between the ground truth and the posterior when combining different numbers of VBMC runs. The red points indicate samples from the posterior approximation, with the target density depicted with colour gradients in the background.

We further use 'corner plots' (Foreman-Mackey, 2016) to visualise exemplar posterior approximations from different algorithms, including S-VBMC, VBMC and BBVI. These plots depict both one-dimensional marginal distributions and all pairwise two-dimensional marginals of the posterior samples. Example results are shown in Figures A.2, A.3, A.4, and A.5, where orange contours and points represent posterior samples obtained from different algorithms while the black contours and points represent ground truth samples. S-VBMC consistently improves the posterior approximations over standard VBMC and generally outperforms BBVI, showing a closer alignment with the target posterior.

---

#### Page 15

> **Image description.** This image contains four panels, each displaying a set of plots related to statistical modeling. Each panel contains three subplots: a histogram along the top, a contour plot in the center, and another histogram on the right. The panels are arranged in a 2x2 grid.
> 
> Here's a breakdown of each panel:
> 
> *   **Panel (a) VBMC:**
>     *   Top Histogram: Shows two overlapping histograms. One is black, and the other is orange. The black histogram has two distinct peaks, while the orange histogram has a single, broader peak.
>     *   Contour Plot: Displays contours representing a bivariate distribution. There appear to be two distinct clusters of contours, one centered around negative x1 and x2 values, and another centered around positive x1 and x2 values. The contours are shaded in gray.
>     *   Right Histogram: Similar to the top histogram, showing overlapping black and orange histograms. The orange histogram has a single peak, while the black histogram has two.
> 
> *   **Panel (b) S-VBMC (20 runs):**
>     *   Top Histogram: Similar to panel (a), with overlapping black and orange histograms. The black histogram has two peaks, and the orange histogram has a single, broader peak.
>     *   Contour Plot: Shows four distinct clusters of contours, indicating a more complex bivariate distribution than in panel (a). The contours are shaded in orange.
>     *   Right Histogram: Overlapping black and orange histograms, with the orange histogram having two peaks and the black histogram having two peaks.
> 
> *   **Panel (c) VBMC (noisy):**
>     *   Top Histogram: Similar to panel (a), with overlapping black and orange histograms.
>     *   Contour Plot: Similar to panel (a), showing two clusters of contours but with more noise or spread. The contours are shaded in gray.
>     *   Right Histogram: Similar to panel (a), with overlapping black and orange histograms.
> 
> *   **Panel (d) S-VBMC (20 runs, noisy):**
>     *   Top Histogram: Similar to panel (b), with overlapping black and orange histograms.
>     *   Contour Plot: Similar to panel (b), showing four clusters of contours. The contours are shaded in orange.
>     *   Right Histogram: Overlapping black and orange histograms, with the orange histogram having two peaks and the black histogram having two peaks.
> 
> Each contour plot has x1 on the horizontal axis and x2 on the vertical axis, ranging from approximately -10 to 10. The histograms appear to represent the marginal distributions of x1 (top) and x2 (right). The y-axis of the histograms appears to range from 0 to 10.

---

#### Page 16

> **Image description.** This image contains four panels, each showing a visualization of a posterior distribution. Each panel contains three subplots: a histogram above, a contour plot in the center, and another histogram to the right.
> 
> The panels are arranged in a 2x2 grid.
> 
> *   **Panel (e):** The title reads "(e) BBVI, MoG (K = 50)". The top histogram shows a bimodal distribution, with an orange line and a black line overlaid. The central contour plot shows four distinct contours, clustered in the corners. The right histogram is similar to the top one, also showing a bimodal distribution with overlaid orange and black lines. The x-axis of the contour plot is labeled "X1", and the y-axis is labeled "X2". The histograms' x-axis is labeled "X2", and the y-axis ranges from -10 to 10.
> 
> *   **Panel (f):** The title reads "(f) BBVI, MoG (K = 500)". The layout is identical to panel (e). The histograms and contour plot show similar patterns to panel (e), but potentially with slightly different shapes.
> 
> *   **Panel (g):** The title reads "(g) BBVI, MoG (K = 50), noisy". The layout is identical to panel (e). The histograms and contour plot show similar patterns to panel (e), but potentially with slightly different shapes.
> 
> *   **Panel (h):** The title reads "(h) BBVI, MoG (K = 500), noisy". The layout is identical to panel (e). The histograms and contour plot show similar patterns to panel (e), but potentially with slightly different shapes.
> 
> In all panels, the x-axes of the contour plots are labeled "X1", and the y-axes are labeled "X2". The histograms' x-axis is labeled "X2", and the y-axis ranges from -10 to 10.

Figure A.2: $\operatorname{GMM}(D=2)$ example posterior visualisation.

---

#### Page 17

> **Image description.** The image contains four panels, each displaying a set of plots related to VBMC (Variational Bayesian Monte Carlo) and S-VBMC (Sequential VBMC) methods. Each panel consists of three plots: a 2D plot in the center and two histograms, one above and one to the right of the 2D plot.
> 
> *   **Panel (a): VBMC**
>     *   The central plot shows a 2D representation with x1 on the horizontal axis and x2 on the vertical axis, both ranging from approximately -6 to 6. A gray ring-like structure is visible, indicating the distribution of points.
>     *   The top plot is a histogram representing the distribution of x1 values. It shows two distributions, one in black and one in orange. The black distribution is flatter, while the orange distribution is more peaked and shifted towards positive x1 values.
>     *   The right plot is a histogram representing the distribution of x2 values. It also shows two distributions in black and orange. The black distribution is relatively flat, while the orange distribution is more concentrated around x2 values near 0.
>     *   The panel is labeled "(a) VBMC" below the central plot.
> 
> *   **Panel (b): S-VBMC (20 runs)**
>     *   The central plot shows a similar 2D representation as in (a), with x1 and x2 axes. The distribution is again ring-shaped, but with a more defined and less noisy structure compared to (a). There are also orange contours visible within the ring.
>     *   The top plot is a histogram of x1 values, showing two distributions in black and orange. The orange distribution is more similar to the black distribution compared to panel (a).
>     *   The right plot is a histogram of x2 values, with black and orange distributions. The orange distribution is more similar to the black distribution compared to panel (a).
>     *   The panel is labeled "(b) S-VBMC (20 runs)" below the central plot.
> 
> *   **Panel (c): VBMC (noisy)**
>     *   The central plot shows a 2D representation with a ring-shaped distribution, similar to (a) but with a more pixelated or blocky appearance, suggesting the presence of noise.
>     *   The top plot is a histogram of x1 values, showing black and orange distributions. The orange distribution is more peaked and shifted towards positive x1 values.
>     *   The right plot is a histogram of x2 values, with black and orange distributions. The orange distribution is more concentrated around x2 values near 0.
>     *   The panel is labeled "(c) VBMC (noisy)" below the central plot.
> 
> *   **Panel (d): S-VBMC (20 runs, noisy)**
>     *   The central plot shows a 2D representation with a ring-shaped distribution, similar to (b) but with a more pixelated appearance, indicating noise. There are orange contours visible within the ring.
>     *   The top plot is a histogram of x1 values, showing black and orange distributions.
>     *   The right plot is a histogram of x2 values, with black and orange distributions.
>     *   The panel is labeled "(d) S-VBMC (20 runs, noisy)" below the central plot.
> 
> In summary, the image compares the performance of VBMC and S-VBMC methods, both with and without noise, by visualizing the distributions of x1 and x2. The S-VBMC method appears to produce more refined distributions, even in the presence of noise. The orange distributions appear to represent the learned distributions, while the black distributions may represent the true distributions.

---

#### Page 18

> **Image description.** The image presents a figure composed of four panels, arranged in a 2x2 grid. Each panel displays a visualization of a posterior distribution, likely from a Bayesian inference process. Each panel contains three subplots: a 2D scatter plot in the center, and two histograms positioned above and to the right of the scatter plot, representing marginal distributions.
> 
> *   **Panel Structure:** Each panel is structured identically:
>     *   The central plot is a 2D scatter plot with axes labeled x1 and x2, ranging from -6 to 6. The data points form a ring-like structure, with varying density indicated by grayscale shading. There are also some contour lines overlaid on the scatter plot in the top left panel.
>     *   The top subplot is a histogram representing the marginal distribution of x1. It displays two histograms, one in black and one in orange.
>     *   The right subplot is a histogram representing the marginal distribution of x2. It also displays two histograms, one in black and one in orange.
> 
> *   **Panel Labels:** Each panel has a label below it indicating the method used and parameters:
>     *   Panel (e): "BBVI, MoG (K = 50)"
>     *   Panel (f): "BBVI, MoG (K = 500)"
>     *   Panel (g): "BBVI, MoG (K = 50), noisy"
>     *   Panel (h): "BBVI, MoG (K = 500), noisy"
> 
> *   **Visual Differences:** The main differences between the panels are in the shape and smoothness of the histograms, and the density of the points in the ring-like structure in the scatter plots. The panels labeled "noisy" appear to have a less defined ring structure and more spread in the distributions. The panels with K=500 appear to have sharper histograms.

Figure A.3: Ring $(D=2)$ example posterior visualisation.

---

#### Page 19

> **Image description.** The image contains two triangular grid plots, one above the other. Each plot displays the pairwise relationships between five variables, labeled x1, x2, x3, x4, and x5.
> 
> Each triangular grid plot consists of a 5x5 grid of subplots. The diagonal subplots display histograms of individual variables, with the variable name (x1 to x5) labeled on the x-axis. The off-diagonal subplots show scatter plots or contour plots representing the joint distribution of two variables. The x-axis of each off-diagonal subplot corresponds to the variable in the column, and the y-axis corresponds to the variable in the row.
> 
> In each subplot, there are two lines or contours: one in black and one in orange. These likely represent two different distributions or estimates of the variables. The scatter plots show a cloud of points, with the density of points indicating the probability density of the joint distribution. The contour plots show lines of equal probability density.
> 
> The top triangular grid plot is labeled "(a) VBMC" at the bottom. The bottom triangular grid plot is labeled "(b) S-VBMC (20 runs)" at the bottom.
> 
> The axes for x1 range from approximately 30 to 45. The axes for x2 range from approximately 1e-5 to 8. The axes for x3 range from approximately 2.5 to 3.5. The axes for x4 range from approximately 0.00 to 0.08. The axes for x5 range from approximately -64.8 to -64.2.

---

#### Page 20

> **Image description.** The image contains two similar triangular plots, arranged one above the other. Each plot visualizes the posterior distribution of five variables (x1 to x5) using a combination of histograms and contour plots.
> 
> Each triangular plot consists of a 5x5 grid of subplots. The diagonal subplots (from top-left to bottom-right) display histograms of the marginal posterior distributions for each variable (x1 to x5). The off-diagonal subplots display contour plots representing the joint posterior distributions between pairs of variables. The lower triangle of the grid is filled with these plots, while the upper triangle is empty.
> 
> The histograms on the diagonal show the distribution of each variable. Each histogram contains two overlaid distributions: one in black and one in orange. The contour plots show the relationship between pairs of variables. These plots also contain two sets of contours, one in black and one in orange. Additionally, the contour plots contain a faint gray scatter plot.
> 
> The x-axis labels for the bottom row of plots are x1, x2, x3, x4, and x5, respectively. The y-axis labels for the leftmost column of plots are x2, x3, x4, and x5, respectively. The x-axis values are labeled with numerical values.
> 
> The plot in the top half of the image is labeled "(c) BBVI, MoG (K = 50)". The plot in the bottom half of the image is labeled "(d) BBVI, MoG (K = 500)".

Figure A.4: Neuronal model $(D=5)$ example posterior visualisation.

---

#### Page 21

> **Image description.** The image shows two sets of plots arranged in a triangular matrix format, a common visualization for exploring relationships between multiple variables. Each set of plots represents a different method: VBMC (top) and S-VBMC (bottom).
> 
> Each triangular matrix consists of six variables, labeled x1 through x6 along the axes.
> 
> *   **Diagonal Plots:** The plots along the diagonal are histograms, showing the marginal distribution of each variable. Each histogram contains two lines: one in black and one in orange.
> 
> *   **Off-Diagonal Plots:** The plots off the diagonal are scatter density plots, showing the joint distribution of two variables. These plots feature a cloud of orange points representing the data density, surrounded by black contour lines.
> 
> *   **Text Labels:** The x-axis labels are x1, x2, x3, x4, x5, and x6. The y-axis labels are x2, x3, x4, x5, and x6. The subplots are labeled as (a) VBMC and (b) S-VBMC (20 runs).

---

#### Page 22

> **Image description.** The image shows two sets of plots, arranged in a matrix format, visualizing posterior distributions. The top set of plots is labeled "(c) BBVI, MoG (K = 50)" and the bottom set is labeled "(d) BBVI, MoG (K = 500)". Each set contains six variables, denoted as x1 through x6, with each variable represented along the diagonal as a histogram and the off-diagonal elements showing pairwise scatter plots with density contours.
> 
> Each set of plots is a 6x6 grid. The diagonal elements, from top-left to bottom-right, display histograms for x1, x2, x3, x4, x5, and x6 respectively. These histograms show the distribution of each variable, with a black line and an orange line representing two different distributions. The off-diagonal plots are scatter plots showing the joint distributions of pairs of variables. For example, the plot in the first row and second column shows the joint distribution of x1 and x2. These scatter plots are visualized using density contours, with both black and orange contours representing different distributions. The axes of the scatter plots are labeled with the values of the corresponding variables. The x-axis of the bottom row of plots is labeled x1, x2, x3, x4, x5, and x6. The y-axis of the left column of plots is labeled x2, x3, x4, x5, and x6.

Figure A.5: Multisensory model $(D=6, \sigma=3)$ example posterior visualisation.

---

#### Page 23

# A.7. ELBO bias 

Here we analyse the ELBO overestimation observed in our results through a simplified example that illustrates one potential mechanism for this bias. While other factors may contribute, this analysis provides insight into why the bias tends to increase with the number of merged VBMC runs.

Consider $M$ VBMC runs that return identical posteriors each with a single component. The stacked posterior takes the form:

$$
q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})=\sum_{m=1}^{M} \hat{w}_{m} q_{m}(\boldsymbol{\theta})
$$

For each single-component posterior, the expected log-joint is approximated as

$$
I_{m}=\mathbb{E}_{q \boldsymbol{\phi}_{m}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})] \approx \mathbb{E}_{q_{\boldsymbol{\phi}_{m}}}\left[f_{m}(\boldsymbol{\theta})\right]
$$

where $f_{m}(\boldsymbol{\theta})$ is the surrogate log-joint from the $m$-th VBMC run. Since all posteriors share identical parameters, their entropies are equal:

$$
\mathcal{H}\left[q_{\boldsymbol{\phi}_{1}}(\boldsymbol{\theta})\right]=\mathcal{H}\left[q_{\boldsymbol{\phi}_{2}}(\boldsymbol{\theta})\right]=\ldots=\mathcal{H}\left[q_{\boldsymbol{\phi}_{M}}(\boldsymbol{\theta})\right]
$$

The stacked posterior is thus a mixture of identical components with different associated values $I_{m}$. The optimal mixture weights $\hat{\mathbf{w}}$ depend solely on the noisy estimates of $I_{m}$ :

$$
\hat{I}_{m}=\mathbb{E}_{q_{\phi_{m}}}\left[f_{m}(\boldsymbol{\theta})\right]=\mathbb{E}_{q_{\phi_{m}}}[\log p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})]+\epsilon_{m}
$$

where $\epsilon_{m} \sim \mathcal{N}\left(0, J_{m}\right)$ represents estimation noise with variance $J_{m}$. Since all posteriors are identical and derived from the same data and model, differences in expected log-joint estimates arise purely from noise deriving from the Gaussian process surrogates $f_{m}$.

Given that entropy remains constant under merging, optimizing $\mathrm{ELBO}_{\text {stacked }}$ reduces to selecting the posterior with the highest expected log-joint estimate. If we denote $\hat{I}_{\max }=$ $\max _{m} \hat{I}_{m}$, the optimal ELBO becomes

$$
\operatorname{ELBO}_{\text {stacked }}^{*}=\hat{I}_{\max }+\mathcal{H}\left[q_{\hat{\boldsymbol{\phi}}}(\boldsymbol{\theta})\right]
$$

Since the true expected log-joint is identical across posteriors, the optimisation selects the most overestimated value. The magnitude of this overestimation increases with both $M$ and the observation noise for $f_{m}$, introducing a positive bias in $\mathrm{ELBO}_{\text {stacked }}^{*}$ that grows with the number of stacked runs and is more substantial for surrogates of noisy log-likelihoods.

While this simplified scenario does not capture the complexity of practical applications - where posteriors have multiple, non-overlapping components - it illustrates a fundamental issue: if we model each $\hat{I}_{m, k}$ as the sum of the true $I_{m, k}$ and noise, the merging process will favour overestimated components, biasing the final $\mathrm{ELBO}_{\text {stacked }}$ estimate upward. Further work is needed to develop debiasing techniques to counteract this tendency.