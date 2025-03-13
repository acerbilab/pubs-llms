```
@article{schmitt2024amortized,
  title={Amortized Bayesian Workflow (Extended Abstract)},
  author={Marvin Schmitt and Chengkun Li and A. Vehtari and Luigi Acerbi and Paul-Christian Burkner and Stefan T. Radev},
  year={2024},
  journal={NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty},
  doi={10.48550/arXiv.2409.04332}
}
```

---

#### Page 1

# Amortized Bayesian Workflow (Extended Abstract)

Marvin Schmitt ${ }^{* 1,3}$ Chengkun $\mathbf{L i}^{* 2}$ Aki Vehtari ${ }^{3}$ Luigi Acerbi ${ }^{2}$ Paul-Christian Bürkner ${ }^{4}$ Stefan T. Radev ${ }^{5}$

## 1 Introduction

In statistics, we often reason about unknown parameters $\theta$ from observables $y$ modeled as a joint distribution $p(\theta, y)$. The posterior $p(\theta \mid y)$ is the statistically optimal solution to this inverse problem, and there are different computational approaches to approximate this costly distribution.

Markov chain Monte Carlo (MCMC) methods constitute the most popular family of sampling algorithms due to their theoretical guarantees and powerful diagnostics [6, 7]. MCMC methods yield autocorrelated draws conditional on a fixed data set $y_{\text {obs }}$. As a consequence, the probabilistic model has to be re-fit for each new data set, which necessitates repeating the entire MCMC procedure from scratch. For such algorithms performed conditionally on a fixed data set, the well-established Bayesian workflow [7] defines an iterative sequence of steps that encompasses model specification, fitting, evaluation, addressing computational issues, modifications, and model comparison.
Differently, amortized Bayesian inference uses deep neural networks to learn a direct mapping from observables $y$ to the corresponding posterior $p(\theta \mid y)$. Amortized inference follows a two-stage approach: (i) a training stage, where neural networks learn to distill information from the probabilistic model based on simulated examples of observations and parameters $(\theta, y) \sim p(\theta) p(y \mid \theta)$; and (ii) an inference stage where the neural networks approximate the posterior distribution for an unseen data set $y_{\text {obs }}$ in near-instant time without repeating the training stage. The Bayesian workflow is not directly transferable to amortized inference because the approximation step is learned over the prior predictive space (see Section 2) while only the inference step is conditional on a fixed data set.

In Bayesian inference, both MCMC (e.g., ChEES-HMC; [9]) and amortized inference lie at the Pareto front of methods that have a favorable trade-off between accuracy and speed. In this paper, we propose an adaptive workflow that yields highquality posterior draws while minimizing the required compute time by moving along the Pareto front to afford fast-andaccurate inference when possible, and slow-but-guaranteedaccurate inference when necessary (see Figure 1). Crucially, our workflow consistently yields high accuracy, as evaluated with tailored diagnostics in all steps. Furthermore, it re-uses computations for subsequent steps in the form of importance sampling proposals and initializations of many-short-chains MCMC. The software implementation encompasses an end-toend workflow featuring model specification via PyMC [15], amortized inference with deep learning via BayesFlow [16], and GPU-enabled ChEES-HMC [9] via Tensorflow Probability [5].

> **Image description.** A graph illustrating the trade-off between accuracy guarantees and inference speed in Bayesian computation methods. The graph has two axes: the vertical axis labeled "Accuracy guarantees" ranging from "weak" to "strong", and the horizontal axis labeled "Inference speed per data set" ranging from "slow" to "instant".
>
> The graph displays a curved red line labeled "Pareto front" that represents the optimal trade-off boundary between these two metrics. Several methods are positioned along this Pareto front, labeled in green text: "ChEES-HMC" at the top left (high accuracy, slower speed), "+ PSIS" in the middle, and "Amortized inference" at the bottom right (faster speed, lower accuracy guarantees).
>
> Gray dots are scattered throughout the space below the Pareto front, representing suboptimal methods. Blue arrows curve around the right side of the graph, suggesting a workflow path that moves adaptively between these methods depending on requirements.
>
> This visualization represents the adaptive workflow approach described in the paper, showing how different Bayesian inference methods balance accuracy and computational efficiency.

Figure 1: Our workflow adaptively moves along the Pareto front and re-uses previous computations.

## 2 Integrating Amortized Inference into the Bayesian Workflow

Our adaptive workflow starts with neural network training to enable subsequent amortized inference on any number of unseen data sets. While this training phase is conceptually identical to standalone amortized inference training, the inference phase features a principled control flow that guides the analysis based on tailored diagnostics in order to select the appropriate inference algorithm for each observed data set while re-using computations along the way.

[^0]
[^0]: \*Equal contribution $\quad{ }^{1}$ University of Stuttgart, GER ${ }^{2}$ University of Helsinki, FI ${ }^{3}$ Aalto University, FI
${ }^{4}$ TU Dortmund University, GER ${ }^{5}$ Rensselaer Polytechnic Institute, US

---

#### Page 2

> **Image description.** This image is a flow diagram illustrating an adaptive workflow for Bayesian inference. It is divided into three main steps, labeled "Step 1", "Step 2", and "Step 3". Each step represents a different sampling algorithm.
>
> - **Step 1: Amortized Inference**: This step involves a "Neural density estimator qφ" which takes "One data set yobs(k)" as input. The estimator produces "S draws θ1(k), ..., θs(k)". A "Diagnostics OK?" check follows. If the diagnostics are satisfactory ("yes"), the "Accept draws" box is highlighted in green. If not ("no"), the process moves to Step 2, indicated by a pink arrow. There's a "re-use draws" blue dashed arrow going to step 2. An "Example draws" plot is shown, displaying a scatter plot with a U-shaped distribution of points. Below this step is a grid of 256 squares, labeled "K=256 observed data sets". Most squares are green, with some pink squares scattered throughout. The caption below states "Accept amortized draws for 192/256 data sets".
>
> - **Step 2: Pareto-smoothed importance sampling**: This step starts with "Importance weights" and "Fit Pareto distribution". A "Diagnostics OK?" check follows. If the diagnostics are satisfactory ("yes"), the "Accept draws" box is highlighted in green. If not ("no"), the process moves to Step 3, indicated by a pink arrow. There's a "re-use draws" blue dashed arrow going to step 3. An "Example draws" plot is shown, displaying a scatter plot with a U-shaped distribution of points. Below this step is a grid of 64 squares, labeled "PSIS on remaining 64 data sets". Most squares are green, with some pink squares scattered throughout. The caption below states "Accept PSIS draws for 56/64 data sets".
>
> - **Step 3: ChEES-HMC with amortized inits**: This step involves "Initialize S superchains" followed by "Warmup and sampling". A "Diagnostics OK?" check follows. If the diagnostics are satisfactory ("yes"), the "Accept draws" box is highlighted in green. If not ("no"), the process moves to a "Use NUTS or revise model" box, indicated by a pink arrow. An "Example draws" plot is shown, displaying a scatter plot with a U-shaped distribution of points. Below this step is a grid of 8 squares, labeled "ChESS-HMC on remaining 8 data sets". All squares are green. The caption below states "Accept ChEES-HMC draws for 8/8 data sets".
>
> The steps are visually separated by gray rounded rectangles. Dashed blue arrows labeled "re-use draws" connect the steps, indicating the transfer of information between them.

Figure 2: Our adaptive workflow leverages near-instant amortized posterior sampling when possible and gradually resorts to slower - but more accurate - sampling algorithms. As indicated by the blue dashed arrows, we re-use the $S$ draws from the amortized posterior in step 1 for the subsequent steps in the form of PSIS proposals (step 2) and initial values in ChEES-HMC (step 3).

# 2.1 Training phase: simulation-based opt

Since most Bayesian models are generative by design, we can readily simulate $M$ tuples of parameters and corresponding observations from the joint model,

$$
\left(\theta^{(m)}, y^{(m)}\right) \sim p(\theta, y) \quad \Leftrightarrow \quad \theta^{(m)} \sim p(\theta), y^{(m)} \sim p(y \mid \theta) \text { for } m=1, \ldots, M
$$

which results in the training set $\left\{\left(\theta^{(m)}, y^{(m)}\right)\right\}_{m=1}^{M}$. The total number $M$ of example tuples is called the training budget, and the quality of the amortized posterior estimator hinges on a sufficient training budget. In the case study, we use flow matching [13] as a flexible neural estimators, but our workflow is agnostic to the exact choice of neural network architecture.
Diagnostics. At this point, there are no observed data sets yet to guide data-conditional diagnostics. However, we can easily simulate a synthetic test set $\left\{\left(\hat{\theta}_{*}^{(j)}, y^{(j)}\right)\right\}_{j=1}^{J}$ of size $J$ from the joint model via Eq. 1. In this closed-world setting, we know which "true" parameter vector $\theta_{*}^{(j)}$ generated each simulated test data set $y^{(j)}$. We evaluate the amortized posterior's bias and variance via the normalized root mean-squared error (NRMSE) and perform simulation-based calibration (SBC; $[19,21])$ checking to evaluate the uncertainty calibration. These evaluations act as a convergence diagnostic to assert that the neural estimator yields faithful posterior draws under idealized conditions (see Appendix A for details). If these closed-world convergence diagnostics fail, we should tune the training hyperparameters (e.g., training duration, simulation budget, neural network architecture).

### 2.2 Inference phase: posterior approximation on observed data sets

We now use the pre-trained neural network to achieve rapid amortized posterior inference on a total of $K$ observed data sets $\left\{y_{\text {obs }}^{(k)}\right\}_{k=1}^{K}$, which naturally do not come with known ground-truth parameters. The diagnostics in this step are evaluated conditional on each observed data set to determine whether the whole set of amortized draws is acceptable for each specific data set.

### 2.2.1 Step 1: Amortized posterior draws

We aim to use the rapid sampling capabilities of the amortized posterior approximator $q_{\phi}$ whenever possible according to the diagnostics. Therefore, the natural first step for each observed data set $y_{\text {obs }}^{(k)}$ is to query the amortized posterior and sample $S$ posterior draws $\hat{\theta}_{1}^{(k)}, \ldots, \hat{\theta}_{S}^{(k)} \sim q_{\phi}\left(\theta \mid y^{(k)}\right)$ in near-instant time (see Figure 2, first panel).
Diagnostics. Amortized inference may yield unfaithful results under distribution shifts [11, 20, 23]. Therefore, we assess whether an observed data set is atypical under the data-generating process of the joint model. We define atypical data as data sets that have a larger maximum mean discrepancy (MMD; [8]) to the training set than $95 \%$ of the training data sets themselves (cf. [20]; see Appendix B for details). Since the amortized approximator has no accuracy guarantees for data outside of the typical set of the joint model, we propagate such atypical data sets to the next step.

[^0]
[^0]: ${ }^{2}$ This data generation scheme is also known as prior predictive sampling.

---

#### Page 3

2.2.2 Step 2: Pareto-smoothed importance sampling

As a first pursuit to improve the quality of the amortized posterior draws with a small overhead in computation time, we use a Pareto-smoothed sampling importance sampling (PSIS) scheme [22] (see Figure 2, second panel). Based on the amortized posterior draws from step 1, we compute the importance weights $w_{s}^{(k)}=p\left(y^{(k)} \mid \hat{\theta}_{s}\right) p\left(\hat{\theta}_{s}\right) / q_{\phi}\left(\hat{\theta}_{s} \mid y^{(k)}\right)$ conditional on each observed data set $y^{(k)}$ and smooth the tail of the weight distribution based on fitting a generalized Pareto distribution (aka. Pareto-smoothing; [22]). These smoothed importance weights are then used for computing posterior expectations and for improving the posterior draws with the sampling importance resampling (SIR) scheme [18]. While the utility of standard importance sampling for improving neural posterior draws has previously been investigated [4], we specifically use the PSIS algorithm which is self-diagnosing and therefore better suited for a principled workflow.
Note. Common neural architectures for amortized inference (e.g., normalizing flows, flow matching) are mode covering. ${ }^{3}$ When the neural network training stage is insufficient (e.g., small simulation budget or poorly optimized network), this may lead to overdispersed posteriors. Fortunately, this errs in the right direction, and PSIS can generally mitigate overdispersed mode-covering draws.

Diagnostics. We use the Pareto- $\hat{k}$ diagnostic to gauge the fidelity of the PSIS-refined posterior draws. According to established guidelines [22, 24], Pareto- $\hat{k} \leq 0.7$ indicates good results, whereas $\hat{k}>0.7$ implies that the draws should be rejected and the respective data sets proceed to step 3.

# 2.2.3 Step 3: ChEES-HMC with amortized initializations

If Pareto-smoothed importance sampling fails according to the diagnostics, we resort to an MCMC sampling scheme which is augmented by re-using computations from the previous steps. Concretely, we use the ChEESHMC algorithm [9] that affords to launch thousands of parallel chains on a GPU. To accelerate convergence, we use the importance weights from step 2 to sample $S$ (e.g., 16) unique draws for initializing $S$ ChEES-HMC superchains ${ }^{4}$, each with $L$ (e.g., 128) subchains for the nested- $\widehat{R}$ diagnostic below. For the purpose of ChEES-HMC initialization, it is also desirable that the amortized posterior draws are generally mode covering (cf. step 2).

> **Image description.** The image is a 2D plot illustrating sampling methods, likely related to Bayesian inference. The plot features several visual elements:
>
> - **Axes:** The horizontal axis is labeled "θ1" and the vertical axis is labeled "θ2".
>
> - **Contour Lines:** A series of nested, gray contour lines represent the "Target posterior". The contours are denser towards the center, suggesting a higher probability density in that region.
>
> - **Amortized Initializations:** These are marked by "X" symbols in various colors (purple, blue, green, and yellow). They are scattered across the plot.
>
> - **ChEES-HMC Samples:** These are represented by dotted lines connecting small circular points. There are several such lines, each starting from a different "Amortized Initialization" and meandering through the space defined by θ1 and θ2. The lines are colored in shades of purple, blue, green, and yellow, corresponding to the colors of the starting "Amortized Initializations".
>
> - **Legend:** A legend in the upper-left corner identifies the different elements: "Amortized initializations" (represented by an "X"), "ChEES-HMC samples" (represented by a dotted line), and "Target posterior" (represented by a solid gray line).

Figure 3: We initialize many ChEESHMC chains with amortized draws.

Diagnostics. In this last step, we use the nested $\widehat{R}$ diagnostic [14] which is specifically designed to assess the convergence of the many-but-short MCMC chains. If the diagnostics in this step indicate unreliable inference, we recommend resorting to the overarching Bayesian workflow [7] and addressing the computational issues that even persist when using the (ChEES-)HMC algorithm. This could involve using the established NUTS-HMC algorithm ([3, 10]) or revising the Bayesian model.

## 3 Empirical Demonstration: Generalized Extreme Value Distribution

In this section, we illustrate the application of the proposed workflow with Bayesian inference on the parameters of a generalized extreme value (GEV) distribution. The GEV distribution is characterized by three parameters: a location parameter $\mu \in \mathbb{R}$, a scale parameter $\sigma \in \mathbb{R}_{>0}$, and a shape parameter $\xi \in \mathbb{R}$, with a cumulative distribution function $G(y)=\exp \left(-\left(1+\xi\left(\frac{y-\mu}{\sigma}\right)\right)^{-1 / \xi}\right)$. Given $N=65$ i.i.d. observations $y=\left(y_{1}, \ldots, y_{65}\right)$ from the GEV distribution, we aim to compute a posterior estimate for the data-generating parameters $\theta=(\mu, \sigma, \xi)$. We first train the amortized posterior approximator on simulated parameters and verify that its closed-world performance is satisfactory, as indexed by high parameter recovery and excellent calibration (see Appendix C).
As summarized in Table 1, we perform inference on a total of $K=1000$ test data sets which are deliberately sampled from a model with a $2 \times$ wider prior distribution to emulate out-of-distribution

[^0]
[^0]: ${ }^{3}$ Conditional flow matching is mode covering [13]. Normalizing flows are mode covering because they optimize the forward KL divergence [17]. In contrast, variational inference algorithms typically optimize the reverse KL divergence, which leads to mode seeking behavior that is less favorable for importance sampling.
${ }^{4}$ If importance sampling resampling without replacement fails to return $S$ valid draws for initializing the chains (e.g., due to less than $S$ non-zero importance weights), we fall back to random initializations.

---

#### Page 4

|                              | Accepted datasets | Time | TPA ${ }^{1}$ |           MMD to reference            |
| :--------------------------- | :---------------: | :--: | :-----------: | :-----------------------------------: |
| Step 1: Amortized inference  |   $678 / 1000$    | 142  |     0.21      | $0.0082[4 \times 10^{-4} \cdot 0.35]$ |
| Step 2: Amortized + PSIS     |    $228 / 322$    | 124  |     0.54      | $0.0010[1 \times 10^{-4} \cdot 0.02]$ |
| Step 3: ChEES-HMC w/ inits   |     $66 / 94$     | 398  |     6.03      | $0.0001[1 \times 10^{-5} \cdot 0.05]$ |
| Total: aggregated over steps |   $972 / 1000$    | 664  |     0.68      |                   -                   |

${ }^{1}$ TPA: time per accepted data set in seconds, computed as the expended time relative to the number of accepted data sets in this step. Table 1: MMD (median, $95 \%$ CI) quantifies the distance between approximate and reference posterior draws. All times are wall-clock seconds on an NVIDIA A100. The time for step 1 includes the training (120s), inference (10s), and diagnostics (12s) stages of the amortized approximator. Our amortized workflow yielded a total of 2 million posterior draws in 11 minutes, whereas using NUTS on all data sets takes approximately 16 hours. While the MMD in step 1 is numerically higher than in steps 2 and 3, spot checks indicated that the posteriors are visually similar to the reference draws.
settings in real applications (see Appendix C for details). In step 1, we draw 2000 posterior samples from the amortized approximator $q_{\mathrm{d}}$, which takes 150 seconds for all 1000 data sets ( 2 million posterior draws in total). We confirm that $678 / 1000$ observed data sets are typical under the datagenerating process and accept the amortized draws. The remaining 322 data sets are passed to stage 2 , where we apply the PSIS algorithm, taking a total of 130 seconds. The Pareto- $k$ diagnostic signals acceptable results for 228 of the 322 data sets, which means that we propagate the remaining 94 data sets to stage 3. Here, we initialize the parallel ChEES-HMC sampler with the amortized draws and observe that the nested $\hat{R}$ values lie below 1.01 for 66 of the data sets, leading to acceptance of the ChEES draws. This leaves only 28 data sets for separate inference with NUTS-HMC. In total, our amortized Bayesian workflow took $\approx 10$ minutes and led to high-quality posterior draws on all steps, as indicated by a small MMD to a reference posterior. In contrast, running NUTS-HMC on all 1000 observed test data sets would have taken $\approx 955$ minutes ( 16 hours), which underscores the efficiency gains of our integrated workflow.

Amortized draws can be good ChEES-HMC inits. To further investigate whether the amortized posterior estimates are indeed beneficial for initializing ChEESHMC chains, we randomly collect 20 test datasets that are passed to step 3 in the workflow. This indicates that both the amortized posterior draws and their Paretosmoothed refinement are deemed unacceptable, as quantified by Pareto- $\hat{k}>0.7$ in step 2 . We initialize the ChEESHMC chains with three different methods: (1) Amortized posterior draws, (2) PSIS-refined amortized draws, and (3) a random initialization scheme similar to Stan [3].We run the chains for different numbers of warmup iterations followed by a single sampling iteration. As described in Section 2, we use the nested $R$ value to gauge whether the chains converged appropriately during the warmup stage (as quantified by common $\hat{R}-1$ thresholds of $10^{-1}$ or $10^{-2}$ ). As shown in Figure 4, amortized posterior draws (and their PSIS-refined counterparts) can significantly reduce the required number of warmup iterations to achieve convergence of ChEES-HMC chains, even though the draws themselves have previously been flagged as unacceptable. This emphasizes that our amortized workflow creates synergies by re-using computations in subsequent steps. However, it is not evident whether initializing ChEES-HMC with the PSIS-refined draws from step 2 has an advantage over using the raw amortized draws from step 1, and we mainly see that PSIS improves the worst-case performance (upper error boundary in Figure 4).

I'll provide a detailed description of the image:

> **Image description.** A logarithmic plot showing the relationship between the number of warmup iterations (x-axis) and Nested R̂-1 values (y-axis) for three different initialization methods in ChEES-HMC chains. The x-axis displays warmup iterations (10, 50, 100, 200, 300, 500) on a non-linear scale, while the y-axis shows Nested R̂-1 values on a logarithmic scale ranging from 10^-3 to 10^1.
>
> Three different initialization methods are represented with distinct markers and line colors:
>
> - Blue triangles connected by a blue line: "Amortized" initialization
> - Yellow/gold diamonds connected by a yellow line: "Amortized + PSIS" initialization
> - Red squares connected by a red line: "Random Initialization"
>
> Two horizontal dashed gray lines mark thresholds at 10^-1 and 10^-2, representing common R̂-1 convergence thresholds.
>
> Each data point includes vertical error bars indicating the interquartile range (IQR) across 20 test datasets. The plot shows that both amortized methods (blue and yellow) converge much faster than random initialization (red). The amortized methods reach the 10^-2 threshold by around 100 warmup iterations, while random initialization requires nearly 500 iterations to reach the same threshold. After about 200 iterations, both amortized methods stabilize around 10^-2 to 10^-3, while random initialization continues to decrease more gradually.

Figure 4: Using amortized posterior draws as inits can reduce the required warmup in ChEES-HMC, but the extent of the benefit varies. The figure shows median ± IQR across 20 test data sets.

# 4 Conclusion

We presented an adaptive Bayesian workflow to combine the rapid speed of amortized inference with the undisputed sampling quality of MCMC in the context of many observed data sets while maintaining a high quality of posterior draws. Our workflow efficiently uses resources by (i) using fast (amortized) inference when the results are accurate; (ii) refining draws with PSIS when possible; and (iii) amortized initializations of slow-but-guaranteed-accurate MCMC chains when needed.
