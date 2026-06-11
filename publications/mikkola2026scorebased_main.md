```
@misc{mikkola2025scorebased,
  title={Score-Based Density Estimation from Pairwise Comparisons},
  author={Petrus Mikkola and Luigi Acerbi and Arto Klami},
  year={2025},
  eprint={2510.09146},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.09146}
}
```

---

# SCORE-BASED DENSITY ESTIMATION FROM PAIRWISE COMPARISONS

Petrus Mikkola Department of Computer Science University of Helsinki petrus.mikkola@helsinki.fi

Luigi Acerbi\\* Department of Computer Science University of Helsinki luigi.acerbi@helsinki.fi

Arto Klami\\* Department of Computer Science University of Helsinki arto.klami@helsinki.fi

## ABSTRACT

We study density estimation from pairwise comparisons, motivated by expert knowledge elicitation and learning from human feedback. We relate the unobserved target density to a tempered winner density (marginal density of preferred choices), learning the winner's score via score- matching. This allows estimating the target by 'de- tempering' the estimated winner density's score. We prove that the score vectors of the belief and the winner density are collinear, linked by a position- dependent tempering field. We give analytical formulas for this field and propose an estimator for it under the Bradley- Terry model. Using a diffusion model trained on tempered samples generated via score- scaled annealed Langevin dynamics, we can learn complex multivariate belief densities of simulated experts, from only hundreds to thousands of pairwise comparisons.

## 1 INTRODUCTION

Several complementary techniques, from flows (Rezende & Mohamed, 2015; Lipman et al., 2023) to diffusion models (Ho et al., 2020), can today efficiently learn complex densities \(p(\mathbf{x})\) from examples \(\mathbf{x}\sim p(\mathbf{x})\) . With sufficiently large data, we can learn accurate densities even over high- dimensional spaces, such as natural images (Rombach et al., 2022). While challenges persist in the most complex cases, these models have achieved a high level of performance, proving sufficient for many tasks.

We consider the fundamentally more challenging problem of learning the density not from direct observations but solely from comparisons of two candidates. Given \(\mathbf{x}\) and \(\mathbf{x}^{\prime}\) that are not sampled from the target \(p(\mathbf{x})\) but rather from a distinct sampling distribution \(\lambda (\mathbf{x})\) satisfying suitable regularity conditions, the task is to learn \(p(\mathbf{x})\) from triplets \((\mathbf{x},\mathbf{x}^{\prime},\mathbf{x}\succ \mathbf{x}^{\prime})\) . The last entry indicates which alternative has higher density (the winner point). Being able to do this enables cognitively easy elicitation of subjective beliefs of an individual over random vectors. The canonical use- case is encoding expert knowledge into statistical models as prior information, with established literature in statistics dedicated to this problem of prior elicitation (O'Hagan, 2019; Mikkola et al., 2023). Here the belief is typically over a relatively low- dimensional space, but it needs to be inferred from a very limited number of observations to keep the expert effort manageable. Recently, elicitation tools have been increasingly used to quantify large language model (LLM) knowledge in probabilistic terms (Capstick et al., 2025; Requeima et al., 2024), for instance, to evaluate calibration or to use them as probabilistic cognitive models (Binz & Schulz, 2024) or forecasting models (Halawi et al., 2024). The current methods require dedicated techniques for direct prompting of probabilities, samples (Requeima et al., 2024) or moments (Capstick et al., 2025), whereas our formulation only requires comparative queries that LLMs can readily answer. Finally, the problem setup is also related to learning from human feedback (Ouyang et al., 2022), in particular to learning an implicit preference distribution for a generative model (Dumoulin et al., 2024).

> **Image description.** A four-panel technical diagram, labeled (a) through (d), illustrating a computational process for density estimation and distribution refinement, likely within the context of score-based generative models. The panels are arranged horizontally, showing a progression from a subjective belief to a refined, tempered distribution.
>
> **Panel (a): Problem setup**
> This panel depicts the initial subjective belief. It features a dark purple background with white text and lines. Two points, labeled 'A' and 'B', are shown. Curved arrows connect these points, indicating a preference or comparison. The text "my belief" is displayed prominently, and a directional arrow points from A toward B, accompanied by the phrase "more probable," visually representing an expert's subjective preference between two configurations.
>
> **Panel (b): MWD (Marginal Winner Density)**
> This panel shows the initial estimated distribution, labeled "MWD." It consists of a dense cloud of small, bright blue dots scattered across a dark background. The dots are concentrated in an irregular, somewhat elongated shape, representing the sampled distribution of winner points.
>
> **Panel (c): Tempering Field**
> This panel displays the estimated tempering field. It is a smooth, symmetrical, glowing field of color. The field is characterized by two prominent, curved, bright lobes that resemble opposing crescents or a stylized hourglass. The color transitions from deep purple and black at the edges to bright yellow and orange at the centers, indicating a gradient or influence field.
>
> **Panel (d): 'Tempered' MWD**
> The final panel, labeled "'Tempered' MWD," shows the result of applying the tempering field to the initial distribution. It contains a dense cloud of bright blue dots, similar to Panel (b), but the distribution is now clearly shaped by the two lobes of the tempering field shown in Panel (c). The samples are concentrated within the two curved, high-density regions, demonstrating how the tempering field refines the original distribution.
>
> The overall visual flow suggests that the initial subjective belief (a) is used to generate the MWD (b), which is then modified by the estimated tempering field (c) to produce the final, refined 'Tempered' MWD (d).

<center>Figure 1: (a) Problem setup. An expert holds a subjective belief over a parameter space, such as the likely hyperparameters of a learning algorithm (e.g. learning rate and weight decay), and can answer questions like "Do you expect configuration A or B to work better?". We learn their belief as a density, to be used e.g. as a prior distribution for finding optimal hyperparameters. (b)-(d) Density estimation from 200 uniformly sampled pairwise comparisons, with the target density shown as a heatmap. (b) Samples and the score field at an intermediate noise level \(\sigma\) , for a diffusion model trained on the (winner, loser) pairs to model the marginal winner density (MWD). (c) Estimated tempering field. (d) Samples from the score-scaled annealed Langevin dynamics with the MWD score and a tempering field estimate. Samples align well with the target density, demonstrating the fundamental relationship between the scores of the estimable MWD and the latent target (belief density). </center>

Recently, Mikkola et al. (2024) proposed the first solution for this problem, learning normalizing flows from pairwise comparisons and rankings. We propose an improved solution that also uses random utility models (RUMs; Train, 2009) for modeling the preferential data and is inspired by their idea of relating the target density \(p(\mathbf{x})\) to a tempered version of the distribution of winner points, \(p_{w}^{\tau}(\mathbf{x})\) , for some tempering parameter \(\tau \geq 1\) . Since we have samples from \(p_{w}(\mathbf{x})\) , this relationship leads to practical algorithms once \(\tau\) is estimated. In contrast to their empirically motivated heuristic link, we characterize this connection in detail and provide an exact relationship between the scores of \(p(\mathbf{x})\) and \(p_{w}(\mathbf{x})\) . Since the relationship holds for the scores, it is natural to also switch to solving the problem with score- based models (Song & Ermon, 2019; Song et al., 2021), instead of flows. This brings additional benefits, for instance in modeling multimodal targets, and we empirically demonstrate a substantial improvement in accuracy compared to Mikkola et al. (2024). While they could learn densities from a modest number of rankings, they needed additional regularization to avoid escaping probability mass (Nicoli et al., 2023). Moreover, their best accuracy required more informative multiple- item rankings. In contrast, we focus solely on pairwise comparisons, which are easier to answer and more reliable (Kendall & Babington Smith, 1940; Shah & Oppenheimer, 2008), and widely used in AI alignment (Ouyang et al., 2022; Wallace et al., 2024).

Denote by \(p_{\mathbf{x}\sim \mathbf{x}^{\prime}}(\mathbf{x},\mathbf{x}^{\prime})\) the joint density of the available data, encoding the preferred candidate in the order of the arguments. The marginal winner density (MWD), denoted by \(p_{w}(\mathbf{x})\) , is obtained as its marginal as \(p_{w}(\mathbf{x}) = \int p_{\mathbf{x}\sim \mathbf{x}^{\prime}}(\mathbf{x},\mathbf{x}^{\prime})d\mathbf{x}^{\prime}\propto \int \mathbb{P}(\mathbf{x}\succ \mathbf{x}^{\prime})\lambda (\mathbf{x})\lambda (\mathbf{x}^{\prime})d\mathbf{x}^{\prime}\) where \(\lambda (\mathbf{x})\) is the sampling density of the (independent) candidates. Our main theoretical contribution is a novel, exact relationship between the target \(p(\mathbf{x})\) and the MWD \(p_{w}(\mathbf{x})\) in terms of their scores: up to a reparameterization of the space, we have \(\nabla \log p(\mathbf{x}) = \tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x})\) . Critically, \(\tau (\mathbf{x})\) is not constant but a position- dependent tempering field. This implies we can perfectly recover \(p(\mathbf{x})\) from the estimable \(p_{w}(\mathbf{x})\) with score- based methods if the tempering field is known. We prove this foundational relationship for the popular Bradley- Terry model (Bradley & Terry, 1952; Touvron et al., 2023) and an exponential noise RUM, providing explicit formulas for the tempering fields.

Our second contribution is a practical algorithm derived from our theoretical insights. First, we propose to model the preference relationships by estimating the score of the joint density \(p_{\mathbf{x}\sim \mathbf{x}^{\prime}}(\mathbf{x},\mathbf{x}^{\prime})\) , then train a continuous- time diffusion model (Karras et al., 2022) to recover the MWD by marginalizing it. Building on the ideal tempering field under the Bradley- Terry model, we estimate the tempering field \(\tau (\mathbf{x})\) by using the analytical formula with importance samples from the trained MWD model and a simple density ratio model trained on the pairwise comparison data. Finally, we sample from the belief density \(p(\mathbf{x})\) by running score- scaled annealed Langevin dynamics (Song & Ermon, 2019) with the MWD score and \(\tau (\mathbf{x})\) . Fig. 1 illustrates our approach.

## 2 BACKGROUND

### 2.1 DENOISING SCORE MATCHING AND ANNEALED LANGEVIN DYNAMICS

The (Stein) score of a probability density function \(p(\mathbf{x})\) , denoted \(\nabla_{\mathbf{x}}\log p(\mathbf{x})\) , is a vector field pointing in the direction of maximum log- density increase. Score- based generative methods approximate this score. They typically start by defining a family of perturbed densities \(p_{\sigma}(\mathbf{x})\) by convolving \(p(\mathbf{x})\) with noise at varying levels \(\sigma >0\) ; for example, \(p_{\sigma}(\mathbf{x}) = p(\mathbf{x})*\mathcal{N}(\mathbf{x};\mathbf{0},\sigma^{2}\mathbf{I})\) , where \(*\) denotes convolution. A neural network \(\mathbf{s}_{\theta}(\mathbf{x},\sigma)\) with parameters \(\theta\) is then trained to model the score of these perturbed densities, \(\nabla_{\mathbf{x}}\log p_{\sigma}(\mathbf{x})\) . This score network \(\mathbf{s}_{\theta}\) is commonly trained through denoising score matching (Vincent, 2011), by minimizing the objective:

\[\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x})}\mathbb{E}_{\sigma \sim p_{\mathrm{train}}(\sigma)}\mathbb{E}_{\tilde{\mathbf{x}}\sim p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x})}\ell (\sigma)\left\| \nabla_{\tilde{\mathbf{x}}}\log p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) - \mathbf{s}_{\theta}(\tilde{\mathbf{x}},\sigma)\right\|^{2}. \quad (1)\]

Here, \(\tilde{\mathbf{x}}\) is a noisy version of a clean sample \(\mathbf{x}\) , generated via the perturbation kernel \(p_{\sigma}(\tilde{\mathbf{x}} |\mathbf{x})\) (e.g., an isotropic Gaussian \(\mathcal{N}(\tilde{\mathbf{x}};\mathbf{x},\sigma^{2}\mathbf{I})\) ). The network is trained to predict the score of \(p_{\sigma}\) by minimizing the objective in Eq. 1, where the perturbation kernel is typically tractable. The function \(\ell (\sigma)\) provides a positive weighting for different noise levels. The noise levels \(\sigma\) are drawn from a distribution \(p_{\mathrm{train}}(\sigma)\) following either a discrete, often uniform schedule \((\sigma_{t})_{t = 1}^{T}\) (Song & Ermon, 2019), or a continuous one (Karras et al., 2022).

Once trained, \(\mathbf{s}_{\theta}(\mathbf{x},\sigma)\) enables sampling from an approximation of \(p(\mathbf{x})\) . One prominent method, besides reverse diffusion processes (discussed later), is annealed Langevin dynamics (ALD) (Song & Ermon, 2019). ALD starts with samples \(\mathbf{x}_{T}^{(0)}\) from a broad prior (e.g., \(\mathcal{N}(\mathbf{x}\mid \mathbf{0},\sigma_{\mathrm{max}}^{2}\mathbf{I})\) ) and iteratively refines them. It runs \(L\) steps of Langevin MCMC per noise level \(\sigma_{t}\) along a decreasing schedule \(\sigma_{\mathrm{max}} = \sigma_{T} > \ldots >\sigma_{1} = \sigma_{\mathrm{min}}\) :

\[\mathbf{x}_{t}^{(l)} = \mathbf{x}_{t}^{(l - 1)} + \epsilon_{t}\mathbf{s}_{\theta}(\mathbf{x}_{t}^{(l - 1)},\sigma_{t}) + \sqrt{2\epsilon_{t}}\mathbf{n}_{t}^{(l)},\quad l = 1,2,\ldots ,L, \quad (2)\]

with step size \(\epsilon_{t} > 0\) and \(\mathbf{n}_{t}^{(l)}\sim \mathcal{N}(\mathbf{0},\mathbf{I})\) . For \(t< T\) , \(\mathbf{x}_{t}^{(0)} = \mathbf{x}_{t + 1}^{(L)}\) . Under ideal conditions \((L\rightarrow \infty\) \(\epsilon_{t}\rightarrow 0\) , accurate \(\mathbf{s}_{\theta}\) ), \(\mathbf{x}_{t}^{(L)}\) approximates a sample from \(p_{\sigma_{\mathrm{min}}}(\mathbf{x})\approx p(\mathbf{x})\) (Welling & Teh, 2011).

### 2.2 DIFFUSION MODELS

A continuous- time diffusion model describes a forward process that gradually transforms a data distribution \(p(\mathbf{x})\) into a simple, known prior distribution (e.g., a Gaussian). This process is often defined by a forward- time stochastic differential equation (SDE) (Song et al., 2021):

\[d\mathbf{x} = f(\mathbf{x},t)d t + g(t)d\mathbf{b},\]

where \(\mathbf{b}\) is Brownian motion, \(f(\mathbf{x},t)\) is the drift coefficient, and \(g(t)\) is the diffusion coefficient. If \(\mathbf{x}(0)\sim p(\mathbf{x})\) (the target density), its time- evolved density is \(p_{t}(\mathbf{x})\) . If \(f\) is an affine transformation, then the transition kernel \(p(\mathbf{x}(t)|\mathbf{x}(0))\) is Gaussian and for a sufficiently large \(T > 0\) , the marginal distribution \(p_{T}(\mathbf{x}(T))\) becomes a pure Gaussian, such as \(\mathcal{N}(\mathbf{0},\mathbf{I})\) or \(\mathcal{N}(\mathbf{0},T^{2}\mathbf{I})\) .

The forward process can be reversed to generate data. Starting from a sample \(\mathbf{x}_{T}\sim p_{T}(\mathbf{x})\) , one can obtain a sample \(\mathbf{x}_{0}\sim p(\mathbf{x})\) by solving the corresponding reverse- time SDE (Anderson, 1982):

\[d\mathbf{x} = \left(f(\mathbf{x},t) - g^{2}(t)\nabla_{\mathbf{x}}\log p_{t}(\mathbf{x})\right)dt + g(t)d\mathbf{b}, \quad (3)\]

where \(\tilde{\mathbf{b}}\) is Brownian motion with time flowing backward from \(T\) to 0. Alternatively, samples can be generated by solving the deterministic probability flow ODE (Song et al., 2021),

\[d\mathbf{x} = \left(f(\mathbf{x},t) - \frac{1}{2} g^{2}(t)\nabla_{\mathbf{x}}\log p_{t}(\mathbf{x})\right)dt. \quad (4)\]

Both reverse methods require the score function \(\nabla_{\mathbf{x}}\log p_{t}(\mathbf{x})\) , typically approximated by a trained score network, \(\mathbf{s}_{\theta}(\mathbf{x},t)\) or \(\mathbf{s}_{\theta}(\mathbf{x},\sigma)\) if parameterized by noise level \(\sigma\) .

The Elucidating Diffusion Models (EDM) framework (Karras et al., 2022; 2024a) parametrizes the diffusion process directly using the noise level \(\sigma \in [\sigma_{\mathrm{min}},\sigma_{\mathrm{max}}]\) rather than an abstract time \(t\) . This can be achieved by assuming \(g(t) = \sqrt{2t}\) and \(f(\mathbf{x},t) = \mathbf{0}\) , and using the initial condition \(\mathbf{x}_{T}\sim \mathcal{N}(\mathbf{0},\sigma_{\mathrm{max}}^{2}\mathbf{I})\) for some fixed, sufficiently large \(\sigma_{\mathrm{max}} > 0\) . The perturbed density can be written as \(p_{t}(\mathbf{x}) = p_{\sigma}(\mathbf{x}) = p(\mathbf{x})*\mathcal{N}(\mathbf{x};\mathbf{0},\sigma^{2}\mathbf{I})\) .

The score network \(\mathbf{s}_{\theta}(\mathbf{x},\sigma)\) is trained via denoising score matching (Eq. 1). Sampling is done by solving the stochastic reverse diffusion SDE (Eq. 3) or the deterministic probability flow ODE (Eq. 4).

### 2.3 RANDOM UTILITY MODELS AND DENSITY ESTIMATION FROM CHOICE DATA

In the context of decision theory, the random utility model (RUM) represents the decision maker's stochastic utility \(U\) as the sum of a deterministic utility and a stochastic perturbation (Train, 2009),

\[U(\mathbf{x}) = u(\mathbf{x}) + W(\mathbf{x}),\]

where \(u:\mathcal{X}\to \mathbb{R}\) is a deterministic utility function, \(W\) is a stochastic noise process, and the choice space \(\mathcal{X}\) is a compact subset of \(\mathbb{R}^{d}\) . Given a set \(C\subset \mathcal{X}\) of possible alternatives, the decision maker selects an alternative \(\mathbf{x}\in C\) by solving the noisy utility maximization problem: \(\mathbf{x}\sim \arg \max_{\mathbf{x}^{\prime}\in C}U(\mathbf{x}^{\prime})\) . Pairwise comparison is the most common form of choice data and corresponds to assuming that the choice set contains only two alternatives, \(C = \{\mathbf{x},\mathbf{x}^{\prime}\}\) . The decision maker chooses \(\mathbf{x}\) from \(\mathcal{C}\) , denoted by \(\mathbf{x}\succ \mathbf{x}^{\prime}\) , if \(u(\mathbf{x}) + w(\mathbf{x}) > u(\mathbf{x}^{\prime}) + w(\mathbf{x}^{\prime})\) for a given realization \(w\) of \(W\) . It is often assumed that \(W\) is independent across \(\mathbf{x}\) , leading to so- called Fechnerian models (Becker et al., 1963), where the choice distribution conditional on \(\mathcal{C}\) reduces to \(F(u(\mathbf{x}) - u(\mathbf{x}^{\prime}))\) , with \(F\) denoting the cumulative distribution function of \(W(\mathbf{x}^{\prime}) - W(\mathbf{x})\) .

Psychophysical experiments suggest that human perception of numerical magnitude follows a logarithmic scale (Dehaene, 2003). Assuming a RUM with utility function \(u(\mathbf{x}) = \log p(\mathbf{x})\) , the model's noise becomes additive in the log- transformed beliefs. In this paper, we consider two RUMs, explicitly including the noise level, as it is crucial for identifying \(p(\mathbf{x})^1\) . First, we study the generalized Bradley- Terry model (Bradley & Terry, 1952) with \(W\sim \mathrm{Gumbel}(0,s)\) , which induces the conditional choice distribution \(F_{\mathrm{Logistic}(0,s)}(u(\mathbf{x}) - u(\mathbf{x}^{\prime}))\) . Second, we consider the exponential RUM with \(W\sim \mathrm{Exp}(s)\) , which yields a heavier- tailed conditional choice distribution \(F_{\mathrm{Laplace}(0,1 / s)}(u(\mathbf{x}) - u(\mathbf{x}^{\prime}))\) .

Under these assumptions, the density estimation task is an instance of expert knowledge elicitation (O'Hagan, 2019; Mikkola et al., 2023), and it is closely related to (probabilistic) reward modeling (Leike et al., 2018; Dumoulin et al., 2024). Expert knowledge elicitation infers an expert's belief as a probability density \(p(\mathbf{x})\) using only queries they can reliably answer, such as requests for specific quantiles or statistics of \(p(\mathbf{x})\) (O'Hagan, 2019; Bockting et al., 2025) or comparisons like here. Recently, Dumoulin et al. (2024) reinterpreted reward modeling by referring to the target distribution as the "implicit preference distribution" and treating the reward as a probability distribution to be modeled.

## 3 BELIEF DENSITY AS A TEMPERED MARGINAL WINNER DENSITY

Let \(p(\mathbf{x})\) be the expert's belief density. We assume the expert's choices follow a RUM with utility function \(u(\mathbf{x}) = \log p(\mathbf{x})\) . They observe two points independently drawn from the sampling density \(\lambda (\mathbf{x})\) , and the expert chooses one of the points. We denote the probability density of that point by \(p_{w}(\mathbf{x})\) and refer to it as the marginal winner density (MWD). By marginalizing out the unobserved loser \(\mathbf{x}^{\prime}\) in a pairwise comparison where \(\mathbf{x}\) is preferred \((\mathbf{x}\succ \mathbf{x}^{\prime})\) , \(p_{w}(\mathbf{x})\) can be expressed as \(2\lambda (\mathbf{x})\int F(\log p(\mathbf{x}) - \log p(\mathbf{x}^{\prime}))\lambda (\mathbf{x}^{\prime})d\mathbf{x}^{\prime}\) (Mikkola et al., 2024).

While Mikkola et al. (2024) empirically showed that \(p(\mathbf{x})\) resembles a tempered version of \(p_{w}(\mathbf{x})\) (i.e., \(p(\mathbf{x})\approx [p_{w}(\mathbf{x})]^{\tau}\) for some constant \(\tau\) ), this relationship was not formally analyzed besides the theoretical limiting case of selecting the winner from infinitely many alternatives. In this section, we establish a more fundamental connection. We demonstrate that under two RUMs—the Bradley- Terry model and the exponential RUM—it is possible to find a tempering field \(\tau (\mathbf{x})\) such that \(\nabla \log p(\mathbf{x}) = \tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x})\) up to reparameterization of the space. This key relationship implies that, in principle, \(p(\mathbf{x})\) can be precisely recovered from \(p_{w}(\mathbf{x})\) using score- based methods if \(\tau (\mathbf{x})\) is known. This finding motivates leveraging score- matching techniques for estimating the belief density. To analyze such score- based relationships and evaluate approximations, we use the Fisher divergence, which quantifies the difference between two distributions based on their scores:

\[F(p,q) = \int_{\mathcal{X}}\| \nabla \log p(\mathbf{x}) - \nabla \log q(\mathbf{x})\|^{2}p(\mathbf{x})d\mathbf{x}.\]

### 3.1 TEMPERING FIELD

Consider a tempered probability density \(p(\mathbf{x})\) constructed from another density \(q(\mathbf{x})\) using a tempering constant \(\tau >0\) ..

\[p(\mathbf{x}) = \frac{q^{\tau}(\mathbf{x})}{\int_{\mathcal{X}}q^{\tau}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}}.\]

For such densities, the relationship between their scores is given by the product rule as

\[\nabla \log p(\mathbf{x}) = \tau \nabla \log q(\mathbf{x}) + \log q(\mathbf{x})\nabla \tau = \tau \nabla \log q(\mathbf{x}).\]

The score of the tempered density becomes directly proportional to that of the original density, i.e., the two score vectors are collinear. Inspired by this relationship, we define a more general concept. We call a function \(\tau :\mathcal{X}\to (0,\infty)\) a tempering field between \(p\) and \(q\) if their scores satisfy the following relation almost everywhere for \(\mathbf{x}\in \mathcal{X}\) ..

\[\nabla \log p(\mathbf{x}) = \tau (\mathbf{x})\nabla \log q(\mathbf{x}). \quad (5)\]

This implies the scores are collinear, with \(\tau (\mathbf{x})\) as a position- dependent scaling. The tempering field thus describes a localized, score- level tempering.

### 3.2 TEMPERING FIELDS UNDER RUMS

With our theoretical framework in place, we analyze the relationship between the belief density \(p\) and the MWD \(p_{w}\) in terms of tempering fields for RUM models with utility \(\log p\) . We prove our main results for both the Bradley- Terry model and the exponential RUM, with \(W\sim \mathrm{Gumbel}(0,s)\) and \(W\sim \mathrm{Exp}(s)\) . The treatment of the latter RUM is deferred to Appendix A.

To facilitate theoretical analysis, with no loss of generality, we assume a uniform sampling distribution \(\lambda\) throughout this section, to remove the tilting of MWD \(p_{w}(\mathbf{x})\) . We then address a non- uniform \(\lambda\) by reparameterizing the space so that it becomes uniform on a hypercube. The invariance of the RUM under the space reparameterizing is derived in Appendix D. The diffusion model is trained in the transformed space, and the generated samples are mapped back to the original space with the inverse transformation. We use the Rosenblatt transformation, which requires the conditional distribution functions of \(\lambda\) (Rosenblatt, 1952), here assumed to be either known (e.g., when \(\lambda\) is Gaussian) or numerically approximated. Other transformations, e.g. ones based on copulas or normalizing flows trained on samples from \(\lambda\) (i.e., the combined data of winners and losers), could be used as well.

Under the following assumptions, a tempering field exists between the belief density and the MWD:

Assumption 1. \(\operatorname {supp}(p)\subseteq \operatorname {supp}(\lambda)\)

Assumption 2. \(\lambda\) is a uniform density over \(\mathcal{X}\)

Assumption 3. \(p\) is smooth, with \(\nabla p\neq \mathbf{0}\) almost everywhere.

Theorem 3.1. Assume \(W\sim \mathrm{Gumbel}(0,s)\) . A tempering field \(\tau (\mathbf{x})\) exists between the belief density \(p\) and the MWD \(p_{w}\) , and it is given by the formula,

\[\tau (\mathbf{x}) = s\left(\frac{\int_{\mathcal{X}}\frac{1}{1 + r_{s}(\mathbf{x},\mathbf{x}^{\prime})}d\mathbf{x}^{\prime}}{\int_{\mathcal{X}}\frac{r_{s}(\mathbf{x},\mathbf{x}^{\prime})}{(1 + r_{s}(\mathbf{x},\mathbf{x}^{\prime}))^{2}}d\mathbf{x}^{\prime}}\right), \quad (6)\]

where \(r_{s}(\mathbf{x},\mathbf{x}^{\prime})\coloneqq p^{\frac{1}{s}}(\mathbf{x}^{\prime})p^{-\frac{1}{s}}(\mathbf{x})\) is the \(1 / s\) - tempered density ratio.

Proof. Our constructive proof derives a scalar field that satisfies the defining relation. Specifically, for any fixed \(\mathbf{x}\in \mathcal{X}\) , direct manipulations yield a scalar \(\tau (\mathbf{x}) > 0\) such that \(\nabla_{\mathbf{x}}\log p(\mathbf{x}) - \tau (\mathbf{x})\nabla_{\mathbf{x}}\log p_{w}(\mathbf{x}) = 0\) . See Appendix B for the full proof. \(\square\)

Fig. 2 illustrates the tempering field in Theorem 3.1 using \(s = \sqrt{6 / \pi^{2}}\) (unit variance noise). There exists a specific invariance relationship between the tempering and the noise scale. Specifically, if \(\tau_{p,s}\) denotes the tempering field of RUM under the belief density \(p\) and noise level \(s > 0\) , then by the tempering field theorems it is clear that for any exponent \(\alpha >0\) : \(\tau_{p^{\alpha},s} = \frac{1}{s}\tau_{p^{\alpha s},1}\) and \(\tau_{p^{\alpha},s} = s\tau_{p^{\alpha / s},1}\) , where the tempering fields are of the exponential RUM and the Bradley- Terry model, respectively.

> **Image description.** This image is a technical illustration, Figure 2, depicting the relationship between score fields and tempering fields for a Twomoons2D distribution. The figure is divided into two main panels, (a) and (b), set against a dark background.
>
> **Panel (a): Score fields**
> This panel displays a dense vector field, representing the score of the distribution $p$. The field consists of numerous red arrows (vectors) arranged in a highly structured pattern. These arrows form two distinct, curved, crescent-like shapes, characteristic of the Twomoons2D distribution. The vectors generally point outward from the center of these two crescent shapes, indicating the direction of the gradient of the log-probability.
>
> **Panel (b): Tempering fields (estimate and ground-truth)**
> This panel is split into two adjacent sub-visualizations, both representing the tempering field $\tau(\mathbf{x})$.
>
> 1.  **Left Sub-image (Estimate):** This visualization shows the estimated tempering field. It is a smooth, continuous scalar field rendered in shades of purple and dark blue. It also forms two distinct, curved, crescent-like shapes, mirroring the structure seen in Panel (a). The color intensity appears to be concentrated within the curves.
> 2.  **Right Sub-image (Ground-truth):** This visualization shows the ground-truth tempering field. It is a smooth, continuous scalar field rendered in shades of yellow and orange. Like the estimated field, it forms two distinct, curved, crescent-like shapes.
>
> **Text and Labels**
> The figure includes the following labels and caption:
>
> *   **Panel Labels:**
>     *   (a) Score fields
>     *   (b) Tempering fields (estimate and ground-truth)
> *   **Figure Caption:**
>     *   Figure 2: Illustration of the relationship $\nabla \log p(\mathbf{x}) = \tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x})$ when $p$ is Twomoons2D (Stimper et al., 2022) and $\lambda$ is uniform. (a) The score of $p$ (red arrows) and the score of $p_{w}$ (orange arrows) under the Bradley–Terry model, scaled for better visualization. (b) The estimated tempering field $\tau (\mathbf{x})$ from 200 pairwise comparisons (left, Section 4.2) and the ground-truth (right, Theorem 3.1). Due to the collinearity of the scores, the red arrows equal the pointwise product of the orange arrows and the tempering field, which can be estimated (with an underestimation in this example).

<center>Figure 2: Illustration of the relationship \(\nabla \log p(\mathbf{x}) = \tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x})\) when \(p\) is Twomoons2D (Stimper et al., 2022) and \(\lambda\) is uniform. (a) The score of \(p\) (red arrows) and the score of \(p_{w}\) (orange arrows) under the Bradley–Terry model, scaled for better visualization. (b) The estimated tempering field \(\tau (\mathbf{x})\) from 200 pairwise comparisons (left, Section 4.2) and the ground-truth (right, Theorem 3.1). Due to the collinearity of the scores, the red arrows equal the pointwise product of the orange arrows and the tempering field, which can be estimated (with an underestimation in this example). </center>

### 3.3 ON THE CONSTANT TEMPERING APPROXIMATION

Even though our method directly estimates the full tempering field, our theory also sheds light on methods assuming constant tempering, such as Mikkola et al. (2024). It allows us to establish three quantities of interest related to approximating \(p\) with a constant- tempered version of \(q\) : (i) the optimal constant tempering \(\tau^{*} > 0\) which minimizes \(F(p,q^{\tau^{*}})\) , (ii) the approximation error \(F(p,q^{\tau^{*}})\) for any constant \(\tau > 0\) , and (iii) the approximation error \(F(p,q^{\tau^{*}})\) for the optimal constant tempering.

Proposition 3.2. Assume that there exists a tempering field \(\tau (\mathbf{x})\) between \(p\) and \(q\) . The optimal tempering constant \(\tau^{*} = \arg \min_{\tau >0}F(p,q^{\tau})\) can be written as,

\[\tau^{*} = \mathbb{E}_{X\sim p}\left(\omega (X)\tau (X)\right), \quad (7)\]

where the stochastic weight \(\omega \geq 0\) is given by \(\omega (X) = \frac{\| \nabla \log q(X)\|^{2}}{\mathbb{E}_{X\sim p}(\| \nabla \log q(X)\|^{2})}\) .

Proof. By the Leibniz integral rule,

\[\frac{\partial}{\partial\tau} F(p,q^{\tau}) = \int_{\mathcal{X}}2\left(\tau \| \nabla \log q(\mathbf{x})\|^{2} - \langle \nabla \log q(\mathbf{x}),\nabla \log p(\mathbf{x})\rangle\right)p(\mathbf{x})d\mathbf{x}.\]

The divergence is quadratic in \(\tau\) and the critical point is the global minimum, and by assumption \(\langle \nabla \log q(\mathbf{x}),\nabla \log p(\mathbf{x})\rangle = \tau (\mathbf{x})\| \nabla \log q(\mathbf{x})\|^{2}\) . Algebraic manipulation gives the result. \(\square\)

The approximation errors can be quantified in terms of the tempering field.

Proposition 3.3. Let \(\tau (\mathbf{x})\) be a tempering field. For any \(\tau >0\) it holds that

\[F(p,q^{\tau}) = \mathbb{E}_{X\sim p}\left(|\tau -\tau (X)|^{2}\| \nabla \log q(X)\|^{2}\right).\]

Further, when \(\tau^{*} > 0\) is the optimal tempering, we have

\[F(p,q^{\tau^{*}}) = \mathbb{E}_{X\sim p}\left(\| \nabla \log q(X)\|^{2}\tau^{2}(X)\right) - \frac{\left(\mathbb{E}_{X\sim p}\left(\tau (X)\| \nabla \log q(X)\|^{2}\right)\right)^{2}}{\mathbb{E}_{X\sim p}\left(\| \nabla \log q(X)\|^{2}\right)}.\]

Proof. See Appendix B.

## 4 SCORE-BASED DENSITY ESTIMATION FROM PAIRWISE COMPARISONS

Building on Section 3, we now introduce our score- based density estimator for eliciting the belief density from pairwise comparisons. The method has two components. First, we train a diffusion model on the joint distribution of winners and losers using a masking scheme that ensures its marginal, that is MWD \(p_{w}(\mathbf{x})\) , can also be evaluated. Second, under the Bradley- Terry model, we provide a simple procedure to estimate the tempering field \(\tau (\mathbf{x})\) and use it to de- temper the score- based estimate of the MWD. Details of both steps are explained next. The sampling distribution \(\lambda (\mathbf{x})\) is assumed known, and we reparameterize the space to make it uniform, as explained in Section 3.2.

### 4.1 MODELING THE MWD

Our goal is to learn the perturbed score model of the MWD \(\nabla \log [p_{w}(\mathbf{x}) * \mathcal{N}(\mathbf{x}; \mathbf{0}, \sigma^{2}\mathbf{I})]\) . We want to utilize all samples, both winners and losers. To do so we simultaneously learn the marginal \(p_{w}(\mathbf{x}) = \int p_{\mathbf{x} \sim \mathbf{x}'}(\mathbf{x}, \mathbf{x}') dx'\) , and the full joint \(p_{\mathbf{x} \sim \mathbf{x}'}(\mathbf{x}, \mathbf{x}')\) from the concatenated data of winners and losers. To learn the marginal, during training, half of the time we randomly mask \(\mathbf{x}'\) and consider the score only with respect to \(\mathbf{x}\) .

We parametrize the score model as \(s_{\theta}(\mathbf{x}, \mathbf{x}', \sigma , \text{joint, temp})\) , where joint \(\in \{0, 1\}\) and temp \(\in \{0, 1\}\) are conditioning flags: (a) When joint \(= 1\) , the network takes both \(\mathbf{x}\) and \(\mathbf{x}'\) as input and models the score of the joint distribution, \(\nabla \log p_{\mathbf{x} \sim \mathbf{x}'}(\mathbf{x}, \mathbf{x}')\) . (b) When joint \(= 0\) , the loser \(\mathbf{x}'\) is masked (replaced with noise), and the network models the MWD score \(\nabla \log p_{w}(\mathbf{x})\) . The flag temp then controls whether the output is scaled by the tempering field: setting temp \(= 1\) yields an approximation to \(\tau (\mathbf{x}) \nabla \log p_{w}(\mathbf{x}) = \nabla \log p(\mathbf{x})\) .

This parametrization allows us to train a single score network on both winners and losers via denoising score matching (Eq. 1), while still enabling sampling from the belief density. During training, we randomly mask the loser with probability 0.5 (see Appendix C.1 for details). The MWD could also be estimated directly from winners alone, but this would ignore that losers carry information about where winners are less likely to be. We demonstrate the value of modeling the joint distribution empirically in Appendix C.1, while also confirming that we can accurately marginalize the joint model.

We stay as close as possible to the EDM- style diffusion model (Karras et al., 2024a). We use the perturbation kernel \(p_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^{2}\mathbf{I})\) , which aligns with EDM and defines a forward diffusion process from \(\sigma_{\min}\) to \(\sigma_{\max}\) , where \(p_{\sigma_{\min}}(\mathbf{x}) \approx p(\mathbf{x})\) and \(p_{\sigma_{\max}}(\mathbf{x}) \approx \mathcal{N}(\mathbf{0}, \sigma_{\max}^{2}\mathbf{I})\) . A detailed description is provided in Appendix C.2. Algorithm 1 summarizes the method.

### 4.2 TEMPERING FIELD ESTIMATION

The tempering field under the Bradley- Terry model (Eq. 6) has a particularly convenient form as it depends only on the belief density ratio \(r(\mathbf{x}, \mathbf{x}') \coloneqq p(\mathbf{x}') / p(\mathbf{x})\) and the RUM noise level \(s > 0\) . Note that this ratio is different from what the phrase density ratio often refers to; this is the ratio of the same density for two inputs, not a ratio of two densities for the same \(\mathbf{x}\) . It does not depend on the normalizing constant of the belief density and is hence straightforward to estimate via maximum- likelihood estimation (MLE), assuming careful regularization. We train a simple estimator for \(r(\mathbf{x}, \mathbf{x}')\) by maximizing the Bradley- Terry model likelihood of the pairwise comparison data. If we parametrize the density ratio (or its logarithm) as a neural network \(r_{\theta}\) , the parameters \(\theta\) can be optimized by minimizing the loss \(\mathcal{L}(\theta) \propto \text{Softplus}(\log r_{\theta}(\mathbf{x}, \mathbf{x}') / s)\) , where \(\mathbf{x}\) and \(\mathbf{x}'\) are winner and loser points, respectively. The Softplus loss comes from the assumption of \(W \sim \text{Gumbel}(0, s)\) .

We plug the learned \(r_{\theta}\) into the integrals in Eq. 6, where the integrals are computed using importance sampling with the MWD model acting as the importance sampler. The resulting plug- in Monte Carlo estimator of the tempering field is consistent but biased. Similar biased ratio estimators have been used in self- normalized importance sampling (Owen, 2013) and in every- visit off- policy value estimation in reinforcement learning (Sutton et al., 1998) due to favorable variance properties. See Appendix C.5 for more discussion on the estimator. For details on evaluating the importance weights, which correspond to the (reciprocal of) density of the MWD diffusion model, see Appendix C.7. Algorithm 2 summarizes the estimation procedure.

### 4.3 BELIEF DENSITY SAMPLING

Given the perturbed MWD score network \(\mathbf{s}_{\theta}(\mathbf{x},\sigma)\approx \nabla \log [p_{w}(\mathbf{x})\ast \mathcal{N}(\mathbf{x};\mathbf{0},\sigma^{2}\mathbf{I})]\) and the estimate of the tempering field \(\tau (\mathbf{x})\) , we can draw approximate samples from the belief density \(p(\mathbf{x})\) using the score- scaled ALD. Specifically, we iteratively run Eq. 2 with the score \(\tau (\mathbf{x})\mathbf{s}_{\theta}(\mathbf{x},\sigma)\) . The tempering field relation (Eq. 5) shows that for \(\sigma = 0\) this procedure would be equivalent to sampling from \(p(\mathbf{x})\) and ALD is theoretically valid at the small- noise limit (Welling & Teh, 2011), making it an appealing choice as a sampling algorithm. However, for \(\sigma >0\) the algorithm is not exact. We show that it works empirically well (Section 5), but characterization of the approximation error remains as future work.

# Algorithm 1 Full algorithm

require: choice data \(\mathcal{D} = \{\left[\mathbf{x}_{i},\mathbf{x}_{i}^{\prime}\right]|\mathbf{x}_{i}\succ \mathbf{x}_{i}^{\prime}\}_{i = 1}^{n}\) output: samples from the belief density or a trained diffusion

model for it it

train \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}^{\prime},\sigma ,\mathrm{joint})\) using DSM (Eq. 1) on \(\mathcal{D}\)

\(50\%\) prob.: set joint \(= 0\) and mask \(\mathbf{x}^{\prime}\) with \(\mathcal{N}(\mathbf{0},\sigma_{t}^{2}\mathbf{I})\)

\(50\%\) prob.: set \(\sigma\) to noise schedule \((\sigma_{t})_{t = 1}^{L}\)

initialize \(\tau (\mathbf{x})\) given \(\mathcal{D}\) and \(\mathbf{s}_{\theta}\)

sample \(\mathcal{D}^{*}\) using \(\tau (\mathbf{x})\) - scaled ALD with the score \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}^{\prime},(\sigma_{t})_{t = 1}^{L},\mathrm{joint} = 0)\)

\(\frac{\mathrm{optional}}{\mathrm{train} s_{\theta_{\mathrm{MWD}}}(\mathbf{x},\sigma,\mathrm{temp})}\) using DSM on \(\mathcal{D}^{*}\)

return: \(\mathcal{D}^{*}\) or \(\mathbf{s}_{\theta_{\mathrm{MWD}}}(\mathbf{x},\sigma ,\mathrm{temp} = 1)\)

# Algorithm 2 \(\tau (\mathbf{x})\)

require: noise level \(\mathbf{s}\) , \(\mathcal{D}\) , \(\mathbf{s}_{\theta}\)

initialize:

train \(r_{\theta}(\mathbf{x},\mathbf{x}^{\prime})\approx \frac{p(\mathbf{x}^{\prime})}{p(\mathbf{x})}\) as

MLE of \(\mathcal{D}\) (Softplus loss)

sample \(m\) points \(\mathbf{X}\) with den

sities \(p_{w}(\mathbf{X})\) using \(\mathbf{s}_{\theta}\)

input: \(\mathbf{x}\)

\(\mathbf{r} = (r_{\theta}(\mathbf{x},\mathbf{X}))^{\frac{1}{2}}\)

return:

\[s\left(\frac{\sum_{i = 1}^{m}\frac{1}{1 + \mathbf{r}_{i}}\frac{p_{w}(\mathbf{x}_{i})}{\mathbf{r}_{i}}}{(\sum_{i = 1}^{m}\frac{1}{(1 + \mathbf{r}_{i})^{2}}\frac{p_{w}(\mathbf{x}_{i})}{\mathbf{r}_{i}})}\right)\]

## 5 EXPERIMENTS

We evaluate the method on synthetic data generated from a RUM. We then consider an experiment where a large language model (LLM) serves as a proxy for a human expert, demonstrating the method's applicability in settings where data does not follow a RUM model. Our experimental setup closely follows that of Mikkola et al. (2024), with the key distinction that we only query pairwise comparisons, not considering the easier case of ranking multiple candidates. We empirically compare against their flow- based model, using their implementation, as the only previous method for the task. We consider two variants of our score- based method: score- \(\tau (\mathbf{x})\) , which uses the full tempering field (Section 4), and score- \(\tau^{*}\) , which uses a constant tempering estimated via Proposition 3.2. This allows direct quantification of the importance of modeling the whole field.

Setup and evaluation For a \(d\) - dimensional target, we query \(1000d\) pairwise comparisons to ensure reliable comparison between the methods but remaining well below the large- sample regime typical for the diffusion model literature. For \(d\leq 4\) , the sampling distribution \(\lambda\) is uniform. For \(d > 4\) , \(\lambda\) is a diagonal Gaussian (Gaussian mixture in Mixturegaussians10D) centered at the target mean, with a variance three times that of the target's.2 The simulated expert follows the Bradley- Terry model with utility given by \(\log p\) , where the belief density \(p\) varies in dimensionality, modality, and detail. We set the noise level \(s = \sqrt{6 / \pi^{2}}\) (unit variance). As the diffusion model, we adopt an EDM- style architecture with a MLP score network, implemented on top of Karras et al. (2024b). For further details, see Appendix C and E. We assess performance qualitatively by visually comparing \(2D\) and \(1D\) marginals of the target density and the estimate, and quantitatively using two metrics: the Wasserstein distance and the mean marginal total variation distance (MMTV; Acerbi, 2020). Results are reported as means and standard deviations over replicate runs.

Experiment 1: Synthetic low dimensional targets with uniform sampling We consider low- dimensional synthetic target densities that may exhibit non- trivial geometry or multimodality. The set includes five target distributions, see Appendix E.1. Table 1 (top) shows that the score- based method is clearly superior for all targets, with at least \(50\%\) (Wasserstein) and \(25\%\) (MMTV) reduction of error

Table 1: Density estimation from pairwise comparisons: score- \(\tau^{*}\) and score- \(\tau (\mathbf{x})\) denote our methods with constant and varying tempering fields, respectively, and flow refers to (Mikkola et al., 2024). Bold indicates the best method, and underline indicates results that are not significantly worse (paired two-sided Wilcoxon signed-rank test, \(p > 0.05\)

| p(x) | wasserstein (↓) flow | wasserstein (↓) score-τ* | wasserstein (↓) score-τ(x) | MMTV (↓) flow | MMTV (↓) score-τ* | MMTV (↓) score-τ(x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Onemoon2D | 1.37 (±0.03) | 0.44 (±0.13) | 0.37 (±0.14) | 0.54 (±0.00) | 0.25 (±0.06) | 0.22 (±0.06) |
| Twomoons2D | 1.29 (±0.06) | 0.54 (±0.14) | 0.44 (±0.09) | 0.53 (±0.01) | 0.15 (±0.03) | 0.14 (±0.02) |
| Ring2D | 0.87 (±0.03) | 0.40 (±0.07) | 0.39 (±0.08) | 0.40 (±0.01) | 0.27 (±0.01) | 0.26 (±0.01) |
| Gaussian4D | 6.12 (±0.05) | 1.69 (±0.22) | 1.40 (±0.24) | 0.72 (±0.01) | 0.41 (±0.05) | 0.44 (±0.08) |
| Mixturegaussians4D | 3.75 (±0.02) | 1.23 (±0.06) | 1.09 (±0.09) | 0.53 (±0.01) | 0.27 (±0.01) | 0.22 (±0.02) |
| Stargaussian6D | 2.25 (±0.02) | 1.55 (±0.04) | 1.28 (±0.04) | 0.19 (±0.00) | 0.18 (±0.02) | 0.16 (±0.01) |
| Mixturegaussians10D | 1.41 (±0.01) | 1.10 (±0.12) | 1.33 (±0.11) | 0.19 (±0.00) | 0.14 (±0.02) | 0.26 (±0.06) |
| Gaussian16D | 5.50 (±0.03) | 5.00 (±0.07) | 5.00 (±0.06) | 0.16 (±0.00) | 0.13 (±0.00) | 0.13 (±0.00) |

compared to the flow method. In most cases, using the full tempering field \(\tau (\mathbf{x})\) is better than relying on the best constant tempering, but we outperform the flow- based method even when restricted to constant tempering as in their approach. Visual inspection (Fig. 3 (a- b) and Figs. F.1- F.3) confirms this. The flow method captures the density relatively well but clearly overestimates the low- density regions, whereas our estimate is essentially perfect here.

Experiment 2: Synthetic targets with Gaussian sampling For higher- dimensional experiments we replace uniform sampling with more concentrated Gaussian sampling, since otherwise the probability that both \(\mathbf{x}\) and \(\mathbf{x}^{\prime}\) fall in low- density regions increases dramatically, making it impossible to learn \(p(\mathbf{x})\) well. The set includes three target distributions, see Appendix E.1. Table 1 (bottom) shows that we again outperform the flow- based method in Wasserstein distance but in terms of MMTV the methods are closer. Visually, the score- based method usually gives sharper and better estimates (e.g., compare Fig. F.6 and F.7), but suffers in terms of the MMTV metric due to occasional too- tight marginals resulting from overestimating the tempering field (e.g., Fig. F.8).

Experiment 3: LLM as a proxy for the expert To illustrate the method in a real belief density estimation task without user experiments, we replicate the LLM experiment of Mikkola et al. (2024), except that we query 220 pairwise comparisons instead of 5- wise rankings. Using the data from (Mikkola et al., 2024), we uniformly sample 220 pairwise comparisons across all eight features and prompt the LLM for belief judgments. The LLM \(^3\) acts as a proxy for a human expert, providing its belief about what houses in California are like, restricted to the features available in the California housing dataset (Pace & Barry, 1997). The LLM's belief is inferred solely from the pairwise queries, without providing the LLM any direct access to the data itself.

This experiment leverages the finding that LLMs learn statistical features from the vast training data and can be queried about it (Brown et al., 2020; Requeima et al., 2024). While the aim is to infer the belief density of the LLM—which is unknown—our main goal here is to validate our method, rather than to analyze the beliefs of this particular LLM. We do this by comparing the belief estimate with the empirical data distribution: Similarity between the two suggests the elicitation method yields a reasonable belief estimate, and differences can be interpreted as possible biases the LLM might have. Fig. 3 (c) shows clear distributional similarities; e.g. the marginals of AveRooms and MedInc exhibit similar shapes. See Appendix F.2 for complete results, with Table F.1 quantifying the accuracy.

## 6 DISCUSSION

We proved the theoretical connection for two common RUMs but we believe it extends to other RUMs as well, although a closed- form tempering field is not guaranteed e.g. for the Thurstone- Mosteller model (Thurstone, 1927) where the choice probability requires integration.

The difficulty of the belief estimation problem, naturally, depends heavily on the sampling distribution \(\lambda (\mathbf{x})\) . For example, when the support of \(p(\mathbf{x})\) is much smaller than that of \(\lambda (\mathbf{x})\) , it becomes

> **Image description.** This image is a technical figure composed of three distinct panels, (a), (b), and (c), illustrating density estimation methods and the results of a learned likelihood model (LLM) experiment.
>
> **Panels (a) and (b): Ring2D Density Estimates**
> These two panels display visualizations of density estimates for a dataset called Ring2D, which is characterized by a ring-like structure. The density is represented by blue points.
>
> *   **Panel (a) - Score-based:** This visualization shows the density estimate generated by the score-based method. The blue points form a ring, but the central area of the ring is relatively sparse.
> *   **Panel (b) - Flow:** This visualization shows the density estimate generated by the flow method. While the points also form a ring, the central area is visibly denser and more filled with blue points compared to Panel (a).
>
> **Panel (c): LLM Experiment**
> This panel presents the results of an LLM experiment, divided into two comparative sections: "Learned LLM prior" and "California housing dataset." Both sections utilize a combination of 2D scatter plots (cross-plots) and 1D histograms (marginal distributions) involving three variables: `AveRooms`, `HouseAge`, and `MedInc`.
>
> *   **Learned LLM prior:** This section displays the estimated prior distribution.
>     *   **Scatter Plots (Top Row):** Three cross-plots show the relationships between the variables: `AveRooms` vs. `HouseAge`, `AveRooms` vs. `MedInc`, and `HouseAge` vs. `MedInc`. In all three, the data points form dense, somewhat elliptical clouds.
>     *   **Histograms (Bottom Row):** Three marginal distributions show the individual distributions for each variable: `AveRooms`, `HouseAge`, and `MedInc`. These histograms show unimodal, somewhat Gaussian-like distributions.
>
> *   **California housing dataset:** This section displays the actual data from the California housing dataset, serving as a comparison to the learned prior.
>     *   **Scatter Plots (Top Row):** The three cross-plots (`AveRooms` vs. `HouseAge`, `AveRooms` vs. `MedInc`, and `HouseAge` vs. `MedInc`) show dense, elliptical clusters of data points, visually mirroring the patterns seen in the "Learned LLM prior" section.
>     *   **Histograms (Bottom Row):** The three marginal distributions for `AveRooms`, `HouseAge`, and `MedInc` are shown, exhibiting distributions that are visually consistent with the prior distributions.
>
> Overall, the figure visually contrasts the density estimation performance of two methods (Score-based vs. Flow) and demonstrates that the learned prior distribution closely approximates the actual distribution of the California housing dataset.

<center>Figure 3: (a-b) Samples from score-based and flow estimates of Ring2D, with contours indicating the true density. Ring2D illustrates an extreme case where the score-based method clearly outperforms the flow method: the flow model oversamples the center of the ring, where the MWD also has moderate density, whereas the score-based method can downweight it using the tempering field. (c) Cross-plot of the first three variables in the LLM expert elicitation experiment. Full cross-plot and comparison to the flow method are shown in Figs. F.10 and F.11. The score-based method tends to generate Gaussian-like marginals in this extremely limited data setting. </center>

nearly impossible to obtain sharp estimates of \(p(\mathbf{x})\) , and the problem is further exacerbated in high- dimensional spaces. For this reason, we see potential in active learning methods that concentrate sampling in high- density regions of \(p(\mathbf{x})\) . In this work, we assume that \(\lambda (\mathbf{x})\) is given as it would be in many applications (e.g. the elicitation protocol in prior elicitation), but for active learning or learning beliefs from public preference data, an additional density estimation step is required to learn \(\lambda (\mathbf{x})\) .

We demonstrated that low- dimensional targets can be learned from a few hundreds of pairwise comparisons (Fig. 1). However, in extremely limited data regimes, say below \(100d\) pairwise comparisons, the robustness of our method is not guaranteed without carefully tuning hyperparameters such as those of the diffusion model—a class of models that are notoriously difficult to train in a stable manner (Karras et al., 2024b, Appendix B.5). Stability could likely be improved by adopting best practices from the field (Karras et al., 2022) and incorporating recent advances in learning score models from limited data (Li et al., 2024). Similarly, the tempering field estimate (Section 4.2) depends on the density ratio estimate \(r_{\theta}\) , which is sensitive to the network's regularization. Misspecified values of the \(\ell_{2}\) regularization for the network weights \(\theta\) or the RUM noise level \(s\) will lead to under- or overestimation of the tempering field, due to MLE struggling to determine the global scale and tails.

While our method is theoretically grounded and clearly outperforms the flow- based method of Mikkola et al. (2024) in estimation accuracy, it has some limitations. The flow allows faster sampling, only requiring a single forward pass, and also more efficient and stable evaluation of the density. Our method requires numerically solving the probability- flow ODE for evaluating the density, and it is still open whether diffusion models provide reliable pointwise density estimates (Zheng et al., 2023).

The connection between the target density \(p\) and the MWD \(p_{w}\) may have applications beyond expert knowledge elicitation, such as fine- tuning generative models (Wallace et al., 2024) using pairwise data, a perspective explored by Dumoulin et al. (2024). Specifically, when \(\lambda (\mathbf{x})\) represents a pretrained generative model conditioned on a prompt \(c\) , our theory suggests that training the MWD and tempering field on individual- level data (rather than pooled data) yields a probabilistic reward model \(\operatorname {reward}(c, \mathbf{x}) = p(\mathbf{x}|c)\) . This conditional tempered MWD captures the distribution of samples (e.g., images) aligned with prompt \(c\) from that specific individual's perspective.

## 7 CONCLUSIONS

Despite two decades of active research on learning from preference data, following the pioneering works of Chu & Ghahramani (2005); Fürnkranz & Hüllermeier (2010), the question of how to learn flexible densities from such data has remained elusive. We established the missing theoretical basis by showing how the score of the target density relates to quantities that can be estimated, enabling the use of powerful modern density estimators for this task. Our approach demonstrates superior performance over recent flow- based solutions in representing human beliefs.

---

*Transcribed with OCR and VLMs; text, equations, tables, and figure descriptions may contain mistakes.*
