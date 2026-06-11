```
@inproceedings{mikkola2026scorebased,
  title={Score-Based Density Estimation from Pairwise Comparisons},
  author={Petrus Mikkola and Luigi Acerbi and Arto Klami},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026},
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

# SCORE-BASED DENSITY ESTIMATION FROM PAIRWISE COMPARISONS - Appendix

---

## APPENDIX

This appendix provides additional technical material complementing the main paper. Appendix A presents extended theoretical results for the exponential noise RUM. Appendix B contains the proofs. Appendix C provides method details and implementation specifications. Appendix D shows the RUM invariance under space reparameterization to a uniform distribution. Appendix E reports experimental details, the runtime breakdown, and additional ablations on RUM model misspecification. Appendix F includes plots of the learned belief densities and a description of the LLM experiment. Appendix G describes the use of LLMs in the preparation of this paper.

## LIST OF NOTATION

x A vectorX A matrixx \(x_{i}\) Element \(i\) of vector \(\mathbf{x}\) or \(i^{t h}\) observation \(X\) A vector- valued random variable \(W\) RUM noise, a scalar random variable or a stochastic process \(W(\mathbf{x})\) A stochastic process at index \(\mathbf{x}\) \(\succ\) A binary strict preference relation \(\mathbf{x}\succ \mathbf{x}^{\prime}\) \(\mathbf{x}\) is chosen over \(\mathbf{x}^{\prime}\) , where \(\mathbf{x}\) is winner point and \(\mathbf{x}^{\prime}\) is loser point \(p(\mathbf{x})\) A probability density function evaluated at \(\mathbf{x}\) \(p\) Belief density \(\lambda\) Sampling density \(p_{w}\) Marginal winner density (MWD) \(p_{\mathbf{x}\succ \mathbf{x}^{\prime}}\) Joint density of winner- loser pairs encoded in the order of the arguments \(u\) Utility function \(\nabla f(\mathbf{x})\) The gradient of \(f\) with respect to \(\mathbf{x}\) , and evaluated at \(\mathbf{x}\) \(\mathbf{s}_{\theta}\) Noise- conditioned score network with parameters \(\theta\) \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}^{\prime},\sigma)\) Noise- conditioned score network evaluated at winner- loser pair \((\mathbf{x},\mathbf{x}^{\prime})\) and noise scale \(\sigma\) \(s\) RUM noise level, a positive scalar \(\tau\) Tempering field, a scalar field \(\tau :\mathcal{X}\to (0,\infty)\) or a positive scalar (in case of constant field) \(\tau (\mathbf{x})\) Tempering field evaluated at \(\mathbf{x}\) , a positive scalar \(\tau^{*}\) Optimal tempering constant, a positive scalar \(\omega\) Weighting function, a scalar field \(\omega :\mathcal{X}\to [0,\infty)\) \(\omega (X)\) Stochastic weight, a scalar random variable \(\| \cdot \|\) \(\ell_{2}\) - norm (Euclidean norm)

## A TEMPERING FIELD UNDER THE EXPONENTIAL NOISE RUM

In this appendix, we state the existence and provide a closed- form expression for the tempering field when the expert's choice model follows the exponential RUM with \(W\sim \mathrm{Exp}(s)\)

Theorem A.1. Assume \(W\sim \mathrm{Exp}(s)\) . A tempering field \(\tau (\mathbf{x})\) exists between the belief density \(p\) and the MWD \(p_{w}\) , and it is given by the formula

\[\tau (\mathbf{x}) = \frac{1}{s}\left(\frac{2v o l(L_{\mathbf{x}}) + 2p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}}{p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} + p^{-s}(\mathbf{x})\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}} - 1\right), \quad (A.1)\]

where the sublevel set \(L_{\mathbf{x}} = \{\mathbf{x}^{\prime}\in \mathcal{X}\mid p(\mathbf{x})\geq p(\mathbf{x}^{\prime})\}\) and the superlevel set \(U_{\mathbf{x}} = \mathcal{X}\setminus L_{\mathbf{x}}\)

Proof Sketch. For a fixed \(\mathbf{x}\in \mathcal{X}\) , after lengthy manipulations, we get \(\tau (\mathbf{x}) > 0\) such that \(\nabla_{\mathbf{x}}\log p(\mathbf{x}) - \tau (\mathbf{x})\nabla_{\mathbf{x}}\log p_{w}(\mathbf{x}) = \mathbf{0}\) . To handle the change in integration domains, we apply the generalized Leibniz integral rule (Flanders, 1973). See Appendix B for the full proof. \(\square\)

Fig. A.1 illustrates the tempering fields from Theorem A.1 and Theorem 3.1, using \(s = \sqrt{6 / \pi^{2}}\) for Bradley- Terry and \(s = 1\) for exponential RUM, both resulting in unit variance for ease of comparison. The tempering fields are extremely similar, but tempering in high- density regions appears slightly more pronounced in the Bradley- Terry model, due to a lighter- tailed conditional choice distribution.

> **Image description.** A comparative visualization consisting of two side-by-side 2D heat maps, illustrating the "tempering fields" ($\tau(\mathbf{x})$) for a probability distribution known as Twomoons2D, under two different statistical models.
>
> The image is dominated by two identical-sized panels, each displaying a continuous color gradient representing the value of the tempering field across a 2D space. Both panels exhibit a highly symmetrical, "two-moon" shape. The central region of the distribution is characterized by a deep, dark purple/blue color, indicating the lowest values of the tempering field. This central low-value region is flanked by two prominent, crescent-shaped lobes that extend outward. These lobes are colored in bright yellow and orange, signifying significantly higher values of the tempering field.
>
> A vertical color bar (legend) is positioned on the right side, spanning both panels, which maps the colors to numerical values of $\tau(\mathbf{x})$. The scale ranges from a dark blue/purple at the bottom to a bright yellow/red at the top. The numerical labels on this scale are: 2.0, 5.6, 9.2, 12.8, 20.0, 23.6, 27.2, 30.8, and 34.4.
>
> The two panels represent the tempering fields derived from two different models (implied by the context to be Bradley-Terry and Exponential RUM). While the overall visual pattern—the central low-value region surrounded by high-value crescent lobes—is similar in both panels, the subtle differences in the intensity and spread of the colors suggest variations in how the tempering field behaves under each model.
>
> Below the visualization, a partial caption is visible, beginning: "Figure A.1: Illustration of the tempering fields under two different RUMs when $p$ is Twomoons2D (Stimper et al., 2022). The tempering field $\tau (\mathbin...$"

<center>Figure A.1: Illustration of the tempering fields under two different RUMs when \(p\) is Twomoons2D (Stimper et al., 2022). The tempering field \(\tau (\mathbf{x})\) of the exponential RUM (left, Theorem A.1) and the Bradley–Terry model (right, Theorem 3.1). </center>

## B PROOFS

Theorem A.1. Assume \(W\sim \mathrm{Exp}(s)\) . A tempering field \(\tau (\mathbf{x})\) exists between the belief density \(p\) and the MWD \(p_{w}\) , and it is given by the formula,

\[\tau (\mathbf{x}) = \frac{1}{s}\left(\frac{2v o l(L_{\mathbf{x}}) + 2p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}}{p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} + p^{-s}(\mathbf{x})\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}} - 1\right),\]

where the sublevel set \(L_{\mathbf{x}} = \{\mathbf{x}^{\prime}\in \mathcal{X}\mid p(\mathbf{x})\geq p(\mathbf{x}^{\prime})\}\) and the superlevel set \(U_{\mathbf{x}} = \{\mathbf{x}^{\prime}\in \mathcal{X}\mid p(\mathbf{x})< p(\mathbf{x}^{\prime})\}\) .

Proof. We want to show that for each \(\mathbf{x}\in \mathcal{X}\) , there exists a scalar \(\tau (\mathbf{x}) > 0\) such that \(\nabla \log p(\mathbf{x}) - \tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x}) = \mathbf{0}\) . Our constructive proof determines this scalar through brute- force calculation. Fix a point \(\mathbf{x}\in \mathcal{X}\) , and denote a constant \(\tau (\mathbf{x}) = \tau >0\) .

Under the exponential RUM, the MWD \(p_{w}(\mathbf{x})\) equals to

\[p_{w}(\mathbf{x}) = 2\lambda (\mathbf{x})\int_{\mathcal{X}}F_{\mathrm{Laplace}(0,1 / s)}\left(\log p(\mathbf{x}) - \log p(\mathbf{x}^{\prime})\right)\lambda (\mathbf{x}^{\prime})d\mathbf{x}^{\prime}. \quad (B.1)\]

For uniform \(\lambda\) , this implies

\[\nabla_{\mathbf{x}}\log p_{w}(\mathbf{x}) = \nabla_{\mathbf{x}}\log \int_{\mathcal{X}}F_{\mathrm{Laplace}(0,1 / s)}\left(\log p(\mathbf{x}) - \log p(\mathbf{x}^{\prime})\right)d\mathbf{x}^{\prime}. \quad (B.2)\]

Let \(L_{\mathbf{x}},U_{\mathbf{x}}\subset \mathcal{X}\) be the regions \(L_{\mathbf{x}} = \{\mathbf{x}^{\prime}\in \mathcal{X}\mid p(\mathbf{x})\geq p(\mathbf{x}^{\prime})\}\) and \(U_{\mathbf{x}} = \{\mathbf{x}^{\prime}\in \mathcal{X}\mid p(\mathbf{x})< p(\mathbf{x}^{\prime})\}\) Straightforward manipulations give us,

\[\tau \nabla_{\mathbf{x}}\log p_{w}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p(\mathbf{x}) =\] \[= \tau \nabla_{\mathbf{x}}\log \left(\int_{U_{\mathbf{x}}}\frac{1}{2} p^{s}(\mathbf{x})p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} + \int_{L_{\mathbf{x}}}\left(1 - \frac{1}{2} p^{-s}(\mathbf{x})p^{s}(\mathbf{x}^{\prime})\right)d\mathbf{x}^{\prime}\right) - \tau \nabla_{\mathbf{x}}\log p^{\frac{1}{\tau}}(\mathbf{x})\] \[= \tau \nabla_{\mathbf{x}}\log \frac{vol(L_{\mathbf{x}}) + \frac{1}{2} p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} - \frac{1}{2} p^{-s}(\mathbf{x})\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}}{p^{\frac{1}{\tau}}(\mathbf{x})}.\]

Because \(\nabla_{\mathbf{x}}\log f(\mathbf{x}) = \nabla_{\mathbf{x}}f(\mathbf{x}) / f(\mathbf{x})\) , the above vanishes, if and only if,

\[\nabla_{\mathbf{x}}\left(p^{-\frac{1}{\tau}}(\mathbf{x})vol(L_{\mathbf{x}}) + \frac{1}{2} p^{s - \frac{1}{\tau}}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} - \frac{1}{2} p^{-s - \frac{1}{\tau}}(\mathbf{x})\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}\right) = 0.\]

We will apply to LHS the generalized Leibniz integral rule (Flanders, 1973) for each fixed dimension \(j\in \{1,\dots,d\}\) by interpreting \(\frac{\partial}{\partial x_{j}}\) as differentiation with respect to the time. To justify the generalized Leibniz integral rule, note that the boundaries \(\partial L_{\mathbf{x}} = \partial U_{\mathbf{x}}\) are defined by an implicit function \(g:\mathcal{X}\to \mathcal{X}\) whose graph is the set \(\{(\mathbf{x},g(\mathbf{x}))\} = \{(\mathbf{x},\mathbf{x}^{\prime})\in \mathcal{X}^{2}\mid f(\mathbf{x},\mathbf{x}^{\prime}) = 0\}\) , where the function \(f(\mathbf{x},\mathbf{x}^{\prime})\coloneqq p(\mathbf{x}) - p(\mathbf{x}^{\prime})\) is continuously differentiable by Assumption 3. Moreover, since \(\nabla_{\mathbf{x}^{\prime}}f(\mathbf{x},\mathbf{x}^{\prime}) = - \nabla p(\mathbf{x}^{\prime})\neq 0\) almost everywhere, the implicit function theorem implies that the level set \(\{(\mathbf{x},\mathbf{x}^{\prime})\mid f(\mathbf{x},\mathbf{x}^{\prime}) = 0\}\) locally defines \(\mathbf{x}^{\prime}\) as a differentiable function of \(\mathbf{x}\) almost everywhere. Therefore, \(g(\mathbf{x})\) is continuously differentiable almost everywhere.

For a smooth function \(f:\mathcal{X}\to \mathbb{R}_{+}\) consider,

\[\frac{\partial}{\partial\mathbf{x}_{j}}\int_{A_{\mathbf{x}}}f(\mathbf{x}^{\prime})d\mathbf{x}^{\prime},\]

where \(A_{\mathbf{x}} = L_{\mathbf{x}}\) or \(A_{\mathbf{x}} = U_{\mathbf{x}}\) . Interpret the scalar \(\mathbf{x}_{j}\) as time, and apply the generalized Leibniz integral rule,

\[\frac{\partial}{\partial\mathbf{x}_{j}}\int_{A_{\mathbf{x}}}f(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} = \int_{A_{\mathbf{x}}}\frac{\partial}{\partial\mathbf{x}_{j}} f(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} + \int_{\partial A_{\mathbf{x}}}f(\mathbf{x}^{\prime})(\mathbf{n}\cdot \mathbf{v})dS,\]

where \(\mathbf{n}\) is the unit normal vector pointing outwards from the boundary \(\partial A_{\mathbf{x}}\) , \(\mathbf{v}\) is the Eulerian velocity of the boundary \(\partial A_{\mathbf{x}}\) when \(\mathbf{x}_{j}\) is interpreted as time, and \(dS\) is the surface element. The first term in RHS vanishes. For the second term, consider the level set \(\{\mathbf{x}^{\prime}\in \mathcal{X}\mid p(\mathbf{x}) - p(\mathbf{x}^{\prime}) = 0\}\) . The normal vector is orthogonal to this level set, which equals to gradient with respect to \(\mathbf{x}^{\prime}\) , \(\mathbf{n} = \nabla_{\mathbf{x}^{\prime}}(p(\mathbf{x}) - p(\mathbf{x}^{\prime})) / \| \nabla_{\mathbf{x}^{\prime}}(p(\mathbf{x}) - p(\mathbf{x}^{\prime}))\| = - \nabla_{\mathbf{x}^{\prime}}p(\mathbf{x}^{\prime}) / \| \nabla_{\mathbf{x}^{\prime}}p(\mathbf{x}^{\prime})\|\) . This corresponds to the normal vector of \(L_{\mathbf{x}}\) while the normal vector of \(U_{\mathbf{x}}\) is with the opposite sign.

Let us consider \(\mathbf{v} = (\frac{\partial\mathbf{x}_{1}^{\prime}}{\partial\mathbf{x}_{j}},\dots,\frac{\partial\mathbf{x}_{N}^{\prime}}{\partial\mathbf{x}_{j}})\) , the velocity of the boundary with respect to \(\mathbf{x}_{j}\) . The total derivative of the boundary should not change,

\[D_{\mathbf{x}_{j}}(p(\mathbf{x}) - p(\mathbf{x}^{\prime})) = 0\] \[\frac{\partial}{\partial\mathbf{x}_{j}}(p(\mathbf{x}) - p(\mathbf{x}^{\prime})) + \sum_{i = 1}^{d}\frac{\partial}{\partial\mathbf{x}_{i}^{\prime}}(p(\mathbf{x}) - p(\mathbf{x}^{\prime}))\frac{\partial\mathbf{x}_{i}^{\prime}}{\partial\mathbf{x}_{j}} = 0\] \[\frac{\partial}{\partial\mathbf{x}_{j}} p(\mathbf{x}) - \sum_{i = 1}^{d}\frac{\partial}{\partial\mathbf{x}_{i}^{\prime}} p(\mathbf{x}^{\prime})\frac{\partial\mathbf{x}_{i}^{\prime}}{\partial\mathbf{x}_{j}} = 0.\]

Taking the dot product of the constraint with the normal vector gives,

\[\mathbf{n}\cdot \mathbf{v} = -\frac{\frac{\partial}{\partial\mathbf{x}_{j}}p(\mathbf{x})}{\|\nabla_{\mathbf{x}^{\prime}}p(\mathbf{x}^{\prime})\|}.\]

Since \(p(\mathbf{x}^{\prime}) = p(\mathbf{x})\) on \(\partial L_{\mathbf{x}} = \partial U_{\mathbf{x}}\) ,

\[\frac{\partial}{\partial x_{j}}\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} = -p^{s}(\mathbf{x})\frac{\partial}{\partial x_{j}} p(\mathbf{x})\int_{\partial L_{\mathbf{x}}}\frac{1}{\|{\nabla_{\mathbf{x}}^{\prime}}p(\mathbf{x}^{\prime})\|}d S(\mathbf{x}^{\prime})\] \[\frac{\partial}{\partial x_{j}}\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} = p^{-s}(\mathbf{x})\frac{\partial}{\partial x_{j}} p(\mathbf{x})\int_{\partial L_{\mathbf{x}}}\frac{1}{\|{\nabla_{\mathbf{x}}^{\prime}}p(\mathbf{x}^{\prime})\|}d S(\mathbf{x}^{\prime})\] \[\frac{\partial}{\partial x_{j}}\int_{L_{\mathbf{x}}}d\mathbf{x}^{\prime} = -\frac{\partial}{\partial x_{j}} p(\mathbf{x})\int_{\partial L_{\mathbf{x}}}\frac{1}{\|{\nabla_{\mathbf{x}}^{\prime}}p(\mathbf{x}^{\prime})\|}d S(\mathbf{x}^{\prime}).\]

Together these imply that,

\[\left(p^{-\frac{1}{\tau}}(\mathbf{x})\frac{\partial}{\partial x_{j}}\int_{L_{\mathbf{x}}}d\mathbf{x}^{\prime} + \frac{1}{2} p^{s - \frac{1}{\tau}}(\mathbf{x})\frac{\partial}{\partial x_{j}}\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} - \frac{1}{2} p^{-s - \frac{1}{\tau}}(\mathbf{x})\frac{\partial}{\partial x_{j}}\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}\right) = 0.\]

We are left with the following terms that vanish,

\[p^{-\frac{1}{\tau} -1}(\mathbf{x})\nabla_{\mathbf{x}}p(\mathbf{x})\left(-\frac{1}{\tau} vol(L_{\mathbf{x}}) + \frac{s - \frac{1}{\tau}}{2} p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} + \frac{s + \frac{1}{\tau}}{2} p^{-s}(\mathbf{x})\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}\right).\]

In order to this hold, the scalar in the parenthesis must vanish,

\[\frac{2}{\tau} vol(L_{\mathbf{x}}) + \frac{1}{\tau} p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} - \frac{1}{\tau} p^{-s}(\mathbf{x})\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}\] \[= s p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} + s p^{-s}(\mathbf{x})\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}.\]

Rearranging the terms give us,

\[\tau = \frac{1}{s}\left(\frac{2vol(L_{\mathbf{x}}) + 2p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}}{p^{s}(\mathbf{x})\int_{U_{\mathbf{x}}}p^{-s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime} + p^{-s}(\mathbf{x})\int_{L_{\mathbf{x}}}p^{s}(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}} - 1\right).\]

Theorem 3.1. Assume \(W\sim \mathrm{Gumbel}(0,s)\) . A tempering field \(\tau (\mathbf{x})\) exists between the belief density \(p\) and the MWD \(p_{w}\) , and it is given by the formula,

\[\tau (\mathbf{x}) = s\left(\frac{\int_{\mathcal{X}}\frac{1}{1 + r_{s}(\mathbf{x},\mathbf{x}^{\prime})}d\mathbf{x}^{\prime}}{\int_{\mathcal{X}}\frac{r_{s}(\mathbf{x},\mathbf{x}^{\prime})}{(1 + r_{s}(\mathbf{x},\mathbf{x}^{\prime}))^{2}}d\mathbf{x}^{\prime}}\right), \quad (B.3)\]

where \(r_{s}(\mathbf{x},\mathbf{x}^{\prime})\coloneqq p^{\frac{1}{s}}(\mathbf{x}^{\prime})p^{-\frac{1}{s}}(\mathbf{x})\) is \(1 / s\) - tempered density ratio.

Proof. Under the Bradley- Terry model, \(W\sim \mathrm{Gumbel}(0,s)\) , and the MWD \(p_{w}(\mathbf{x})\) now equals to

\[p_{w}(\mathbf{x}) = 2\lambda (\mathbf{x})\int_{\mathcal{X}}F_{\mathrm{Logistic}(0,s)}\left(\log p(\mathbf{x}) - \log p(\mathbf{x}^{\prime})\right)\lambda (\mathbf{x}^{\prime})d\mathbf{x}^{\prime}. \quad (B.4)\]

Following same lines of reasoning as in the constructive proof of Theorem A.1, we fix \(\mathbf{x}\in \mathcal{X}\) and aim to find a constant \(\tau (\mathbf{x}) = \tau >0\) solving the tempering field condition. Because \(p\) is uniform, the necessary and sufficient condition for the existence of \(\tau\) is that it solves the equation,

\[p^{-1 - \frac{1}{s}}(\mathbf{x})\int_{\mathcal{X}}\frac{\frac{1}{s}p^{-\frac{1}{s}}(\mathbf{x})p^{\frac{1}{s}}(\mathbf{x}^{\prime}) - \frac{1}{\tau}\left(1 + p^{-\frac{1}{s}}(\mathbf{x})p^{\frac{1}{s}}(\mathbf{x}^{\prime})\right)}{(1 + p^{-\frac{1}{s}}(\mathbf{x})p^{\frac{1}{s}}(\mathbf{x}^{\prime}))^{2}} d\mathbf{x}^{\prime} = 0.\]

This is equivalent to,

\[\tau = s\left(\frac{\int_{\mathcal{X}}\frac{1}{1 + r_{s}(\mathbf{x},\mathbf{x}^{\prime})}d\mathbf{x}^{\prime}}{\int_{\mathcal{X}}\frac{r_{s}(\mathbf{x},\mathbf{x}^{\prime})}{(1 + r_{s}(\mathbf{x},\mathbf{x}^{\prime}))^{2}} d\mathbf{x}^{\prime}}\right), \quad (B.5)\]

where for clarity we denote \(r_{s}(\mathbf{x},\mathbf{x}^{\prime})\coloneqq p^{\frac{1}{s}}(\mathbf{x}^{\prime})p^{-\frac{1}{s}}(\mathbf{x}) = \left(\frac{p(\mathbf{x}^{\prime})}{p(\mathbf{x})}\right)^{1 / s}\) , which is \(1 / s\) - tempered density ratio between density values at compared points \(\mathbf{x}^{\prime}\) and \(\mathbf{x}\) . \(\square\)

Proposition 3.2. Assume that there exists a tempering field \(\tau (\mathbf{x})\) between \(p\) and \(q\) . A tempering parameter \(\tau^{*} > 0\) defined by,

\[\tau^{*} = \mathbb{E}_{X\sim p}\left(\omega (X)\tau (X)\right), \quad (B.6)\]

where a stochastic weight \(\omega \geq 0\) is given by

\[\omega (X) = \frac{\| \nabla \log q(X)\|^{2}}{\mathbb{E}_{X\sim p}\left(\| \nabla \log q(X)\|^{2}\right)},\]

minimizes the Fisher divergence between \(p\) and \(q\) ,

\[\tau^{*} = \underset {\tau >0}{\arg \min}F(p,q^{\tau}). \quad (B.7)\]

Proof. By the Leibniz formula,

\[\frac{\partial}{\partial\tau}\int_{\mathcal{X}}\| \nabla_{\mathbf{x}}\log q^{\tau}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p(\mathbf{x})\|^{2}p(\mathbf{x})d\mathbf{x}\] \[= \int_{\mathcal{X}}\frac{\partial}{\partial\tau}\| \nabla_{\mathbf{x}}\log q^{\tau}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p(\mathbf{x})\|^{2}p(\mathbf{x})d{\mathbf{x}}\] \[= \int_{\mathcal{X}}2\left(\tau \| \nabla_{\mathbf{x}}\log q(\mathbf{x})\|^{2} - \langle \nabla_{\mathbf{x}}\log q(\mathbf{x}),\nabla_{\mathbf{x}}\log p(\mathbf{x})\rangle\right)p(\mathbf{x})d{\mathbf{x}}.\]

Since the Fisher score is quadratic in \(\tau\) , the critical point corresponds to the global minimum. By the assumption, \(\langle \nabla_{\mathbf{x}}\log q(\mathbf{x}),\nabla_{\mathbf{x}}\log p(\mathbf{x})\rangle = \tau (\mathbf{x})\| \nabla_{\mathbf{x}}\log q(\mathbf{x})\|^{2}\) . Hence,

\[\tau^{*} = \frac{\mathbb{E}_{X\sim p}\left(\tau(X)\left\|\nabla\log q(X)\right\|^{2}\right)}{\mathbb{E}_{X\sim p}\left(\left\|\nabla\log q(X)\right\|^{2}\right)}.\]

Lemma B.4. Let \(\tau (\mathbf{x})\) be a tempering field. For any \(\tau >0\) it holds that

\[F(p,q^{\tau}) = \int_{\mathcal{X}}|\tau -\tau (\mathbf{x})|^{2}\| \nabla_{\mathbf{x}}\log q(\mathbf{x})\|^{2}p(\mathbf{x})d\mathbf{x}.\]

Proof. Since \(\tau (\mathbf{x})\) is a tempering field,

\[\| \nabla_{\mathbf{x}}\log q^{\tau}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p(\mathbf{x})\|\] \[= \| (\tau -\tau (\mathbf{x}))\nabla_{\mathbf{x}}\log q(\mathbf{x}) + (\tau (\mathbf{x})\nabla_{\mathbf{x}}\log q(\mathbf{x}) - \nabla_{\mathbf{x}}\log p(\mathbf{x}))\|\] \[= \| (\tau -\tau (\mathbf{x}))\nabla_{\mathbf{x}}\log q(\mathbf{x})\| .\]

Proposition B.5. Let \(\tau^{*}\) be the optimal tempering and \(\tau (\mathbf{x})\) a tempering field.

\[F(p,q^{\tau^{*}}) = \mathbb{E}\left(\| \nabla \log q(X)\|^{2}\tau^{2}(X)\right) - \frac{\left(\mathbb{E}\left(\tau(X)\| \nabla \log q(X)\|^{2}\right)\right)^{2}}{\mathbb{E}\left(\| \nabla \log q(X)\|^{2}\right)}.\]

Proof. By Proposition 3.2 and Lemma B.4,

\[\int_{X}\left\| \nabla_{\mathbf{x}}\log q^{\tau^{*}}(\mathbf{x}) - \nabla_{\mathbf{x}}\log p(\mathbf{x})\right\|^{2}p(\mathbf{x})d\mathbf{x}\] \[= \mathbb{E}\left(\left|\tau^{*} - \tau (X)\right|^{2}\left\| \nabla \log q(X)\right\|^{2}\right)\] \[= \mathbb{E}\left(\left(\frac{\mathbb{E}\left(\tau(X^{\prime})\left\| \nabla\log q(X^{\prime})\right\|^{2}\right)}{\mathbb{E}\left(\left\| \nabla\log q(X^{\prime})\right\|^{2}\right)} -\tau (X)\right)^{2}\left\| \nabla \log q(X)\right\|^{2}\right)\] \[= \mathbb{E}\left(\left(\frac{\left\| \nabla\log q(X)\right\| \mathbb{E}\left(\tau(X^{\prime})\left\| \nabla\log q(X^{\prime})\right\|^{2}\right)}{\mathbb{E}\left(\left\| \nabla\log q(X^{\prime})\right\|^{2}\right)} -\left\| \nabla \log q(X)\right\| \tau (X)\right)^{2}\right)\] \[= \frac{\left(\mathbb{E}\left(\tau(X)\left\| \nabla\log q(X)\right\|^{2}\right)\right)^{2}}{\mathbb{E}\left(\left\| \nabla\log q(X)\right\|^{2}\right)} -2\frac{\left(\mathbb{E}\left(\tau(X)\left\| \nabla\log q(X)\right\|^{2}\right)\right)^{2}}{\mathbb{E}\left(\| \nabla\log q(X)\|^{2}\right)} +\mathbb{E}\left(\| \nabla\log q(X)\|^{2}\tau^{2}(X)\right)\] \[= \mathbb{E}\left(\| \nabla\log q(X)\|^{2}\tau^{2}(X)\right) - \frac{\left(\mathbb{E}\left(\tau(X)\left\| \nabla\log q(X)\right\|^{2}\right)\right)^{2}}{\mathbb{E}\left(\| \nabla\log q(X)\|^{2}\right)}.\]

Proposition 3.3. Let \(\tau (\mathbf{x})\) be a tempering field. For any \(\tau >0\) it holds that

\[F(p,q^{\tau}) = \mathbb{E}_{X\sim p}\left(|\tau -\tau (X)|^{2}\left\| \nabla \log q(X)\right\|^{2}\right). \quad (B.8)\]

Further, when \(\tau^{*} > 0\) is the optimal tempering

\[F(p,q^{\tau^{*}}) = \mathbb{E}_{X\sim p}\left(\| \nabla \log q(X)\|^{2}\tau^{2}(X)\right) - \frac{\left(\mathbb{E}_{X\sim p}\left(\tau(X)\left\| \nabla \log q(X)\right\|^{2}\right)\right)^{2}}{\mathbb{E}_{X\sim p}\left(\| \nabla \log q(X)\|^{2}\right)}. \quad (B.9)\]

Proof. This is a combined result of Lemma B.4 and Proposition B.5.

Corollary B.7. Assume that the expert choice model follows the Bradley- Terry model or the exponential RUM. The scores of the belief and the MWD are collinear. That is, there exists a scalar- valued function \(\tau (\mathbf{x})\) such that,

\[\nabla \log p(\mathbf{x}) = \tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x}).\]

Proof. The result follows directly from Definition 5 and Theorems 3.1 and A.1.

## C METHOD

## C.1 TRAINING JOINT AND MARGINALS DISTRIBUTIONS

Recent works discuss in detail how diffusion models can be used to learn between joint and arbitrary conditional distributions, whereas modeling the marginals is not always straightforward (Gloeckler et al., 2024). We adopt a simplified approach to estimate the marginal score function by leveraging a corruption- based marginalization strategy.

To model simultaneously both the joint distribution \(p_{\mathbf{x} > \mathbf{x}^{\prime}}(\mathbf{x},\mathbf{x}^{\prime})\) and the marginal distribution \(p_{w}(\mathbf{x})\) , we introduce a binary conditioning variable joint \(\in \{\mathrm{true},\mathrm{false}\}\) into the score model. During training, we randomly set joint \(=\) false with \(50\%\) probability, and in this case, we mask the input \(\mathbf{x}_{t}^{\prime}\) by replacing it with Gaussian noise \(\mathcal{N}(\mathbf{0},\sigma_{t}^{2}\mathbf{I})\) , where \(\sigma_{t}\) is the current noise level. We then compute the denoising score matching loss only over the winner dimensions \(\mathbf{x}\) (i.e., the first \(d\)

> **Image description.** A technical figure consisting of two side-by-side panels, (a) and (b), which visually compare the quality of a density estimate (referred to as "Tempered MWD") under two different training conditions. Both panels display a dark, deep purple background against which two crescent-shaped probability distributions are rendered using a combination of blue data points and a faint, lighter blue density field.
>
> Panel (a), labeled "(a) Tempered MWD, w/o joint training," shows the density estimate generated without utilizing joint training data. The two crescent shapes are visible, but the surrounding density field appears somewhat less defined and less cohesive. The blue data points are scattered within the shapes, and the overall distribution seems slightly less refined.
>
> Panel (b), labeled "(b) Tempered MWD, w/ joint training," shows the density estimate generated after training on the full joint data. In this panel, the two crescent shapes are significantly more clearly defined. The density field is smoother, more concentrated, and appears to more accurately capture the underlying distribution compared to panel (a). The blue data points are densely packed within the shapes, indicating a higher quality and more accurate marginal distribution estimate.
>
> The figure visually demonstrates the improvement in the quality of the final density estimate when the score model is trained using the full joint data, as indicated by the clearer and more refined shapes in panel (b) compared to panel (a).

<center>Figure C.1: Replication of Fig. 1 using (a) only winner samples, with the score model trained only for the MWD \(p_{w}(\mathbf{x})\) . The quality of the final density estimate, i.e., the 'tempered' MWD, is clearly inferior compared to (b) training on the full joint \(p_{\mathbf{x} > \mathbf{x}'}(\mathbf{x},\mathbf{x}')\) using both winners and losers. </center>

Table C.1: Wasserstein distance between the winner marginal samples using \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}',\sigma ,\mathrm{joint} = 0,\mathrm{temp} = 0)\) and \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}',\sigma ,\mathrm{joint} = 1,\mathrm{temp} = 0)\) . The last column reports the distance as a percentage relative to the fourth column of Table 1.

| $p(\mathbf{x})$ | Wasserstein | Relative proportion |
| :--- | :--- | :--- |
| Onemoon2D | 0.085 ($\pm$ 0.009) | 23% |
| Ring2D | 0.077 ($\pm$ 0.012) | 18% |
| Twomoons2D | 0.072 ($\pm$ 0.010) | 18% |
| Mixturegaussians4D | 0.071 ($\pm$ 0.002) | 5% |
| Onegaussian4D | 0.069 ($\pm$ 0.001) | 6% |
| Stargaussian6D | 0.156 ($\pm$ 0.001) | 12% |
| Mixturegaussians10D | 0.366 ($\pm$ 0.001) | 28% |
| Onegaussian16D | 0.671 ($\pm$ 0.001) | 13% |

components of \(\nabla \log p_{\mathbf{x} > \mathbf{x}'}(\mathbf{x},\mathbf{x}')\) . When joint \(= \mathrm{true}\) , we train the model to predict the full joint score over both \(\mathbf{x}\) and \(\mathbf{x}'\) .

At sampling time, to generate samples from the marginal distribution \(p_{w}(\mathbf{x})\) using ALD, we similarly set joint \(= \mathrm{false}\) and replace \(\mathbf{x}_i'\) with Gaussian noise \(\mathcal{N}(0,\sigma_i^2\mathbf{I})\) at each iteration of ALD. Fig. C.1 demonstrates the benefit of training the score model on the full joint data while using the proposed marginalization method to model the marginal score.

Does the diffusion model learn to marginalize? To empirically study how well our approach for the diffusion marginalization is able to learn to marginalize, we analyze the difference between the marginal samples from the joint model, and the samples from the marginal model. Specifically, Table C.1 reports the Wasserstein distance between the samples from (i) the true winner marginal of the joint model: \(\mathbf{x}\) where \((\mathbf{x},\mathbf{x}') \sim p_{\mathbf{x} > \mathbf{x}'}(\mathbf{x},\mathbf{x}')\) using the score model \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}',\sigma ,\mathrm{joint} = 1,\mathrm{temp} = 0)\) , and (ii) the winner marginal of the joint model: \(\mathbf{x}\) where \(\mathbf{x} \sim p_{w}(\mathbf{x})\) using the score model \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}',\sigma ,\mathrm{joint} = 0,\mathrm{temp} = 0)\) . To quantify the similarity, we can contrast the Wasserstein distance between these two estimates to the one we have between the target and the score- based estimate (reported in the fourth column in Table 1). This relative difference is typically below \(20\%\) , indicating that the marginalization method performs well.

## C.2 DETAILS ON MODELING THE 'TEMPERED' MWD

To learn the MWD \(p_{w}(\mathbf{x})\) , having only access to samples from the joint distribution of winners and losers \(p_{\mathbf{x} > \mathbf{x}^{\prime}}(\mathbf{x},\mathbf{x}^{\prime})\) , we propose learning the full joint and the first marginal to capture the preference relationships in the data while enabling sampling from the tempered marginal via score- scaled ALD. To that end, we parametrize the score model \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}^{\prime},\sigma ,\mathrm{joint},\mathrm{temp})\) such that \((\mathbf{x},\mathbf{x}^{\prime})\mapsto\) \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}^{\prime},\sigma_{\mathrm{min}},\mathrm{joint} = 1,\mathrm{temp} = 0)\) models the score of the joint distribution of winners and losers, i.e., \(\nabla \log p_{\mathbf{x} > \mathbf{x}^{\prime}}(\mathbf{x},\mathbf{x}^{\prime})\) , and \((\mathbf{x},\mathrm{temp})\mapsto \mathbf{s}_{\theta}(\mathbf{x},\emptyset ,\sigma_{\mathrm{min}},\mathrm{joint} = 0,\mathrm{temp})\) models the score of the MWD and its 'tempered' version, i.e., \(\nabla \log p_{w}(\mathbf{x})\) and \(\tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x})\) , respectively. This allows training the score network with both winners and losers, while still enabling sampling from the belief density via the approximation \(\mathbf{s}_{\theta}(\mathbf{x},\emptyset ,\sigma_{\mathrm{min}},\mathrm{joint} = 0,\mathrm{temp} = 1)\approx \tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x}) = \nabla \log p(\mathbf{x})\) by Corollary B.7. Fig. C.1 validates that the joint score model is superior to directly learning the marginal from only the winner samples.

We implement the method by training a score model through the denoising score matching equation 1 on the concatenation of winners and losers, with random masking of losers during training (for more details, see Appendix C.1). Technically speaking, to enable sampling via reverse diffusion and ALD, we use a noise distribution \(p_{\mathrm{train}}(\sigma)\) during training, defined as a mixture of a Dirac delta on a cosine noise schedule and a \(\mathrm{LogNormal}(P_{\mathrm{mean}},P_{\mathrm{std}}^{2})\) , with mixture weight \(\phi = 0.5\) assigned to the Dirac delta component. We stay as close as possible to the EDM- style diffusion model (Karras et al., 2024a). Specifically, we use the perturbation kernel \(p_{\sigma}(\tilde{\mathbf{x}} |\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}};\mathbf{x},\sigma^{2}\mathbf{I})\) , which aligns with EDM and defines a forward diffusion process from \(\sigma_{\mathrm{min}}\) to \(\sigma_{\mathrm{max}}\) , where \(p_{\sigma_{\mathrm{min}}}(\mathbf{x})\approx p(\mathbf{x})\) and \(p_{\sigma_{\mathrm{max}}}(\mathbf{x})\approx \mathcal{N}(\mathbf{0},\sigma_{\mathrm{max}}^{2}\mathbf{I})\) . After training \(\mathbf{s}_{\theta}(\mathbf{x},\mathbf{x}^{\prime},\sigma ,\mathrm{joint},0)\) on perturbed winners and losers, we sample from the belief density using the score- scaled ALD. We can either stop here, or optionally train the tempered marginal score network \(\mathbf{s}_{\theta_{\mathrm{MWD}}}(\mathbf{x},\sigma ,\mathrm{temp})\) whose weights can be initialized to that of \(\mathbf{s}_{\theta}\) , through denoising score matching using the sampled data. Finally, we use the loss weighting \(\ell (\sigma) = \sigma^{2}\) . Algorithms 1- 2 summarize the method.

## C.3 SCORE MODEL

We follow as closely as possible the EDM2 specifications used in the 2D toy experiment in (Karras et al., 2024a). For both the joint score network and the tempered MWD score network, we use an MLP with one input layer and four hidden layers, SiLU activation functions (Hendrycks & Gimpel, 2016) are applied after each hidden layer, and implemented using the magnitude- preserving primitives from EDM2 (Karras et al., 2024b). In the joint score network, the input is a \((2d + 3)\) - dimensional vector \((x,x^{\prime},\sigma ,\mathrm{joint},\mathrm{temp})\) , and the output of each hidden layer has \(h\) features, where \(h\in \{32,64,96,128\}\) depending on the experiment dimensionality. In the MWD score network, the input is a \((d + 3)\) - dimensional vector \((x,\sigma ,0,\mathrm{temp})\) . The binary variables joint and temp are linearly embedded into an \(h / 4\) - dimensional space. Further, a simple residual connection is applied to the embedded variables through all hidden layers. Otherwise, we use the same preconditioning for the score network as described in EDM (Karras et al., 2022).

## C.4 BELIEF DENSITY RATIO MODEL

We parametrize the belief density ratio \(r_{\theta}(\mathbf{x},\mathbf{x}^{\prime})\approx p(\mathbf{x}^{\prime}) / p(\mathbf{x})\) via parameterizing the unnormalized log density \(f_{\theta}(\mathbf{x})\approx \log p(\mathbf{x}) + constant\) as an MLP with three hidden layers with SiLu activations, and one output layer, such that \(\log r_{\theta}(\mathbf{x},\mathbf{x}^{\prime}) = f_{\theta}(\mathbf{x}^{\prime}) - f_{\theta}(\mathbf{x})\) . The number of hidden units is tied to that of the score model (Section C.3). Regularization of the weights \(\theta\) is important for obtaining sensible results. To this end, we apply adaptive \(\ell_{2}\) - regularization using the Adam optimizer (Kingma & Ba, 2015) with weight decay. In contrast, standard \(\ell_{2}\) - regularization, corresponding to AdamW (Loshchilov & Hutter, 2019), yielded slightly inferior empirical performance. We set the weight decay to \(10^{- 3}\) , except in the small data \(n = 100d\) experiments, where we use a higher value of \(3\times 10^{- 3}\) .

## C.5 TEMPERING FIELD ESTIMATE

We estimate the integral ratio in Eq. 6 by approximating both the numerator and the denominator using Monte Carlo estimates, using the MWD \(p_{w}(\mathbf{x})\) as importance sampler,

\[\tau (\mathbf{x})\approx s\left(\frac{\sum\frac{1}{p_{w}(\mathbf{x}_{i})}\frac{1}{1+r_{s}(\mathbf{x},\mathbf{x}_{i})}}{\sum\frac{1}{p_{w}(\mathbf{x}_{i})}\frac{r_{s}(\mathbf{x},\mathbf{x}_{i})}{(1+r_{s}(\mathbf{x},\mathbf{x}_{i}))^{2}}}\right), \quad (C.1)\]

where the sums are over the importance samples \(\mathbf{x}_{i}\sim p_{w}\) . This Monte Carlo ratio estimator is biased, but consistent.

For the \(d\) - dimensional target, we use \(2000d\) importance samples to estimate the integrals in the tempering field. The importance weights are computed using the probability- flow ODE of the MWD diffusion model (see Appendix C.7 for details). The estimated tempering field \(\tau (\mathbf{x})\) is clipped such that \(1\leq \tau (\mathbf{x})\leq Q_{\tau}(0.99)\) , where \(Q_{\tau}(0.99)\) denotes the \(99\%\) quantile of the estimated tempering field values (for visualization, we use the \(99.9\%\) quantile). The lower bound follows directly from the theory (i.e., from formula 6), while the upper bound is introduced for numerical stability to remove outliers.

## C.6 SCORE-SCALED ALD

Score- scaled ALD uses the scaled score \(\tau (\mathbf{x})\nabla \log p_{w}(\mathbf{x})\) , where \(\nabla \log p_{w}(\mathbf{x})\) is replaced by our estimated score. While \(\tau (\mathbf{x})\) is not the ALD step size \(\epsilon >0\) , it is clear that \(\tau (\mathbf{x})\) influences the ALD update in a manner similar to \(\epsilon\) . To ensure convergence of score- scaled ALD, at each ALD step we use the step size \(\epsilon = \frac{\epsilon_{\mathrm{base}}}{\tau(\mathbf{x})}\frac{\sigma^{2}}{\sigma_{\mathrm{max}}^{2}}\) , where \(\epsilon_{\mathrm{base}}\) is the base step size to be specified. The required number of iterations \(T\) should be chosen with respect to \(\epsilon_{\mathrm{base}}\) . In our experiments, we use \(L = 50\) , \(T = 40\) , and \(\epsilon_{\mathrm{base}} = 0.15\) , except in the \(2D\) - experiments, where we use \(\epsilon_{\mathrm{base}} = 7.0\) with \(L = 15\) and \(T = 40\) . In the \(2D\) - experiments, we keep the original domain scale \(([- 3,3]^{d})\) and do not rescale it to \([- 0.5,0.5]^{d}\) , unlike in the other experiments.

Regarding the injection of a deterministic ALD noise schedule during denoising score- matching training, we find that the cosine schedule yields better empirical performance, while the noise schedule corresponding to the EDM time- step discretization is also a natural option.

## C.7 DENSITY EVALUATION OF A DIFFUSION MODEL

Chen et al. (2018) showed that for a random variable whose probability density evolves over time, with dynamics \(d\mathbf{x} = \tilde{f} (\mathbf{x}_{t},t)d t\) (where \(\tilde{f}\) is Lipschitz continuous in \(\mathbf{x}\) and continuous in \(t\) ), the log- density at a point \(\mathbf{x}_{0}\) is given by

\[\log p_{0}(\mathbf{x}_{0}) = \log p_{T}(\mathbf{x}_{T}) + \int_{0}^{T}\nabla \cdot \tilde{f} (\mathbf{x}_{t},t)d t, \quad (C.2)\]

where \(\nabla\) denotes the divergence operator, which is equal to the trace of the Jacobian. In practice, the divergence is often approximated using the Skilling- Hutchinson trace estimator (Grathwohl et al., 2019),

\[\nabla \cdot \tilde{f} (\mathbf{x}) = \mathbb{E}_{\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})}[\epsilon^{\mathsf{T}}J_{\tilde{f}}(\mathbf{x})\epsilon ],\]

where the expectation is typically estimated using a finite number of samples.

Now, applying this to EDM- type diffusion model, which is characterized by the probability- flow ODE

\[d\mathbf{x} = -\sigma \nabla \log p_{\sigma}(\mathbf{x})d\sigma , \quad (C.3)\]

we can compute the probability density at a point \(\mathbf{x}\) as

\[\log p(\mathbf{x})\approx \log \mathcal{N}(\mathbf{x}_{\sigma_{\mathrm{max}}};\mathbf{0},\sigma_{\mathrm{max}}\mathbf{I}) - \int_{\sigma_{\mathrm{min}}}^{\sigma_{\mathrm{max}}}\sigma \nabla \cdot \nabla \log p_{\sigma}(\mathbf{x}_{\sigma})d\sigma , \quad (C.4)\]

where the score \(\nabla \log p_{\sigma}(\mathbf{x})\) is approximated by the score network \(\mathbf{s}_{\theta}(\mathbf{x},\sigma)\) , and the divergence term is estimated using the Skilling- Hutchinson estimator.

The ODE in Eq. C.3 is often stiff at small noise scales, requiring a careful choice of numerical integration method for stable results. To integrate Eq. C.4, we solve the coupled system

\[\frac{d\mathbf{x}}{d\sigma} = -\sigma \mathbf{s}_{\theta}(\mathbf{x},\sigma),\qquad \frac{d\log p(\mathbf{x})}{d\sigma} = \sigma \nabla \cdot \mathbf{s}_{\theta}(\mathbf{x},\sigma),\]

where the latter ODE tracks the accumulation of the log- density.

To solve the coupled system, we use the implicit Adams- Bashforth- Moulton black- box ODE solver implemented in the torchdiffeq package (Chen, 2018). We compute the divergence exactly using automatic differentiation, although the Skilling- Hutchinson estimator also worked, sometimes even yielding better results. Finally, to further stabilize the density estimates, we clamp the importance weights \(1 / p(\mathbf{x})\) to the interval defined by their 1st and 90th percentiles.

## D RUM UNDER THE SPACE REPARAMETERIZATION

This appendix studies the exponential RUM and the Bradley- Terry model under space reparameterization, and verifies that both RUMs are invariant under this transformation. This justifies our approach of transforming possible non- uniform \(\lambda (\mathbf{x})\) into a uniform distribution.

The invariance to the space reparameterization holds for Fechnerian RUMs with the winner density,

\[p_{w}(\mathbf{x}) = 2\lambda (\mathbf{x})\int_{\mathcal{X}}F\left(\log p(\mathbf{x}) - \log p(\mathbf{x}^{\prime})\right)\lambda (\mathbf{x}^{\prime})d\mathbf{x}^{\prime},\]

where \(F\) is a cumulative distribution function of the choice probability. Let a transformation \(T\) push the sampling density \(\lambda\) into the uniform distribution on the hypercube, that is \(T_{\#}\lambda = \mathrm{Unif}([0,1]^{d})\) . It is enough to show that the winner density in the transformed space has the same form but with uniform sampling density and different belief density. To that end, denote \(\mathbf{y} = T(\mathbf{x})\) and note that

\[1 = \lambda (T^{-1}(\mathbf{y}))|\operatorname *{det}\nabla T^{-1}(\mathbf{y})|.\]

Hence, \(\lambda (\mathbf{x}) = \lambda (T^{- 1}(\mathbf{y}))\) in the front of the integral cancels the volume change under the transformation \(T\) . The integral in the new coordinate system \(\mathbf{y}^{\prime}\) can be computed by the change of variable \(\begin{array}{r}{\int_{\mathcal{X}}G_{\mathbf{y}^{\prime}}d([T^{- 1}]_{\#}\mu) = \int_{[0,1]^{d}}[G_{\mathbf{y}}\circ T^{- 1}]d\mu} \end{array}\) for \(G_{\mathbf{y}}(\mathbf{x}^{\prime})\coloneqq F\left(\log p(T^{- 1}(\mathbf{y})) - \log p(\mathbf{x}^{\prime})\right)\) and \(\mu\) denotes the Lebesgue measure on the hypercube \([0,1]^{d}\) . The winner density in the transformed space reads as

\[p_{w,trans}(\mathbf{y}) = 2\int_{[0,1]^{d}}F\left(\log p(T^{-1}(\mathbf{y})) - \log p(T^{-1}(\mathbf{y}^{\prime}))\right)d\mathbf{y}^{\prime}.\]

That is, the winner density in the transformed space follows the same RUM than in the original space, but with a uniform sampling distribution and a belief density \(p_{trans}(\mathbf{y})\propto p(T^{- 1}(\mathbf{y}))\) . Note that the normalization of \(p_{trans}(\mathbf{y})\) is irrelevant, as the constant cancels out within the logarithmic utility difference. Consequently, the scale of the RUM noise (parameterized by \(F\) ) remains invariant under the space reparameterization.

## E EXPERIMENTAL DETAILS

## E.1 TARGET DISTRIBUTIONS

The log unnormalized densities of the target distributions used in the synthetic experiments are provided below.

\[\mathbf{Onemoon2D}:\qquad -\frac{1}{2}\left(\frac{\|{\bf x}\| - 2}{0.2}\right)^{2} - \frac{1}{2}\left(\frac{{\bf x}_{1} + 2}{0.3}\right)^{2}\]

\[\mathbf{Twomoons2D}:\frac{-(\|{\bf x}\| - 1)^{2}}{0.08} -\frac{(|{\bf x}_{1}| - 2)^{2}}{0.18} +\log \left(1 + e^{-\frac{4{\bf x}_{1}}{0.09}}\right)\]

\[\mathbf{Ring2D}:\log \left(\sum_{i = 1}^{k}\left(\frac{32}{\pi} e^{-32(\|{\bf x}\| - i - 1)^{2}}\right)\right),\quad \mathrm{where} k = 1\]

\[\mathbf{Stargaussian6D}:\qquad \log \left(\frac{1}{2}\mathcal{N}(\mathbf{x}\mid \pmb {\mu},\Sigma_{1}) + \frac{1}{2}\mathcal{N}(\mathbf{x}\mid \pmb {\mu},\Sigma_{2})\right),\]

\[\sigma^{2} = 1, \rho = 0.9, d = 6, \mu = 31_{d}, \Sigma_{1} = \left( \begin{array}{cccc}\sigma^{2} & \rho \sigma^{2} & \rho \sigma^{2} & \dots & \rho \sigma^{2} \\ \rho \sigma^{2} & \sigma^{2} & \rho \sigma^{2} & \dots & \rho \sigma^{2} \\ \rho \sigma^{2} & \rho \sigma^{2} & \sigma^{2} & \dots & \rho \sigma^{2} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \rho \sigma^{2} & \rho \sigma^{2} & \rho \sigma^{2} & \dots & \sigma^{2} \end{array} \right),\]

\[\Sigma_{2} = \left( \begin{array}{cccc}\sigma^{2} & -\rho \sigma^{2} & \rho \sigma^{2} & \dots & (-1)^{d - 1}\rho \sigma^{2} \\ -\rho \sigma^{2} & \sigma^{2} & -\rho \sigma^{2} & \dots & (-1)^{d - 2}\rho \sigma^{2} \\ \rho \sigma^{2} & -\rho \sigma^{2} & \sigma^{2} & \dots & (-1)^{d - 3}\rho \sigma^{2} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ (-1)^{d - 1}\rho \sigma^{2} & (-1)^{d - 2}\rho \sigma^{2} & (-1)^{d - 3}\rho \sigma^{2} & \dots & \sigma^{2} \end{array} \right)\]

\[\mathbf{Mixturegaussians}, d \in \{4, 10\} : \qquad \log \left(\frac{1}{4} \sum_{i = 1}^{4} \exp \left(-\frac{1}{2} (\mathbf{x} - \pmb{\mu}_{i})^{\top} \Sigma_{i}^{-1} (\mathbf{x} - \pmb{\mu}_{i})\right)\right),\]

\[\mathbf{where} \quad \pmb{\mu}_{i} = r \cdot \frac{\mathbf{v}_{i}}{\| \mathbf{v}_{i}\|}, \quad r = 3, \quad \mathbf{v}_{1} = \mathbf{1}_{d}, \quad \mathbf{v}_{2} = -\mathbf{1}_{d}, \quad \mathbf{v}_{3} = \left[(-1)^{j}\right]_{j = 1}^{d}, \quad \mathbf{v}_{4} = -\mathbf{v}_{3},\]

\[\Sigma_{i} = \mathbf{Q}_{i} \cdot \mathrm{diag}(\sigma_{0}^{2}, \sigma^{2}, \dots , \sigma^{2}) \cdot \mathbf{Q}_{i}^{\top}, \quad \sigma_{0}^{2} = 1, \sigma^{2} = 0.1, \mathbf{Q}_{i} = [\hat{\pmb{\mu}}_{i}, \dots ] \in \mathbb{R}^{d \times d}\]

\[\mathbf{Gaussian}, D \in \{4, 16\} : \quad -\frac{1}{2} (\mathbf{x} - \pmb {\mu})^{\top} \Sigma^{-1} (\mathbf{x} - \pmb {\mu}), \quad \pmb {\mu} = 2 \left( \begin{array}{c} (-1)^{1} \\ \vdots \\ (-1)^{d} \end{array} \right),\]

\[\Sigma = \left( \begin{array}{cccc}\frac{d}{10} & \frac{d}{15} & \dots & \frac{d}{15} \\ \frac{d}{15} & \frac{d}{10} & \dots & \frac{d}{15} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{d}{15} & \frac{d}{15} & \dots & \frac{d}{10} \end{array} \right)\]

## E.2 OTHER EXPERIMENTAL DETAILS

Hyperparameters and optimization details. The score models are trained for varying number of iterations \((d = 2:8192, 2 < d < 10:12288, d \geq 10:15360)\) with the Adam optimizer (Kingma & Ba, 2015) and a batch size of \(\min \{n, 4000\}\) pairwise comparisons, where \(n\) is the number of pairwise comparisons in the dataset. For the 2D experiments, we follow (Karras et al., 2024a) and use an adaptive learning rate, specifically a decay schedule of \(\alpha_{ref} / \max (\mathrm{iter}, \mathrm{iter}_{ref}, 1)\) , with \(\alpha_{ref} = 0.005\) and \(\mathrm{iter}_{ref} = 1024\) iterations. We use a power- function EMA profile with \(\sigma_{ref} = 0.01\) . The setup is somewhat sensitive to hyperparameters, and performance can vary depending on their tuning. We expect to achieve better or worse performance in the experiments depending on how well

Table E.1: Robustness to RUM noise family and noise level. The method score- \(\tau (\mathbf{x})\) assumes the Bradley- Terry model with noise level \(s = 0.7797\) , while the data-generating process is varied across three RUMs: Exponential RUM \((W \sim \mathrm{Exp}(s))\) , Bradley- Terry \((W \sim \mathrm{Gumbel}(0, s))\) , and Thurstone- Mosteller \((W \sim \mathcal{N}(0, s^2))\) , each evaluated at noise levels \(s \in \{0.1, 0.7797, 1.0, 5.0\}\) . Reported values are averages over 10 repetitions of the Wasserstein distance \((\downarrow)\) .

| Dataset / RUM | $s = 0.1$ | $s = 0.7797$ | $s = 1.0$ | $s = 5.0$ |
| :--- | :--- | :--- | :--- | :--- |
| **Onemoon2D** | | | | |
| $W \sim \mathrm{Exp}(s)$ | 0.693 (±0.078) | 0.266 (±0.122) | 0.252 (±0.122) | 0.242 (±0.117) |
| $W \sim \mathrm{Gumbel}(0, s)$ | 0.358 (±0.138) | 0.366 (±0.145) | 0.372 (±0.149) | 0.599 (±0.132) |
| $W \sim \mathcal{N}(0, s^2)$ | 0.294 (±0.139) | 0.318 (±0.124) | 0.321 (±0.122) | 0.549 (±0.119) |
| **Twomoons2D** | | | | |
| $W \sim \mathrm{Exp}(s)$ | 0.577 (±0.069) | 0.348 (±0.125) | 0.371 (±0.127) | 0.383 (±0.139) |
| $W \sim \mathrm{Gumbel}(0, s)$ | 0.481 (±0.090) | 0.440 (±0.089) | 0.446 (±0.094) | 0.500 (±0.060) |
| $W \sim \mathcal{N}(0, s^2)$ | 0.375 (±0.144) | 0.345 (±0.158) | 0.346 (±0.160) | 0.447 (±0.080) |
| **Ring2D** | | | | |
| $W \sim \mathrm{Exp}(s)$ | 0.557 (±0.071) | 0.423 (±0.066) | 0.427 (±0.068) | 0.453 (±0.089) |
| $W \sim \mathrm{Gumbel}(0, s)$ | 0.462 (±0.128) | 0.387 (±0.075) | 0.368 (±0.074) | 0.449 (±0.041) |
| $W \sim \mathcal{N}(0, s^2)$ | 0.495 (±0.109) | 0.430 (±0.086) | 0.420 (±0.081) | 0.491 (±0.073) |

the hyperparameters are tuned. The chosen hyperparameters are likely suboptimal, and we expect performance gains, especially in higher- dimensional experiments, if the hyperparameters are well tuned.

Environment. All experiments are conducted on a server equipped with nodes containing dual Intel Xeon Cascade Lake processors (20 cores each, 2.1GHz). While exact training times and memory usage were not recorded, the datasets and score network architectures used are relatively lightweight.

Experiment replications. Every experiment was replicated with 10 different seeds, ranging from 1 to 10.

Baseline. We used the official implementation of (Mikkola et al., 2024) and the provided config files to match the hyperparameter configuration used in their experiments to the closest experiment in our paper. For example, for 2D experiments, we use the config file that was used in their Onemoon2D experiment.

## E.3 ROBUSTNESS TO RUM NOISE FAMILY AND NOISE LEVEL

To study robustness of the method for misspecification in the data- generation process, we rerun Onemoon2D, Twomoons2D, and Ring2D by varying the true data- generation RUM noise family \(W\) and the noise level \(s\) . In all cases, our method assumes the Bradley- Terry model with fixed noise level \(s = \sqrt{6 / \pi^2}\) . Table E.1 reports the results.

The results suggest that lower true noise levels generally lead to higher- quality estimates (note that for \(W \sim \mathrm{Exp}(s)\) , a smaller \(s\) corresponds to a higher noise level). However, using a true noise level that roughly matches the assumed model often yields better or comparable performance than choosing a very small noise level (note that the model is correctly specified only when \(W \sim \mathrm{Gumbel}(0, s)\) ). This observation aligns with the theoretical identifiability of the problem discussed in Section 2.3. Overall, the results appear relatively robust to misspecification of both the noise family and the noise level.

## E.4 RUNTIME BREAKDOWN

The computation times are profiled for the experiments in Section 5. Table E.2 reports the total runtime allocated to three main components: DSM training of the score network (Section 4.1); estimation of the tempering field, including training the density ratio model \(r_{\theta}\) (Section 4.2); and score- scaled ALD sampling (Section 4.3) with the number of samples varying from 25k to 40k. The final column shows the average time required to generate a single sample. In practice, this

also depends on the sampling batch size, which we have not optimized and which is constrained by available memory.

Table E.2: Breakdown of computation times (in seconds). The table reports mean and standard deviation over 10 replicates.

| Model | DSM training | $\tau(x)$ estimation | Sampling Total | Sampling Per Sample |
| :--- | :--- | :--- | :--- | :--- |
| Onemoon2D | 80 ($\pm$ 16) | 49 ($\pm$ 12) | 690 ($\pm$ 45) | 0.0276 |
| Ring2D | 82 ($\pm$ 15) | 47 ($\pm$ 10) | 692 ($\pm$ 41) | 0.0277 |
| Twomoons2D | 81 ($\pm$ 14) | 48 ($\pm$ 10) | 701 ($\pm$ 42) | 0.0280 |
| Mixturegaussians4D | 262 ($\pm$ 51) | 4560 ($\pm$ 873) | 6958 ($\pm$ 743) | 0.2783 |
| Onegaussian4D | 283 ($\pm$ 46) | 2660 ($\pm$ 279) | 7187 ($\pm$ 404) | 0.2872 |
| Stargaussian6D | 440 ($\pm$ 54) | 3364 ($\pm$ 496) | 12053 ($\pm$ 729) | 0.4018 |
| Mixturegaussians10D | 653 ($\pm$ 47) | 6728 ($\pm$ 1030) | 27314 ($\pm$ 853) | 0.6829 |
| Onegaussian16D | 797 ($\pm$ 24) | 4977 ($\pm$ 223) | 47477 ($\pm$ 727) | 1.1870 |

## F PLOTS

## F.1 PLOTS OF LEARNED BELIEF DENSITIES

> **Image description.** A comparative visualization consisting of two side-by-side panels, illustrating the results of density estimation from two different machine learning models, labeled "Score-based" and "Flow." Both panels feature a dark, near-black background with bright blue elements representing the learned distribution.
>
> The image is divided into two distinct sections:
>
> 1.  **Panel (a) - Score-based:**
>     *   **Label:** Located at the bottom left, the panel is labeled "(a) Score-based."
>     *   **Visual Content:** This panel displays a continuous, smooth, crescent-shaped distribution in bright blue. The shape is oriented vertically, resembling a "moon" or a curved arc. The blue color is concentrated along the curve, suggesting a dense probability field or heatmap representation of the learned distribution.
>
> 2.  **Panel (b) - Flow:**
>     *   **Label:** Located at the bottom right, the panel is labeled "(b) Flow."
>     *   **Visual Content:** This panel also displays a curved, crescent-shaped distribution, similar in overall form to Panel (a). However, instead of a continuous density field, the distribution is represented by a dense cluster of numerous individual, small blue dots (samples). This suggests that the Flow model generates discrete samples from the learned distribution.
>
> In summary, the image visually compares the output of a Score-based model (a smooth, continuous density field) and a Flow model (a dense collection of discrete samples) when attempting to model a specific, curved target distribution (likely the "One-moon2D" distribution mentioned in the context).

<center>Figure F.1: Onemoon2D experiment. The target distribution is shown as a heatmap, and samples from the learned model are overlaid in blue. (a) Samples from the score-base model. For this particular seed, the estimated tempering field is not too far from the true field, resulting in a good fit. (b) Samples from the flow model. </center>

> **Image description.** This is a technical figure consisting of two side-by-side panels, (a) and (b), which visually compare the learned density distributions generated by two different generative modeling techniques: Score-based and Flow. Both panels display two distinct, crescent-shaped distributions, commonly referred to as a "two moons" dataset, against a dark, near-black background.
>
> Panel (a), labeled "Score-based," shows the output of the score-based model. The two crescent shapes are rendered in a smooth, concentrated blue density. The samples appear highly refined and tightly clustered, closely following the intended curved boundaries of the target distribution. The density is uniform and well-defined within the crescent shapes, indicating a high-quality fit and effective sampling.
>
> Panel (b), labeled "Flow," shows the output of the flow model. While the overall shape of the two crescent distributions is maintained, the visual representation is significantly more scattered and noisy compared to Panel (a). The blue density is spread out, and there are numerous small, dispersed blue points and pixels throughout the area, suggesting that the samples generated by the flow model are less concentrated and exhibit higher variance or noise.
>
> In summary, the image visually demonstrates the difference in the quality of the learned distributions, with the score-based method (a) producing smoother, more concentrated, and more accurate samples, while the flow method (b) produces a more scattered and less refined set of samples.

<center>Figure F.2: Twomoons2D experiment. The target distribution is shown as a heatmap, and samples from the learned model are overlaid in blue. (a) Samples from the score-base model. For this particular seed, the estimated tempering field is not too far from the true field, resulting in a good fit. (b) Samples from the flow model. </center>

## F.2 LLM EXPERIMENT

Details of the data generation for the LLM experiment, including the prompts used, are provided in Mikkola et al. (2024, Appendix C.2). We reuse their data and scripts to convert the 5- wise rankings into the pairwise comparisons assumed in our setup: https://github.com/petrus- mikkola/prefflow.

Fig. F.10 completes the partial plot in main text for the LLM experiment. The elicited \(2D\) and \(1D\) marginals have the same support as the true data distribution marginals, and their shapes are also similar, with the distinction that score- based methods tend to generate Gaussian- like marginals. The only exception is the variable \(AveOccup\) , whose marginal appears to have an unreasonably long tail.

Fig. F.11 compares our score- based diffusion method and the flow method of learning the LLM prior from pairwise comparisons in the LLM experiment, highlighting that the score- based method results in smoother estimates than the flow method. Table F.1 summarizes the densities in a quantitative manner by reporting the means for all variables.

> **Image description.** A comparative visualization consisting of two side-by-side panels, labeled (a) and (b), illustrating the output of two different generative models—a score-based model and a flow model—when attempting to reproduce a target ring distribution.
>
> The overall composition features a dark, near-black background in both panels, against which dense clusters of bright blue points are displayed.
>
> **Panel (a): Score-based**
> *   **Label:** The panel is labeled "(a) Score-based" in the bottom left corner.
> *   **Visual Content:** This panel displays a dense collection of bright blue points arranged in a clear, circular ring or torus shape. The points are tightly packed, forming a continuous, well-defined ring structure that suggests the model has successfully learned the target distribution.
>
> **Panel (b): Flow**
> *   **Label:** The panel is labeled "(b) Flow" in the bottom left corner.
> *   **Visual Content:** This panel also displays a dense collection of bright blue points arranged in a circular ring or torus shape. The distribution of points is highly similar to that in Panel (a), forming a complete and well-defined ring.
>
> Both panels visually demonstrate the successful generation of samples that conform to a target ring distribution, comparing the performance of the score-based model versus the flow model. The blue points represent the samples generated by the respective models, while the overall ring shape represents the target distribution.

<center>Figure F.3: Ring2D experiment. The target distribution is shown as a heatmap, and samples from the learned model are overlaid in blue. (a) Samples from the score-base model. (b) Samples from the flow model. </center>

> **Image description.** A complex grid of eight plots arranged in four rows and two columns, illustrating the joint and marginal distributions of four variables (x1, x2, x3, and x4). The plots are used to visualize a multi-dimensional distribution, likely comparing a target distribution with samples from a learned model.
>
> The overall visual style uses a dark background for the plots, with data represented by a dense, dark blue/purple heatmap (representing the target distribution) and a smooth, bright pink curve (representing the marginal distribution or samples from the learned model).
>
> The plots are organized by the variable they represent:
>
> 1.  **Variable x1 (Top Row):**
>     *   The left panel is a 2D joint distribution plot, showing the relationship between x1 and an implied second variable (likely x2, based on the overall structure). It features a dense, dark blue heatmap and a pink curve.
>     *   The right panel is a 1D marginal distribution plot for x1. It displays a smooth pink curve centered around zero, spanning an x-axis range from -5.0 to 5.0.
>
> 2.  **Variable x2 (Second Row):**
>     *   The left panel is a 2D joint distribution plot, showing the relationship between x1 (horizontal axis) and x2 (vertical axis). It contains a dense, dark blue heatmap and a pink curve. Both axes range from -5.0 to 5.0.
>     *   The right panel is a 1D marginal distribution plot for x2. It shows a smooth pink curve centered around zero, spanning an x-axis range from -5.0 to 5.0.
>
> 3.  **Variable x3 (Third Row):**
>     *   The left panel is a 2D joint distribution plot, showing the relationship between x1 (horizontal axis) and x3 (vertical axis). It features a dense, dark blue heatmap and a pink curve. Both axes range from -5.0 to 5.0.
>     *   The right panel is a 1D marginal distribution plot for x3. It displays a smooth pink curve centered around zero, spanning an x-axis range from -5.0 to 5.0.
>
> 4.  **Variable x4 (Bottom Row):**
>     *   The left panel is a 2D joint distribution plot, showing the relationship between x1 (horizontal axis) and x4 (vertical axis). It contains a dense, dark blue heatmap and a pink curve. Both axes range from -5.0 to 5.0.
>     *   The right panel is a 1D marginal distribution plot for x4. It shows a smooth pink curve centered around zero, spanning an x-axis range from -5.0 to 5.0.
>
> In summary, the image systematically presents the joint distributions (2D heatmaps) and the corresponding marginal distributions (1D pink curves) for the variables x1, x2, x3, and x4, all within a consistent coordinate system ranging from -5.0 to 5.0.

<center>Figure F.4: Gaussian4D experiment. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned diffusion model is depicted by greenish blue contour sample points and its marginal by a black curve. </center>

> **Image description.** A technical visualization consisting of a 4x4 grid of density plots, illustrating the marginal and joint distributions of a 4D dataset (Mixturegaussians4D experiment). The grid displays the relationship between four variables, labeled x1, x2, x3, and x4, across both the horizontal and vertical axes.
>
> The overall structure is a matrix where the rows and columns are labeled with the variables x1, x2, x3, and x4. Each of the 16 subplots represents a 2D projection (a marginal or joint distribution) of the original 4D data.
>
> **Visual Elements and Data Representation:**
> Each subplot contains two primary sets of data, representing two different distributions:
> 1.  **Target Distribution:** This is depicted using light blue contour points (or a light blue density field) and a corresponding pink curve, which represents the true marginal distribution.
> 2.  **Learned Diffusion Model:** This is depicted using greenish blue contour sample points (or a greenish blue density field) and a corresponding black curve, which represents the distribution learned by the diffusion model.
>
> **Plot Characteristics:**
> *   **Axes:** All subplots share a consistent coordinate system, with both the horizontal and vertical axes ranging from -4 to 4.
> *   **Diagonal Plots (1D Marginal Distributions):** The plots along the main diagonal (e.g., x1 vs x1, x2 vs x2) show the distribution of a single variable. These plots display multi-modal shapes, where the pink curve (target) and the black curve (learned model) generally align, showing multiple distinct peaks.
> *   **Off-Diagonal Plots (2D Joint Distributions):** The plots off the diagonal (e.g., x1 vs x2, x3 vs x4) show the joint distribution between two variables. These plots display clustered density patterns, indicating the correlation between the variables. In these plots, the light blue/pink target distribution and the greenish blue/black learned distribution show a high degree of visual similarity, suggesting the model successfully captured the underlying structure of the data.
>
> The visualization effectively compares the true underlying data distribution (target) against the distribution generated by the learned model (diffusion model) across all possible 2D projections of the 4D space.

<center>Figure F.5: Mixturegaussians4D experiment. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned diffusion model is depicted by greenish blue contour sample points and its marginal by a black curve. </center>

> **Image description.** This image is a technical figure, Figure F.6, titled "Stargaussian6D experiment," which displays a series of statistical distribution plots across six dimensions ($x_1$ through $x_6$). The figure is organized into six horizontal panels, each representing a specific dimension and containing two distinct types of plots: a 2D joint distribution contour plot and a 1D marginal distribution plot.
>
> **Overall Structure and Layout:**
> The figure consists of six rows, labeled $x_1$ through $x_6$ on the left side of each row. Each row is divided into two visual components:
> 1.  **Left Plot (2D):** A contour plot showing the joint distribution of two variables.
> 2.  **Right Plot (1D):** A marginal distribution plot (density curve/histogram) showing the distribution of a single variable.
>
> **Visual Elements and Data Representation:**
> The plots compare two distributions: the target distribution and the distribution learned by a diffusion model.
>
> *   **Target Distribution:** Represented by light blue contour lines in the 2D plots and a pink curve in the 1D marginal plots.
> *   **Learned Diffusion Model:** Represented by greenish-blue contour lines in the 2D plots and a black curve in the 1D marginal plots.
>
> **Detailed Panel Analysis:**
> Across all six panels ($x_1$ through $x_6$), the visual patterns are highly consistent:
>
> *   **2D Contour Plots (Left side of each panel):** These plots show a bivariate distribution centered near the origin (0, 0). The light blue contours (target) and the greenish-blue contours (learned model) are closely aligned, indicating that the learned model successfully approximates the target distribution in the joint space.
> *   **1D Marginal Plots (Right side of each panel):** These plots display a unimodal distribution, characteristic of a Gaussian or bell curve. The pink curve (target) and the black curve (learned model) are nearly superimposed, demonstrating that the learned model accurately captures the marginal distribution of the variable.
>
> In summary, the figure visually demonstrates the high fidelity of the learned diffusion model, as evidenced by the near-perfect overlap between the target (light blue/pink) and learned (greenish blue/black) distributions across both the 2D joint and 1D marginal views for all six dimensions.

<center>Figure F.6: Stargaussian6D experiment. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned diffusion model is depicted by greenish blue contour sample points and its marginal by a black curve. </center>

> **Image description.** A multi-panel technical figure, Figure F.7, displaying the results of a Stargaussian6D experiment using a flow-based method. The figure is organized into six horizontal rows, labeled $x_1$ through $x_6$, with each row containing two distinct plots: a 2D joint distribution plot on the left and a 1D marginal distribution plot on the right.
>
> The overall visual theme is the comparison between a target distribution and a learned flow model across six different dimensions.
>
> **Structure and Content of Each Row ($x_1$ to $x_6$):**
>
> 1.  **2D Joint Distribution Plots (Left Column):**
>     *   Each plot shows a density distribution centered roughly between 2 and 8 on both axes.
>     *   The axes for these plots range from 0 to 10.
>     *   The target distribution is represented by light blue contour points.
>     *   The learned flow model is represented by greenish-blue contour sample points.
>     *   In all six panels, the greenish-blue points closely match the shape and density of the light blue contours, indicating a high degree of similarity between the target and the learned model in the joint space.
>
> 2.  **1D Marginal Distribution Plots (Right Column):**
>     *   Each plot displays a density curve (histogram) representing the marginal distribution for the corresponding dimension.
>     *   The x-axis ranges from 0 to 10, and the y-axis represents density.
>     *   The target distribution is depicted by a pink curve.
>     *   The learned flow model is depicted by a black curve.
>     *   Across all six dimensions, the black curve (learned flow model) closely follows the shape and peak of the pink curve (target distribution), demonstrating that the model successfully captures the marginal distribution of each dimension.
>
> The figure visually confirms that the baseline flow-based method successfully approximates both the joint and marginal distributions of the target Stargaussian6D distribution across all six dimensions.

<center>Figure F.7: Stargaussian6D experiment when using the baseline flow-based method. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned flow model is depicted by greenish blue contour sample points and its marginal by a black curve. </center>

> **Image description.** A grid of ten two-dimensional scatter plots, labeled x1 through x10, illustrating the results of a "Mixturegaussians10D experiment." Each panel displays a 2D projection of a high-dimensional distribution, comparing a target distribution with a distribution learned by a diffusion model.
>
> The overall layout consists of two rows and five columns, with panels labeled sequentially from x1 (top left) to x10 (bottom right). All panels share a consistent coordinate system, with both the horizontal and vertical axes ranging from -5 to 5, marked with numerical ticks.
>
> Each panel contains two primary types of visual data:
> 1.  **Contour/Sample Points:** These represent the distribution of data points.
>     *   The **target distribution** is depicted using light blue contour points.
>     *   The **learned diffusion model** is depicted using greenish blue contour sample points.
> 2.  **Marginal Distribution:** A smooth curve is overlaid on the point cloud in each panel, representing the marginal distribution.
>     *   The marginal distribution for the **target** is shown as a pink curve.
>     *   The marginal distribution for the **learned model** is shown as a black curve.
>
> The distributions vary significantly in shape across the ten panels. Some panels (e.g., x1, x2) show distributions that are relatively circular or slightly elongated. Others (e.g., x3, x4, x5) exhibit more complex, multi-modal, or highly elongated shapes, demonstrating the diverse nature of the 10D data being analyzed. The visual comparison across the panels allows for the observation of how closely the learned diffusion model (greenish blue points and black curve) approximates the target distribution (light blue points and pink curve) in various dimensions.

<center>Figure F.8: Mixturegaussians10D experiment. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned diffusion model is depicted by greenish blue contour sample points and its marginal by a black curve. </center>

> **Image description.** A technical figure, Figure F.9, consisting of a large grid of 2D scatter and contour plots, illustrating the results of a "Gaussian16D experiment." The figure compares a target distribution with a distribution learned by a diffusion model across multiple dimensions or stages.
>
> The grid is organized into multiple rows and columns, with each cell containing a visualization of a probability distribution. The plots generally show two main types of data:
>
> 1.  **Target Distribution:** This is represented by light blue contour points or density clusters. Its corresponding marginal distribution (a 1D projection) is shown as a pink curve.
> 2.  **Learned Diffusion Model:** This is represented by greenish blue contour sample points or density clusters. Its corresponding marginal distribution is shown as a black curve.
>
> The plots demonstrate a visual progression across the rows, suggesting an evolution or refinement in the model's ability to match the target distribution. In the initial plots, the distributions may be less defined, but as the rows progress, the greenish blue contours increasingly resemble the light blue target contours, indicating improved performance of the learned model.
>
> Many of the plots include a marginal distribution curve on the right side, showing the 1D projection of the 2D data. The labels visible near the plots (e.g., "1", "2", "3", "4", "5", "6") likely correspond to different dimensions or experimental steps within the 16D Gaussian experiment. The overall visual effect is a systematic comparison of how well the learned model captures the characteristics of the target distribution in various dimensions.

<center>Figure F.9: Gaussian16D experiment. The target distribution is depicted by light blue contour points and its marginal by a pink curve. The learned diffusion model is depicted by greenish blue contour sample points and its marginal by a black curve. </center>

> **Image description.** A complex technical visualization consisting of two large, side-by-side scatterplot matrices (pair plots), which are used to visualize the joint and marginal distributions of multiple variables. The image is titled "Figure F.10: Full result plot for the LLM expert elicitation experiment, complementing the partial plot presented in Fig. 3."
>
> The visualization is divided into two distinct panels, both featuring an $8 \times 8$ grid of plots, where the variables are listed along the axes. The variables included in the analysis are: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, and Long.
>
> **Panel 1: Learned LLM prior from pairwise comparisons (Left)**
> This panel displays the joint probability distributions derived from the LLM's expert elicitation.
> *   **Structure:** The diagonal plots show the univariate (marginal) distribution of each variable (e.g., the histogram/KDE for MedInc). The off-diagonal plots show the bivariate (joint) distribution between any two variables.
> *   **Visual Characteristics:** The distributions in this panel exhibit various shapes and correlations. For instance, the joint distribution between MedInc and AveRooms shows a clear positive correlation, while the marginal distributions show complex, often non-Gaussian shapes, indicating multimodal or skewed data.
>
> **Panel 2: California housing dataset (Right)**
> This panel displays the joint probability distributions of the actual California housing dataset, serving as the ground truth for comparison.
> *   **Structure:** It follows the identical $8 \times 8$ matrix structure as the left panel.
> *   **Visual Characteristics:** The distributions here also show correlations between variables (e.g., between AveRooms and AveBedrms). While the overall structure and types of correlations are similar to the LLM prior panel, the specific shapes and densities of the distributions differ, allowing for a visual comparison of the learned prior against the true data distribution.
>
> **Summary of Elements:**
> *   **Type:** Pair Plot / Scatterplot Matrix.
> *   **Variables (8):** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long.
> *   **Layout:** Two $8 \times 8$ grids placed horizontally.
> *   **Function:** To visually compare the learned distribution (LLM prior) against the true distribution (California housing dataset) across all possible bivariate and univariate combinations of the eight variables.

<center>Figure F.10: Full result plot for the LLM expert elicitation experiment, complementing the partial plot presented in Fig. 3. </center>

> **Image description.** This image is a technical figure consisting of two large, side-by-side matrices of joint distribution plots, comparing the results of two different methods for learning an LLM prior.
>
> **Overall Structure and Variables**
> The figure is titled "Figure F.11: Comparison of the results when learning the LLM prior from pairwise comparisons using our score-based diffusion method (left) and the flow based method (right)." Both panels display an $8 \times 8$ grid of 2D density plots, representing the joint distributions of eight variables: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, and Long.
>
> **Left Panel: Learned LLM Prior (Score-based)**
> The left panel is labeled "Learned LLM prior (score-based)." It contains an $8 \times 8$ matrix of plots.
> *   **Axes:** The variables are listed along both the horizontal and vertical axes of the grid.
> *   **Content:** Each plot shows the learned joint distribution of two variables. The distributions vary across the grid, illustrating the learned correlations and marginal distributions between the variables. For example, the plots involving MedInc and AveOccup show distinct, somewhat elongated or clustered distributions.
>
> **Right Panel: Learned LLM Prior (Flow)**
> The right panel is labeled "Learned LLM prior (flow)." It mirrors the structure of the left panel, containing an $8 \times 8$ matrix of joint distribution plots.
> *   **Axes:** The same eight variables (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long) are used for the axes.
> *   **Content:** Similar to the score-based method, this panel displays the joint distributions of variable pairs. Visually, the patterns and shapes of the distributions in the flow-based method appear comparable to those in the score-based method, suggesting a comparison of the quality of the learned priors.
>
> The overall visual presentation is a direct comparison of the learned prior distributions generated by two distinct mathematical approaches: score-based diffusion and flow-based modeling.

<center>Figure F.11: Comparison of the results when learning the LLM prior from pairwise comparisons using our score-based diffusion method (left) and the flow based method (right). </center>

Table F.1: The means of the variables based on (first row) the distribution of the California housing dataset, (second row) the sample from the score-based diffusion model fitted to the LLM's feedback, and (third row) the sample from the flow model.

| | MedInc | HouseAge | AveRooms | AveBedrms | Population | AveOccup | Lat | Long |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| True data | 3.87 | 28.64 | 5.43 | 1.1 | 1425.48 | 3.07 | 35.63 | -119.57 |
| Score-based | 5.89 | 27.56 | 6.66 | 1.53 | 2997.96 | 4.70 | 36.69 | -119.37 |
| Flow | 5.83 | 28.48 | 6.68 | 1.49 | 2948.17 | 3.36 | 36.73 | -119.30 |

## G THE USE OF LARGE LANGUAGE MODELS (LLMS)

The data for Experiment 3 "LLM as a proxy for the expert" in Section 5 was obtained by prompting an LLM (Claude 3 Haiku by Anthropic, March 2024). Further, the first version of the Rosenblatt transformation implementation was developed using an LLM. This version was later improved, and the final version was verified to correctly transform points to the hypercube. The inverse transformation also worked in the tested cases. Finally, an LLM was used to check for writing and content errors in both the text and the code.

# SCORE-BASED DENSITY ESTIMATION FROM PAIRWISE COMPARISONS - Backmatter

---

## ACKNOWLEDGMENTS

The authors acknowledge the research environment provided by ELLIS Institute Finland. The work was supported by the Research Council of Finland Flagship programme: Finnish Center for Artificial Intelligence FCAI, and additionally by the grants 363317 and 358980. The authors acknowledge support from CSC - IT Center for Science, Finland, for computational resources.

Reproducibility statement The source code for reproducing the experiments is available at https://github.com/petrus- mikkola/pairwise2diffusion. Section 4 describes the method and its implementation, while Appendix C details the specific components and training settings required to replicate the experiments. Appendix B provides the full proofs of the theoretical results presented in the main text.

## REFERENCES

Luigi Acerbi. Variational Bayesian Monte Carlo with noisy likelihoods. In Advances in Neural Information Processing Systems, volume 33, pp. 8211- 8222, 2020.

Brian Anderson. Reverse- time diffusion equation models. Stochastic Processes and their Applications, 12(3):313- 326, 1982.

Anthropic. The Claude 3 Model Family: Opus, Sonnet, Haiku. https://www- cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3. pdf, 2024. Model card.

Gordon M Becker, Morris H DeGroot, and Jacob Marschak. Stochastic models of choice behavior. Behavioral science, 8(1):41- 55, 1963.

M Binz and E Schulz. Turning large language models into cognitive models. In Twelfth International Conference on Learning Representations, 2024.

F. Bockting, 
S. 
T. Radev, and 
P. 
C. Burkner. Expert-elicitation method for non-parametric joint priors using normalizing flows. Statistics and Computing, 2025.

Ralph Allan Bradley and Milton Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324- 345, 1952.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert- Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few- shot learners. In Advances in Neural Information Processing Systems, volume 33, pp. 1877- 1901, 2020.

Alexander Capstick, Rahul Krishnan, and Payam Barnaghi. AutoElicit: Using large language models for expert prior elicitation in predictive modelling. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste- Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267, pp. 6746- 6777, 2025.

Ricky TQ Chen. torchdiffeq, 2018. URL https://github.com/rtqichen/torchdiffeq.

Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, volume 31, 2018.

Wei Chu and Zoubin Ghahramani. Preference learning with Gaussian processes. In Proceedings of the 22nd International Conference on Machine learning, pp. 137- 144, 2005.

Stanislas Dehaene. The neural basis of the Weber- Fechner law: a logarithmic mental number line. Trends in cognitive sciences, 7(4):145- 147, 2003.

Vincent Dumoulin, Daniel D. Johnson, Pablo Samuel Castro, Hugo Larochelle, and Yann Dauphin. A density estimation perspective on learning from pairwise human preferences. Transactions on Machine Learning Research, 2024. ISSN 2835- 8856.

Harley Flanders. Differentiation under the integral sign. The American Mathematical Monthly, 80(6): 615- 627, 1973.

Johannes Fürnkranz and Eyke Hüllermeier. Preference learning and ranking by pairwise comparison. In Preference learning, pp. 65- 82. Springer, 2010.

Manuel Gloeckler, Michael Deistler, Christian Dietrich Weilbach, Frank Wood, and Jakob H Macke. All- in- one simulation- based inference. In Proceedings of the 41th International Conference on Machine Learning, pp. 15735- 15766, 2024.

Will Grathwohl, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free- form continuous dynamics for scalable reversible generative models. In International Conference on Learning Representations, 2019.

Danny Halawi, Fred Zhang, Chen Yueh- Han, and Jacob Steinhardt. Approaching human- level forecasting with language models. In Advances in Neural Information Processing Systems, volume 37, pp. 50426- 50468, 2024.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840- 6851, 2020.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion- based generative models. In Advances in Neural Information Processing Systems, volume 35, pp. 26565- 26577, 2022.

Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. In Advances in Neural Information Processing Systems, volume 37, pp. 52996- 53021, 2024a.

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24174- 24184, 2024b.

M. G. Kendall and B. Babington Smith. On the method of paired comparisons. Biometrika, 31(3/4): 324- 345, March 1940.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871, 2018.

Xiang Li, Yixiang Dai, and Qing Qu. Understanding generalizability of diffusion models requires rethinking the hidden Gaussian structure. In Advances in Neural Information Processing Systems, volume 37, pp. 57499- 57538, 2024.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben- Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

Petrus Mikkola, Osvaldo A. Martin, Suyog Chandramouli, Marcelo Hartmann, Oriol Abril Pla, Owen Thomas, Henri Pesonen, Jukka Corander, Aki Vehtari, Samuel Kaski, Paul- Christian Bürkner, and Arto Klami. Prior Knowledge Elicitation: The Past, Present, and Future. Bayesian Analysis, pp. 1- 33, 2023.

Petrus Mikkola, Luigi Acerbi, and Arto Klami. Preferential normalizing flows. In Advances in Neural Information Processing Systems, volume 37, pp. 55667- 55702, 2024.

Kim A Nicoli, Christopher J Anders, Tobias Hartung, Karl Jansen, Pan Kessel, and Shinichi Nakajima. Detecting and mitigating mode- collapse for flow- based sampling of lattice field theories. Physical Review D, 108(11):114501, 2023.

Anthony O'Hagan. Expert knowledge elicitation: Subjective but scientific. The American Statistician, 73:69- 81, 2019.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730- 27744, 2022.

Art B. Owen. Monte Carlo theory, methods and examples. 2013.

R Kelley Pace and Ronald Barry. Sparse spatial autoregressions. Statistics & Probability Letters, 33 (3):291- 297, 1997.

James Requeima, John Bronskill, Dami Choi, Richard Turner, and David K Duvenaud. Llm processes: Numerical predictive distributions conditioned on natural language. In Advances in Neural Information Processing Systems, volume 37, pp. 109609- 109671, 2024.

Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine learning, pp. 1530- 1538, 2015.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High- resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684- 10695, June 2022.

Murray Rosenblatt. Remarks on a multivariate transformation. The annals of mathematical statistics, 23(3):470- 472, 1952.

Anuj K Shah and Daniel M Oppenheimer. Heuristics made easy: an effort- reduction framework. Psychological Bulletin, 134(2):207, 2008.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, 2019.

Yang Song, Jascha Sohl- Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score- based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

Vincent Stimper, Bernhard Scholkopf, and José Miguel Hernández- Lobato. Resampling base distributions of normalizing flows. In International Conference on Artificial Intelligence and Statistics, pp. 4915- 4936, 2022.

Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction. 1998.

L. L. Thurstone. A law of comparative judgment. Psychological Review, 34(4):273-286, 1927.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Kenneth E Train. Discrete choice methods with simulation. Cambridge university press, 2009.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661- 1674, 2011.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228- 8238, 2024.

Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th International Conference on Machine learning, pp. 681–688, 2011.

Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Improved techniques for maximum likelihood estimation for diffusion odes. In International Conference on Machine learning, pp. 42363–42389, 2023.

---

*Transcribed with OCR and VLMs; text, equations, tables, and figure descriptions may contain mistakes.*
