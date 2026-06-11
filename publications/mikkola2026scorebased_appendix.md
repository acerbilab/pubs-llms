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

---

*Transcribed with OCR and VLMs; text, equations, tables, and figure descriptions may contain mistakes.*
