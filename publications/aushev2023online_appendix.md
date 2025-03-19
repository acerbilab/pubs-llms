# Online Simulator-Based Experimental Design for Cognitive Model Selection - Appendix

---

# Supplementary information

The article has the following accompanying supplementary materials:

- Appendix A shows the validity of the approximation of the entropy gain for the design selection rule;
- Appendix B details the algorithm for the proposed BOSMOS method;
- Appendix C contains tables with full experimental results, which shows additional design evaluation points;
- Appendix D showcases a side-by-side comparison of the posterior evolution resulting from BOSMOS and MINEBED for the signal detection task.

---

#### Page 21

# A Approximation of the entropy gain for design selection

Since the expected values in the entropy gain from Equation (3) are not tractable, we rely on a Monte-Carlo estimation of these quantities. We focus on the first term, as the second one only features one of the approximations.
Following the SMC framework [Del Moral et al., 2006], we propose to sequentially update a particle population between trials. This allows the online update of the posterior along the experiment.
At each step we estimate the distribution $p\left(m, \boldsymbol{\theta}_{m} \mid \mathcal{D}_{1: t}\right)$ with a population of $N_{1}$ particles $\left(m^{i}, \boldsymbol{\theta}_{m}^{i}\right)$. Following importance sampling, we know that $\sum_{i=1}^{N_{1}} w_{i} \delta_{\left(m^{i}, \boldsymbol{\theta}_{m}^{i}\right)}$ converges to the distribution associated with density $p\left(m, \boldsymbol{\theta}_{m} \mid\right.$ $\mathcal{D}_{t}$ ). This population of particles then evolves according to the SMC algorithm [Del Moral et al., 2006].
To estimate the expected value in Equation (3), we use a standard Monte-Carlo estimation, making use of the particles at time $t$ as an estimation of the posterior at time $t$. For each particle, we simulate $N_{2}$ vectors $\boldsymbol{x}_{t}^{i}(j) \sim p(\cdot \mid$ $\left.\boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}^{i}, m^{i}\right)$. The convergence to prove is then:

$$
\sum_{i, j}\left(N_{2}\right)^{-1} w_{i} \hat{H}\left(\boldsymbol{x}_{t}^{i_{1}}(j) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right) \rightarrow_{N_{1}, N_{2} \rightarrow \infty} \mathbb{E}_{q\left(m, \boldsymbol{\theta}_{m} \mid \mathcal{D}_{t-1}\right)}\left[H\left(\boldsymbol{x}_{t}^{i} \mid m, \boldsymbol{\theta}_{m}\right)\right]
$$

where $\hat{H}$ is a modified version of the entropy. Note that it is also possible to estimate the gradient of this quantity with respect to the design. If we truly computed $H\left(\boldsymbol{x}_{t}^{i}\right)$ as the entropy with respect to the measure $\sum_{i} \delta_{x_{i}^{i}}$, it would lead to constantly null value. Thus, we decide to use a kernel approximation of the distribution: $\hat{H}\left(\boldsymbol{x}_{t}^{i_{1}}(j) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right)=$ $H\left(\sum_{i} \mathcal{N}\left(\cdot \mid x_{i}^{\prime}(j), \sigma_{N_{2}}\right) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right)$, with $\sigma_{N_{2}} \rightarrow_{N_{2} \rightarrow \infty} 0$.
The convergence of the estimator in Equation 42 to the true entropy requires two results.
First, the convergence in $N_{2}$ :

$$
\sum_{i=1}^{N_{1}} \sum_{j=1}^{N_{2}}\left(N_{2}\right)^{-1} w_{i} \hat{H}\left(\boldsymbol{x}_{t}^{i_{1}}(j) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right) \rightarrow_{N_{2} \rightarrow \infty} \sum_{i=1}^{N_{1}} w_{i} \hat{H}\left(p\left(\cdot \mid \boldsymbol{d}_{t}, \boldsymbol{\theta}_{m}^{i}, m^{i}\right) \mid m^{i}, \boldsymbol{\theta}_{m}^{i}\right)
$$

using the convergence of $\mathcal{N}\left(\cdot \mid x_{i}^{\prime}(j), \sigma_{N_{2}}\right)$ to $\delta_{x_{i}^{\prime}(j)}$ in distribution and the law of large numbers. Second, the convergence for $N_{1} \rightarrow \infty$ comes from the results on SMC [Del Moral et al., 2006].

## B Algorithms

```
Algorithm 1 Bayesian optimization for simulator-based model selection
    Input: prior over models \(p(m)\) and parameters \(p\left(\boldsymbol{\theta}_{m} \mid m\right)_{m} ;\) set of all models \(\mathcal{M}=\left\{p\left(\boldsymbol{x} \mid \boldsymbol{\theta}_{m}, m, d\right)\right\} ;\) design
    budget \(N_{d}\); total number of particles \(N_{q}\);
    Output: selected model \(m^{\prime}\) and its parameters \(\boldsymbol{\theta}_{m}^{\prime}\);
    initialize current beliefs from the priors:
        \(q\left(m, \boldsymbol{\theta}_{m}\right):=\left\{\left(m^{\prime}, \boldsymbol{\theta}_{m}^{i}\right): \boldsymbol{\theta}_{m}^{i} \sim p\left(\boldsymbol{\theta}_{m} \mid m^{\prime}\right), m^{\prime} \sim p(m)\right\}_{i=0}^{N_{d}} ;\)
    initialize an empty set for the collected data: \(\mathcal{D}_{0}=\{ \}\);
    for \(i:=1: N_{d}\) do
        get the design \(\boldsymbol{d}^{\prime}\) with Equation (5);
        collect the data \(\boldsymbol{x}^{\prime}\) at the design location \(\boldsymbol{d}^{\prime}\) and store it in \(\mathcal{D}_{i}\);
        for \(m^{\prime} \in \mathcal{M}\) do
            get the likelihood \(\mathcal{L}_{\epsilon_{m^{\prime}}}\left(\boldsymbol{x}_{i} \mid \boldsymbol{\theta}_{m^{\prime}}\right)\) with Equation (7);
        end for
        get the marginal likelihood \(\mathcal{L}\left(\boldsymbol{x}_{i} \mid m, \mathcal{D}_{i-1}\right)\) with Equation (9);
        update \(q\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{i}\right)\) with Equation (12);
    end for
    apply the decision rule (e.g. MAP):
        \(m^{\prime}=\arg \max _{m} \sum_{\boldsymbol{\theta}_{m}} q\left(\boldsymbol{\theta}_{m}, m \mid \mathcal{D}_{N_{d}}\right)\);
        \(\boldsymbol{\theta}_{m}^{\prime}=\arg \max _{\boldsymbol{\theta}_{m}} q\left(\boldsymbol{\theta}_{m} \mid m, \mathcal{D}_{N_{d}}\right)\)
```

---

#### Page 22

# C Full experimental results

We provide full experimental results data in Tables 2-4, where evaluations of performance metrics were made after different numbers of design iterations. Moreover, we report time costs of running 100 design trials for each method in Table 5, where our method was 80-100 times faster than the other LFI method, MINEBED. The rest of the section discusses three additional minor points with relation to the performance of ADO for the demonstrative example, the bias in the model space for the memory task, and results of testing two decision rules in the risky choice.
As we have seen in the main text, MINEBED had the fastest behavioural fitness convergence rate in the demonstrative example, while BOSMOS was the close second. Hence, ADO had the slowest convergence rate among design optimization methods for behavioural fitness (Table 2) and also for parameter estimation (Table 3). This result is somewhat counterintuitive, as we expected ADO, with its ground-truth likelihood and design optimization, to be the fastest to converge. Since the only other factor influencing this outcome, Bayesian updates, had access to the ground-truth and avoided LFI approximations, suboptimal designs are likely to blame for the poor performance. This problem is likely to be mitigated by expanding the size of the grid used by ADO to calculate the utility objective. However, expanding it would likely increase ADO's convergence at the expense of more calculations; therefore we aimed to get its running time closer to that of BOSMOS, so both methods could be fairly compared.
In the results of model selection for the memory retention task discussed in the main text, MINEBED showed a marginally better average model accuracy than BOSMOS. However, upon closer inspection (Table 4), this accuracy can be solely attributed to the strong bias towards the POW model; the other approaches show it as well, albeit less dramatically. This suggests that the two models in the memory task are separable, but the EXP model cannot likely replicate parts of the behaviour space that the POW model can, resulting in this skewness towards the more flexible model. This is also more broadly related to non-identifiability: since these cognitive models were designed to explain the same target behaviour, it is inevitable that there will be an overlap in their response (or behavioural data) space, complicating model selection.
Since the risky choice model had four models of varied complexity, we experimented with two distinct decisionmaking rules for estimating the models and parameters: the default MAP and Bayesian information criterion (BIC) [Schwarz, 1978, Vrieze, 2012]. Both decision-making rules include a penalty for the size of the model (artificially for BIC, and by definition for MAP). Interestingly, the results are the same for both decision-making rule, indicating that the EU model cannot be completely replaced by a more flexible model. BIC's slightly superior parameter estimates for the BOSMOS technique is most likely explained by the poorer model prediction accuracy. Nevertheless, the BIC rule remains a viable option for model selection in situations when there is a risk of having a more flexible and complex model alongside few-parameter alternatives, despite being less supported theoretically.

## D Posterior evolution examples for BOSMOS and MINEBED

In Figures 5 and 6, we compare posterior distributions returned by MINEBED and BOSMOS for two synthetic participants in the signal detection task. In both examples, the methods have successfully identified the ground-truth POW model, as the majority of the posterior density (shaded area in the figure) has moved to the correct model. Nevertheless, BOSMOS and MINEBED have quite different posteriors, which emphasizes the influence of the design selection strategy on the resulting convergence, as one of the method performs better than the other in each of the provided examples.

---

#### Page 23

> **Image description.** This image is a figure containing a series of 2D density plots showing the evolution of posterior distributions. The figure is arranged in a 4x4 grid of panels.
>
> - **Overall Structure:** The figure is organized into four rows and four columns. Each column represents a different number of trials (1, 4, 20, and 100, from left to right). The first and third rows are labeled "POW", while the second and fourth rows are labeled "PR".
>
> - **Panel Contents:** Each panel contains a 2D density plot. The x-axis is labeled "$\theta_{sens}$", and the y-axis is labeled "$\theta_{hit}$". The y-axis ranges from 0 to 8. The x-axis ranges from 0.0 to 0.9.
>
> - **Density Plots:** The density plots in the first two rows are green, while the density plots in the third and fourth rows are red. The density plots show the distribution of data points, with darker areas indicating higher density. A black "X" is present in the POW plots (rows 1 and 3).
>
> - **Text Labels:** Each column has a title indicating the number of trials: "Trials = 1", "Trials = 4", "Trials = 20", and "Trials = 100". The rows are labeled "POW" and "PR" on the left side, along with "$\theta_{hit}$" and numbers 8, 6, 4, 2, and 0.
>
> - **Observations:** As the number of trials increases from left to right, the density plots in the first two rows (green) become more concentrated. In the third row (red), the density plots also become more concentrated, but the fourth row (red) becomes empty for the higher trial numbers.

Figure 5: The first example of evolution of the posterior distribution resulting from MINEBED (green; rows 1 and 2) and BOSMOS (red; rows 3 and 4) for the signal detection task. For each method, the first row shows the distributions of parameters of the POW model (ground-truth), followed by the PR model parameter distributions in the second row. The axes correspond to the model parameters: sensor-noise (x-axis) and hit value (y-axis); $\theta_{\text {low }}$ and $\theta_{\text {len }}$ of the PR model are omitted to simplify visualization. The last bottom row panels are empty as in both cases the posterior probability of the PR model becomes negligible, so that the particle approximation of this posterior does not contain any more particle.

---

#### Page 24

> **Image description.** This image contains a set of plots arranged in a 4x4 grid, displaying the evolution of posterior distributions.
>
> Each plot in the grid is a 2D contour plot with the x-axis labeled "$\theta_{sens}$" ranging from 0.0 to 0.9, and the y-axis labeled "$\theta_{hit}$" ranging from 0 to 8. The plots are arranged in rows and columns, with the columns representing different numbers of trials (1, 4, 20, and 100).
>
> The first two rows display plots related to the "MINEBED" method, with the first row showing distributions for the "POW" model and the second row for the "PR" model. These plots are colored in shades of green. A black "X" is present in each of the POW model plots. The last two plots in the second row are empty.
>
> The last two rows display plots related to the "BOSMOS" method, with the third row showing distributions for the "POW" model and the fourth row for the "PR" model. These plots are colored in shades of red. A black "X" is present in each of the POW model plots. The last two plots in the fourth row are empty.

Figure 6: The second example of evolution of the posterior distribution resulting from MINEBED (green; rows 1 and 2) and BOSMOS (red; rows 3 and 4) for the signal detection task. For each method, the first row shows the distributions of parameters of the POW model (ground-truth), followed by the PR model parameter distributions in the second row. The axes correspond to the model parameters: sensor-noise (x-axis) and hit value (y-axis); $\theta_{\text {low }}$ and $\theta_{\text {hit }}$ of the PR model are omitted to simplify visualization. The last bottom row panels are empty as in both cases the posterior probability of the PR model becomes negligible, so that the particle approximation of this posterior does not contain any more particle.

---

#### Page 25

|    Methods    |     Tasks: number of design trials     |                 |                 |                 |                 |
| :-----------: | :------------------------------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|               |         Demonstrative example          |                 |                 |                 |                 |
|               |                1 trial                 |    2 trials     |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |            $0.03 \pm 0.03$             | $0.02 \pm 0.02$ | $0.02 \pm 0.02$ | $0.01 \pm 0.01$ |        -        |
|    MINEBED    |            $0.02 \pm 0.07$             | $0.01 \pm 0.00$ | $0.01 \pm 0.00$ | $0.01 \pm 0.00$ |        -        |
|    BOSMOS     |            $0.05 \pm 0.07$             | $0.01 \pm 0.01$ | $0.01 \pm 0.00$ | $0.01 \pm 0.00$ |        -        |
|     LBIRD     |            $0.36 \pm 0.24$             | $0.33 \pm 0.25$ | $0.29 \pm 0.24$ | $0.14 \pm 0.18$ |        -        |
|     Prior     | Baseline for 0 trials: $0.33 \pm 0.3$  |                 |                 |                 |                 |
|               |            Memory retention            |                 |                 |                 |                 |
|               |                1 trial                 |    2 trials     |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |            $0.20 \pm 0.16$             | $0.17 \pm 0.14$ | $0.15 \pm 0.10$ | $0.07 \pm 0.06$ | $0.05 \pm 0.03$ |
|    MINEBED    |            $0.27 \pm 0.22$             | $0.24 \pm 0.19$ | $0.24 \pm 0.19$ | $0.23 \pm 0.15$ | $0.23 \pm 0.17$ |
|    BOSMOS     |            $0.24 \pm 0.19$             | $0.19 \pm 0.16$ | $0.17 \pm 0.14$ | $0.15 \pm 0.13$ | $0.13 \pm 0.11$ |
|     LBIRD     |            $0.20 \pm 0.17$             | $0.17 \pm 0.15$ | $0.14 \pm 0.11$ | $0.08 \pm 0.06$ | $0.05 \pm 0.03$ |
|     Prior     | Baseline for 0 trials: $0.33 \pm 0.47$ |                 |                 |                 |                 |
|               |            Signal detection            |                 |                 |                 |                 |
|               |                1 trial                 |    2 trials     |    4 trials     |    20 trials    |   100 trials    |
|    MINEBED    |            $0.27 \pm 0.24$             | $0.24 \pm 0.20$ | $0.23 \pm 0.17$ | $0.21 \pm 0.18$ | $0.20 \pm 0.17$ |
|    BOSMOS     |            $0.25 \pm 0.21$             | $0.20 \pm 0.17$ | $0.17 \pm 0.15$ | $0.17 \pm 0.15$ | $0.15 \pm 0.12$ |
|     Prior     | Baseline for 0 trials: $0.40 \pm 0.49$ |                 |                 |                 |                 |
|               |              Risky choice              |                 |                 |                 |                 |
|               |                1 trial                 |    2 trials     |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |            $0.32 \pm 0.11$             | $0.30 \pm 0.13$ | $0.27 \pm 0.12$ | $0.14 \pm 0.08$ | $0.07 \pm 0.04$ |
|    MINEBED    |            $0.30 \pm 0.11$             | $0.31 \pm 0.12$ | $0.26 \pm 0.12$ | $0.21 \pm 0.12$ | $0.22 \pm 0.13$ |
|    BOSMOS     |            $0.26 \pm 0.11$             | $0.23 \pm 0.12$ | $0.24 \pm 0.13$ | $0.18 \pm 0.11$ | $0.14 \pm 0.08$ |
| MINEBED (BIC) |            $0.25 \pm 0.11$             | $0.26 \pm 0.13$ | $0.23 \pm 0.11$ | $0.21 \pm 0.12$ | $0.22 \pm 0.13$ |
| BOSMOS (BIC)  |            $0.24 \pm 0.11$             | $0.24 \pm 0.13$ | $0.23 \pm 0.11$ | $0.19 \pm 0.12$ | $0.15 \pm 0.10$ |
|     LBIRD     |            $0.31 \pm 0.12$             | $0.30 \pm 0.13$ | $0.26 \pm 0.13$ | $0.14 \pm 0.07$ | $0.08 \pm 0.03$ |
|     Prior     | Baseline for 0 trials: $0.44 \pm 0.50$ |                 |                 |                 |                 |

Table 2: Convergence of behavioural fitness error $\eta_{\mathrm{b}}$ (mean $\pm$ std. across 100 simulated participants) for comparison methods (rows) with increased number of trials (columns).

---

#### Page 26

|    Methods    | Tasks: number of design trials |                                        |                 |                 |                 |
| :-----------: | :----------------------------: | :------------------------------------: | :-------------: | :-------------: | :-------------: |
|               |                                |         Demonstrative example          |                 |                 |                 |
|               |            1 trial             |                2 trials                |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |        $0.05 \pm 0.06$         |            $0.04 \pm 0.05$             | $0.03 \pm 0.04$ | $0.01 \pm 0.01$ |        -        |
|    MINEBED    |        $0.00 \pm 0.02$         |            $0.00 \pm 0.00$             | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |        -        |
|    BOSMOS     |        $0.05 \pm 0.07$         |            $0.01 \pm 0.02$             | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |        -        |
|     LBIRD     |        $0.34 \pm 0.22$         |            $0.32 \pm 0.24$             | $0.29 \pm 0.25$ | $0.17 \pm 0.18$ |        -        |
|     Prior     |                                | Baseline for 0 trials: $0.33 \pm 0.23$ |                 |                 |                 |
|               |        Memory retention        |                                        |                 |                 |                 |
|               |            1 trial             |                2 trials                |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |        $0.25 \pm 0.21$         |            $0.25 \pm 0.20$             | $0.23 \pm 0.19$ | $0.19 \pm 0.14$ | $0.14 \pm 0.12$ |
|    MINEBED    |        $0.47 \pm 0.38$         |            $0.46 \pm 0.38$             | $0.46 \pm 0.39$ | $0.48 \pm 0.40$ | $0.48 \pm 0.43$ |
|    BOSMOS     |        $0.29 \pm 0.21$         |            $0.27 \pm 0.22$             | $0.28 \pm 0.20$ | $0.27 \pm 0.22$ | $0.29 \pm 0.22$ |
|     LBIRD     |        $0.25 \pm 0.21$         |            $0.25 \pm 0.20$             | $0.22 \pm 0.18$ | $0.22 \pm 0.20$ | $0.15 \pm 0.13$ |
|     Prior     |                                | Baseline for 0 trials: $0.33 \pm 0.20$ |                 |                 |                 |
|               |        Signal detection        |                                        |                 |                 |                 |
|               |            1 trial             |                2 trials                |    4 trials     |    20 trials    |   100 trials    |
|    MINEBED    |        $0.60 \pm 0.24$         |            $0.56 \pm 0.28$             | $0.43 \pm 0.34$ | $0.45 \pm 0.35$ | $0.49 \pm 0.34$ |
|    BOSMOS     |        $0.37 \pm 0.22$         |            $0.35 \pm 0.19$             | $0.36 \pm 0.21$ | $0.35 \pm 0.20$ | $0.35 \pm 0.19$ |
|     Prior     |                                | Baseline for 0 trials: $0.35 \pm 0.20$ |                 |                 |                 |
|               |          Risky choice          |                                        |                 |                 |                 |
|               |            1 trial             |                2 trials                |    4 trials     |    20 trials    |   100 trials    |
|      ADO      |        $0.42 \pm 0.22$         |            $0.44 \pm 0.23$             | $0.43 \pm 0.22$ | $0.33 \pm 0.24$ | $0.26 \pm 0.23$ |
|    MINEBED    |        $0.86 \pm 0.28$         |            $0.87 \pm 0.27$             | $0.81 \pm 0.32$ | $0.76 \pm 0.36$ | $0.86 \pm 0.29$ |
|    BOSMOS     |        $0.41 \pm 0.25$         |            $0.40 \pm 0.21$             | $0.45 \pm 0.26$ | $0.29 \pm 0.25$ | $0.21 \pm 0.23$ |
| MINEBED (BIC) |        $0.85 \pm 0.33$         |            $0.83 \pm 0.31$             | $0.87 \pm 0.30$ | $0.86 \pm 0.31$ | $0.84 \pm 0.35$ |
| BOSMOS (BIC)  |        $0.27 \pm 0.23$         |            $0.26 \pm 0.19$             | $0.40 \pm 0.26$ | $0.23 \pm 0.22$ | $0.21 \pm 0.22$ |
|     LBIRD     |        $0.51 \pm 0.25$         |            $0.48 \pm 0.24$             | $0.40 \pm 0.24$ | $0.35 \pm 0.22$ | $0.24 \pm 0.19$ |
|     Prior     |                                | Baseline for 0 trials: $0.48 \pm 0.25$ |                 |                 |                 |

Table 3: Convergence of parameter estimation error $\eta_{\mathrm{p}}$ (mean $\pm$ std. across 100 simulated participants) when the model is predicted correctly for comparison methods (rows) with increased number of trials (columns).

---

#### Page 27

|    Methods    |        Tasks: number of design trials        |               |               |               |               |
| :-----------: | :------------------------------------------: | :-----------: | :-----------: | :-----------: | :-----------: |
|               |        Demonstrative example (PM, NM)        |               |               |               |               |
|               |                   1 trial                    |   2 trials    |   4 trials    |   20 trials   |  100 trials   |
|      ADO      |                 $0.98,0.96$                  |  $0.98,1.00$  |  $0.98,1.00$  |  $1.00,1.00$  |       -       |
|    MINEBED    |                 $0.93,0.98$                  |  $0.98,1.00$  |  $0.98,1.00$  |  $0.98,1.00$  |       -       |
|    BOSMOS     |                 $0.87,0.83$                  |  $1.00,0.98$  |  $1.00,0.98$  |  $1.00,1.00$  |       -       |
|     LBIRD     |                 $0.51,0.48$                  |  $0.56,0.59$  |  $0.60,0.65$  |  $0.82,0.87$  |       -       |
|     Prior     |       Baseline for 0 trials: $0.5,0.5$       |               |               |               |               |
|               |         Memory retention (POW, EXP)          |               |               |               |               |
|               |                   1 trial                    |   2 trials    |   4 trials    |   20 trials   |  100 trials   |
|      ADO      |                 $0.43,0.89$                  |  $0.61,0.80$  |  $0.74,0.76$  |  $0.91,0.78$  |  $0.96,0.82$  |
|    MINEBED    |                 $0.53,0.68$                  |  $0.60,0.55$  |  $0.58,0.50$  |  $0.96,0.11$  |  $0.93,0.03$  |
|    BOSMOS     |                 $0.30,0.96$                  |  $0.33,0.96$  |  $0.22,0.96$  |  $0.26,0.96$  |  $0.24,0.96$  |
|     LBIRD     |                 $0.37,0.91$                  |  $0.48,0.84$  |  $0.69,0.87$  |  $0.94,0.70$  |  $0.98,0.82$  |
|     Prior     |       Baseline for 0 trials: $0.5,0.5$       |               |               |               |               |
|               |          Signal detection (PPO, PR)          |               |               |               |               |
|               |                   1 trial                    |   2 trials    |   4 trials    |   20 trials   |  100 trials   |
|    MINEBED    |                 $0.02,0.54$                  |  $0.21,0.39$  |  $0.31,0.22$  |  $0.33,0.17$  |  $0.33,0.17$  |
|    BOSMOS     |                 $0.77,0.76$                  |  $0.92,0.52$  |  $0.94,0.46$  |  $0.92,0.43$  |  $0.92,0.43$  |
|     Prior     |       Baseline for 0 trials: $0.5,0.5$       |               |               |               |               |
|               |       Risky choice (EU, WEU; OPT, CPT)       |               |               |               |               |
|               |                   1 trial                    |   2 trials    |   4 trials    |   20 trials   |  100 trials   |
|      ADO      |                $0.26,0.39 ;$                 | $0.11,0.39 ;$ | $0.22,0.64 ;$ | $0.07,0.71 ;$ | $0.11,0.86 ;$ |
|               |                 $0.13,0.19$                  |  $0.35,0.24$  |  $0.35,0.38$  |  $0.52,0.57$  |  $0.70,0.81$  |
|    MINEBED    |                $0.38,0.19 ;$                 | $0.50,0.27 ;$ | $0.58,0.23 ;$ | $0.38,0.38 ;$ | $0.38,0.15 ;$ |
|               |                 $0.12,0.29$                  |  $0.29,0.33$  |  $0.35,0.24$  |  $0.12,0.05$  |  $0.18,0.14$  |
|    BOSMOS     |                $0.48,0.36 ;$                 | $0.48,0.36 ;$ | $0.52,0.50 ;$ | $0.56,0.36 ;$ | $0.48,0.32 ;$ |
|               |                 $0.13,0.05$                  |  $0.09,0.14$  |  $0.13,0.29$  |  $0.09,0.33$  |  $0.09,0.33$  |
| MINEBED (BIC) |                $0.55,0.00 ;$                 | $0.00,0.14 ;$ | $0.55,0.00 ;$ | $0.55,0.00 ;$ | $0.55,0.00 ;$ |
|               |                 $0.00,0.00$                  |  $0.23,0.12$  |  $0.00,0.00$  |  $0.00,0.00$  |  $0.00,0.00$  |
| BOSMOS (BIC)  |                $1.00,0.00 ;$                 | $0.33,0.36 ;$ | $0.52,0.43 ;$ | $0.59,0.29 ;$ | $0.63,0.29 ;$ |
|               |                 $0.00,0.00$                  |  $0.04,0.00$  |  $0.00,0.05$  |  $0.22,0.05$  |  $0.22,0.10$  |
|     LBIRD     |                $0.33,0.43 ;$                 | $0.30,0.43 ;$ | $0.33,0.43 ;$ | $0.33,0.39 ;$ | $0.19,0.50 ;$ |
|               |                 $0.17,0.14$                  |  $0.13,0.29$  |  $0.13,0.38$  |  $0.39,0.29$  |  $0.57,0.67$  |
|     Prior     | Baseline for 0 trials: $0.25,0.25,0.25,0.25$ |               |               |               |               |

Table 4: Convergence of model prediction accuracy $\eta_{\mathrm{m}}$ (proportion of correct predictions of models across 100 simulated participants) for comparison methods (rows) with increased number of trials (columns).

|         | Demonstrative example |   Memory retention   |   Signal detection   |     Risky choice     |
| :-----: | :-------------------: | :------------------: | :------------------: | :------------------: |
|   ADO   |   $10.46 \pm 1.06$    |   $75.44 \pm 9.04$   |          -           |  $134.00 \pm 17.08$  |
| MINEBED |  $786.47 \pm 87.63$   | $3614.11 \pm 272.79$ | $6757.20 \pm 399.90$ | $6698.34 \pm 310.16$ |
| BOSMOS  |    $7.65 \pm 0.39$    |   $35.56 \pm 3.54$   |   $73.41 \pm 8.94$   |   $88.32 \pm 5.75$   |
|  LBIRD  |    $3.16 \pm 0.37$    |   $20.07 \pm 1.32$   |          -           |  $35.95 \pm 12.58$   |

Table 5: Empirical time cost (mean $\pm$ std. in minutes across 100 simulated participants) of applying comparison methods (rows) in cognitive tasks (columns) with 100 sequential designs. ADO and LBIRD for the signal detection task need available likelihoods and therefore cannot be used for this task.
