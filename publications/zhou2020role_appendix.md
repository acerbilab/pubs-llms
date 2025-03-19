# The role of sensory uncertainty in simple contour integration - Appendix

---

#### Page 1

## S1 Appendix <br> Supplemental methods

Yanli Zhou ${ }^{1,2 \# *}$, Luigi Acerbi ${ }^{1,3 \#}$, Wei Ji Ma ${ }^{1,2}$

${ }^{1}$ Center for Neural Science, New York University, New York, New York, USA
${ }^{2}$ Department of Psychology, New York University, New York, New York, USA
${ }^{3}$ Department of Computer Science, University of Helsinki, Helsinki, Finland
\# These authors contributed equally to this work. \* yanlizhou@nyu.edu

- Contact: yanlizhou@nyu.edu

## Contents

1 Analysis of learning ..... 1
2 Model specification ..... 2
2.1 Bayesian model ..... 2
2.2 Fixed-criterion model ..... 3
2.3 Linear heuristic model ..... 3
2.4 Response probabilities ..... 3
2.5 Lapse rate ..... 3
2.6 Height judgment task model ..... 3
3 Model recovery analysis ..... 4
4 Posterior distributions of model parameters ..... 4

## 1 Analysis of learning

We found no main effect of learning across sessions on participants' accuracy in the collinearity task, as shown in S1 Fig (two-way repeated-measures ANOVA with Greenhouse-Geisser correction; $F_{(2.49,17.4)}=$ $0.859, \epsilon=0.828, p=0.462, \eta_{p}^{2}=0.109)$. There is a significant main effect of eccentricity $\left(F_{(2.28,16.0)}=\right.$ $62.4, \epsilon=0.761, p<0.001, \eta_{p}^{2}=0.899$ ), which is expected from the experimental manipulations. We also found no significant interaction between session and eccentricity $\left(F_{(3.44,24.1)}=0.624, \epsilon=0.382, p=0.627\right.$, $\eta_{p}^{2}=0.082$ ). These analyses suggest that participants quickly learnt the task and their performance was stationary across sessions.

---

#### Page 2

> **Image description.** This is a line graph showing accuracy across four sessions for different retinal eccentricity levels.
>
> The graph has the following key features:
>
> - **Axes:** The x-axis is labeled "Session #" and ranges from 1 to 4. The y-axis is labeled "Accuracy" and ranges from 0.5 to 1.0, with tick marks at intervals of 0.1.
>
> - **Data:** There are four lines plotted on the graph, each representing a different retinal eccentricity level, indicated by "y = 0", "y = 4.8", "y = 9.6", and "y = 16.8". The lines are colored in shades of gray, with "y = 0" being black, "y = 4.8" being dark gray, "y = 9.6" being a medium gray, and "y = 16.8" being a light gray. Each data point on the lines has error bars, represented by short vertical lines with horizontal caps, indicating the standard error of the mean (SEM).
>
> - **Legend:** A legend is present in the upper right corner, associating each line color with its corresponding "y" value.
>
> - **Overall Trend:** The black line (y=0) shows the highest accuracy and remains relatively constant across all four sessions. The other lines show lower accuracy, with some fluctuation across sessions.

S1 Fig. Analysis of learning across sessions. Accuracy across four sessions (chance probability $=0.5$ ). Error bars indicate Mean $\pm 1$ SEM across subjects, with retinal eccentricity level plotted as separate lines. We found no significant change in performance across sessions.

# 2 Model specification

### 2.1 Bayesian model

In each trial, an observer utilizes the noisy measurements $x_{L}, x_{R}$ of the true stimuli $y_{L}, y_{R}$ to produce an estimate of the collinearity state $\hat{C}$. Under the Bayesian model, the observer accounts for sensory uncertainty when deciding whether a measured offset between the line segments is due to non-collinearity or to sensory noise by computing the log posterior ratio $d\left(x_{L}, x_{R}\right)$ (Eq S1) of the two competing hypotheses so that the probability of answering correctly is maximized,

$$
d\left(x_{L}, x_{R}\right)=\log \frac{p\left(C=1 \mid x_{L}, x_{R}\right)}{p\left(C=0 \mid x_{L}, x_{R}\right)}=\log \frac{p\left(x_{L}, x_{R} \mid C=1\right)}{p\left(x_{L}, x_{R} \mid C=0\right)}+\log \frac{p(C=1)}{p(C=0)}
$$

Having no direct knowledge of the true vertical positions, the observer marginalizes over $y_{L}$ and $y_{R}$. This gives rise to the following expression for $d\left(x_{L}, x_{R}\right)$,

$$
d\left(x_{L}, x_{R}\right)=\log \frac{\mathcal{N}\left(x_{L} ; x_{R}, 2 \sigma_{x}^{2}\right) \mathcal{N}\left(\frac{x_{L}+x_{R}}{2} ; y, \frac{\sigma_{x}^{2}}{2}+\sigma_{y}^{2}\right)}{\mathcal{N}\left(x_{L} ; y, \sigma_{x}^{2}+\sigma_{y}^{2}\right) \mathcal{N}\left(x_{R} ; y, \sigma_{x}^{2}+\sigma_{y}^{2}\right)}+\log \frac{p(C=1)}{1-p(C=1)}
$$

where $\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ is the probability density of a normal distribution with mean $\mu$ and variance $\sigma^{2}$, and we left out the $y$-dependence of $\sigma_{x}^{2}(y)$ for notational simplicity.

The Bayesian decision rule is to report "collinear" when $d\left(x_{L}, x_{R}\right)>0$,

$$
\hat{C}= \begin{cases}1 & \text { if } d>0 \\ 0 & \text { if } d<0\end{cases}
$$

The boundary equality $d\left(x_{\mathrm{L}}, x_{\mathrm{R}}\right)=0$ defines a curve in $\left(x_{\mathrm{L}}, x_{\mathrm{R}}\right)$-space, which we visualize in Fig 3B of the main text for several values of $\sigma_{x}^{2}$.

---

#### Page 3

# 2.2 Fixed-criterion model

We also tested a non-Bayesian model in which the observer applies a fixed, uncertainty-independent decision boundary $\kappa$ (Fig 3A in the main text). The estimated collinearity state is determined as follows,

$$
\hat{C}= \begin{cases}1 & \text { if }|\Delta x|>\kappa \\ 0 & \text { if }|\Delta x|<\kappa\end{cases}
$$

This is equivalent to using a decision variable $d\left(x_{L}, x_{R}\right)=\left|x_{L}-x_{R}\right|-\kappa$. This model describes the strategy in which the observer uses only the measurements of the line segments, and reports collinearity if the measured offset is within a fixed threshold $\kappa$.

### 2.3 Linear heuristic model

Our third main model is a non-Bayesian probabilistic model in which the decision boundary is represented by a linear function of sensory noise (Fig 3C in the main text),

$$
\hat{C}= \begin{cases}1 & \text { if }|\Delta x|>\kappa_{0}+\kappa_{1} \sigma_{x} \\ 0 & \text { if }|\Delta x|<\kappa_{0}+\kappa_{1} \sigma_{x}\end{cases}
$$

This is equivalent to using a decision variable $d\left(x_{L}, x_{R}\right)=\left|x_{L}-x_{R}\right|-k_{0}-k_{1} \sigma_{x}$. This observer accounts for uncertainty but not in the optimal way.

### 2.4 Response probabilities

The probability of reporting collinearity given stimuli $y_{L}$ and $y_{R}$ and eccentricity level $y$ is then equal to the probability that the measurements fall within the boundary defined by the model $M$,

$$
\begin{aligned}
p_{M}\left(\hat{C}=1 \mid y_{L}, y_{R}, \sigma_{x}^{2}\right) & =\iint p\left(\hat{C}=1 \mid x_{L}, x_{R}, M\right) p\left(x_{L} \mid y_{L}\right) p\left(x_{R} \mid y_{R}\right) d x_{L} d x_{R} \\
& =\iint_{d_{M}\left(x_{L}, x_{R}\right) \geq 0} \mathcal{N}\left(x_{L} ; y_{L}, \sigma_{x}^{2}\right) \mathcal{N}\left(x_{R} ; y_{R}, \sigma_{x}^{2}\right) d x_{L} d x_{R}
\end{aligned}
$$

Eq S6 generally does not have an analytical solution for an arbitrary decision rule $d_{M}\left(x_{L}, x_{R}\right)$, in which case we estimated the response probabilities via numerical integration on a grid.

### 2.5 Lapse rate

For all models, we fitted a lapse rate $\lambda$ for each subject. We define a lapse as a trial in which the subject randomly reports collinearity or non-collinearity, each with a probability of 0.5 .

Adding the lapse rate to Eq S6 we obtain

$$
p_{M, \text { lapse }}\left(\hat{C}=1 \mid y_{L}, y_{R}, \sigma_{x}^{2}\right)=(1-\lambda) \cdot p_{M}\left(\hat{C} \mid y_{L_{i}}, y_{R_{i}}, \sigma_{x}^{2}\right)+\frac{\lambda}{2}
$$

### 2.6 Height judgment task model

During the height judgment task, the observer reports whether the right line segment is higher than the left line segment. The observer's decision $\hat{C}=$ "right higher" or $\hat{C}=$ "left higher" depends only on the sign of the offset between measurements $\Delta x=x_{R}-x_{L}$, which in turn depends on the offset between the

---

#### Page 4

true vertical positions of the stimuli $\Delta y=y_{R}-y_{L}$ at the current eccentricity level. The decision rule is simply to report "right higher" when $\Delta x>0$ and vice versa,

$$
\hat{C}= \begin{cases}\text { "right higher" } & \text { if } \Delta x>0 \\ \text { "left higher" } & \text { if } \Delta x<0\end{cases}
$$

The response probability is therefore

$$
\begin{aligned}
p(\hat{C}= & \text { "right higher" } \mid \Delta y)=\int p(\hat{C}= \text { "right higher" } \mid \Delta x) p(\Delta x \mid \Delta y) d \Delta x \\
& =\int \chi(\Delta x>0) \mathcal{N}\left(\Delta x ; \Delta y, 2 \sigma_{x}^{2}\right) d \Delta x
\end{aligned}
$$

where $\chi(\cdot)$ is the indicator function which is equal to 1 when the argument is true, 0 otherwise.

# 3 Model recovery analysis

To ensure our models are distinguishable, and as a further validation of our model fitting pipeline, we performed a model recovery analysis which verifies that the fitted models are matched to the datagenerating model for simulated data. In this analysis, we considered our three main models: Fixed, Bayes and Lin.

We first generated 50 synthetic datasets from each of these three models using parameter vectors randomly drawn from the posterior samples of a randomly drawn subject obtained through MCMC sampling, so as to represent 'typical' observers. Following this procedure, we generated 150 datasets in total ( 3 generating models $\times 50$ simulated observers). We then fitted all models to each generated dataset via maximum likelihood estimation for computational tractability. Here we used the Akaike Information Criterion (AIC) as the metric for model comparison, and computed the fraction of times that a model wins the model comparison for a given generating model (S2 Fig). We did not use LOO scores for model comparison due to the impracticality of performing MCMC on all synthetic datasets and models, as we found AIC scores and LOO scores to be highly consistent for our models and datasets (see main text).

On average, $90.0 \%$ of the 150 datasets were successfully recovered. This result suggests that our main models are distinguishable, and also validates our model fitting pipeline.

## 4 Posterior distributions of model parameters

In our analyses, we first performed maximum-likelihood estimation for every model and every subject. We then followed up with MCMC sampling, using the MLE solutions as starting points for the MCMC chains. This procedure enabled us to not only obtain point estimates of model parameters for each subject but also samples from the posterior landscapes of all model parameters (examples shown below).

---

#### Page 5

> **Image description.** This image is a matrix representing model recovery analysis. The matrix is a 3x3 grid, with each cell containing a numerical value. The rows are labeled "Generating models" and the columns are labeled "Fitted models." The models are "Fixed," "Bayes," and "Lin."
>
> - The rows, from top to bottom, represent the generating models: Fixed, Bayes, and Lin.
> - The columns, from left to right, represent the fitted models: Fixed, Bayes, and Lin.
> - The cell at the intersection of "Fixed" (generating) and "Fixed" (fitted) contains the value 0.82 and is colored light gray.
> - The cell at the intersection of "Fixed" (generating) and "Bayes" (fitted) is black.
> - The cell at the intersection of "Fixed" (generating) and "Lin" (fitted) contains the value 0.08 and is colored dark gray.
> - The cell at the intersection of "Bayes" (generating) and "Fixed" (fitted) is black.
> - The cell at the intersection of "Bayes" (generating) and "Bayes" (fitted) contains the value 1.00 and is white.
> - The cell at the intersection of "Bayes" (generating) and "Lin" (fitted) contains the value 0.04 and is black.
> - The cell at the intersection of "Lin" (generating) and "Fixed" (fitted) contains the value 0.18 and is dark gray.
> - The cell at the intersection of "Lin" (generating) and "Bayes" (fitted) is black.
> - The cell at the intersection of "Lin" (generating) and "Lin" (fitted) contains the value 0.88 and is colored light gray.
>
> The color intensity of the cells varies, with white representing the highest value (1.00) and black representing the lowest values. The values in the cells represent the fraction of datasets that were best fitted from a model (columns), for a given generating model (rows).

S2 Fig. Model recovery analysis. Each square represents the fraction of datasets that were best fitted from a model (columns), for a given generating model (rows), according to the AIC score. The light shade of the diagonal squares indicates that the true generating model was, on average, the best-fitting model in all cases, leading to a successful model recovery.

> **Image description.** The image consists of three triangular grids of plots, each representing the posterior distribution landscapes for different models: "Fixed", "Bayes", and "Lin" from left to right. Each grid is an upper triangular matrix of plots.
>
> - **Overall Structure:** Each grid shows 1-D marginal posterior distributions on the diagonal and 2-D marginal posteriors for every parameter pair below the diagonal.
>
> - **Diagonal Plots:** The diagonal plots in each grid are density plots (histograms or kernel density estimates) showing the marginal posterior distribution of a single parameter. These plots are gray lines.
>
> - **Off-Diagonal Plots:** The off-diagonal plots display 2-D marginal posterior distributions for each parameter pair. These are represented as contour plots with multiple contour lines. The contours are blue, and the central area within the contours is filled with a yellow-green color, indicating areas of higher probability density.
>
> - **Model "Fixed":** The "Fixed" model grid displays the parameters σ1, σ2, σ3, σ4, κ (kappa), and λ (lambda). The x and y axes of the off-diagonal plots are labeled with these parameter symbols. The values on the x and y axes are visible.
>
> - **Model "Bayes":** The "Bayes" model grid displays the parameters σ1, σ2, σ3, σ4, p(C=1), and λ. The x and y axes of the off-diagonal plots are labeled with these parameter symbols. The values on the x and y axes are visible.
>
> - **Model "Lin":** The "Lin" model grid displays the parameters σ2, σ3, σ4, κ1, κ0, and λ. The x and y axes of the off-diagonal plots are labeled with these parameter symbols. The values on the x and y axes are visible.

S3 Fig. Posterior distribution landscapes. From left to right, the figure shows the posterior distributions of the model parameters obtained via MCMC for the three main models (Bayes, Fixed, and Lin, respectively), for a representative subject. In each panel, the plots represent the 1-D marginal posterior distributions of each parameter (on the diagonal), and 2-D marginal posteriors for every parameter pair (below the diagonal).
