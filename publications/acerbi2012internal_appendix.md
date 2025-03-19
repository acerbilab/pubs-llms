# Internal Representations of Temporal Statistics and Feedback Calibrate Motor-Sensory Interval Timing - Appendix

---

## Supporting Information

Table S1 Bayesian model comparison: most supported observer model components for Experiments 1-4. Most supported observer model components (posterior probability), for each subject, according to the Bayesian model comparison.

Text S1 Additional models and analyses. This supporting text includes sections on: computation of response bias and standard deviation of the response for the basic Bayesian observer model; a Bayesian observer model with non-quadratic loss function; a Bayesian observer model with lapse; an extended analysis of subjects' sensory and motor variability. Figures S1, S2, S3, S4 are included.

Text S2 Results of Experiments 3 and 4. Plots of mean response bias and standard deviation of the response for Experiment 3 and 4. Figures S5 and S6 are included.

---

#### Page 1

|   Subject    | Sensory likelihood | Motor likelihood | Subjective prior | Loss function |
| :----------: | :----------------: | :--------------: | :--------------: | :-----------: |
| Experiment 1 |                    |                  |                  |               |
|      LA      |     Sc (1.000)     |    Sc (1.000)    |    b (1.000)     |  Sk (1.000)   |
|      JW      |     Sc (0.967)     |    Sc (1.000)    |    b (0.960)     |  St (1.000)   |
|      TL      |     Cn (1.000)     |    Sc (1.000)    |    b (1.000)     |  Sk (1.000)   |
|      DB      |     Sc (1.000)     |    Sc (0.974)    |    e (0.997)     |  St (1.000)   |
| Experiment 2 |                    |                  |                  |               |
|      LA      |     Sc (1.000)     |    Sc (0.997)    |    g (1.000)     |  Sk (1.000)   |
|      AC      |     Cn (1.000)     |    Sc (1.000)    |    f (0.978)     |  St (1.000)   |
|      AP      |     Cn (1.000)     |    Sc (0.981)    |    b (1.000)     |  Sk (1.000)   |
|      HH      |     Sc (0.997)     |    Sc (0.997)    |    g (0.998)     |  Sk (0.875)   |
|      JB      |     Sc (0.998)     |    Cn (0.996)    |    f (0.997)     |  Sk (1.000)   |
|      TZ      |     Sc (1.000)     |    Cn (1.000)    |    d (0.976)     |  Sk (1.000)   |
| Experiment 3 |                    |                  |                  |               |
|      LA      |     Cn (0.910)     |    Sc (0.990)    |    b (1.000)     |  St (0.993)   |
|      NY      |     Sc (0.988)     |    Sc (0.780)    |    b (1.000)     |  Fr (1.000)   |
|      JL      |     Sc (0.528)     |    Sc (1.000)    |    b (0.999)     |  St (1.000)   |
|      RD      |     Cn (1.000)     |    Sc (0.996)    |    b (0.998)     |  Fr (1.000)   |
|      PD      |     Sc (0.758)     |    Cn (1.000)    |    b (0.999)     |  St (1.000)   |
|      JE      |     Cn (0.896)     |    Sc (0.912)    |    b (1.000)     |  St (1.000)   |
| Experiment 4 |                    |                  |                  |               |
|      RR      |     Cn (0.986)     |    Sc (0.950)    |    a (0.998)     |   St $(-)$    |
|      DD      |     Cn (0.726)     |    Cn (0.641)    |    f (0.511)     |   St $(-)$    |
|              |                    |                  |    g (0.486)     |               |
|      NG      |     Cn (0.980)     |    Sc (0.973)    |    b (0.503)     |   St $(-)$    |
|              |                    |                  |    g (0.458)     |               |

Table S1. Bayesian model comparison: most supported observer model components for Experiments 1-4. Most supported observer model components (posterior probability), for each subject, according to the Bayesian model comparison. A posterior probability $p>0.95$ should be considered suggestive evidence, and $p>0.99$ significant (posterior probability $p>0.9995$ is written as 1.000 , with a slight abuse of notation). The sensory and motor likelihoods can either be constant ( Cn ) or scalar (Sc); the subjective priors (a-g) are described in the Methods section (see main text); the loss function can be Skewed (Sk), Standard (St) or Fractional (Fr) (see also Figure 6 in main text). Note the switch in preferred loss function from Experiments 1 and 2 (which received Skewed feedback) to Experiment 3 (which received Standard feedback). In Experiment 4 the loss function was fixed to Standard to constrain the model selection.

---

#### Page 1

# Supporting Text S1 - Additional models and analyses

## Contents

1 Computation of response bias and standard deviation of the response ..... 2
2 Non-quadratic loss function ..... 4
3 Bayesian observer model with lapse ..... 4
4 Sensory and motor variability ..... 5
4.1 Measuring sensory and motor noise ..... 6
4.2 Internal knowledge of estimation variability ..... 7
4.3 Generalized law for motor noise ..... 8

---

#### Page 2

# 1 Computation of response bias and standard deviation of the response

The key equations of our model are Eqs. 1 (optimal action $u^{*}(y)$ for internal measurement $y$ ) and 2 (probability of response $r$ given the true time interval $x$ ); see main text. In particular, they allow us to compute the response bias and standard deviation (sd) which are shown in the plots.

As intermediate calculations, the mean response for interval $x$ is

$$
\mathbb{E}[r]_{p(r \mid x)}=\int p(r \mid x) r d r=\int\left[\int p_{s}(y \mid x) p_{m}\left(r \mid u^{*}(y)\right) d y\right] r d r=\int p_{s}(y \mid x) u^{*}(y) d y
$$

and, analogously, the the second moment of the response reads

$$
\begin{aligned}
\mathbb{E}\left[r^{2}\right]_{p(r \mid x)} & =\int p(r \mid x) r^{2} d r=\int\left[\int p_{s}(y \mid x) p_{m}\left(r \mid u^{*}(y)\right) d y\right] r^{2} d r \\
& =\int p_{s}(y \mid x)\left[u^{*}(y)^{2}+\sigma_{m}^{2}\left(u^{*}(y)\right)\right] d y
\end{aligned}
$$

In the above derivations we have used the fact that the motor likelihood $p_{m}(r \mid u)$ is modelled as a Gaussian with mean $u$ and variance $\sigma_{m}^{2}(u)$ (the specific function for the variance depends on the observer model; see Methods).

From Eqs. S1 and S2 we can compute

$$
\text { Reponse bias }(x)=\mathbb{E}[r]_{p(r \mid x)}-x, \quad \text { Response } \operatorname{sd}(x)=\sqrt{\mathbb{E}\left[r^{2}\right]_{p(r \mid x)}-\left(\mathbb{E}[r]_{p(r \mid x)}\right)^{2}}
$$

Note that the optimal action $u^{*}(y)$ is a key element of all equations. In particular, the mean response in Eq. S1 is obtained by convolving the optimal action with the sensory likelihood. In other words, plots of the mean response are smoothed versions of plots of the optimal action; the same relationship holds for the response bias and shifted optimal action $u^{*}(y)-y$ (see Figure S1).

---

#### Page 3

> **Image description.** This image consists of eight line graphs arranged in a 2x4 grid. The top row of graphs is labeled "Response bias (ms)" on the y-axis and "Time interval (ms)" on the x-axis. The bottom row of graphs is labeled "Shifted optimal action (ms)" on the y-axis and "Internal measurement (ms)" on the x-axis.
>
> Each column of graphs is labeled with a letter: "a", "b", "c", and "d" at the top.
>
> Each graph has a horizontal line at y=0. The x-axis ranges from approximately 550 to 1050, with tick marks at 600, 750, 900, and 1050. The y-axis ranges from -60 to 60, with tick marks at -60, -40, -20, 0, 20, 40, and 60.
>
> The graphs in columns "a" and "b" show a line that decreases linearly. The graphs in columns "c" and "d" show a non-linear curve that decreases. Each line/curve has four colored dots along it: green, light green, gray, and magenta, from left to right. The dots appear to mark specific data points along the line.

Figure S1. Comparison of response bias and (shifted) optimal action. Response bias and (shifted) optimal action for four different ideal observers (columns a-d) are shown (see Figure 1 in main text). Top: Response bias for the example observer models taken from Figure 1 in the paper. Bottom: Shifted optimal action $u^{*}(y)-y$ for the same models. For ease of comparison, different colored dots mark a discrete set of interval durations. Note the similarity between the two rows; the mean response is in fact obtained by convolving the optimal action with the sensory noise (Eq. S1).

---

#### Page 4

# 2 Non-quadratic loss function

Our basic model assumed a quadratic (or pseudo-quadratic) loss function that was obtained by squaring the subjective error map $\tilde{f}(x, r)$ (Eq. 1 in main text). The exponent 2 allowed a semi-analytical solution of Eq. 1, which made tractable the problems of (a) computing the marginal likelihood for a relatively large class of models and (b) nonparametrically inferring the subjects' priors (see main text). However, previous work has shown that people in sensorimotor tasks may be instead following a sub-quadratic loss function [1].

For the sake of completeness, we explored an extended model with non-quadratic loss functions. For computational reasons we could not perform a full Bayesian model comparison, but we considered only the 'best' observer model per subject. We used datasets from Experiments 1 and 2 as they comprised two distinct blocks per subject, which provided more data points and reduced risk of overfitting. For each subject we chose the most supported model components for the sensory and motor likelihood and the shape of the subjective error mapping (Standard, Skewed or Fractional), whereas for the prior we took the nonparametrically inferred priors. However, the exponent of the loss function was now free to vary, so that the equation for the optimal action reads

$$
u^{*}(y)=\arg \min _{u} \int p_{s}\left(y \mid x ; w_{s}\right) q(x) p_{m}\left(r \mid u ; w_{m}\right)\left|\tilde{f}(x, r)\right|^{\kappa} d x d r
$$

where $\kappa>0$ is a continuous free parameter representing the exponent of the loss function. Eq. S4 was solved numerically (function fminbnd and trapz in MATLAB) for various values of $y$ and then interpolated. Through Eqs. 2 and 7 we computed for each subject the posterior probability of the exponent $\operatorname{Pr}(\kappa \mid$ data $) \propto \operatorname{Pr}($ data $ \mid \kappa) \operatorname{Pr}(\kappa)$, where we assumed an (improper) uniform prior on $\kappa$.

Results are shown in Figure S2 as a box plot for each subject's inferred $\kappa$. Taking the median of the posterior distribution as the inferred value for $\kappa$, the exponent averaged across subjects (excluding one outlier) is $1.88 \pm 0.06$ which is marginally lower than 2 (one-sample t-test $p<0.07$ ). (Taking the mean of the posterior instead of the median renders analogous results.) This result is in qualitative agreement with [1] which found that subjects were following a sub-quadratic loss function (with exponent $1.72 \pm 0.03$ for a power law). Our average inferred exponent is however higher, and only marginally lower than 2, but this might be due to the fact that the subjects' priors have been inferred under the assumption of a quadratic loss function, and therefore priors may be already 'fitting' some features of the data that were due instead to a sub-quadratic loss function. The structure of the model does not currently allow for a simultaneous inference of both nonparametric priors and exponent of the loss function computationally, which is an open problem for future work.

## 3 Bayesian observer model with lapse

We extended the Bayesian observer model described in the paper (Eqs. 1 and 2) by introducing for each subject in Experiments 1 and 2 a third continuous parameter, the probability of lapse $\lambda$. For each trial, the observer has some probability $\lambda$ of ignoring the current stimulus and responding with uniform probability over the range of allowed responses - a very simple model of data outliers due to subjects' errors. The response probability with lapse reads

$$
p_{\text {lapse }}\left(r \mid x ; w_{s}, w_{m}, \lambda\right)=\lambda \frac{1}{L}+(1-\lambda) p\left(r \mid x ; w_{s}, w_{m}\right)
$$

where $L$ is the allowed response window duration (which is block-dependent, see Data Analysis). By using Eq. S5 in Eq. 7 (see Methods) we computed the marginal likelihood of models with lapse, extracted the most supported model components and hence inferred the subjective priors.

---

#### Page 5

> **Image description.** The image is a box plot comparing the inferred exponents of a loss function for multiple subjects.
>
> - **Axes:** The x-axis is labeled "Subjects" and ranges from 1 to 10. The y-axis is labeled "Loss function exponent" and ranges from 0 to 4, with increments of 0.5.
> - **Data Representation:** The data is displayed as box plots. Each box plot represents the distribution of inferred exponents for a single subject. The boxes show the interquartile range (IQR), with a line indicating the median. Whiskers extend from the boxes, representing the range of the data, excluding outliers.
> - **Horizontal Line:** A solid black horizontal line is drawn at the y-value of 2.
> - **Box Plot Details:** Each box plot has a different vertical position, indicating the range of inferred exponents for each subject. Subject 3 has a noticeably higher box plot compared to the rest.

Figure S2. Non-quadratic loss function. Inferred exponents of the loss function for subjects in Experiment 1 (ss $1-4$ ) and 2 (ss $5-10$ ). The box plots have lines at the lower quartile, median, and upper quartile values; whiskers cover $95 \%$ confidence interval. Excluding one outlier (s 3), the average inferred exponent is marginally lower than $2(p<0.07)$.

The average moments of the reconstructed priors did not differ significantly from the ones computed with the basic model without lapse (see Table 2), and in particular the kurtosis was similar, being in general systematically higher than the true distribution kurtosis. The excess kurtosis for the observers with lapse, computed by averaging the moments of sampled priors pooled from all subjects, was (mean $\pm 1$ s.d.): $0.85 \pm 1.30$ (Short Uniform), $0.70 \pm 1.01$ (Long Uniform); $0.91 \pm 1.57$ (Medium Uniform), $1.87 \pm 1.84$ (Medium Peaked); as opposed to a true excess kurtosis of -1.27 (Uniform blocks) and 0.09 (Peaked block).

# 4 Sensory and motor variability

The sensory (estimation) and motor (reproduction) likelihoods in our observer's model were represented by normal distributions whose standard deviation (either constant or 'scalar', Figure 6 i and ii, see paper) was governed by the two parameters $w_{s}, w_{m}$, respectively for the sensory and motor component. We describe here a set of additional experiments and analyses which tested various hypotheses about our subjects' sensorimotor likelihoods.

First of all, we examined whether the parameter values $w_{s}, w_{m}$ inferred from the data corresponded to direct measures of sensory and motor variability gathered in different tasks. We found a good agreement at the group level for both parameters and a good correlation for the individual values of the sensory noise (see 'Measuring sensory and motor noise').

With an additional model comparison, we checked whether, according to our data, subjects' 'knew' their own sensory (estimation) variability; that is, we examined whether their internal estimate of their sensory variability matched their objective sensory variability (both quantities were computed from the model). The analysis suggests that subjects were generally 'aware' of their own sensory variability (refer

---

#### Page 6

to subsection below on 'Internal knowledge of estimation variability'). We did not perform an analogous study on the motor variability as the problem becomes under-constrained (see below).

At last, to see whether we could better understand the form of the motor noise we analyzed our data with a 'generalized' model with 2 parameters governing the growth of the standard deviation of the motor noise. Interestingly, the generalized model did not perform better in terms of model comparison than the 1-parameter scalar model (refer to subsection below on 'Generalized law for motor noise').

# 4.1 Measuring sensory and motor noise

For each subject in Experiments 1 and 2 we computed the posterior distribution of $w_{s}, w_{m}$ (weighted average over all models) and took the mean of the posterior as the 'model-inferred' sensory and motor variability. We examined whether the model-inferred values corresponded to direct measures of sensory and motor variability ( $w_{s}^{\prime}, w_{m}^{\prime}$ ) obtained through additional experiments. We directly measured each subject's sensory variability $w_{s}^{\prime}$ in a two-alternative forced choice time interval discrimination task, and analogously we directly measured the subjects' motor variability $w_{m}^{\prime}$ in a time interval 'production' task (see below, Methods, for details).

The comparison between the model-inferred values and the directly-measured ones is shown in Figure S3 for the sensory (left) and motor (right) noise parameters. For sensory variability, we found that $w_{s}^{\prime}$ had a good correlation $\left(R^{2}=0.77\right)$ with $w_{s}$, and the group means were in good agreement $\left(\overline{w_{s}}=0.157 \pm 0.002\right.$, $\left.\overline{w_{s}^{\prime}}=0.166 \pm 0.009\right)$. For the motor variability, the group means were quantitatively similar, even though in slight statistical disagreement $\left(\overline{w_{m}}=0.072 \pm 0.001, \overline{w_{m}^{\prime}}=0.078 \pm 0.001\right)$, but we did not find a correlation between $w_{m}^{\prime}$ and $w_{m}$ (see Discussion). These results suggest that the model parameters for the 'noise properties' extracted from the full model were in agreement with independent measures of these noise properties in isolation. Interestingly, independent measurements of the sensory noise had predictive power on the subjects' performance even at the individual level (data not shown), due to the good correlation with the sensory model parameter.

The lack of correlation for the motor noise parameter at the individual level may have been due to other noise factors, not contemplated in the model, that influenced the variance of the produced response (e.g. noise in the decision making process, non-Gaussian likelihoods, deviations from the exact scalar property, etc.).

## Methods

Each participant of Experiments 1 and 2 took part in a side sensory and motor measurement session. In these sessions all stimuli and materials were identical to the ones presented in the main experiment (see Methods in main text); the design of these experiments itself was chosen to be as similar as possible to the main experiment, but focussing only on the sensory (estimation) or motor (reproduction) part of the task.

In the sensory noise measurement session, $\sim 320$ trials, in each trial subjects clicked on a mouse button and a dot flashed on screen after a given duration ( $x_{1} \mathrm{~ms}$ ). Subjects clicked again on the mouse button, and a second dot flashed on screen after $x_{2} \mathrm{~ms}$. At the end of each trial subjects had to specify which interval was longer through a two-alternative forced choice. Correct responses received a tone as positive feedback. Intervals $x_{1}$ and $x_{2}$ were adaptively chosen from the range $300-1275 \mathrm{~ms}$ on a trial by trial basis in order to approximately maximize the expected gain in information about the sensory variability of the subject (we adapted the algorithm described in [2]).

In the motor noise measurement session, each trial subjects had to reproduce a given block-dependent interval by holding the mouse button. Subjects received visual feedback of their performance through the Skewed error mapping (as in Experiments 1 and 2). For each block the target interval was always the same ( 500,750 or 1000 ms ) and the subjects were instructed about it. Subjects performed on the

---

#### Page 7

three intervals twice, in a randomized order, for a total of six blocks ( 30 trials per block, the first five trials were discarded).

For each subject we built simple ideal observer models of the interval discrimination and interval reproduction tasks in which the sensory and motor variability could either be constant or scalar (according to the results of the model comparison in the main experiment). We computed the posterior distributions of the sensory and motor noise parameters, and took the mean of the posterior as the 'directly-measured' noise parameters $\left(w_{s}^{\prime}, w_{m}^{\prime}\right)$.

> **Image description.** This image consists of two scatter plots side-by-side, comparing sensory and motor noise parameters.
>
> - **Left Panel: Sensory Variability Comparison**
>
>   - Title: "Sensory variability comparison"
>   - X-axis: "Model average for w_s" ranging from 0.0 to 0.3.
>   - Y-axis: "Measured w'\_s" ranging from 0.0 to 0.3.
>   - Data points: Multiple circular points, each with error bars extending vertically and horizontally.
>   - Group mean: A cross mark is present, surrounded by a shaded area, presumably indicating a confidence interval.
>   - Line: A solid blue line shows a linear fit to the data.
>   - Diagonal line: A dashed black line represents the identity line (y=x).
>   - R-squared value: "R^2 = 0.77" is displayed in the lower right corner.
>
> - **Right Panel: Motor Variability Comparison**
>   - Title: "Motor variability comparison"
>   - X-axis: "Model average for w_m" ranging from 0.04 to 0.1.
>   - Y-axis: "Measured w'\_m" ranging from 0.04 to 0.1.
>   - Data points: Multiple circular points, each with error bars extending vertically and horizontally.
>   - Group mean: A cross mark is present, surrounded by a shaded area, presumably indicating a confidence interval.
>   - Diagonal line: A dashed black line represents the identity line (y=x).
>   - R-squared value: "R^2 = 0.03" is displayed in the lower right corner.
>
> The plots compare model-averaged noise parameters with directly measured noise parameters for sensory and motor variability, respectively. The R-squared values indicate the goodness of fit for the linear relationship between the two sets of parameters.

Figure S3. Comparison of sensory and motor noise parameters (main experiment vs direct measurements). For each participant of the main experiments (Experiment 1 and $2, n=10$ ) we independently measured the sensory $\left(w_{s}^{\prime}\right)$ and motor $\left(w_{m}^{\prime}\right)$ variabilities in a time-interval discrimination session and a time interval reproduction session with performance feedback (see text for details). We built simple probabilistic models for the above tasks and computed the posterior mean and standard deviation for $w_{s}^{\prime}$ and $w_{m}^{\prime}$. For each subject we also calculated the posterior mean and standard deviation for the parameters $w_{s}$ and $w_{m}$ that appear in our Bayesian ideal observer model, averaged over all models (weighted by the model posterior probability). Ideally, the couples of parameters $\left(w_{s}, w_{s}^{\prime}\right)$ and $\left(w_{m}, w_{m}^{\prime}\right)$ reflected the same objective features of the subjects measured in distinct, indepedent tasks. The parameters are compared in the figure, $\left(w_{s}, w_{s}^{\prime}\right)$ to the left and $\left(w_{m}, w_{m}^{\prime}\right)$ to the right, each circle is a participant's parameters mean $\pm 1$ s.d. We also plotted the group mean (crosses, shaded area $95 \%$ confidence interval). The group means are $\bar{w}_{s}=0.157 \pm 0.002, \bar{w}_{s}^{\prime}=0.166 \pm 0.009 ; \bar{w}_{m}=0.072 \pm 0.001$, $\overline{w_{m}^{\prime}}=0.078 \pm 0.001$.

# 4.2 Internal knowledge of estimation variability

Our modelling framework allowed us to ask whether subjects 'knew' their own sensory (estimation) variability in the task [3-5]. We extended our original model by introducing a distinction between the objective sensory variability $w_{s}$ and the subjective estimate the Bayesian observer had of its value, $\widetilde{w}_{s}$. The computation of the optimal action was modified accordingly,

$$
u^{*}(y)=\arg \min _{u} \int p_{s}\left(y \mid x ; \widetilde{w}_{s}\right) q(x) p_{m}\left(r \mid u ; w_{m}\right) \widetilde{f}^{2}(x, r) d x d r
$$

which is identical to Eq. 1 but note that the expected loss depends now on the subjective value $\widetilde{w}_{s}$ instead of $w_{s}$. The other equations of the model remained unchanged as they depend on the objective sensory noise.

---

#### Page 8

We performed a full Bayesian model comparison with the extended model, where all components (likelihoods, prior, loss function) were free to vary as per the basic model comparison (see paper); the only difference being the presence of three continuous parameters $\left(w_{s}, w_{m}, \widetilde{w}_{s}\right)$ and Eq. S6. We limited our analysis to Experiment 1 and 2, as they had two distinct blocks per subject and therefore more data and reduced ambiguity and risk of overfitting. Results of the model comparison showed that the extended models did not gain a significant advantage in terms of marginal likelihood (data not shown), i.e. the distinction between objective and subjective sensory variability did not appear to be a relevant feature in explaining our data. This result suggests that most subjects had a reasonably accurate estimate of their own sensory variability.

Note that an analogous study for the motor (reproduction) variability is not feasible with our dataset as the problem becomes in this case under-constrained. In fact, if we separate the objective motor variability $w_{m}$ from its subjective estimate $\widetilde{w}_{m}$, some observer models do not even depend on $\widetilde{w}_{m}$ (e.g. an observer with constant motor likelihood and Standard loss function), and others show only a weak dependence. In order to meaningfully test whether people 'knew' their own motor variability a much stronger asymmetry in the loss function is needed, along with some other experimental manipulations (see for instance [3]).

# 4.3 Generalized law for motor noise

A recent study has shown violations of the scalar property for motor timing [6]; see also [7] for a review. The scalar property, taken literally, entails that motor variability decreases to zero for vanishing time intervals, which is quite unlikely; a more realistic assumption is that motor noise must reach a lower bound. The fact that many studies in time interval reproduction have shown a good agreement with the scalar property may simply mean that the lower bound was negligible for the considered interval ranges.

To verify whether this is the case for our work, we considered a 2-parameters model for the motor variability which consists of two independent noise sources, one of which is constant and the other which follows the scalar property. In this model, the equation for the motor variance is

$$
\sigma_{m}^{2}(u)=\sigma_{0}^{2}+w_{m}^{2} u^{2} \quad(\text { generalized motor variability })
$$

where $u$ is the desired reproduction interval, $w_{m}$ is the scalar coefficient (Weber's fraction) and $\sigma_{0}$ represents the lower bound for the motor noise.

We ran a full Bayesian model comparison on all datasets (including the new ones), adding the 'generalized' motor variability as a possible choice for the motor likelihood component, in addition to the basic constant and scalar motor components considered before. All other components (sensory likelihood, prior, loss function) were free to vary as per the basic model comparison (see paper).

We found that observer models with generalized motor variability obtained slightly better fits in some cases (Figure S4), but they performed better in terms of marginal likelihood (with respect to the scalar or constant models) only for two subjects in Experiment 1. For all remaining subjects and experiments the extended model did not represent an improvement in marginal likelihood - that is, the minimal gain in model fitting was hampered by the 'cost' of the additional parameter $\sigma_{0}$, meaning that in general the model does not represent a better explanation for the data.

It is not surprising that the subjects who gained some benefit from the addition of the constant noise term belonged to Experiment 1, since this experiment included a Short block and therefore might be more sensitive to the presence of a constant error for short intervals. These results show that while Eq. S7 probably applies to small intervals [6], it seems that in our study the lower bound $\sigma_{0}$ is not relevant for explaining the data and can therefore be ignored with a good approximation.

---

#### Page 9

> **Image description.** This image presents a series of plots comparing experimental data with model fits, focusing on response bias and standard deviation in relation to physical time intervals. The image is divided into two columns, labeled "Single subject" (left) and "Group mean" (right). Each column contains three subplots.
>
> - **Top Row:** The top subplot in each column displays a series of colored squares. In the "Single subject" plot, there are four red squares followed by four green squares. The "Group mean" plot shows the same arrangement of four red squares followed by four green squares. The x-axis is labeled with values 450, 600, 750, 900, and 1050.
>
> - **Middle Row:** The middle subplot in each column shows "Response bias (ms)" plotted against "Physical time interval (ms)". The y-axis ranges from -150 to 150. Data points are shown as circles, with red circles representing one condition and green circles representing another. Error bars are present on each data point. Solid and dashed lines represent model fits. The "Single subject" plot shows a more complex, curved relationship, while the "Group mean" plot shows a more linear relationship, with the green data points showing a negative slope and the red data points showing a positive slope.
>
> - **Bottom Row:** The bottom subplot in each column shows "Response sd (ms)" plotted against "Physical time interval (ms)". The y-axis ranges from 40 to 120. Again, data points are shown as red and green circles with error bars, and solid and dashed lines represent model fits. The plots show the standard deviation of the response as a function of the physical time interval.
>
> - **Text:** The x-axis of the middle and bottom subplots is labeled "Physical time interval (ms)" with values 450, 600, 750, 900, and 1050. The y-axis of the middle subplots is labeled "Response bias (ms)" and ranges from -150 to 150. The y-axis of the bottom subplots is labeled "Response sd (ms)" and ranges from 40 to 120. The "Group mean" plot includes the text "n = 4".

Figure S4. Experiment 1: comparison between basic models and models with generalized motor variability. Very top: Experimental distributions for Short Uniform (red) and Long Uniform (green) blocks, repeated on top of both columns. Left column: Mean response bias (average difference between the response and true interval duration, top) and standard deviation of the response (bottom) for a representative subject in both blocks (red: Short Uniform; green: Long Uniform). Error bars denote s.e.m. Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported basic model components (scalar or constant); dashed lines are model fits which include the generalized motor variability in the model comparison. The subject shown is the one who gained the most by the introduction of the general linear motor likelihood. Right column: Mean response bias (top) and standard deviation of the response (bottom) across subjects in both blocks (mean $\pm$ s.e.m. across subjects). Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported basic models across subjects; dashed lines are model fits which include the generalized motor variability. Although providing slightly better fits, the extended model did not represent a substantial improvement over the 1-parameter motor noise models.

# References

1. Körding KP, Wolpert DM (2004) The loss function of sensorimotor learning. Proc Natl Acad Sci U S A 101: 9839-9842.
2. Kontsevich LL, Tyler CW (1999) Bayesian adaptive estimation of psychometric slope and threshold.

---

#### Page 10

Vision Res 39: 2729-37. 3. Hudson T, Maloney L, Landy M (2008) Optimal compensation for temporal uncertainty in movement planning. PLoS Comput Biol 4: e1000130. 4. Barthelmé S, Mamassian P (2009) Evaluation of objective uncertainty in the visual system. PLoS Comput Biol 5: e1000504. 5. Battaglia PW, Kersten D, Schrater PR (2011) How haptic size sensations improve distance perception. PLoS Comput Biol 7: e1002080. 6. Laje R, Cheng K, Buonomano D (2011) Learning of temporal motor patterns: an analysis of continuous versus reset timing. Front Integr Neurosci 5. 7. Lewis PA, Miall RC (2009) The precision of temporal judgement: milliseconds, many minutes, and beyond. Proc Biol Sci 364: 1897-1905.

---

#### Page 1

# Supporting Text S2 - Results of Experiments 3 and 4

> **Image description.** This image presents a set of four scatter plots arranged in a 2x2 grid, comparing data for a single subject versus a group mean in an experiment related to interval timing. The plots are labeled and show the relationship between "Physical time interval (ms)" on the x-axis and either "Response bias (ms)" or "Response sd (ms)" on the y-axis.
>
> - **Top Row:** Each plot in the top row is accompanied by a visual representation of the experimental distribution, consisting of five equally spaced small green squares aligned horizontally above the plot. The x-axis values 600, 750, and 900 are marked below the squares.
>
> - **Left Column:** The plots in the left column are labeled "Single subject."
>
>   - The top plot shows "Response bias (ms)" versus "Physical time interval (ms)." Data points are plotted as green circles with error bars. A curved green line represents a Bayesian model fit.
>   - The bottom plot shows "Response sd (ms)" versus "Physical time interval (ms)." Data points are plotted as green circles with error bars. A curved green line represents a Bayesian model fit.
>
> - **Right Column:** The plots in the right column are labeled "Group mean." The label "n = 6" is also present.
>
>   - The top plot shows "Response bias (ms)" versus "Physical time interval (ms)." Data points are plotted as green circles with error bars. A straight green line represents a Bayesian model fit.
>   - The bottom plot shows "Response sd (ms)" versus "Physical time interval (ms)." Data points are plotted as green circles with error bars. A straight green line represents a Bayesian model fit.
>
> - **Axes:** All plots share the same x-axis labels (600, 750, 900) for "Physical time interval (ms)." The y-axes for "Response bias (ms)" range from -150 to 150, while the y-axes for "Response sd (ms)" range from 40 to 120.

Figure S5. Experiment 3: Medium Uniform block with Standard feedback. Very top: Experimental distribution for Medium Uniform block, repeated on top of both columns. Left column: Mean response bias (average difference between the response and true interval duration, top) and standard deviation of the response (bottom) for a representative subject. Error bars denote s.e.m. Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models (Bayesian model averaging). Right column: Mean response bias (top) and standard deviation of the response (bottom) across subjects (mean $\pm$ s.e.m. across subjects). Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models across subjects.

---

#### Page 2

> **Image description.** This image contains two columns of plots comparing data for a single subject versus a group mean. Each column contains three subplots arranged vertically.
>
> - **Top Subplots (Histograms):** Each column's top subplot is a histogram. The x-axis is labeled with values 600, 750, and 900. The left histogram is titled "Single subject" and the right one is titled "Group mean". The left histogram has a single blue bar above the 750 mark, while the right histogram also has a single blue bar above the 750 mark.
>
> - **Middle Subplots (Response Bias):** The middle subplots in both columns are scatter plots showing "Response bias (ms)" on the y-axis, ranging from -150 to 150, and "Physical time interval (ms)" on the x-axis, ranging from 600 to 900. Data points are marked with blue circles and error bars. A horizontal gray line is at y=0. A blue curve is fitted to the data points in each plot. The plot on the right has the label "n = 3".
>
> - **Bottom Subplots (Response sd):** The bottom subplots in both columns are scatter plots showing "Response sd (ms)" on the y-axis, ranging from 40 to 120, and "Physical time interval (ms)" on the x-axis, ranging from 600 to 900. Data points are marked with blue circles and error bars. A blue curve is fitted to the data points in each plot.

Figure S6. Experiment 4: Medium High-Peaked block. Very top: Experimental distribution for Medium High-Peaked block, repeated on top of both columns. Left column: Mean response bias (average difference between the response and true interval duration, top) and standard deviation of the response (bottom) for a representative subject. Error bars denote s.e.m. Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models (Bayesian model averaging). Right column: Mean response bias (top) and standard deviation of the response (bottom) across subjects (mean $\pm$ s.e.m. across subjects). Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models across subjects.
