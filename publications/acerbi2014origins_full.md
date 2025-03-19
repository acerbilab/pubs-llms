```
@article{acerbi2014origins,
  title={On the Origins of Suboptimality in Human Probabilistic Inference},
  author={Luigi Acerbi and Sethu Vijayakumar and Daniel M. Wolpert},
  year={2014},
  journal={PLoS Computational Biology},
  doi={10.1371/journal.pcbi.1003661},
}
```

---

#### Page 1

# On the Origins of Suboptimality in Human Probabilistic Inference

Luigi Acerbi ${ }^{1,2}$, Sethu Vijayakumar ${ }^{1}$, Daniel M. Wolpert ${ }^{3}$<br>1 Institute of Perception, Action and Behaviour, School of Informatics, University of Edinburgh, Edinburgh, United Kingdom, 2 Doctoral Training Centre in Neuroinformatics and Computational Neuroscience, School of Informatics, University of Edinburgh, Edinburgh, United Kingdom, 3 Computational and Biological Learning Lab, Department of Engineering, University of Cambridge, Cambridge, United Kingdom

#### Abstract

Humans have been shown to combine noisy sensory information with previous experience (priors), in qualitative and sometimes quantitative agreement with the statistically-optimal predictions of Bayesian integration. However, when the prior distribution becomes more complex than a simple Gaussian, such as skewed or bimodal, training takes much longer and performance appears suboptimal. It is unclear whether such suboptimality arises from an imprecise internal representation of the complex prior, or from additional constraints in performing probabilistic computations on complex distributions, even when accurately represented. Here we probe the sources of suboptimality in probabilistic inference using a novel estimation task in which subjects are exposed to an explicitly provided distribution, thereby removing the need to remember the prior. Subjects had to estimate the location of a target given a noisy cue and a visual representation of the prior probability density over locations, which changed on each trial. Different classes of priors were examined (Gaussian, unimodal, bimodal). Subjects' performance was in qualitative agreement with the predictions of Bayesian Decision Theory although generally suboptimal. The degree of suboptimality was modulated by statistical features of the priors but was largely independent of the class of the prior and level of noise in the cue, suggesting that suboptimality in dealing with complex statistical features, such as bimodality, may be due to a problem of acquiring the priors rather than computing with them. We performed a factorial model comparison across a large set of Bayesian observer models to identify additional sources of noise and suboptimality. Our analysis rejects several models of stochastic behavior, including probability matching and sample-averaging strategies. Instead we show that subjects' response variability was mainly driven by a combination of a noisy estimation of the parameters of the priors, and by variability in the decision process, which we represent as a noisy or stochastic posterior.

## Introduction

Humans have been shown to integrate prior knowledge and sensory information in a probabilistic manner to obtain optimal (or nearly so) estimates of behaviorally relevant stimulus quantities, such as speed [1,2], orientation [3], direction of motion [4], interval duration [5-8] and position [9-11]. Prior expectations about the values taken by the task-relevant variable are usually assumed to be learned either from statistics of the natural environment [1-3] or during the course of the experiment [4-$6,8-11]$; the latter include studies in which a pre-existing prior is modified in the experimental context $[12,13]$. Behavior in these perceptual and sensorimotor tasks is qualitatively and often quantitatively well described by Bayesian Decision Theory (BDT) $[14,15]$.

The extent to which we are capable of performing probabilistic inference on complex distributions that go beyond simple Gaussians, and the algorithms and approximations that we might
use, is still unclear [14]. For example, it has been suggested that humans might approximate Bayesian computations by drawing random samples from the posterior distribution [16-19]. A major problem in testing hypotheses about human probabilistic inference is the difficulty in identifying the source of suboptimality, that is, separating any constraints and idiosyncrasies in performing Bayesian computations per se from any deficiencies in learning and recalling the correct prior. For example, previous work has examined Bayesian integration in the presence of experimentallyimposed bimodal priors [4,8,9,20]. Here the normative prescription of BDT under a wide variety of assumptions would be that responses should be biased towards one peak of the distribution or the other, depending on the current sensory information. However, for such bimodal priors, the emergence of Bayesian biases can require thousands of trials [9] or be apparent only on pooled data [4], and often data show at best a complex pattern of biases which is only in partial agreement with the underlying distribution [8,20]. It is unknown whether this mismatch is due to

---

#### Page 2

## Author Summary

The process of decision making involves combining sensory information with statistics collected from prior experience. This combination is more likely to yield 'statistically optimal' behavior when our prior experiences conform to a simple and regular pattern. In contrast, if prior experience has complex patterns, we might require more trial-and-error before finding the optimal solution. This partly explains why, for example, a person deciding the appropriate clothes to wear for the weather on a June day in Italy has a higher chance of success than her counterpart in Scotland. Our study uses a novel experimental setup that examines the role of complexity of prior experience on suboptimal decision making. Participants are asked to find a specific target from an array of potential targets given a cue about its location. Importantly, the 'prior' information is presented explicitly so that subjects do not need to recall prior events. Participants' performance, albeit suboptimal, was mostly unaffected by the complexity of the prior distributions, suggesting that remembering the patterns of past events constitutes more of a challenge to decision making than manipulating the complex probabilistic information. We introduce a mathematical description that captures the pattern of human responses in our task better than previous accounts.
the difficulty of learning statistical features of the bimodal distribution or if the bimodal prior is actually fully learned but our ability to perform Bayesian computation with it is limited. In the current study we look systematically at how people integrate uncertain cues with trial-dependent 'prior' distributions that are explicitly made available to the subjects. The priors were displayed as an array of potential targets distributed according to various density classes - Gaussian, unimodal or bimodal. Our paradigm allows full control over the generative model of the task and separates the aspect of computing with a probability distribution from the problem of learning and recalling a prior. We examine subjects' performance in manipulating probabilistic information as a function of the shape of the prior. Participants' behavior in the task is in qualitative agreement with Bayesian integration, although quite variable and generally suboptimal, but the degree of suboptimality does not differ significantly across different classes of distributions or levels of reliability of the cue. In particular, performance was not greatly affected by complexity of the distribution per se - for instance, people's performance with bimodal priors is analogous to that with Gaussian priors, in contrast to previous learning experiments [8,9]. This finding suggests that major deviations encountered in previous studies are likely to be primarily caused by the difficulty in learning complex statistical features rather than computing with them.

We systematically explore the sources of suboptimality and variability in subjects' responses by employing a methodology that has been recently called factorial model comparison [21]. Using this approach we generate a set of models by combining different sources of suboptimality, such as different approximations in decision making with different forms of sensory noise, in a factorial manner. Our model comparison is able to reject some common models of variability in decision making, such as probability matching with the posterior distribution (posterior-matching) or a sampling-average strategy consisting of averaging a number of samples from the posterior distribution. The observer model that best describes the data is a Bayesian observer with a slightly mismatched representation of the likelihoods, with sensory noise in the estimation of the parameters of the prior, that occasionally
lapses, and most importantly has a stochastic representation of the posterior that may represent additional variability in the inference process or in action selection.

## Results

Subjects were required to locate an unknown target given probabilistic information about its position along a target line (Figure 1a-b). Information consisted of a visual representation of the a priori probability distribution of targets for that trial and a noisy cue about the actual target position (Figure 1b). On each trial a hundred potential targets (dots) were displayed on a horizontal line according to a discrete representation of a trialdependent 'prior' distribution $p_{\text {prior }}(x)$. The true target, unknown to the subject, was chosen at random from the potential targets with uniform probability. A noisy cue with horizontal position $x_{\text {cue }}$, drawn from a normal distribution centered on the true target, provided partial information about target location. The cue had distance $d_{\text {cue }}$ from the target line, which could be either a short distance, corresponding to added noise with low-variance, or a long distance, with high-variance noise. Both prior distribution and cue remained on screen for the duration of the trial. (See Figure 1c-d for the generative model of the task.) The task for the subjects involved moving a circular cursor controlled by a manipulandum towards the target line, ending the movement at their best estimate for the position of the real target. A 'success' ensued if the true target was within the cursor radius.

To explain the task, subjects were told that the each dot represented a child standing in a line in a courtyard, seen from a bird's eye view. On each trial a random child was chosen and, while the subject was 'not looking', the child threw a yellow ball (the cue) directly ahead of them towards the opposite wall. Due to their poor throwing skills, the farther they threw the ball the more imprecise they were in terms of landing the ball straight in front of them. The subject's task was to identify the child who threw the ball, after seeing the landing point of the ball, by encircling him or her with the cursor. Subjects were told that the child throwing the ball could be any of the children, chosen randomly each trial with equal probability.

Twenty-four subjects performed a training session in which the 'prior' distributions of targets shown on the screen (the set of children) corresponded to Gaussian distributions with a standard deviation (SD) that varied between trials ( $\sigma_{\text {prior }}$ from 0.04 to 0.18 standardized screen units; Figure 2a). On each trial the location (mean) of the prior was chosen randomly from a uniform distribution. Half of the trials provided the subjects with a 'short-distance' cue about the position of the target (low noise: $\sigma_{\text {low }}=0.06$ screen units; a short throw of the ball); the other half had a 'long-distance' cue (high noise: $\sigma_{\text {high }}=0.14$ screen units; a long throw). The actual position of the target (the 'child' who threw the ball) was revealed at the end of each trial and a displayed score kept track of the number of 'successes' in the session (full performance feedback). The training session allowed subjects to learn the structure of the task in a setting in which humans are known to perform in qualitative and often quantitative agreement with Bayesian Decision Theory, i.e. under Gaussian priors [5,9-11]. Note however that, in contrast with the previous studies, our subjects were required to compute each trial with a different Gaussian distribution (Figure 2a). The use of Gaussian priors in the training session allowed us to assess whether our subjects could use explicit priors in our novel experimental setup in the same way in which they have been shown to learn Gaussian priors through extended implicit practice.

---

#### Page 3

> **Image description.** This image shows a figure with five panels (a-e) illustrating an experimental procedure and generative model of a task.
>
> Panel a: Shows a schematic of the experimental setup. A person is seated, holding the handle of a "Robotic manipulandum". Above the manipulandum is a "Mirror" reflecting the image from a "CRT display" onto the plane of the hand, creating a "Virtual cursor".
>
> Panel b: Depicts the screen setup. A coordinate system is shown with x and y axes. A "home position" is represented by a large red circle with a grey outline. A line of potential targets is represented by a horizontal line of small grey dots. A "visual cue" is indicated by a yellow dot.
>
> Panel c: Is a directed acyclic graph representing the generative model of the task. The nodes are labeled as follows: p_prior (in a circle), x (in a circle), d_cue (in a circle), and x_cue (in a circle). There are directed edges from p_prior to x, from x to x_cue, and from d_cue to x_cue.
>
> Panel d: Illustrates details of the generative model. At the top, there is a graph of p_prior(x) as a curve with a shaded area underneath, and P_prior(x) as a cumulative distribution function. Horizontal lines connect the two graphs. Below the graphs is a horizontal line with potential targets (grey dots) and the true target (red dot). Two Gaussian distributions are shown below: one labeled "p(x_cue | x, d_short)" representing a low-noise cue (yellow curve), and the other labeled "p(x_cue | x, d_long)" representing a high-noise cue (yellow curve).
>
> Panel e: Shows the prior, likelihood, posterior, and expected loss. At the top is "Prior" represented as a horizontal line with potential targets (grey dots). A specific target is marked as x*. Below is the "Likelihood" represented as a yellow curve. Below that is the "Posterior" represented as a cyan curve. At the bottom is "Expected loss" represented as a magenta curve. A vertical dashed magenta line connects the x* target to the peak of the expected loss curve.

Figure 1. Experimental procedure. a: Setup. Subjects held the handle of a robotic manipulandum. The visual scene from a CRT monitor, including a cursor that tracked the hand position, was projected into the plane of the hand via a mirror. b: Screen setup. The screen showed a home position (grey circle), the cursor (red circle) here at the start of a trial, a line of potential targets (dots) and a visual cue (yellow dot). The task consisted in locating the true target among the array of potential targets, given the position of the noisy cue. The coordinate axis was not displayed on screen, and the target line is shaded here only for visualization purposes. c: Generative model of the task. On each trial the position of the hidden target $x$ was drawn from a discrete representation of the trial-dependent prior $p_{\text {prior }}(x)$, whose shape was chosen randomly from a sessiondependent class of distributions. The vertical distance of the cue from the target line, $d_{\text {cue }}$, was either 'short' or 'long', with equal probability. The horizontal position of the cue, $x_{\text {cue }}$, depended on $x$ and $d_{\text {cue }}$. The participants had to infer $x$ given $x_{\text {cue }}, d_{\text {cue }}$ and the current prior $p_{\text {prior }}$. d: Details of the generative model. The potential targets constituted a discrete representation of the trial-dependent prior distribution $p_{\text {prior }}(x)$; the discrete representation was built by taking equally spaced samples from the inverse of the cdf of the prior, $P_{\text {prior }}(x)$. The true target (red dot) was chosen uniformly at random from the potential targets, and the horizontal position of the cue (yellow dot) was drawn from a Gaussian distribution, $p\left(x_{\text {cue }}\left(x, d_{\text {cue }}\right)\right.$, centered on the true target $x$ and whose $S D$ was proportional to the distance $d_{\text {cue }}$ from the target line (either 'short' or 'long', depending on the trial, for respectively low-noise and high-noise cues). Here we show the location of the cue for a high-noise trial. e: Components of Bayesian decision making. According to Bayesian Decision Theory, a Bayesian ideal observer combines the prior distribution with the likelihood function to obtain a posterior distribution. The posterior is then convolved with the loss function (in this case whether the target will be encircled by the cursor) and the observer picks the 'optimal' target location $x^{\prime \prime}$ (purple dot) that corresponds to the minimum of the expected loss (dashed line).

doi:10.1371/journal.pcbi. 1003661 . g 001

After the training session, subjects were randomly divided in three groups ( $n=8$ each) to perform a test session. Test sessions differed with respect to the class of prior distributions displayed during the session. For the 'Gaussian test' group, the distributions were the same eight Gaussian distributions of varying SD used during training (Figure 2a). For the 'unimodal test' group, on each trial the prior was randomly chosen from eight unimodal distributions with fixed SD $\left(\sigma_{\text {prior }}=0.11\right.$ screen units) but with varying skewness and kurtosis (see Methods and Figure 2b). For the 'bimodal test' group, priors were chosen from eight (mostly) bimodal distributions with fixed SD (again, $\sigma_{\text {prior }}=0.11$ screen units) but variable separation and weighting
between peaks (see Methods and Figure 2c). As in the training session, on each trial the mean of the prior was drawn randomly from a uniform distribution. To preserve global symmetry during the session, asymmetric priors were 'flipped' along their center of mass with a probability of $1 / 2$. During the test session, at the end of each trial subjects were informed whether they 'succeeded' or 'missed' the target but the target's actual location was not displayed (partial feedback). The 'Gaussian test' group allowed us to verify that subjects' behavior would not change after removal of full performance feedback. The 'unimodal test' and 'bimodal test' groups provided us with novel information on how subjects perform

---

#### Page 4

> **Image description.** The image is a figure displaying prior probability density distributions. It consists of three panels labeled "a. Gaussian session", "b. Unimodal session", and "c. Bimodal session". Each panel contains a 4x2 grid of plots, each showing a different probability density function.
>
> - **Overall Layout:** The figure is organized into three columns (a, b, c) representing different types of prior distributions (Gaussian, Unimodal, and Bimodal, respectively). Each column has eight plots arranged in four rows and two columns. Each plot is enclosed in a square frame. A number (1-8) is located in the upper left corner of each plot.
>
> - **Axes and Labels:** The y-axis is labeled "Prior probability density" and is shared by all three panels. The x-axis is labeled "Target position (screen units)" and ranges from -0.5 to 0.5.
>
> - **Panel a: Gaussian Session:** The plots in this panel display Gaussian distributions. The distributions vary in width, with some being very narrow and peaked, while others are wider and flatter.
>
> - **Panel b: Unimodal Session:** The plots in this panel show unimodal distributions that are not necessarily symmetrical. Some are skewed to the left or right, and some have a flat top. One plot (row 4, column 1) shows a uniform distribution.
>
> - **Panel c: Bimodal Session:** The plots in this panel display bimodal distributions, each having two peaks. The relative height and separation of the peaks vary across the plots.
>
> The plots are all rendered in black lines on a white background, with the area under the curves filled in gray.

Figure 2. Prior distributions. Each panel shows the (unnormalized) probability density for a 'prior' distribution of targets, grouped by experimental session, with eight different priors per session. Within each session, priors are numbered in order of increasing differential entropy (i.e. increasing variance for Gaussian distributions). During the experiment, priors had a random location (mean drawn uniformly) and asymmetrical priors had probability $1 / 2$ of being 'flipped'. Target positions are shown in standardized screen units (from -0.5 to 0.5 ). a: Gaussian priors. These priors were used for the training session, common to all subjects, and in the Gaussian test session. Standard deviations cover the range $\sigma_{\text {prior }}=0.04$ to 0.18 screen units in equal increments. b: Unimodal priors. All unimodal priors have fixed SD $\sigma_{\text {prior }}=0.11$ screen units but different skewness and kurtosis (see Methods for details). c: Bimodal priors. All priors in the bimodal session have fixed SD $\sigma_{\text {prior }}=0.11$ screen units but different relative weights and separation between the peaks (see Methods).

doi:10.1371/journal.pcbi. 1003661 . g002
probabilistic inference with complex distributions. Moreover, non-Gaussian priors allowed us to evaluate several hypotheses about subjects' behavior that are not testable with Gaussian distributions alone [22].

## Human performance

We first performed a model-free analysis of subjects' performance. Figure 3 shows three representative prior distributions and the pooled subjects' responses as a function of the cue position for low (red) and high (blue) noise cues. Note that pooled data are used here only for display and all subjects' datasets were analyzed individually. The cue positions and responses in Figure 3 are reported in a coordinate system relative to the mean of the prior (set as $\mu_{\text {prior }}=0$ ). For all analyses we consider relative coordinates without loss of generality, having verified the assumption of translational invariance of our task (see Section 1 in Text S1).

Figure 3 shows that subjects' performance was affected by both details of the prior distribution and the cue. Also, subjects' mean performance (continuous lines in Figure 3) show deviations from the prediction of an optimal Bayesian observer (dashed lines), suggesting that subjects behavior may have been suboptimal.

Linear integration with Gaussian priors. We examined how subjects performed in the task under the well-studied case of Gaussian priors [9,10]. Given a Gaussian prior with SD $\sigma_{\text {prior }}$ and a noisy cue with horizontal position $x_{\text {cue }}$ and known variability $\sigma_{\text {cue }}$ (assuming Gaussian noise), the most likely target location can be computed through Bayes' theorem. In the relative coordinate system $\left(\mu_{\text {prior }}=0\right)$, the optimal target location takes the simple linear form:

$$
\begin{aligned}
& x^{*}\left(x_{\text {cue }}\right)=w \cdot x_{\text {cue }} \\
& \text { with } w=\frac{\sigma_{\text {prior }}^{2}}{\sigma_{\text {prior }}^{2}+\sigma_{\text {cue }}^{2}} \quad \text { (relative coordinates) }
\end{aligned}
$$

where $w$ is the linear weight assigned to the cue.
We compared subjects' behavior with the 'optimal' strategy predicted by Eq. 1 (see for instance Figure 3a; the dashed line corresponds to the optimal strategy). For each subject and each combination of $\sigma_{\text {prior }}$ and cue type (either 'short' or 'long', corresponding respectively to low-noise and high-noise cues), we fit the responses $r$ as a function of the cue position $x_{\text {cue }}$ with a robust linear fit. The slopes of these fits for the training session are plotted in Figure 4; results were similar for the Gaussian test session. Statistical differences between different conditions were assessed using repeated-measures ANOVA (rm-ANOVA) with Green-house-Geisser correction (see Methods).

In general, subjects did not perform exactly as predicted by the optimal strategy (dashed lines), but they took into account the probabilistic nature of the task. Specifically, subjects tended to give more weight to low-noise cues than to high-noise ones (main effect: Low-noise cues, High-noise cues; $F_{(1,23)}=145, p<0.001$ ), and the weights were modulated by the width of the prior (main effect: prior width $\sigma_{\text {prior }} ; F_{(3.45,79.2)}=88, \epsilon=0.492, p<0.001$ ), with wider priors inducing higher weighting of the cue. Interestingly, cue type and width of the prior seemed to influence the weights independently, as no significant interaction was found (interaction: prior width $\times$ cue type; $F_{(4.86,112)}=0.94, \epsilon=0.692$, $p=0.46$ ). Analogous patterns were found in the Gaussian test session.

---

#### Page 5

> **Image description.** The image contains three panels (a, b, and c), each consisting of three subplots arranged vertically. Each panel represents data related to different prior distributions: Gaussian (a), unimodal (b), and bimodal (c).
>
> - **Top Row (Prior Distributions):** Each panel's top subplot displays a probability distribution.
>
>   - Panel a: A bell-shaped curve, representing a Gaussian distribution, filled in gray. The x-axis ranges from approximately -0.2 to 0.2. The y-axis is labeled "Probability".
>   - Panel b: A flat-topped, unimodal distribution, filled in gray. The x-axis ranges from approximately -0.2 to 0.2.
>   - Panel c: A bimodal distribution with two peaks, filled in gray. The x-axis ranges from approximately -0.2 to 0.2.
>
> - **Middle Row (Low-Noise Cue):** Each panel's middle subplot displays a scatter plot of "Response" versus "Cue position (screen units)" for low-noise cues.
>
>   - The x-axis ranges from approximately -0.2 to 0.2.
>   - The y-axis ranges from approximately -0.2 to 0.2 and is labeled "Response".
>   - Data points are shown as small red dots.
>   - A solid red line represents a kernel regression estimate of the mean response.
>   - A dashed black line represents the Bayes optimal strategy.
>   - The text "Low-noise cue" is positioned near the bottom right of each subplot.
>
> - **Bottom Row (High-Noise Cue):** Each panel's bottom subplot displays a scatter plot of "Response" versus "Cue position (screen units)" for high-noise cues.
>   - The x-axis ranges from approximately -0.2 to 0.2 and is labeled "Cue position (screen units)".
>   - The y-axis ranges from approximately -0.2 to 0.2 and is labeled "Response".
>   - Data points are shown as small blue dots.
>   - A solid blue line represents a kernel regression estimate of the mean response.
>   - A dashed black line represents the Bayes optimal strategy.
>   - The text "High-noise cue" is positioned near the bottom right of each subplot.

Figure 3. Subjects' responses as a function of the position of the cue. Each panel shows the pooled subjects' responses as a function of the position of the cue either for low-noise cues (red dots) or high-noise cues (blue dots). Each column corresponds to a representative prior distribution, shown at the top, for each different group (Gaussian, unimodal and bimodal). In the response plots, dashed lines correspond to the Bayes optimal strategy given the generative model of the task. The continuous lines are a kernel regression estimate of the mean response (see Methods). a. Exemplar Gaussian prior (prior 4 in Figure 2a). b. Exemplar unimodal prior (platykurtic distribution: prior 4 in Figure 2b). c. Exemplar bimodal prior (prior 5 in Figure 2c). Note that in this case the mean response is not necessarily a good description of subjects' behavior, since the marginal distribution of responses for central positions of the cue is bimodal. doi:10.1371/journal.pcbi. 1003661 . g 003

> **Image description.** This image consists of two line graphs, one above the other, comparing response slopes under different conditions.
>
> The top graph is titled "Single subject". The bottom graph is titled "Group mean (n = 24)". Both graphs share the same x and y axes. The y-axis is labeled "Response slope w" and ranges from 0 to 1. The x-axis is labeled "σprior (screen units)" and ranges from approximately 0.04 to 0.18.
>
> Each graph contains four lines:
>
> - A red dashed line labeled "Bayes optimal (short cue)".
> - A blue dashed line labeled "Bayes optimal (long cue)".
> - A solid red line labeled "Subj. average (short cue)". This line has error bars.
> - A solid blue line labeled "Subj. average (long cue)". This line also has error bars.
>
> The lines represent the response slope as a function of the SD of the Gaussian prior distribution. The dashed lines represent the Bayes optimal strategy, while the solid lines represent the subject average. The red lines represent trials with low noise ('short' cues), and the blue lines represent trials with high noise ('long' cues). The error bars on the solid lines indicate the standard error.
>
> In both graphs, the red lines are generally higher than the blue lines. The dashed lines are generally below the solid lines. The "Single subject" graph shows more variation in the subject average lines compared to the "Group mean" graph, which shows smoother lines.

Figure 4. Response slopes for the training session. Response slope $w$ as a function of the SD of the Gaussian prior distribution, $\sigma_{\text {prior }}$, plotted respectively for trials with low noise ('short' cues, red line) and high noise ('long' cues, blue line). The response slope is equivalent to the linear weight assigned to the position of the cue (Eq. 1). Dashed lines represent the Bayes optimal strategy given the generative model of the task in the two noise conditions. Top: Slopes for a representative subject in the training session (slope $\pm$ SE). Bottom: Average slopes across all subjects in the training session ( $n=24$, mean $\pm$ SE across subjects). doi:10.1371/journal.pcbi. 1003661 .g004

We also examined the average bias of subjects' responses (intercept of linear fits), which is expected to be zero for the optimal strategy. On average subjects exhibited a small but significant rightward bias in the training session of $(5.2 \pm 1.2) \cdot 10^{-3}$ screen units or $1 \sim 2 \mathrm{~mm}$ (mean $\pm$ SE across subjects, $p<10^{-3}$ ). The average bias was only marginally different than zero in the test session: $(3.2 \pm 1.6) \cdot 10^{-3}$ screen units ( $\sim 1$ $\mathrm{mm}, p=0.08$ ).

Optimality index. We developed a general measure of performance that is applicable beyond the Gaussian case. An objective measure of performance in each trial is the success probability, that is, the probability that the target would be within a cursor radius' distance from the given response (final position of the cursor) under the generative model of the task (see Methods). We defined the optimality index for a trial as the success probability normalized by the maximal success probability (the success probability of an optimal response). The optimality index allows us to study variations in subjects' performance which are not trivially induced by variations in the difficulty of the task. Figure 5 shows the optimality index averaged across subjects for different conditions, in different sessions. Data are also summarized in Table 1. Priors in Figure 5 are listed in order of differential entropy (which corresponds to increasing variance for Gaussian priors), with the exception of 'unimodal test' priors which are in order of increasing width of the main peak in the prior, as computed through a Laplace approximation. We chose this ordering for priors in the unimodal test session as it highlights the pattern in subjects' performance (see below).

For a comparison, Figure 5 also shows the optimality index of two suboptimal models that represent two extremal response strategies. Dash-dotted lines correspond to the optimality index of a Bayesian observer that maximizes the probability of locating the

---

#### Page 6

> **Image description.** The image is a set of bar graphs comparing optimality indices across different conditions. It is arranged as a 2x2 grid of plots.
>
> - **Overall Structure:** Each of the four panels contains two sets of bar graphs, one for "Low-noise cue" and one for "High-noise cue." Each bar graph represents a different "prior distribution," indexed by numbers along the x-axis. The y-axis, labeled "Optimality index," ranges from 0 to 1. Error bars are present on top of each bar. A dotted line and a dash-dotted line are overlaid on each set of bars, representing "Cue-only model" and "Prior-only model" respectively. A shaded gray area is present near the top of each plot, labeled "Synergistic integration area."
>
> - **Panel 1: Gaussian training (n = 24):** The title indicates that this panel represents data from a "Gaussian training" condition with a sample size of 24. The x-axis is labeled with the numbers 1 through 8, representing different prior distributions. The bars for "Low-noise cue" are red, gradually fading to lighter shades of red. The bars for "High-noise cue" are blue, gradually fading to lighter shades of blue.
>
> - **Panel 2: Gaussian test (n = 8):** The title indicates that this panel represents data from a "Gaussian test" condition with a sample size of 8. The x-axis is labeled with the numbers 1 through 8, representing different prior distributions. The bars for "Low-noise cue" are red, gradually fading to lighter shades of red. The bars for "High-noise cue" are blue, gradually fading to lighter shades of blue.
>
> - **Panel 3: Unimodal test (n = 8):** The title indicates that this panel represents data from a "Unimodal test" condition with a sample size of 8. The x-axis is labeled with the numbers 6, 1, 3, 2, 5, 7, 8, and 4, representing different prior distributions. The bars for "Low-noise cue" are red, gradually fading to lighter shades of red. The bars for "High-noise cue" are blue, gradually fading to lighter shades of blue.
>
> - **Panel 4: Bimodal test (n = 8):** The title indicates that this panel represents data from a "Bimodal test" condition with a sample size of 8. The x-axis is labeled with the numbers 1 through 8, representing different prior distributions. The bars for "Low-noise cue" are red, gradually fading to lighter shades of red. The bars for "High-noise cue" are blue, gradually fading to lighter shades of blue.
>
> - **Legend:** A legend is present at the top of the image, indicating that the dotted line represents the "Cue-only model," the dash-dotted line represents the "Prior-only model," and the shaded gray area represents the "Synergistic integration area."

Figure 5. Group mean optimality index. Each bar represents the group-averaged optimality index for a specific session, for each prior (indexed from 1 to 8 , see also Figure 2) and cue type, low-noise cues (red bars) or high-noise cues (blue bars). The optimality index in each trial is computed as the probability of locating the correct target based on the subjects' responses divided by the probability of locating the target for an optimal responder. The maximal optimality index is 1 , for a Bayesian observer with correct internal model of the task and no sensorimotor noise. Error bars are SE across subjects. Priors are arranged in the order of differential entropy (i.e. increasing variance for Gaussian priors), except for 'unimodal test' priors which are listed in order of increasing width of the main peak in the prior (see text). The dotted line and dash-dotted line represent the optimality index of a suboptimal observer that takes into account respectively either only the cue or only the prior. The shaded area is the zone of synergistic integration, in which an observer performs better than using information from either the prior or the cue alone.

doi:10.1371/journal.pcbi. 1003661 . g 005
correct target considering only the prior distribution (see below for details). Conversely, dotted lines correspond to an observer that only uses the cue and ignores the prior: that is, the observer's response in a trial matches the current position of the cue. The shaded gray area specifies the 'synergistic integration' zone, in which the subject is integrating information from both prior and cue in a way that leads to better performance than by using either the prior or the cue alone. Qualitatively, the behavior in the gray area can be regarded as 'close to optimal', whereas performance below the gray area is suboptimal. As it is clear from Figure 5, in all sessions participants were sensitive to probabilistic information from both prior and cue - that is, performance is always above the
minimum of the extremal models (dash-dotted and dotted lines) in agreement with what we observed in Figure 4 for Gaussian sessions, although their integration was generally suboptimal. Human subjects were analogously found to be suboptimal in a previous task that required to take into account explicit probabilistic information [23].

We examined how the optimality index changed across different conditions. From the analysis of the training session, it seems that subjects were able to integrate low-noise and high-noise cues for priors of any width equally well, as we found no effect of cue type on performance (main effect: Low-noise cues, High-noise cues; $F_{(1,23)}=0.015, p=0.90$ ) and no significant interaction between

Table 1. Group mean optimality index.

| Session           | Low-noise cue   | High-noise cue  | All cues        |
| :---------------- | :-------------- | :-------------- | :-------------- |
| Gaussian training | $0.86 \pm 0.02$ | $0.87 \pm 0.01$ | $0.87 \pm 0.01$ |
| Gaussian test     | $0.89 \pm 0.02$ | $0.88 \pm 0.02$ | $0.89 \pm 0.01$ |
| Unimodal test     | $0.85 \pm 0.03$ | $0.80 \pm 0.04$ | $0.83 \pm 0.02$ |
| Bimodal test      | $0.90 \pm 0.02$ | $0.89 \pm 0.01$ | $0.89 \pm 0.01$ |
| All sessions      | $0.87 \pm 0.01$ | $0.87 \pm 0.01$ | $0.87 \pm 0.01$ |

Each entry reports mean $\pm$ SE of the group optimality index for a specific session and cue type, or averaged across all sessions/cues. See also Figure 5. doi:10.1371/journal.pcbi. 1003661 . t001

---

#### Page 7

cue types and prior width (interaction: prior width $\times$ cue type; $F_{(5.64,129.6)}=1.56, \epsilon=0.81, p=0.17$ ). However, relative performance was significantly affected by the width of the prior per se (main effect: prior width $\sigma_{\text {prior }} ; F_{(2.71,62.3)}=17.94, \epsilon=0.387$, $p<0.001$ ); people tended to perform worse with wider priors, in a way that is not simply explained by the objective decrease in the probability of locating the correct target due to the less available information (see Discussion).

Results in the Gaussian test session (Figure 5 top right) replicated what we had obtained in the training session. Subjects' performance was not influenced by cue type (main effect: Lownoise cues, High-noise cues; $F_{(1,7)}=0.026, p=0.88$ ) nor by the interaction between cue types and prior width (interaction: prior width $\times$ cue type; $F_{(2.65,18.57)}=0.67, \epsilon=0.379, p=0.56$ ). Conversely, as before, the width of the prior affected performance significantly (main effect: prior width $\sigma_{\text {prior }} ; F_{(1.47,10.3)}=5.21$, $\epsilon=0.21, p<0.05$ ); again, wider priors were associated with lower relative performance.

A similar pattern of results was found also for the bimodal test session (Figure 5 bottom right). Performance was affected significantly by the shape of the prior (main effect: prior shape; $F_{(4.01,28.1)}=3.93, \epsilon=0.573, p<0.05$ ) but otherwise participants integrated cues of different type with equal skill (main effect: Lownoise cues, High-noise cues; $F_{(1,7)}=1.42, p=0.27$; interaction: prior shape $\times$ cue type; $F_{(2.84,19.9)}=1.1, \epsilon=0.406, p=0.37$ ). However, in this case performance was not clearly correlated with a simple measure of the prior or of the average posterior (e.g. differential entropy).

Another scenario emerged in the unimodal test session (Figure 5 bottom left). Here, subjects' performance was affected not only by the shape of the prior (main effect: prior shape; $F_{(3.79,26.5)}=20.7$, $\epsilon=0.542, p<0.001$ ) but also by the type of cue (main effect: Lownoise cues, High-noise cues; $F_{(1,7)}=9.85, p<0.05$ ) and the specific combination of cue and prior (interaction: prior shape $\times$ cue type; $F_{(3.53,24.7)}=5.27, \epsilon=0.504, p<0.01$ ). Moreover, in this session performance improved for priors whose main peak was broader (see Discussion).

Notwithstanding this heterogeneity of results, an overall comparison of participants' relative performance in test sessions (averaging results over priors) did not show statistically significant differences between groups (main effect: group; $F_{(2,21)}=2.13$, $p=0.14$ ) nor between the two levels of reliability of the cue (main effect: Low-noise cues, High-noise cues; $F_{(1,21)}=3.36, p=0.08$ ); only performance in the unimodal session for high-noise cues was at most marginally worse. In particular, relative performance in the Gaussian test and the bimodal test sessions was surprisingly similar, unlike previous learning experiments (see Discussion).

Effects of uncertainty on reaction time. Lastly, we examined the effect of uncertainty on subjects' reaction time (time to start movement after the 'go' beep) in each trial. Uncertainty was quantified as the SD of the posterior distribution in the current trial, $\sigma_{\text {post }}$ (an alternative measure of spread, exponential entropy [24], gave analogous results). We found that the average subjects' reaction time grew almost linearly with $\sigma_{\text {post }}$ (Figure 6). The average change in reaction times (from lowest to highest uncertainty in the posterior) was substantial during the training session ( $\sim 50 \mathrm{~ms}$, about $15 \%$ change), although less so in subsequent test sessions.

## Suboptimal Bayesian observer models

Our model-free analysis showed that subjects' performance in the task was suboptimal. Here we examine the source of this apparent suboptimality. Subjects' performance is modelled with a
family of Bayesian ideal observers which incorporate various hypotheses about the decision-making process and internal representation of the task, with the aim of teasing out the major sources of subjects' suboptimality; see Figure 1e for a depiction of the elements of decision making in a trial. All these observers are 'Bayesian' because they build a posterior distribution through Bayes' rule, but the operations they perform with the posterior can differ from the normative prescriptions of Bayesian Decision Theory (BDT).

We construct a large model set with a factorial approach that consists in combining different independent model 'factors' that can take different 'levels' [8,21]. The basic factors we consider are:

1. Decision making (3 levels): Bayesian Decision Theory ('BDT'), stochastic posterior ('SPK'), posterior probability matching ('PPM').
2. Cue-estimation sensory noise (2 levels): absent or present (' S ').
3. Noisy estimation of the prior ( 2 levels): absent or present ('P').
4. Lapse (2 levels): absent or present ('L').

Observer models are identified by a model string, for example 'BDT-P-L' indicates an observer model that follows BDT with a noisy estimate of the prior and suffers from occasional lapses. Our basic model set comprises 24 observer models; we also considered several variants of these models that are described in the text. All main factors are explained in the following sections and summarized in Table 2. The term 'model component' is used through the text to indicate both factors and levels.

Decision making: Standard BDT observer ('BDT'). The 'decision-making' factor comprises model components with different assumptions about the decision process. We start describing the 'baseline' Bayesian observer model, BDT, that follows standard BDT. Suboptimality, in this case, emerges if the observer's internal estimates of the parameters of the task take different values from the true ones. As all subsequent models are variations of the BDT observer we describe this model in some detail.

On each trial the information available to the observer is comprised of the 'prior' distribution $p_{\text {prior }}(x)$, the cue position $x_{\text {cue }}$, and the distance $d_{\text {cue }}$ of the cue from the target line, which is a proxy for cue variability, $\sigma_{\text {cue }} \equiv \sigma\left(d_{\text {cue }}\right)$. The posterior distribution of target location, $x$, is computed by multiplying together the prior with the likelihood function. For the moment we assume the observer has perfect access to the displayed cue location and prior, and knowledge that cue variability is normally distributed. However, we allow the observer's estimate of the variance of the likelihood $\left(\hat{\sigma}_{b c u}^{2}\right.$ and $\left.\hat{\sigma}_{b i g h}^{2}\right)$ to mismatch the actual variance $\left(\sigma_{b c u}^{2}\right.$ and $\sigma_{b i g h}^{2}$ ). Therefore the posterior is given by:

$$
p_{\text {post }}(x)=p_{\text {post }}\left(x \mid x_{\text {cue }, \sigma} d_{\text {cue }, p_{\text {prior }}}\right) \propto p_{\text {prior }}(x) \mathcal{N}\left(x_{\text {cue }} \mid x, \hat{\sigma}_{\text {cue }}^{2}\right)
$$

where $\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ denotes a normal distribution with mean $\mu$ and variance $\sigma^{2}$.

In general, for any given trial, the choice the subject makes (desired pointing location for $x$ ) can be a probabilistic one, leading to a 'target choice' distribution. However, for standard BDT, the choice is deterministic given the trial parameters, leading to a 'target choice' distribution that collapses to a delta function:

$$
p_{\text {target }}\left(x \mid x_{\text {cue }, \sigma} d_{\text {cue }, p_{\text {prior }}}\right)=\delta\left[x-x^{*}\left(x_{\text {cue }, \sigma} d_{\text {cue }, p_{\text {prior }}}\right)\right]
$$

where $x^{*}$ is the 'optimal' target position that minimizes the observer's expected loss. The explicit task in our experiment is to

---

#### Page 8

> **Image description.** The image consists of four scatter plots arranged horizontally. Each plot displays the relationship between the standard deviation of the posterior distribution (sigma_post) on the x-axis and the mean reaction time on the y-axis. All plots use the same axes scales, with the x-axis ranging from 0.04 to 0.12 "screen units" and the y-axis ranging from 0.25 to 0.35 seconds.
>
> Each plot contains a thick, solid purple line representing the average reaction times. Two thinner, solid purple lines are plotted above and below the average line, likely representing the standard error. A dashed purple line represents a robust linear fit to the reaction time data.
>
> The plots are titled as follows:
>
> 1. "Gaussian training (n = 24)"
> 2. "Gaussian test (n = 8)"
> 3. "Unimodal test (n = 8)"
> 4. "Bimodal test (n = 8)"
>
> The x-axis is labeled "sigma_post (screen units)" and the y-axis is labeled "Mean reaction time (s)".

Figure 6. Average reaction times as a function of the SD of the posterior distribution. Each panel shows the average reaction times (mean $\pm$ SE across subjects) for a given session as a function of the SD of the posterior distribution, $\sigma_{\text {post }}$ (individual data were smoothed with a kernel regression estimate, see Methods). Dashed lines are robust linear fits to the reaction times data. For all sessions the slope of the linear regression is significantly different than zero $\left(p<10^{-3}\right)$.

doi:10.1371/journal.pcbi. 1003661 .g006
place the target within the radius of the cursor, which is equivalent to a 'square well' loss function with a window size equal to the diameter of the cursor. For computational reasons, in our observer models we approximate the square well loss with an inverted Gaussian (see Methods) that best approximates the square well, with fixed SD $\sigma_{\hat{\tau}}=0.027$ screen units (see Section 3 in Text S1).

In our experiment all priors were mixtures of $m$ (mainly 1 or 2) Gaussian distributions of the form $p_{\text {prior }}(x)=$ $\sum_{i=1}^{m} \pi_{i} \mathcal{N}\left(x \mid \mu_{i}, \sigma_{i}^{2}\right)$, with $\sum_{i=1}^{m} \pi_{i}=1$. It follows that the expected loss is a mixture of Gaussians itself, and the optimal target that minimizes the expected loss takes the form (see Methods for details):

$$
\begin{aligned}
x^{*}\left(x_{\text {cue }}\right) & =x^{*}\left(x_{\text {cue }} ; d_{\text {cue }, p_{\text {prior }}}\right) \\
& =\arg \min _{x^{*}}\left\{-\sum_{i=1}^{m} \gamma_{i} \mathcal{N}\left(x^{*} \mid v_{i}, \tau_{i}^{2}+\sigma_{i}^{2}\right)\right\}
\end{aligned}
$$

where we defined:

$$
\begin{gathered}
\gamma_{i} \equiv \pi_{i} \mathcal{N}\left(x_{\text {cue }} \mid \mu_{i}, \sigma_{i}^{2}+\tilde{\sigma}_{\text {cue }}^{2}\right) \\
v_{i} \equiv \frac{\mu_{i} \tilde{\sigma}_{\text {cue }}^{2}+x_{\text {cue }} \sigma_{i}^{2}}{\sigma_{i}^{2}+\tilde{\sigma}_{\text {cue }}^{2}}, \quad \tau_{i}^{2} \equiv \frac{\sigma_{i}^{2} \tilde{\sigma}_{\text {cue }}^{2}}{\sigma_{i}^{2}+\tilde{\sigma}_{\text {cue }}^{2}}
\end{gathered}
$$

For a single-Gaussian prior $(m=1), p_{\text {prior }}=\mathcal{N}\left(x \mid \mu_{1}, \sigma_{1}^{2}\right)$ and the posterior distribution is itself a Gaussian distribution with mean $\mu_{\text {post }}=v_{1}$ and variance $\sigma_{\text {post }}^{2}=\tau_{1}^{2}$, so that $x^{*}\left(x_{\text {cue }}\right)=\mu_{\text {post }}$.

We assume that the subject's response is corrupted by motor noise, which we take to be normally distributed with SD $\sigma_{\text {motor. }}$. By convolving the target choice distribution (Eq. 3) with motor noise we obtain the final response distribution:

$$
p\left(r \mid x_{\text {cue },} d_{\text {cue }, p_{\text {prior }}}\right)=\mathcal{N}\left(r \mid x^{*}\left(x_{\text {cue }}\right), \sigma_{\text {motor }}^{2}\right)
$$

The calculation of the expected loss in Eq. 4 does not explicitly take into account the consequences of motor variability, but this

Table 2. Set of model factors.

| Label |                 Model description                 | $\#$ parameters |                                       Free parameters $\left(\hat{v}_{H}\right)$                                       |
| :---: | :-----------------------------------------------: | :-------------: | :--------------------------------------------------------------------------------------------------------------------: |
|  BDT  |               Decision making: BDT                |        4        |     $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {live }},\left(\tilde{\sigma}_{\text {high }} \times 2\right)$     |
|  PPM  |  Decision making: Posterior probability matching  |        4        |     $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {live }},\left(\tilde{\sigma}_{\text {high }} \times 2\right)$     |
|  SPK  |       Decision making: Stochastic posterior       |        6        | $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {live }},\left(\tilde{\sigma}_{\text {high }}, \kappa\right) \times 2$ |
|  PSA  |  Decision making: Posterior sampling average (')  |        6        | $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {live }},\left(\tilde{\sigma}_{\text {high }}, \kappa\right) \times 2$ |
|   S   |               Cue-estimation noise                |      $+2$       |                                     $+\left(\Delta_{\text {cue }} \times 2\right)$                                     |
|   P   |              Prior estimation noise               |      $+2$       |                                     $+\left(\eta_{\text {prior }} \times 2\right)$                                     |
|   L   |                       Lapse                       |      $+2$       |                                                 $+(\lambda \times 2)$                                                  |
|  MV   |     Gaussian approximation: mean/variance (')     |        -        |                                                           -                                                            |
|  LA   | Gaussian approximation: Laplace approximation (') |        -        |                                                           -                                                            |

Table of all major model factors, identified by a label and short description. An observer model is built by choosing a model level for decision making and then optionally adding other components. For each model component the number of free parameters is specified. $\mathrm{A}^{\prime} \times 2^{\prime}$ means that a parameter is specified independently for training and test sessions; otherwise parameters are shared across sessions. See main text and Methods for the meaning of the various parameters. (') These additional components appear in the comparison of alternative models of decision making. doi:10.1371/journal.pcbi. 1003661 .t002

---

#### Page 9

approximation has minimal effects on the inference (see Discussion).

The behavior of observer model BDT is completely described by Eqs. 4, 5 and 6. This observer model is subjectively Bayes optimal; the subject applies BDT to his or her internal model of the task, which might be wrong. Specifically, the observer will be close to objective optimality only if his or her estimates for the likelihood parameters, $\hat{\sigma}_{\text {low }}$ and $\hat{\sigma}_{\text {high }}$, match the true likelihood parameters of the task ( $\sigma_{\text {low }}$ and $\sigma_{\text {high }}$ ). As extreme cases, if $\hat{\sigma}_{\text {low }}, \hat{\sigma}_{\text {high }} \rightarrow 0$ the BDT observer will ignore the prior and only use the noiseless cues (cue-only observer model; dashed lines in Figure 5), whereas for $\hat{\sigma}_{\text {low }}, \hat{\sigma}_{\text {high }} \rightarrow \infty$ the observer will use only probabilistic information contained in the priors (prior-only observer model; dotted lines in Figure 5).

Decision making: Noisy decision makers ('SPK' and 'PPM'). An alternative to BDT is a family of observer models in which the decision-making process is probabilistic, either because of noise in the inference or stochasticity in action selection. We model these various sources of variability without distinction as stochastic computations that involve the posterior distribution.

We start our analysis by considering a specific model, SPK (stochastic posterior, $\kappa$-power), in which the observer minimizes the expected loss (Eq. 4) under a noisy, approximate representation of the posterior distribution, as opposed to the deterministic, exact posterior of BDT (Figure 7a and 7d); later we will consider other variants of stochastic computations. As before, we allow the SD of the likelihoods, $\hat{\sigma}_{\text {low }}$ and $\hat{\sigma}_{\text {high }}$, to mismatch their true values. For mathematical and computational tractability, we do not directly simulate the noisy inference during the model comparison. Instead, we showed that different ways of introducing stochasticity in the inference process - either by adding noise to an explicit representation of the observer's posterior (Figure 7b and 7e), or by building a discrete approximation of the posterior via sampling (Figure 7c and 7f) - induce variability in the target choice that is well approximated by a power function of the posterior distribution itself; see Text S2 for details.

We, therefore, use the power function approximation with power $\kappa$ - hence the name of the model - to simulate the effects of a stochastic posterior on decision making, without committing to a specific interpretation. The target choice distribution in model SPK takes the form:

$$
p_{\text {target }}\left(x \mid x_{\text {cue }, d_{\text {cue }}, p_{\text {prior }}}\right) \propto\left[p_{\text {prior }}(x)\right]^{\kappa}
$$

where the power exponent $\kappa \geq 0$ is a free parameter inversely related to the amount of variability. Eq. 7 is convolved with motor noise to give the response distribution. The power function conveniently interpolates between a posterior-matching strategy (for $\kappa=1$ ) and a maximum a posteriori (MAP) solution $(\kappa \rightarrow \infty)$.

We consider as a separate factor the specific case in which the power exponent $\kappa$ is fixed to 1 , yielding a posterior probability matching observer, PPM, that takes action according to a single draw from the posterior distribution $[25,26]$.

Observer models with cue-estimation sensory noise ('S'). We consider a family of observer models, S, in which we drop the assumption that the observer perfectly knows the horizontal position of the cue. We model sensory variability by
adding Gaussian noise to the internal measurement of $x_{\text {cue }}$, which we label $\zeta_{\text {cue }}$ :

$$
\begin{aligned}
p\left(\zeta_{\text {cue }} \mid x_{\text {cue }, d_{\text {cue }}}\right)= & \mathcal{N}\left(\zeta_{\text {cue }} \mid x_{\text {cue }, \Sigma^{2}}\left(d_{\text {cue }}\right)\right) \\
& \text { with } \Sigma^{2}\left(d_{\text {cue }}\right) \in\left\{\Sigma_{\text {low }}^{2}, \Sigma_{\text {high }}^{2}\right\}
\end{aligned}
$$

where $\Sigma_{\text {low }}^{2}, \Sigma_{\text {high }}^{2}$ represent the variances of the estimates of the position of the cue, respectively for low-noise (short-distance) and high-noise (long-distance) cues. According to Weber's law, we assume that the measurement error is proportional to the distance from the target line $d_{\text {cue }}$, so that the ratio of $\Sigma_{\text {high }}$ to $\Sigma_{\text {low }}$ is equal to the ratio of $d_{\text {long }}$ to $d_{\text {short }}$, and we need to specify only one of the two parameters $\left(\Sigma_{\text {high }}\right)$. Given that both the cue variability and the observer's measurement variability are normally distributed, their combined variability will still appear to the observer as a Gaussian distribution with variance $\hat{\sigma}_{\text {cue }}^{2}+\Sigma_{\text {cue }}^{2}$, assuming independence. Therefore, the observer's internal model of the task is formally identical to the description we gave before by replacing $x_{\text {cue }}$ with $\zeta_{\text {cue }}$ in Eq. 2 (see Methods). Since the subject's internal measurement is not accessible during the experiment, the observed response probability is integrated over the hidden variable $\zeta_{\text {cue }}$ (Eq. 18 in Methods). A model with cue-estimation sensory noise ('S') tends to the equivalent observer model without noise for $\Sigma_{\text {cue }} \rightarrow 0$.

Observer models with noisy estimation of the prior ('P'). We introduce a family of observer models, P , in which subjects have access only to noisy estimates of the parameters of the prior, $p_{\text {prior }}$. For this class of models we assume that estimation noise is structured along a task-relevant dimension.

Specifically, for Gaussian priors we assume that the observers take a noisy internal measurement of the SD of the prior, $\hat{\sigma}_{\text {prior }}$, which according to Weber's law follows a log-normal distribution:

$$
p\left(\hat{\sigma}_{\text {prior }} \mid \sigma_{\text {prior }}\right)=\log \mathcal{N}\left(\hat{\sigma}_{\text {prior }} \mid \sigma_{\text {prior }, \eta_{\text {prior }}^{2}}\right)
$$

where $\sigma_{\text {prior }}$, the true SD, is the log-scale parameter and $\eta_{\text {prior }} \geq 0$ is the shape parameter of the log-normally distributed measurement (respectively mean and SD in log space). We assume an analogous form of noise on the width of the platykurtic prior in the unimodal session. Conversely, we assume that for priors that are mixtures of two Gaussians the main source of error stems from assessing the relative importance of the two components. In this case we add lognormal noise to the weights of each component, which we assume to be estimated independently:

$$
p\left(\hat{\pi}_{i} \mid \pi_{i}\right)=\log \mathcal{N}\left(\hat{\pi}_{i} \mid \pi_{i}, \eta_{\text {prior }}^{2}\right) \quad \text { for } i=1,2
$$

where $\pi_{i}$ are the true mixing weights and $\eta_{\text {prior }}$ is the noise parameter previously defined. Note that Eq. 10 is equivalent to adding normal noise with SD $\sqrt{2} \eta_{\text {prior }}$ to the log weights ratio in the 'natural' log odds space [27].

The internal measurements of $\hat{\sigma}_{\text {prior }}$ (or $\hat{\pi}_{i}$ ) are used by the observer in place of the true parameters of the priors in the inference process (e.g. Eq. 5). Since we cannot measure the internal measurements of the subjects, the actual response probabilities are computed by integrating over the unobserved values of $\hat{\sigma}_{\text {prior }}$ or $\hat{\pi}_{i}$ (see Methods). Note that for $\eta_{\text {prior }} \rightarrow 0$ an observer model with prior noise ('P') tends to its corresponding version with no noise.

---

#### Page 10

> **Image description.** This image contains six plots arranged in a 2x3 grid, labeled a through f. Each plot displays data related to decision-making with stochastic posterior distributions.
>
> **Plots a, b, and c:**
>
> - Each plot has an x-axis labeled from -0.5 to 0.5 and a y-axis labeled "Expected loss" and "Probability".
> - A black line represents the posterior distribution, with the area under the curve filled in gray in plot a. In plot b, the black line is jagged, representing a noisy posterior. In plot c, the black line is replaced by vertical gray bars, representing a sample-based posterior. A faint gray line shows the original posterior distribution in plots b and c.
> - A purple line shows the expected loss.
> - A blue arrow labeled "x\*" points from the x-axis to the expected loss curve, indicating the subjectively optimal target.
>
> **Plots d, e, and f:**
>
> - Each plot has an x-axis labeled "Target position (screen units)" ranging from -0.5 to 0.5 and a y-axis labeled "p_target(x)".
> - A gray line represents the original posterior distribution.
> - A blue line represents the distribution of target choices. In plot d, this is a single vertical line. In plots e and f, it's a curve similar to the gray posterior distribution.
> - A dashed red line in plots e and f represents a power function of the posterior distribution.
>
> **Titles:**
>
> - Plot a is titled "Standard BDT posterior".
> - Plot b is titled "Noisy posterior".
> - Plot c is titled "Sample-based posterior".

Figure 7. Decision making with stochastic posterior distributions. a-c: Each panel shows an example of how different models of stochasticity in the representation of the posterior distribution, and therefore in the computation of the expected loss, may affect decision making in a trial. In all cases, the observer chooses the subjectively optimal target $x^{*}$ (blue arrow) that minimizes the expected loss (purple line; see Eq. 4) given his or her current representation of the posterior (black lines or bars). The original posterior distribution is showed in panels b-f for comparison (shaded line). a: Original posterior distribution. b: Noisy posterior: the original posterior is corrupted by random multiplicative or Poisson-like noise (in this example, the noise has caused the observer to aim for the wrong peak). c: Sample-based posterior: a discrete approximation of the posterior is built by drawing samples from the original posterior (grey bars; samples are binned for visualization purposes). d-f: Each panel shows how stochasticity in the posterior affects the distribution of target choices $p_{\text {target }}(x)$ (blue line). d: Without noise, the target choice distribution is a delta function peaked on the minimum of the expected loss, as per standard BDT. e: On each trial, the posterior is corrupted by different instances of noise, inducing a distribution of possible target choices $p_{\text {target }}(x)$ (blue line). In our task, this distribution of target choices is very well approximated by a power function of the posterior distribution, Eq. 7 (red dashed line); see Text S2 for details. f: Similarly, the target choice distribution induced by sampling (blue line) is fit very well by a power function of the posterior (red dashed line). Note the extremely close resemblance of panels e and f (the exponent of the power function is the same).

doi:10.1371/journal.pcbi. 1003661 . g007

A different type of measurement noise on the the prior density is represented by 'unstructured', pointwise noise which can be shown to be indistinguishable from noise in the posterior under certain assumptions (see Text S2).

Observer models with lapse ( $\mathbf{L}^{\prime}$ ). It is possible that the response variability exhibited by the subjects could be simply explained by occasional lapses. Observer models with a lapse term are common in psychophysics to account for missed stimuli and additional variability in the data [28]. According to these models, in each trial the observer has a typically small, fixed probability $0 \leq \lambda \leq 1$ (the lapse rate) of making a choice from a lapse probability distribution instead of the optimal target $x^{*}$. As a representative lapse distribution we choose the prior distribution (prior-matching lapse). The target choice for an observer with lapse has distribution:

$$
\begin{aligned}
p_{\text {target }}^{\text {lapse }}(x \mid x_{\text {cise }, d_{\text {cise }, p_{\text {prior }}})= & (1-\lambda) \cdot p_{\text {target }}\left(x \mid x_{\text {cise }, d_{\text {cise }, p_{\text {prior }}})\right. \\
& \left.+\lambda \cdot p_{\text {prior }}(x)\right.
\end{aligned}
$$

where the first term in the right hand side of the equation is the target choice distribution (either Eq. 3 or Eq. 7, depending on the decision-making factor), weighted by the probability of not making a lapse, $1-\lambda$. The second term is the lapse term, with probability $\lambda$, and it is clear that the observer model with lapse ( $\mathrm{L}^{\prime}$ ) reduces to an observer with no lapse in the limit $\lambda \rightarrow 0$. Eq. 11 is then convolved with motor noise to provide the response distribution. We also tested a lapse model in which the lapse distribution was uniform over the range of the displayed prior distribution.

Observer models with uniform lapse performed consistently worse than the prior-matching lapse model, so we only report the results of the latter.

## Model comparison

For each observer model $M$ and each subject's dataset we evaluated the posterior distribution of parameters $p\left(\theta_{M} \mid \mathrm{data}\right)$, where $\theta_{M}$ is in general a vector of model-dependent parameters (see Table 2). Each subject's dataset comprised of two sessions (training and test), for a total of about 1200 trials divided in 32 distinct conditions ( 8 priors $\times 2$ noise levels $\times 2$ sessions). In general, we assumed subjects shared the motor parameter $\sigma_{\text {motor }}$ across sessions. We also assumed that from training to test sessions people would use the same high-noise to low-noise ratio between cue variability $\left\langle\hat{\sigma}_{\text {high }} / \hat{\sigma}_{\text {low }}\right\rangle$; so only one cue-noise parameter $\left\langle\hat{\sigma}_{\text {high }}\right\rangle$ needed to be specified for the test session. Conversely, we assumed that the other noise-related parameters, if present ( $\kappa, \Sigma_{\text {high }}, \eta_{\text {prior }}$, $\lambda$ ), could change freely between sessions, reasoning that additional response variability can be affected by the presence or absence of feedback, or as a result of the difference between training and test distributions. These assumptions were validated via a preliminary model comparison (see Section 5 in Text S1). Table 2 lists a summary of observer models and their free parameters.

The posterior distributions of the parameters were obtained through a slice sampling Monte Carlo method [29]. In general, we assumed noninformative priors over the parameters except for motor noise parameter $\sigma_{\text {motor }}$ and cue-estimation sensory noise parameter $\Sigma_{\text {high }}$ (when present), for which we determined a

---

#### Page 11

reasonable range of values through an independent experiment (see Methods and Text S3). Via sampling we also computed for each dataset a measure of complexity and goodness of fit of each observer model, the Deviance Information Criterion (DIC) [30], which we used as an approximation of the marginal likelihood to perform model comparison (see Methods).

We compared observer models according to a hierarchical Bayesian model selection (BMS) method that treats subjects and models as random effects [31]. That is, we assumed that multiple observer models could be present in the population, and we computed how likely it is that a specific model (or model level within a factor) generated the data of a randomly chosen subject, given the model evidence represented by the subjects' DIC scores (see Methods for details). As a Bayesian metric of significance we used the exceedance probability $P^{*}$ of one model (or model level) being more likely than any other model (or model levels within a factor). In Text S1 we report instead a classical (frequentist) analysis of the group difference in DIC between models (GDIC), which assumes that all datasets have been generated by the same unknown observer model. In spite of different assumptions, BMS and GDIC agree on the most likely observer model, validating the robustness of our main findings. The two approaches exhibit differences with respect to model ranking, due to the fact that, as a 'fixed effect' method, GDIC does not account for group heterogeneity and outliers [31] (see Section 4 in Text S1 for details). Finally, we assessed the impact of each factor on model performance by computing the average change in DIC associated with a given component.

Results of model comparison. Figure 8 shows the results of the BMS method applied to our model set. Figure 8a shows the model evidence for each individual model and subject. For each subject we computed the posterior probability of each observer model using DIC as an approximation of the marginal likelihood (see Methods). We calculated model evidence as the Bayes factor (posterior probability ratio) between the subject's best model and a given model. In the graph we report model evidence in the same scale as DIC, that is as twice the log Bayes factor. A difference of more than 10 in this scale is considered very strong evidence [32]. Results for individual subjects show that model SPK-P-L (stochastic posterior with estimation noise on the prior and lapse) performed consistently better than other models for all conditions. A minority of subjects were also well represented by model SPK-P (same as above, but without the lapse component). All other models performed significantly worse. In particular, note that the richer SPK-S-P-L model was not supported, suggesting that that sensory noise on estimation of cue location was not needed to explain the data. Figure 8b confirms these results by showing the estimated probability of finding a given observer model in the population (assuming that multiple observer models could be present). Model SPK-P-L is significantly more represented $(P=0.72$; exceedance probability $P^{*}>0.999$ ), followed by model SPK-P $(P=0.10)$. For all other models the probability is essentially the same at $P<0.01$. The probability of single model factors reproduced an analogous pattern (Figure 8c). The majority of subjects (more than $80 \%$ in each case) are likely to use a stochastic decision making (SPK), to have noise in the estimation of the priors ( P ), and lapse ( L ). Only a minority ( $10 \%$ ) would be described by an observer model with sensory noise in estimation of the cue. The model comparison yielded similar results, although with a more graded difference between models, when looking directly at DIC scores (see Section 4 in Text S1; lower is better).

To assess in another way the relative importance of each model component in determining the performance of a model, we measured the average contribution to DIC of each model level
within a factor across all tested models (Figure 4 in Text S1). In agreement with our previous findings, the lowest DIC (better score) in decision making is obtained by observer models containing the SPK factor. BDT increases (i.e. worsens) average DIC scores substantially (difference in DIC, $\Delta \mathrm{DIC}=173 \pm 14$; mean $\pm$ SE across subjects) and PPM has devastating effects on model performance ( $\Delta \mathrm{DIC}=422 \pm 72$ ), where 10 points of $\Delta \mathrm{DIC}$ may already be considered a strong evidence towards the model with lower DIC [30]. Regarding the other factors ( $\mathrm{S}, \mathrm{P}, \mathrm{L}$ ) we found that in general lacking a factor increases DIC (worse model performance; see Section 4 in Text S1 for discussion about factor S). Overall, this analysis confirms the strong impact that an appropriate modelling of variability has on model performance (see Section 4 in Text S1 for details).

We performed a number of analyses on an additional set of observer models to validate the finding that model SPK-P-L provides the best explanation for the data in our model set.

Firstly, in all the observer models described so far the subjects' parameters of the likelihood, $\hat{\sigma}_{l o w}$ and $\hat{\sigma}_{h i j g h}$, were allowed to vary. Preliminary analysis had suggested that observer models with mismatching likelihoods always outperformed models with true likelihood parameters, $\sigma_{\text {low }}$ and $\sigma_{\text {high }}$. We tested whether this was the case also with our current best model, or if we could assume instead that at least some subjects were using the true parameters. Model SPK-P-L-true performed considerably worse than its counterpart with mismatching likelihood parameters ( $P=0.01$ with $P^{*} \approx 1$ for the other model; $\Delta \mathrm{DIC}=178 \pm 33$ ), suggesting that mismatching likelihoods are invariably necessary to explain our subjects' data.

We then checked whether the variability of subjects' estimates of the priors may have arisen instead due to the discrete representation of the prior distribution in the experiment (see Figure 1d). We therefore considered a model SPK-D-L in which priors were not noisy, but the model component ' D ' replaces the continuous representations of the priors with their true discrete representation (a mixture of a hundred narrow Gaussians corresponding to the dots shown on screen). Model SPK-D-L performed worse than model SPK-P-L $(P=0.01$ with $P^{*} \approx 1$ for the other model; $\Delta \mathrm{DIC}=$ $145 \pm 25$ ) and, more interestingly, also worse than model SPK-L $(P=0.09$ with $P^{*} \approx 1$ for the other model; $\Delta \mathrm{DIC}=59 \pm 15)$. The discrete representation of the prior, therefore, does not provide a better explanation for subjects' behavior.

Lastly, we verified whether our subjects' behavior and apparent variability could be explained by a non-Bayesian iterative model applied to the training datasets. A basic iterative model failed to explain subjects' data (see Section 6 in Text S1 and Discussion).

In conclusion, all analyses identify as the main sources of subjects' suboptimal behavior the combined effect of both noise in estimating the shape of the 'prior' distributions and variability in the subsequent decision, plus some occasional lapses.

Comparison of alternative models of decision making. Our previous analyses suggest that subjects exhibit variability in decision making that conforms to some nontrivial transformation of the posterior distribution (such as a power function of the posterior, as expressed by model component SPK). We perform a second factorial model comparison that focusses on details of the decision-making process, by including additional model components that describe different transformations of the posterior. We consider in this analysis the following factors (in italic the additions):

1. Decision making (4 levels): Bayesian Decision Theory ('BDT'), stochastic posterior ('SPK'), posterior probability matching ('PPM'), posterior sampling-average ('PSA').

---

#### Page 12

> **Image description.** This image presents a model comparison between individual models, displayed in three panels (a, b, and c).
>
> **Panel a:** This panel is a heatmap-like grid.
>
> - **Axes:** The y-axis is labeled "Models" and lists various model configurations (e.g., "SPK," "BDT," "PPM") combined with factors like "S," "P," and "L." The x-axis is labeled "Subject number" and ranges from 1 to 24.
> - **Data Representation:** Each cell in the grid is colored according to a color scale on the right, representing the "2 log Bayes Factor." The scale ranges from 0 (dark red) to >50 (dark blue). The color of a cell indicates the model's evidence against the best model for that subject.
> - **Grouping:** The subjects are divided into three groups: "Gaussian group," "Unimodal group," and "Bimodal group," separated by vertical dashed lines. Above each group, numbers (1 or 2) are displayed, possibly indicating a ranking or category within the group.
> - **Column Headers:** Above the grid, there are column headers indicating model factors: "Decision making," "Cue noise," "Prior noise," and "Lapse."
>
> **Panel b:** This panel displays a horizontal bar chart.
>
> - **Axes:** The y-axis lists model configurations (e.g., "SPK-P-L," "SPK-P," "BDT-S-L"). The x-axis is labeled "Model probability for random subject" and ranges from 0 to 1.
> - **Data Representation:** Each bar represents the probability that a given model generated the data of a randomly chosen subject. The length of the bar corresponds to the probability value. The top bar is dark red and has three asterisks "\*\*\*" next to it.
>
> **Panel c:** This panel also displays a horizontal bar chart.
>
> - **Axes:** The y-axis lists "Model Factors" (SPK, BDT, PPM, ~S, S, ~P, P, ~L, L). The x-axis is labeled "Factor probability for random subject" and ranges from 0 to 1.
> - **Data Representation:** Each bar represents the probability that a given model level within a factor generated the data of a randomly chosen subject. The length of the bar corresponds to the probability value. The bars for SPK, ~S, ~P, and L are dark red and have three asterisks "\*\*\*" next to them.
>
> The asterisks in panels b and c likely indicate a significant exceedance probability. The brown bars in panels b and c represent the most supported models (or model levels within a factor).

Figure 8. Model comparison between individual models. a: Each column represents a subject, divided by test group (all datasets include a Gaussian training session), each row an observer model identified by a model string (see Table 2). Cell color indicates model's evidence, here displayed as the Bayes factor against the best model for that subject (a higher value means a worse performance of a given model with respect to the best model). Models are sorted by their posterior likelihood for a randomly selected subject (see panel b). Numbers above cells specify ranking for most supported models with comparable evidence (difference less than 10 in 2 log Bayes factor [32]). b: Probability that a given model generated the data of a randomly chosen subject. Here and in panel c, brown bars represent the most supported models (or model levels within a factor). Asterisks indicate a significant exceedance probability, that is the posterior probability that a given model (or model component) is more likely than any other model (or model component): $\left({ }^{* * *}\right) P^{*}>0.999$. c: Probability that a given model level within a factor generated the data of a randomly chosen subject. doi:10.1371/journal.pcbi. 1003661 . g008

2. Gaussian approximation of the posterior (3 levels): no approximation, mean/variance approximation ('MV') or Laplace approximation ('LA').
3. Lapse (2 levels): absent or present ('L').

Our extended model set comprises 18 observer models since some combinations of model factors lead to equivalent observer models. In order to limit the combinatorial explosion of models, in this factorial analysis we do not include model factors $S$ and $P$ that were previously considered, since our main focus here is on decision making (but see below). All new model components are explained in this section and summarized in Table 2.

Firstly, we illustrate an additional level for the decision-making factor. According to model PSA (posterior sampling-average), we assume that the observer chooses a target by taking the average of $\kappa \geq 1$ samples drawn from the posterior distribution [33]. This corresponds to an observer with a sample-based posterior that applies a quadratic loss function when choosing the optimal target. For generality, with an interpolation method we allow $\kappa$ to be a real number (see Methods).

We also introduce a new model factor according to which subjects may use a single Gaussian to approximate the full posterior. The mean/variance model (MV) assumes that subjects approximate the posterior with a Gaussian with matching loworder moments (mean and variance). For observer models that act according to BDT, model MV is equivalent to the assumption of a quadratic loss function during target selection, whose optimal target choice equals the mean of the posterior. Alternatively, a commonly used Gaussian approximation in Bayesian inference is the Laplace approximation (LA) [34]. In this case, the observer approximates the posterior with a single Gaussian centered on the mode of the posterior and whose variance depends on the local curvature at the mode (see Methods). The main difference of the Laplace approximation from other models is that the posterior is usually narrower, since it takes into account only the main peak.

Crucially, the predictions of these additional model components differ only if the posterior distribution is non-Gaussian; these observer models represent different generalizations of how a noisy decision process could affect behavior beyond the Gaussian case. Therefore we include in this analysis only trials in which the theoretical posterior distribution is considerably non-Gaussian (see Methods); this restriction immediately excludes from the analysis the training sessions and the Gaussian group, in which all priors and posteriors are strictly Gaussian.

Figure 9 shows the results of the BMS method applied to this model set. As before, we consider first the model evidence for each individual model and subject (Figure 9a). Results are slighly different depending on the session (unimodal or bimodal) but in both cases model SPK-L (stochastic posterior with lapse) performs consistently better than other tested models for all conditions. Only a couple of subjects are better described by a different approximation of the posterior (either PSA or SPK-MV-L). These results are summarized in Figure 9b, which shows the estimated probability that a given model would be responsible of generating the data of a randomly chosen subject. We show here results for both groups; a separate analysis of each group did not show qualitative differences. Model SPK-L is significantly more represented $\left(P=0.64\right.$; exceedance probability $\left.P^{*}>0.99\right)$, followed by model PSA $\left(P=0.10\right)$ and SPK-MV-L $\left(P=0.08\right)$. For all other models the probability is essentially the same at $P \approx 0.01$. The probability of single model factors reproduces the pattern seen before (Figure 9c). The majority of subjects (more than $75 \%$ in

---

#### Page 13

each case) are likely to use a stochastic decision making (SPK), to use the full posterior (no Gaussian approximations) and lapse (L).

The model comparison performed on group DIC scores (GDIC) obtained mostly similar results although with a more substantial difference between the unimodal group and the bimodal group (Figure 3 in Text S1). In particular, group DIC scores fail to find significant differences between distinct types of approximation of the posterior in the unimodal case. The reason is that for several subjects in the unimodal group differences between models are marginal, and GDIC does not have enough information to disambiguate between these models. Nonetheless, results in the bimodal case are non-ambigous, and overall the SPK-L model emerges again as the best description of subjects' behavior (see Section 4 in Text S1 for details).

As mentioned before, in order to limit model complexity we did not include model factors S and P in the current analysis. We can arguably ignore sensory noise in cue estimation, S , since it was already proven to have marginal effect on subjects' behavior, but this is not the case for noisy estimation of the prior, P. We need, therefore, to verify that our main results about decision making in the case of non-Gaussian posteriors were not affected by the lack of this factor. We compared the four most represented models of the current analysis (Figure 9b) augmented with the P factor: SPK-P-L, PSA-P, SPK-MV-P-L and PSA-P-L. Model SPK-P-L was still the most representative model ( $P=0.80$, exceedance probability $P^{\prime \prime}>0.99$ ), showing that model factor P does not affect our conclusions on alternative models of decision making. We also found that model SPK-P-L obtained more evidence than any other model tested in this section ( $P=0.72$, exceedance probability $P^{\prime \prime}>0.99$ ), in agreement with the finding of our first factorial model comparison.

Finally, even though the majority of subjects' datasets is better described by the narrow loss function of the task, a few of them support also observer models that subtend a quadratic loss. To explore this diversity, we examined an extended BDT model in which the loss width $\sigma_{\ell}$ is a free parameter (see Section 3 in Text S1). This model performed slightly better than a BDT model with fixed $\sigma_{\ell}$, but no better than the equivalent SPK model, so our findings are not affected.

In summary, subjects' variability in our task is compatible with them manipulating the full shape of the posterior corrupted by noise (SPK), and applying a close approximation of the loss function of the task. Our analysis marks as unlikely alternative models of decision making that use instead a quadratic loss or different low-order approximations of the posterior.

## Analysis of best observer model

After establishing model SPK-P-L as the 'best' description of the data among the considered observer models, we examined its properties. First of all, we inspected the posterior distribution of the model parameters given the data for each subject. In almost all cases the marginalized posterior distributions were unimodal with a well-defined peak. We therefore summarized each posterior distribution with a point estimate (a robust mean) with minor loss of generality; group averages are listed in Table 3. For the analyses in this section we ignored outlier parameter values that fell more than 3 SDs away from the group mean (this rule excluded at most one value per parameter). In general, we found a reasonable statistical agreement between parameters of different sessions, with some discrepancies in the unimodal test session only. In this section, inferred values are reported as mean $\pm$ SD across subjects.

The motor noise parameter $\sigma_{\text {motor }}$ took typical values of $(4.8 \pm 2.0) \cdot 10^{-3}$ screen units ( $\sim 1.4 \mathrm{~mm}$ ), somewhat larger on
average than the values found in the sensorimotor estimation experiment, although still in a reasonable range (see Text S3). The inferred amount of motor noise is lower than estimates from previous studies in reaching and pointing (e.g. [10]), but in our task subjects had the possibility to adjust their end-point position.

The internal estimates of cue variability for low-noise and highnoise cues ( $\hat{\sigma}_{l o w}$ and $\hat{\sigma}_{h i g h}$ ) were broadly scattered around the true values ( $\sigma_{l o w}=0.06$ and $\sigma_{h i g h}=0.14$ screen units). In general, individual values were in qualitative agreement with the true parameters but showed quantitative discrepancies. Differences were manifest also at the group level, as we found statistically significant disagreement for both low and high-noise cues in the unimodal test session ( $t$-test, $p<0.01$ ) and high-noise cues in the bimodal test session $(p<0.05)$. The ratio between the two likelihood parameters, $\hat{\sigma}_{h i g h} / \hat{\sigma}_{l o w}=2.00 \pm 0.54$, differed significantly from the true ratio, $\sigma_{h i g h} / \sigma_{l o w}=2.33(p<0.01)$.

A few subjects $(n=5)$ were very precise in their decision-making process, with a power function exponent $\kappa>20$. For the majority of subjects, however, $\kappa$ took values between 1.8 and 14 (median 6.4 ), corresponding approximately to an amount of decision noise of $\sim 7-55 \%$ of the variance of the posterior distribution (median $\sim 15 \%$ ). The range of exponents is compatible with values of $\kappa$ ( number of samples) previously reported in other experiments, such as a distance-estimation task [33] or 'intuitive physics' judgments [35]. In agreement with the results of our previous model comparison, the inferred exponents suggest that subjects' stochastic decision making followed the shape of a considerably narrower version of the posterior distribution $(\kappa \gg 1)$ which is not simply a form of posterior-matching $(\kappa=1)$.

The Weber's fraction of estimation of the parameters of the priors' density took typical values of $\eta_{\text {prior }}=0.48 \pm 0.19$, with similar means across conditions. These values denote quite a large amount of noise in estimating (or manipulating) properties of the priors. Nonetheless, such values are in qualitative agreeement with a density/numerosity estimation experiment in which a change of $\sim 40 \%$ in density or numerosity of a field of random dots was necessary for subjects to note a difference in either property [36]. Although the two tasks are too different to allow a direct quantitative comparison, the thresholds measured in [36] suggest that density/numerosity estimation can indeed be as noisy as we found.

Finally, even though we did not set an informative prior over the parameter, the lapse rate took reasonably low values as expected from a probability of occasional mistakes [28,37]. We found $\lambda=0.03 \pm 0.03$, and the inferred lapse rate averaged over training and test session was less than 0.06 for all but one subject.

We examined the best observer model's capability to reproduce our subjects' performance. For each subject and group, we generated 1000 datasets simulating the responses of the SPK-P-L observer model to the experimental trials experienced by the subject. For each simulated dataset, model parameters were sampled from the posterior distribution of the parameters given the data. For each condition (shape of prior and cue type) we then computed the optimality index and averaged it across simulated datasets. The model's 'postdictions' are plotted in Figure 10 as continuous lines (SE are omitted for clarity) and appear to be in good agreement with the data. Note that the postdiction is not exactly a fit since (a) the parameters are not optimized specifically to minimize performance error, and (b) the whole posterior distribution of the parameters is used and not just a 'best' point estimate. As a comparison, we also plotted in Figure 10 the postdiction for the best BDT observer model, BDT-P-L (dashed line). As the model comparison suggested, standard Bayesian Decision Theory fails to capture subjects' performance.

---

#### Page 14

> **Image description.** The image presents a comparison between alternative models of decision-making, displayed in three panels labeled a, b, and c.
>
> Panel a: This panel is a heatmap. The y-axis is labeled "Models" and lists various model configurations such as "SPK", "PSA", "PPM", and "BDT", sometimes combined with "MV," "LA," or "L." The x-axis is labeled "Subject number" and ranges from 9 to 24. The subjects are divided into two groups, "Unimodal group" (subjects 9-16) and "Bimodal group" (subjects 18-24), separated by a vertical dashed line. The cells in the heatmap are colored according to a color scale to the right, labeled "2 log Bayes Factor," ranging from blue (low values) to yellow and red (high values). Specific numerical values (1, 2, 3, 4, 5) are overlaid on some of the cells. The top of the panel has labels "Decision making," "Gaussian approx.," and "Lapse."
>
> Panel b: This panel is a horizontal bar chart. The y-axis lists specific model configurations like "SPK-L," "PSA," "SPK-MV-L," "PSA-L," "SPK," and "PPM-LA." The x-axis is labeled "Model probability for random subject" and ranges from 0 to 1. The bars are colored brown, and some bars have asterisks indicating significance levels (e.g., "\*\*").
>
> Panel c: This panel is another horizontal bar chart. The y-axis is labeled "Model Factors" and lists factors like "SPK," "PPM," "BDT," "-GA," "MV," "LA," "SPK," "PSA," "L," and "-L." The x-axis is labeled "Factor probability for random subject" and ranges from 0 to 1. The bars are colored either brown or blue, and some bars have asterisks indicating significance levels (e.g., "**," "\***").

Figure 9. Comparison between alternative models of decision making. We tested a class of alternative models of decision making which differ with respect to predictions for non-Gaussian trials only. a: Each column represents a subject, divided by group (either unimodal or bimodal test session), each row an observer model identified by a model string (see Table 2). Cell color indicates model's evidence, here displayed as the Bayes factor against the best model for that subject (a higher value means a worse performance of a given model with respect to the best model). Models are sorted by their posterior likelihood for a randomly selected subject (see panel b). Numbers above cells specify ranking for most supported models with comparable evidence (difference less than 10 in 2 log Bayes factor [32]). b: Probability that a given model generated the data of a randomly chosen subject. Here and in panel c, brown bars represent the most supported models (or model levels within a factor). Asterisks indicate a significant exceedance probability, that is the posterior probability that a given model (or model component) is more likely than any other model (or model component): $\left({ }^{* *}\right) P^{*}>0.99,\left({ }^{* * *}\right) P^{*}>0.999$. c: Probability that a given model level within a factor generated the data of a randomly chosen subject. Label "-GA" stands for no Gaussian approximation (full posterior). doi:10.1371/journal.pcbi. 1003661 . g009

For each subject and group (training and test) we also plot the mean optimality index of the simulated sessions against the optimality index computed from the data, finding a good correlation $\left(R^{2}=0.98\right.$; see Figure 11).

Lastly, to gain an insight on subjects' systematic response biases, we used our framework in order to nonparametrically reconstruct what the subjects' priors in the various conditions would look like $[2,3,8,9]$ (see Methods). Due to limited data per condition and computational constraints, we recovered the subjects' priors at the group level and for model SPK-L, without additional noise on the priors $\langle\mathrm{P}\rangle$. The reconstructed average priors for distinct test sessions are shown in Figure 12. Reconstructed priors display a
very good match with the true priors for the Gaussian session and show minor deviations in the other sessions. The ability of the model to reconstruct the priors - modulo residual idiosyncrasies is indicative of the goodness of the observer model in capturing subjects' sources of suboptimality.

## Discussion

We have explored human performance in probabilistic inference (a target estimation task) for different classes of prior distributions and different levels of reliability of the cues. Crucially, in our setup subjects were required to perform Bayesian

Table 3. Best observer model's estimated parameters.

| Session           | $\sigma_{\text {recon }}$     | $\hat{\sigma}_{\text {lim }}$ | $\hat{\sigma}_{\text {logit }}$ | $\kappa\left({ }^{*}\right)$ | $\eta$          | $\lambda$       |
| :---------------- | :---------------------------- | :---------------------------- | :------------------------------ | :--------------------------- | :-------------- | :-------------- |
| Gaussian training | $(4.8 \pm 2.0) \cdot 10^{-3}$ | $0.07 \pm 0.02$               | $0.13 \pm 0.07$                 | $7.67 \pm 4.33$              | $0.48 \pm 0.15$ | $0.03 \pm 0.02$ |
| Gaussian test     | $(5.7 \pm 2.9) \cdot 10^{-3}$ | $0.07 \pm 0.02$               | $0.14 \pm 0.07$                 | $7.31 \pm 3.83$              | $0.47 \pm 0.20$ | $0.02 \pm 0.02$ |
| Unimodal test     | $(6.3 \pm 4.8) \cdot 10^{-3}$ | $0.05 \pm 0.01$               | $0.08 \pm 0.02$                 | $4.01 \pm 2.77$              | $0.48 \pm 0.20$ | $0.04 \pm 0.02$ |
| Bimodal test      | $(4.0 \pm 1.1) \cdot 10^{-3}$ | $0.06 \pm 0.02$               | $0.11 \pm 0.03$                 | $6.38 \pm 2.17$              | $0.49 \pm 0.28$ | $0.04 \pm 0.04$ |
| True values       | -                             | $\sigma_{\text {lim }}=0.06$  | $\sigma_{\text {logit }}=0.14$  | -                            | -               | -               |

[^0]
[^0]: Group-average estimated parameters for the 'best' observer model (SPK-P-L), grouped by session (mean $\pm$ SD across subjects). For each subject, the point estimates of the parameters were computed through a robust mean of the posterior distribution of the parameter given the data. For reference, we also report the true noise values of the cues; $\sigma_{\text {lim }}$ and $\sigma_{\text {logit }}$. (.) We ignored values of $\kappa>20$.
doi:10.1371/journal.pcbi. 1003661 . t003

---

#### Page 15

computations with explicitly provided probabilistic information, thereby removing the need either for statistical learning or for memory and recall of a prior distribution. We found that subjects performed suboptimally in our paradigm but that their relative degree of suboptimality was similar across different priors and different cue noise. Based on a generative model of the task we built a set of suboptimal Bayesian observer models. Different methods of model comparison among this large class of models converged in identifying a most likely observer model that deviates from the optimal Bayesian observer in the following points: (a) a mismatching representation of the likelihood parameters, (b) a noisy estimation of the parameters of the prior, (c) a few occasional lapses, and (d) a stochastic representation of the posterior (such that the target choice distribution is approximated by a power function of the posterior).

# Human performance in probabilistic inference

Subjects integrated probabilistic information from both prior and cue in our task, but rarely exhibited the signature of full 'synergistic integration', i.e. a performance above that which could be obtained by using either the prior or the cue alone (see Figure 5). However, unlike most studies of Bayesian learning, on each trial in our study subjects were presented with a new prior. A previous study on movement planning with probabilistic information (and fewer conditions) similarly found that subjects violated conditions of optimality [23].
More interestingly, in our data the relative degree of suboptimality did not show substantial differences across distinct
classes of priors and noise levels of the cue (low-noise and highnoise). This finding suggests that human efficacy at probabilistic inference is only mildly affected by complexity of the prior per se, at least for the distributions we have used. Conversely, the process of learning priors is considerably affected by the class of the distribution: for instance, learning a bimodal prior (when it is learnt at all) can require thousands of trials [9], whereas mean and variance of a single Gaussian can be acquired reliably within a few hundred trials [11].
Within the same session, subjects' relative performance was influenced by the specific shape of the prior. In particular, for Gaussian priors we found a systematic effect of the variance subjects performed worse with wider priors, more than what would be expected by taking into account the objective decrease in available information. Interestingly, neither noise in estimation of the prior width (factor P) nor occasional lapses that follow the shape of the prior itself (factor L) are sufficient to explain this effect. Model postdictions of model BDT-P-L show large systematic deviations from subjects' performance in the Gaussian sessions, whereas the best model with decision noise, SPK-P-L, is able to capture subjects' behavior; see top left and top right panels in Figure 10. Moreover, the Gaussian priors recovered under model SPK-L match extremely well the true priors, furthering the role of the stochastic posterior in fully explaining subjects' performance with Gaussians. The crucial aspect of model SPK may be that decision noise is proportional to the width of the posterior, and not merely of the prior.

> **Image description.** The image consists of four bar charts arranged in a 2x2 grid. Each chart displays the "Optimality index" on the y-axis, ranging from 0.7 to 1.0, and the "Prior distribution" on the x-axis, with values from 1 to 8 (or a subset thereof). Each chart has two distinct sets of bars: one labeled "Low-noise cue" and the other "High-noise cue".
>
> - **General Layout:** The charts are organized as follows:
>
>   - Top Left: "Gaussian training (n = 24)"
>   - Top Right: "Gaussian test (n = 8)"
>   - Bottom Left: "Unimodal test (n = 8)"
>   - Bottom Right: "Bimodal test (n = 8)"
>
> - **Bar Chart Details:**
>
>   - The bars in the "Low-noise cue" group are colored in shades of red, transitioning from a darker red for the first bar to a lighter red.
>   - The bars in the "High-noise cue" group are colored in shades of blue, transitioning from a darker blue to a lighter blue.
>   - Each bar has a small black error bar extending vertically from its top.
>   - The x-axis labels are numerical, but their order varies in the "Unimodal test" chart.
>
> - **Overlaid Lines:**
>
>   - Each chart also contains two overlaid lines: a solid purple line labeled "Best stochastic model (SPK-P-L)" and a dashed black line labeled "Best BDT model (BDT-P-L)". These lines appear to represent model predictions or fits to the data represented by the bars.
>
> - **Text Elements:**
>   - The y-axis is labeled "Optimality index".
>   - The x-axis is labeled "Prior distribution".
>   - Each chart has a title indicating the type of distribution (Gaussian, Unimodal, Bimodal) and whether it's a training or test set, along with the sample size (n = ...).
>   - The legend at the top identifies the two overlaid lines.
>
> In summary, the image presents a comparison of optimality indices across different prior distributions, cue noise levels, and distribution types (Gaussian, Unimodal, Bimodal), with overlaid model predictions.

Figure 10. Model 'postdiction' of the optimality index. Each bar represents the group-averaged optimality index for a specific session, for each prior (indexed from 1 to 8 , see also Figure 2) and cue type, either low-noise cues (red bars) or high-noise cues (blue bars); see also Figure 5. Error bars are SE across subjects. The continuous line represents the 'postdiction' of the best suboptimal Bayesian observer model, model SPK-P-L; see 'Analysis of best observer model' in the text). For comparison, the dashed line is the 'postdiction' of the best suboptimal observer model that follows Bayesian Decision Theory, BDT-P-L.

doi:10.1371/journal.pcbi. 1003661 . g010

---

#### Page 16

> **Image description.** The image is a scatter plot comparing measured and simulated performance, specifically the optimality index.
>
> - **Axes:** The x-axis is labeled "Measured optimality index," and the y-axis is labeled "Simulated optimality index." Both axes range from approximately 0.65 to 1.0, with tick marks at intervals of 0.05.
> - **Data Points:** The plot contains numerous data points, represented by small circles. There are two distinct colors: yellow and cyan. According to the legend in the upper left corner, yellow dots represent "Training sessions," and cyan dots represent "Test sessions." The data points are clustered along a diagonal line.
> - **Diagonal Line:** A dashed black line runs diagonally across the plot, representing the line of equality where measured and simulated values are identical.
> - **R-squared Value:** In the lower right corner of the plot, the text "R² = 0.98" is displayed, indicating a high coefficient of determination.

Figure 11. Comparison of measured and simulated performance. Comparison of the mean optimality index computed from the data and the simulated optimality index, according to the 'postdiction' of the best observer model (SPK-P-L). Each dot represents a single session for each subject (either training or test). The dashed line corresponds to equality between observed and simulated performance. Model-simulated performance is in good agreement with subjects' performance ( $R^{2}=0.98$ ).

doi:10.1371/journal.pcbi. 1003661 .g011

In the unimodal test session, subjects' performance was positively correlated with the width of the main peak of the distribution. That is, non-Gaussian, narrow-peaked priors (such as priors 1 and 6 in Figure 12b) induced worse performance than broad and smooth distributions (e.g. priors 4 and 8). Subjects tended to 'mistrust' the prior, especially in the high-noise condition, giving excess weight to the cue $\left\langle\hat{\sigma}_{\text {high }}\right.$ is significantly lower than it should be; see Table 3), which can be also interpreted as an overestimation of the width of the prior. In agreement with this description, the reconstructed priors in Figure 12b show a general tendency to overestimate the width of the narrower peaks, as we found in a previous study of interval timing [8]. This behavior is compatible with a well-known human tendency of underestimating (or, alternatively, underweighting) the probability of occurrence of highly probable results and overestimating (overweighting) the frequency of rare events (see [27,38,39]). Similar biases in estimating and manipulating prior distributions may be explained with an hyperprior that favors more entropic and, therefore, smoother priors in order to avoid 'overfitting' to the environment [40].

# Modelling suboptimality

In building our observer models we made several assumptions. For all models we assumed that the prior adopted by observers in Eq. 2 corresponded to a continuous approximation of the probability density function displayed on screen, or a noisy estimate thereof. We verified that using the original discrete representation does not improve model performance. Clearly, subjects may have been affected by the discretization of the prior in other ways, but we assumed that such errors could be absorbed by other model components. We also assumed subjects quickly

> **Image description.** The image is a composite figure showing a series of probability density plots. It is divided into three columns labeled "a. Gaussian session", "b. Unimodal session", and "c. Bimodal session". Each column contains four rows, resulting in 12 individual plots.
>
> - **Overall Structure:** The plots are arranged in a 4x3 grid. Each plot is contained within a square frame. A number from 1 to 8 is placed in the top left corner of each plot.
>
> - **Axes:** All plots share the same x and y axes. The x-axis is labeled "Target position (screen units)" and ranges from -0.5 to 0.5, with tick marks at -0.5, 0, and 0.5. The y-axis is labeled "Prior probability density".
>
> - **Plot Content:** Each plot displays two overlapping curves: a black curve and a magenta curve. The area under the black curve is shaded in gray. The shape of the curves varies across the plots, reflecting different probability distributions.
>
>   - **Gaussian Session (Column a):** The plots in this column primarily show Gaussian-like distributions, with a single peak centered around 0.
>   - **Unimodal Session (Column b):** These plots show distributions with a single peak, but the shape is less symmetrical than the Gaussian distributions. Some distributions are skewed or have a wider peak.
>   - **Bimodal Session (Column c):** The plots in this column display bimodal distributions, characterized by two distinct peaks.
>
> - **Curve Overlap:** In most plots, the magenta curve closely follows the black curve, but there are some discrepancies. In some cases, the magenta curve appears to smooth out or slightly deviate from the black curve.

Figure 12. Reconstructed prior distributions. Each panel shows the (unnormalized) probability density for a 'prior' distribution of targets, grouped by test session, as per Figure 2. Purple lines are mean reconstructed priors (mean $\pm 1$ s.d.) according to observer model SPK-L. a) Gaussian session. Recovered priors in the Gaussian test session are very good approximations of the true priors (comparison between SD of the reconstructed priors and true SD: $R^{2}=0.94$ ). b) Unimodal session. Recovered priors in the unimodal test session approximate the true priors (recovered SD: $0.105 \pm 0.007$, true SD: 0.11 screen units) although with systematic deviations in higher-order moments (comparison between moments of the reconstructed priors and true moments: skewness $R^{2}=0.47$; kurtosis $R^{2}<0$ ). Reconstructed priors are systematically less kurtotic (less peaked, lighter-tailed) than the true priors. c) Bimodal session. Recovered priors in the bimodal test session approximate the true priors with only minor systematic deviations (recovered SD: $0.106 \pm 0.004$, true SD: 0.11 screen units; coefficient of determination between moments of the reconstructed priors and true moments: skewness $R^{2}=0.99$; kurtosis $R^{2}=0.80$ ).

doi:10.1371/journal.pcbi. 1003661 .g012

---

#### Page 17

acquired a correct internal model of the probabilistic structure of the task, through practice and feedback, although quantitative details (i.e. model parameters) could be mismatched with respect to the true parameters. Formally, our observer models were not 'actor' models in the sense that they did not take into account the motor error in the computation of the expected loss. However, this was with negligible loss of generality since the motor term has no influence on the inference of the optimal target for single Gaussians priors, and yields empirically negligible impact for other priors for small values of the motor error $\sigma_{\text {motor }}$ (as those measured in our task; see Text S3).

Suboptimality was introduced into our observer models in three main ways: (a) miscalibration of the parameters of the likelihood; (b) models of approximate inference; and (c) additional stochasticity, either on the sensory inputs or in the decision-making process itself. Motor noise was another source of suboptimality, but its contribution was comparably low.

Miscalibration of the parameters of the likelihood means that the subjective estimates of the reliability of the cues ( $\hat{\sigma}_{\text {low }}$ and $\hat{\sigma}_{\text {high }}$ ) could differ from the true values ( $\sigma_{\text {low }}$ and $\sigma_{\text {high }}$ ). In fact, we found slight to moderate discrepancies, which became substantial in some conditions. Previous studies have investigated whether subjects have (or develop) a correct internal estimate of relevant noise parameters (i.e. the likelihood) which may correspond to their own sensory or motor variability plus some externally injected noise. In several cases subjects were found to have a miscalibrated model of their own variability which led to suboptimal behavior [33,41-43], although there are cases in which subjects were able to develop correct estimates of such parameters $[10,44,45]$.

More generally, it could be that subjects were not only using incorrect parameters for the task, but built a wrong internal model or were employing approximations in the inference process. For our task, which has a relatively simple one-dimensional structure, we did not find evidence that subjects were using low-order approximations of the posterior distribution. Also, the capability of our models to recover the subjects' priors in good agreement with the true priors suggest that subjects' internal model of the task was not too discrepant from the true one.

Crucial element in all our models was the inclusion of extra sources of variability, in particular in decision making. Whereas most forms of added noise have a clear interpretation, such as sensory noise in the estimation of the cue location, or in estimating the parameters of the prior, the so-called 'stochastic posterior' deserves an extended explanation.

## Understanding the stochastic posterior

We introduced the stochastic posterior model of decision making, SPK, with two intuitive interpretations, that is a noisy posterior or a sample-based approximation (see Figure 7 and Text S2), but clearly any process that produces a variability in the target choice distribution that approximates a power function of the posterior is a candidate explanation. The stochastic posterior captures the main trait of decision noise, that is a variability that depends on the shape of the posterior [33], as opposed to other forms of noise that do not depend on the decision process. Outstanding open questions are therefore which kind of process could be behind the observed noise in decision making, and during which stage it arises, e.g. whether it is due to inference or to action selection [46].

A seemingly promising candidate for the source of noise in the inference is neuronal variability in the nervous system [47]. Although the noisy representation of the posterior distribution in Figure 7b through a population of units may be a simplistic
cartoon, the posterior could be encoded in subtler ways (see for instance [48]). However, neuronal noise itself may not be enough to explain the amount of observed variability (see Text S2). An extension of this hypothesis is that the noise may emerge since suboptimal computations magnify the underlying variability [49].

Conversely, another scenario is represented by the sampling hypothesis, an approximate algorithm for probabilistic inference which could be implemented at the neural level [19]. Our analysis ruled out an observer whose decision-making process consists in taking the average of $\kappa$ samples from the posterior - operation that implicitly assumes a quadratic loss function - showing that averaging samples from the posterior is not a generally valid approach, although differences can be small for unimodal distributions. More generally, the sampling method should always take into account the loss function of the task, which in our case is closer to a delta function (a MAP solution) rather than to a quadratic loss. Our results are compatible with a proper sampling approach, in which an empirical distribution is built out of a small number of samples from the posterior, and then the expected loss is computed from the sampled distribution [19].

As a more cognitive explanation, decision variability may have arisen because subjects adopted a probabilistic instead of deterministic strategy in action selection as a form of exploratory behavior. In reinforcement learning this is analogous to the implementation of a probabilistic policy as opposed to a deterministic policy, with a 'temperature' parameter that governs the amount of variability [50]. Search strategies have been hypothesized to lie behind suboptimal behaviors that appear random, such as probability matching [51]. While generic exploratory behavior is compatible with our findings, our analysis rejected a simple posterior-matching strategy $[25,26]$.

All of these interpretations assume that there is some noise in the decision process itself. However, the noise could emerge from other sources, without the necessity of introducing deviations from standard BDT. For instance, variability in the experiment could arise from lack of stationarity: dependencies between trials, fluctuations of subjects' parameters or time-varying strategies would appear as additional noise in a stationary model [52]. We explored the possibility of nonstationary behavior without finding evidence for strong effects of nonstationarity (see Section 6 in Text S1). In particular, an iterative (trial-dependent) non-Bayesian model failed to model the data in the training dataset better than the stochastic posterior model. Clearly, this does not exclude that different, possibly Bayesian, iterative models could explain the data better, but our task design with multiple alternating conditions and partial feedback should mitigate the effect of dependencies between trials, since each trial typically displays a different condition from the immediately preceding ones.

In summary, we show that a decision strategy that implements a 'stochastic posterior' that introduces variability in the computation of the expected loss has several theoretical and empirical advantages when modelling subjects' performance, demonstrating improvement over previous models that implemented variability only through a 'posterior-matching' approach or that implicitly assume a quadratic loss function (sampling-average methods).

## Methods

## Ethics statement

The Cambridge Psychology Research Ethics Committee approved the experimental procedures and all subjects gave informed consent.

---

#### Page 18

## Participants

Twenty-four subjects ( 10 male and 14 female; age range 18-33 years) participated in the study. All participants were naïve to the purpose of the study. All participants were right-handed according to the Edinburgh handedness inventory [53], with normal or corrected-to-normal vision and reported no neurological disorder. Participants were compensated for their time.

## Behavioral task

Subjects were required to reach to an unknown target given probabilistic information about its position. Information consisted of a visual representation of the a priori probability distribution of targets for that trial and a noisy cue about the actual target position.

Subjects held the handle of a robotic manipulandum (vBOT, [54]). The visual scene from a CRT monitor (Dell UltraScan P1110, 21-inch, 100 Hz refresh rate) was projected into the plane of the hand via a mirror (Figure 1a) that prevented the subjects from seeing their hand. The workspace origin, coordinates [0,0], was $\sim 35 \mathrm{~cm}$ from the torso of the subjects, with positive axes towards the right ( $x$ axis) and away from the subject ( $y$ axis). The workspace showed a home position ( 1.5 cm radius circle) at $[0,-15] \mathrm{cm}$ and a cursor ( 1.25 cm radius circle) that tracked the hand position.

On each trial 100 potential targets ( 0.1 cm radius dots) were shown around the target line at positions $\left[\mu_{i}, v_{j}\right]$, for $j=1, \ldots, 100$, where the $\mu_{i}$ formed a fixed discrete representation of the trialdependent 'prior' distribution $p_{\text {prior }}(x)$, obtained through a regular sample of the cdf (see Figure 1d), and the $v_{j}$ were small random offsets used to facilitate visualization ( $v_{j} \sim \operatorname{Uniform}(-0.3,0.3) \mathrm{cm}$ ). The true target was chosen by picking one of the potential targets at random with uniform probability. A cue ( 0.25 cm radius circle) was shown at position $\left[x_{\text {cue }},-d_{\text {cue }}\right]$. The horizontal position $x_{\text {cue }}$ provided a noisy estimate of the target position, $x_{\text {cue }}=x+c \sigma_{\text {cue }}$, with $x$ the true (horizontal) position of the target, $\sigma_{\text {cue }}$ the cue variability and $c$ a normal random variable with zero mean and unit variance. The distance of the cue from the target line, $d_{\text {cue }}$, was linearly related to the cue variability: cues distant from the target line were noisier than cues close to it. In our setup, the noise level $\sigma_{\text {cue }}$ could only either be low for 'short-distance' cues, $\sigma_{\text {low }}=1.8 \mathrm{~cm}\left(d_{\text {short }}=3.9 \mathrm{~cm}\right)$, or high for 'long-distance' cues, $\sigma_{\text {high }}=4.2 \mathrm{~cm}\left(d_{\text {long }}=9.1 \mathrm{~cm}\right)$. Both the prior distribution and cue remained on the screen for the duration of a trial.

After a 'go' beep, subjects were required to move the handle towards the target line, choosing an endpoint position such that the true target would be within the cursor radius. The manipulandum generated a spring force along the depth axis $\left(F_{y}=-5.0 \mathrm{~N} / \mathrm{cm}\right)$ for cursor positions past the target line, preventing subjects from overshooting. The horizontal endpoint position of the movement (velocity of the cursor less than $0.5 \mathrm{~cm} /$ s), after contact with the target line, was recorded as the subject's response $r$ for that trial.

At the end of each trial, subjects received visual feedback on whether their cursor encircled (a 'success') or missed the true target (partial feedback). On full feedback trials, the position of the true target was also shown ( 0.25 cm radius yellow circle). Feedback remained on screen for 1 s . Potential targets, cues and feedback then disappeared. A new trial started 500 ms after the subject had returned to the home position.

For simplicity, all distances in the experiment are reported in terms of standardized screen units (window width of 1.0), with $x \sigma[-0.5,0.5]$ and 0.01 screen units corresponding to 3 mm . In screen units, the cursor radius is 0.042 and the SD of noise for
short and long distance cues is respectively $\sigma_{\text {low }}=0.06$ and $\sigma_{\text {high }}=0.14$.

## Experimental sessions

Subjects performed one practice block in which they were familiarized with the task ( 64 trials). The main experiment consisted of a training session with Gaussian priors ( 576 trials) followed by a test session with group-dependent priors (576-640 trials). Sessions were divided in four runs. Subjects could take short breaks between runs and there was a mandatory 15 minutes break between the training and test sessions.

Each session presented eight different types of priors and two cue noise levels (corresponding to either 'short' or 'long' cues), for a total of 16 different conditions (36-40 trials per condition). Trials from different conditions were presented in random order. Depending on the session and group, priors belonged to one of the following classes (see Figure 2):

Gaussian priors. Eight Gaussian distributions with evenly spread SDs between 0.04 and 0.18 i.e. $\sigma_{\text {prior }} \in(0.04,0.06, \ldots, 0.18)$ screen units.

Unimodal priors. Eight unimodal priors with fixed SD $\sigma_{\text {prior }}=0.11$ and variable skewness and kurtosis. With the exception of platykurtic prior 4 , which is a mixture of 11 Gaussians, and prior 8 , which is a single Gaussian, all other priors were realized as mixtures of two Gaussians that locally maximize differential entropy for given values of the first four central moments. In the maximization we included a constraint on the SDs of the individual components so to prevent degenerate solutions $\left\langle 0.02 \leq \sigma_{i} \leq 0.2\right.$ screen units, for $i=1,2$ ). Skewness and excess kurtosis were chosen to represent various shapes of unimodal distributions, within the strict bounds that exist between skewness and kurtosis of a unimodal distribution [55]. The values of (skewness, kurtosis) for the eight distributions, in order of increasing differential entropy: 1: $(2,5) ; 2:(0,5) ; 3:(0.78,0) ; 4$ : $(0,-1) ; 5:(0.425,-0.5) ; 6:(0,1) ; 7:(0.5,0) ; 8:(0,0)$.

Bimodal priors. Eight (mostly) bimodal priors with fixed SD $\sigma_{\text {prior }}=0.11$ and variable separation and relative weight. The priors were realized as mixtures of two Gaussians with equal variance: $p_{\text {prior }}(x)=\pi \lambda^{2}\left(x \mid \mu_{1}, \sigma^{2}\right)+(1-\pi) \lambda^{2}\left(x \mid \mu_{2}, \sigma^{2}\right)$. Separation was computed as $d=\frac{\mu_{1}-\mu_{2}}{\sigma}$, and relative weight was defined as $w=\frac{\sigma}{1-\sigma}$. The values of (separation, relative weight) for the eight distributions, in order of increasing differential entropy: 1: $(5,1) ; 2:(4,3) ; 3:(4,2) ; 4:(4,5) ; 5:(4,1) ; 6:(3,1) ; 7:(2,1) ; 8$ : $(0,-)$ (the last distribution is a single Gaussian).

For all priors, the mean $\mu_{\text {prior }}$ was drawn from a uniform distribution whose bounds were chosen such that the extremes of the discrete representation would fall within the active screen window (the actual screen size was larger than the active window). Also, asymmetric priors had $50 \%$ probability of being flipped horizontally about the mean.

## Data analysis

Analysis of behavioral data. Data analysis was conducted in MATLAB 2010b (Mathworks, U.S.A.). To avoid edge artifacts in subjects' response, we discarded trials in which the cue position, $x_{\text {cue }}$, was outside the range of the discretized prior distribution (2691 out of 28672 trials: $9.4 \%$ ). We included these trials in the experimental session in order to preserve the probabilistic relationships between variables of the task.

For each trial, we recorded the response location $r$ and the reaction time (RT) was defined as the interval between the 'go' beep and the start of the subject's movement. For each subject and

---

#### Page 19

session we computed a nonlinear kernel regression estimate of the average RT as a function of the SD of the posterior distribution, $\sigma_{\text {post }}$. We only considered a range of $\sigma_{\text {post }}$ for which all subjects had a significant density of data points. Results did not change qualitatively for other measures of spread of the posterior, such as the exponential entropy [24].

All subjects' datasets are available online in Dataset S1.
Optimality index and success probability. We calculated the optimality index for each trial as the success probability for response $r, p_{\text {success }}(r)$, divided by the maximal success probability $p_{\text {success }}^{\prime}$, which we used to quantify performance of a subject (or an observer model). The optimality index of our subjects in the task is plotted in Figure 5 and success probabilities are shown in Figure 1 in Text S1.

The success probability $p_{\text {success }}(r)$ in a given trial represents the probability of locating the correct target according to the generative model of the task (independent of the actual position of the target). For a trial with cue position $x_{\text {cue }}$, cue noise variance $\sigma_{\text {cue }}^{2}$, and prior distribution $p_{\text {prior }}(x)$, the success probability is defined as:
$p_{\text {success }}(r)=$
$\int_{r-\frac{t}{2}}^{r+\frac{t}{2}}\left[\int \frac{1}{d x^{\prime} p_{\text {prior }}\left(x^{\prime}\right) \mathcal{N}\left(x^{\prime} \mid x_{\text {cue }}, \sigma_{\text {cue }}^{2}\right)} p_{\text {prior }}(x) \mathcal{N}\left(x \mid x_{\text {cue }}, \sigma_{\text {cue }}^{2}\right)\right] d x$
where the integrand is the posterior distribution according to the continuous generative model of the task and $f$ is the diameter of the cursor. Solving the integral in Eq. 12 for a generic mixture-ofGaussians prior, $p_{\text {prior }}(x)=\sum_{i=1}^{\mathrm{se}} \pi_{i} \mathcal{N}\left(x \mid \mu_{i}, \sigma_{i}^{2}\right)$, we obtain:
$p_{\text {success }}(r)=$
$\left(\sum_{j=1}^{m} \gamma_{j}\right)^{-1} \sum_{i=1}^{m} \frac{\gamma_{i}}{2}\left[\operatorname{erf}\left(\frac{r+\frac{t}{2}-v_{i}}{\sqrt{2} \tau_{i}}\right)-\operatorname{erf}\left(\frac{r-\frac{t}{2}-v_{i}}{\sqrt{2} \tau_{i}}\right)\right]$
where the symbols $\gamma_{i}, v_{i}$ and $\tau_{i}$ have been defined in Eq. 5. The maximal success probability is simply computed as $p_{\text {success }}^{\prime}=\max _{r} p_{\text {success }}(r)$.

Note that a metric based on the theoretical success probability is more appropriate than the observed fraction of successes for a given sample of trials, as the latter introduces additional error due to mere chance (the observed fraction of successes fluctuates around the true success probability with binomial statistics, and the error can be substantial for small sample size).

The priors for the Gaussian, unimodal and bimodal sessions were chosen such that the average maximal success probability of each class was about the same ( $\sim 51.5 \%$ ) making the task challenging and of equal difficulty across the task.

Computing the optimal target. According to Bayesian Decision Theory (BDT), the key quantity an observer needs to compute in order to make a decision is the (subjectively) expected loss for a given action. In our task, the action corresponds to a choice of a cursor position $x^{\prime}$, and the expected loss takes the form:

$$
\mathcal{E}\left[x^{\prime} ; p_{\text {post }}, \mathcal{L}\right]=\int p_{\text {post }}(x) \mathcal{L}\left(x^{\prime}, x\right) d x
$$

where $p_{\text {post }}(x)$ is the subject's posterior distribution of target position, described by Eq. 2, and the loss associated with choosing position $x^{\prime}$ when the target location is $x$ is represented by loss function $\mathcal{L}\left(x^{\prime}, x\right)$.

Our task has a clear 'hit or miss' structure that is represented by the square well function:

$$
\mathcal{L}_{\text {well }}\left(x^{\prime}, x ; \ell\right)=\left\{\begin{array}{cc}
-\frac{1}{\ell} & \text { for }\left|x^{\prime}-x\right|<\frac{\ell}{2} \\
0 & \text { otherwise }
\end{array}\right.
$$

where $x^{\prime}-x$ is the distance of the chosen response from the target, and $\ell$ is the size of the allowed window for locating the target (in the experiment, the cursor diameter). The square well loss allows for an analytical expression of the expected loss, but the optimal target still needs to be computed numerically. Therefore we make a smooth approximation to the square well loss represented by the inverted Gaussian loss:

$$
\mathcal{L}_{\text {Gauss }}\left(x^{\prime}, x ; \sigma_{\ell}\right)=-\mathcal{N}\left(x \mid x^{\prime}, \sigma_{\ell}^{2}\right)
$$

where the parameter $\sigma_{\ell}$ governs the scale of smoothed detection window. The Gaussian loss approximates extremely well the predictions of the square well loss in our task, to the point that performance under the two forms of loss is empirically indistinguishable (see Section 3 in Text S1). However, computationally the Gaussian loss is preferrable as it allows much faster calculations of optimal behavior.

For the decision process, BDT assumes that observers choose the 'optimal' target position $x^{*}$ that minimizes the expected loss:

$$
\begin{aligned}
x^{*} & =\arg \min _{x^{\prime}} \mathcal{E}\left[x^{\prime} ; p_{\text {post }}, \mathcal{L}_{\text {Gauss }}\right] \\
& =\arg \min _{x^{\prime}}\left\{-\sum_{i=1}^{m} \pi_{i}\left\{\mathcal{N}\left(x \mid \mu_{i}, \sigma_{i}^{2}\right) \mathcal{N}\left(x \mid x_{\text {cue }}, \hat{\sigma}_{\text {cue }}^{2}\left(d_{\text {cue }}\right)\right) \mathcal{N}\left(x \mid x^{\prime}, \sigma_{\ell}^{2}\right) d x\right\}\right.
\end{aligned}
$$

where we have used Eqs. 2, 14 and 16. With some algebraic manipulations, Eq. 17 can be reformulated as Eq. 4. Given the form of the expected loss, the solution of Eq. 4 is equivalent to finding the maximum (mode) of a Gaussian mixture model. In general no analytical solution is known for more than one model component $(m>1)$, so we implemented a fast and accurate numerical solution adapting the algorithm in [56].

Computing the response probability. The probability of observing response $r$ in a trial, $p(r \mid$ trial $)$ (e.g., Eq. 6) is the key quantity for our probabilistic modelling of the task. For basic observer models, $p(r \mid$ trial $)$ is obtained as the convolution between a Gaussian distribution (motor noise) and a target choice distribution in closed form (e.g. a power function of a mixture of Gaussians), such as in Eqs. 3, 7 and 11. Response probabilities are integrated over latent variables of model factor $\mathrm{S}\left\langle\hat{\zeta}_{\text {cue }}\right.$; see Eq. 8) and of model factor $\mathrm{P}\left\langle\log \hat{\sigma}_{\text {prior }}\right.$ and $\log \frac{\mathrm{s}}{\mathrm{s}}$; see Eqs. 9 and 10). Integrations were performed analytically when possible or otherwise numerically (trapz in MATLAB or Gauss-Hermite quadrature method for non-analytical Gaussian integrals [57]). For instance, the observed response probability for model factor S takes the shape:

$$
\begin{aligned}
p\left(r \mid x_{\text {cue }}, d_{\text {cue }}, p_{\text {prior }}\right)= & \int\left[\int \mathcal{N}\left(r\left(x, \sigma_{i n}^{2}\right) p_{\text {target }}\left(x \mid \zeta_{\text {cue }}, d_{\text {cue }}, p_{\text {prior }}\right) d x\right]\right. \\
& \left.\mathcal{N}\left(\zeta_{\text {cue }} \mid x_{\text {cue }}, \Sigma_{\text {cue }}^{2}\right) d \zeta_{\text {cue }}\right)
\end{aligned}
$$

where we are integrating over the hidden variables $\zeta_{\text {cue }}$ and $x$. The target choice distribution $p_{\text {target }}$ depends on the decision-making model component (see e.g. Eqs. 3 and 7). Without loss of generality, we assumed that the observers are not aware of their

---

#### Page 20

internal variability. Predictions of model S do not change whether we assume that the observer is aware of his or her measurement error $\Sigma_{\text {cue }}^{2}$ or not; differences amount just to redefinitions of $\hat{\sigma}_{\text {cue }}^{2}$.

For a Gaussian prior with mean $\mu_{\text {prior }}$ and variance $\sigma_{\text {prior }}^{2}$, the response probability has the following closed form solution:

$$
p\left(r \mid x_{\text {cue }}, d_{\text {cue }}, \mu_{\text {prior }}, \sigma_{\text {prior }}^{2}\right)=\mathcal{N}\left(r \mid \mu_{\text {resp }}, \sigma_{\text {response }}^{2}\right)
$$

with

$$
\begin{aligned}
& \mu_{\text {resp }} \equiv \frac{\mu_{\text {prior }} \hat{\sigma}_{\text {cue }}^{2}+x_{\text {cue }} \sigma_{\text {prior }}^{2}}{\sigma_{\text {prior }}^{2}+\hat{\sigma}_{\text {cue }}^{2}} \\
& \sigma_{\text {resp }}^{2} \equiv \sigma_{\text {au }}^{2}+\frac{1}{\kappa} \frac{\sigma_{\text {prior }} \hat{\sigma}_{\text {cue }}^{2}}{\sigma_{\text {prior }}^{2}+\hat{\sigma}_{\text {cue }}^{2}}+\left(\frac{\sigma_{\text {prior }}^{2}}{\sigma_{\text {prior }}^{2}+\hat{\sigma}_{\text {cue }}^{2}}\right)^{2} \Sigma_{\text {cue }}^{2}
\end{aligned}
$$

where $\kappa$ is the noise parameter of the stochastic posterior in model component SPK ( $\kappa=1$ for PPM; $\kappa \sim \infty$ for BDT) and $\Sigma_{\text {cue }}$ is the sensory noise in estimation of the cue position in model $\mathrm{S}\left(\Sigma_{\text {cue }}=0\right.$ for observer models without cue-estimation noise). For observer models P with noise on the prior, Eq. 19 was numerically integrated over different values of the internal measurement (here corresponding to $\log \sigma_{\text {prior }}$ ) with a Gauss-Hermite quadrature method [57].

For non-Gaussian priors there is no closed form solution similar to Eq. 19 and the calculation of the response probability, depending on active model components, may require up to three nested numerical integrations. Therefore, for computational tractability, we occasionally restricted our analysis to a subset of observer models, as indicated in the main text.

For model class PSA (posterior sampling average), the target choice distribution is the probability distribution of the average of $\kappa$ samples drawn from the posterior distribution. For a posterior that is a mixture of Gaussians and integer $\kappa$, it is possible to obtain an explicit expression whose number of terms grows exponentially in $\kappa$. Fortunately, this did not constitute a problem as observer models favored small values of $\kappa$ (also, a Gaussian approximation applies for large values of $\kappa$ due to the central limit theorem). Values of the distribution for non-integer $\kappa$ were found by linear interpolation between adjacent integer values. For model class LA (Laplace approximation) we found the mode of the posterior numerically [56] and analytically evaluated the second derivative of the log posterior at the mode. The mean of the approximate Gaussian posterior is set to the mode and the variance to minus the inverse of the second derivative [34].

For all models, when using the model-dependent response probability, $p(r \mid$ trial $)$, in the model comparison, we added a small regularization term:

$$
p^{(\text {reg })}(r \mid \text { trial })=(1-\epsilon) p(r \mid \text { trial })+\epsilon
$$

with $\epsilon=1.5 \cdot 10^{-6}$ (the value of the pdf of a normal distribution at 5 SDs from the mean). This change in probability is empirically negligible, but from the point of view of model comparison the regularization term introduces a lower bound $\log \epsilon$ on the log probability of a single trial, preventing single outliers from having unlimited weight on the log likelihood of a model, increasing therefore the robustness of the inference.

Sampling and model comparison. For each observer model and each subject's dataset (comprised of training and test session) we
calculated the posterior distribution of the model parameters given the data, $\operatorname{Pr}\left(\theta_{M} \mid\right.$ data, model $) \propto \operatorname{Pr}($ data $) \theta_{M}$, model $)$ $\operatorname{Pr}\left(\theta_{M} \mid\right.$ model $)$, where we assumed a factorized prior over parameters, $\operatorname{Pr}\left(\theta_{M} \mid\right.$ model $)=\Pi_{i} \operatorname{Pr}\left(\theta_{i} \mid\right.$ model $)$. Having obtained independent measures of typical sensorimotor noise parameters of the subjects in a sensorimotor estimation experiment, we took informative log-normal priors on parameters $\sigma_{\text {motor }}$ and $\Sigma_{\text {high }}$ (when present), with log-scale respectively $\log 3.4 \cdot 10^{-3}$ and $\log 7.7 \cdot 10^{-3}$ screen units and shape parameters 0.38 and 0.32 (see Text S3; results did not depend crucially on the shape of the priors). For the other parameters we took a noninformative uniform prior $\sim$ Uniform $[0,1]$ (dimensionful parameters were measured in normalized screen units), with the exception of the $\eta_{\text {prior }}$ and $\kappa$ parameters. The $\eta_{\text {prior }}$ parameter that regulates the noise in the prior could occasionally be quite large (see main text) so we adopted a broader range $\sim$ Uniform $[0,4]$ to avoid edge effects. A priori, the $\kappa$ parameter that governs noise in decision making could take any positive nonzero value (with higher probability mass on lower values), so we assumed a prior $\sim$ Uniform $[0,1]$ on $1 /(\kappa+1)$, which is equivalent to a prior $\sim 1 /(\kappa+1)^{2}$, for $\kappa \in[0, \infty)$. Formally, a value of $\kappa$ less than one represents a performance more variable than posterior-matching (for $\kappa \rightarrow 0$ the posterior distribution tends to a uniform distribution). Results of the model comparison were essentially identical whether we allowed $\kappa$ to be less than one or not. We took a prior $\sim 1 / \kappa^{2}$ on the positive real line since it is integrable; an improper prior such as a noninformative prior $\sim 1 / \kappa$ is not recommendable in a model comparison between models with non-common parameters due to the 'marginalization paradox' [58].

The posterior distribution of the parameters is proportional to the data likelihood, which was computed in logarithmic form as:

$$
\log \operatorname{Pr}(\text { data } \mid \theta_{M}, \text { model })=\sum_{i=1}^{N} \log p^{(\text {reg })}\left(r^{(i)} \mid \text { trial }_{i}\right)
$$

where $p^{(\text {reg })}$ is the regularized probability of response given by Eq. 21, and trial $i$ represents all the relevant variables of the $i$-th trial. Eq. 22 assumes that the trials are independent and that subjects' parameters are fixed throughout each session (stationarity). The possibility of dependencies between trials and nonstationarity in the data is explored in Section 6 of Text S1.

A convenient way to compute a probability distribution whose unnormalized pdf is known (Eq. 22) is by using a Markov Chain Monte Carlo method (e.g. slice sampling [29]). For each dataset and model, we ran three parallel chains with different starting points ( $10^{3}$ to $10^{4}$ burn-in samples, $2 \cdot 10^{3}$ to $5 \cdot 10^{4}$ saved samples per chain, depending on model complexity) obtaining a total of $6 \cdot 10^{3}$ to $1.5 \cdot 10^{3}$ sampled parameter vectors. Marginal pdfs of sampled chains were visually checked for convergence. We also searched for the global minimum of the (minus log) marginal likelihood by running a minimization algorithm (fininsearch in MATLAB) from several starting points ( 30 to 100 random locations). With this information we verified that, as far as we could tell, the chains were not stuck in a local minimum. Finally, we computed Gelman and Rubin's potential scale reduction statistic $R$ for all parameters [59]. Large values of $R$ indicate convergence problems whereas values close to 1 suggest convergence. Longer chains were run when suspicion of a convergence problem arose from any of these methods. In the end average $R$ (across parameters, participants and models) was

---

#### Page 21

1.003 and almost all values were $<1.1$ suggesting good convergence.

Given the parameter samples, we computed the DIC score (deviance information criterion) [30] for each dataset and model. The DIC score is a metric that combines a goodness of fit term and a penality for model complexity, similarly to other metrics adopted in model comparison, such as Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC), with the advantage that DIC takes into account an estimate of the effective complexity of the model and it is particularly easy to compute given a MCMC output. DIC scores are computed as:

$$
\begin{aligned}
& \mathrm{DIC}=2\left[\frac{1}{n_{s}} \sum_{i=1}^{n_{s}} D\left(\theta_{M}^{(i)}\right)\right]-D\left(\widehat{\theta}_{M}\right) \\
& D(\theta) \equiv-2 \log \operatorname{Pr}(\text { data } \mid \theta)
\end{aligned}
$$

where $D(\theta)$ is the deviance given parameter vector $\theta$, the $\theta^{(i)}$ are MCMC parameter samples and $\widehat{\theta}$ is a 'good' parameter estimate for the model (e.g. the mean, median or another measure of central tendency of the sampled parameters). As a robust estimate of $\widehat{\theta}_{M}$ we computed a trimmed mean (discarding $10 \%$ from each side, which eliminated outlier parameter values). DIC scores are meaningful only in a comparison, so we only report DIC scores differences between models ( $\triangle$ DIC). Although a difference of 3-7 points is already suggested to be significant [30], we follow a conservative stance, for which the difference in DIC scores needs to be 10 or more to be considered significant [33]. In Section 4 of Text S1 we report a set of model comparisons evaluated in terms of group DIC (GDIC). The assumption of GDIC is that all participants' datasets have been generated by the same observer model, and all subjects contribute equally to the evidence of each model.

In the main text, instead, we compared models according to a hierarchical Bayesian model selection method (BMS) [31] that treats both subjects and models as random factors, that is, multiple observer models may be present in the population. BMS uses an iterative algorithm based on variational inference to compute model evidence from individual subjects' marginal likelihoods (or approximations thereof, such as DIC, with the marginal likelihood being $\approx-\frac{1}{2}$ DIC). BMS is particularly appealing because it naturally deals with group heterogeneity and outliers. Moreover, the output of the algorithm has an immediate interpretation as the probability that a given model is responsible for generating the data of a randomly chosen subject. BMS also allows to easily compute the cumulative evidence for groups of models and we used this feature to compare distinct levels within factors [31]. As a Bayesian metric of significance we report the exceedance probability $P^{*}$ of a model (or model level within a factor) being more likely than any other model (or level). We consider values of $P^{*}>0.95$ to be significant. The BMS algorithm is typically initialized with a symmetric Dirichlet distribution that represents a prior over model probabilities with no preference for any specific model [31]. Since we are comparing a large number of models generated by the factorial method, we chose for the concentration parameter of the Dirichlet distribution a value $\alpha_{0}=0.25$ that corresponds to a weak prior belief that only a few observer models are actually present in the population ( $\alpha_{0} \rightarrow 0$ would correspond to the prior belief that only one model is true, similarly to GDIC, and $\alpha_{0}=1$ that any number
of models are true). Results are qualitatively independent of the specific choice of $\alpha_{0}$ for a large range of values.

When looking at alternative models of decision making in our second factorial model comparison, we excluded from the analysis 'uninteresting' trials in which the theoretical posterior distribution (Eq. 2 with the true values of $\sigma_{\text {low }}$ and $\sigma_{\text {high }}$ ) was too close in shape to a Gaussian; since predictions of these models are identical for Gaussian posteriors, Gaussian trials constitute only a confound for the model comparison. A posterior distribution was considered 'too close' to a Gaussian if the Kullback-Leibler divergence between a Gaussian approximation with matching low-order moments and the full posterior was less than a threshold value of 0.02 nats (results were qualitatively independent of the chosen threshold). In general, this preprocessing step removed about $45-60 \%$ of trials from unimodal and bimodal sessions (clearly, Gaussian sessions were automatically excluded).

Nonparametric reconstruction of the priors. We reconstructed the group priors as a means to visualize the subjects' common systematic biases under a specific observer model (SPK-L). Each group prior $q_{\text {prior }}(x)$ was 'nonparametrically' represented by a mixture of Gaussians with a large number of components $(m=31)$. The components' means were equally spaced on a grid that spanned the range of the discrete representation of the prior; SDs were equal to the grid spacing. The mixing weights $\left\{\pi_{i}\right\}_{i=1}^{m}$ were free to vary to define the shape of the prior (we enforced symmetric values on symmetric distributions, and the sum of the weigths to be one). The representation of the prior as a mixture of Gaussians allowed us to cover a large class of smooth distributions using the same framework as the rest of our study.

For this analysis we fixed subjects' parameters to the values inferred in our main model comparison for model SPK-L (i.e. to the robust means of the posterior of the parameters). For each prior in each group (Gaussian, unimodal and bimodal test sessions), we simultaneously inferred the shape of the nonparametric prior that explained each subject's dataset, assuming the same distribution $q_{\text {prior }}$ for all subjects. Specifically, we sampled from the posterior distribution of the parameters of the group priors, $\operatorname{Pr}\left(q_{\text {prior }} \mid\right.$ data), with a flat prior over $\log$ values of the mixing weights $\left\{\pi_{i}\right\}_{i=1}^{m}$. We ran 5 parallel chains with a burn-in of $10^{3}$ samples and $2 \cdot 10^{3}$ samples per chain, for a total of $10^{4}$ sampled vectors of mixing weights (see previous section for details on sampling). Each sampled vector of mixing weights corresponds to a prior $q_{\text {prior }}^{i j}$, for $j=1 \ldots 10^{4}$. Purple lines in Figure 12 show the mean ( $\pm 1 \mathrm{SD}$ ) of the sampled priors, that is the average reconstructed priors (smoothed with a small Gaussian kernel for visualization purposes). For each sampled prior we also computed the first four central moments (mean, variance, skewness and kurtosis) and calculated the posterior average of the moments (see Figure 12).

Statistical analyses. All regressions in our analyses used a robust procedure, computed using Tukey's 'bisquare' weighting function (robustfit in MATLAB). Robust means were computed as trimmed means, discarding $10 \%$ of values from each side of the sample. Statistical differences were assessed using repeated-measures ANOVA (rm-ANOVA) with Green-house-Geisser correction of the degrees of freedom in order to account for deviations from sphericity [60]. A logit transform was applied to the optimality index measure before performing rm-ANOVA, in order to improve normality of the data (results were qualitatively similar for non-transformed data). Nonlinear

---

#### Page 22

kernel regression estimates to visualize mean data (Figure 3 and 6) were computed with a Nadaraya-Watson estimator with rule-of-thumb bandwidth [61]. For all analyses the criterion for statistical significance was $p<0.05$.

---

# On the Origins of Suboptimality in Human Probabilistic Inference - Backmatter

---

## Colophon

Citation: Acerbi L, Vijayakumar S, Wolpert DM (2014) On the Origins of Suboptimality in Human Probabilistic Inference. PLoS Comput Biol 10(6): e1003661. doi:10.1371/journal.pcbi. 1003661
Editor: Jeff Beck, Duke University, United States of America
Received November 25, 2013; Accepted April 25, 2014; Published June 19, 2014
Copyright: © 2014 Acerbi et al. This is an open-access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.
Funding: This work was supported in part by grants EP/F500385/1 and BB/F529254/1 for the University of Edinburgh School of Informatics Doctoral Training Centre in Neuroinformatics and Computational Neuroscience from the UK Engineering and Physical Sciences Research Council, UK Biotechnology and Biological Sciences Research Council, and the UK Medical Research Council (LA). This work was also supported by the Wellcome Trust (DMW), the Human Frontiers Science Program (DMW), and the Royal Society Noreen Murray Professorship in Neurobiology to DMW. SV is supported through grants from Microsoft Research, Royal Academy of Engineering and EU FP7 programs. The work has made use of resources provided by the Edinburgh Compute and Data Facility, which has support from the eDRT initiative. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.
Competing Interests: The authors have declared that no competing interests exist.

- Email: L.Acerbi@ms.ed.ac.uk

## Acknowledgments

We thank Sophie Denève, Jan Drugowitsch, Megan A. K. Peters, Paul R. Schrater and Angela J. Yu for useful discussions, Sue Franklin for assistance with the experiments and James Ingram for technical assistance. We also thank the editor and two anonymous reviewers for helpful feedback.

## Author Contributions

Conceived and designed the experiments: LA SV DMW. Performed the experiments: LA. Analyzed the data: LA. Wrote the paper: LA SV DMW.

# References

1. Weiss Y, Simoncelli EP, Adelson EH (2002) Motion illusions as optimal percepts. Nat Neurosci 5: 598-604.
2. Stocker AA, Simoncelli EP (2006) Noise characteristics and prior expectations in human visual speed perception. Nat Neurosci 9: 578-585.
3. Girshick A, Landy M, Simoncelli E (2011) Cardinal rules: visual orientation perception reflects knowledge of environmental statistics. Nat Neurosci 14: 926932 .
4. Chalk M, Seitz A, Seriès P (2010) Rapidly learned stimulus expectations alter perception of motion. J Vis 10: 1-18.
5. Miyazaki M, Nozaki D, Nakajima Y (2005) Testing bayesian models of human coincidence timing. J Neurophysiol 94: 395-399.
6. Jazayeri M, Shaffen MN (2010) Temporal context calibrates interval timing. Nat Neurosci 13: 1020-1026.
7. Alireus MB, Sahani M (2011) Observers exploit stochastic models of sensory change to help judge the passage of time. Curr Biol 21: 200-206.
8. Acechi L, Wolpert DM, Vijayakumar S (2012) Internal representations of temporal statistics and feedback calibrate motor-sensory interval timing. PLoS Comput Biol 8: e1002771.
9. Kording KP, Wolpert DM (2004) Bayesian integration in sensorimotor learning. Nature 427: 244-247.
10. Tassinari H, Hudson T, Landy M (2006) Combining priors and noisy visual cues in a rapid pointing task. J Neurosci 26: 10154-10163.
11. Berniker M, Voss M, Kording K (2010) Learning priors for bayesian computations in the nervous system. PLoS One 5: e12686.
12. Adams WJ, Graf EW, Ernst MO (2004) Experience can change the 'light-fromabove' price. Nature 7: 1057-1058.
13. Sotiropoulos G, Seitz A, Seriès P (2011) Changing expectations about speed alters perceived motion direction. Curr Biol 21: R883-R884.
14. Kording K, Wolpert D (2006) Bayesian decision theory in sensorimotor control. Trends Cogn Sci 10: 319-326.
15. Trommershäuser J, Maloney L, Landy M (2008) Decision making, movement planning and statistical decision theory. Trends Cogn Sci 12: 291-297.
16. Sundareswara R, Schrater PR (2008) Perceptual multistability predicted by search model for bayesian decisions. J Vis 8: 1-19.
17. Vul E, Pashler H (2008) Measuring the crowd within: Probabilistic representations within individuals. Psychol Sci 19: 645-647.
18. Vul E, Goodman ND, Griffiths TL, Tenenbaum JB (2009) One and done? optimal decisions from very few samples. In: Proceedings of the 31st annual conference of the cognitive science society, volume 1, pp. 66-72.
19. Fiser J, Berkes P, Orbán G, Leugyel M (2010) Statistically optimal perception and learning: from behavior to neural representations. Trends Cogn Sci 14: 119-130.
20. Grkas N, Chalk M, Seitz AR, Seriès P (2013) Complexity and specificity of experimentally-induced expectations in motion perception. J Vis 13: 1-18.
21. van den Berg R, Asch E, Ma WJ (2014) Factorial comparison of working memory models. Psychol Rev 121: 124-149.
22. Körding KP, Wolpert DM (2004) The loss function of sensorimotor learning. Proc Natl Acad Sci U S A 101: 9839-9842.
23. Hudson TE, Maloney LT, Landy MS (2007) Movement planning with probabilistic target information. J Neurophysiol 98: 3034-3046.
24. Campbell L (1966) Exponential entropy as a measure of extent of a distribution. Probab Theory Rel 5: 217-225.
25. Mamassian P, Landy MS, Maloney LT (2002) Bayesian modelling of visual perception. In: Rao R, Ohhausen B, Lewicki M, editors, Probabilistic models of the brain: Perception and neural function, MIT Press. pp. 13-36.
26. Wozny DR, Beierholm UR, Shams L (2010) Probability matching as a computational strategy used in perception. PLoS Comput Biol 6: e1000871.
27. Zhang H, Maloney L (2012) Ubiquitous log odds: a common representation of probability and frequency distortion in perception, action, and cognition. Front Neurosci 6: 1-14.
28. Wichmann FA, Hill NJ (2001) The psychometric function: I. fitting, sampling, and goodness of fit. Percept Psychophys 63: 1293-1313.
29. Neal R (2003) Slice sampling. Ann Stat 31: 705-741.
30. Spiegelhalter DJ, Best NG, Carlin BP, Van Der Linde A (2002) Bayesian measures of model complexity and fit. J R Stat Soc B 64: 583-639.
31. Stephan KE, Penny WD, Daunierau J, Moran RJ, Friston KJ (2009) Bayesian model selection for group studies. Neuroimage 46: 1004-1017.
32. Kass RE, Raftery AE (1995) Bayes factors. J Am Stat Assoc 90: 773-795.
33. Battaglia PW, Kersten D, Schrater PR (2011) How haptic size sensations improve distance perception. PLoS Comput Biol 7: e1002080.
34. MacKay DJ (2003) Information theory, inference and learning algorithms. Cambridge University Press.
35. Battaglia PW, Hamrick JB, Tenenbaum JB (2013) Simulation as an engine of physical scene understanding. Proc Natl Acad Sci U S A 110: 18327-18332.
36. Dakin SC, Tibber MS, Greenwood JA, Morgan MJ, et al. (2011) A common visual metric for approximate number and density. Proc Natl Acad Sci USA 108: 19552-19557.
37. Kuss M, Jäkel F, Wichmann FA (2005) Bayesian inference for psychometric functions. J Vis 5: 478-492.
38. Kahneman D, Tversky A (1979) Prospect theory: An analysis of decision under risk. Econometrica 47: 263-291.
39. Tversky A, Kahneman D (1992) Advances in prospect theory: Cumulative representation of uncertainty. J Risk Uncertainty 5: 297-323.
40. Feldman J (2013) Tuning your priors to the world. Top Cogn Sci 5: 13-34.
41. Mamassian P (2008) Overconfidence in an objective anticipatory motor task. Psychol Sci 19: 601-606.
42. Zhang H, Morvan C, Maloney LT (2010) Gambling in the visual periphery: a conjoint-measurement analysis of human ability to judge visual uncertainty. PLoS Comput Biol 6: e1001023.
43. Zhang H, Daw ND, Maloney LT (2013) Testing whether humans have an accurate model of their own motor uncertainty in a speeded reaching task. PLoS Comput Biol 9: e1003080.
44. Trommershäuser J, Gepshtein S, Maloney LT, Landy MS, Baeke MS (2005) Optimal compensation for changes in task-relevant movement variability. J Neurosci 25: 7169-7178.
45. Gepshtein S, Seydell A, Trommershäuser J (2007) Optimality of human movement under natural variations of visual-motor uncertainty. J Vis 7: 1-18.
46. Drugowitsch J, Wyarta V, Koechlin E (2014). The origin and structure of behavioral variability in perceptual decision-making. Cosyne Abstracts 2014, Salt Lake City USA.
47. Faisal AA, Selou LP, Wolpert DM (2008) Noise in the nervous system. Nat Rev Neurosci 9: 292-303.
48. Ma WJ, Beck JM, Latham PE, Pouget A (2006) Bayesian inference with probabilistic population codes. Nat Neurosci 9: 1432-1438.
49. Beck JM, Ma WJ, Pirkow X, Latham PE, Pouget A (2012) Not noisy, just wrong: the role of suboptimal inference in behavioral variability. Neuron 74: 30-39.
50. Sutton RS, Barto AG (1998) Reinforcement learning: An introduction. MIT press.

---

#### Page 23

51. Gaismauer W, Schooler LJ (2008) The smart potential behind probability matching. Cognition 109: 416-422.
52. Green C, Benson C, Kersten D, Schraier P (2010) Alterations in choice behavior by manipulations of world model. Proc Natl Acad Sci U S A 107: 16401-16406.
53. Oldfield RC (1971) The assessment and analysis of handedness: the edinburgh inventory. Neuropsychologia 9: 97-113.
54. Howard IS, Ingram JN, Wolpert DM (2009) A modular planar robotic manipulandum with endpoint torque control. J Neurosci Methods 181: 199211 .
55. Teuscher F, Guiard V (1995) Sharp inequalities between skewness and kurtosis for unimodal distributions. Stat Probabil Lett 22: 257-260.
56. Carreira-Pequinan MA (2000) Mode-finding for mixtures of gaussian distributions. IEEE T Pattern Anal 22: 1318-1323.
57. Press WH, Flannery BP, Teukolsky SA, Vetterling WT (2007) Numerical recipes 3rd edition: The art of scientific computing. Cambridge University Press.
58. Dawid A, Stone M, Zalek JV (1973) Marginalization paradoxes in bayesian and structural inference. J R Stat Soc B: 189-233.
59. Gelman A, Rubin DB (1992) Inference from iterative simulation using multiple sequences. Stat Sci 7: 457-472.
60. Greenhouse SW, Grisser S (1959) On methods in the analysis of profile data. Psychometrika 24: 95-112.
61. Hardle W, Müller M, Sperlich S, Werwatz A (2004) Nonparametric and semiparametric models, An introduction. Springer.

---

# On the Origins of Suboptimality in Human Probabilistic Inference - Appendix

---

## Supporting Information

Dataset S1 Subject's datasets. Subjects' datasets for the main experiment $(n=24$, training and test sessions) and for the sensorimotor estimation experiment $(n=10)$, with relevant metadata, in a single MATLAB data file.
(ZIP)

Text S1 Additional analyses and observer models. This supporting text includes sections on: Translational invariance of subjects' behavior; Success probability; Inverted Gaussian loss function; Model comparison with DIC; Model comparison for different shared parameters between sessions; Nonstationary analysis.
(PDF)

Text S2 Noisy probabilistic inference. Description of the models of stochastic probabilistic inference ('noisy posterior' and 'sample-based posterior') and discussion about unstructured noise in the prior.
(PDF)

Text S3 Sensorimotor estimation experiment. Methods and results of the additional experiment to estimate the range of subjects' sensorimotor parameters.
(PDF)

---

#### Page 1

# Supporting Text S1 - Additional analyses and observer models

# Contents

1 Translational invariance of subjects' targeting behavior ..... 2
2 Success probability ..... 3
3 Inverted Gaussian loss function ..... 4
3.1 Observer model with variable loss width $\sigma_{\ell}$ ..... 4
4 Model comparison with DIC ..... 5
4.1 Basic model comparison ..... 5
4.2 Comparison of alternative models of decision making ..... 5
4.3 Comparison of distinct model components ..... 8
5 Model comparison for different shared parameters between sessions ..... 9
6 Nonstationary analysis ..... 11
6.1 Iterative non-Bayesian observer model ..... 11

---

#### Page 2

# 1 Translational invariance of subjects' targeting behavior

In this section we show that subject's behavior depends only on the relative position of the cue with respect to the prior. This result allows us to express all positions in a 'prior-centric' coordinate system $\left(\mu_{\text {prior }}=0\right)$ without loss of generality.

In the paper we assumed that all variables (e.g. cue position $x_{\text {cue }}$, subjects' response $r$, target position $x$ ) can be expressed relative to the current location of the prior ( $\mu_{\text {prior }}$ ); a shift of $\mu_{\text {prior }}$ simply produces an equal shift in all other position variables. That is, subjects' behavior is independent of screen coordinates (translational invariance). The alternative hypothesis is that subjects' responses instead show some form of bias that is screen-coordinate dependent, for example a central tendency towards the middle of the screen.

In order to test whether subjects' relative responses depend on the absolute location of the prior, for each subject we fit a linear regression line to the relation between the relative response $\tilde{r}=r-\mu_{\text {prior }}$ and the prior mean $\mu_{\text {prior }}$ across all trials. Given the generative model of our task, we expected the average relative response to be zero irrespective of prior mean, $\langle\tilde{r}\rangle=0$ and therefore tested whether the slope or intercept are different than zero.

For almost all subjects, the slope and intercept were not significantly different than zero $(p>0.05)$. For two subjects we found that slope or intercept may be significantly different from zero ( $p=0.002$ and $p=0.04$ ). However, even in these cases a correction for multiple comparisons $(n=24)$ suggests that the these differences are not statistically significant or at most marginally so. This analysis confirms that subjects' responses in general do not show statistically significant departures from the assumption of translational invariance.

---

#### Page 3

# 2 Success probability

Figure 1 shows the success probability (see Methods in the paper) averaged across subjects, divided by sessions.

> **Image description.** The image shows a set of four bar charts arranged in a 2x2 grid. Each chart displays the "Success probability" on the y-axis, ranging from 0 to 1, and "Prior distribution" on the x-axis, labeled with numbers 1 through 8. Each chart is further divided into two sections, representing "Low-noise cue" (shown in shades of red) and "High-noise cue" (shown in shades of blue).
>
> Here's a breakdown of each chart:
>
> - **Top Left:** Titled "Gaussian training (n = 24)". It shows eight red bars (low-noise cue) gradually decreasing in height from left to right, followed by eight blue bars (high-noise cue), also decreasing in height. Error bars are visible on top of each bar. A solid dark line and a dashed line both curve downwards, starting from the top-left and going towards the right, above the bars.
>
> - **Top Right:** Titled "Gaussian test (n = 8)". The structure is similar to the top-left chart, with eight red bars (low-noise cue) and eight blue bars (high-noise cue), both sets decreasing in height. Error bars are visible on top of each bar. A solid dark line and a dashed line both curve downwards, starting from the top-left and going towards the right, above the bars.
>
> - **Bottom Left:** Titled "Unimodal test (n = 8)". Similar structure with eight red bars (low-noise cue) and eight blue bars (high-noise cue). The bars are more uniform in height compared to the top charts. Error bars are visible on top of each bar. A solid dark line and a dashed line both curve downwards, starting from the top-left and going towards the right, above the bars.
>
> - **Bottom Right:** Titled "Bimodal test (n = 8)". Similar structure with eight red bars (low-noise cue) and eight blue bars (high-noise cue). The bars are more uniform in height compared to the top charts. Error bars are visible on top of each bar. A solid dark line and a dashed line both curve downwards, starting from the top-left and going towards the right, above the bars.
>
> A legend at the top of the image reads: "Best stochastic model (BDT-P-L)" (represented by a solid dark line) and "Maximal success probability" (represented by a dashed line).

Figure 1. Group mean success probability for all sessions. Each bar represents the groupaveraged success probability for a specific session, for each prior (indexed from 1 to 8 , see also Figure 2 in the paper) and cue type, low-noise cues (red bars) or high-noise cues (blue bars). Error bars are SE across subjects. Priors are arranged in the order of differential entropy (i.e. increasing variance for Gaussian priors), except for 'unimodal test' priors which are listed in order of increasing width of the main peak in the prior (see main paper). The dashed line represents the maximal success probability for an ideal observer. The continuous line represents the 'postdiction' of the best Bayesian model, BDT-P-L (see 'Analysis of best observer model' in the paper). Also, compare this figure with Figure 5 in the paper, which shows the optimality index.

---

#### Page 4

# 3 Inverted Gaussian loss function

In this section we show that the inverted Gaussian loss function described by Eq. 16 in the paper is a very good approximation of the true loss model of the task, the square well loss (Eq. 15 in the paper), meaning that in our analysis we can adopt the inverted Gaussian without loss of generality.

In order to compare the Gaussian loss with the square well loss, we first compute the theoretical distribution of observed cues, given each combination of prior and cue (low-noise and high-noise). The distribution of cues is a convolution between the prior and the cue variability, $p\left(x_{\text {cue }} \mid p_{\text {prior }}, d_{\text {cue }}\right)=$ $\int p_{\text {prior }}(x) \mathcal{N}\left(x_{\text {cue }} \mid x, \sigma_{x}^{2}\left(d_{\text {cue }}\right)\right) d x$. For each combination we calculate the RMSE between the 'optimal target' predicted by the two loss functions for a certain cue position, weighted by cue probability:

$$
R M S E\left(p_{\text {prior }}, d_{\text {cue }}\right)=\left\{\int_{\mathcal{D}} p\left(x_{\text {cue }} \mid p_{\text {prior }}, d_{\text {cue }}\right)\left[x_{\text {Gauss }}^{*}\left(x_{\text {cue }}\right)-x_{\text {well }}^{*}\left(x_{\text {cue }}\right)\right]^{2} d x_{\text {cue }}\right\}^{\frac{1}{2}}
$$

where $\mathcal{D}$ is the range of the discrete representation of $p_{\text {prior }}(x)$. We exclude from the analysis singleGaussian priors, as in that case the predicted optimal target is identical for both loss models. We repeat the calculation for a range of values of the scale of the inverted Gaussian, $\sigma_{\ell}$, while we keep the window size of the square well loss fixed to the 'true' value ( $\ell^{*}=0.083$ screen units, the cursor diameter).

This procedure allow us to find the value of $\sigma_{\ell}$ for which the inverted Gaussian loss best approximates the true loss function of the task in terms of observable behavior, by minimizing the average RMSE across all our experimental conditions. We find an optimal value of $\sigma_{\ell}^{*} \approx 0.027$ screen units, close to the SD of a uniform distribution of range $\ell^{*}$, which is 0.024 screen units (the square well loss can be thought of as an 'inverted uniform distribution'). For $\sigma_{\ell}^{*}$, the total RMSE is $1.2 \cdot 10^{-4} \pm 1.5 \cdot 10^{-4}$ screen units (mean $\pm$ SD across different conditions), which is on average less than a tenth of a mm. In terms of performance, the optimality index of an ideal Bayesian observer that uses the inverted Gaussian loss in place of the square-well loss is $0.9999 \pm 0.0001$ (mean $\pm$ SE across conditions) which is empirically indistinguishable from 1. This analysis shows that the inverted Gaussian loss approximates the behavior of the square well loss far below empirical error for our set of distributions. Hence we can use the inverted Gaussian loss function for our Bayesian observer models without loss of generality.

The inverted Gaussian loss has several advantages over the square well loss. Primarily for us, it allows us to derive an analytic expression of the expected loss that involves only sums of Gaussian distributions (see Eq. 4 in the paper). In general, the inverted Gaussian loss is also a very flexible model, as the scale parameter $\sigma_{\ell}$ allows to interpolate between two very well-known models of loss, a delta function (for $\sigma_{\ell} \rightarrow 0$, which leads to a MAP solution) and a quadratic loss (for $\sigma_{\ell} \rightarrow \infty$, corresponding to the mean of the posterior). In addition to theoretical appeal, experimentally the inverted Gaussian loss has been proven to account very well for people's behavior in a spatial targeting task [1].

### 3.1 Observer model with variable loss width $\sigma_{\ell}$

In the paper we either fixed $\sigma_{\ell}$ to the value that best approximates the square well loss or we considered models that explictly or implicitly assume a quadratic loss $\left(\sigma_{\ell} \rightarrow \infty\right)$. Here we examine the performance of an extended BDT-P-L model (the best model that follows BDT) in which the loss width $\sigma_{\ell}$ is allowed to vary freely. Since the parameter $\sigma_{\ell}$ is irrelevant for Gaussian posteriors, we perform this analysis only for non-Gaussian posteriors (see Methods in the paper). Given the typical scale of the posteriors in the task, a value of $\sigma_{\ell} \gtrsim 0.2$ screen units should be considered near-quadratic for all practical purposes.

We find that subjects fall in two classes with respect to the posterior distribution of parameter $\sigma_{\ell}$. For the majority of subjects ( 10 out of 16), mostly in the bimodal session, the posterior is peaked around $\sigma_{\ell}=0.11 \pm 0.02$ screen units (mean $\pm$ SE across subjects), which is significantly higher than the 'true' value ( $\sigma_{\ell}^{*}=0.027$ screen units; signed rank test, $p<0.01$ ) but still qualitatively different from a nearquadratic loss. For the other six subjects the posterior is much broader and flat in the range of $\sigma_{\ell}$

---

#### Page 5

from 0.2 to 1 screen units, compatibly with a near-quadratic loss. In fact, according to the comparison between alternative models of decision making, these subjects show some preference for a quadratic loss or, similarly, a low-order approximation of the posterior (see Figure 9a in the paper and Figure 3a here, subjects 10-14 and 18). However, note that most of these subjects belong to the unimodal group, where posteriors are still very close to Gaussians and therefore the exact value of the loss width may not be necessarily meaningful. The reason why we find a relatively large loss width in the case of a BDT observer is that it needs to account for large, posterior-dependent targeting errors that are explained instead by stochasticity in decision making by the SPK observer (in neither case posterior-dependent errors can be adequately explained by constant motor noise $\sigma_{\text {motor }}$ ).

Performance of model BDT-P-L with variable loss is better than its corresponding version with fixed $\sigma_{\ell}(\Delta \mathrm{DIC}=-11.5 \pm 4.0, p<0.05)$, but still slightly worse than a model with variability in decision making with the same number of parameters, SPK-L $(\Delta \mathrm{DIC}=22.5 \pm 8.9, p<0.05)$. In conclusion, allowing a degree of freedom to the loss function at most slightly improves model performance for BDT but does not seem to provide a better explanation for the data than models with variability in decision making.

# 4 Model comparison with DIC

We report in this section the DIC scores of invidual models for all subjects, and results of the group DIC (GDIC) model comparison. DIC scores are used in the paper to approximate the marginal likelihood of each dataset and model within a hierarchical Bayesian model selection (BMS) framework [2]. Here we also use DIC scores to compute the average impact of each model factor.

### 4.1 Basic model comparison

Figure 2a shows the model evidence for each individual model and subject. We calculated model evidence as the difference in DIC between a given model and the subject's best model (lower values are better). A difference of more than 10 in this scale should be considered strong evidence for the model with lower DIC. Individual results show that model SPK-P-L performed better than other models for almost all datasets, with the exception of a minority that favored model SPK-P instead. Unlike our BMS analysis, here we see a considerable similarity of performance between model SPK-P-L and SPK-S-P-L, although the latter performs slightly worse than the former in almost all cases. Figure 2b shows the group average DIC (GDIC), relative to the model with lowest average DIC (lower scores are better). SPK-P-L is confirmed as the best model. Model SPK-S-K-L comes second in terms of average score, but note that the difference with SPK-P-L is very significant (pairwise signed-rank test with Bonferroni correction for multiple comparisons, $p<0.001$ ). This suggests that the extra model factor S is not improving model performance, and therefore that SPK-S-P-L is not a 'good' model, in agreement with the small support it obtained in the BMS analysis (see main paper).

### 4.2 Comparison of alternative models of decision making

We consider first the model evidence for each individual model and subject (Figure 3a). Results differ depending on the session (unimodal or bimodal). In both sessions model SPK-L performs consistently well, closely followed by model SPK. However, in the unimodal session there are quite a few subjects whose behavior is well described by several other models. These results are summarized in Figure 3b, which shows the group DIC relative to the model with lowest average DIC (lower scores are better). Due to the difference between sessions we separately computed the group averages for the unimodal and bimodal group. GDIC analysis in the unimodal session alone fails to find significant differences between

---

#### Page 6

> **Image description.** The image presents a model comparison between individual models, displayed in two panels labeled 'a' and 'b'.
>
> **Panel a:**
>
> - This panel is a heatmap-like representation of DIC scores for different models across subjects and test groups.
> - The y-axis is labeled "Models" and lists various models such as "SPK", "BDT", and "PPM", each potentially with suffixes like "-S", "-P", or "-L". These suffixes are also listed as column headers: "Decision making", "Cue noise", "Prior noise", and "Lapse".
> - The x-axis is labeled "Subject number" and ranges from 1 to 24.
> - The subjects are divided into three groups: "Gaussian group," "Unimodal group," and "Bimodal group," separated by vertical dashed lines.
> - Each cell in the grid is colored according to the model's evidence, represented as the DIC difference (ΔDIC) with the best model for that subject. A color bar to the right indicates the ΔDIC range, from 0 (dark red) to >50 (dark blue).
> - Numbers within some cells indicate the ranking of the most supported models with comparable evidence (ΔDIC less than 10).
> - The grid has a dark blue background, with cells varying in color from dark blue to red, indicating the ΔDIC values.
>
> **Panel b:**
>
> - This panel is a horizontal bar chart showing the group average ΔDIC score for each model, relative to the best model.
> - The y-axis lists the same models as in panel a.
> - The x-axis is labeled "ΔDIC" and ranges from 0 to 600.
> - Each bar represents the mean ΔDIC score for a model, with error bars indicating the standard error (SE).
> - Asterisks above the bars denote significant differences in DIC between a given model and the best model, after correction for multiple comparisons: (\*) p<0.05, (\*\*\*) p<0.001.
> - The bars are primarily blue, with the bar for "SPK-P-L" being red.

Figure 2. Model comparison between individual models (DIC scores). a: Each column represents a subject, divided by test group (all datasets include a Gaussian training session), each row an observer model identified by a model string (see Table 2 in the paper). Cell color indicates model's evidence, here displayed as the DIC difference ( $\Delta \mathrm{DIC}$ ) with the best model for that subject (a higher value means a worse performance of a model with respect to the best model). Models are sorted by their group average DIC score (see panel b). Numbers above cells specify ranking for most supported models with comparable evidence ( $\Delta$ DIC less than 10). b: Group average $\Delta$ DIC score, relative to the best model (mean $\pm$ SE). Higher scores indicate worse performance. Asterisks denote significant difference in DIC between a given model and the best model, after correction for multiple comparisons: (\*) $p<0.05$, (\*\*\*) $p<0.001$.

SPK-L and several other observer models. Conversely, GDIC shows significant results in the bimodal session, finding that all models but SPK perform worse than SPK-L.

These results agree with the BMS analysis in the paper in indicating SPK-L as the best model, but otherwise present quite a different pattern. Discrepancies between the two model comparison methods emerge for the following reasons. Firstly, as mentioned in the paper, BMS is not affected by outliers and by construction takes into account group heterogeneity, contrarily to DIC. Secondly, posteriors in the unimodal session may still be very close to Gaussian and therefore distinct models share very similar predictions, which DIC scores alone cannot disambiguate. The hierarchical probabilistic structure of BMS, instead, allows information to flow between global model evidence and individual model evidence for each subject (respectively $\alpha$ and $u_{n k}$ in [2]), at each iteration of the model comparison algorithm. This propagation of belief led BMS to discard less likely models in the paper.

---

#### Page 7

> **Image description.** The image presents a comparison of decision-making models using DIC scores, displayed in two panels labeled 'a' and 'b'.
>
> Panel a: This panel is a heatmap. The y-axis is labeled "Models" and lists various models (SPK, BDT, PSA, PPM) combined with parameters such as 'MV', 'LA', and 'L'. The x-axis is labeled "Subject number" and ranges from 9 to 24. The subjects are divided into two groups: "Unimodal group" (subjects 9-16) and "Bimodal group" (subjects 17-24), separated by a vertical dashed line. Above the subject groups are labels "Decision making", "Gaussian approx.", and "Lapse". Each cell in the heatmap represents a subject and a model. Cell color indicates the model's evidence, displayed as the DIC difference (ΔDIC) with the best model for that subject. A colorbar to the right of the heatmap shows the color scale, ranging from dark blue (0) to red (>50), representing the ΔDIC value. Numbers within the cells (1 to 7) indicate the ranking of models with comparable evidence.
>
> Panel b: This panel is a horizontal bar chart. The y-axis lists the same models as in panel a. The x-axis is labeled "ΔDIC" and ranges from 0 to 500. Each model has two horizontal bars representing the group average ΔDIC for the "Unimodal group" and "Bimodal group". Error bars are present on each bar, representing the standard error. The "Unimodal group" is represented by white diamonds, and the "Bimodal group" is represented by cyan circles. Asterisks above the bars indicate significant differences in DIC between a given model and the best model, with one asterisk (\*) indicating p < 0.05, two asterisks (**) indicating p < 0.01, and three asterisks (\***) indicating p < 0.001. A legend in the top right corner clarifies which color represents each group.

Figure 3. Model comparison between alternative models of decision making (DIC scores). We tested a class of alternative models of decision making which differ with respect to predictions for non-Gaussian trials only. a: Each column represents a subject, divided by group (either unimodal or bimodal test session), each row an observer model identified by a model string (see Table 2 in the paper). Cell color indicates model's evidence, here displayed as the DIC difference ( $\Delta \mathrm{DIC}$ ) with the best model for that subject (a higher value means a worse performance of a model with respect to the best model). Models are sorted by their group average DIC score across both sessions (see panel b). Numbers above cells specify ranking for most supported models with comparable evidence ( $\Delta \mathrm{DIC}$ less than 10). b: Group average $\Delta$ DIC, divided by test group (unimodal or bimodal session), relative to the best model (mean $\pm$ SE). Higher scores indicate worse performance. Asterisks denote significant difference in DIC between a given model and the best model, after correction for multiple comparisons: $\left(^{*}\right) p<0.05,\left({ }^{* *}\right) p<0.01$, $\left({ }^{* * *}\right) p<0.001$.

---

#### Page 8

# 4.3 Comparison of distinct model components

We assess the relevance of each model level within a factor by measuring the average contribution to DIC of each level across all tested observer models, relative to the best level (Figure 4). This is the GDIC counterpart of the BMS computation of the posterior likelihood of each model component (Figures 8c and 9 c in the paper). Results of the GDIC analysis are qualitatively similar to BMS for all factors, with the sole exception of factor S (sensory noise in estimation of the cue position). BMS rejects factor S, whereas from GDIC we can see that, on average, it seems that not having factor S decreases model performance ( $\Delta$ DIC: $33.0 \pm 5.6$, mean $\pm$ SE across subjects). This is not a contradiction: for many simple observer models the addition of any reasonable form of noise, including cue-estimation noise, will improve model performance. However, model factor S becomes redundant when other more fitting forms of noise are present. Since GDIC weights equally all model contributions, model S appears to have a useful influence on model performance due to the average contribution of 'simpler' models. On the contrary, BMS weights evidence differentially and component S appears to be irrelevant for the most likely models (see paper).

> **Image description.** This image contains two horizontal bar charts, labeled "a" and "b".
>
> **Panel a:**
>
> - The title of the chart is "All test trials".
> - The y-axis is labeled "Model Factors" and lists the following factors from top to bottom: SPK, BDT, PPM, S, ¬S, P, ¬P, L, ¬L.
> - The x-axis is labeled "ΔDIC" and ranges from 0 to 500.
> - There are horizontal blue bars corresponding to each model factor, with error bars at the end of each bar.
> - Asterisks are used to denote significance levels. "BDT", "PPM", "¬S", "¬P", and "¬L" have three asterisks each, indicating high significance.
>
> **Panel b:**
>
> - The title of the chart is "Non-Gaussian test trials".
> - The y-axis is labeled "Model Factors" and lists the following factors from top to bottom: SPK, PPM, BDT, ¬GA, MV, LA, SPK, PSA, L, ¬L.
> - The x-axis is labeled "ΔDIC" and ranges from 0 to 200.
> - There are horizontal blue bars corresponding to each model factor, with error bars at the end of each bar.
> - Asterisks are used to denote significance levels. "PPM", "BDT", and "¬L" have three asterisks each. "MV" has one asterisk, and "PSA" has two asterisks.

Figure 4. Influence of different model factors on DIC. Difference in DIC between different levels within factors, relative to the best level (lowest DIC); highest scores denote worse performance. Each group of bars represent a factor, each bar a level within the factor, identified by a model label (see Table 2 in the paper). Error bars are SE across subjects. Asterisks denote significant difference in DIC between a given level and the best level, after correction for multiple comparisons. a: Factors in the basic model comparison. b: Factors in the comparison of alternative models of decision making. Label ' $\neg$ GA' stands for no Gaussian approximation (full posterior).

---

#### Page 9

# 5 Model comparison for different shared parameters between sessions

In the paper we assumed that each subject shared two parameters between the training session and the test session (the motor noise $\sigma_{\text {motor }}$ and the ratio between the cue noise, $\tilde{\sigma}_{\text {high }} / \tilde{\sigma}_{\text {low }}$ ), whereas all the other parameters were specified separately for the two sessions (see 'Model comparison' section in the paper). Here we motivate our modelling choice by showing that it is optimal, at least on a subset of observer models. By 'optimal' we mean that models that share more parameters between sessions perform substantially worse, whereas models that share less parameters (and therefore have more free parameters to specify) do not provide a significant advantage.

For the current analysis we consider a set of variants of observer model SPK (stochastic posterior). We focus on this model since it is the simplest model with the 'best' decision-making component, as found in the paper. These variants differ from the standard SPK model only with respect to the number of parameters shared between training and test sessions. For a single session, model SPK can be characterized by four parameters ( $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa$; see 'Suboptimal Bayesian observer models' section in the paper). Table 1 lists the considered variants, labelled by number of parameters shared across sessions (model SPK\#2 corresponds to the variant adopted in the paper). ${ }^{1}$

| Model  | Total number of parameters |                                 Free parameters $\left(\boldsymbol{\theta}_{M}\right)$                                 |
| :----- | :------------------------: | :--------------------------------------------------------------------------------------------------------------------: |
| SPK\#4 |             4              |            $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa$            |
| SPK\#3 |             5              |       $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa \times 2$        |
| SPK\#2 |             6              |       $\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }},(\tilde{\sigma}_{\text {high }}, \kappa) \times 2$       |
| SPK\#1 |             7              |       $\sigma_{\text {motor }},(\tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa) \times 2$       |
| SPK\#0 |             8              | $\left(\sigma_{\text {motor }}, \tilde{\sigma}_{\text {low }}, \tilde{\sigma}_{\text {high }}, \kappa\right) \times 2$ |

Table 1. Observer model SPK with different shared parameters. Table of observer models based on SPK (stochastic posterior) but with different number of shared parameters (model SPK\#2 corresponds to the version in the paper). The number after the '\#' symbol represents the number of parameters the model shares between training and test session. For each model it is also specified the total number of free parameters used to characterize both sessions. A ' $\times 2$ ' means that a parameter is specified independently for training and test sessions; otherwise parameters are shared across sessions. See main text and Methods in the paper for the meaning of the various parameters.

Here we use GDIC instead of BMS since we want to find the modelling choice that works best on average for all subjects. Figure 5 shows the relative DIC scores of the model for different number of shared parameters. Unsurprisingly, the model with lowest group DIC is the model with the highest number of parameters (SPK\#0). However, models SPK\#1 and SPK\#2 closely match the performance of model SPK\#0. In particular, the difference between SPK\#2 and SPK\#0 is nonsignificant ( $\Delta \mathrm{DIC}=3.5 \pm 2.1$; $p=0.55$ ). Conversely, observer models with 3 or more shared parameters perform significantly worse (e.g., for SPK\#3: $\Delta \mathrm{DIC}=32.4 \pm 7.3 ; p<0.001$ ).

These results show that a model that shares the motor noise parameter and the ratio between the estimated cues' SDs between sessions achieves the optimal balance between model fit and simplicity, supporting our choice in the paper.

[^0]
[^0]: ${ }^{1}$ Although there are in total $2^{4}$ variants of model SPK that share different combinations of parameters between sessions, the five models in Table 1 represent the most natural combinations, in order of increasing model complexity.

---

#### Page 10

> **Image description.** This image consists of two panels, labeled 'a' and 'b', presenting a comparison of statistical models.
>
> Panel a: This panel is a heatmap-like representation.
>
> - The y-axis is labeled "Models" and lists five models: "SPK #0", "SPK #1", "SPK #2", "SPK #3", and "SPK #4".
> - The x-axis is labeled "Subject number" and ranges from 1 to 24. The subjects are divided into three groups: "Gaussian group", "Unimodal group", and "Bimodal group", separated by dashed vertical lines.
> - Above the subject groups are the labels "Decision making" and "# Shared parameters", oriented diagonally.
> - The heatmap itself consists of colored cells, each containing a number from 1 to 5. The color of each cell corresponds to a value on the colorbar to the right, ranging from 0 (red) to >50 (dark blue). The colorbar is labeled "ΔDIC".
>
> Panel b: This panel is a horizontal bar graph.
>
> - The y-axis lists the same five models as in panel a: "SPK #0", "SPK #1", "SPK #2", "SPK #3", and "SPK #4".
> - The x-axis is labeled "ΔDIC" and ranges from 0 to 100.
> - Each model has a corresponding horizontal bar, with error bars extending from the end of the bar. The bars are colored to match the colors in the heatmap of panel a.
> - Asterisks are present above the bars for "SPK #3" and "SPK #4", indicating statistical significance.

Figure 5. Comparison of models with different number of shared parameters. Model comparison between observer models based on model SPK but with different number of shared parameters between sessions. a: Each column represents a subject, divided by test group (all datasets include a Gaussian training session), each row an observer model identified by a model string (see Table 1). Cell color indicates model's evidence, here displayed as the DIC difference ( $\Delta \mathrm{DIC}$ ) with the best model for that subject (a higher value means a worse performance of a model with respect to the best model). Models are sorted by their group average DIC score (see panel b). Numbers above cells specify ranking for most supported models with comparable evidence ( $\Delta$ DIC less than 10). b: Group average $\Delta$ DIC score, relative to the best model (mean $\pm$ SE). Higher scores indicate worse performance. Asterisks denote significant difference in DIC between a given model and the best model, after correction for multiple comparisons: $\left({ }^{(* * *}\right) p<0.001$.

---

#### Page 11

# 6 Nonstationary analysis

In our analysis of the data in the paper we have assumed stationarity of participants' behavior: in first approximation, trials are statistically independent and observers' parameters do not change during the course of a session. Stationarity is a common simplifying assumption in the analysis of psychophysical data, although deviations from stationarity can lead to misestimation of the participants' parameters [3]. A typical source of nonstationarity is 'memory', the influence of recent trials on the current response [4]. This is of prime interest to our study, as it could be the case that the variability that we observe in decision making is not random but due to recency effects.

As a simple, model-free test for recency effects, we look at correlations between trial variables at trial $i$ and trial $i+1$. In particular, we define the error at trial $i, \operatorname{Error}(i)$, as the difference between the subjects's response $r$ and the true target position $x$. The shift at trial $i, \operatorname{Shift}(i)$, is the difference between subject's response $r$ and the cue position $x_{\text {cue }}$. In formulas:

$$
\operatorname{Error}(i)=r^{(i)}-x^{(i)}, \quad \operatorname{Shift}(i)=r^{(i)}-x_{\text {cue }}^{(i)}
$$

Note that subjects explicitly knew the error only during the training session, in which they received full performance feedback. During the test trials they only received a qualitative feedback on whether they succeded or missed in the trial.

For each subject we analyze separately training and test sessions, computing the correlations between Error and/or Shift between trial $i$ and trial $i+1$ for each dataset ( $n=24$ training sessions and $n=24$ test sessions, for four possible combinations of variable interaction). Figure 6 shows the trial to trial correlations for individual subjects and their mean. In all cases we find a small but statistically significant anticorrelation between trial variables in the training sessions (t-test, $p<0.05$ ) and no significant correlation in the test sessions. The anticorrelation in the training sessions could easily emerge from a strategy that produces small adjustments in the opposite direction of the experienced error vector. Since the test sessions did not provide full feedback, we do not see any significant effect. These small and null effects suggest that the major variability in the subjects' responses, observed in both training and test sessions, was not due to some trivial trial-to-trial correlation.

### 6.1 Iterative non-Bayesian observer model

Although the correlations seen in Figure 6 are modest, it may be that an iterative (trial-to-trial) model that captures longer-term correlations may fare better at explaining the data. Iterative Bayesian models have been successful at explaining subjects' perfomance in different domains, such as target estimation [5], distance perception [6] and motor adaptation [7]. Simple heuristics may reproduce a behavior that is very close to the Bayesian prediction $[4,8]$. We consider here an iterative, non-Bayesian linear observer model with lapse (IT-L) that implements a simple trial-to-trial heuristic.

In a trial without lapse, the non-Bayesian iterative observer chooses the target $x$ according to a linear mapping $f$ of the current cue position $x_{\text {cue }}$ (in prior-centric coordinates), depending on the current cue type $d_{\text {cue }}$ and prior $p_{\text {prior }}:$

$$
x=f\left(x_{\text {cue }} ; d_{\text {cue }}, p_{\text {prior }}\right)=W\left(d_{\text {cue }}, p_{\text {prior }}\right) \cdot x_{\text {cue }}
$$

where $W$ is a table of linear weights with one entry for each combination of prior and cue type. The table $W$ is updated on a trial by trial basis according to the feedback received each trial (see below for implementation details). To account for mistakes and other sources of variability, we include a probability of lapse, according to model factor L (see paper) . The final response is as usual obtained by adding motor noise with SD $\sigma_{\text {motor }}$. Although conceptually simple, the model has a total of eight free parameters, most of which are involved in the update rule in order to allow for maximum flexibility (see below).

---

#### Page 12

> **Image description.** The image is a scatter plot showing trial-to-trial correlations between different variables related to error and shift in a learning experiment.
>
> The plot has the following key features:
>
> - **Axes:**
>
>   - The vertical axis is labeled "Trial to trial correlation" and ranges from -0.25 to 0.25 with increments of 0.05.
>   - The horizontal axis is labeled "Trial variables".
>
> - **Data Points:** The data is presented as scatter plots. Each plot consists of:
>
>   - Yellow diamonds: Represent "Training sessions (subjects)".
>   - Blue diamonds: Represent "Test sessions (subjects)".
>   - Yellow circles with error bars: Represent "Training sessions (mean)". The error bars appear to be confidence intervals.
>   - Blue circles with error bars: Represent "Test sessions (mean)". The error bars appear to be confidence intervals.
>
> - **Groups of Data:** The data points are grouped into four categories along the horizontal axis, each representing a different correlation:
>
>   - "Error(i) vs Error(i + 1)"
>   - "Error(i) vs Shift(i + 1)"
>   - "Shift(i) vs Error(i + 1)"
>   - "Shift(i) vs Shift(i + 1)"
>
> - **Legend:** A legend is present in the upper right corner explaining the meaning of the different data point markers.
>
> - **Horizontal Line:** A horizontal line is drawn at y = 0.

Figure 6. Trial to trial correlations between Error and Shift. Correlations between trial variables at trial $i$ and trial $i+1$ for four possible combinations of relevant variables Error (difference between response and target position) and Shift (difference between response and cue position). Each data point is an individual session (training sessions in green, test sessions in blue). Mean correlations, averaged across subjects, are plotted as circles. Error bars are $95 \%$ confidence intervals, computed via bootstrap.

Since in our previous analysis only training sessions showed significant trial-to-trial correlations and, moreover, our update rule assumes that full feedback is available to the subjects, we test the model on the training sessions only. We compare the non-Bayesian iterative model against model SPK (stochastic posterior), the simplest Bayesian observer that includes variability in decision making. GDIC analysis shows that model SPK significantly outperforms model IT-L (20 subjects out of 24 ; paired signed-rank test $p<0.01$ ). We, therefore, reject the hypothesis that our data can be explained by this simple iterative non-Bayesian model (see Discussion in the paper).

# Implementation of the model

For model IT-L, we substitute Eq. 11 of the paper with a trial-dependent equation,

$$
p_{\text {target }}^{(l a p s e)}\left(x^{(i)} \mid x_{c u e}^{(i)}, d_{c u e}^{(i)}, p_{p r i o r}^{(i)}\right)=(1-\lambda) \cdot \delta\left[x^{(i)}-f\left(x_{c u e}^{(i)}, d_{c u e}^{(i)}, p_{p r i o r}^{(i)}\right)\right]+\lambda \cdot p_{p r i o r}^{(i)}\left(x^{(i)}\right)
$$

where all variables now show a dependence on the trial number $i$, but otherwise all symbols have a comparable role as in Eq. 11 in the paper. Here, $f\left(x_{c u e}, d_{c u e}, p_{p r i o r}\right)$ is assumed to be a linear mapping from the position of the cue to the chosen target (see Eq. S3), whose linear weights are stored in table $W^{(i)}$, which is updated each trial. The table contains a separate entry for each combination of prior type ( $p_{\text {prior }}$, or equivalently $\sigma_{\text {prior }}$ for Gaussian priors) and cue type ( $d_{\text {cue }}$, either 'short' or 'long'). We assume for simplicity that the table is initialized with two weight values, respectively one for all short, low-noise

---

#### Page 13

cues, $w_{\text {short }}^{(0)}$, and another one for long, high-noise cues, $w_{\text {long }}^{(0)}$, irrespective of prior type.
In a noise-free scenario, in each trial the error term between current weight and 'correct' weight (according to feedback) can be computed as:

$$
\delta^{(i)}=\frac{r^{(i)}}{x_{c u e}^{(i)}}-\frac{x^{(i)}}{x_{c u e}^{(i)}}=\left(r^{(i)}-x^{(i)}\right) \frac{1}{x_{c u e}^{(i)}}
$$

where $x^{(i)}$ is the actual target position (all positions are measured in coordinates relative to the mean of the prior). However, due to noise, Eq. S5 can take arbitrarily large values because of $x_{c u e}^{(i)}$ at the denominator. We therefore apply a regularization factor to the error, so that

$$
\delta^{(i)}=\left(r^{(i)}-x^{(i)}\right) \cdot \frac{\operatorname{sgn}\left(x_{c u e}^{(i)}\right)}{\left|x_{c u e}^{(i)}\right|+\omega}
$$

with $\omega>0$. For the update rule we take a delta-rule [8]:

$$
W^{(i+1)}\left(\sigma_{\text {prior }}, d_{c u e}\right)=W^{(i)}\left(\sigma_{\text {prior }}, d_{c u e}\right)-\eta \cdot \delta^{(i)} \cdot g\left(\sigma_{\text {prior }}, d_{c u e}, \sigma_{\text {prior }}^{(i)}, d_{c u e}^{(i)}\right)
$$

where $\eta>0$ is a learning factor and $g\left(\sigma_{\text {prior }}, d_{\text {cue }}, \sigma_{\text {prior }}^{\prime}, d_{\text {cue }}^{\prime}\right)$ a transfer function assessing how the learning about a specific combination of prior and cue generalizes to another combination. We assume a simple local learning of the form

$$
g\left(\sigma_{\text {prior }}, d_{\text {cue }}, \sigma_{\text {prior }}^{\prime}, d_{\text {cue }}^{\prime}\right)=e^{-\frac{\left(\sigma_{\text {prior }}-\sigma_{\text {prior }}^{\prime}\right)^{2}}{2 \Delta_{\sigma}^{2}}} \cdot e^{-\frac{\left(d_{\text {cue }}-d_{\text {cue }}^{\prime}\right)^{2}}{2 \Delta_{\text {cue }}^{2}}}
$$

where $\Delta_{\sigma}$ and $\Delta_{\text {cue }}$ are two parameters measuring the generalization length respectively in prior and cue space. Overall, the model has eight parameters: the motor variability $\sigma_{\text {motor }}$ and the lapse rate $\lambda$, the initial weights $w_{\text {short }}^{(0)}$ and $w_{\text {long }}^{(0)}$, the learning factor $\eta$, the regularization parameter $\omega$ and the generalization lengths $\Delta_{\sigma}$ and $\Delta_{\text {cue }}$.

This wide array of parameters allows the model to capture different possible classes of non-Bayesian strategies and update rules. Given the number of parameters and complexity of the log likelihood space of the model, when computing the posterior distribution of the parameters we ran much longer chains in order to improve convergence of the sampling algorithm $\left(3 \cdot 10^{5}\right.$ burn-in samples, $3 \cdot 10^{5}$ saved samples per chain).

# References

1. Körding KP, Wolpert DM (2004) The loss function of sensorimotor learning. Proc Natl Acad Sci U S A 101: 9839-9842.
2. Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009) Bayesian model selection for group studies. Neuroimage 46: 1004-1017.
3. Fründ I, Haenel NV, Wichmann FA (2011) Inference for psychometric functions in the presence of nonstationary behavior. J Vis 11: 1-19.
4. Raviv O, Ahissar M, Loewenstein Y (2012) How recent history affects perception: the normative approach and its heuristic approximation. PLoS Comput Biol 8: e1002731.
5. Berniker M, Voss M, Kording K (2010) Learning priors for bayesian computations in the nervous system. PLoS One 5: e12686.

---

#### Page 14

6. Petzschner F, Glasauer S (2011) Iterative bayesian estimation as an explanation for range and regression effects: a study on human path integration. J Neurosci 31: 17220-17229.
7. Verstynen T, Sabes PN (2011) How each movement changes the next: an experimental and theoretical study of fast adaptive priors in reaching. J Neurosci 31: 10050-10059.
8. Nassar MR, Wilson RC, Heasly B, Gold JI (2010) An approximately bayesian delta-rule model explains the dynamics of belief updating in a changing environment. J Neurosci 30: 12366-12378.

---

#### Page 1

## Supporting Text S2 - Noisy probabilistic inference

We introduce two alternative models of stochastic computations in Bayesian inference ('Stochastic posterior models'). The first one (noisy posterior) comprises a representation of the posterior corrupted by noise; in the second one (sample-based posterior), a discrete, approximate representation of the posterior distribution is built out of a number of samples drawn from the posterior.

We show that, for the loss function of our task, for both models the predicted distribution of chosen targets is quantitatively very close to a power function of the posterior distribution in the trial ('Results'). The generality of this result motivates the power function approximation used for decision-making model level SPK (stochastic posterior), Eq. 7 in the paper.

Lastly, we show that, under specific assumptions, the stochasticity in the posterior can also represent a certain type of noise in the prior ('Stochastic posterior from unstructured noise in the prior').

## Stochastic posterior models

According to Bayesian Decision Theory (BDT), the computation of the optimal target $x^{*}$ for a given loss function $\mathcal{L}$ requires three steps:

1. Computation of the posterior probability $p_{\text {post }}(x)$.
2. Computation of the expected loss, $\mathcal{E}(x)=\int p_{\text {post }}(x) \mathcal{L}\left(x, x^{\prime}\right) d x^{\prime}$.
3. Computation of the target $x^{*}$ that minimizes the expected loss, $x^{*}=\arg \min _{x} \mathcal{E}(x)$.

Step 1 corresponds to the inference step and is described by Eq. 2 in the paper. Steps 2 and 3 correspond to action selection (Eq. 4 in the paper).

In principle, noise in decision making could be added to any of the above steps. For parsimony, here we consider models that adds stochasticity to the computation (or representation) of the posterior distribution (step 1), and we analyze how this noise propagates to the inferred optimal target $x^{*}$. However, our results are compatible also with noise injected at later stages (e.g. in action selection).

## Noisy posterior

For ease of calculation, we convert the continuous posterior distribution $p_{\text {post }}(x)$ to a discrete probability distribution $p_{i}=p_{\text {post }}\left(x_{i}\right)$ for a discrete set of target values $\left\{x_{i}\right\}_{1 \leq i \leq N}$ where we assume that the $x_{i}$ cover uniformly the target space with dense spacing $\Delta x .^{1}$

[^0]
[^0]: ${ }^{1}$ The discretization step could be skipped by modelling continuous noise with a Gaussian process [1]. However, the discrete representation makes the model simpler and easier to interpret. The lattice spacing $\Delta x$ is related to the correlation length of a Gaussian process and affects the amount of noise and discretization error.

---

#### Page 2

We model the computation of a 'noisy posterior' (step 1) by adding normally distributed noise to the posterior (see Figure 7 b in the paper):

$$
\widetilde{p}_{\text {post }}(x)=\sum_{i=1}^{N} y_{i} \delta\left(x-x_{i}\right) \quad \text { with } \quad y_{i}=p_{i}+\sigma\left(p_{i}\right) \eta_{i}
$$

where the $\eta_{i}$ are i.i.d. normal random variables and $\sigma\left(p_{i}\right)$ is the SD of the 'decision noise', that in general depends on the value $p_{i}{ }^{2}$ For simplicity, the $\eta_{i}$ are assumed to be statistically independent but it is easy to extend the model to take into account correlations in the noise.

For the form of $\sigma(p)$ we consider two common alternative rules:

- A Poisson-like law: $\sigma_{\text {Poisson }}(p)=\sqrt{p / g}$, where we have defined $g>0$ as a 'neuronal gain' parameter; higher gain corresponds to less noise. The rationale for this rule is that the $y_{i}$ can be thought of as a population of $N$ independent units or channels ('neurons'), each one noisily encoding the posterior probability at a given target value $x_{i}$ (see Figure 7b in the paper). The activation of each unit ('firing rate'), with a global rescaling factor $g$, takes the form $y_{i}=g p_{i}+\sqrt{g p_{i}} \eta_{i}$ which approximates the response of a Poisson neuron with mean activation $g p_{i}$.
- Weber's law (multiplicative noise), in which the noise is proportional to the probability itself, a form of variability which is typical to many sensory magnitudes: $\sigma_{W e b e r}(p)=w \cdot p$, with $w>0$ the Weber's fraction.

For a fixed lattice spacing $\Delta x$, this model of noise in decision making has only one free parameter, $g$ (or $w$ ), that sets the amount of variability in the inference. Note that the 'neural population' description allows for an intuitive understanding of Eq. S1, but the noisy posterior model does not require to commit to this intepretation.

# Sample-based posterior

This model assumes that a discrete, approximate representation of the posterior is constructed by drawing $K$ samples from the posterior [2-4] (see Figure 7c in the paper):

$$
\widetilde{p}_{\text {post }}(x)=\frac{1}{K} \sum_{i=1}^{K} \delta\left(x-x^{(i)}\right) \quad \text { with } \quad x^{(i)} \sim p_{\text {post }}
$$

where the $x^{(i)}$ are i.i.d. samples from the posterior. The parameter $K$ is inversely proportional to the noise in the representation.

## Target choice distribution

For a given posterior distribution $p_{\text {post }}(x)$, Eqs. S1 and S2 allow us to compute several instances of a stochastic posterior $\widetilde{p}_{\text {post }}(x)$ which, after minimization of the expected loss, entail different chosen targets $x^{*}$. By repeating this procedure and binning the results, we can obtain the shape of the distribution of target choices $p_{\text {target }}(x)$ for a given model of stochasticity (see Figure 7e \& 7 f in the paper). However, this method is computationally very expensive.

A simple expression for $p_{\text {target }}(x)$ is needed in order to make efficient use of a stochastic posterior model in data analysis, e.g. to compute the marginal likelihood of a dataset. Our goal is to show that the

[^0]
[^0]: ${ }^{2}$ Formally, $\widetilde{p}_{\text {post }}(x)$ as defined in Eq. S1 is not a probability distribution since, aside of normalization, it is not always non-negative (the $p_{i}$ 's may take negative values for large amounts of noise in the inference). In this case the 'noisy posterior' should be more correctly interpreted simply as an intermediate step in a noisy computation of the expected loss.

---

#### Page 3

target choice probability of these noisy decision-making models is well approximated by a power function of the posterior distribution:

$$
p_{\text {target }}(x) \sim\left[p_{\text {post }}(x)\right]^{\kappa}
$$

where $\kappa \geq 0$ is an appropriate exponent that is the direct equivalent of the noise parameter $g, w$ or $K$; higher values of $\kappa$ correspond to less decision noise. In general, we would like the exponent in Eq. S3 to be a function of the noise parameter, that is for example $\kappa=\kappa(g)$, where the mapping does not depend on the posterior distribution itself but only on the decision noise level (note that the mapping will depend on other fixed details of the model such as the loss function, and the chosen discretization spacing $\Delta x$ for the 'noisy posterior' model).

# Results

We computed the target choice probability predicted by the stochastic posterior models in our task (noisy posterior with either Poisson-like or Weber's law noise, and sample-based posterior). We chose as loss function the inverted Gaussian approximation used by the observer models in the paper (see Methods in the paper; results did not qualitatively change with the square well loss). We took as posterior distributions a representative set of all posterior distributions of the task, built out of several combinations of prior, cue position and cue type (low-noise and high-noise cues), for a total of about 1000 posterior distributions. We took several levels of decision noise (values of $g, w$ or $K$, depending on the model), ranging from an approximately correct inference to an extremely noisy inference. For each posterior distribution and decision noise level we calculated the shape of the target choice distribution via Monte Carlo sampling ( $10^{5}$ samples per distribution).

Figure 1 shows the target choice distributions and related posterior-power fit distributions (Eq. S3) for three illustrative posteriors and five levels of decision noise for the noisy posterior model with Poisson-like noise. For high levels of decision noise, the target choice distribution resembles the posterior distribution (i.e. a posterior-matching strategy), whereas for low levels of decision noise it becomes a narrow distribution peaked on the mode of the posterior (the model tends to a MAP strategy for $g \rightarrow \infty$ ). This may intuitively explain why a power function of the posterior would be a good approximation of the target choice distribution.

We quantified how well a power function of the posterior can approximate the target choice distributions in terms of Kullback-Leibler (KL) divergence. For each noise level, we computed the exponent $\kappa$ that minimizes the KL divergence between posterior-power distributions and target choice distributions in the set (crucially, the same exponent $\kappa$ fit simultaneously all $\sim 1000$ distributions). To assess the goodness of fit in our experiment, we computed mean and SD of the KL divergence according to a log-normal approximation of the posterior distribution of values of $\kappa$ found in the test sessions for our subjects (see paper, section 'Analysis of best observer model').

In general, we found that the posterior-power fit approximates quite well the target choice distribution of all stochastic posterior models. The KL divergence between true distribution and its approximation was $\sim 0.02 \pm 0.02$ nats (mean $\pm$ s.d. across the distribution of values of $\kappa$ ) for all distinct models of noisy inference. These values are equivalent to the KL divergence between two Gaussian distributions with same SD and whose means differ by about one-fourth of their SD.

This analysis shows that a power function of the posterior represents a good approximation of the distribution of target choices of a Bayesian oberver that takes action according to a noisy or sample-based representation of the posterior. This result provides a sound basis for the analytical form chosen for model level SPK (stochastic posterior), Eq. 7 in the paper.

---

#### Page 4

> **Image description.** The image is a figure composed of multiple plots arranged in a grid, comparing target choice distributions with posterior-power approximations under varying noise levels and posterior distributions.
>
> The figure consists of three columns, labeled "Gaussian posterior," "Unimodal posterior," and "Bimodal posterior" at the top. Each column represents a different type of posterior distribution.
>
> Each column contains six plots stacked vertically. The y-axis of each plot is labeled "Probability." The x-axis of the bottom row of plots is labeled "Optimal target (screen units)." The y-axis ranges from 0 to 0.05 in the first three rows and from 0 to 0.1 in the fourth row, and from 0 to 0.2 in the fifth and sixth rows. The x-axis ranges from 0 to 0.8 in all plots.
>
> The first plot in each column shows the posterior distribution as a filled gray curve. The subsequent five plots in each column show a blue line representing the "Decision distribution" and a dashed red line representing the "Posterior-power fit." The level of noise decreases from top to bottom, as indicated by an arrow labeled "Decreasing noise" on the right side of the figure.
>
> At the bottom of the figure, there is a key indicating the representation of each type of data: a filled gray curve for "Posterior," a blue line for "Decision distribution," and a dashed red line for "Posterior-power fit."

Figure 1. Posterior-power approximation of the noisy posterior model. Comparison between the target choice distributions computed according to the true noisy posterior model (with Poisson-like noise) and their posterior-power approximations. The various panels show the target choice distributions $p_{\text {target }}(x)$ (blue lines) and the associated posterior-power fits (red dashed lines) for different posterior distribution and noise level $g$ in the computation. Each column corresponds to a different illustrative posterior distribution, shown on top, divided by class (Gaussian, unimodal and bimodal). Each row, excluding the first, corresponds to a different level of decision noise, with noise decreasing from top to bottom. Analogous fits were found for the sample-based approximation of the posterior.

---

#### Page 5

# Stochastic posterior from unstructured noise in the prior

We show here that the posterior noise model SPK may also subsume the unstructured components of noise in the prior.

If we assume that the internal measurement of the prior is corrupted by multiplicative sensory noise (according to the approximate Weber's law for density or numerosity esimation [5]) and that it changes smoothly in the target position, the estimated prior can be written as:

$$
\tilde{p}_{\text {prior }}(x)=p_{\text {prior }}(x) \cdot(1+\epsilon(x))
$$

where $\epsilon(x)$ is a Gaussian process with zero mean and some appropriately chosen SD and covariance function (see [1]). Crucially, if the observer uses Eq. S4 to build a posterior distribution, we obtain:

$$
\tilde{p}_{\text {post }}(x)=p_{\text {post }}(x)(1+\epsilon(x))
$$

where $p_{\text {post }}(x)$ is the usual, non-noisy posterior (Eq. 2 in the paper). Eq. S5, once appropriately discretized, is formally equivalent to the equation we used to describe a noisy posterior with multiplicative noise (Eq. S1; see also Figure 7b in the paper). Therefore, under these assumptions, the random, unstructured components of noise in the prior can be absorbed within the noisy posterior model.

Note that the estimation noise on the prior that we considered in the paper, model factor P , is a structured form of noise that varies along task-relevant dimensions (such as the width of the prior or the relative weights of bimodal priors). Whereas structured noise can be identified at least in principle, teasing out which stage or component unstructured noise belongs to represents a greater challenge. For example, an experiment that involves a variable number of inference step may be able to distinguish whether noise stems from the computation of the posterior, which is repeated at every step, or from noise in the encoding of the original prior, which happens only once. A paradigm of this kind has been recently used to explore similar issues in a perceptual categorization task [6]. However, this method is still unable to distinguish whether noise appears in the first step (in the encoding or recall of the prior) or at the very last stage, during action selection. Another way to identify noise in the prior could consist in imposing a strong hyperprior on the subjects via extensive training. The level of attraction to such hyperpriors, once learned, may be indicative of the amount of uncertainty in the subjects' measurement of the prior.

## References

1. Rasmussen C, Williams CKI (2006) Gaussian Processes for Machine Learning. The MIT Press.
2. Sundareswara R, Schrater PR (2008) Perceptual multistability predicted by search model for bayesian decisions. J Vis 8: 1-19.
3. Vul E, Goodman ND, Griffiths TL, Tenenbaum JB (2009) One and done? optimal decisions from very few samples. In: Proceedings of the 31st annual conference of the cognitive science society. volume 1, pp. 66-72.
4. Fiser J, Berkes P, Orbán G, Lengyel M (2010) Statistically optimal perception and learning: from behavior to neural representations. Trends Cogn Sci 14: 119-130.
5. Ross J (2003) Visual discrimination of number without counting. Perception 32: 867-870.
6. Drugowitsch J, Wyarta V, Koechlin E (2014). The origin and structure of behavioral variability in perceptual decision-making. Cosyne Abstracts 2014, Salt Lake City USA.

---

#### Page 1

## Supporting Text S3 - Sensorimotor estimation experiment

We performed a sensorimotor estimation experiment to obtain an independent measure of subjects' sensorimotor variability (see 'Methods'). The sensorimotor variability includes subjects' noise in determining the location of the cue and projecting it back onto the target line as well as any motor noise in indicating that location. We found that in general the sensorimotor variability was small and had a negligible impact on performance ('Results'). The estimated parameters were used to construct informative priors for the model comparison in the paper ('Informative priors for the model comparison').

## Methods

Ten subjects ( 3 male and 7 female; age range $21-33$ years) that had taken part in the main experiment also participated in the control experiment.

The experimental setup had the same layout as the main experiment (see Methods and Figure 1 in the paper), with the following differences: (a) no discrete distribution of targets was shown on screen, only a horizontal target line; (b) in all trials the target was drawn randomly from a uniform distribution whose range covered the width of the active screen window; (c) as usual, half of the trials featured short-distance cues and the other half long-distance cues, but both types of cues had no added noise. In each trial the target was always perfectly above the shown cue, with $x \equiv x_{\text {cue }}$.

Subjects performed a short practice session ( 64 trials) followed by a test session (288 trials). Full performance feedback was provided during both practice and test. Feedback consisted in a visual display of the true position of the target and an integer-valued score that was maximal ( 10 points) for a perfect 'hit' and decreased rapidly away from the target, according to the following equation:

$$
\operatorname{Score}(r, x)=\left\lfloor 10 \cdot e^{-\frac{(r-x)^{2}}{2 \sigma_{\text {score }}^{2}}}+0.5\right\rfloor
$$

where $r$ is the response in the trial, $x$ is the target position, $\sigma_{\text {score }}$ is one-tenth of the cursor diameter $\left(8.3 \cdot 10^{-3}\right.$ screen units or 2.5 mm$)$ and $\lfloor x\rfloor$ denotes the floor function.

All subjects' datasets for the sensorimotor estimation session are available online in Dataset S1.

## Results

Results of the sensorimotor estimation session for all subjects are plotted in Figure 1. The root-meansquared error (RMSE) of the response with respect to the true target position was on average ( $9.3 \pm$ $0.8) \cdot 10^{-3}$ screen units for long-distance cues and $(5.2 \pm 0.3) \cdot 10^{-3}$ screen units for short-distance cues (mean $\pm \mathrm{SE}$ across subjects). In general, the RMSE can be divided in a constant bias term and a variance term, but the bias term was overall small, on average $(0.6 \pm 0.5) \cdot 10^{-3}$ screen units, and not significantly different than zero $(p=0.26)$, which means that the error arose almost entirely from the subject's response variability.

---

#### Page 2

> **Image description.** This image contains two scatter plots comparing targeting error for short-distance and long-distance cues. The plots are labeled 'a' and 'b'.
>
> **Panel a:**
>
> - The plot shows individual RMSE (Root Mean Squared Error) on the y-axis, labeled "RMSE (screen units)", ranging from 0 to 0.02. The x-axis represents "Subjects", numbered from 1 to 10.
> - Two distinct sets of data points are plotted: red dots represent "Short-distance cues," and blue dots represent "Long-distance cues."
> - Each data point has a vertical error bar representing the 95% confidence interval.
> - The red dots, representing short-distance cues, are generally clustered lower on the y-axis than the blue dots, indicating lower RMSE values. The blue dots tend to decrease from left to right.
>
> **Panel b:**
>
> - This plot shows the mean RMSE, averaged across all subjects.
> - The y-axis is the same as in panel a, "RMSE (screen units)", ranging from 0 to 0.02. The x-axis is labeled "Mean."
> - Similar to panel a, there are two data points: a red dot for short-distance cues and a blue dot for long-distance cues, each with a vertical error bar representing the 95% confidence interval.
> - The blue dot (long-distance cues) is higher on the y-axis than the red dot (short-distance cues), indicating a higher mean RMSE for long-distance cues.

Figure 1. Targeting error for short-distance and long-distance cues. RMSE of the responses, with respect to true target position, for different distance of the cues from the target line, either 'short' (brown dots) or 'long' (blue dots); 0.01 screen units correspond to 3 mm . a: Individual RMSE. For visualization, subjects are ordered by average precision. b: Mean RMSE, averaged across subjects. In both graphs error bars are $95 \%$ confidence intervals computed via bootstrap.

Since subjects knew that the cues were fully informative about the target position, all variability in their responses originated from two sources: sensory noise (error in projecting the cue position on the target line) and motor noise. We assumed that sensory and motor noise were independent and normally distributed, and that sensory variability was proportional to the distance of the cue from the target line (Weber's law). Under these assumptions, variance of subjects' responses was described by the following formula:

$$
\sigma_{\text {response }}^{2}=\sigma_{\text {motor }}^{2}+w_{\text {sensory }}^{2} d_{\text {cue }}^{2}
$$

where $w_{\text {sensory }}$ is Weber's fraction and $d_{\text {cue }}$ is the distance of the cue from the target line. Using Eq. S1 we were able to estimate participants' sensorimotor parameters; results are reported in Table 1.

|         Parameter         | Description                | Mean $\pm$ Std <br> (screen units) | Mean $\pm$ Std <br> $(\mathrm{mm})$ |
| :-----------------------: | :------------------------- | :--------------------------------: | :---------------------------------: |
| $\sigma_{\text {motor }}$ | Motor noise                |   $(3.6 \pm 1.1) \cdot 10^{-3}$    |            $1.1 \pm 0.3$            |
|  $\Sigma_{\text {low }}$  | Sensory noise (short cues) |   $(3.5 \pm 1.1) \cdot 10^{-3}$    |            $1.1 \pm 0.3$            |
| $\Sigma_{\text {high }}$  | Sensory noise (long cues)  |   $(8.1 \pm 2.6) \cdot 10^{-3}$    |            $2.4 \pm 0.8$            |

Table 1. Average estimated sensorimotor parameters. Group-average estimated motor and sensory noise parameters. Estimates were obtained from the data through Eq. S1.

The estimated parameters in Table 1 allowed us to assess the typical impact of realistic values of sensorimotor noise on subjects' performance. First, we computed the performance of the optimal ideal observer model with added realistic noise. In order to do so, we generated 1000 subjects by sampling from the distribution of estimated sensorimotor parameters and we then simulated their behavior on our subjects' datasets according to the optimal observer model. We found an average optimality index of $0.997 \pm 0.001$ which is empirically indistinguishable from one. The difference in performance induced by the sensorimotor noise was analogously negligible for the simulations of other ideal observer models, such

---

#### Page 3

as the 'prior-only' or 'cue-only' models (see Figure 5 in the paper). These results show that motor and sensory noise had a very limited impact on subjects' performance.

# Informative priors for the model comparison

The pooled estimated parameters summarized in Table 1 were used to construct informative priors for the motor and sensory parameters that were applied in our model comparison (see paper and Text S1). Bootstrapped parameters were fit with log-normal distributions with log-scale $\mu$ and shape parameter $\sigma$ (which correspond to mean and SD in log space; see Figure 2). The resulting parameters of the priors were $\mu=\log 3.4 \cdot 10^{-3}$ screen units, $\sigma=0.38$ for $\sigma_{\text {motor }}$; and $\mu=\log 7.7 \cdot 10^{-3}$ screen units, $\sigma=0.32$ for $\Sigma_{\text {high }}$. The prior on $\sigma_{\text {motor }}$ was used in all observer models, whereas the prior on $\Sigma_{\text {high }}$ was used only in the observer models with sensory noise (model factor S).

Using an independent experiment to construct informative priors can be thought of as a 'soft' generalization of the typical procedure that consists in directly applying independently estimated parameters to an observer model [1]. In that case, the constructed priors are delta functions on point estimates of the subjects' parameters. Here, instead, pooled measured parameters were used to compute distributions that represent realistic values for the model parameters in our task (that is, informative priors).

> **Image description.** This image is a plot showing probability density curves for motor and sensory noise parameters, along with experimental data points.
>
> - **Axes:** The horizontal axis is labeled "Parameter value (screen units)" and ranges from 0 to 0.02. The vertical axis is labeled "Probability density" and ranges from 0 to 1.2.
>
> - **Curves:** There are two probability density curves:
>
>   - A brown/olive-green curve representing "Motor noise parameter σmotor". This curve has a peak around 0.003-0.004.
>   - A blue curve representing "Sensory noise parameter Σhigh". This curve has a peak around 0.007.
>
> - **Data Points:** There are two sets of data points plotted on the graph:
>
>   - Brown/olive-green dots, clustered around the left side of the graph, corresponding to motor noise.
>   - Blue dots, clustered more towards the center of the graph, corresponding to sensory noise.
>   - Each data point has a horizontal error bar associated with it.
>
> - **Legend:** A legend in the upper right corner identifies the curves:
>   - A brown line is labeled "Motor noise parameter σmotor".
>   - A blue line is labeled "Sensory noise parameter Σhigh".

Figure 2. Priors over sensorimotor parameters. The experimental estimates of individual parameters for motor noise (brown dots) and sensory noise (purple dots) are used to construct informative log-normal priors for $\sigma_{\text {motor }}$ (brown line) and $\Sigma_{\text {high }}$ (purple line) in the main experiment. Error bars are $95 \%$ confidence intervals, computed via bootstrap.

## References

1. Tassinari H, Hudson T, Landy M (2006) Combining priors and noisy visual cues in a rapid pointing task. J Neurosci 26: 10154-10163.