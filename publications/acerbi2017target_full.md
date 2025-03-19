```
@article{acerbi2017target,
  title={Target Uncertainty Mediates Sensorimotor Error Correction},
  author={Acerbi, Luigi and Sethu, Vijayakumar and Wolpert, Daniel M},
  year={2017},
  journal={PLoS ONE},
  doi={10.1371/journal.pone.0170466},
}
```

---

#### Page 1

# Target Uncertainty Mediates Sensorimotor Error Correction

Luigi Acerbi ${ }^{1,2 * *}$, Sethu Vijayakumar ${ }^{1}$, Daniel M. Wolpert ${ }^{3}$<br>1 Institute of Perception, Action and Behaviour, School of Informatics, University of Edinburgh, Edinburgh, United Kingdom, 2 Doctoral Training Centre in Neuroinformatics and Computational Neuroscience, School of Informatics, University of Edinburgh, Edinburgh, United Kingdom, 3 Computational and Biological Learning Lab, Department of Engineering, University of Cambridge, Cambridge, United Kingdom

- Current address: Center for Neural Science, New York University, New York, NY, United States of America \* luigi.acerbi@nyu.edu

#### Abstract

Human movements are prone to errors that arise from inaccuracies in both our perceptual processing and execution of motor commands. We can reduce such errors by both improving our estimates of the state of the world and through online error correction of the ongoing action. Two prominent frameworks that explain how humans solve these problems are Bayesian estimation and stochastic optimal feedback control. Here we examine the interaction between estimation and control by asking if uncertainty in estimates affects how subjects correct for errors that may arise during the movement. Unbeknownst to participants, we randomly shifted the visual feedback of their finger position as they reached to indicate the center of mass of an object. Even though participants were given ample time to compensate for this perturbation, they only fully corrected for the induced error on trials with low uncertainty about center of mass, with correction only partial in trials involving more uncertainty. The analysis of subjects' scores revealed that participants corrected for errors just enough to avoid significant decrease in their overall scores, in agreement with the minimal intervention principle of optimal feedback control. We explain this behavior with a term in the loss function that accounts for the additional effort of adjusting one's response. By suggesting that subjects' decision uncertainty, as reflected in their posterior distribution, is a major factor in determining how their sensorimotor system responds to error, our findings support theoretical models in which the decision making and control processes are fully integrated.

## Introduction

Sensorimotor tasks typically involve both estimating the state of the world (e.g., target and limb positions) and controlling actions so as to achieve goals. Two major frameworks, Bayesian estimation and stochastic optimal feedback control (OFC), have emerged to explain how the sensorimotor system estimates uncertain states and controls its actions. Together these frameworks have provided a normative account of human motor coordination which is able to account for a range of behavioral phenomena, including how humans correct for

---

#### Page 2

perturbations of various kind in fast directed movements. Here we will investigate the relation between estimation and online control.

Estimation is a nontrivial task due to sensory noise [1] and the ambiguity of the stimuli [2]. Optimal estimates need to take into account the statistics of the stimuli, the currently available information, and the cost associated with errors in the estimate [3, 4]. Humans have been shown to combine prior information with sensory data in a manner broadly consistent with Bayesian integration in a variety of sensorimotor tasks, such as reaching [5], interval timing [6, 7], pointing to hidden targets [8-10], speed estimation [11, 12], orientation estimation [13], and motion estimation [14]. Humans are also sensitive to the reward/loss structure imposed by the task [15]. Within the framework of Bayesian Decision Theory (BDT), this means that probabilistic 'posterior' estimates are combined with a cost function so as to maximize the expected gain [16]. Estimation performance compatible with BDT, with an explicitly imposed loss function, has been observed, for example, in visual 'offset' estimation [17], orientation estimation [18], motor planning [19], and sensorimotor timing [7]. These studies suggest that people keep track of the uncertainty, and possibly build a full probabilistic representation [20], of perceptual and sensorimotor variables of interest, and use it to compute optimal estimates.

Optimal feedback control (OFC) is a prominent theory of motor control whereby optimal feedback gains are computed by minimizing the cost of movement over the space of all possible feedback control strategies [21-23]. The ability of the sensorimotor system to make online corrections in OFC is crucial in the presence of errors that can arise from both the inaccuracies in internal models that are involved in generating the commands [24, 25] and from the noise and variability inherent in sensory inputs and motor outputs [1, 26]. The cost function in OFC takes into account various factors, with a trade-off between task goals (accuracy) and effort (energy, movement time, computation); see, e.g., [27]. A prediction of OFC is the minimal intervention principle, according to which errors are corrected and movement variability is minimized only along task-relevant dimensions [21]. OFC also suggests how the motor system should react to perturbations throughout the movement. For example, for fast directed reaching, late perturbations afford a lesser correction gain due to a trade-off between accuracy and stability [28]. A few studies have investigated online control of movement in the presence of uncertain targets, finding agreement with the optimal solution given by a Kalman filter, which describes iterative Bayesian integration of sensory information according to its reliability [2931]. Recent work on the interaction between uncertainty and control has also found that human sensorimotor behavior exhibits risk-sensitivity, that is sensitivity to the uncertainty in the reward [32, 33], which may stem from target variability [34]. In sum, there are both theoretical and empirical reasons to suggest that uncertainty in the estimate may interfere with the way in which humans correct online for their sensorimotor errors.

Online error correction during reaching has typically been studied by observing how subjects react to either mechanical perturbations or explicit or subliminal alterations of visual feedback of the hand (e.g., $[35,36]$ ) or of the target (see [37] for a review). Measured correction gains have been shown to change across the movement [38] and according to task demands, in agreement with OFC [28]. Also, as mentioned before, subjects do not correct indiscriminately for all perturbations, but mostly only for those along task-relevant dimensions [21]. For example, a recent study has shown that subjects used a flexible control strategy that adapted to task demands (target shape) according to the minimal intervention principle on a trial-by-trial basis [39]. These studies, however, mostly examine feedback control in the presence of welldefined, visible targets and with fast movements (duration under 1 second). Here we investigate the relation between estimation and control, by asking if uncertainty in the estimation influences the process of online error correction even when sufficient time is available to fully correct for any errors, here represented by a late visual perturbation.

---

#### Page 3

In our experiment, subjects performed a center of mass estimation task in which they were presented with a visual dumbbell (a bar with disks on each end) and were required to place their finger on the bar to balance the dumbbell. The task was designed so that the estimation would have low variability on some trials (same size disks requiring a simple line bisection to balance) or high variability (unequal sized disks). During the reaching movement to indicate the balance point the location of the finger was occluded and on some trials, unbeknownst to participants, when the finger reappeared its position had been visually shifted so that we could then examine the extent to which subjects corrected for the shift.

We can consider three scenarios. If subjects estimate the center of mass position as a point estimate and then simply report this with a reach, then we would expect that they should correct for the entire perturbation to be as accurate as possible-or, if there is a cost of correction, they should correct just as much for the high and low uncertainty conditions. If subjects represent the full posterior of the position but have no cost on corrections then we would expect that they should correct for the entire perturbation to be as accurate as possible. However, if subjects represent their uncertainty in the center of mass location, as reflected in their posterior distribution, they may be less willing to correct in the high-uncertainty condition as the cost of correction (e.g., energy, movement time, computation) may outweigh the potential increases in accuracy that can be achieved through correction.

Even though participants were given enough time to compensate for the perturbation, they only fully corrected for the induced error on trials with low uncertainty about target location and corrected partially in conditions with more uncertainty (where partial correction was just enough to make their performance practically indistinguishable from the unperturbed trials). Our findings suggest that subjects' decision uncertainty, as reflected in the width of the posterior, is a factor in determining how their sensorimotor system responds to errors, providing new evidence for the link between decision making and control processes.

# Materials and Methods

## Participants

Sixteen naïve subjects ( 8 male and 8 female; age range 19-27 years) participated in the study. All participants were right-handed [40], with normal or corrected-to-normal vision and reported no neurological disorder. The Cambridge Psychology Research Ethics Committee approved the experimental procedures and all subjects gave written informed consent.

## Behavioral task

Subjects performed a center of mass estimation task, designed to probe subject's behavior in a natural sensorimotor task. We used an Optotrak 3020 (Northern Digital Inc, Ontario, Canada) to track the tip of a subject's right index finger at 500 Hz . The visual image from a LCD monitor (Apple Cinema HD, $64 \mathrm{~cm} \times 40 \mathrm{~cm}, 60 \mathrm{~Hz}$ refresh rate) was projected into the plane of the hand via a mirror that prevented the subjects from seeing their arm (Fig 1A). The workspace origin, coordinates $(0,0)$, was $\sim 20 \mathrm{~cm}$ in front of the subject's torso in the mid-sagittal plane, with positive axes towards the right ('horizontal' $x$ axis) and away from the subject ('vertical' $y$ axis). The workspace showed a home position ( 1.5 cm radius circle) at the origin and a cursor ( 0.25 cm radius circle) could be displayed that tracked the finger position.

On each trial a virtual object consisting of two filled circles (disks) and a thin horizontal line (target line) connecting the centers of the two disks [41] was displayed on the screen (Fig 1B). The centers of the disks were $\ell=24 \mathrm{~cm}$ apart (length of the target line) and at vertical position $y=20 \mathrm{~cm}$. To prevent subjects from responding to a stereotypical location in the workspace, on each trial the object was horizontally displaced with a uniformly random jitter $\sim$

---

#### Page 4

> **Image description.** The image is a composite figure illustrating an experimental setup and related probability distributions. It is divided into three panels labeled A, B, and C.
>
> Panel A:
>
> - Shows a side view of a person sitting and looking at an LCD monitor.
> - A mirror is positioned between the monitor and the person's hand.
> - Red lines indicate the reflection of the monitor's display onto the mirror and then to the person's eyes.
> - A dashed red line points from the mirror to a "Virtual cursor" near the person's right index finger.
> - An "Optotrak marker" is labeled near the fingertip.
>
> Panel B:
>
> - Displays a top-down view of a screen.
> - A grey circle at the bottom represents a "home position" with a red dot inside.
> - Two green circles (one larger than the other) are connected by a green line at the top of the screen, representing the object.
> - A dashed line marks the center of mass of the object.
> - A coordinate system (x and y axes) is shown in the bottom left corner.
> - The upper portion of the screen is shaded in light grey.
>
> Panel C:
>
> - Presents a graph with "Probability" on the y-axis and "Center of mass position (cm)" on the x-axis, ranging from -12 to 12.
> - A horizontal green line extends across the graph.
> - Three dumbbell-shaped objects, similar to the one in Panel B, are positioned above the x-axis at -6, 0, and 6 cm. Each is labeled with "p = 1/3".
> - Red curves labeled "High uncertainty" are centered at -6 and 6 cm.
> - A blue vertical line labeled "Low uncertainty" is positioned at 0 cm.
> - A red bell curve is centered on the blue line.
> - Diffuse green and yellow circles are present at -12 and 12 cm.

Fig 1. Experimental setup. A: Subjects wore an Optotrak marker on the tip of their right index finger. The visual scene from a CRT monitor, including a virtual cursor that tracked the finger position, was projected into the plane of the hand via a mirror. B: The screen showed a home position at the bottom (grey circle), the cursor (red circle), here at the start of a trial, and the object at top (green dumbbell). The task consisted of locating the center of mass of the object, here indicated by the dashed line. Visual feedback of the cursor was removed in the region between the home position and the target line (here shaded for visualization purposes). C: The two disks were separated by 24 cm and, depending on the disks size ratio, the target (center of mass) was either exactly halfway between the two disks ( $p=1 / 3$; low uncertainty; blue distribution) or to the right $(p=1 / 3)$ or left $(p=1 / 3)$ of the midpoint (high uncertainty; red distributions), leading to a trimodal distribution of center of mass.
doi:10.1371/journal.pone.0170466.g001

$[-3,3] \mathrm{cm}$ from the center of the screen. The radius of one of the disks was drawn from a lognormal distribution with mean $\log 1 \mathrm{~cm}$ and SD 0.1 in log space. The radius of the other disk was chosen so that on $1 / 3$ of the trials the disks were of equal size, making the task equivalent to a simple line bisection, and on $2 / 3$ of the trials the ratio of the disk radii was drawn from a log-normal distribution with mean $\log 1.5$ and SD 0.1 in log space, leading to a trimodal distribution of center of mass locations (Fig 1C). The bulk of the distribution over locations was far ( $\gg 2 \mathrm{~cm}$ ) from the largest disks' edges, so as to avoid edge effects [41]. The position (left or right) of the larger disk in unequal-size trials was chosen randomly and counterbalanced within each experimental block. We expected that the uncertainty in the center of mass location would be low for the equal-disk trials ('Low-uncertainty'), when the task was equivalent to line bisection, but would be high for the unequal-disk trials ('High-uncertainty') due to both the spread of the experimental distribution and the nonlinear mapping between the disks' ratio and center of mass, see below.

---

#### Page 5

After a 'go' tone, participants were required to reach from the home position to the center of mass of the disks (the target) on the target line, thereby balancing the object on their finger. Subjects were explicitly told in the instructions that the circles were to be interpreted as disks in the center of mass estimation. Importantly, during the reaching movement, visual feedback of the cursor was removed in the region $y \in[2,19] \mathrm{cm}$ (shaded area in Fig 1B). Subjects were given 1.5 s to arrive in the proximity of the target line $(y>19.5 \mathrm{~cm})$. After reaching the target line, subjects were allowed 3 seconds to adjust their endpoint position to correct for any errors that might have arisen during the movement when the cursor was hidden. The remaining time for adjustment was indicated by a pie-chart animation of the cursor, which gradually turned from red to yellow. The cursor's horizontal position at the end of the adjustment phase constituted the subject's response for that trial. If participants were still moving at the end of the adjustment phase (velocity of the finger greater than $0.5 \mathrm{~cm} / \mathrm{s}$ ), the trial was terminated with an error message. Such missed trials were presented again later during the session.

# Experimental sessions

Participants performed a preliminary training session (120 trials) in which they received performance feedback at the end of each trial. Performance feedback consisted of displaying the correct location of the center of mass, an integer score and, if the error was greater than 1 cm , a tilted dumbbell in the appropriate direction. The score depended on the (horizontal) distance of the cursor from the center of mass, $\Delta s$, according to a squared exponential formula:

$$
\operatorname{Score}(\Delta s)=\operatorname{Round}\left(10 \cdot \exp \left\{-\frac{\Delta s^{2}}{2 \sigma_{\text {score }}^{2}}\right\}\right)
$$

where $\sigma_{\text {score }}$ is the score length scale and Round $(z)$ denotes the value of $z$ rounded to the nearest integer. We chose the numerical constants in Eq (1) $\left(\sigma_{\text {Score }} \approx 0.41 \mathrm{~cm}\right)$ such that the score had a maximum of 10 and was nonzero up to 1 cm away from the center of mass. A new trial started 500 ms after the subject had returned to the home position.

Subjects then performed a test session ( 576 trials) which included standard trials (192 trials) identical to the training session, and 'perturbation' trials in which, unbeknownst to the subjects, the visual feedback of the cursor was displaced horizontally from the finger when the cursor reappeared at the end of the movement $(y>19 \mathrm{~cm})$, near the target line. Cursor displacement could either be small (drawn from a Gaussian distribution with mean $\pm 0.5 \mathrm{~cm}$ and SD $0.2 \mathrm{~cm} ; 192$ trials), or large (mean $\pm 1.5 \mathrm{~cm}$ and SD 0.2 cm ; 192 trials). To avoid overlap between distinct perturbation levels, the Gaussian distributions were truncated at 2.5 SDs ( 0.5 cm away from the mean). All trials were presented in a pseudorandom order and left and right perturbations were counterbalanced within the session. To keep subjects motivated throughout the test session while minimizing the chances that subjects would either adapt their behavior or become aware of the shifts, we only provided participants with performance feedback on unperturbed trials [5]. We also provided the sum of the scores for all trials in blocks of 36 trials [17]. All participants answered a short debriefing questionnaire at the end of the session, the results of which showed that they were unaware of the perturbations or of any other difference between trials with or without performance feedback (see S1 Appendix for details).

## Data analysis

For all analyses the criterion for statistical significance was $p<.05$, and we report uncorrected $p$-values. Even after applying a conservative Bonferroni correction for multiple comparisons with $m=20$ (for the about twenty different analyses we conducted) all of our main findings

---

#### Page 6

remain statistically significant. Unless specified otherwise, summary statistics are reported in the text as mean $\pm$ SE between subjects.

# Trial response data

For each trial, we recorded the final horizontal position $r$ of the visual cursor, the horizontal position of the hidden cursor at the time of exiting the no visual feedback zone $x_{\text {exit }}$, and the effective adjustment time (time before the subject stopped moving during the adjustment phase). We computed the response error $\Delta s$ as the signed difference between the final position of the visual cursor and position of the center of mass of the current stimulus.

## Variation of mean residual error and SD of the error

We analyzed how the mean residual error (or 'bias') and SD of the error depended on the class of stimuli presented (Low-uncertainty and High-uncertainty) and on the mean perturbation level $(-1.5,-0.5,0,0.5,1.5)$. For the High-uncertainty trials we had counterbalanced whether the larger disk was on the right or left. An examination of the mean residual error and SD of the error with factor of side (Left, Right) and perturbation mean level showed no significant difference and we therefore pooled data from Left trials with Right trials.

Statistical differences between conditions in these analyses were assessed using repeatedmeasures ANOVA (rm-ANOVA) with Greenhouse-Geisser correction of the degrees of freedom in order to account for deviations from sphericity [42]. A logarithmic transformation was applied to the SDs before performing rm-ANOVA, in order to improve normality of the data (results were qualitatively similar for non-transformed data). We report effect sizes as partial eta squared, denoted with $\eta_{\mathrm{p}}^{2}$.

## Slope of the mean residual error

For each subject, we performed linear regression of the mean residual error as a function of perturbation size (a continuous variable from -2 to 2 cm ) for the Low and High uncertainty conditions. The slope of the regression fit is a measure of the fraction of the applied perturbation that was not corrected for. In the plots, we remove the mean residual error for the 0 perturbation condition from each subject's data to allow a direct comparison between subjects; this has no effect on the estimation of the slope. The difference in slope between conditions was assessed with a paired Student's $t$-test on the individual slope coefficients.

## Observer model

We built a Bayesian observer model to investigate whether our subjects' correction biases could be explained as the interaction of probabilistic inference and the correction cost. In order to account for the residual errors (lack of correction) in the perturbation condition, we introduced a modification to the structure of the loss function that takes effort into account. As described below, subjects' datasets were fit individually and model fits were averaged to obtain the group prediction. To limit model complexity and avoid overfitting, some model parameters were either estimated from the individual training datasets or fixed to theoretically motivated values.

## Perception stage

We assume that the observer estimates the log ratio of the radii of the two disks, whose true value is $\rho=\log \left(r_{2} / r_{1}\right)$, where $r_{i}$ with $i=1,2$ is the radius of the two disks (left and right) presented on a trial. This logarithmic representation was chosen as it naturally embodies Weber's

---

#### Page 7

law. It also unifies different possible transformations of radius to weight for each disk that the subject might use. For example if subjects use the radius to calculate the area or volume (as though the object was a sphere) then the log ratio can be simply expressed as $\log \left(r_{2}^{2} / r_{1}^{2}\right)=2 \rho$ and $\log \left(r_{2}^{2} / r_{1}^{2}\right)=3 \rho$, respectively.

In the estimation process, the true ratio is corrupted by normally distributed noise with magnitude $\sigma_{\rho}$ in log space, which yields a noisy measurement $\rho_{m}$. The parameter $\sigma_{\rho}$ represents both log-normally distributed sensory noise in estimating the radii of the disks and additional independent sources of error in computing the ratio (see Discussion). The conditional measurement probability takes the form:

$$
p_{\text {meas }}\left(\rho_{m} \mid \rho\right)=\mathcal{N}\left(\rho_{m} \mid \rho, \sigma_{\rho}^{2}\right)
$$

where $\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ is a normal distribution with mean $\mu$ and variance $\sigma^{2}$.
The experimental distribution of log ratios is a mixture of three components: two Gaussians centered at $\pm \log 1.5 \approx \pm 0.405$ with SD 0.1 and a delta function at $\rho=0$ (Fig 1B). For simplicity, we assume the observer's prior in log-ratios space, $q_{\text {prior }}(s)$, corresponds to the experimental distribution:

$$
q_{\text {prior }}(\rho)=\frac{1}{3} \sum_{i=1}^{3} \mathcal{N}\left(\rho \mid \mu_{\text {prior }}^{(i)}, \sigma_{\text {prior }}^{(i) 2}\right)
$$

with $\boldsymbol{\mu}_{\text {prior }}=(-\log 1.5,0, \log 1.5)$ and $\boldsymbol{\sigma}_{\text {prior }}=(0.1,0,0.1)$, using the formal definition $\mathcal{N}(x \mid \mu, 0) \equiv \delta(x-\mu)$.

Combining Eqs (2) and (3), after some algebraic manipulations, the posterior can be expressed as a mixture of Gaussians [10]:

$$
q_{\text {post }}\left(\rho \mid \rho_{m}\right)=\frac{1}{\mathcal{Z}} \sum_{i=1}^{3} \mathcal{Z}^{(i)} \mathcal{N}\left(\left.\rho\right|_{\text {post }}, \sigma_{\text {post }}^{(i) 2}\right)
$$

where the normalization factor $\mathcal{Z}$, the posterior mixing weights, means, and variances have all a closed-form expression (see S1 Appendix).

The observer uses the inferred values of $\rho$ to compute the location of the center of mass of the two-disk object (here measured with respect to the midpoint between the two disks). We denote with $f_{D}(\rho)$ the generally nonlinear mapping that identifies the location of the center of mass $s$ of two $D$-dimensional spheres with radii of log ratio $\rho$ (see S1 Appendix). We assume that the observer computes the center of mass using this mapping $f_{D}$ with some fixed value of $D>0$, although not necessarily the correct value $D=2$ for two-dimensional disks, nor we restrict $D$ to be an integer. Knowing the expression for $f_{D}(\rho)$, we can compute the posterior distribution of the location of the estimated center of mass, $q_{\text {post }}\left(s \mid \rho_{m}\right)$ (see S1 Appendix for the derivation). Due to the generally nonlinear form of $f_{D}$, this posterior is a mixture of non-Gaussian distributions. However, we find that it is well approximated by a mixture of Gaussians:

$$
q_{\text {post }}\left(s \mid \rho_{m}\right) \approx \frac{1}{\mathcal{Z}} \sum_{i=1}^{3} \mathcal{Z}^{(i)} \mathcal{N}\left(s \mid m_{\text {post }}^{(i)}, s_{\text {post }}^{(i) 2}\right)
$$

where $m_{\text {post }}^{(i)}$ and $s_{\text {post }}^{(i)}$ are respectively the mean and SD of the mixture components of the posterior (see S1 Appendix for details).

---

#### Page 8

# Decision-making stage

According to Bayesian Decision Theory (BDT), the observer chooses the final cursor position that minimizes his or her expected loss [16]. The typical loss functions used in perceptual and even sensorimotor tasks take into account only the error (distance between response and target). However, although the explicit goal of our task consists of minimizing endpoint error, subjects appeared to be influenced by other considerations.

We assume that the subjects' full loss function depends on an error-dependent cost term, $\mathcal{L}_{\text {err }}(r-s)$, which assesses the deviation of the response $(r)$ from the target $(s)$, and a second adjustment cost, $\mathcal{L}_{\text {adj }}\left(r-r_{0}\right)$, which expresses the cost of moving from the perturbed endpoint position $r_{0}$ (originally planned endpoint position plus perturbation $b$ ). The rationale is that there is an additional cost in moving from the initially planned endpoint position, possibly due to the effort involved in an additional unplanned movement (e.g., for replanning the action).

In a preliminary motor planning stage, the endpoint $s_{\text {pre }}^{r}$ is chosen by minimizing the error loss:

$$
\begin{aligned}
s_{\text {pre }}^{r}\left(\rho_{m}\right) & =\arg \min _{s}\left[\int_{-l / 2}^{l / 2} q_{\text {post }}\left(s \mid \rho_{m}\right) \mathcal{L}_{\text {err }}(\hat{s}-s) d s\right] \\
& =\arg \min _{s}\left[-\sum_{i=1}^{2} \mathrm{Z}^{(i)} \mathcal{N}(\hat{s}) m_{\text {post }}^{(i)} s_{\text {post }}^{(i)} 2+\sigma_{\text {err }}^{2}\right]
\end{aligned}
$$

where we assumed for the loss function a continuous approximation of the discrete scoring system (Eq (1)), that is a (rescaled) inverted Gaussian, $\mathcal{L}_{\text {err }}(\hat{s}-s)=-\exp \left\{-(\hat{s}-s)^{2} / 2 \sigma_{\text {err }}^{2}\right\}$. In addition to being both in agreement with the reward structure of the task and a loss that well describes human sensorimotor behavior [43], this loss allowed us to derive an analytic solution for the expected loss (Eq (6)). To limit model complexity, we assumed subjects conformed to the error length scale of the performance feedback, that is $\sigma_{\text {err }}=\sigma_{\text {score }}$ (Eq (1)).

After the initial movement, subjects are allowed plenty of time to adjust their endpoint position. Due to the applied perturbation $b$, the (average) endpoint position after movement will be $r_{0} \equiv s_{\text {pre }}^{r}\left(\rho_{m}\right)+b$. We introduce, therefore, the adjustment cost in the final loss function:

$$
\mathcal{L}\left(r, s, r_{0}\right)=\mathcal{L}_{\text {err }}(r-s)+\alpha \mathcal{L}_{\text {adj }}\left(r-r_{0}\right)
$$

where $\alpha \geq 0$ specifies the relative weight of the adjustment loss with respect to the error term. In Eq (7), $r_{0}$ represents the (average) end point before adjustment and $r$ the endpoint after adjustment, so that the adjustment loss is a function of the distance covered in the adjustment phase, $r-r_{0}$. The key characteristic of this loss function is that for Low-uncertainty trials the first term can be significantly reduced by adjustments, whereas for High-uncertainty trials there is less to be gained through adjustments (as the location of the center of mass has high variance) and the second term can become dominant leading to partial correction, with $\alpha$ controlling this trade-off. The 'optimal' final position $s^{*}$ that minimizes the expected loss in Eq (7) is:

$$
s^{*}\left(\rho_{m}, r_{0}\right)=\arg \min _{s}\left[\alpha \mathcal{L}_{\text {adj }}\left(\hat{s}-r_{0}\right)+\int_{-l / 2}^{l / 2} q_{\text {post }}\left(s \mid \rho_{m}\right) \mathcal{L}_{\text {err }}(\hat{s}-s) d s\right]
$$

For simplicity, for $\mathcal{L}_{\text {adj }}\left(\hat{s}-r_{0}\right)$ we also assume the shape of an inverted Gaussian loss with length scale $\sigma_{\text {adj }}$, a free parameter of the model representing the scale of the cost of moving away from the originally planned target. For the chosen loss functions, Eq (8) can be further simplified (see S1 Appendix for details), but still only admits numerical solution. In section 'Alternative

---

#### Page 9

observer models', we will see how the solution of Eq (8) changes depending on the shape of the loss functions.

# Full observer model

In each trial, the decision-making process is simulated in two stages. First, the observer computes the preliminary endpoint position $s_{\text {pre }}^{*}\left(\rho_{m}\right)$ for a given internal measurement $\rho_{m}$ (Eq (6)). For simplicity, we assume that the endpoint position is systematically altered only by the external perturbation $b$, so that (on average) the arrival position is $r_{0}=s_{\text {pre }}^{*}\left(\rho_{m}\right)+b$. In the second step, the observer adjusts his or her endpoint position, moving to the optimal target as per Eq (8). Gaussian noise with variance $\sigma_{\text {motor }}^{2}$ is added to the final choice $s^{*}$ to simulate any residual noise in the response.

According to this model, the response probability of observing response $r$ in a trial with perturbation $b$ and disks' ratio $\rho$ is:

$$
\operatorname{Pr}(r \mid \rho, b ; \boldsymbol{\theta})=\int_{-\infty}^{\infty} \mathcal{N}\left(\rho_{m} \mid \rho, \sigma_{\rho}^{2}\right) \mathcal{N}(r) s^{*}\left(\rho_{m}, x_{\text {pre }}^{*}\left(\rho_{m}\right)+b\right), \sigma_{\text {motor }}^{2}\right) d \rho_{m}
$$

where we marginalized over the internal measurement $\rho_{m}$ which is not directly accessible in our experiment, and $\boldsymbol{\theta}=\left\{\sigma_{\rho}, D, \alpha, \sigma_{\text {adj }}, \sigma_{\text {motor }}\right\}$ is the vector of model parameters.

We estimated the model parameters for each subject via maximum-likelihood (see S1 Appendix for details). To limit the possibility of overfitting, the sensory variability parameter of each subject, $\sigma_{\rho}$, was estimated from a separate model fit of the training datasets. The observer model fit to the individual test datasets had, therefore, effectively 3 free parameters: $D, \alpha$ and $\sigma_{\text {adj }}$ representing the dimensionality of the transformation from disk radius to weight, the trade-off between error and effort and the length-scale of the loss function for adjustments, respectively. The parameter $\sigma_{\text {motor }}^{2}$ represents the mean square of the residuals and is not typically counted as a free parameter.

## Results

## Human performance

Subjects found the task natural and straightforward to perform and the debriefing questionnaire at the end of the session showed that they were unaware of the perturbations on the trials. On unperturbed Low-uncertainty trials they received on average $7.36 \pm 0.43$ points and balanced the object on $97.4 \%$ of trials. In contrast on High-uncertainty trials they received on average $3.35 \pm 0.15$ points and balanced the object on $60.2 \%$ of trails. Example subject trajectories and velocity profiles are shown in S1 Fig.

## Mean residual error and variability

We analyzed the participants' response (visual location of cursor at the end of the adjustment phase) as a function of trial uncertainty (Low, High) and mean perturbation level ( $-1.5,-0.5$, $0,0.5,1.5)$. To confirm that the trials with equal-sized and unequal-sized disks correspond to low and high-uncertainty we examined the variability (SD) of subjects' response. As expected, we found that the variability was significantly affected only by trial uncertainty (main effect: Low, High; $F_{(1,15)}=297, p<.001, \eta_{p}^{2}=0.94$ ) with average SD of $0.40 \pm 0.06 \mathrm{~cm}$ and $1.02 \pm 0.05 \mathrm{~cm}$ for the Low and High-uncertainty trials, respectively. We found no significant effect of perturbation and no interaction ( $p>.40$ and $\eta_{p}^{2}<0.04$ for both). This confirms that subjects were more variable in their judgments of the center of mass in 'High-uncertainty' trials.

---

#### Page 10

> **Image description.** The image consists of two scatter plots, labeled A and B, comparing mean residual error against mean perturbation size for low and high uncertainty trials.
>
> Panel A:
> This panel shows a single scatter plot.
>
> - The x-axis is labeled "Mean perturbation size (cm)" and ranges from approximately -1.75 to 1.75.
> - The y-axis is labeled "Mean residual error (cm)" and ranges from -0.4 to 0.4.
> - There are two sets of data points, one for "Low uncertainty" (blue) and one for "High uncertainty" (red). Each data point has error bars, presumably representing standard error of the mean (SEM).
> - Linear regression lines are fitted to each set of data points. The blue line (low uncertainty) has a shallow, almost flat, positive slope. The red line (high uncertainty) has a steeper positive slope.
> - A horizontal black line is present at y=0.
> - The text "n = 16" is present in the lower right corner of the plot.
>
> Panel B:
> This panel consists of 16 smaller scatter plots arranged in a 4x4 grid.
>
> - Each subplot represents data from a single subject.
> - Each subplot has the same x and y axis labels and ranges as Panel A, though the axis labels only appear on the bottom row of subplots.
> - Each subplot contains "Low uncertainty" (blue) and "High uncertainty" (red) data points with error bars, and corresponding linear regression lines, similar to Panel A.
> - The data points in each subplot appear to have been shifted such that the mean residual error for the 0 perturbation condition is removed for each subject.
> - A horizontal black line is present at y=0 in each subplot.

Fig 2. Mean residual error against mean perturbation size, for Low-uncertainty (blue) and High-uncertainty (red) trials. A: Group mean residual error against mean perturbation size. Error bars are SEM between subjects. Fits are are linear regressions to the mean data. $B$ : Each panel reports the mean residual error against mean perturbation size for a single subject, for Low-uncertainty (blue) and High-uncertainty (red) trials. Error bars are SEM between trials. Fits are linear regressions to the individual data. For both panels each subject's data have been shifted so as to remove the mean residual error for the 0 perturbation condition for that subject.
doi:10.1371/journal.pone.0170466.g002

We also examined the subjects' mean residual error (mean difference between cursor endpoint and center of mass). The mean residual error was not significantly affected by trial uncertainty (main effect: Low, High; $F_{(1,15)}=0.69, p>.40, \eta_{p}^{2}=0.04$ ) but was significantly affected by the perturbation level (main effect: perturbation level; $F_{(3.88,58.1)}=25.7, \epsilon=0.969$, $p<.001, \eta_{p}^{2}=0.63$ ) and in particular by the interaction between the two (interaction: perturbation $\times$ uncertainty; $F_{(3.64,54.7)}=15.1, \epsilon=0.91, p<.001, \eta_{p}^{2}=0.50$ ). This suggests that uncertainty modulates the effect of the perturbation on subjects' biases.

To assess the proportion of the perturbation which subjects corrected for, we performed a linear regression of their mean residual error as a function of the perturbation size for Low and High uncertainty trials (after subtracting the baseline mean residual error from unperturbed trials, Fig 2). A slope of zero would correspond to no residual error and hence a full correction, whereas a positive slope correspond to a smaller fraction of the perturbation that subjects correct for, with a slope of 1 corresponding to no correction at all. The regression slopes were small $(0.03 \pm 0.01)$ for Low uncertainty trials but large $(0.16 \pm 0.02)$ for High uncertainty trials, both significantly different than zero ( $t$-test Low: $t_{(15)}=3.61, p=.003$, $d=0.90$; High: $t_{(15)}=8.15, p<.001, d=2.04$ ) and significantly different from each other (paired $t$-test $t_{(15)}=6.80, p<.001, d=1.70$ ). These results show that subjects corrected almost entirely for the perturbation in the Low-uncertainty condition and left sizeable errors in the High-uncertainty trials by only correcting on average for $84 \%$ of the perturbation.

---

#### Page 11

# Exit position

On each trial we also recorded the hidden cursor horizontal position when it crossed the end of the no-feedback zone ( $y=19 \mathrm{~cm}$ ), before applying visual perturbations, as exit position $x_{\text {exit }}$. As a sanity check, we verified that subjects' behavior in perturbation trials before applying the perturbation was identical to no-perturbation trials. In particular, we examined the empirical distribution of $x_{\text {exit }}$ relative to the position of the center of mass for three different levels of perturbation $(-1.5,0,1.5)$ and distinct target locations (left, center, and right). The empirical cumulative distribution functions were well overlapping, meaning that indeed there was no systematic difference between perturbation vs. no-perturbation trials.

Then, we examined the variability (SD) of exit position to investigate subjects' reaching behavior. The SD of $x_{\text {exit }}$ was respectively $0.89 \pm 0.05 \mathrm{~cm}$ (Low uncertainty trials) and $1.70 \pm 0.07 \mathrm{~cm}$ (High uncertainty). We found a statistically significant correlation between the target position and the exit position in the High uncertainty trials (considering Left and Right separately), with a correlation coefficient of $r=.36 \pm 0.02$ ( $t$-test $t_{(15)}=15.0, p<.001, d=3.76$ ). Accordingly, the variability of exit position when considered with respect to target position was statistically significantly lower than the variability of $x_{\text {exit }}$ itself, although not very different in practice ( $1.64 \pm 0.08 \mathrm{~cm}$; paired $t$-test $t_{(15)}=3.8, p=.002, d=0.96$ ). Also, note that the variability of exit position in Low and High uncertainty trials was substantially higher than the corresponding endpoint variability ( $p<.001$ for both). Together, these findings suggest that the subjects' strategy consisted of aiming at a general area depending on the target broad location (left, center, or right), and then refined their endpoint position in the adjustment phase.

## Effective adjustment time

We assessed the time subjects spent in the adjustment phase before they stopped making corrections as a function of trial uncertainty (Low, High) and absolute perturbation size ( $0,0.5$, 1.5). The mean effective adjustment time $(1.60 \pm 0.06 \mathrm{~s})$ was not affected by trial uncertainty per se (main effect: Low, High; $F_{(1,15)}=0.2, p=.66, \eta_{p}^{2}=0.01$ ), but was significantly influenced by perturbation size (main effect: perturbation size; $F_{(1.93,28.9)}=20.9, \epsilon=0.96, p<.001$, $\eta_{p}^{2}=0.58$ ) with no interaction (interaction: uncertainty $\times$ perturbation size; $F_{(2,30)}=0.74, \epsilon \approx$ $1, p>.40, \eta_{p}^{2}=0.05$ ). On average, there was no difference in adjustment time between baseline and small ( 0.5 ) perturbation trials (time difference $1 \pm 16 \mathrm{~ms}, p=.95, d=0.01$ ). However, subjects spent significantly more time adjusting their endpoint position in large (1.5) perturbation trials than baseline trials (time difference $93 \pm 14 \mathrm{~ms}$, paired $t$-test $t_{(15)}=6.89, p<.001$, $d=1.72$ ). Effective adjustment times were broadly scattered in the range $0-3 \mathrm{~s}$ and approximately symmetric around the mean (skewness $0.03 \pm 0.08$ ), with no sign of an accumulation near 3 s . We found qualitatively similar results by defining as 'effective ajustment time' the fraction of time that subjects spent moving in the adjustment phase, instead of the time elapsed before they stopped moving. Velocity profiles during the adjustment phase show that subjects performed a rapid, large correction for perturbations in perturbation trials, followed by occasional small adjustments that become less frequent with time (panel B in S1 Fig). Together, these results suggest that subjects had ample time to make the needed corrections in both Low and High uncertainty trials.

## Analysis of performance

Overall, subjects showed significant mean absolute residual errors (Fig 3A) that depended on the uncertainty level (Low, High) and perturbation size $(0,0.5,1.5)$. To determine how these biases affected performance, we analyzed their mean score per trial as a function of trial

---

#### Page 12

> **Image description.** This image contains two bar graphs, labeled A and B.
>
> **Panel A:**
>
> - **Type:** Bar graph
> - **Title:** None
> - **X-axis:** "Mean absolute perturbation (cm)" with values 0, 0.5, and 1.5.
> - **Y-axis:** "Mean absolute residual error (cm)" with values ranging from 0 to 0.3.
> - **Data:** The graph displays two sets of bars, one for "Low uncertainty" and one for "High uncertainty" at each x-axis value. The "Low uncertainty" bars are blue, while the "High uncertainty" bars are red. Error bars are present on top of each bar.
> - **Annotations:** Horizontal lines with "\*\*\*" above them connect the "Low uncertainty" and "High uncertainty" bars at the 0.5 and 1.5 perturbation levels.
>
> **Panel B:**
>
> - **Type:** Bar graph
> - **Title:** None
> - **X-axis:** "Mean absolute perturbation (cm)" with values 0, 0.5, and 1.5.
> - **Y-axis:** "Mean score" with values ranging from 0 to 10.
> - **Data:** Similar to panel A, the graph displays two sets of bars, one for "Low uncertainty" and one for "High uncertainty" at each x-axis value. The "Low uncertainty" bars are blue, and the "High uncertainty" bars are red. Error bars are present on top of each bar. The blue bars are significantly higher than the red bars.
> - **Annotations:** A horizontal line with "\*\*\*" above it connects the "Low uncertainty" and "High uncertainty" bars at the 0 perturbation level.

Fig 3. Participants' mean absolute residual errors and mean scores. A: Mean absolute residual error (mean $\pm$ SE across subjects; residual errors are computed after removing the residual error for the 0 perturbation condition) by perturbation size $(0, \pm 0.5, \pm 1.5 \mathrm{~cm})$ and trial uncertainty (Low, High). These data are the same as in Fig 2A, here shown in absolute value and aggregated by perturbation size. B: Participants' mean scores (mean $\pm$ SE between subjects) by perturbation size and trial uncertainty. Even though the residual errors (panel A) are significantly different from zero and significantly modulated by perturbation size ( $p<.001$ ) and the interaction between the uncertainty and perturbation size ( $p<.001$ ), the scores (panel B) are significantly affected only by the trial uncertainty ( $p<.001$ ).
doi:10.1371/journal.pone.0170466.g003

uncertainty and perturbation size (Fig 3B). Interestingly, the mean score was significantly influenced only by trial uncertainty (Low: $7.36 \pm 0.38$, High: $3.32 \pm 0.14$; main effect: $F_{(1,15)}=$ $177, p<.001, \eta_{p}^{2}=0.92$ ), with no significant effect of perturbation size nor interaction ( $p>0.60$ and $\eta_{p}^{2}<0.03$ for both). Analogous results hold if we split the High-uncertainty trials in left and right, depending on their location (having, thus, three levels of trial uncertainty: High-Left, Low-Middle, High-Right), and five levels of perturbations ( $-1.5,-0.5,0,0.5,1.5$ ), suggesting that differences are not hidden by the pooling procedure. These findings suggest that subjects' partial lack of correction did not significantly affect their performance.

We compared subjects' average score with that of optimal Bayesian observers (see Methods) which shared the same disks' ratio estimation noise $\sigma_{p}$ as the subjects but correctly computed the location of the center of mass $(D=2)$ and fully compensated for any movement error in the adjustment phase $(\alpha=0)$. The mean score expected from the ideal observer was $9.88 \pm 0.12$ for Low uncertainty trials and $6.03 \pm 0.26$ for High uncertainty ones (mean $\pm$ SD computed via bootstrap). Overall, subjects' average score was significantly lower (paired $t$-test $p<.001$ for both conditions), with a relative efficiency of about $\sim 0.75$ and $\sim 0.55$ for respectively Low and High uncertainty trials.

Our previous analysis ('Mean residual error and variability') showed that subjects' corrective strategy differed between the two levels of uncertainty, with an 'almost-full' correction for Low uncertainty trial ( $\sim 3 \%$ uncorrected perturbation) and a 'partial' correction for High uncertainty trials ( $\sim 16 \%$ uncorrected perturbation). We estimated what would have been the score in the perturbed Low uncertainty conditions, had the participants adopted the partial amount of correction as in the High uncertainty trials. To estimate subjects' score in this hypothetical case we considered their baseline, unperturbed responses and added the mean residual

---

#### Page 13

error from baseline, which we had previously estimated from both Low and High uncertainty trials (corresponding respectively to almost-full and partial correction). We simulated also the almost-full correction strategy as a control, expecting to observe no difference with baseline. The score in each trial was recomputed through Eq (1). The original mean score in the Low uncertainty condition, without perturbation, was $7.36 \pm 0.43$ (see Fig 3B). As expected, hypothetical mean scores under the almost-full correction strategy were not significantly different from baseline ( $7.52 \pm 0.35$ and $7.40 \pm 0.39$, respectively for small, $\pm 0.5$, and large, $\pm 1.5$, perturbations; main effect: perturbation size, $F_{(1.96,29.4)}=0.88, \epsilon=0.84, p=.41, \eta_{p}^{2}=0.06$ ). On the contrary, hypothetical mean scores under the partial correction strategy were significantly different from baseline ( $6.59 \pm 0.49$ and $6.41 \pm 0.41$; main effect: perturbation size, $F_{(1.99,29.9)}=$ $16.1, \epsilon \approx 1, p<.001, \eta_{p}^{2}=0.52$ ). These numbers mean that had the participants been equally sloppy in their correction strategy in the Low uncertainty trials as they were in the High uncertainty trials, the drop in score would have been statistically significant and notable ( $\Delta$ Score $-0.97 \pm 0.18$; paired $t$-test $t_{(15)}=-6.31, p<.001, d=1.58$ ). Conversely, the data show that had the participants been (almost) fully correcting for perturbations in the High uncertainty trials as they were in the Low uncertainty trials, the difference in score would have been negligible (no difference in score between perturbed and unperturbed High uncertainty trials, Fig 3B). This suggests that participants' adjustment strategy took into account task demands, even in the absence of performance feedback in perturbation trials.

# Bayesian model fit

We examined subjects' mean residual errors as a function of the actual center of mass location relative to the midpoint of the bar and mean perturbation level (Fig 4). Even though individual participants' datasets are variable, their mean residual errors exhibited a clear nonlinear pattern as a function of center of mass location, partly driven by the prior over center of mass locations (Fig 1C). We fit the Bayesian observer model to the individual datasets and obtained a good qualitative agreement with the group data (Fig 4) and quantitative agreement for the slope of mean residual error with respect to perturbation for individual subjects ( $R^{2}=0.84$; see Fig 5). Fig 4 shows that there is a separation of biases (vertical shifts) for different amount of perturbation, indicative of the influence of target uncertainty when making corrections. Moreover, we observe a regression to the means of each prior component (left and right), which stems from the shape of the prior.

A crucial element of the model is a loss function that takes into account both a final targeting error cost and an additional cost of moving in the adjustment phase. Due to the width of the posterior distribution in the High-uncertainty condition, the expected gain for an adjustment is smaller than in the Low-uncertainty condition and therefore subjects may be less willing to adjust. Our model qualitatively predicts that the lack of correction to external perturbations should correlate with the trial uncertainty (as measured by the spread of the posterior distribution).

The best fit model parameters to the data were: $\sigma_{\beta}=0.063 \pm 0.004$ (estimated from the training session), $D=1.94 \pm 0.04$ (not significantly different from the correct value $D=2 ; t$-test $t_{(15)}$ $=1.51, p=.15, d=0.38), \sigma_{\text {motor }}=0.76 \pm 0.06 \mathrm{~cm}$. Fits of the loss-related parameters showed that for 3 subjects the adjustment loss was almost constant $\left(\sigma_{\text {adj }} \rightarrow \infty\right)$. For the other 13 subjects we found: $\alpha=3.1 \pm 0.9, \sigma_{\text {adj }}=2.8 \pm 0.5 \mathrm{~cm}$, suggesting that the cost changed slowly, with a large length scale (at least as large as the largest perturbations of $\approx \pm 2 \mathrm{~cm}$ ), and in general these subjects were giving a sizeable weight to the adjustment term ( $\alpha>1 ; t$-test $t_{(12)}=2.19$, $p=.049, d=0.61$ ). Interpreting the adjustment cost as effort, this result is in qualitative agreement with a previous study that found that effort had a considerabily greater relative weight in

---

#### Page 14

> **Image description.** This image is a line graph showing the mean residual error as a function of the location of the center of mass.
>
> The graph has the following characteristics:
>
> - **Axes:** The horizontal axis represents the "Center of mass position (cm)" and ranges from -6.5 to 6.5. The vertical axis represents "Mean residual error (cm)" and ranges from -1 to 1. A dotted horizontal line is present at y=0.
> - **Data Points:** The graph displays data points as circles with error bars. The error bars are vertical lines extending above and below each data point. The data points are clustered around specific x-axis values: -6.5, -5.6, -4.6, -3.6, -2.4, 0, 2.4, 3.6, 4.6, 5.6, and 6.5.
> - **Lines:** Continuous lines are plotted through the data points. These lines are different colors, each representing a different "Mean perturbation" level.
> - **Colors and Labels:**
>   - Purple: "Mean perturbation 1.5 cm"
>   - Dark Purple: "Mean perturbation 0.5 cm"
>   - Gray: "No perturbation"
>   - Dark Green: "Mean perturbation -0.5 cm"
>   - Green: "Mean perturbation -1.5 cm"
> - **Text:** The text "n = 16" is present near the bottom-left of the graph.
>
> The graph shows a relationship between the center of mass position and the mean residual error, with different levels of perturbation affecting the shape and position of the curves. The curves are generally symmetrical around the y-axis, with the error increasing as the center of mass position moves away from zero.

Fig 4. Mean residual error (bias) as a function of the location of the center of mass. Data points and error bars are mean data $\pm \mathrm{SE}$ across subjects in the test session (binned for visualization). Colors correspond to different mean perturbation levels. Continuous lines are the fits of the Bayesian model to each individual dataset, averaged over subjects (asymmetries are due to asymmetries in the data). For both data and model fits, distinct perturbation levels are displayed with a slight offset on the $x$ axis for visualization purposes. Vertical shifts in residual error for different levels of perturbation correspond to different amounts of average lack of correction (absolute residual errors shown in Fig 3A).
doi:10.1371/journal.pone.0170466.g004

the loss function than the error term (relative weight $\sim 7$ for the force production task described in the study; see [44]).

# Alternative observer models

We also analyzed the predictions of a number of alternative observer models: (1) a quadratic loss model for the error term in Eq (7); (2) a power-function loss model for the adjustment loss; (3) an alternative model which explains lack of correction as a miscalibration of the perceived position of the cursor. Alternative models (1) and (3) are unable to account for the principal effect that we observed in the experiment, that is a modulation of the amount of correction that depends on target uncertainty. We found that model (2) is empirically indistinguishable from the inverted Gaussian adjustment loss (as previoulsy reported in another context [43]), meaning that the exact shape that we posited for the adjustment loss is not critical to explain our results. In conclusion, results from these alternative observer models further validate our modelling choices. Detailed description and analysis of these alternative observer models can be found in S1 Appendix.

## Discussion

We used a task in which we could control the uncertainty of the location of a target (the center of mass) and examine the extent to which subjects corrected for perturbations of their reach to indicate the target location. We found that target uncertainty significantly affected subjects'

---

#### Page 15

> **Image description.** This is a scatter plot comparing model predictions with experimental data. The plot shows the relationship between "Model fit slope" on the x-axis and "Measured slope" on the y-axis.
>
> - **Axes:** The x-axis is labeled "Model fit slope" and ranges from approximately -0.05 to 0.4. The y-axis is labeled "Measured slope" and ranges from approximately -0.05 to 0.4.
> - **Data Points:** The plot contains two sets of data points, distinguished by color:
>   - Blue dots represent "Low uncertainty" trials.
>   - Red dots represent "High uncertainty" trials.
> - **Arrangement:** The data points are scattered across the plot. The blue dots (low uncertainty) appear to be clustered more closely to the origin and generally have lower slope values than the red dots (high uncertainty). The red dots (high uncertainty) are more spread out and tend to have higher slope values.
> - **Diagonal Line:** A dashed diagonal line extends from the bottom-left corner to the top-right corner of the plot, representing the line of perfect agreement between the model and the data.
> - **Text:** The text "R² = 0.84" is present in the lower-right quadrant of the plot, indicating the coefficient of determination for the overall fit.
> - **Legend:** A legend is present in the upper-left corner, indicating that blue dots correspond to "Low uncertainty" and red dots correspond to "High uncertainty".

Fig 5. Slope of mean residual error with respect to perturbation, comparison between data and model. Each circle represents the slope of the mean residual error (Fig 2B) for a single subject for Lowuncertainty trials (blue dots) and High-uncertainty trials (red dots). The $x$ axis indicates the slope predicted by the Bayesian observer model, while the $y$ axis reports the slope measured from the data (slope of linear regressions in Fig 2B). The model correctly predicts the substantial difference between Low-uncertainty and High-uncertainty trials and is in good quantitative agreement with individual datasets.
doi:10.1371/journal.pone.0170466.g005

error correction strategy for perturbations of the visual feedback on a trial-by-trial basis, but in such a way that the overall performance would not be hindered. That is, subjects almost fully corrected for the perturbation when target uncertainty was low but only partially corrected when the target uncertainty was high.

# Effect of uncertainty on reaching behaviour

Our study differs from previous work that examines how uncertainty affects sensorimotor behavior. Studies which show that subjects can integrate priors with sensory evidence to produce optimal, yet biased, estimates are consistent with a point estimate being used by the motor system when enacting a movement [5]. The bias we show here is a bias arising from error correction which acts in addition to any biases from Bayesian integration, and would not be predicted if the motor system only had a point estimate of the target location. Moreover, the partial corrections we see relate to the posterior width within a trial. This is in contrast with studies which show that the distribution of perturbations can affect the corrections seen from one trial to the next [45].

Qualitatively similar trial-to-trial, context-dependent responses to perturbations have been observed when people are required to reach to spatially extended target [39]. In that case, corrections happened during fast reaching movements and were compatible with external task demands: errors along the larger dimension of the targets required smaller compensations to still successfully hit the targets (according to the principle of minimal intervention). In our

---

#### Page 16

experiment, however, subjects were sensitive to the implicit posterior width, as opposed to explicit visual target width. Optimal feedback control predicts that, under time constraints, subjects should fail to fully correct for errors that arise late in a movement due to additional requirements of endpoint stability as well as temporal limitations, even if there is no target uncertainty [28]. However, our bias is unrelated to time constraints or requirements of stability as a 3 second adjustment time ensures that sensory delays cannot prevent corrections [29], and our data show that subjects had ample time to correct for mistakes up to their desired precision. Also, note that our use of a long, fixed adjustment time window prevented decision strategies that are available if subjects can choose when to end the adjustment period and move to the next trial, thereby choosing to skip the more difficult trials [46].

An interaction between target uncertainty and response bias has been previously reported in motor planning by Grau-Moya et al. [34]. In their task subjects were required to hit a visual target whose horizontal location uncertainty was manipulated. A robotic interface was used to generate a resistive force that opposed motion in the outward direction with the force linearly related to the horizontal location of the hand. They found that on higher uncertainty trials subjects chose to err on the side of the target with the lower resistive force. There are several key differences of this previous study to ours. In their study, the 'effort' cost is explicit and externally imposed, hit/miss performance feedback is provided on all trials, and explicit manipulations of the cost are blocked by session. By contrast, here we showed an implicit, unconscious trade-off between accuracy and effort in online error correction during a naturalistic task. Moreover, in our study task-relevant perturbations (i.e., implicit manipulations of the cost) were unbeknownst to the subjects and intermixed on a trial-by-trial basis, and we did not provide performance feedback on perturbed trials. Critically, their work does not address correction to ongoing motor commands and shows that subjects can pre-plan a trade-off whereas we show that the online error correction is affected by target uncertainty. Our work provides, therefore, a stronger test of the interaction between uncertainty in the estimate and feedback control.

Finally, target uncertainty in our experiment emerged primarily from a complex mapping from stimulus to target location in a naturalistic task (calculation of the center of mass of a visual dumbbell). This type of uncertainty may differ from uncertainty arising purely from sensory estimation noise, such as with visually blurred targets [29]. On the other hand, 'computational' uncertainty is a common component of everyday problems the motor system needs to deal with, such as with object manipulation (see [47] for a review of different types of sensorimotor uncertainty). In our modelling, for convenience we grouped all sources of noise under the labels of 'sensory' (input) and 'motor' (output) noise but other components may well be present. Our analysis applies here irrespective of the exact nature of uncertainty in target location.

# Uncertainty and lack of correction

A somewhat surprising finding is that subjects did not fully correct for the perturbations, but in a way that did not significantly affect performance. Clearly, a null effect on score differences might simply be due to lack of statistical power in our analysis, but we demonstrated that had subjects used the same partial correction strategy in all trials, their performance would have dropped by almost one point on average. This means that subjects' correction strategy for Low and High uncertainty trials was well adapted to task demands.

A similar finding of partial, yet 'optimal', correction has been reported in a recent study by van Dam and Ernst [48], that looked at subjects' awareness of their own pointing errors. Participants performed a reaching movement to a one-dimensional target, and visual feedback of both the hand and target position was withheld after the commencement of the movement.

---

#### Page 17

After movement termination, subjects responded in a 2AFC task whether they had landed to the left or to the right of the target. In the condition that is most related to our work, subjects were also allowed to correct for their natural pointing mistakes, with no time limit. Also, at this point subjects would receive a brief visual feedback (with small or large blur) about their current endpoint position. The study reports that subjects hardly corrected for their mistakes, but analysis showed that the applied correction gains were sensible (if not 'optimal') when taking into account the information subjects had about their own pointing errors and their current endpoint position [48].

Our study differs from the work by van Dam and Ernst in several fundamental aspects. Most importantly, their work probes a form of Bayesian integration between (a) the current knowledge of endpoint position or, equivalently, estimated distance from the target (due to proprioception and provided noisy visual feedback) and (b) the prior knowledge of the error distribution (and target position). One of their main findings is that subjects seem to acquire more detailed information of the endpoint position only after the end of the movement, even for slow reaches [48]. We showed instead that in our task the lack of correction cannot be explained by a simple form of Bayesian integration. Even if subjects integrated visual feedback of the cursor with (conflicting) proprioceptive information, the expected biases would not yield the observed pattern of uncertainty-dependent corrections.

# The cost of effort and alternative explanations

Our data are consistent with an additional term in the loss function that can be interpreted as 'effort' (whether energy, time or computation; see [15, 21, 44]). The exact nature of this cost is left open, as our experiment does not allow us to pinpoint the specific cost. Our model provides good fits to the subjects' data, and, moreover, we showed that other common models of loss used in Bayesian estimation and motor planning, which either ignore the cost of adjustment or use a quadratic error loss term, fail to account for the key features of our datasets.

However, our model hinges on several assumptions, and more targeted experiments may be needed to completely rule out specific alternative explanations. For example, one assumption of the model is that the observer's posterior distribution over target location is stable within a trial and, for instance, unaffected by the reappearance of the cursor. If subjects took the reappearance of the cursor as an independent piece of evidence, an incorrect belief update (e.g., via a Kalman filter [49]) might produce effects similar to those that we observe. Such behavior is sub-optimal and unlikely since the stimulus was always present on the screen and subjects had plenty of time after the reappearance of the cursor to adjust their endpoint. Our results are also consistent with an interpetation of subjects' behavior as a form of risk-sensitivity [32,34]. An interesting alternative hypothesis inspired by [48] is that subjects built an internal expectation of their average error during the trials with performance feedback, and, therefore, were less willing to correct for large perturbations that were reputed to be unlikely. This interpretation predicts, among other things, that the length scale of the adjustment cost, $\sigma_{\text {adj }}$, should correlate with the spread of the errors made by the subject, but we did not find any evidence for this pattern in the data.

A stronger empirical test for the interaction between effort and cost would consist of a 'Bayesian transfer' type of task [50], in which the same observers are tested on different scoring functions, amounts of required effort [34], and training. In such a task, observers would not necessarily be able to learn any arbitrary reward function, but we expect them to at least adapt to qualitative features of the provided cost, such as skewness-as found, for example, in our previous work on sensorimotor timing [7]. Arguably, the performance (and 'optimal laziness')

---

#### Page 18

will be correlated to the amount of training and inversely related to the complexity of the provided cost.

In conclusion, our results show that even for simple, naturalistic tasks such as center of mass estimation, the inertia against additional correction can be noticeable and is significantly modulated by trial uncertainty. At the same time, somewhat paradoxically, the effects on performance of this lack of correction are negligible, suggesting that subjects' may have been 'optimally lazy' in correcting for their mistakes, according to the minimal intervention principle [21, 23], even in the absence of performance feedback. Our findings suggest that there is no clear-cut separation between the decision making and motor component of a task, since perceptual or cognitive uncertainty affects subsequent motor behavior beyond action initiation, as the posterior distribution is used even in the adjustment period.

---

# Target Uncertainty Mediates Sensorimotor Error Correction - Backmatter

---

## Colophon

OPEN ACCESS

Citation: Acerbi L, Vijayakumar S, Wolpert DM (2017) Target Uncertainty Mediates Sensorimotor Error Correction. PLoS ONE 12(1): e0170466. doi:10.1371/journal.pone. 0170466

Editor: Gavin Buckingham, University of Exeter, UNITED KINGDOM

Received: May 4, 2016
Accepted: January 5, 2017
Published: January 27, 2017
Copyright: © 2017 Acerbi et al. This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

Data Availability Statement: All relevant data are within the paper and its Supporting Information files.

Funding: This work was supported in part by grants EP/F500385/1 and BB/F529254/1 for the University of Edinburgh School of Informatics Doctoral Training Centre in Neuroinformatics and Computational Neuroscience from the UK Engineering and Physical Sciences Research Council (EPSRC), UK Biotechnology and Biological Sciences Research Council (BBSRC), and the UK Medical Research Council (MRC) to L. Acerbi. This work was also supported by the Wellcome Trust, the Human Frontiers Science Program, and the Royal Society Noreen Murray Professorship in Neurobiology (D. M. Wolpert). S. Vijayakumar is supported through grants from Microsoft Research, Royal Academy of Engineering and EU FP7 programs. The work has made use of resources provided by the Edinburgh Compute and Data Facility, which has support from the eDIKT initiative. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

Competing Interests: I have read the journal's policy and the authors of this manuscript have the following competing interests: SV received funding from Microsoft Research as Microsoft Senior Research Fellow in Learning Robotics. This does not alter our adherence to PLOS ONE policies on sharing data and materials.

## Acknowledgments

We thank Sae Franklin and James Ingram for technical assistance.

## Author Contributions

Conceptualization: LA SV DMW.
Data curation: LA.
Formal analysis: LA.
Funding acquisition: LA SV DMW.
Investigation: LA.
Methodology: LA DMW.

---

#### Page 19

Project administration: LA.
Resources: DMW.
Software: LA.
Supervision: SV DMW.
Validation: LA.
Visualization: LA.
Writing - original draft: LA DMW.
Writing - review \& editing: LA SV DMW.

## References

1. Faisal AA, Selen LP. Wolpert DM. Noise in the nervous system. Nat Rev Neurosci. 2008; 9(4):292-303. doi: 10.1038/nrn2258 PMID: 18319728
2. Kersten D, Mamassian P, Yuille A. Object perception as Bayesian inference. Annu Rev Psychol. 2004; 55:271-304. doi: 10.1146/annurev.psych.55.090902.142005 PMID: 14744217
3. Simoncelli EP. Optimal estimation in sensory systems. In: Gazzaniga MS, editor. The cognitive neurosciences, IV edition. MIT Press; 2009. p. 525-535.
4. Ma WJ. Organizing probabilistic models of perception. Trends Cogn Sci. 2012; 16(10):511-518. doi: 10.1016/j.tics.2012.08.010 PMID: 22981359
5. Körding KP, Wolpert DM. Bayesian integration in sensorimotor learning. Nature. 2004; 427(6971):244247. doi: 10.1038/nature02169 PMID: 14724638
6. Jazayeri M, Shadlen MN. Temporal context calibrates interval timing. Nat Neurosci. 2010; 13(8):10201026. doi: 10.1038/nn. 2590 PMID: 20581842
7. Acerbi L, Wolpert DM, Vijayakumar S. Internal representations of temporal statistics and feedback calibrate motor-sensory interval timing. PLoS Comput Biol. 2012; 8(11):e1002771. doi: 10.1371/journal.pcbi. 1002771 PMID: 23209386
8. Tassinari H, Hudson TE, Landy MS. Combining priors and noisy visual cues in a rapid pointing task. J Neurosci. 2006; 26(40):10154-10163. doi: 10.1523/JNEUROSCI.2779-06.2006 PMID: 17021171
9. Berniker M, Voss M, Kording K. Learning priors for Bayesian computations in the nervous system. PloS One. 2010; 5(9):e12686. doi: 10.1371/journal.pone. 0012686 PMID: 20844766
10. Acerbi L, Vijayakumar S, Wolpert DM. On the Origins of Suboptimality in Human Probabilistic Inference. PLoS Comput Biol. 2014; 10(6):e1003661. doi: 10.1371/journal.pcbi. 1003661 PMID: 24945142
11. Weiss Y, Simoncelli EP, Adelson EH. Motion illusions as optimal percepts. Nat Neurosci. 2002; 5 (6):598-604. doi: 10.1038/nn0602-858 PMID: 12021763
12. Stocker AA, Simoncelli EP. Noise characteristics and prior expectations in human visual speed perception. Nat Neurosci. 2006; 9(4):578-585. doi: 10.1038/nn1669 PMID: 16547513
13. Girshick AR, Landy MS, Simoncelli EP. Cardinal rules: Visual orientation perception reflects knowledge of environmental statistics. Nat Neurosci. 2011; 14(7):926-932. doi: 10.1038/nn. 2831 PMID: 21642976
14. Chalk M, Seitz AR, Seriès P. Rapidly learned stimulus expectations alter perception of motion. J Vis. 2010; 10(8):1-18. doi: 10.1167/10.8.2
15. Trommershäuser J, Maloney LT, Landy MS. Statistical decision theory and the selection of rapid, goaldirected movements. J Opt Soc Am A. 2003; 20(7):1419-1433. doi: 10.1364/JOSAA.20.001419 PMID: 12868646
16. Körding KP, Wolpert DM. Bayesian decision theory in sensorimotor control. Trends Cogn Sci. 2006; 10 (7):319-326. doi: 10.1016/j.tics.2006.05.003 PMID: 16807063
17. Whiteley L, Sahani M. Implicit knowledge of visual uncertainty guides decisions with asymmetric outcomes. J Vis. 2008; 8(3):1-15. doi: 10.1167/8.3.2 PMID: 18484808
18. Landy MS, Goutcher R, Trommershäuser J, Mamassian P. Visual estimation under risk. J Vis. 2007; 7 (6):1-15. doi: 10.1167/7.6.4 PMID: 17685787
19. Seydell A, McCann BC, Trommershäuser J, Knill DC. Learning stochastic reward distributions in a speeded pointing task. J Neurosci. 2008; 28(17):4356-4367. doi: 10.1523/JNEUROSCI.0647-08.2008 PMID: 18434514

---

#### Page 20

20. Mitrovic D, Klanke S, Vijayakumar S. Adaptive optimal feedback control with learned internal dynamics models. In: Sigaud O, Peters J, editors. From Motor Learning to Interaction Learning in Robots. Springer; 2010. p. 65-84.
21. Todorov E, Jordan MI. Optimal feedback control as a theory of motor coordination. Nat Neurosci. 2002; 5(11):1226-1235. doi: 10.1038/nn963 PMID: 12404008
22. Scott SH. Optimal feedback control and the neural basis of volitional motor control. Nat Rev Neurosci. 2004; 5(7):532-546. doi: 10.1038/nrn1427 PMID: 15208695
23. Todorov E. Optimality principles in sensorimotor control. Nat Neurosci. 2004; 7(9):907-915. doi: 10. 1038/nn1309 PMID: 15332089
24. Shadmehr R, Smith MA, Krakauer JW. Error correction, sensory prediction, and adaptation in motor control. Annu Rev Neurosci. 2010; 33:89-108. doi: 10.1146/annurev-neuro-060909-153135 PMID: 20367317
25. Franklin DW, Wolpert DM. Computational mechanisms of sensorimotor control. Neuron. 2011; 72 (3):425-442. doi: 10.1016/j.neuron.2011.10.006 PMID: 22078503
26. Mitrovic D, Klanke S, Osu R, Kawato M, Vijayakumar S. A computational model of limb impedance control based on principles of internal model uncertainty. PloS One. 2010; 5(10):e13601. doi: 10.1371/ journal.pone. 0013601 PMID: 21049061
27. Rawlik K, Toussaint M, Vijayakumar S. On stochastic optimal control and reinforcement learning by approximate inference. In: Rossi F, editor. Proceedings of the Twenty-Third international joint conference on Artificial Intelligence. AAAI Press; 2013. p. 3052-3056.
28. Liu D, Todorov E. Evidence for the flexible sensorimotor strategies predicted by optimal feedback control. J Neurosci. 2007; 27(35):9354-9368. doi: 10.1523/JNEUROSCI.1110-06.2007 PMID: 17728449
29. Izawa J, Shadmehr R. On-line processing of uncertain information in visuomotor control. J Neurosci. 2008; 28(44):11360-11368. doi: 10.1523/JNEUROSCI.3063-08.2008 PMID: 18971478
30. Stevenson IH, Fernandes HL, Vilares I, Wei K. Körding KP. Bayesian integration and non-linear feedback control in a full-body motor task. PLoS Comput Biol. 2009; 5(12):e1000629. doi: 10.1371/journal. pcbi. 1000629 PMID: 20041205
31. Crevecoeur F, Scott SH. Priors engaged in long-latency responses to mechanical perturbations suggest a rapid update in state estimation. PLoS Comput Biol. 2013; 9(8):e1003177. doi: 10.1371/journal.pcbi. 1003177 PMID: 23966846
32. Nagengast AJ, Braun DA, Wolpert DM. Risk-sensitive optimal feedback control accounts for sensorimotor behavior under uncertainty. PLoS Comput Biol. 2010; 6(7):e1000857. doi: 10.1371/journal.pcbi. 1000857 PMID: 20657657
33. Braun DA, Nagengast AJ, Wolpert D. Risk-sensitivity in sensorimotor control. Front Hum Neurosci. 2011; 5:1. doi: 10.3389/fnhum.2011.00001 PMID: 21283556
34. Grau-Moya J, Ortega PA, Braun DA. Risk-sensitivity in Bayesian sensorimotor integration. PLoS Comput Biol. 2012; 8(9):e1002698. doi: 10.1371/journal.pcbi. 1002698 PMID: 23028294
35. Saunders JA, Knill DC. Humans use continuous visual feedback from the hand to control fast reaching movements. Exp Brain Res. 2003; 152(3):341-352. doi: 10.1007/s00221-003-1525-2 PMID: 12904935
36. Saunders JA, Knill DC. Visual feedback control of hand movements. J Neurosci. 2004; 24(13):32233234. doi: 10.1523/JNEUROSCI.4319-03.2004 PMID: 15056701
37. Sarlegna FR, Mutha PK. The influence of visual target information on the online control of movements. Vision Res. 2014;.
38. Dimitriou M, Wolpert DM, Franklin DW. The temporal evolution of feedback gains rapidly update to task demands. J Neurosci. 2013; 33(26):10898-10909. doi: 10.1523/JNEUROSCI.5669-12.2013 PMID: 23804109
39. Knill DC, Bondada A, Chhabra M. Flexible, task-dependent use of sensory feedback to control hand movements. J Neurosci. 2011; 31(4):1219-1237. doi: 10.1523/JNEUROSCI.3522-09.2011 PMID: 21273407
40. Oldfield RC. The assessment and analysis of handedness: The Edinburgh inventory. Neuropsychologia. 1971; 9(1):97-113. doi: 10.1016/0028-3932(71)90067-4 PMID: 5146491
41. Friedenberg J, Liby B. Perception of two-body center of mass. Percept Psychophys. 2002; 64(4):531539. doi: 10.3758/BF03194724 PMID: 12132756
42. Greenhouse SW, Geisser S. On methods in the analysis of profile data. Psychometrika. 1959; 24 (2):95-112. doi: 10.1007/BF02289823
43. Körding KP. Wolpert DM. The loss function of sensorimotor learning. Proc Natl Acad Sci USA. 2004; 101(26):9839-9842. doi: 10.1073/pnas.0308394101 PMID: 15210973

---

#### Page 21

44. O'Sullivan I, Burdet E, Diedrichsen J. Dissociating variability and effort as determinants of coordination. PLoS Comput Biol. 2009; 5(4):e1000345. doi: 10.1371/journal.pcbi. 1000345 PMID: 19360132
45. Wei K, Körding K. Uncertainty of feedback and state estimation determines the speed of motor adaptation. Front Comput Neurosci. 2010; 4. doi: 10.3389/fncom.2010.00011 PMID: 20485466
46. Drugowitsch J, Moreno-Bote R, Churchland AK, Shadlen MN, Pouget A. The cost of accumulating evidence in perceptual decision making. J Neurosci. 2012; 32(11):3612-3628. doi: 10.1523/JNEUROSCI. 4010-11.2012 PMID: 22423085
47. Orbán G, Wolpert DM. Representations of uncertainty in sensorimotor control. Current Opinion in Neurobiology. 2011; 21(4):629-635. doi: 10.1016/j.conb.2011.05.026 PMID: 21689923
48. van Dam LC, Ernst MO. Knowing Each Random Error of Our Ways, but Hardly Correcting for It: An Instance of Optimal Performance. PloS One. 2013; 8(10):e78757. doi: 10.1371/journal.pone. 0078757 PMID: 24205308
49. Burge J, Ernst MO. Banks MS. The statistical determinants of adaptation rate in human reaching. Journal of Vision. 2008; 8(4):1-19. doi: 10.1167/8.4.20 PMID: 18484859
50. Maloney LT, Mamassian P. Bayesian decision theory as a model of human visual perception: Testing Bayesian transfer. Visual Neuroscience. 2009; 26(1):147-155. doi: 10.1017/S0952523808080905 PMID: 19193251

---

# Target Uncertainty Mediates Sensorimotor Error Correction - Appendix

---

# Supporting Information

> **Image description.** A scientific figure showing movement trajectory and velocity profiles under different uncertainty conditions, organized in two panels (A and B).
>
> **Panel A** (top) displays movement trajectories along x-y coordinates (measured in cm). It contains three graphs labeled "High uncertainty (left)," "Low uncertainty," and "High uncertainty (right)." Each graph shows trajectories from approximately 20 cm height down to 0 cm, with an "Occluder" area marked in the middle portion. The trajectories show thin lines in pink and green (individual trials) and thick lines representing mean trajectories for different perturbation conditions (shown in magenta for +1.5 cm perturbation, black for no perturbation, and green for -1.5 cm perturbation, as indicated in the legend at bottom right). All trajectories originate from a gray circle at position (0,0) and extend upward with varying degrees of lateral deviation.
>
> **Panel B** (bottom) shows corresponding x-velocity profiles (measured in cm/s) over adjustment time (in seconds, from 0 to 3). Like Panel A, it contains three graphs for the three uncertainty conditions. The velocity profiles show oscillating patterns that quickly stabilize, with peak velocities reaching approximately +10 cm/s for green lines (mean perturbation -1.5 cm) and -8 cm/s for magenta lines (mean perturbation +1.5 cm). The black line (no perturbation) shows minimal velocity change. Thin light lines represent individual trials while thick colored lines show mean velocities for each perturbation condition.
>
> The figure illustrates how subjects react and adjust to different perturbation conditions under varying levels of uncertainty, with the middle "Low uncertainty" condition showing more consistent trajectories compared to the "High uncertainty" conditions on either side.

S1 Fig. Trajectory and velocity profiles. Full movement trajectory $(A)$ and velocity profiles in the adjustment phase $(B)$ for a representative subject, for respectively High-Left uncertainty (left), Low uncertainty (middle), and High-Right uncertainty (right) targets. Thick lines are mean trajectories and velocity profiles, thin lines are individual trials (subsampled for visualization). Different colors correspond to different mean perturbation levels (we show here only $-1.5,0$, and 1.5 cm ). A: Full movement trajectories. For visualization, we removed from the $x$ position the random jitter of the dumbbell (linearly from $y=0$ to $y=19 \mathrm{~cm}$ ). B: Velocity profiles along the $x$ axis during the 3 s adjustment phase. Subjects quickly reacted to the perturbation in perturbed trials, and then performed minor adjustments.
(PDF)

S1 Appendix. Additional analyses and models. Extended definitions and derivations: posterior over center of mass; posterior distribution of estimated center of mass; optimal target after adjustment with inverted Gaussian loss. Model fitting. Alternative observer models: quadratic error loss; power-law adjustment loss; miscalibrated observer model. Debriefing questionnaire.
(PDF)

S1 Dataset. Subjects' datasets. MATLAB (MathWorks) file containing all sixteen subjects' datasets for training and test session.
(MAT)

---

#### Page 1

# Target Uncertainty Mediates Sensorimotor Error Correction Supporting Information: Appendix S1 Additional analyses and models

Luigi Acerbi, Sethu Vijayakumar and Daniel M. Wolpert

## Contents

1 Extended definitions and derivations ..... 1
1.1 Posterior over center of mass ..... 1
1.2 Posterior distribution of estimated center of mass ..... 2
1.3 Optimal target after adjustment with inverted Gaussian loss ..... 3
2 Model fitting ..... 3
3 Alternative observer models ..... 4
3.1 Quadratic error loss ..... 4
3.2 Power-law adjustment loss ..... 4
3.3 Miscalibrated observer model ..... 4
4 Debriefing questionnaire ..... 6
Supplemental References ..... 6
1 Extended definitions and derivations
1.1 Posterior over center of mass

The posterior over center of mass is represented by Eq. 4 in the main text, where we defined the mixing weights, means and variances as:

$$
\begin{aligned}
& Z^{(i)} \equiv \mathcal{N}\left(\left.\rho_{m}\right|_{\text {prior }}(i) \sigma_{\text {prior }}^{(i)}+\sigma_{\rho}^{2}\right) \quad \text { and } \quad \mathcal{Z} \equiv \sum_{i=1}^{3} Z^{(i)} \quad \text { (normalization constant) } \\
& \mu_{\text {post }}^{(i)} \equiv \frac{\rho_{m} \sigma_{\text {prior }}^{(i) 2}+\mu_{\text {prior }}^{(i)} \sigma_{\rho}^{2}}{\sigma_{\text {prior }}^{(i) 2}+\sigma_{\rho}^{2}}, \quad \sigma_{\text {post }}^{2} \equiv \frac{\sigma_{\text {prior }}^{(i) 2} \sigma_{\rho}^{2}}{\sigma_{\text {prior }}^{(i) 2}+\sigma_{\rho}^{2}}, \quad \text { for } i=1,2,3
\end{aligned}
$$

where $\rho_{m}$ is the noisy measurement of log disks ratio, $\sigma_{\rho}^{2}$ the noise variance, and $\mu_{\text {prior }}^{(i)}, \sigma_{\text {prior }}^{(i) 2}$ are, respectively, mean and variance of the $i$-th mixture component of the prior over log disks ratios (for $i=1,2,3)$.

---

#### Page 2

# 1.2 Posterior distribution of estimated center of mass

The horizontal location of the center of mass $s$ of two $D$-dimensional spheres with radii $r_{1}$ and $r_{2}$, separated by a distance $\ell$, is given by:

$$
s=\frac{\ell}{2} \cdot\left(\frac{r_{2}^{D}-r_{1}^{D}}{r_{1}^{D}+r_{2}^{D}}\right)
$$

with respect to the midpoint of the target line. From Eq. S1, the relationship between the log ratio of the radii of two spherical objects in a $D$-dimensional space and their center of mass, $s$, can be represented by the mapping $f_{D}$ :

$$
f_{D}(\rho)=\frac{\ell}{2} \cdot\left[\frac{(-1) \cdot r_{1}^{D}+(+1) \cdot r_{2}^{D}}{r_{1}^{D}+r_{2}^{D}}\right]=\frac{\ell}{2} \cdot\left[\frac{\frac{r^{D}}{r_{2}^{D}}-1}{\frac{r_{2}^{D}}{r_{1}^{D}}+1}\right]=\frac{\ell}{2} \cdot\left[\frac{e^{D \rho}-1}{e^{D \rho}+1}\right]
$$

whose inverse is $f_{D}^{-1}(s)=\frac{1}{D} \log [(\ell / 2+s) /(\ell / 2-s)]$. We assume that the observer uses the mapping $f_{D}$ from Eq. S2 with some fixed value of $D>0$, although not necessarily the correct value $D=2$ for two-dimensional disks, nor we restrict $D$ to be an integer.

Combining Eqs. 4 of the main text with Eq. S2, the posterior distribution of the location of the estimated center of mass takes the form:

$$
\begin{aligned}
q_{\text {post }}\left(s \mid \rho_{m}\right) & =\int_{-\infty}^{\infty} \delta\left[f_{D}(\rho)-s\right] q_{\text {post }}\left(\rho \mid \rho_{m}\right) d \rho \\
& =\frac{1}{\mathcal{Z}} \sum_{i=1}^{3} Z^{(i)} \int_{-\infty}^{\infty} \delta\left(\rho^{\prime}-s\right) \mathcal{N}\left(\left.f_{D}^{-1}\left(\rho^{\prime}\right) \right\rvert\, \rho_{\text {post }}^{(i)}, \sigma_{\text {post }}^{(i)}^{2}\right) \frac{d \rho^{\prime}}{\frac{d f_{D}}{d \rho}\left(f^{-1}\left(\rho^{\prime}\right)\right)} \\
& =\frac{1}{\mathcal{Z}} \sum_{i=1}^{3} Z^{(i)} \mathcal{N}\left(\frac{1}{D} \log \left[\frac{\ell / 2+s}{\ell / 2-s}\right] \mid \rho_{\text {post }}^{(i)}, \sigma_{\text {post }}^{(i)}^{2}\right) \frac{\ell}{D \cdot\left(\ell^{2} / 4-s^{2}\right)}
\end{aligned}
$$

where in the second passage we have performed the change of variable $\rho^{\prime}=f_{D}(\rho)$.
The posterior in Eq. S3 is not a mixture of Gaussian distributions due to the nonlinear relationship in the argument of the Gaussian function. To allow efficient computation of optimal behavior we can approximate each mixture component with a Gaussian:

$$
q_{\text {post }}\left(s \mid \rho_{m}\right) \approx \frac{1}{\mathcal{Z}} \sum_{i=1}^{3} Z^{(i)} \mathcal{N}\left(s \mid m_{\text {post }}^{(i)}, s_{\text {post }}^{(i)^{2}}\right)
$$

where $m_{\text {post }}^{(i)}$ and $s_{\text {post }}^{(i)}$ are respectively the mean and SD of the mixture components in Eq. S3 (computed numerically, function trapz in MATLAB).

The 'Gaussianized' components in Eq. S4 are a very good approximation of their counterparts in Eq. S3 for the parameters in our task, as measured by an average Kullback-Leibler divergence of $(3.1 \pm 0.7)$. $10^{-3}$ nats (mean $\pm \mathrm{SD}$ across all posteriors in the task). This amounts to roughly the KL divergence between two Gaussians with same variance and whose difference in means is one-twelfth of their SD. The approximation we chose works better than a Laplace approximation [1], which yields worse values for the KL divergence of $(7.8 \pm 1.8) \cdot 10^{-3}$ nats.

---

#### Page 3

# 1.3 Optimal target after adjustment with inverted Gaussian loss

Combining Eqs. 5 and 8 of the main text with inverted Gaussian loss functions for both error and adjustment,

$$
\mathcal{L}_{\mathrm{err}}(\hat{s}-s)=-\exp \left\{-\frac{(\hat{s}-s)^{2}}{2 \sigma_{\mathrm{err}}^{2}}\right\}, \quad \mathcal{L}_{\mathrm{adj}}(\hat{r}-r)=-\exp \left\{-\frac{(\hat{r}-r)^{2}}{2 \sigma_{\mathrm{adj}}^{2}}\right\}
$$

we obtain the final expression for the 'optimal' target after adjustment,

$$
\begin{aligned}
s^{*}\left(\rho_{m}, \tilde{r}\right) & =\arg \min _{\hat{s}}\left[\alpha \mathcal{L}_{\mathrm{adj}}(\hat{s}-\tilde{r})+\int_{-\ell / 2}^{\ell / 2} q_{\mathrm{post}}\left(s \mid \rho_{m}\right) \mathcal{L}_{\mathrm{err}}(\hat{s}-s) d s\right] \\
& =\arg \min _{\hat{s}}\left[-\alpha \sqrt{2 \pi} \sigma_{\mathrm{adj}} \mathcal{N}\left(\hat{s} \mid \tilde{r}, \sigma_{\mathrm{adj}}^{2}\right)-\sqrt{2 \pi} \sigma_{\mathrm{err}} \int_{-\ell / 2}^{\ell / 2} q_{\mathrm{post}}\left(s \mid \rho_{m}\right) \mathcal{N}\left(s \mid \hat{s}, \sigma_{\mathrm{err}}^{2}\right) d s\right] \\
& =\arg \min _{\hat{s}}\left[-\widetilde{\alpha} \mathcal{N}\left(\hat{s} \mid \tilde{r}, \sigma_{\mathrm{adj}}^{2}\right)-\sum_{i=1}^{3} Z^{(i)} \mathcal{N}\left(\hat{s} \mid m_{\mathrm{post}}^{(i)}, s_{\mathrm{post}}^{(i) 2}+\sigma_{\mathrm{err}}^{2}\right)\right]
\end{aligned}
$$

where we have collected all scaling factors in $\widetilde{\alpha} \equiv \alpha \mathcal{Z} \sigma_{\text {adj }} / \sigma_{\text {err }}$, by using the property that the arg min is invariant to rescaling of its argument by a constant.

In order to find $s_{\text {pre }}^{*}$ (Eq. 6 in the main text) and subsequently the 'optimal' final position $s^{*}$ (Eq. S5), we used gmm1max, a fast numerical algorithm for finding the mode of a mixture of Gaussians [2,3], implemented in the gmm1 package for MATLAB (https://github.com/lacerbi/gmm1), as both equations have no known analytical solution.

## 2 Model fitting

Using Eq. 9 in the main text for the probability of response in a given trial, we estimated the model parameters for each subject by maximizing the (log) likelihood of the data:

$$
\log \mathcal{L}(\boldsymbol{\theta})=\sum_{i=1}^{N} \log \left[\operatorname{Pr}^{*}\left(r^{(i)} \mid \rho^{(i)}, b^{(i)} ; \boldsymbol{\theta}\right)\right]
$$

where $N=576$ is the total number of trials and we have assumed conditional independence between trials. We modified Eq. 9 in the main text with a tiny probability $\epsilon$ to improve robustness of the inference:

$$
\operatorname{Pr}^{*}(r \mid \rho, b ; \boldsymbol{\theta})=(1-\epsilon) \operatorname{Pr}(r \mid \rho, b ; \boldsymbol{\theta})+\epsilon
$$

where $\epsilon=1.5 \cdot 10^{-5}$ is the value of the pdf of a normal distribution five SDs away from the mean. Eq. S7 allows for a very small lapse rate that prevents a few outliers in a subject's dataset from having a large effect on the log likelihood of the data in Eq. S6.

Maximum likelihood fits were obtained via the Nelder-Mead simplex method (fminsearch in MATLAB), with ten randomized starting points per dataset. The log likelihood landscapes proved to be relatively simple, with almost all optimization runs converging to the same optimum within each dataset.

---

#### Page 4

# 3 Alternative observer models

### 3.1 Quadratic error loss

As an alternative model of loss for the error term in Eq. 7 of the main text, we considered a quadratic loss function: $\mathcal{L}_{\text {quad }}(r-s)=(r-s)^{2}$. In this case, the 'optimal' target (Eq. 8 in the main text, and see also Eq. S5 for comparison) takes the form:

$$
\begin{aligned}
s^{*}\left(\rho_{m}, x_{0}\right) & =\arg \min _{\hat{s}}\left[\alpha \mathcal{L}_{\mathrm{adj}}(\hat{s}-\bar{r})+\int_{-\ell / 2}^{\ell / 2} q_{\mathrm{post}}\left(s \mid \rho_{m}\right)(\hat{s}-s)^{2} d s\right] \\
& =\arg \min _{\hat{s}}\left[\alpha \mathcal{L}_{\mathrm{adj}}(\hat{s}-\bar{r})+\hat{s}^{2}-2 \hat{s} \cdot m_{\mathrm{post}}\right]
\end{aligned}
$$

where $m_{\text {post }}$ is the mean of the full posterior in Eqs. S3 and S4. For example, for a quadratic adjustment loss, Eq. S8 reduces to a simple analytical solution: $s^{*}\left(\rho_{m}, \bar{r}\right)=\left(m_{\text {post }}+\alpha \bar{r}\right) /(1+\alpha)$. However, for any shape of the adjustment loss, Eq. S8 predicts that the optimal target, and therefore the amount of correction to a perturbation, does not depend on the variance (i. e., on the uncertainty) of the posterior distribution [4]. This prediction is at odds with the patterns observed in the experiment, so the quadratic loss model of the error is unable to account for the key feature of our data.

### 3.2 Power-law adjustment loss

To explore alternative models of adjustment cost in Eq. 7 of the main text, we assume an inverted Gaussian for the error combined with a power function for the adjustment loss. A power function with power $\nu>0$ includes several common models of loss as specific cases (absolute, sub-quadratic, quadratic) and therefore represents a valid alternative to the inverted Gaussian [5]. For this case, the 'optimal' target is:

$$
s^{*}\left(\rho_{m}, x_{0}\right)=\arg \min _{\hat{s}}\left[\frac{\alpha \mathcal{Z}}{\sigma_{\mathrm{err}}}|\hat{s}-\bar{r}|^{\nu}-\sum_{i=1}^{3} Z^{(i)} \mathcal{N}\left(\left.\hat{s}\right|_{m_{\mathrm{post}}}, s_{\mathrm{post}}^{(i) 2}+\sigma_{\mathrm{err}}^{2}\right)\right]
$$

We compared the performances of the observer models with power loss and inverted Gaussian loss in terms of (log) maximum likelihood. We do not need to consider a term for complexity as the two models have the same number of parameters. The performance of the two models was largely similar, with a nonsignificant advantage of the inverted Gaussian model $(0.7 \pm 1.6$ difference in average log likelihood; $t$-test $t_{(15)}=0.44, p>.60, d=0.11)$. An analogous lack of empirical difference between the inverted Gaussian loss and the power loss was reported in a previous study [5]. The choice between the two functions must therefore be driven by other considerations, such as mathematical tractability (e.g., in our case the inverted Gaussian loss allows to write the expected loss in closed form, see Eq. S5).

### 3.3 Miscalibrated observer model

Finally, we examined the predictions for an observer model built from a different set of assumptions. For this model, we hypothesized that the source of the lack of correction is not effort, but a miscalibration of the perceived position of the cursor. Even though in our task the visual feedback of the cursor location during the adjustment phase is unambiguous, subject's perception may be systematically altered by proprioceptive feedback according to the relative reliability of vision and proprioception (see $[6,7]$ ). In particular, the posterior distribution of cursor position $r$ for visual cue $x_{\text {vis }}$ and proprioceptive cue position

---

#### Page 5

$x_{\text {prop }}$ is given by [8]:

$$
\begin{aligned}
q_{\text {cursor }}\left(r \mid x_{\text {vis }}, x_{\text {prop }}\right) & =\mathcal{N}\left(r \mid x_{\text {vis }}, \sigma_{\text {vis }}^{2}\right) \mathcal{N}\left(r \mid x_{\text {prop }}, \sigma_{\text {prop }}^{2}\right) \\
& \propto \mathcal{N}\left(r \mid \frac{x_{\text {vis }} \sigma_{\text {prop }}^{2}+x_{\text {prop }} \sigma_{\text {vis }}^{2}}{\sigma_{\text {vis }}^{2}+\sigma_{\text {prop }}^{2}}, \frac{\sigma_{\text {vis }}^{2} \sigma_{\text {prop }}^{2}}{\sigma_{\text {vis }}^{2}+\sigma_{\text {prop }}^{2}}\right)
\end{aligned}
$$

where $\sigma_{\text {vis }}^{2}$ and $\sigma_{\text {prop }}^{2}$ are the (subjective) variances of visual and proprioceptive noise. Noting that in our setup $x_{\text {prop }} \approx x_{\text {vis }}-b$, where $b$ is the visual perturbation applied in the trial, we can rewrite Eq. S10 as:

$$
\begin{aligned}
q_{\text {cursor }}\left(r \mid x_{\text {vis }}, x_{\text {prop }}\right) & \propto \mathcal{N}\left(r \mid x_{\text {vis }}-b \frac{\sigma_{\text {vis }}^{2}}{\sigma_{\text {vis }}^{2}+\sigma_{\text {prop }}^{2}}, \frac{\sigma_{\text {vis }}^{2} \sigma_{\text {prop }}^{2}}{\sigma_{\text {vis }}^{2}+\sigma_{\text {prop }}^{2}}\right) \\
& \equiv \mathcal{N}\left(r \mid x_{\text {vis }}-\mu_{\text {cursor }}, \sigma_{\text {cursor }}^{2}\right)
\end{aligned}
$$

where we have defined $\mu_{\text {cursor }}$ and $\sigma_{\text {cursor }}$ for notational convenience. We assume now that, according to BDT, the observer places the visual cursor so as to match the 'optimal' visual cursor location $x_{\text {vis }}^{*}$ that minimizes the expected loss of the task:

$$
\begin{aligned}
x_{\text {vis }}^{*}\left(\rho_{m}\right) & =\arg \min _{\hat{x}_{\text {vis }}}\left[-\int_{-\ell / 2}^{\ell / 2} q_{\text {post }}\left(s \mid \rho_{m}\right) \mathcal{N}\left(r \mid \hat{x}_{\text {vis }}-\mu_{\text {cursor }}, \sigma_{\text {cursor }}^{2}\right) \mathcal{N}\left(r \mid s, \sigma_{\text {score }}^{2}\right) d s d r\right] \\
& =\arg \min _{\hat{x}_{\text {vis }}}\left[-\int_{-\ell / 2}^{\ell / 2} q_{\text {post }}\left(s \mid \rho_{m}\right) \mathcal{N}\left(s \mid \hat{x}_{\text {vis }}-\mu_{\text {cursor }}, \sigma_{\text {cursor }}^{2}+\sigma_{\text {score }}^{2}\right) d s\right]
\end{aligned}
$$

where the subjective error in the loss function is computed with respect to the unknown exact cursor position $r$, whose distribution is inferred via cue combination (Eq. S11). The posterior over target locations, $q_{\text {post }}\left(s \mid \rho_{m}\right)$, is defined by the mixture of Gaussians of Eq. S4. For the sake of argument, let us consider the case in which the posterior is mainly represented by a single Gaussian component with mean $m_{\text {post }}$ and variance $s_{\text {post }}^{2}$ (both depend on $\rho_{m}$ ). This is a reasonable approximation in most of the trials as our observers' sensory noise on the estimation of the disks' (log) ratio, $\sigma_{\rho}$, is much smaller than the separation between components of the prior over (log) ratios $(0.063 \ll 0.405$, see main text). Using a single Gaussian in Eq. S12 we obtain:

$$
\begin{aligned}
x_{\text {vis }}^{*}\left(\rho_{m}\right) & =\arg \min _{\hat{x}_{\text {vis }}}\left[-\int_{-\ell / 2}^{\ell / 2} \mathcal{N}\left(s \mid m_{\text {post }}, s_{\text {post }}^{2}\right) \mathcal{N}\left(s \mid \hat{x}_{\text {vis }}-\mu_{\text {cursor }}, \sigma_{\text {cursor }}^{2}+\sigma_{\text {score }}^{2}\right) d s\right] \\
& =\arg \min _{\hat{x}_{\text {vis }}}\left[-\mathcal{N}\left(\hat{x}_{\text {vis }} \mid m_{\text {post }}+\mu_{\text {cursor }}, \sigma_{\text {cursor }}^{2}+\sigma_{\text {score }}^{2}+s_{\text {post }}^{2}\right)\right] \\
& =m_{\text {post }}+b \frac{\sigma_{\text {vis }}^{2}}{\sigma_{\text {vis }}^{2}+\sigma_{\text {prop }}^{2}}
\end{aligned}
$$

where we used the definition of $\mu_{\text {cursor }}$ from Eq. S11. Similarly to the quadratic error loss model, the solution in Eq. S13 does not depend on the variance of the posterior $s_{\text {post }}^{2}$, meaning that for the majority of trials the observer model does not present the features that we observe in our data. Still, for trials with multimodal posteriors, the solution of Eq. S12 might depend on the uncertainty in the trial. If so, albeit unlikely, these trials alone might be enough to produce the effect that we observe in our data. We, therefore, analyzed the behavior that this observer model would predict for our observers for a range of reasonable values of the parameters $\sigma_{\text {vis }}$ and $\sigma_{\text {prop }}$ from 0.1 cm to 2.5 cm [6]. For all combinations of parameters and for all subjects we did not find any sign of interaction between trial uncertainty and residual error, confirming the results of our first approximation (Eq. S13). In conclusion, subjects' perception of the perturbed cursor position may be altered by proprioceptive feedback, but this effect alone cannot account for the uncertainty-dependent modulation of the residual error.

---

#### Page 6

# 4 Debriefing questionnaire

Participants were presented with a short debriefing questionnaire at the end of the test session. The questionnaire asked:

1. Did you notice any difference between trials in which you received performance feedback and trials in which you did not receive performance feedback? [Yes / No]
2. Did you notice any discrepancy between location of the cursor and position of your fingertip in one or more trials during this session? [Yes / No]

If the answer was 'Yes' to any question, subjects could write a short explanation. Out of 16 subjects, one did not fill in the questionnaire; 13 subjects answered 'No' to the 1st question; and 15 subjects answered 'No' to the 2 nd question. One of the participants who had answered 'Yes' to the 1st question wrote that "After two/three trials with feedback it was harder". Participants reacted with surprise when they were explained the experimental manipulation, after they had completed the questionnaire. Overall, we take these results as evidence that subjects were unaware of cursor displacement or of any other difference between trials with or without performance feedback.

## Supplemental References

1. MacKay DJ. Information theory, inference and learning algorithms. Cambridge University Press; 2003 .
2. Carreira-Perpiñán MÁ. Mode-finding for mixtures of Gaussian distributions. IEEE T Pattern Anal. 2000;22(11):1318-1323.
3. Acerbi L, Vijayakumar S, Wolpert DM. On the Origins of Suboptimality in Human Probabilistic Inference. PLoS Comput Biol. 2014;10(6):e1003661.
4. Grau-Moya J, Ortega PA, Braun DA. Risk-sensitivity in Bayesian sensorimotor integration. PLoS Comput Biol. 2012;8(9):e1002698.
5. Körding KP, Wolpert DM. The loss function of sensorimotor learning. Proc Natl Acad Sci USA. 2004;101(26):9839-9842.
6. van Beers RJ, Sittig AC, van der Gon JJD. How humans combine simultaneous proprioceptive and visual position information. Exp Brain Res. 1996;111(2):253-261.
7. van Beers RJ, Sittig AC, van der Gon JJD. Integration of proprioceptive and visual positioninformation: An experimentally supported model. J Neurophysiol. 1999;81(3):1355-1364.
8. Ernst MO, Banks MS. Humans integrate visual and haptic information in a statistically optimal fashion. Nature. 2002;415(6870):429-433.