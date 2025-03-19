```
@article{acerbi2012internal,
  title={Internal Representations of Temporal Statistics and Feedback Calibrate Motor-Sensory Interval Timing},
  author={Luigi Acerbi and Daniel M. Wolpert and Sethu Vijayakumar},
  year={2012},
  journal={PLoS Computational Biology},
  doi={10.1371/journal.pcbi.1002771},
}
```

---

#### Page 1

# Internal Representations of Temporal Statistics and Feedback Calibrate Motor-Sensory Interval Timing

Luigi Acerbi ${ }^{1,2}$, Daniel M. Wolpert ${ }^{3}$, Sethu Vijayakumar ${ }^{1}$<br>1 Institute of Perception, Action and Behaviour, School of Informatics, University of Edinburgh, Edinburgh, United Kingdom, 2 Doctoral Training Centre in Neuroinformatics and Computational Neuroscience, School of Informatics, University of Edinburgh, Edinburgh, United Kingdom, 3 Computational and Biological Learning Lab, Department of Engineering, University of Cambridge, Cambridge, United Kingdom

#### Abstract

Humans have been shown to adapt to the temporal statistics of timing tasks so as to optimize the accuracy of their responses, in agreement with the predictions of Bayesian integration. This suggests that they build an internal representation of both the experimentally imposed distribution of time intervals (the prior) and of the error (the loss function). The responses of a Bayesian ideal observer depend crucially on these internal representations, which have only been previously studied for simple distributions. To study the nature of these representations we asked subjects to reproduce time intervals drawn from underlying temporal distributions of varying complexity, from uniform to highly skewed or bimodal while also varying the error mapping that determined the performance feedback. Interval reproduction times were affected by both the distribution and feedback, in good agreement with a performance-optimizing Bayesian observer and actor model. Bayesian model comparison highlighted that subjects were integrating the provided feedback and represented the experimental distribution with a smoothed approximation. A nonparametric reconstruction of the subjective priors from the data shows that they are generally in agreement with the true distributions up to third-order moments, but with systematically heavier tails. In particular, higher-order statistical features (kurtosis, multimodality) seem much harder to acquire. Our findings suggest that humans have only minor constraints on learning lower-order statistical properties of unimodal (including peaked and skewed) distributions of time intervals under the guidance of corrective feedback, and that their behavior is well explained by Bayesian decision theory.

## Introduction

The ability to estimate motor-sensory time intervals in the subsecond range and react accordingly is fundamental in many behaviorally relevant circumstances [1], such as dodging a blow or assessing causality ("was it me producing that noise?"). Since sensing of time intervals is inherently noisy [2], it is typically advantageous to enhance time estimates with previous knowledge of the temporal context. It has been shown in various timing experiments that humans can take into account some relevant temporal statistics of a task according to Bayesian decision theory, such as in sensorimotor coincidence timing [3], tactile simultaneity judgements [4], planning movement duration [5] and time interval estimation $[6-8]$.

Most of these studies $[3,4,6,8]$ exposed the participants to time intervals whose duration followed some simple distribution (e.g. a Gaussian or a uniform distribution), and then assumed that the subjects' internal representation of it corresponded to the experimental distribution. As a more realistic working hypothesis, we can expect the observers to have acquired, after training, an internal representation of the statistics of the temporal intervals which is an approximation of the true, objective experimental
distribution. It can be argued that this approximation in most cases would be 'similar enough' to the true distribution, so that in practice the distinction between subjective and objective distribution is an unnecessary complication. This is not exact though, first of all because it is unknown whether the similarity assumption would hold for complex temporal distributions, and secondly because the specific form of the approximation can lead to observable differences in behavior even for simple cases (see Figure 1).

We propose that understanding how humans learn and approximate temporal statistics in a given context can help explaining observed temporal biases and illusions [9]. Previous studies have shown that human observers exhibit specific idiosyncrasies in judging simultaneity and temporal order of stimuli after repeated exposure to a specific inter-stimulus lag (temporal recalibration) $[4,10,11]$, in encoding certain kinds of temporal distributions in the subsecond range [12] or in estimating durations of very rare stimuli (oddballs) [13], so it is worth asking whether people are able to acquire an internal representation of complex (e.g. very peaked, bimodal) distributions of inter-stimulus intervals in the first place, and what are their limitations.

---

#### Page 2

## Author Summary

Human performance in a timing task depends on the context of recently experienced time intervals. In fact, people may use prior experience to improve their timing performance. Given the relevance of time for both sensing and acting in the world, how humans learn and represent temporal information is a fundamental question in neuroscience. Here, we ask subjects to reproduce the duration of time intervals drawn from different distributions (different temporal contexts). We build a set of models of how people might behave in such a timing task, depending on how they are representing the temporal context. Comparison between models and data allows us to establish that in general subjects are integrating taskrelevant temporal information with the provided error feedback to enhance their timing performance. Analysis of the subjects' responses allows us to reconstruct their internal representation of the temporal context, and we compare it with the true distribution. We find that with the help of corrective feedback humans can learn good approximations of unimodal distributions of time intervals used in the experiment, even for skewed distributions of durations; on the other hand, under similar conditions, we find that multimodal distributions of timing intervals are much harder to acquire.

Bayesian decision theory (BDT) provides a neat and successful framework for representing the internal beliefs of an ideal observer in terms of a (subjective) prior distribution, and it gives a normative account on how the ideal observer should take action [14]. A large number of behavioral studies are consistent with a Bayesian interpretation [15-17] and some results suggest that human subjects build internal representations of priors and likelihoods $[15,18,19]$ or likelihood and loss functions [20]. We therefore adopted BDT as a framework to infer the subjects' acquired beliefs about the experimental distributions. However, the behavior of a Bayesian ideal observer depends crucially not only on the prior, but also on the likelihoods and the loss function, with an underlying degeneracy, i.e. distinct combinations of distributions can lead to the same empirical behavior [21]. It follows that a proper analysis of the internal representations cannot be separated from an appropriate modelling of the likelihoods and the loss function as well.

With this in mind, we analyzed the timing responses of human observers for progressively more complex temporal distributions of durations in a motor-sensory time interval reproduction task. We provided performance feedback (also known as 'knowledge of results', or KR) on a trial-by-trial basis, which constrained the loss function, speeded up learning and allowed the subjects to adjust their behavior, therefore providing an upper bound on human performance [22,23]. We carried out a full Bayesian model comparison analysis among a discrete set of candidate likelihoods, priors and loss functions in order to find the observer model most supported by the data, characterizing the behavior of each individual subject across multiple conditions. Having inferred the form of the likelihoods and loss functions for each subject, we could then perform a nonparametric reconstruction [24] of what the subjects' prior distributions would look like under the assumptions of our framework and we compared them with the experimental distributions. The inferred priors suggest that people learn smoothed approximations of the experimental distributions which take into account not only mean and variance but also higher-order statistics, although some complex features (kurtosis,
bimodality) seem to deviate systematically from those of the experimental distribution.

## Results

Subjects took part in a time interval reproduction task with performance feedback (trial structure depicted in Figure 2 top; see Methods for full details). On each trial subjects clicked a mouse button and, after a time interval ( $x \mathrm{~ms}$ ) that could vary from trial-to-trial, saw a yellow dot flash on the screen. They were then required to hold down the mouse button to reproduce the perceived interval between the original click and the flash. The duration of this mouse press constituted their response ( $r \mathrm{~ms}$ ) for their trial. Subjects received visual feedback on their performance, with an error bar that was displayed either to the left or right of a central zero-error line, depending on whether their response was shorter or longer than the true interval duration. In different experimental blocks we varied both the statistical distribution of the intervals, $p(x)$, and the nature of the performance feedback, i.e. mapping between the interval/response pair and the error display, $f(x, r)$, relative to the zero-error line. For each experimental block, subjects first performed training sessions until their performance was stable (around 500 to 1500 trials), followed by two test sessions (about 500 trials per session). Testing with a block was completed before starting a new one.

Different groups of subjects took part in five experiments, whose setup details are summarized in Table 1 (see also Methods). In brief, Experiment 1 represented a basic test for the experimental paradigm and modelling framework with simple (Uniform) distributions over different ranges. Experiment 2 compared subjects' responses in a simple condition (Uniform) vs a complex one (Peaked, one interval was over-represented), over the same range of intervals. Experiment 3 verified the effect of feedback on subjects' responses by imposing a different error mapping $f(x, r)$. Experiment 4 tested subjects in a more extreme version of the Peaked distribution. Experiment 5 verified the limits of subjects' capability of learning with bimodal distributions of intervals.

We first present the results of the first two experiments in a qualitative manner, and then describe a quantitative model. Results of the other three experiments that test specific aspects of the model or more complex distributions are presented thereafter.

## Experiment 1: Uniform distributions over different ranges

In the first experiment the distribution of time intervals consisted of a set of six equally spaced discrete times with equal probability according to either a Short Uniform ( $450-825 \mathrm{~ms}$ ) or Long Uniform ( $750-1125 \mathrm{~ms}$ ) distribution. The order of these blocks was randomized across subjects. The feedback followed a Skewed error mapping $f_{S k} \propto \frac{r-x}{x}$. The 'artificial' responsedependent asymmetry in the Skewed mapping was chosen to test whether participants would integrate the provided feedback error into their decision process, as opposed to other possibly more natural forms of error, such as the Standard error $f_{S t} \propto r-x$ or the Fractional error $f_{F t} \propto \frac{r-x}{x}$ (see later, Bayesian model comparison).

We examined the mean bias in the response (mean reproduction interval minus actual interval, $\bar{r}-x$, also termed 'constant error' in the psychophysical literature), as a function of the actual interval (Figure 3 top). Subjects' responses showed a regression to the mean consistent with a Bayesian process that integrates the prior with sensory evidence $[4,6,8,15]$. That is, little bias was seen for intervals that matched the mean of the prior ( 637.5 ms for

---

#### Page 3

> **Image description.** The image is a figure comparing response profiles for different ideal observers in a timing task. It is organized as a 4x6 grid of subplots, labeled a, b, c, and d across the top, and "Stimuli," "Likelihood," "Prior," "Loss function," "Response bias (ms)," and "Response sd (ms)" down the left side.
>
> - **Top Row (Stimuli):** Each subplot shows a bar graph representing stimuli durations. The x-axis is labeled with values 600, 750, 900, and 1050. The bars are colored green, gray, purple, and magenta, respectively.
>
> - **Second Row (Likelihood):** Each subplot shows a set of overlapping Gaussian curves. The curves are colored green, gray, purple, and magenta, corresponding to the stimuli durations. In columns b, c, and d, the background is shaded yellow.
>
> - **Third Row (Prior):** Each subplot shows a filled curve representing a prior distribution. In columns a and b, the curve is bell-shaped (Gaussian) and shaded gray. In columns c and d, the curve is rectangular (uniform) and shaded gray, with a yellow background.
>
> - **Fourth Row (Loss function):** Each subplot shows a set of curves resembling parabolas. The curves are colored green, gray, purple, and magenta, corresponding to the stimuli durations. In column d, the background is shaded yellow.
>
> - **Fifth Row (Response bias (ms)):** Each subplot shows a scatter plot with a line. The x-axis is labeled "Time interval (ms)" with values 600, 750, 900, and 1050. The y-axis is labeled "Response bias (ms)" with values ranging from -50 to 50. Data points are colored green, gray, purple, and magenta, corresponding to the stimuli durations.
>
> - **Sixth Row (Response sd (ms)):** Each subplot shows a scatter plot with a line. The x-axis is labeled "Time interval (ms)" with values 600, 750, 900, and 1050. The y-axis is labeled "Response sd (ms)" with values ranging from 50 to 100. Data points are colored green, gray, purple, and magenta, corresponding to the stimuli durations.
>
> The columns a, b, c, and d represent different ideal observer models, each with a different combination of temporal sensorimotor noise (likelihood), prior expectations, and loss functions. Yellow shading highlights the changes of each model (column) from model (a).

Figure 1. Comparison of response profiles for different ideal observers in the timing task. The responses of four different ideal observers (columns a-d) to a discrete set of possible stimuli durations are shown (top row); for visualization purpose, each stimulus duration in this plot is associated with a specific color. The behavior crucially depends on the combination of the modelled observer's temporal sensorimotor noise (likelihood), prior expectations and loss function (rows 2-4); see Figure 2 bottom for a description of the observer model. For instance, the observer's sensorimotor variability could be constant across all time intervals (column a) or grow linearly in the interval, according to the 'scalar' property of interval timing (column b-d). An observer could be approximating the true, discrete distribution of intervals as a Gaussian (columns a-b) or with a uniform distribution (columns c-d). Moreover, the observer could be minimizing a typical quadratic loss function (columns a-c) or a skewed cost imposed through an external source of feedback (column d). Yellow shading highlights the changes of each model (column) from model (a). All changes to the observer's model components considerably affect the statistics of the predicted responses, summarized by response bias, i.e. average difference between the response and true stimulus duration, and standard deviation (bottom two rows). For instance, all models predict a central tendency in the response (that is, a bias that shifts responses approximately towards the center of the interval range), but bias profiles show characteristic differences between models.

doi:10.1371/journal.pcbi. 1002771 .g001

---

#### Page 4

Time

> **Image description.** This image is a diagram illustrating a time interval reproduction task and its corresponding generative model. The diagram is divided into two rows labeled "Experiment" and "Generative Model" on the left side. The columns are labeled "Stimulus", "Response", and "Feedback" from left to right, indicating the progression of the task.
>
> The top row, "Experiment", depicts the experimental procedure:
>
> - **Stimulus:** A hand clicks a mouse button. A yellow dot then flashes on a screen. The time interval between the click and the flash is denoted as "x ms".
> - **Response:** The subject presses the mouse button for a duration of "r ms" and then releases it.
> - **Feedback:** An "Error Display" shows a horizontal bar with two yellow markers, presumably representing the target interval and the reproduced interval. The function "f(x, r)" is associated with this stage.
>
> The bottom row, "Generative Model", describes the underlying statistical model:
>
> - **Estimation:** The "True interval" x is drawn from a probability distribution p(x), indicated by "x ~ p(x)". This interval leads to a "Noisy sensory measurement" y, modeled as "y ~ ps(y|x;ws)".
> - **Reproduction:** The "Optimal action" u*(y) is determined based on the sensory measurement. A black arrow points from the "Noisy sensory measurement" box to the "Optimal action" box. This action leads to a "Noisy motor response" r, modeled as "r ~ pm(r/u*(y);wm)".
> - **Loss function:** The "Loss function" is represented as "f²(x, r)". A dashed arrow points from the "Optimal action" box to the "Loss function" box.
>
> The entire diagram is oriented along a horizontal "Time" axis at the top. The background of each cell is light gray.

Figure 2. Time interval reproduction task and generative model. Top: Outline of a trial. Participants clicked on a mouse button and a yellow dot was flashed $x$ ms later at the center of the screen, with $x$ drawn from a block-dependent distribution (estimation phase). The subject then pressed the mouse button for a matching duration of $r$ ms (reproduction phase). Performance feedback was then displayed according to an error map $f(x, r)$. Bottom: Generative model for the time interval reproduction task. The interval $x$ is drawn from the probability distribution $p(x)$ (the objective distribution). The stimulus induces in the observer the noisy sensory measurement $y$ with conditional probability density $p_{s}\left(y \mid x ; w_{s}\right)$ (the sensory likelihood), with $w_{s}$ a sensory variability parameter. The action $u$ subsequently taken by the ideal observer is assumed to be the 'optimal' action $u^{*}$ that minimizes the subjectively expected loss (Eq. 1); $u$ is therefore a deterministic function of $y, u=u^{*}(y)$. The subjectively expected loss depends on terms such as the prior $q(x)$ and the loss function (squared subjective error map $\bar{f}(x, r)$ ), which do not necessarily match their objective counterparts. The chosen action is then corrupted by motor noise, producing the observed response $r$ with conditional probability density $p_{m}\left(r \mid w ; w_{m}\right)$ (the motor likelihood), where $w_{m}$ is a motor variability parameter.

doi:10.1371/journal.pcbi. 1002771 . g002

Short Uniform, red points, and 937.5 ms for Long Uniform, green points). However, at other intervals a bias was seen towards the mean interval of that experimental block, with subjects reporting intervals longer than the mean as shorter than they really were and conversely intervals shorter than the mean as being longer than they really were. Moreover, this bias increased almost linearly with the difference between the mean interval and the actual interval. Qualitatively, this bias profile is consistent with most reasonable hypotheses for the prior, likelihoods and
loss functions of an ideal Bayesian observer (even though details may differ).

The standard deviation of the response (Figure 3 bottom) showed a roughly linear increase with interval duration, in agreement with the 'scalar property' of interval timing [25], according to which the variability in a timing task grows in proportion to the interval duration.

These results qualitatively suggest that the temporal context influences subjects' performance in the motor-sensory timing task in a way which may be compatible with a Bayesian interpretation,

---

#### Page 5

Table 1. Summary of experimental layout for all experiments.

| Experiment | Subjects | Interval range | Distribution |  Peak probability   | Feedback |
| :--------: | :------: | :------------: | :----------: | :-----------------: | :------: |
|     1      |  $n=4$   |     Short      |   Uniform    |         $-$         |  Skewed  |
|            |          |      Long      |   Uniform    |         $-$         |          |
|     2      |  $n=6$   |     Medium     |   Uniform    |         $-$         |  Skewed  |
|            |          |     Medium     |    Peaked    |      $7 / 12$       |          |
|     3      |  $n=6$   |     Medium     |   Uniform    |         $-$         | Standard |
|     4      |  $n=3$   |     Medium     | High-Peaked  |      $19 / 24$      | Standard |
|    5 a     |  $n=4$   |     Medium     |   Bimodal    | $1 / 3$ and $1 / 3$ | Standard |
|    5 b     |  $n=4$   |      Wide      | Wide-Bimodal |      See text       | Standard |

Each line represents an experimental block, which are grouped by experiment; subjects in Experiment 1 and 2 took part in two blocks, whereas in Experiment 5 two distinct groups of subjects took part in the two blocks. For each block, the table reports number of subjects ( $n$ ), interval ranges, type of distribution, probability of the 'peak' (i.e. most likely) intervals and shape of performance feedback. Tested ranges were Short (450-825 ms), Medium (600-975 ms), Long (750-1125 ms) and Wide (450-1125 ms), each covered by 6 intervals ( 10 for the Wide block) separated by 75 ms steps. Distributions of intervals were Uniform ( $1 / 6$ probability per interval), Peaked/High-peaked (the 'peak' interval at 675 ms appeared with higher probability than non-peak stimuli, which were equiprobable), Bimodal (intervals at 600 and 975 ms appeared with higher probability) and Wide-Bimodal (intervals at 450-600 ms and 975-1125 ms appeared with higher probability). The Skewed feedback takes the form $x \frac{r-x}{r}$ whereas the Standard feedback $x \cdot r-x$, where $r$ is the reproduced duration and $x$ is the target interval in a trial.
doi:10.1371/journal.pcbi. 1002771 .t001
and in agreement with previous work which considered purely sensory intervals and uniform distributions [6,8,26].

## Experiment 2: Uniform and Peaked distributions on the same range

As in the first experiment six different equally-spaced intervals were used, with two different distributions. However, in this experiment both blocks had the same range of intervals (Medium: $600-975 \mathrm{~ms}$ ). In one block (Medium Peaked) one of the intervals (termed the 'peak') occurred more frequently than the other 5 intervals, that were equiprobable. That is, the 675 ms interval occurred with $p=7 / 12$ with the other 5 intervals occurring each with $p=1 / 12$. In the other block (Medium Uniform) the 6 intervals were equiprobable. The feedback gain for both blocks was again the Skewed error map $f_{S k} \propto \frac{r-x}{r}$.

Examination of the responses showed a central tendency as encountered in the previous experiment (Figure 4 top). However, despite the identical range of intervals in both blocks, subjects were sensitive to the relative probability of the intervals [27]. In particular, the responses in the Peaked block (light blue points) appeared to be generally shifted towards shorter durations and this shift was interval dependent (see Figure 5). This behavior is qualitatively consistent with a simple Bayesian inference process, according to which the responses are 'attracted' towards the regions of the prior distribution with greatest probability mass. Intuitively, the average ('global') shift of responses can be thought of as arising from the shift in the distribution mean, from the Uniform distribution (mean 787.5 ms ) to the Peaked distribution (mean 731.3 ms ); whereas interval-dependent ('local') effects are a superimposed modulation by the probability mass assignments of the distribution. This is only a simplified picture, as the biases depend on a nonlinear inference process, which is also influenced by other details of the Bayesian model (such as the loss function), but the qualitative outcome is likely to be similar in many relevant cases.

The standard deviation of the responses showed a significant decrease in variability around the peak for the Peaked condition (Figure 4 bottom; two-sample F-test $p<0.001$ ). This effect could be simply due to practice as subjects received feedback more often at peak intervals, however the local modulation of bias previously described (Figure 5) suggests a Bayesian interpretation. In fact,
because of the local 'attraction' effect, interval durations close to the peak would elicit responses that map even closer to it, therefore compressing the perceptual variability, an example of biasvariance trade-off [6].

The results of the second experiment show that people take into account the different nature of the two experimental distributions, in agreement with previous work that found differential effects in temporal reproduction for skewed vs uniform distributions of temporal intervals on a wider, suprasecond range [27]. The performance of the subjects in the two blocks is consistent with a Bayesian 'attraction' in the response towards the intervals with higher prior probability mass. Moreover, although the average negative shift in the response observed in the Peaked condition versus the Uniform one might be compatible with a temporal recalibration effect that shortens the perceived duration between action and effect [11,28,29], the interval-dependent bias modulation (Figure 5) and the reduction in variability around the peak (Figure 4 bottom) suggest there may instead be in this case a Bayesian explanation.

In order to address more specific, quantitative questions about our results we set up a formal framework based on a Bayesian observer and actor model.

## Bayesian observer model

We modelled the subjects' performance with a family of Bayesian ideal observer (and actor) models which incorporated both the perception (time interval estimation) and action (reproduction) components of the task; see Figure 2 (bottom) for a depiction of the generative model of the data. We assume that on a given trial a time interval $x$ is drawn from a probability distribution $p(x)$ (the experimental distribution) and the observer makes an internal measurement $y$ that is corrupted by sensory noise according to the sensory likelihood $p_{x}\left(y \mid x ; w_{x}\right)$, where $w_{x}$ is a parameter that determines the sensory (estimation) variability. Subjects then reproduce the interval with a motor command of duration $u$. This command is corrupted by motor noise, producing the response duration $r$ - the observed reproduction time interval - with conditional probability density $p_{m}\left(r \mid u ; w_{m}\right)$ (the motor likelihood), with $w_{m}$ a motor (reproduction) variability parameter. Subjects receive an error specified by a mapping $f(x, r)$ and we assume they try to minimize a (quadratic) loss based on this error.

---

#### Page 6

> **Image description.** The image presents a set of graphs comparing experimental data with a Bayesian model. It's structured as a 2x2 grid, with an additional row of visual cues at the very top.
>
> - **Top Row (Visual Cues):** Each column has a horizontal arrangement of colored squares. The left column has red squares from 450 to 750, and green squares from 750 to 1050. The right column has the same arrangement of red and green squares. These likely indicate the range of "Short Uniform" (red) and "Long Uniform" (green) blocks.
>
> - **Left Column (Single Subject):**
>
>   - _Top Graph:_ A scatter plot showing "Response bias (ms)" on the y-axis (ranging from -150 to 150) and "Physical time interval (ms)" on the x-axis (ranging from 450 to 1050). Red data points with error bars represent the "Short Uniform" condition, and green data points with error bars represent the "Long Uniform" condition. A continuous red line and a continuous green line show the model fit for each condition. The red data points and line show a negative slope. The green data points and line show a negative slope.
>   - _Bottom Graph:_ A scatter plot showing "Response sd (ms)" (standard deviation) on the y-axis (ranging from 40 to 120) and "Physical time interval (ms)" on the x-axis (ranging from 450 to 1050). Red data points with error bars represent the "Short Uniform" condition, and green data points with error bars represent the "Long Uniform" condition. A continuous red line and a continuous green line show the model fit for each condition. The red data points and line show a positive slope. The green data points and line show a positive slope.
>
> - **Right Column (Group Mean):**
>
>   - _Top Graph:_ A scatter plot showing "Response bias (ms)" on the y-axis (ranging from -150 to 150) and "Physical time interval (ms)" on the x-axis (ranging from 450 to 1050). Red data points with error bars represent the "Short Uniform" condition, and green data points with error bars represent the "Long Uniform" condition. A continuous red line and a continuous green line show the model fit for each condition. The red data points and line show a negative slope. The green data points and line show a negative slope. The label "n = 4" is present.
>   - _Bottom Graph:_ A scatter plot showing "Response sd (ms)" (standard deviation) on the y-axis (ranging from 40 to 120) and "Physical time interval (ms)" on the x-axis (ranging from 450 to 1050). Red data points with error bars represent the "Short Uniform" condition, and green data points with error bars represent the "Long Uniform" condition. A continuous red line and a continuous green line show the model fit for each condition. The red data points and line show a positive slope. The green data points and line show a positive slope.
>
> - **Text:** The graphs are labeled with "Single subject" above the left column and "Group mean" above the right column. The axes are labeled consistently across columns as "Response bias (ms)", "Response sd (ms)", and "Physical time interval (ms)".

Figure 3. Experiment 1: Short Uniform and Long Uniform blocks. Very top: Experimental distributions for Short Uniform (red) and Long Uniform (green) blocks, repeated on top of both columns. Left column: Mean response bias (average difference between the response and true interval duration, top) and standard deviation of the response (bottom) for a representative subject in both blocks (red: Short Uniform; green: Long Uniform). Error bars denote s.e.m. Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models (Bayesian model averaging). Right column: Mean response bias (top) and standard deviation of the response (bottom) across subjects in both blocks (mean $\pm$ s.e.m. across subjects). Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models across subjects.

doi:10.1371/journal.pcbi.1002771.g003

In our model we assume that subjects develop an internal estimate of both the experimental distribution and error mapping (the feedback associated with a response $r$ to stimulus $x$ ), which leads to the construction of a (subjective) prior, $q(x)$, and subjective error mapping $\tilde{f}(x, r)$; the latter is then squared to obtain the loss function. This allows the prior and subjective error mapping to deviate from their objective counterparts, respectively $p(x)$ and $f(x, r)$.

Following Bayesian decision theory, the 'optimal' action $u^{*}(y)$ is calculated as the action $u$ that minimizes the subjectively expected loss:

$$
u^{*}(y)=\arg \min _{u} \int p_{x}\left(y \mid x ; w_{x}\right) q(x) p_{m}(r) u ; w_{m} \tilde{f}^{2}(x, r) d x d r
$$

where the integral on the right hand side is proportional to the subjectively expected loss. Combining Eq. 1 with the generative model of Figure 2 (bottom) we computed the distribution of responses of an ideal observer for a target time interval $x$, integrating over the hidden
internal measurement $y$ which was not directly accessible in our experiment.

Therefore the reproduction time $r$ of an ideal observer, given the target interval $x$, is distributed according to:

$$
p\left(r \mid x ; w_{x}, w_{m}\right)=\int p_{x}\left(y \mid x ; w_{x}\right) p_{m}(r) u^{*}(y) ; w_{m} d y
$$

Eqs. 1 and 2 are the key equations that allow us to simulate our task, in particular by computing the mean response bias and standard deviation of the response for each interval (Section 1 in Text S1). Eq. 1 represents the internal model and deterministic decision process adopted by the subject whereas Eq. 2 represents probabilistically the objective generative process of the data. Notice that the experimental distribution $p(x)$ and objective error mapping $f(x, r)$ do not appear in any equation: the distribution of responses of ideal observers only depends on their internal representations of prior and loss function.

---

#### Page 7

> **Image description.** The image presents a series of plots comparing experimental data with a Bayesian model. It consists of four main panels arranged in a 2x2 grid, each associated with either a "Single subject" or "Group mean" analysis. The top row of panels displays distributions, while the bottom two rows show scatter plots with error bars and fitted lines.
>
> - **Top Row: Distributions**
>
>   - Each of the two panels in the top row shows a bar graph. The x-axis is labeled with values 600, 750, and 900. Each x-value has two bars, one light brown and one light blue. The light blue bar is taller at x=750 in both panels. The left panel is labeled "Single subject", and the right panel is labeled "Group mean".
>
> - **Middle Row: Response Bias vs. Physical Time Interval**
>
>   - The two panels in the middle row show scatter plots. The y-axis is labeled "Response bias (ms)" and ranges from -150 to 150. The x-axis is labeled "Physical time interval (ms)" and ranges from 600 to 900. Each panel contains two sets of data points, one in light blue and one in light brown. Each data point has error bars. The light blue points are connected by a light blue line, and the light brown points are connected by a light brown line. A horizontal line is drawn at y=0. The left panel is labeled "Single subject", and the right panel is labeled "Group mean". In the right panel, the text "n = 6" is present.
>
> - **Bottom Row: Response Standard Deviation vs. Physical Time Interval**
>   - The two panels in the bottom row show scatter plots. The y-axis is labeled "Response sd (ms)" and ranges from 40 to 120. The x-axis is labeled "Physical time interval (ms)" and ranges from 600 to 900. Each panel contains two sets of data points, one in light blue and one in light brown. Each data point has error bars. The light blue points are connected by a light blue line, and the light brown points are connected by a light brown line. The left panel is labeled "Single subject", and the right panel is labeled "Group mean".

Figure 4. Experiment 2: Medium Uniform and Medium Peaked blocks. Very top: Experimental distributions for Medium Uniform (light brown) and Medium Peaked (light blue) blocks, repeated on top of both columns. Left column: Mean response bias (average difference between the response and true interval duration, top) and standard deviation of the response (bottom) for a representative subject in both blocks (light blue: Medium Uniform; light brown: Medium Peaked). Error bars denote s.e.m. Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models (Bayesian model averaging). Right column: Mean response bias (top) and standard deviation of the response (bottom) across subjects in both blocks (mean $\pm$ s.e.m. across subjects). Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models across subjects.

doi:10.1371/journal.pcbi. 1002771 . g 004
Eqs. 1 and 2 describe a family of Bayesian observer models, a single Bayesian ideal observer is fully specified by picking (i) a noise model for the sensory estimation process, $p_{s}\left(y \mid x ; w_{s}\right)$; (ii) a noise model for the motor reproduction process $p_{m}(r \mid n ; w_{m})$; (iii) the form of the prior $q(x)$; and (iv) the loss function $\hat{f}^{2}(x, r)$ (Figure 6 and Methods). To limit model complexity, in the majority of our analyses we used the same likelihood functions ( $p_{s}, p_{m}$ and their parameters $w_{s}, w_{m}$ ) for both the generative model (Eq. 2) and the internal model (Eq. 1). Analogously, for computational reasons in our basic model we assumed a quadratic exponent for the loss function (Eq. 1); in a subsequent analysis we relaxed this requirement (Section 2 in Text S1).

## Bayesian model comparison

To study the nature of the internal model adopted by the participants, we performed a full Bayesian model comparison over the family of Bayesian ideal observer models. For each participant we assumed that the sensory and motor noise, the approximation
strategy for the priors, and the loss function were shared across different experimental blocks. The model comparison was performed over a discrete set of model components, that is, possible choices for the priors, loss functions and shape of likelihoods (Figure 6). In particular, priors and loss functions did not have continuous parameters, as a parametric model would likely be ambiguous or hard to interpret, with multimodal posterior distributions over the parameters (as multiple combinations of likelihoods, prior and cost function can make identical predictions). Instead, we considered a finite number of parameterfree models of loss function, prior and shape of likelihoods, leaving only two continuous parameters for characterizing the sensory and motor variability.

Both sensory and motor noise were modelled with Gaussian distributions whose means were centered on the interval and whose standard deviations could either be constant or 'scalar', that is, grow linearly with the interval (Figure 6 i and ii). We used two parameters, $w_{s}$ and $w_{m}$, which represent the coefficient of variation of the subject's sensory and motor noise. For the scalar

---

#### Page 8

> **Image description.** The image is a graph that plots the difference in response (in milliseconds) on the y-axis against the physical time interval (in milliseconds) on the x-axis.
>
> - **Axes:**
>
>   - The y-axis is labeled "Difference in response (ms)" and ranges from -60 to 0 in increments of 10.
>   - The x-axis is labeled "Physical time interval (ms)" and has values 600, 675, 750, 825, 900, and 975.
>
> - **Data Points and Error Bars:**
>
>   - There are six data points, each represented by a gray diamond with error bars extending vertically above and below the diamond. The data points are located at x-axis values of 600, 675, 750, 825, 900, and 975.
>   - Two of the data points, at x=600 and x=825, have an asterisk symbol above them.
>
> - **Reference Lines and Regions:**
>
>   - A horizontal dashed black line is present, labeled "Average response shift."
>   - A shaded yellow region surrounds the dashed line, likely representing the standard error of the mean (s.e.m.) for the average response shift. The text "Average response shift" is written within this shaded region.
>
> - **Experimental Distributions:**
>   - At the bottom of the graph, near the x-axis, are pairs of short horizontal bars at each x-axis value. One bar in each pair is dark yellow, and the other is light blue, except for the pair at x=675, where the light blue bar is taller than the dark yellow bar. These represent the experimental distributions.

Figure 5. Experiment 2: Difference in response between Medium Peaked and Medium Uniform blocks. Difference in response between the Medium Peaked and the Medium Uniform conditions as a function of the actual interval, averaged across subjects ( $\pm 1$ s.e.m.). The experimental distributions (light brown: Medium Uniform; light blue: Medium Peaked) are plotted for reference at bottom of the figure. The dashed black line represents the average response shift (difference in response between blocks, averaged across all subjects and stimuli), with the shaded area denoting $\pm$ s.e.m. The average response shift is significantly different from zero ( $-32.2 \pm 7.9 \mathrm{~ms}$; two-sample t-test $p<10^{-7}$ ), meaning that the two conditions elicited consistently different performance. Additionally, the responses were subject to a 'local' (i.e. interval-dependent) modulation superimposed to the average shift, that is, intervals close to the peak of the distribution ( 675 ms ) were attracted towards it, in addition to the average shift, while intervals far away from the peak were less affected. (\*) The response shift at 600 ms and 825 ms is significantly different from the average response shift; $p<0.01$.

doi:10.1371/journal.pcbi. 1002771 . g005
case this simply specifies the coefficient of proportionality of the standard deviation with respect to the mean, whereas in the constant case it specifies the proportion of noise with respect to a fixed interval $(787.5 \mathrm{~ms})$.

We considered three different possible subjective error metrics corresponding to the Skewed error $\tilde{f}_{S k}(x, r) \propto \frac{r-x}{r}$ (the error map we provided experimentally), the Standard error $\tilde{f}_{S t}(x, r) \propto r-x$, and a Fractional error $\tilde{f}_{F r}(x, r) \propto \frac{r-x}{x}$ (Figure 6 iv), which were then squared to obtain the loss function (see also Methods). Note that scaling these mappings does not change the optimal actions and hence the model selection process.

We compared different approximation schemes for the priors, such as the true discrete distribution (Figure 6 iii, a) or a single Gaussian whose mean and standard deviation matched those of the true prior (b). We also considered two smoothed versions of the experimental distribution with a weak (c) and strong (d) smoothing parameter, or some other block-dependent approximations, e.g. for the Uniform blocks we considered a uniform distribution over the stimulus range (e); see Methods for a full description. To constrain the model selection process, we assumed that subjects adopted a consistent approximation scheme across blocks.

For each participant we computed the support for each model based on the psychophysical data, that is the posterior probability of the model, Pr (model| data). Assuming an a priori indifference among the models, this corresponds (up to a normalization factor) to the model marginal likelihood $\operatorname{Pr}($ data $\mid$ model $)$, which was obtained by numerical integration over the two-dimensional parameter space $\left(w_{e}\right.$ and $\left.w_{m}\right)$.

We then calculated the Bayesian model average for the response mean bias and standard deviation, shown by the continuous lines in Figure 3 and 4. Note that the Bayesian model 'fits' are obtained by computing the marginal likelihood of the models and integrating the model predictions over the posterior of the parameters (model averaging), with no parameter fitting. The mean biases fits show a good quantitative match with the group averages ( $R^{2} \geq 0.95$ for all blocks); the standard deviations are typically more erratic and we found mainly a qualitative agreement, as observed in previous work [6].

For each participant of Experiments 1 and 2 we computed the most probable (i) sensory and (ii) motor likelihoods, (iii) priors and (iv) loss function (Table S1). The model comparison confirmed that the best noise models were represented by the 'scalar' variability, which had relevant support for both the sensory component ( 7 subjects out of 10 ) and the motor component ( 8 subjects out of 10 ). This result is consistent with previous work in both the sensory and motor domain [5,6,25,30]. The most supported subjective error map was the Skewed error ( 7 subjects out of 10), which matched the feedback we provided experimentally. The priors most supported by the data were typically smooth, peaked versions of the experimental distributions. In particular, according to the model comparison, almost all subjects ( 9 out of 10) approximated the discrete uniform distributions in the Uniform blocks with normal distributions (same mean and variance as the true distribution; Figure 6 iii top, b). However, in Experiment 2 most people ( 5 out of 6 ) seemed to approximate the experimental distribution in the Peaked block not with a standard Gaussian, but with a skewed variant of a normal distribution (Figure 6 iii bottom, d, f and g), suggesting that their

---

#### Page 9

> **Image description.** The image presents a figure with four panels, labeled i through iv, illustrating components of a Bayesian observer and actor model. Each panel contains multiple plots.
>
> Panel i, "Sensory Likelihood," displays two sets of graphs. The top graph shows "Noise sd (ms)" on the y-axis, ranging from 0 to 100, and "Time interval (ms)" on the x-axis, ranging from 600 to 1050. A horizontal line is plotted at approximately y=65, with green, gray, and magenta dots along the line. Below this, a series of Gaussian curves are plotted, with peaks centered at approximately 600, 750, 900, and 1050 ms, colored in green, gray, and magenta. The bottom graph in panel i also has "Noise sd (ms)" on the y-axis (0-100) and "Time interval (ms)" on the x-axis (600-1050). A black line with green, gray, and magenta dots shows a linear relationship, increasing from approximately y=45 at x=600 to y=80 at x=1050. Below this line, a series of Gaussian curves are plotted, similar to the top graph.
>
> Panel ii, "Motor Likelihood," mirrors the structure of panel i. The top graph shows a horizontal line at approximately y=65 with green, gray, and magenta dots. The bottom graph shows a linear relationship increasing from approximately y=45 at x=600 to y=80 at x=1050, again with green, gray, and magenta dots. Both graphs have "Noise sd (ms)" on the y-axis (0-100) and "Time interval (ms)" on the x-axis (600-1050), and Gaussian curves below the line plots. A large "X" symbol is placed between panels i and ii.
>
> Panel iii, "Priors," presents a series of plots labeled "Uniform" and "Peaked." Under "Uniform," plots a through e are shown. Plot a consists of vertical lines at various intervals. Plots b, c, and d show curves with varying degrees of spread. Plot e shows a rectangle. Under "Peaked," plots a, b, c, d, f, and g are shown. Plot a consists of vertical lines of varying heights. Plots b, c, d, f, and g show curves with different shapes and peaks. All plots in panel iii have the x-axis labeled "Time interval (ms)" ranging from 450 to 1050. A large "X" symbol is placed between panels ii and iii.
>
> Panel iv, "Error maps," contains three graphs labeled "Skewed," "Standard," and "Fractional." Each graph plots curves with varying slopes, with green, gray, and magenta dots along a horizontal line at y=0. The x-axis is labeled "Response (ms)" ranging from 600 to 1050. A large "X" symbol is placed between panels iii and iv.

Figure 6. Bayesian observer and actor model components. Candidate (i) sensory and (ii) motor likelihoods, independently chosen for the sensory and motor noise components of the model. The likelihoods are Gaussians with either constant or 'scalar' (i.e. homogeneous linear) variability. The amount of variability for the sensory (resp. motor) component is scaled by parameter $w_{s}$ (resp. $w_{m}$ ). iii) Candidate priors for the Medium Uniform (top) and Medium Peaked (bottom) blocks. The candidate priors for the Short Uniform (resp. Long Uniform) blocks are identical to those of the Medium Uniform block, shifted by 150 ms in the negative (resp. positive) direction. See Methods for a description of the priors. iv) Candidate subjective error maps. The graphs show the error as a function of the response duration, for different discrete stimuli (drawn in different colors). From top to bottom: Skewed error $\hat{f}_{S k}(r, x) \cdot \frac{r-s}{r}$; Standard error $\hat{f}_{S s}(r, x) \cdot x \cdot r-x$; and Fractional error $\hat{f}_{F r}(r, x) \cdot \frac{r-s}{x}$. The scale is irrelevant, as the model is invariant to rescaling of the error map. The squared subjective error map defines the loss function (as per Eq. 1).

doi:10.1371/journal.pcbi. 1002771 . g006
responses were influenced by higher order moments of the true distribution and not just the mean and variance (see Discussion).

For Experiment 2 we also relaxed some constraints on the priors, allowing the model selection to pick a Medium Uniform prior for the Medium Peaked block and vice versa. Nevertheless, the model comparison showed that the most supported models were still the ones in which the priors matched the block distribution, supporting our previous findings that subjects' responses were consistent with the temporal context and changed when switching from one block to another (as visible in Figure 4).

## Nonparametric reconstruction of the priors

To study in detail the internal representations, we relaxed the constraint on the priors. Rather than choosing from a fixed set of candidate priors (Figure 6 iii), we allowed the prior to vary over a much wider class of smooth, continuous distributions. We assumed that the noise models and loss function emerging from the model comparison were a good description of the subjects' decision making and sensorimotor processing in the task. We therefore fixed these components of the observer's model and inferred nonparametrically, on an individual basis, the shape of the priors most compatible with the measured responses (Figure 7; see Methods for details).

Examination of the recovered priors shows that the subjective distributions were significantly different from zero only over the range corresponding to the experimental distribution, with only occasional tails stretching outside the interval range (e.g. Figure 7 bottom left). This suggests that in general people were able to localize the stimulus range in the blocks. The priors did not typically take a bell-like shape, but rather we observed a more or less pronounced peak at the mean of the true distribution, with the remaining probability mass spread over the rest of the range. Interestingly, the group averages for the Uniform priors over the Short, Medium and Long ranges (Figure 7 top right, both, and bottom right, light brown) exhibit very similar, roughly symmetrical shapes, shifted over the appropriate stimulus range. Conversely, the Peaked prior (Figure 7 bottom right, light blue) had a distinct, skewed shape.

To compare the inferred priors with the true distribution, we calculated their distribution moments (Table 2). We found that the first three moments of the inferred priors (in the table reported as mean, standard deviation and skewness) were statistically indistinguishable from those of the true distributions for all experimental conditions (Hotelling's multivariate one-sample $T^{2}$ test considering the joint distribution of mean, standard deviation and skewness against the true values; $p>0.45$ for all blocks). This result confirmed the previously stated hypothesis that participants had

---

#### Page 10

> **Image description.** This image contains four plots arranged in a 2x2 grid, showing probability density functions. The plots are related to an experiment, likely in cognitive science or psychology, as indicated by the labels and the nature of the data.
>
> - **Overall Layout:** The plots are arranged in two rows and two columns. The left column is labeled "Single subjects" and the right column is labeled "Group average". The top row appears to represent one condition or block type, while the bottom row represents another.
>
> - **Top Row Plots:**
>
>   - The top left plot ("Single subjects") shows two probability density curves, one in red and one in green. These curves represent the inferred priors for individual participants. Shaded regions around the curves indicate $\pm 1$ standard deviation. Below the curves, there are several small squares along the x-axis, with some colored red and some colored green. These squares likely represent the discrete experimental distributions. The x-axis is labeled from 300 to 1200. The y-axis is labeled "Probability density" and ranges from 0 to 0.03.
>   - The top right plot ("Group average") is similar to the top left plot, but the curves represent the average inferred priors across a group of participants (n=4). The curves are also red and green, with shaded regions indicating standard deviation. The discrete experimental distributions are also shown as red and green squares below the curves. The axes are labeled the same as the top left plot.
>
> - **Bottom Row Plots:**
>
>   - The bottom left plot ("Single subjects") shows two probability density curves, one in light blue and one in light brown. These curves represent the inferred priors for individual participants. Shaded regions around the curves indicate $\pm 1$ standard deviation. Below the curves, there are several small squares along the x-axis, with some colored light blue and some colored light brown. These squares likely represent the discrete experimental distributions. The x-axis is labeled "Physical time interval (ms)" and ranges from 450 to 1050. The y-axis is labeled "Probability density" and ranges from 0 to 0.03.
>   - The bottom right plot ("Group average") is similar to the bottom left plot, but the curves represent the average inferred priors across a group of participants (n=6). The curves are also light blue and light brown, with shaded regions indicating standard deviation. The discrete experimental distributions are also shown as light blue and light brown squares below the curves. The axes are labeled the same as the bottom left plot.
>
> - **Text Elements:**
>   - "Single subjects" (above the left column)
>   - "Group average" (above the right column)
>   - "Probability density" (y-axis label, repeated on the left side of both rows)
>   - "Physical time interval (ms)" (x-axis label, repeated on the bottom row)
>   - "n = 4" (in the top right plot)
>   - "n = 6" (in the bottom right plot)

Figure 7. Nonparametrically inferred priors (Experiment 1 and 2). Top row: Short Uniform (red) and Long Uniform (green) blocks. Bottom row: Medium Uniform (light brown) and Medium Peaked (light blue) blocks. Left column: Nonparametrically inferred priors for representative participants. Right column: Average inferred priors. Shaded regions are $\pm 1 \mathrm{~s}$.d. For comparison, the discrete experimental distributions are plotted under the inferred priors.

doi:10.1371/journal.pcbi. 1002771 . g007
developed an internal representation which included higher order moments and not just the mean and variance of the experimental distribution. However, when including the fourth moment (kurtosis) in the analysis, we observed a statistically significant deviation of the recovered priors with respect to the true distributions (Hotelling's $T^{2}$ test with the joint distribution of the first four moments; $p<10^{-4}$ for all blocks); in particular, the inferred priors seem to have more pronounced peaks and/or heavier tails. First of all, note that the heightened kurtosis is not an artifact due to the averaging process across subjects or the sampling process within subjects, as we averaged the moments computed for each sampled distribution (see Methods) rather than computing the moments of the average distribution. In other words, all recovered priors are (on average) heavy tailed, it's not just the mean prior that it is 'accidentally' heavy tailed as a mixture of light-tailed distributions. So this result could mean that the subjects' internal representations are actually heavy-tailed, for instance to allow for unexpected stimuli. However, there could be a simpler explanation that the presence of outliers arise from occasional trivial mistakes of the participants. We, therefore, considered a straightforward extension of our model which added the possibility of occasional 'lapses' with a lapse rate $\lambda$, where the response in a lapse trial is simply modelled as a uniform distribution over a wide range of intervals (Section 3 in Text S1). In terms of marginal likelihood, generally the models with lapse performed better than the original models, but with no qualitative difference in the preferred model components. Crucially, we did not observe a significant change in the kurtosis of the recovered priors, ruling out the possibility that the heightened kurtosis had been caused by trivial outliers.

Our analysis therefore showed that, according to the inferred priors, people generally acquired internal representations that were smooth, heavy-tailed approximations to the experimental
distributions of intervals, in agreement up to the first three moments.

## Experiment 3: Effect of the shape of feedback on the loss function

In our ideal observer model we compared three candidate loss functions: Skewed, Standard and Fractional (Figure 6 iv). The results of the model comparison in the first two experiments with Skewed feedback showed that there was a good match between experimentally provided feedback and subjective error metric. However, we could not rule out the possibility, albeit unlikely, that participants were ignoring the experimental feedback and following an internal error signal that just happened to be similar in shape to the Skewed error. We therefore performed an additional experiment to verify that subjects behavior is driven by the feedback provided.

We again used a Medium Uniform block but now with Standard error $f(\mathrm{~V}) / x r-x$ as feedback (see Figure S5 in Text S2). The model comparison for this group showed that the responses of 4 subjects out of 6 were best explained with a Standard loss function. Moreover, no subject appeared to be using the Skewed loss function (Table S1). These results confirm that most people correctly integrate knowledge of results with sensory information in order to minimize the average (squared) error, or an empirically similar metric. Furthermore, all inferred individual priors showed a remarkable agreement with a smoothed approximation of the experimental distribution of intervals (Figure 8 top), suggesting that the Standard error feedback may be easier to use for learning. As in the previous experiments, the average moments of the inferred priors (up to skewness) were statistically indistinguishable from those of the true distribution, with a significant difference in the kurtosis (Table 3 left; Hotelling's $T^{2}$ test, first three moments: $p>0.95$; first four moments: $p<10^{-7}$ ).

---

#### Page 11

Table 2. Main statistics of the experimental distributions and nonparametrically inferred priors (Experiment 1 and 2; Skewed feedback).

|              | Short Uniform  |            |       |      | Long Uniform  |            |       |      |
| :----------- | :------------- | :--------- | :---- | :--- | :------------ | :--------- | :---- | :--- |
|              | Objective      | Subjective |       |      | Objective     | Subjective |       |      |
| Mean (ms)    | 637.5          | 644.2      | $\pm$ | 12.8 | 937.5         | 929.9      | $\pm$ | 19.6 |
| Std (ms)     | 128.1          | 117.4      | $\pm$ | 13.3 | 128.1         | 131.2      | $\pm$ | 16.9 |
| Skewness     | 0              | -0.17      | $\pm$ | 0.24 | 0             | -0.12      | $\pm$ | 0.41 |
| Ex. Kurtosis | -1.27          | 0.86       | $\pm$ | 1.24 | -1.27         | 0.82       | $\pm$ | 0.98 |
|              | Medium Uniform |            |       |      | Medium Peaked |            |       |      |
|              | Objective      | Subjective |       |      | Objective     | Subjective |       |      |
| Mean (ms)    | 787.5          | 805.7      | $\pm$ | 27.4 | 731.3         | 724.1      | $\pm$ | 24.0 |
| Std (ms)     | 128.1          | 130.4      | $\pm$ | 23.5 | 106.6         | 110.13     | $\pm$ | 18.5 |
| Skewness     | 0              | -0.16      | $\pm$ | 0.41 | 1.14          | 0.78       | $\pm$ | 0.42 |
| Ex. Kurtosis | -1.27          | 0.80       | $\pm$ | 1.44 | 0.09          | 2.20       | $\pm$ | 2.39 |

Comparison between the main statistics of the 'objective' experimental distributions and the 'subjective' priors nonparametrically inferred from the data. The subjective moments are computed by averaging the moments of sampled priors pooled from all subjects ( $\pm 1 \mathrm{~s} . \mathrm{d}$ ); see Figure 7, right column and Methods for details. In statistics, the excess kurtosis is defined as kurtosis -5 , such that the excess kurtosis of a normal distribution is zero. Heavy tailed distributions have a positive excess kurtosis. doi:10.1371/journal.pcbi. 1002771 .t002

## Experiment 4: High-Peaked distribution

In the Peaked block we did not observe any significant divergence from the Bayesian prediction. However, the ratio of presentations of 'peak' intervals ( 675 ms ) to the others was low (1.4) and possibly not enough to induce other forms of temporal adaptation [29,31]. To examine whether we might see deviations from Bayesian integration for larger ratios we therefore tested another group of subjects on a more extreme variant of the Peaked distribution in which the peak stimulus had a probability of $p \approx 0.8$ and therefore a ratio of about 4.0. We provided feedback through
the Standard error mapping, as the previous experiment had showed that subjects can follow it at least as well as the Skewed mapping.

Due to the large peak interval presentation frequency we had fewer test data points in the model fitting. Therefore, we constrained the model comparison by only considering the Standard loss in order to prevent the emergence of spurious model components capturing random patterns in the data. We found that the recovered internal priors were in good qualitative agreement with the true distribution, with statistically indistin-

> **Image description.** The image contains four plots arranged in a 2x2 grid. Each plot displays a probability density function against a physical time interval.
>
> - **Overall Structure:** The plots are organized into two columns labeled "Single subjects" (left column) and "Group average" (right column). The top row displays data in yellow, while the bottom row displays data in blue.
>
> - **Axes:** Each plot has a horizontal axis labeled "Physical time interval (ms)" ranging from approximately 450 to 1050. The vertical axis is labeled "Probability density" and ranges from 0 to 0.03.
>
> - **Top Row (Yellow):**
>
>   - The "Single subjects" plot (top left) shows a yellow shaded region with a black line running through its center, representing the probability density function. Small olive green squares are plotted along the x-axis at approximately 600, 700, 750, 800, 900, and 1000 ms.
>   - The "Group average" plot (top right) also shows a similar yellow shaded region with a black line. The x-axis also has olive green squares plotted at the same locations as the "Single subjects" plot. The text "n = 6" is present in the top right corner of the plot.
>
> - **Bottom Row (Blue):**
>   - The "Single subjects" plot (bottom left) displays a blue shaded region with a dark blue line. Short blue lines are plotted along the x-axis, with a taller blue line at approximately 650 ms.
>   - The "Group average" plot (bottom right) shows a similar blue shaded region with a dark blue line. The x-axis also has blue lines plotted at the same locations as the "Single subjects" plot, with a taller blue line at approximately 650 ms. The text "n = 3" is present in the top right corner of the plot.

Figure 8. Nonparametrically inferred priors (Experiment 3 and 4). Top row: Medium Uniform (light brown) block. Bottom row: Medium HighPeaked (dark blue) block. Left column: Nonparametrically inferred priors for representative participants. Right column: Average inferred priors. Shaded regions are $\pm 1 \mathrm{~s} . \mathrm{d}$. For comparison, the discrete experimental distributions are plotted under the inferred priors.

doi:10.1371/journal.pcbi. 1002771 .g008

---

#### Page 12

Table 3. Main statistics of the experimental distributions and nonparametrically inferred priors (Experiment 3 and 4; Standard feedback).

|              | Medium Uniform |            |       |      | Medium High-Peaked |            |       |      |
| :----------- | :------------- | :--------- | :---- | :--- | :----------------- | :--------- | :---- | :--- |
|              | Objective      | Subjective |       |      | Objective          | Subjective |       |      |
| Mean (ms)    | 787.5          | 782.6      | $\pm$ | 18.7 | 703.1              | 702.0      | $\pm$ | 17.9 |
| Std (ms)     | 128.1          | 131.7      | $\pm$ | 13.6 | 80.5               | 119.5      | $\pm$ | 17.9 |
| Skewness     | 0              | 0.03       | $\pm$ | 0.30 | 2.25               | 0.67       | $\pm$ | 0.37 |
| Ex. Kurtosis | -1.27          | 0.42       | $\pm$ | 0.53 | -0.86              | 1.66       | $\pm$ | 1.32 |

Comparison between the main statistics of the 'objective' experimental distributions and the 'subjective' priors nonparametrically inferred from the data. The subjective moments are computed by averaging the moments of sampled priors pooled from all subjects ( $\pm 1$ s.d.); see Figure 8, right column and Methods for details. I doi:10.1371/journal.pcbi. 1002771 .t003
guishable means (Figure 8 bottom, and Table 3; one sample twotailed t -test $p>0.90$ ). When variance and higher moments were included in the analysis, though, the distributions were significantly different (Hotelling's $T^{2}$ test, mean and variance: $p<0.05$; first three moments: $p<0.01$; first four moments: $p<10^{-7}$ ) suggesting that the distribution may have been 'too peaked' to be learnt exactly; see Discussion. Nevertheless, the observed biases of the responses were well explained by the basic Bayesian models (group mean: $R^{2}=0.95$ ), and the standard deviations were in qualitative agreement with the data (Figure S6 in Text S2).

## Experiment 5: Bimodal distributions

Our previous experiments show that people are able to learn good approximation of flat or unimodal distributions of intervals relatively quickly (a few sessions), under the guidance of corrective feedback. Previous work in sensorimotor learning [15] and motion perception [32] has shown that people can learn bimodal distributions. Whether the same is attainable for temporal distributions is unclear; a recent study of time interval reproduction [27] obtained less definite results with a bimodal 'V-shaped' distribution, although training might have been too short, as subjects were exposed only to 120 trials in total and without performance feedback.

To examine whether subjects could easily learn bimodality of a temporal distribution with the help of feedback we tested two new groups of subjects on bimodal distributions of intervals on a Medium range ( $600-975 \mathrm{~ms}$, as before) and on a Wide range ( $450-1125 \mathrm{~ms}$ ), providing in both cases Standard feedback. In the Medium Bimodal block the intervals at 600 and 975 ms had each probability $p=4 / 12$, whereas the other four middle intervals ( 675 , $750,825,900 \mathrm{~ms}$ ) had each probability $p=1 / 12$. In the Wide Bimodal block the six 'extremal' intervals ( $450,525,600 \mathrm{~ms}$ and $975,1050,1125 \mathrm{~ms}$ ) had each probability $p=4 / 28$ whereas the middle intervals had probability $p=1 / 28$. Note that in both cases extremal intervals were four times as frequent as middle intervals.

In the Medium Bimodal block, subjects' responses exhibited a typical central tendency effect (Figure 9 top left) which suggests that people did not match the bimodality of the underlying distribution. To constrain the model comparison we inferred the subjects' priors under the assumption of scalar sensory and motor noise models and Standard loss function, as found by our previous analyses. As before, we first used a discrete set of priors (see Methods) that we used to compute the model 'fit' to the data and then we performed a nonparametric inference. The nonparametrically inferred priors for the Medium Bimodal distribution (Figure 9 top right) suggest that on average subjects developed an
internal representation that differed from those seen in previous experiments and, as before, we found a good agreement between moments of the experimental distribution and moments of the inferred priors up to skewness (Table 4 left). However, results of the Bayesian model comparison among a discrete class of flat, unimodal or bimodal priors do not support the hypothesis that subjects actually learnt the bimodality of the experimental distribution (data not shown). Part of the problem may have been that in the Medium Bimodal distribution the two modes were relatively close, and due to sensory and motor uncertainty subjects could not gather enough evidence that the experimental distribution was not unimodal (but see Discussion). We repeated the experiment therefore on a wider range with a different group of subjects.

The pattern of subjects' responses in the Wide Bimodal block shows a characteristic 'S-shaped' bias profile (Figure 9 top right) which is compatible with either a flat or a slightly bimodal prior. The nonparametrically inferred priors for the Wide Bimodal distribution (Figure 9 bottom right) again suggest that on average subjects acquired, albeit possibly with less accuracy (Table 4 right), some broad features of the experimental distribution; however individual datasets are quite noisy and again we did not find strong evidence for learning of bimodality.

Our results with bimodal distributions confirm our previous finding that people seem to be able to learn broad features of experimental distributions of intervals (mean, variance, skewness) with relative ease (a few sessions of training with feedback). However, more complex features (kurtosis, bimodality) seem to be much harder to learn (see Discussion).

## Discussion

Our main finding is that humans, with the help of corrective feedback, are able to learn various statistical features of both simple (uniform, symmetric) and complex (peaked, asymmetric or bimodal) distributions of time intervals. In our experiments, the inferred internal representations were smooth, heavy tailed approximations of the experimental distributions, in agreement typically up to third-order moments. Moreover, our results suggest that people take into account the shape of the provided feedback and integrate it with knowledge of the statistics of the task in order to perform their actions.

The statistics of the responses of our subjects in the Uniform blocks were consistent with results from previous work; in particular, we found biases towards the mean of the range of intervals (central tendency) $[6,8,26,33]$ and the variability of the responses grew roughly linearly in the sample interval duration (scalar property) $[6,34]$. The responses in the Peaked and High-

---

#### Page 13

> **Image description.** This image contains two sets of plots, each set consisting of three subplots arranged vertically. The set on the left represents "Medium Bimodal" data, while the set on the right represents "Wide Bimodal" data.
>
> - **Top Subplots:** These show the experimental distributions.
>
>   - Left: A bar plot with two prominent bars at approximately 600 and 1000, and smaller bars at 750 and 900. The x-axis ranges from 450 to 1050, labeled "Group mean".
>   - Right: A bar plot with several bars distributed across the x-axis, which ranges from 300 to 1200, labeled "Group mean". The bars are located at approximately 300, 450, 600, 750, 900, 1050 and 1200.
>
> - **Middle Subplots:** These show the mean response bias.
>
>   - Left: A scatter plot with data points and error bars. The x-axis is labeled from 450 to 1050. The y-axis, labeled "Response bias (ms)", ranges from -150 to 150. A curved line is fitted through the data points. "n = 4" is displayed in the upper right.
>   - Right: Similar to the left, but the x-axis ranges from 300 to 1200. The y-axis, labeled "Response bias (ms)", ranges from -150 to 150. A curved line is fitted through the data points. "n = 4" is displayed in the upper right.
>
> - **Bottom Subplots:** These show the average inferred priors.
>   - Left: A plot showing "Probability density" on the y-axis (ranging from 0 to 0.02) and "Physical time interval (ms)" on the x-axis (ranging from 450 to 1050). A shaded region represents $\pm 1$ standard deviation around a central curve. A bar plot is shown at the bottom of the graph, with bars at approximately 600 and 1000, and smaller bars at 750 and 900.
>   - Right: Similar to the left, but the x-axis ranges from 300 to 1200. A shaded region represents $\pm 1$ standard deviation around a central curve. A bar plot is shown at the bottom of the graph, with bars at approximately 300, 450, 600, 750, 900, 1050 and 1200.

Figure 9. Experiment 5: Medium Bimodal and Wide Bimodal blocks, mean bias and nonparametrically inferred priors. Very top: Experimental distributions for Medium Bimodal (dark purple, left) and Wide Bimodal (light purple, right) blocks. Top: Mean response bias across subjects (mean $\pm$ s.e.m. across subjects) for the Medium Bimodal (left) and Wide Bimodal (right) blocks. Continuous lines represent the Bayesian model 'fit' obtained averaging the predictions of the most supported models across subjects. Bottom: Average inferred priors for the Medium Bimodal (left) and Wide Bimodal (right) blocks. Shaded regions are $\pm 1$ s.d. For comparison, the experimental distributions are plotted again under the inferred priors.

doi:10.1371/journal.pcbi. 1002771 . g009

Peaked blocks showed analogous biases, but they were directed towards the mean of the distribution rather than the mean of the range of intervals (the two means overlapped in the Uniform case) [27]. We also observed a significant reduction in variability at the peak. These results were sufficient to suggest that subjects considered the temporal statistics of the context in their decision making processes. We found a similar regression to the mean for a 'narrow' bimodal distribution (Medium Bimodal), in qualitative agreement with previous work that found a simple central tendency with a 'V-shaped' temporal distribution [27] (although with very limited training, no feedback and a suprasecond range). However, for a bimodal distribution on a wider range we observed 'S-shaped' biases which seem compatible with a nonlinear decision
making process [15]. However, more refined conclusions needed the support of a formal framework.

# Bayesian model

Our modelling approach consisted of building a family of Bayesian observer and actor models, which provided us with a mathematical structure in which to ask specific questions about our subjects [35], going beyond mere statements about Bayesian optimality. In particular, we were interested in (1) whether people would be able to learn nontrivial temporal distributions of intervals and what approximations they might use, and (2) how their responses would be affected by performance feedback. Our observer model resembled the Bayesian Least Squares (BLS)

Table 4. Main statistics of the experimental distributions and nonparametrically inferred priors for bimodal distributions (Experiment 5; Standard feedback).

|              | Medium Bimodal |            |       |      | Wide Bimodal |            |       |      |
| :----------- | :------------- | :--------- | :---- | :--- | :----------- | :--------- | :---- | :--- |
|              | Objective      | Subjective |       |      | Objective    | Subjective |       |      |
| Mean (ms)    | 787.5          | 794.5      | $\pm$ | 34.2 | 787.5        | 822.1      | $\pm$ | 70.7 |
| Std (ms)     | 160.6          | 155.7      | $\pm$ | 37.2 | 251.6        | 219.2      | $\pm$ | 29.3 |
| Skewness     | 0              | -0.33      | $\pm$ | 0.39 | 0            | -0.22      | $\pm$ | 0.57 |
| Ex. Kurtosis | -1.72          | -0.08      | $\pm$ | 0.90 | -1.64        | -0.40      | $\pm$ | 0.51 |

[^0]
[^0]: Comparison between the main statistics of the 'objective' experimental distributions and the 'subjective' priors nonparametrically inferred from the data. The subjective moments are computed by averaging the moments of sampled priors pooled from all subjects ( $\pm 1$ s.d.); see Figure 9, bottom and Methods for details. doi:10.1371/journal.pcbi. 1002771 . t004

---

#### Page 14

observer described in [6], but it explicitly included an action component as part of the internal model. Moreover, to answer (1) we allowed the prior to differ from the experimental distribution, and to study (2) we considered additional shapes for the loss function in addition to the Standard squared loss $\propto(r-x)^{2}$.

The Bayesian model comparison gave us specific answers for each of our subjects, and a first validation came from the success of the most supported Bayesian observer and actor models in capturing the statistics of the subjects' responses in the task. However, goodness of fit per se is not necessarily an indicator that the components found by the model comparison reflected true findings about the subjects, rather than 'overfitting' arbitrary statistical relationships in the data. This is of particular relevance for Bayesian models, because of the underlying degeneracy among model components [21].

Our approach consisted in considering a large, 'reasonable' set of observer models that we could link to objective features of the experiment. This does not solve the degeneracy problem per se but it prevents the model comparison from finding arbitrary solutions. In particular, the set of experiments was designed in order to provide evidence that each element of the model mapped on to an experimentally verifiable counterpart; crucially, we found that a change in a component of the experimental setup (e.g. experimental distribution and feedback) correctly induced a switch in the corresponding inferred component of the model (prior and loss function). We also avoided overfitting by limiting our basic models to only two continuous noise parameters, which were then computed through model averaging and further validated by independent direct measures.

To further validate our methods, we directly measured the subject's noise parameters (sensory and motor noise, $w_{s}^{\prime}$ and $w_{m}^{\prime}$ ) in separate tasks and compared them with the model parameters $w_{s}$, $w_{m}$ inferred from the main experiments (see Section 4.1 in Text S1 for full description). The rationale is that, in an idealized situation, we would be able to measure some features of the subjects with an objective, independent procedure and the same features would be predictive of the individual performances in related tasks [16]. The measured parameters were highly predictive of the group behavior, and reasonably predictive at the individual level for the sensory parameter, confirming that the model parameters were overall correctly representing objective 'noise properties' of the subjects.

Overall, our modelling techniques were therefore validated by (a) goodness of fit, (b) consistency between inferred model components and experimental manipulations, and (c) consistency between the model parameters and independent measurements of the same quantities.

## Comparison between inferred priors and experimental distributions

Given the validation of the results of the model comparison, we performed a nonparametric inference of the priors acquired by participants during the task. Other recent works have inferred the shape of subjective 'natural' perceptual priors nonparametrically, such as in visual orientation [24] and speed [36] perception, but studies that focussed on experimentally acquired priors mostly recovered them under parametric models (e.g. Gaussian priors with variable mean and variance) [35,37-39]. The nonparametric method allowed us to study the accuracy of the subjects in learning the experimental distributions, comparing summary statistics such as the moments of the distributions up to fourth order. Note that the significance and reliability of the recovered priors is based on the correctness of our assumptions regarding the observer and actor model; unconstrained priors might capture all sorts of
statistical details, one of the typical objections to Bayesian modelling [40]. However, by dividing the model selection stage (and its validation) from the prior reconstruction process we prevented the most pathological forms of overfitting.

The internal representations inferred from the data show a good agreement with the central moments of the true distributions typically up to third order (mean, variance and skewness). Subjects however showed some difficulties in learning variance and skewness when the provided distribution was extremely peaked, with a width less than the subjects' perceptual variability. This discrepancy observed in the High-Peaked block may have arisen because (a) the experimental distribution's standard deviation was equal or lower in magnitude compared to the perceptual variability of the subjects (experimental distribution standard deviation: 80.5 ms ; subject's average sensory standard deviation at the mean of the distribution: $96.1 \pm 12.1 \mathrm{~ms}$; mean $\pm$ sd across subjects) and (b) due to the shape of the distribution, subjects had much less practice with intervals away from the peak. Another explanation is that subjects' representation of relative frequencies of different time intervals was systematically distorted, with overestimation of small relative frequencies and underestimation of large relative frequencies (see [41] for a critical review), but note that this would arguably produce a change in the mean of the distribution as well, which we did not observe.

Moreover, the recovered priors in all blocks had systematically heavier tails (higher kurtosis) than the true distributions. By exploring an extended model that included lapses we ruled out that this particular result was due to trivial outliers in our datasets. However, our results are compatible with other more sophisticated reasons for the heavy tails we recovered, in particular (a) the objective likelihoods might be non-Gaussian, with heavier tails [42], and (b) the loss functions might follow a less-than-quadratic power law [43], hypothesis for which we found some evidence, although inconclusive, by studying observer models with nonquadratic loss functions (Section 2 in Text S1). Experimentally, both (a) and (b) would imply that in our datasets there would be more outliers than we would expect from a Gaussian noise model with quadratic losses.

Our experiments with bimodal distributions show that, although people's responses were affected by the experimental distribution of intervals in a way which is clearly different from our previous experiments with uniform or peaked distributions, the inferred priors in general fail to capture bimodality and are consistent instead with a broad uniform or multimodal prior (where the peaks however do not necessarily fall at the right places). Note that the average sensory standard deviation for subjects in Experiment 5 was $87 \pm 18 \mathrm{~ms}$ (Medium Bimodal; mean $\pm$ sd across subjects) and $106 \pm 28 \mathrm{~ms}$ (Wide Bimodal), calculated at the center of the interval range. In other words, in both blocks, the centers of the peaks were well-separated in terms of perceptual discriminability (on average by at least four standard deviations). This suggests that most subjects did not simply fail to learn the bimodality of the distributions because they had problems distinguishing between the two peaks.

## Temporal recalibration and feedback

Lag adaptation is a robust phenomenon for which the perceived duration between two inter-sensory or motor-sensory events shortens after repeated exposure to a fixed lag between the two $[10,11,44]$; see [45] for a review. It is currently uncertain whether lag adaptation is a 'global' temporal recalibration effect (affecting all intervals) [46], 'local' (affecting only intervals in a neighborhood of the adapter lag) [47], or both. What is clear is that lag adaptation cannot be interpreted as a Bayesian effect in terms of

---

#### Page 15

prior expectations represented by the sample distribution of adaptation and test intervals, since its signature is a 'repulsion' from the adapter as opposed to the 'attraction' induced by a prior $[4,47,48]$.

Our experimental setup for the peaked blocks mimicked the distributions of intervals of typical lag adaptation experiments [11,29], with the adapter interval set at 675 ms (the 'peak'). However, we did not detect any noticeable disagreement with the predictions of our Bayesian observer model and, in particular, there was no significant 'repulsion effect' from the peak, neither global nor local. Our results suggest that people are not subject to the effects of lag adaptation, or can easily compensate for them, in the presence of corrective feedback.

Sensorimotor lag adaptation seems to belong to a more general class of phenomena of temporal recalibration which induce an adjustment of the produced (or estimated) timing of motor commands to meet the goals of the task at hand. In the case of experimentally induced actuator delays in a time-critical task, such as controlling a spaceship through a minefield in a videogame [49] or driving a car in a simulated environment [50], visual temporal information about delays provides an obvious, compelling reason to recalibrate the timing of actions. However, feedback regarding timing performance need not be provided only in temporal ways. Previous studies have shown that people take into account performance feedback (knowledge of results) when the feedback about the timing of their motor response is provided in various ways, such as verbal or visual report in milliseconds [23,51] or bars of variable length [52]. Interestingly, people tend to also follow 'erroneous' feedback [52-54]. However, this can be explained by the fact that people's behavior in a timing task is goal-oriented (e.g. minimizing feedback error), and therefore these experiments suggest that people are able to follow external, rather than erroneous, feedback. In fact, when participants are told that feedback might sometimes be incorrect, which corresponds to setting different expectations regarding the goal of the task, they adjust their timing estimates taking feedback less into account [53]. Ambiguity regarding the goal of a timing task with non-obvious consequences - as opposed to actions that have obvious sensorimotor consequences, such as catching a ball - can be reduced by imposing an explicit gain/loss function [5,55], and it has been found that people can act according to an externally presented asymmetric cost (even though their timing behavior is not necessarily 'optimal' [55]).

Our work extends these previous findings by performing a model comparison with different types of symmetric and asymmetric loss functions and providing additional evidence that most people are able to correctly integrate an arbitrary external feedback in their decision process, while executing a sensorimotor timing task, so to minimize the feedback error.

## Bayesian sensorimotor timing

There is growing evidence that many aspects of human sensorimotor timing can be understood in terms of Bayesian decision theory $[3,5,6]$. The mechanism through which people build time estimates, e.g. an 'internal clock', is still unclear (see [56] for a review), but it has been proposed that observers may integrate both internal and external stochastic sources of temporal information in order to estimate the passage of time $[7,57]$.

Inspired by these results, in our work we assumed that people build an internal representation of the temporal distribution of intervals presented in the experiment. However, for all timing tasks in which more or less explicit knowledge of results is given to the subjects (e.g. ours, [6,26]), an alternative explanation is that people simply learn a mapping from a duration measurement to a
given reproduction time (strategy known as table look-up), with no need of learning of a probability distribution [58]. At the moment we cannot completely discard this possibility, but other timing studies have shown that people perform according to Bayesian integration even in the absence of feedback both for simple [4,8] and possibly skewed distributions [27], suggesting that people indeed take into account the temporal statistics of the task in a context-dependent way. Moreover, previous work in motor learning in the spatial domain has shown that people do not simply learn a mapping from a stimulus to a response, but adjust their performance according to the reliability of the sensory information [15], a signature of probabilistic inference [59]. Analogous findings have been obtained in multisensory integration [18,60,61] and for visual judgements ('offset' discrimination task) under different externally imposed loss functions [20], crucially in all cases without knowledge of results. All these findings together support the idea that sensorimotor learning follows Bayesian integration, also in the temporal domain. However, the full extent of probabilistic inference in sensorimotor timing needs further study, possibly involving transfer between different conditions in the absence of knowledge of results [58].

Our results answer some of the questions raised in [6], in particular about the general shape of the distributions internalized by the subjects and the influence of feedback on the responses. An avenue for further work is related to the detailed profile of the likelihoods and possible departures from the scalar property $[34,62]$ (see also Section 4 in Text S1), especially in the case of complex experimental distributions. It is reasonable to hypothesize that strongly non-uniform samples of intervals might affect the shape of the likelihood itself, if only for the simple reason that people practice more on some given intervals. Cognitive, attentional and adaptation mechanisms might play various roles in the interaction between nonuniform priors and likelihoods, in particular without the mitigating effect of knowledge of results. A relatively less explored but important research direction involves extending the model to a biologically more realistic observer and actor model, examining the connections with network dynamics $[12,63]$ or population coding [31], bridging the gap between a normative description and mechanistic accounts of time perception. Another extension of the model would consider a nonstationary observer, whose response strategy changes from trial to trial (even after training), possibly in order to account for sequential effects of judgement which may be due to an iterative update of the prior [64-66]. Finally, whereas our analysis suggests that subjects found it relatively easy to learn unimodal distributions of intervals, bimodal distributions seemed to represent a much harder challenge. Further work is needed to understand human performance and limitations with multimodal temporal distributions.

## Methods

## Ethics statement

The University of Edinburgh School of Informatics ethics committee approved the experimental procedures and all subjects gave informed consent.

## Participants

Twenty-five subjects ( 17 male and 8 female; age range 19-34 years) including the first author participated in the study. Except for the first author all participants were naïve to the purpose of the study. All participants were right-handed, with normal or corrected-to-normal vision and reported no neurological disorder. Participants were compensated for their time and an additional

---

#### Page 16

monetary prize was awarded to the three best naïve performers (lowest mean squared error).

The first author took part in three of the experiments and was included as he represents a highly trained and motivated participant. Therefore it allowed an informal means to assess whether the author's data was different from those of the naïve participants which could reflect a lack of training or motivation. However, analysis of the author's datasets (response biases and moments of the inferred priors) were statistically indistinguishable from the other participants and therefore his data was included in the analysis.

## Materials and stimuli

Participants sat in a dimly lit room, $\sim 50 \mathrm{~cm}$ in front of a Dell M782p CRT monitor ( 160 Hz refresh rate, $640 \times 480$ resolution). Participants rested their hand on a high-performance mouse which was fixed to a table and hidden from sight under a cover. The mouse button was sampled at 1 kHz (with a $13 \pm 1 \mathrm{~ms}$ latency). Participants wore ear-enclosing headphones (Sennheiser EH2270) playing white noise at a moderate volume, thereby masking any experimental noise. Stimuli were generated by a custom-written program in MATLAB (Mathworks, U.S.A.) using the Psychophysics Toolbox extensions [67,68]. All timings were calibrated and verified with an oscilloscope.

## Task

Each trial started with the appearance of a grey fixation cross at the center of the screen ( 27 pixels, $1.5^{\prime}$ diameter). Participants were required to then click on the mouse button at a time of their choice and this led to a visual flash being displayed on the screen after a delay of $x \mathrm{~ms}$ which could vary from trial to trial. The flash consisted of a circular yellow dot ( $1.5^{\prime}$ diameter and $1.5^{\prime}$ above the fixation cross) which appeared on the screen for 18.5 ms ( 3 frames). The 'target' interval $x \mathrm{~ms}$ was defined from the start of the button press to the first frame of the flash, and was drawn from a block-dependent distribution $p(x)$. Participants were then required to reproduce the target interval by pressing and holding the mouse button for the same duration. The duration of button press ( $r \mathrm{~ms}$ ) was recorded on each trial. Participants were required to wait at least 250 ms after the flash before starting the interval reproduction, otherwise the trial was discarded and re-presented later. After the button release, $450-850 \mathrm{~ms}$ later (uniform distribution), feedback of the performance was displayed for 62 ms . This consisted of a rectangular box (height $2.5^{\prime}$, width $20^{\prime}$ ) in the lower part of the screen with a central vertical line representing zero error and a dotted line representing the reproduction error on that trial. The horizontal position of the error line relative to the zeroerror line was computed as either $f_{S k}(x, r)=\kappa \cdot \frac{r-x}{r}$ (Skewed feedback) or $f_{S t d}(x, r)=\kappa \cdot \frac{r-x}{787.5}$ (Standard feedback), depending on the experimental condition, with $\kappa=400$ pixels $\left(22.2^{\prime}\right)$. Therefore, for a response $r$ that was shorter than the target interval $x$ the error line was displayed to the left of the zero-error line, and the converse for a response longer than the target interval. The fixation cross disappeared $500-750 \mathrm{~ms}$ after the error feedback, followed by a blank screen for another 500750 ms and the reappearance of the fixation cross signalled the start of a new trial.

## Experiments

Each session consisted of around 500 trials and was broken up into runs of 84-96 trials. Within each run the number of each
interval type was set to reflect the underlying distribution exactly and the order of the presentations was then randomized. However, for the High-Peaked session we ensured that each less likely interval was always preceded by 3-5 'peak' intervals. Subjects could take short breaks between runs.

Each experiment consisted of a number of blocks, each comprising of several sessions. Within each block, the sessions were identical with regard to interval and feedback type. The participants were divided into experimental groups as follows (see also Table 1):

Experiment 1: Short Uniform and Long Uniform blocks with Skewed feedback (4 participants, including the first author). Experiment 2: Medium Uniform and Medium Peaked blocks with Skewed feedback (6 participants, including the first author). Experiment 3: Medium Uniform block with Standard feedback (6 participants, including the first author). Experiment 4: Medium HighPeaked block with Standard feedback (3 participants). Experiment 3: Medium Bimodal with Standard feedback (4 participants) and Wide Bimodal with Standard feedback (4 participants).

The order of the blocks for Experiments 1 and 2 were randomized across subjects. Each block consisted of three to six sessions, terminating when the participant's performance had stabilized (fractional change in mean squared timing error between sessions less than 0.08). For Experiment 5 we required participants to perform a minimum of five sessions.

## Data analysis

We examined the last two sessions of each block, when performance had plateaued so as to exclude any learning period of the experiment. We analysed all trials for the uniform distributions and Wide Bimodal block. For the non-uniform distributions, we picked a random subset of the frequently-sampled intervals such that all intervals contributed equally in the model comparison (results were mostly independent of the chosen random subset), with the exception of the Wide Bimodal block for which we would have had too few data points per interval. For each subject we therefore analysed about 1000 trials for the Uniform or Wide Bimodal blocks, $\sim 500$ for the Peaked or Medium Bimodal block and $\sim 200$ trials for the HighPeaked block. We discarded trials with timestamp errors (e.g. multiple or non-detected clicks) and trials whose response durations fell outside a block-dependent allowed window of $225-1237 \mathrm{~ms}$ (Short), 3001462 ms (Medium), 375-1687 ms (Long) and 225-1687 ms (Wide), giving 124 discarded trials out of a total of $\sim 30000$ trials ( $\sim 0.4 \%$ ). Note that $95 \%$ of the discarded trials had response intervals less than 150 ms , which we attribute to accidental mouse presses.

Bayesian observer model components. Eqs. 1 and 2 describe the family of Bayesian observers models. The behavior of an observer is defined by the choice of four components:
(i) a noise model for the sensory estimation process, which can be either constant or scalar:

$$
p_{s}\left(y \mid x ; w_{s}\right)=\left\{\begin{array}{lr}
\mathcal{N}\left(y \mid x, w_{s} \cdot 787.5\right) & \text { (constant) } \\
\mathcal{N}\left(y \mid x, w_{s} x\right) & \text { (scalar) }
\end{array}\right.
$$

where $\mathcal{N}(x \mid \mu, \sigma)$ is a normal distribution with mean $\mu$ and standard deviation $\sigma$.
(ii) a noise model for the motor reproduction process, which can be either constant or scalar:

$$
p_{m}\left(r \mid u ; w_{m}\right)=\left\{\begin{array}{ll}
\mathcal{N}\left(r \mid u, w_{m} \cdot 787.5\right) & \text { (constant) } \\
\mathcal{N}\left(r \mid u, w_{m} u\right) & \text { (scalar) }
\end{array}\right.
$$

---

#### Page 17

(iii) the approximation scheme for the priors. We considered: (a) the true, discrete distribution; (b) a single Gaussian with same mean and variance as the true distribution; (c) a mixture of six (ten for the Wide range) 37.5 ms standard deviation Gaussians centered on the true discrete intervals with mixing weights equal to the relative probability of the true intervals; (d) as c but with standard deviation of 75 ms ; (e) a continuous uniform distribution from the shortest to the longest interval. For Experiment 2 and 4 we also considered a mixture of two Gaussians with mixing weights $\pi$ and $1-\pi$, with $\pi$ equal to the proportion of 'peak' intervals that emerge from the uniform background distribution ( $\pi=0$ for the Uniform block, $\pi=0.5$ for the Peaked block and $\pi=0.75$ for the High-Peaked block). The first Gaussian is centered on the peak ( 675 ms ) and with a small (f: 37.5 ms ) or large (g: 75 ms ) standard deviation, the second Gaussian is centered on the mean of the Medium range ( 787.5 ms ) and with standard deviation equal to the discrete Uniform distribution ( 128.7 ms ). Therefore, for the Medium Uniform block approximation schemes f and g reduce to a single Gaussian. Analogously, for Experiment 5 we considered a mixture of three Gaussians with mixing weights $\pi, \pi$ and $1-2 \pi$, with $\pi$ equal to the total frequency of one of the two 'peaks' emerging from the uniform background distribution ( $\pi=1 / 4$ for the Medium Bimodal block and $\pi=9 / 28$ for the Wide Bimodal block). The first two Gaussians are centered on the peaks (Medium: 600 ms and 975 ms ; Wide: 525 ms and 1050 ms ) and with a small (f: Medium: 37.5 ms ; Wide: 61.2 ms ) or large (g: twice the small) standard deviation. The third Gaussian is centered on the mean of the range $(787.5 \mathrm{~ms})$ and with standard deviation equal to the discrete Uniform distribution over the range (Medium: 128.7 ms ; Wide: 251.6 ms ). The values of standard deviations for the 'peak' Gaussians (small 37.5 ms , large 75 ms ) were chosen as 75 ms is the gap between time intervals in all experimental distributions. For the Wide Bimodal block, 61.2 ms is the standard deviation of the sample for three intervals separated by 75 ms .
(iv) the loss function

$$
\tilde{f}^{2}(x, r)= \begin{cases}\left(\frac{r-x}{r}\right)^{2} & \text { (Skewed) } \\ (r-x)^{2} & \text { (Standard) } \\ \left(\frac{r-x}{x}\right)^{2} & \text { (Fractional) }\end{cases}
$$

Note that the Fractional error was not used as a feedback shape in the experiments, but we included it as a possibility for the Bayesian observer as it might represent an appropriate error signal if time has a logarithmic representation in the brain [69]. In fact, the logarithmic squared loss reads:

$$
\begin{aligned}
(\log r-\log x)^{2} & =\left(\log \frac{r}{x}\right)^{2}=\left(\log \left|1+\frac{r-x}{x}\right|\right)^{2} \\
& \approx\left(\frac{r-x}{x}\right)^{2} \quad \text { for }\left|\frac{r-x}{x}\right| \ll 1
\end{aligned}
$$

For an analysis with non-quadratic loss function see also Section 2 in Text S1.

Bayesian model comparison. For each observer model and each subject's dataset (that is all blocks within an experiment) we
calculated the posterior probability of the model given the data, $\operatorname{Pr}(\text { model } \mid \text { data }) \propto \operatorname{Pr}(\text { data } \mid \text { model })$, assuming a flat prior over the models.

The marginal likelihood is given by

$$
\begin{aligned}
\operatorname{Pr}(\text { data } \mid \text { model })= & \int d w_{s} d w_{m} \operatorname{Pr}(\text { data } \mid w_{s}, w_{m}, \text { model }) \\
& \operatorname{Pr}\left(w_{s}, w_{m} \mid \text { model }\right)
\end{aligned}
$$

where $\operatorname{Pr}\left(w_{s}, w_{m} \mid\right.$ model) is the prior over the parameters and $\operatorname{Pr}\left(\right.$ data $\mid w_{s}, w_{m}$, model) is the likelihood of the data given a specific model and value of the parameters. For the prior over the parameters we assumed independence between parameters and models $\operatorname{Pr}\left(w_{s}, w_{m} \mid\right.$ model $)=\operatorname{Pr}\left(w_{s}\right) \operatorname{Pr}\left(w_{m}\right)$ and for both parameters we used a broad Beta prior $\sim \operatorname{Beta}(1.3,2.6)$ that slightly favors the range $0.03-0.3$ in agreement with a vast literature on human timing errors [34]. The likelihood of the data was computed according to our observer model, Eq. 2, assuming independence across trials:

$$
\operatorname{Pr}(\text { data } \mid w_{s}, w_{m}, \text { model })=\prod_{i=1}^{n} p\left(r^{i \mid}\left|x^{i}\right| ; w_{s}, w_{m}\right)
$$

with $n$ the total number of test trials and $x^{(i)}, r^{(i)}$ respectively the target interval and response in the $i-\mathrm{th}$ test trial. Note that the calculation of $p(r \mid x)$ (Eq. 2) requires a computation of the optimal action $u^{*}$, that is, the action $u$ that minimizes the expected loss [Eq. 1). The minimization was performed analytically for the Standard and Fractional loss function and numerically for the Skewed loss function (function fminbnd in MATLAB; we assumed that $u^{*}$ always fell in the interval $20-2000 \mathrm{~ms}$; the results were checked against analytical results obtained through a polynomial expansion approximation of the loss function that holds for $\left|\frac{r-x}{x}\right| \ll 1$ ).
We computed the marginal likelihood through Eqs. 6 and 7 both with a full numerical integration and using a Laplace approximation (both methods gave identical results). Given the posterior probability for each model, for each subject we calculated the posterior probability for each model component (by fixing a model component and summing over the others); see Table S1. The 'Bayesian fits' in Figure 3, 4, 9 top and Figure S5 and S6 in Text S2 were obtained by calculating the model average for the response bias and response standard deviation (the average was taken both over parameters and over models, but typically only one of the models contributed significantly to the integral).

Nonparametric reconstruction of the priors. To examine the subjects' priors using a nonparametric approach, for each subject we took the (i) sensory and (ii) motor noise and (iv) loss function, as inferred from the model comparison. We then allowed the priors to vary independently over a broad class of smooth, continuous distributions. For each block, the log prior was specified by the values of ten ( 14 for the Wide range) control points at 75 ms steps over the ranges: Short $300-1025 \mathrm{~ms}$, Medium $450-1175 \mathrm{~ms}$, Long $600-1325 \mathrm{~ms}$ and Wide $300-$ 1325 ms . The control points were centered on the interval range of the block but extended outside the range to allow for tails or shifts. The prior $q(x)$ was calculated by interpolating the values of the prior in log space with a Gaussian process [70] with squared exponential covariance function with fixed scale ( $\sigma_{\varepsilon}=1$ in log space, $\ell=75 \mathrm{~ms}$ ) and a small nonzero noise term to favor conditioning. The Gaussian processes were used only as a smooth interpolating method and not as a part of the inference. In order to infer the prior for each subject and block, we sampled

---

#### Page 18

from the posterior distribution of priors $\propto$ Pr(data| prior, model) using a slice sampling Markov Chain Monte Carlo algorithm [71]. We ran ten parallel chains ( 3000 burn-in samples, 1500 saved samples per chain) obtaining a total of 15000 sampled priors per subject and block. For each sampled prior we calculated the first four moments (mean, standard deviation, skewness and excess kurtosis) and computed the mean and standard deviation of the moments across the sample sets of individual subjects and over the sample set of all subjects (the latter are shown in Table 2 and 3).

---

# Internal Representations of Temporal Statistics and Feedback Calibrate Motor-Sensory Interval Timing - Appendix

---

## Colophon

Citation: Acerbi L, Wolpert DM, Vijayakumar S (2012) Internal Representations of Temporal Statistics and Feedback Calibrate Motor-Sensory Interval Timing. PLoS Comput Biol 8(11): e1002771. doi:10.1371/journal.pcbi. 1002771

Editor: Laurence T. Maloney, New York University, United States of America
Received May 25, 2012; Accepted September 24, 2012; Published November 29, 2012
Copyright: © 2012 Acerbi et al. This is an open-access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.
Funding: This study was supported by an Engineering and Physical Sciences Research Council/Medical Research Council scholarship granted to LA from the Neuroinformatics and Computational Neuroscience Doctoral Training Centre at the University of Edinburgh. DMW is supported by the Wellcome Trust and the Human Frontiers Science Program. SV is supported through grants from Microsoft Research, Royal Academy of Engineering and EU FP7 programs. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.
Competing Interests: The authors have declared that no competing interests exist.

- E-mail: LAcerb@sms.ed.ac.uk

## Acknowledgments

We thank Iain Murray for useful discussion regarding Monte Carlo methods and Paolo Puggioni for comments on an earlier draft of this manuscript.

## Author Contributions

Conceived and designed the experiments: LA DMW SV. Performed the experiments: LA. Analyzed the data: LA. Wrote the paper: LA DMW SV.

## References

1. Maak MD, Buonomano DV (2004) The neural basis of temporal processing. Annu Rev Neurosci 27: 307-340.
2. Bubusi C, Meck W (2005) What makes us tick? Functional and neural mechanisms of interval timing. Nat Rev Neurosci 6: 755-765.
3. Miyazaki M, Nozaki D, Nakajima Y (2005) Testing bayesian models of human coincidence timing. Journal of neurophysiology 94: 395-399.
4. Miyazaki M, Yamamoto S, Uchida S, Kitazawa S (2006) Bayesian calibration of simultaneity in tactile temporal order judgment. Nat Neurosci 9: 875-877.
5. Hudson T, Maloney L, Landy M (2008) Optimal compensation for temporal uncertainty in movement planning. PLoS Comput Biol 4: e1000130.
6. Jazayeri M, Shadreu MN (2010) Temporal context calibrates interval timing. Nat Neurosci 13: 1020-1026.
7. Alterno MB, Sabani M (2011) Observers exploit stochastic models of sensory change to help judge the passage of time. Curr Biol 21: 200-206.
8. Cicchini G, Arrighi R, Cecchetti L, Giusti M, Burr D (2012) Optimal encoding of interval timing in expert percussionists. J Neurosci 32: 1056-1060.
9. Eagleman DM (2008) Human time perception and its illusions. Curr Opin Neurobiol 18: 131-136.
10. Fujisaki W, Shimojo S, Kashino M, Nishida S (2004) Recalibration of audiovisual simultaneity. Nat Neurosci 7: 773-778.
11. Sietson C, Cui X, Montague P, Eagleman D (2006) Motor-sensory recalibration leads to an illusory reversal of action and sensation. Neuron 51: 651-659.
12. Karmarkar UR, Buonomano DV (2007) Timing in the absence of clocks: encoding time in neural network states. Neuron 53: 427-438.
13. Pariyadath V, Eagleman D (2007) The effect of predictability on subjective duration. PLoS One 2: e1264.
14. Kording K, Wolpert D (2006) Bayesian decision theory in sensorimotor control. Trends Cogn Sci 10: 319-326.
15. Kording KP, Wolpert DM (2004) Bayesian integration in sensorimotor learning. Nature 427: 244-247.
16. Taminari H, Hudson T, Landy M (2006) Combining priors and noisy visual cues in a rapid pointing task. J Neurosci 26: 10154-10163.
17. Trommershäuser J, Maloney L, Landy M (2008) Decision making, movement planning and statistical decision theory. Trends Cogn Sci 12: 291-297.
18. Beierholm U, Quartz S, Shams L (2009) Bayesian priors are encoded independently from likelihoods in human multisensory perception. J Vis 9: 1-9.
19. Vilares I, Howard J, Fernandes H, Gottfried J, Kording K (2012) Differential representations of prior and likelihood uncertainty in the human brain. Curr Biol 22: 1641-1648.
20. Whiteley L, Sabani M (2008) Implicit knowledge of visual uncertainty guides decisions with asymmetric outcomes. J Vis 8: 1-15.
21. Mamassian P, Landy MS (2010) It's that time again. Nat Neurosci 13: 914-916.
22. Salmoni A, Schmidt R, Walter C (1984) Knowledge of results and motor learning: a review and critical reappraisal. Psychol Bull 95: 355-386.
23. Blackwell J, Newell K (1996) The informational role of knowledge of results in motor learning. Acta Psychol (Amst) 92: 119-129.
24. Girshick A, Landy M, Simoncelli E (2011) Cardinal rules: visual orientation perception rejects knowledge of environmental statistics. Nat Neurosci 14: 926932 .
25. Rakitin B, Gibbon J, Penney T, Malapani C, Hinton S, et al. (1998) Scalar expectancy theory and peak-interval timing in humans. J Exp Psychol Anim Behav Process 24: 15-33.
26. Jones MR, McAuley JD (2005) Time judgments in global temporal contexts. Percept Psychophys 67: 398-417.
27. Lawrence R (2011) Temporal context affects duration reproduction. J Cogn Psychol 23: 157-170.
28. Haggard P, Clark S, Kalogeras J (2002) Voluntary action and conscious awareness. Nat Neurosci 5: 382-385.
29. Heron J, Hanson JVM, Whitaker D (2009) Effect before cause: supramodal recalibration of sensorimotor timing. PLoS One 4: e7681.
30. Mates J, Müller U, Radil T, Pöppel E (1994) Temporal integration in sensorimotor synchronization. J Cogn Neurosci 6: 332-340.
31. Heron J, Aaco-Sockdale C, Hotchkiss J, Roach N, McGraw P, et al. (2012) Duration channels mediate human time perception. Proc Biol Sci 279: 690-698.
32. Chalk M, Seitz A, Seriès P (2010) Rapidly learned stimulus expectations alter perception of motion. J Vis 10: 1-18.
33. Hollingsorth H (1910) The central tendency of judgment. J Philos Psychol Sci Methods 7: 461-469.
34. Lewis PA, Miall RC (2009) The precision of temporal judgement: milliseconds, many minutes, and beyond. Proc Biol Sci 364: 1897-1905.
35. Battaglia PW, Kresten D, Schraer PR (2011) How haptic size sensations improve distance perception. PLoS Comput Biol 7: e1002080.
36. Stocker AA, Simoncelli EP (2006) Noise characteristics and prior expectations in human visual speed perception. Nat Neurosci 9: 570-585.
37. Berniker M, Voss M, Kording K (2010) Learning priors for bayesian computations in the nervous system. PLoS One 5: e12686.
38. Sotiropoulos G, Seitz A, Seriès P (2011) Changing expectations about speed alters perceived motion direction. Curr Biol 21: R883-R884.
39. Turnham E, Braun D, Wolpert D (2011) Inferring visuomotor priors for sensorimotor learning. PLoS Comput Biol 7: e1001112.
40. Jones M, Love B (2011) Bayesian Fundamentalism or Enlightenment? On the explanatory status and theoretical contributions of Bayesian models of cognition. Behav Brain Sci 34: 169-188.
41. Zhang H, Maloney L (2012) Ubiquitous log odds: a common representation of probability and frequency distortion in perception, action, and cognition. Front Neurosci 6.
42. Natarajan R, Murray I, Shams L, Zemel RS (2009) Characterizing response behavior in multisensory perception with conicting cues. Adv Neural Inf Process Syst 21: 1155-1160.
43. Kording KP, Wolpert DM (2004) The loss function of sensorimotor learning. Proc Natl Acad Sci U S A 101: 9839-9842.
44. Vroomen J, Keetels M, de Gelder R, Bertelson P (2004) Recalibration of temporal order perception by exposure to audio-visual asynchrony. Cogn Brain Res 22: 32-35.
45. Vroomen J, Keetels M (2010) Perception of intersensory synchrony: a tutorial review. Atten Percept Psychophys 72: 871-884.
46. Di Luca M, Machulla Tk, Ernst MO (2009) Recalibration of multisensory simultaneity: cross-modal transfer coincides with a change in perceptual latency. J Vis 9: 1-16.
47. Roach N, Heron J, Whitaker D, McGraw P (2011) Asynchrony adaptation reveals neural population code for audio-visual timing. Proc Biol Sci 278: 13141322 .
48. Stocker A, Simoncelli E (2006) Sensory adaptation within a bayesian framework for perception. Adv Neural Inf Process Syst 18: 1291-1298.

---

#### Page 19

49. Cunningham D, Chatziasmos A, Von der Heyde M, Bühlhoff H (2001) Driving in the future: temporal visuomotor adaptation and generalization. J Vis 1: 88 98.
50. Cunningham DW, Billock VA, Tson BH (2001) Sensorimotor adaptation to violations of temporal contiguity. Psychol Sci 12: 532-535.
51. Franssen V, Vandierendonck A (2002) Time estimation: does the reference memory mediate the effect of knowledge of results? Acta Psychol (Amst) 109: 239-267.
52. Ryan L, Robey T (2002) Learning and performance effects of accurate and erroneous knowledge of results on time perception. Acta Psychol (Amst) 111: 83100 .
53. Ryan L, Henry K, Robey T, Edwards J (2004) Resolution of conicts between internal and external information sources on a time reproduction task: the role of perceived information reliability and attributional style. Acta Psychol (Amst) 117: 205-229.
54. Ryan L, Fritz M (2007) Erroneous knowledge of results affects decision and memory processes on timing tasks. J Exp Psychol Hum Percept Perform 33: $1468-1482$.
55. Mamassian P (2008) Overconfidence in an objective anticipatory motor task. Psychol Sci Public Interest 19: 601-606.
56. Grondin S (2010) Timing and time perception: a review of recent behavioral and neuroscience findings and theoretical directions. Atten Percept Psychophys 72: $561-582$.
57. Haas J, Herrmann J (2012) The neural representation of time: An informationtheoretic perspective. Neural Comput 24: 1519-1552.
58. Maloney L, Mamassian P, et al. (2009) Bayesian decision theory as a model of human visual perception: testing Bayesian transfer. Vis Neurosci 26: 147-155.
59. Ma W (2012) Organizing probabilistic models of perception. Trends Cogn Sci 16: 511-518.
60. Ernst M, Banks M (2002) Humans integrate visual and haptic information in a statistically optimal fashion. Nature 415: 429-433.
61. Alais D, Burr D (2004) The ventriloquint effect results from near-optimal bimodal integration. Curr Biol 14: 257-262.
62. Zarco W, Merchant H, Peads L, Meudez JC (2009) Subsecond timing in primates: comparison of interval production between human subjects and rhesus monkeys. J Neurophysiol 102: 5191-202.
63. Buonomano D, Luje R (2010) Population clocks: motor timing with neural dynamics. Trends Cogn Sci 14: 520-527.
64. Stewart N, Brown G, Chater N (2005) Absolute identification by relative judgment. Psychol Rev 112: 881-911.
65. Petzschner F, Glasauer S (2011) Iterative bayesian estimation as an explanation for range and regression effects: a study on human path integration. J Neurosci 31: 17220-17229.
66. Saunders I, Vijayakumar S (2012) Continuous evolution of statistical estimators for optimal decision-making. PLoS One 7: e37547.
67. Brainard D (1997) The psychophysics toolbox. Spat Vis 10: 433-436.
68. Pelli D (1997) The videotoolbox software for visual psychophysics: transforming numbers into movies. Spat Vis 10: 437-442.
69. Gibbon J (1981) On the form and location of the psychometric bisection function for time. J Math Psychol 24: 58-87.
70. Rasmussen C, Williams CKI (2006) Gaussian Processes for Machine Learning. The MIT Press.
71. Neal R (2003) Slice sampling. Ann Stat 31: 705-741.

---

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