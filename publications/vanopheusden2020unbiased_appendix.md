# Unbiased and Efficient Log-Likelihood Estimation with Inverse Binomial Sampling - Appendix

---

#### Page 67

# Supplementary Material 

## A Further theoretical analyses

## A. 1 Why inverse binomial sampling works

We start by showing that the inverse binomial sampling policy described in Section 2.4, combined with the estimator $\hat{\mathcal{L}}_{\text {ibs }}$ (Equation 14), yields a uniformly unbiased estimate of $\log p$. This derivation follows from de Groot (1959, Theorem 4.1), adapted to our special case of estimating $\log p$ instead of an arbitrary function $f(p)$ :

$$
\begin{aligned}
\mathbb{E}\left[\hat{\mathcal{L}}_{\text {ibs }}\right] & =-\mathbb{E}\left[\sum_{k=1}^{K-1} \frac{1}{k}\right]=-\mathbb{E}\left[\sum_{k=1}^{\infty} \frac{1}{k} \mathbb{1}_{k<K}\right] \\
& =-\sum_{k=1}^{\infty} \frac{1}{k} \mathbb{E}\left[\mathbb{1}_{k<K}\right]=-\sum_{k=1}^{\infty} \frac{1}{k} \operatorname{Pr}(k<K) \\
& =-\sum_{k=1}^{\infty} \frac{1}{k}(1-p)^{k}=\log p
\end{aligned}
$$

The first equality is the definition of $\hat{\mathcal{L}}_{\text {ibs }}$ (Equation 14), using the notational convention that $\sum_{k=1}^{0}=0$. In the second equality we introduce the indicator function $\mathbb{1}_{k<K}$ which is 1 when $k<K$ and 0 otherwise. The third equality follows by linearity of the expectation and the fourth directly from the definition of the indicator function. The fifth and second-to last equality uses the formula for the cumulative distribution function of a geometric variable, that is $\operatorname{Pr}(K \leq k)=1-(1-p)^{k}$, and thus $\operatorname{Pr}(k<K)=(1-p)^{k}$. The final equality is the definition of the Taylor series of $\log p$ expanded around $p=1$. Note that this series converges for all $p \in(0,1]$.

In the derivation above, we can replace $\frac{1}{k}$ by an arbitrary set of coefficients $a_{k}$ and

---

#### Page 68

show that

$$
\mathbb{E}\left[\sum_{k=1}^{K-1} a_{k}\right]=\sum_{k=1}^{\infty} a_{k}(1-p)^{k}
$$

for all $p$ for which the resulting Taylor series converges. Equation S2 immediately proves two corollaries. First, we can use the inverse binomial sampling policy to estimate any analytic function of $p$. Second, since we can rewrite any estimator $\hat{\mathcal{L}}(K)$ as $\sum_{k=1}^{K-1} a_{k}$, and since Taylor series are unique, $a_{k}=\frac{1}{k}$ is the only choice for which $\mathbb{E}[\hat{\mathcal{L}}(K)]$ equals $\log p$. In other words, $\hat{\mathcal{L}}_{\text {ibs }}$ is the only uniformly unbiased estimator of $\log p$ with the inverse sampling policy. Therefore, it trivially is the uniformly minimumvariance unbiased estimator under this policy, since no other unbiased estimator exist.

# A. 2 Analysis of bias of fixed sampling 

We provide here a more formal analysis of the bias of fixed sampling. We initially consider the estimator $\hat{\mathcal{L}}_{\text {fixed }}$ defined by Equation 11 in the main text, but we will see that our arguments hold generally for any estimator based on a fixed sampling policy.

We showed in Figure 2 that in the regime of $p \rightarrow 0, M \rightarrow \infty$, while keeping $p M \rightarrow \lambda$, the bias of $\hat{\mathcal{L}}_{\text {fixed }}$ tends to a master curve. This follows since, in this limit, the binomial distribution $\operatorname{Binom}\left(\frac{\lambda}{M}, M\right)$ converges to a Poisson distribution $\operatorname{Pois}(\lambda)$ and therefore the bias converges to

$$
\begin{aligned}
\text { Bias }\left[\hat{\mathcal{L}}_{\text {fixed }} \mid p\right] & =\mathbb{E}\left[\hat{\mathcal{L}}_{\text {fixed }}-\log \frac{\lambda}{M}\right] \\
& =\exp (-\lambda) \sum_{m=0}^{\infty} \frac{\lambda^{m}}{m!} \log (m+1)-\log (M+1)-\log \frac{\lambda}{M} \\
& \underset{\substack{M \rightarrow 0 \\
p M \rightarrow \lambda}}{\longrightarrow} \exp (-\lambda) \sum_{m=0}^{\infty} \frac{\lambda^{m}}{m!} \log (m+1)-\log \lambda
\end{aligned}
$$

---

#### Page 69

which is the master curve in Figure 2. In particular, the bias is close to zero for $\lambda \gg 1$ and it diverges when $\lambda \ll 1$, or equivalently, for $M \gg \frac{1}{p}$ and $M \ll \frac{1}{p}$, respectively.

This asymptotic behavior is not a coincidence. In fact, it is mathematically guaranteed since the Fisher information of $\operatorname{Pois}(\lambda)$ equals $\frac{1}{\lambda}$ and the reparametrization identity for the Fisher information yields $\mathcal{I}_{f}(\log \lambda)=\lambda$ (Lehmann and Casella, 2006). In the limit of $p \ll \frac{1}{\lambda^{2}}$, which corresponds to $\lambda \ll 1$, this Fisher information vanishes and the outcome of fixed sampling simply provides zero information about $\log \lambda$ or $\log p$. Therefore, any estimates of $\log p$ are not informed by the data and instead are a function of the regularization chosen in the estimator $\hat{\mathcal{L}}_{\text {fixed }}$ (Equation 11). Note that the argument above does not invoke the specific form of the estimator, and therefore holds for any choice of regularization.

We can express the problem with fixed sampling more clearly using Bayesian statistics, in a formal treatment of the 'gambling' analogy we presented in the main text. The 'correct' belief about $\log \lambda$ given the outcome of fixed sampling $(m)$ is quantified by the posterior distribution $p(\log \lambda \mid m)$, which is a product of the likelihood $\operatorname{Pr}(m \mid \log \lambda)$ and a prior $p(\log \lambda)$. In the limit $\lambda \ll 1$, the Poisson distribution converges to a Kronecker delta distribution concentrated on $m=0$. In other words, almost surely none of the samples taken by the behavioral model will match the participant's response. When $m=0$, the likelihood equals $\exp (-\lambda)$, which is mostly flat (when considered as a function of $\log \lambda$, see Figure S1) for $\log \lambda \in[-\infty,-2]$ and therefore our posterior belief ought to be dominated by the prior $p(\log \lambda)$ and become independent of the data. Therefore, we once again conclude that in the limit $p \ll \frac{1}{\lambda^{2}}$, the fixed sampling policy provides no information to base an estimate of $\log p$ on, and it is impossible to avoid

---

#### Page 70

bias.

> **Image description.** This is a line graph showing the relationship between log λ and Pr(λ|m = 0).
> 
> The graph has the following features:
> 
> *   **Axes:**
>     *   The x-axis is labeled "log λ" and ranges from -8 to 2.
>     *   The y-axis is labeled "Pr(λ|m = 0)" and ranges from 0 to 1. Tick marks are present at 0, 0.5, and 1.
> *   **Data:**
>     *   A blue line starts at y=1 when x is -8.
>     *   The line remains approximately flat at y=1 until x is around -2.
>     *   The line then curves downwards, decreasing rapidly as x increases.
>     *   The line approaches y=0 as x approaches 2.

Figure S1: Likelihood function of $\log \lambda$ given that fixed sampling returns $m=0$ (none of the samples from the model match the participant's response). The likelihood is approximately flat for all $\log \lambda \leq-2$. Since $\lambda$ is defined as $\frac{p}{M}$, this implies that the posterior distribution over $p$ will be dominated by a prior rather than evidence, as quantified by the likelihood.

# A. 3 Derivation of IBS variance 

In this section, we derive the expression for the variance of the IBS estimator (Equation 15 in the main text). We compute the variance of $\hat{L}_{\text {ibs }}$ starting from the identity

$$
\operatorname{Var}\left[\hat{L}_{\mathrm{ibs}}\right]=\mathbb{E}\left[\left(\hat{L}_{\mathrm{ibs}}\right)^{2}\right]-\left(\mathbb{E}\left[\hat{L}_{\mathrm{ibs}}\right]\right)^{2}
$$

We already know the second term is equal to $(\log p)^{2}$, but for the purpose of this derivation, and for reasons that will become clear later, we re-write it as

$$
\left(\mathbb{E}\left[\hat{L}_{\mathrm{ibs}}\right]\right)^{2}=\left(\sum_{m=1}^{\infty} \frac{1}{m}(1-p)^{m}\right)^{2}=\sum_{m, n=1}^{\infty} \frac{1}{m n}(1-p)^{m+n}
$$

In order to write this equation as a power series in $1-p$, we collect terms with the same exponent together. Specifically, we re-index this double summation as a summation

---

#### Page 71

over all values of $n$ and $m+n$ (which we label $k$ ), and substitute $k-n$ for $m$.

$$
\sum_{m, n=1}^{\infty} \frac{1}{m n}(1-p)^{m+n}=\sum_{k=1}^{\infty} \sum_{n=1}^{k-1} \frac{1}{(k-n) n}(1-p)^{k}
$$

Note that in the second summation over $n$ we only have to sum to $n=k-1$ since $m \equiv n-k$ has to be positive. We can carry out the internal summation over $n$ explicitly,

$$
\sum_{n=1}^{k-1} \frac{1}{(k-n) n}=\sum_{n=1}^{k-1}\left[\frac{1}{k(k-n)}+\frac{1}{k n}\right]=\frac{2}{k} \sum_{n=1}^{k-1} \frac{1}{n}=\frac{2}{k} H_{k-1}
$$

The first equality is an algebraic manipulation, the second follows by symmetry and the final equality defines $H_{k-1}$ as the $(k-1)$-th harmonic number. Therefore, we find that

$$
\left(\mathbb{E}\left[\hat{L}_{\mathrm{ibs}}\right]\right)^{2}=2 \sum_{k=1}^{\infty} \frac{H_{k-1}}{k}(1-p)^{k}
$$

To calculate $\mathbb{E}\left[\left(\hat{L}_{\mathrm{ibs}}\right)^{2}\right]$, we use a similar rationale as Equation S1,

$$
\begin{aligned}
\mathbb{E}\left[\left(\hat{L}_{\mathrm{ibs}}\right)^{2}\right] & =\mathbb{E}\left[\left(-\sum_{m=1}^{K-1} \frac{1}{m}\right)^{2}\right]=\mathbb{E}\left[\sum_{m, n=1}^{K-1} \frac{1}{m n}\right] \\
& =\mathbb{E}\left[\sum_{m, n=1}^{\infty} \frac{1}{m n} \mathbb{1}_{m<K} \mathbb{1}_{n<K}\right]=\sum_{m, n=1}^{\infty} \frac{1}{m n} \mathbb{E}\left[\mathbb{1}_{m<K, n<K}\right] \\
& =\sum_{m, n=1}^{\infty} \frac{1}{m n} \operatorname{Pr}\left(\mathbb{1}_{m<K, n<K}\right)=\sum_{m, n=1}^{\infty} \frac{1}{m n}(1-p)^{\max (m, n)}
\end{aligned}
$$

In these equations, $\mathbb{1}$ again denotes an indicator function, and we use the fact that the product of indicator functions for two different events is the indicator function for the joint event. Additionally, we use that the event $m<K, n<K$ is logically equivalent to $\max (m, n)<K$. To write this double summation as a power series, we split it into three parts: one where $m<n$, one where $m=n$ and one where $m>n$. By symmetry, the first and last part are equal, and we can write

$$
\mathbb{E}\left[\left(\hat{L}_{\mathrm{ibs}}\right)^{2}\right]=\sum_{m=1}^{\infty} \frac{1}{m m}\left[(1-p)^{\max (m, m)}\right]+2 \sum_{m=1}^{\infty} \sum_{n=1}^{m-1} \frac{1}{m n}\left[(1-p)^{\max (m, n)}\right]
$$

---

#### Page 72

By re-arranging some terms, and using the fact that $\max (m, m)=m$ and $\max (m, n)=$ $m$ for all $n<m$, we can reduce this to

$$
\mathbb{E}\left[\left(\hat{L}_{\mathrm{ibs}}\right)^{2}\right]=\sum_{m=1}^{\infty} \frac{1}{m^{2}}(1-p)^{m}+2 \sum_{m=1}^{\infty}\left[\sum_{n=1}^{m-1} \frac{1}{n}\right] \frac{1}{m}(1-p)^{m}
$$

We can now explicitly perform the summation over $n$ in the second term and write

$$
\mathbb{E}\left[\left(\hat{L}_{\mathrm{ibs}}\right)^{2}\right]=\sum_{m=1}^{\infty} \frac{1}{m^{2}}(1-p)^{m}+2 \sum_{m=1}^{\infty} \frac{H_{m-1}}{m}(1-p)^{m}
$$

Finally, putting everything together, we obtain

$$
\begin{aligned}
\operatorname{Var}\left[\hat{L}_{\mathrm{ibs}}\right] & =\mathbb{E}\left[\left(\hat{L}_{\mathrm{ibs}}\right)^{2}\right]-\left(\mathbb{E}\left[\hat{L}_{\mathrm{ibs}}\right]\right)^{2} \\
& =\sum_{m=1}^{\infty} \frac{1}{m^{2}}(1-p)^{m}+2 \sum_{m=1}^{\infty} \frac{H_{m-1}}{m}(1-p)^{m}-2 \sum_{k=1}^{\infty} \frac{H_{k-1}}{k}(1-p)^{k} \\
& =\sum_{m=1}^{\infty} \frac{1}{m^{2}}(1-p)^{m}
\end{aligned}
$$

# A. 4 Estimator variance and information inequality 

We proved in Section A. 1 that $\hat{\mathcal{L}}_{\text {IBS }}$ is the minimum-variance unbiased estimator of $\log p$ given the inverse binomial sampling policy. Here we show that the estimator also comes close to saturating the information inequality, the analogue of a Cramer-Ráo bound for an arbitrary function $f(p)$ and a non-fixed sampling policy (de Groot, 1959),

$$
\operatorname{Std}(\hat{f} \mid p) \geq \sqrt{\frac{p(1-p)}{\mathbb{E}[K|p|)}}\left|\frac{\mathrm{d} f(p)}{\mathrm{d} p}\right|
$$

In our case, where $f(p)=\log p$, the information inequality reduces to $\operatorname{Std}\left(\hat{\mathcal{L}}_{\text {IBS }}\right) \geq$ $\sqrt{1-p}$. In Figure S2, we plot the standard deviation of IBS compared to this lower bound.

---

#### Page 73

> **Image description.** A line graph displays the standard deviation of IBS and its lower bound against the variable *p*.
> 
> *   The x-axis is labeled "*p*" and ranges from 0 to 1 in increments of 0.2.
> *   The y-axis is labeled "Standard deviation" and ranges from 0 to 1.5 in increments of 0.5.
> *   A blue curve represents "IBS" and starts at approximately 1.3 on the y-axis when *p* is 0, gradually decreasing as *p* increases.
> *   A black curve represents "lower bound" and starts at 1 on the y-axis when *p* is 0, decreasing more rapidly than the blue curve as *p* increases. The two curves converge around *p* = 0.7 and continue to decrease, reaching 0 at *p* = 1.
> *   A legend in the upper right corner identifies the blue curve as "IBS" and the black curve as "lower bound".

Figure S2: Standard deviation of IBS (Blue curve) and the lower bound given by the information inequality (black, see Equation S4). The standard deviation of IBS is within $30 \%$ of the lower bound across the entire range of $p$.

It may be disappointing that IBS does not match the information inequality. Kolmogorov (1950) showed that the only functions $f(p)$ for which the fixed sampling policy with $M$ samples allows an unbiased estimator are polynomials of degree up to $M$, and those estimators can saturate the information equality. Dawson (1953) and later de Groot (1959) showed that if an unbiased estimator of a non-polynomial function $f(p)$ exists and it matches the information inequality, it must use the inverse binomial sampling policy. Moreover, de Groot derived necessary and sufficient conditions for $f(p)$ to allow such estimators (de Groot, 1959). Applying this argument to $f(p)=\log (p)$, the standard deviation in IBS is close (within $30 \%$ ) to its theoretical minimum.

To compare the variance of IBS and fixed sampling on equal terms, we use the scaling behavior of $\hat{\mathcal{L}}_{\text {fixed }}$ as $M \rightarrow \infty$. Specifically, for fixed sampling, we plot $\sqrt{M} \times$ $\operatorname{Std}\left(\hat{\mathcal{L}}_{\text {fixed }}\right)$ and for IBS we plot $\frac{1}{\sqrt{p}} \times \operatorname{Std}\left(\hat{\mathcal{L}}_{\text {IBS }}\right)$ (see Figure S3). With this scaling, the curves for fixed sampling again collapse onto a master curve ${ }^{4}$. Note that repeated-

[^0]
[^0]:    ${ }^{4}$ These curves converge pointwise on $(0,1]$ and uniformly on any interval $(\varepsilon, 1]$, but not uniformly on $(0,1]$. The limits $M \rightarrow \infty$ and $p \rightarrow 0$ are not exchangeable.

---

#### Page 74

sampling IBS estimators $\hat{\mathcal{L}}_{\text {IBS- } R}$ (see Section 4.4), obtained by averaging multiple IBS estimates, overlap with the curve for regular IBS for any $R$.

> **Image description.** A line graph compares the standard deviation of different sampling methods.
> 
> The graph has the x-axis labeled "p" and the y-axis labeled "Standard deviation x √samples". The x-axis ranges from 0 to 1. The y-axis ranges from 0 to 5.
> 
> Several lines are plotted on the graph, each representing a different sampling method. The lines are colored as follows:
> - Pale pink: "Fixed: 1 sample"
> - Light pink: "Fixed: 2 samples"
> - Light red: "Fixed: 5"
> - Red: "Fixed: 10"
> - Dark red: "Fixed: 20"
> - Darker red: "Fixed: 50"
> - Darkest red: "Fixed: 100"
> - Black: "master curve"
> - Blue: "IBS"
> 
> All the lines start at the right side of the graph, near x=1, and increase as x approaches 0. The lines representing "Fixed" sampling methods peak at different points before decreasing, with the peak increasing as the number of samples increases. The "master curve" and "IBS" lines do not peak but continue to increase as x approaches 0.

Figure S3: Standard deviation times square root of the expected number of samples drawn by IBS (blue) and fixed sampling (red), and the master curve (black) that fixed sampling converges to when $M \rightarrow \infty$.

All these curves increase and diverge as $p \rightarrow 0$, reflecting the fact that estimating log-likelihoods for small $p$ is hard. The standard deviation of fixed sampling is always lower than that of IBS, especially when $p \rightarrow 0$ (specifically when $p \ll \frac{1}{M}$ ). In other words, fixed sampling produces low-variance estimators exactly in the range in which its estimates are biased, as guaranteed by the Cramer-Ráo bound. However, in the large- $M$ limit, fixed sampling does saturate the information inequality, so its master curve lies below IBS. In other words, if one is able to draw so many samples that bias is no issue, then fixed sampling provides a slightly better trade-off between variance and computational time. However, in Section C.2, we discuss an improvement to IBS which decreases its variance by a factor 2-20, in which case IBS is clearly superior. Finally, a quantity of interest for the researcher may not be the variance of the estimator per se, but a measure of the error such as the RMSE, for which IBS is also consistently superior

---

#### Page 75

(see Section A.6).

# A. 5 A Bayesian derivation of the IBS estimator 

In Sections 3 and A. 2 we hinted at a Bayesian interpretation of the problem of estimating $\log p$. We show here that indeed we can see the IBS estimator as a Bayesian point estimate of $\log p$ with a specific choice of prior for $p$. For the rest of this section, we use $q$ to denote the likelihood of a trial (instead of $p$ ); that is $q$ is the parameter of the Bernoulli distribution and $\log q$ the quantity we are seeking to estimate. We changed notation to avoid confusion with expressions such as the prior probability of $q$, which is $p(q)$.

Let $K$ be the number of samples until a 'hit', as per the IBS sampling policy. Following Bayes' rule, we can write the posterior over $q$ given $K$ as

$$
\begin{aligned}
p(q \mid K) & =\frac{\operatorname{Pr}(K \mid q) p(q)}{\operatorname{Pr}(K)} \\
& =\frac{(1-q)^{K-1} q \operatorname{Beta}(q ; \alpha, \beta)}{\int_{0}^{1}(1-q)^{K-1} q \operatorname{Beta}(q ; \alpha, \beta) d q} \\
& =\frac{\Gamma(K+\alpha+\beta)}{\Gamma(\alpha+1) \Gamma(K+\beta-1)}(1-q)^{K+\beta-2} q^{\alpha}
\end{aligned}
$$

where we used the fact that $\operatorname{Pr}(K \mid q)$ follows a geometric distribution, and we assumed a Beta $(\alpha, \beta)$ prior over $q$.

In particular, let us compute the posterior mean of $\log q$ under the Haldane prior, $\operatorname{Beta}(0,0)$ (Haldane, 1932). Thanks to the 'law of the unconscious statistician', we can

---

#### Page 76

compute the posterior mean of $p(\log q \mid K)$ directly from Equation S5,

$$
\begin{aligned}
\mathbb{E}_{p(\log q \mid K)}[\log q] & =(K-1) \int_{0}^{1}(\log q)(1-q)^{K-2} d q \\
& =\int_{0}^{1}(\log q) \text { Beta }(q ; 1, K-1) d q \\
& =\psi(1)-\psi(K) \\
& =-\sum_{k=1}^{K-1} \frac{1}{k}
\end{aligned}
$$

where the first row follows from setting $\alpha=0$ and $\beta=0$; it can be shown that the third row is the expectation of $\log q$ for a Beta distribution, with $\psi(z)$ the digamma function (Abramowitz and Stegun, 1948); and the last equality follows from the relationship between the digamma function and harmonic numbers, that is $\psi(n)=-\gamma+\sum_{k=1}^{n-1} \frac{1}{k}$, where $\gamma$ is Euler-Mascheroni constant. We also used the notational convention that $\sum_{k=1}^{0} a_{k}=0$ for any $a_{k}$. Note that the last row is equal to the IBS estimator, $\hat{\mathcal{L}}_{\text {IBS }}(K)$, as defined in Equation 14 in the main text.

Crucially, Equation S5 shows that we can recover the IBS estimator as the posterior mean of $\log q$ given $K$, under the Haldane prior for $q$. This interpretation allows us to also define naturally the variance of our estimate for a given $K$, as the variance of the posterior over $\log q$,

$$
\operatorname{Var}_{p(\log q \mid K)}[\log q]=\psi_{1}(1)-\psi_{1}(K)
$$

where $\psi_{1}(z)$ is the trigamma function, the derivative of the digamma function; the equality follows from a known expression for the variance of $\log q$ under a Beta distribution for $q$.

---

#### Page 77

# A. 6 Estimator RMSE 

In the main text and in previous comparisons we have discussed the bias and the variance of estimators of the log-likelihood, which are important statistical properties, but one might wonder how bias and variance combine to yield an error metric of practical relevance such as the root mean squared error (RMSE). Crucially, this analysis depends on the number of trials $N$ (because bias and standard deviation scale differently with $N$ ) and on the distribution of values of the likelihood for different trials, $p_{i}$.

For illustrative purposes, we took as an example the psychometric model described in Section 5.2, and calculated the distribution of $p_{i}$ for typical datasets and parameters settings. We then calculated the RMSE in estimating the total log-likelihood of a number of randomly generated datasets (sampled from the empirical distribution of $p_{i}$ ) with different number of trials; for different numbers of samples used by the IBS and fixed-sampling estimators.

> **Image description.** This image consists of three line graphs arranged horizontally. Each graph plots the Root Mean Squared Error (RMSE) on the y-axis against the "Number of samples" on the x-axis.
> 
> Here's a breakdown of the common elements and the variations between the graphs:
> 
> *   **Common Elements:**
>     *   **Axes:** Each graph has an x-axis labeled "Number of samples" ranging from 0 to 100. The y-axis is labeled "RMSE."
>     *   **Curves:** Each graph displays two curves: a red curve labeled "Fixed" and a blue curve labeled "IBS." Both curves generally decrease as the number of samples increases, indicating a reduction in RMSE with more samples.
>     *   **Legend:** A legend is present in the first graph, labeling the red curve as "Fixed" and the blue curve as "IBS."
>     *   **General Shape:** Both the red and blue curves show a steep decline initially, followed by a more gradual decrease as the number of samples increases.
> 
> *   **Variations (Panel-Specific Details):**
>     *   **Panel 1 (Left):**
>         *   Title: "10 Trials"
>         *   Y-axis range: 0 to 2.5
>     *   **Panel 2 (Center):**
>         *   Title: "100 Trials"
>         *   Y-axis range: 0 to 25
>     *   **Panel 3 (Right):**
>         *   Title: "500 Trials"
>         *   Y-axis range: 0 to 120
> 
> In summary, the image compares the RMSE of two estimators, "Fixed" and "IBS," as a function of the number of samples, with each panel representing a different number of trials (10, 100, and 500). The RMSE decreases with the number of samples for both estimators, but the specific values and the relative performance of the estimators vary depending on the number of trials.

Figure S4: RMSE of the log-likelihood estimate as a function of number of samples, for the IBS and fixed-sampling estimators. Different panels display the RMSE curves for different number of trials.

Figure S4 shows that starting from even a handful of trials $(N=10)$, IBS is con-

---

#### Page 78

sistently better than fixed sampling at estimating the true value of the log-likelihood of a given parameter vector, and overwhelmingly so for a moderate number of trials $(N \geq 100)$.

# B Experimental details 

In this section, we report details for the three numerical experiments described in the main text and supplementary results.

## B. 1 Orientation discrimination

The parameters of the orientation discrimination model are the (inverse) slope, or sensory noise, represented as $\eta \equiv \log \sigma$, the bias $\mu$, and the lapse rate $\gamma$. The logarithmic representation for $\sigma$ is a natural choice for scale parameters (and more in general, for positive parameters that can span several orders of magnitude).

We define the lower bound (LB), upper bound (UB), plausible lower bound (PLB), and plausible upper bound (PUB) of the parameters as per Table S1. The upper and lower bounds are hard constraints, whereas the plausible bounds provide information to the algorithm to where the global optimum is likely to be, and are used by BADS, for example, to draw a set of initial points to start building the surrogate Gaussian process, and to set priors over the Gaussian process hyperparameters (Acerbi and Ma, 2017). Here we also use the plausible bounds to select ranges for the parameters used to generate simulated datasets, and to initialize the optimization, as described below.

To generate synthetic data sets, we select 120 'true' parameter settings for the ori-

---

#### Page 79

Table S1: Parameters and bounds of the orientation discrimination model.

| Parameter | Description | LB | UB | PLB | PUB |
| :--: | :--: | :--: | :--: | :--: | :--: |
| $\eta \equiv \log \sigma$ | Slope | $\log 0.1$ | $\log 10$ | $\log 0.1$ | $\log 5$ |
| $\mu$ | Bias $\left({ }^{\circ}\right)$ | -2 | 2 | -1 | 1 |
| $\gamma$ | Lapse | 0.01 | 1 | 0.01 | 0.2 |

entation discrimination task as follows. We set the baseline parameter $\boldsymbol{\theta}_{0}$ as $\eta=\log 2^{\circ}$, $\mu=0.1^{\circ}$, and $\gamma=0.1$. Then, for each parameter $\theta_{j} \in\{\eta, \mu, \gamma\}$, we linearly vary the value of $\theta_{j}$ in 40 increments from $\mathrm{PLB}_{j}$ to $\mathrm{PUB}_{j}$ as defined in Table S 1 (e.g., from $-1^{\circ}$ to $1^{\circ}$ for $\mu$ ), while keeping the other two parameters fixed to their baseline value. For each one of the 120 parameter settings $\boldsymbol{\theta}_{\text {true }}$ defined in this way, we randomly generated stimuli and responses for 100 datasets from the generative model, resulting in 12000 distinct data sets for which we know the true generating parameters.

We evaluated the log-likelihood with the following methods: fixed sampling with $M$ samples, with $M \in\{1,2,3,5,10,15,20,35,50,100\}$; IBS with $R$ repeats, with $R \in\{1,2,3,5,10,15,20,35,50\}$; and exact. To avoid wasting computations on particularly 'bad' parameter settings, for IBS we used the 'early stopping threshold' technique described in Section C.1, setting a lower bound on the log-likelihood of IBS equal to the log-likelihood of a chance model, that is $\mathcal{L}_{\text {lower }}=-N \log 2$. While this might seemingly provide an advantage to IBS with respect to Fixed sampling, note that it is simply a way to ameliorate a weakness of IBS (spending too much time on 'bad' parameters vectors, which are largely inconsequential for optimization), which Fixed does not suffer from. Even so, the stopping threshold was rarely reached ( $2 \%$ of evaluations).

---

#### Page 80

For each data set and method, we optimized the log-likelihood by running BADS 8 times with different starting points. We selected starting points as the points that lie on one-third or two-third of the distance between the plausible upper and lower bound for each parameter, that is all combinations of $\eta \in\{-0.998,0.305\}, \mu \in$ $\left\{-0.333^{\circ}, 0.333^{\circ}\right\}, \gamma \in\{0.073,0.137\}$. Each of these optimization runs returns a candidate for $\widehat{\boldsymbol{\theta}}_{\text {MLE }}$. For methods that return a noisy estimate of the log-likelihood, we then re-evaluate $\hat{\mathcal{L}}(\boldsymbol{\theta})$ for each of these 8 candidates with higher precision (for fixed sampling, we use $10 M$ samples; for IBS, we use $10 R$ repeats). Finally, we select the candidate with highest (estimated) log-likelihood.

When estimating parameters using IBS or fixed sampling, we enabled the 'uncertainty handling' option in BADS, informing it to incorporate measurement noise into its model of the objective function. Note that during the optimization, the algorithm iteratively infers a single common value for the observation noise $\sigma_{\text {obs }}$ associated with the function values in a neighborhood of the current point (Acerbi and Ma, 2017). A future extension of BADS may allow the user to explicitly provide the noise associated with each data point, which is easily computed for the IBS estimates (Equation 16 in the main text), affording the construction of a better surrogate model of the log-likelihood.

# Alternative fixed sampling estimator 

In the main text, we considered the fixed-sampling estimator defined by Equation 11. We performed an additional analysis to empirically validate that our results do not depend on the specific choice of estimator for fixed sampling (as expected given the theoretical arguments in Section 3).

---

#### Page 81

An alternative way of avoiding the divergence of fixed sampling is to correct samples that happen to be all zeros, for example with

$$
\hat{\mathcal{L}}_{\text {fixed-bound }}(\boldsymbol{x})=\log \left(\frac{\max \left\{m(\boldsymbol{x}), m_{\min }\right\}}{M}\right)
$$

for some $0<m_{\min }<1$, which sets a lower bound for the log-likelihood equal to $\log \left(m_{\min } / M\right)$. We then performed our analyses of the orientation discrimination task using the $\hat{\mathcal{L}}_{\text {fixed-bound }}$ estimator with $m_{\min }=\frac{1}{2}$. As shown in Figure S5, the results are remarkably similar to what we found using the fixed-sampling estimator $\hat{\mathcal{L}}_{\text {fixed }}$ defined by Equation 11. Finally, we also tried $\hat{\mathcal{L}}_{\text {fixed-bound }}$ with a small value $m_{\min }=10^{-3}$, which yielded even worse results (data not shown).

> **Image description.** This image contains six plots, arranged in a 2x3 grid, labeled A through F. The plots compare different methods ("exact", "ibs", and "fixed") for estimating parameters, likely related to a statistical model or simulation.
> 
> **Panel A:**
> *   This is a scatter plot.
> *   The x-axis is labeled "η".
> *   The y-axis is labeled "ή".
> *   Three data sets are plotted: "exact" (green plus signs), "ibs 2.22" (blue plus signs), and "fixed 10" (red plus signs).
> *   A black line, representing a linear relationship, is also present.
> 
> **Panel B:**
> *   This is a line graph.
> *   The x-axis is labeled "Number of samples". The scale ranges from 0 to 100.
> *   The y-axis is labeled "ή".
> *   Four data sets are plotted as lines: "exact" (green), "IBS" (blue), "fixed" (red), and "true" (black dashed line).
> 
> **Panel C:**
> *   This is a line graph.
> *   The x-axis is labeled "Number of samples". The scale ranges from 0 to 100.
> *   The y-axis is labeled "RMSE (η)".
> *   Three data sets are plotted as lines: "exact" (green), "fixed" (red), and "IBS" (blue).
> 
> **Panel D:**
> *   This is a scatter plot.
> *   The x-axis is labeled "γ".
> *   The y-axis is labeled "ŷ".
> *   Three data sets are plotted: "exact" (green plus signs), "ibs 6.49" (blue plus signs), and "fixed 20" (red plus signs).
> *   A black line, representing a linear relationship, is also present.
> 
> **Panel E:**
> *   This is a line graph.
> *   The x-axis is labeled "Number of samples". The scale ranges from 0 to 100.
> *   The y-axis is labeled "ŷ".
> *   Three data sets are plotted as lines with shaded regions around them: "exact" (green), "fixed" (red), and "IBS" (blue).
> *   A black dashed line is also present.
> 
> **Panel F:**
> *   This is a line graph.
> *   The x-axis is labeled "Number of samples". The scale ranges from 0 to 100.
> *   The y-axis is labeled "RMSE (γ)".
> *   Three data sets are plotted as lines with shaded regions around them: "exact" (green), "fixed" (red), and "IBS" (blue).
> 
> In summary, the figure compares the performance of different estimation methods ("exact", "ibs", "fixed") in terms of their estimates of parameters (η and γ) and their Root Mean Squared Error (RMSE) as a function of the number of samples.

Figure S5: Same as Figure 5 in the main text, but for the alternative fixed-sampling estimator defined by Equation S8. The results are qualitatively identical.

---

#### Page 82

# Complete parameter recovery results 

For completeness, we report in Figure S6 the parameter recovery results for fixed sampling, inverse binomial sampling and 'exact' analytical methods for the orientation discrimination task, for all tested number of samples $M$ and IBS repeats $R$. All estimates were obtained via maximum-likelihood estimation using the Bayesian Adaptive Direct Search (Acerbi and Ma, 2017), as described previously in this section.

## B. 2 Change localization

First, we derive the trial likelihood of the change localization model. Assuming that the change happens at location $c \in\{1, \ldots, 6\}$, by symmetry we can write

$$
\operatorname{Pr}(\text { respond } i \mid c \text { changed })= \begin{cases}P_{\text {correct }}\left(\Delta_{s}^{(c)} ; \boldsymbol{\theta}\right) & \text { if } i=c \\ \frac{1}{5}\left(1-P_{\text {correct }}\left(\Delta_{s}^{(c)} ; \boldsymbol{\theta}\right)\right) & \text { otherwise }\end{cases}
$$

where $\Delta_{s}^{(c)}=\left|d_{\text {circ }}\left(s_{c}^{(1)}, s_{c}^{(2)}\right)\right|$ is the absolute circular distance between the true orientations of patch $c$ in the first and second display. We can derive an expression for $P_{\text {correct }}\left(\Delta_{s}^{(c)} ; \boldsymbol{\theta}\right)$ by marginalizing over the circular distance between the respective measurements,
$P_{\text {correct }}\left(\Delta_{s}^{(c)} ; \boldsymbol{\theta}\right)=\frac{\gamma}{6}+(1-\gamma) \int_{0}^{2 \pi} \operatorname{Pr}\left(\Delta_{x}^{(c)} \mid \Delta_{s}^{(c)}\right) \operatorname{Pr}\left(\forall i \neq c: \Delta_{x}^{(i)} \leq \Delta_{x}^{(c)} \mid \Delta_{s}^{(i)}=0\right) \mathrm{d} \Delta_{x}^{(c)}$,
where we have defined $\Delta_{x}^{(i)}=\left|d_{\text {circ }}\left(x_{i}^{(1)}, x_{i}^{(2)}\right)\right|$ and we suppressed the dependence on $\kappa$ to simplify the notation. The first term in this equation is the probability density function (pdf) of the circular distance between two von Mises random variables whose centers are $\Delta_{s}^{(j)}$ apart. The second term simplifies, since $\Delta_{x}^{(i)}$ for all $i \neq j$ are all

---

#### Page 83

> **Image description.** This image contains a set of nine scatter plots arranged in a 3x3 grid, labeled A through I. Each plot visualizes the relationship between a parameter and its estimate, with different methods used for estimation across the columns. The rows represent different parameters: eta (η), mu (μ), and gamma (γ).
> 
> *   **General Layout:** The plots are organized as follows:
>     *   Top row (A, B, C): Plots for parameter η (eta).
>     *   Middle row (D, E, F): Plots for parameter μ (mu).
>     *   Bottom row (G, H, I): Plots for parameter γ (gamma).
>     *   Left column (A, D, G): Plots using "fixed sampling" with varying numbers of samples.
>     *   Middle column (B, E, H): Plots using "IBS" (Iterative Bayesian Sequential sampling) with varying numbers of repeats.
>     *   Right column (C, F, I): Plots using the "exact" log-likelihood function.
> 
> *   **Plot Details:**
>     *   Each plot has a horizontal axis representing the true parameter value (η, μ, or γ) and a vertical axis representing the estimated parameter value (denoted with a hat: ῆ, μ̂, or γ̂).
>     *   A black diagonal line (y=x) is present in each plot, representing perfect estimation (estimate equals true value).
>     *   Different colored lines represent different numbers of samples (left and middle columns) or the "exact" method (right column).
>     *   Colors range from light to dark, with lighter shades indicating fewer samples/repeats and darker shades indicating more.
>     *   The legends in plots A, B, E, and H indicate the number of samples/repeats associated with each colored line.
>     *   Plot A shows several lines ranging from light pink to dark red, representing 1, 2, 3, 5, 10, 15, 20, 35, 50, and 100 samples.
>     *   Plot B shows several lines ranging from light blue to dark blue, representing 2.22, 4.36, 6.48, 10.75, 21.38, 32.00, 42.59, 74.40, and 106.23 samples.
>     *   Plot C shows a single green line representing the "exact" method.
>     *   Plot D shows several lines ranging from light pink to dark red.
>     *   Plot E shows several lines ranging from light blue to dark blue, representing 2.22, 4.40, 6.56, 10.89, 21.67, 32.47, 43.22, 75.44, and 107.58 samples.
>     *   Plot F shows a single green line representing the "exact" method.
>     *   Plot G shows several lines ranging from light pink to dark red, representing 1, 2, 3, 5, 10, 15, 20, 35, 50, and 100 samples.
>     *   Plot H shows several lines ranging from light blue to dark blue, representing 2.20, 4.35, 6.49, 10.78, 21.51, 32.24, 42.95, 75.03, and 107.11 samples.
>     *   Plot I shows a single green line representing the "exact" method.
> 
> *   **Axes Labels:**
>     *   The horizontal axes are labeled with the parameter symbols: η, μ, and γ.
>     *   The vertical axes are labeled with the estimated parameter symbols: ῆ, μ̂, and γ̂.

Figure S6: Full parameter recovery results for the orientation discrimination model. A.

Mean estimates recovered by fixed sampling with different number of samples. Error bars are omitted to avoid visual clutter. B. Mean estimates recovered by IBS with different numbers of repeats. The legend reports the average number of samples per trial that IBS uses to obtain these estimates. C. Mean estimate recovered using the 'exact' log-likelihood function (Equation 20). D-F Same, for the bias parameter $\mu$. G-I Same, for the lapse rate $\gamma$. Overall, fixed sampling produces highly biased estimates of $\eta$ and $\gamma$, while IBS is much more accurate. The bias parameter $\mu$ can be accurately estimated by either method regardless of the number of samples or repeats.

---

#### Page 84

independent and identically distributed. Therefore, we can rewrite this equation as

$$
P_{\text {correct }}\left(\Delta_{s}^{(c)} ; \boldsymbol{\theta}\right)=\frac{\gamma}{6}+(1-\gamma) \int_{0}^{2 \pi} \operatorname{Pr}\left(\Delta_{x}^{(c)} \mid \Delta_{s}^{(c)}\right) \operatorname{Pr}\left(\Delta \leq \Delta_{x}^{(c)}\right)^{5} \mathrm{~d} \Delta_{x}^{(c)}
$$

where $\Delta$ is an auxiliary variable generated by taking the absolute circular difference between two von Mises random variables that are centered at 0 with concentration parameter $\kappa$. The second term of the integrand, therefore, is the fifth power of the cumulative distribution function (cdf) of $\Delta$. We can compute the distribution of the circular distance between two von Mises random variables analytically, but the cdf is non-analytic. Moreover, the integral in Equation S11 is analytically intractable as well. We can, however, evaluate it numerically via trapezoidal integration (see Figure 6B).

We now describe the settings used for maximum-likelihood estimation. The model parameters are the sensory noise, represented as $\eta \equiv \log \sigma$ (with $\sigma=\frac{1}{\sqrt{\kappa}}$ ), and the lapse rate $\gamma$, with bounds defined in Table S2. We use the same procedure and settings for BADS as for the orientation discrimination task (see Section B.1). For IBS, we use an early-stopping threshold of $\mathcal{L}_{\text {lower }}=-N \log 6$, and we use repeats $R \in$ $\{1,2,3,5,10,15,20\}$ (since due to the larger response space IBS uses more samples per run). We run BADS 4 times, with starting values of $\eta \in\{-1.535,-0.767\}$ and $\gamma \in\{0.173,0.337\}$. For data generation, we select 40 parameter vectors with $\eta=$ $\log 0.3$ and $\gamma$ linearly spaced from 0.01 to 0.5 and 40 data sets with $\gamma=0.03$ and $\eta$ between $\log 0.1$ and $\log 1$. Again, we generate 100 data sets for each such parameter combination.

---

#### Page 85

Table S2: Parameters and bounds of the change localization model.

| Parameter | Description | LB | UB | PLB | PUB |
| :--: | :-- | :--: | :--: | :--: | :--: |
| $\eta \equiv \log \sigma$ | Sensory noise | $\log 0.05$ | $\log 2$ | $\log 0.1$ | $\log 1$ |
| $\gamma$ | Lapse | 0.01 | 1 | 0.01 | 0.5 |

# Complete parameter recovery results 

We report in Figure S7 the parameter recovery results for fixed sampling, inverse binomial sampling and 'exact' methods for the change localization task, for all tested number of samples $M$ and IBS repeats $R$. For this task, the exact method relies on numerical integration.

## B. 3 Four-in-a-row game

The four-in-a-row game model parameters are the value noise $\eta \equiv \log \sigma$, the pruning threshold $\xi$, and the feature dropping rate $\delta$, with bounds defined in Table S3. We use the same procedure and settings for BADS as for the orientation discrimination task (see Section B.1), unless noted otherwise. For IBS, we use an early-stopping threshold of $\mathcal{L}_{\text {lower }}=-N \log 20$, and due to computational cost we use only $R \in\{1,2,3\}$. For fixed sampling we consider $M \in\{1,2,3,5,10,15,20,35,50,100\}$. We have no expression for the likelihood of the four-in-a-row game model, not even in numerical form, so there is no 'exact' method.

We run BADS 8 times, with starting values of $\eta \in\{-0.707,0.196\}, \xi \in\{4,7\}$ and $\delta \in\{0.167,0.333\}$. For data generation, we set as baseline parameter vector $\eta=\log 1$, $\xi=5$ and $\delta=0.2$ and for each parameter we select 40 parameter vectors linearly

---

#### Page 86

> **Image description.** This image contains six scatter plots arranged in a 2x3 grid, labeled A through F. Each plot displays the relationship between two variables, with a diagonal black line serving as a reference.
> 
> *   **Plots A, B, and C:** These plots show the relationship between η (horizontal axis) and ῆ (vertical axis).
>     *   Plot A: Contains multiple lines in shades of red, from light to dark, representing different sample sizes (1, 2, 3, 5, 10, 15, 20, 35, 50, and 100 samples). The lines generally curve upwards, deviating from the diagonal black line, especially at higher values of η.
>     *   Plot B: Contains multiple lines in shades of blue, representing different sample sizes (6.19, 12.25, 18.36, 30.56, 61.12, and 122.52 samples). These lines are closer to the diagonal black line compared to plot A.
>     *   Plot C: Contains a single green line labeled "exact". This line closely follows the diagonal black line, indicating a more accurate relationship between η and ῆ.
> 
> *   **Plots D, E, and F:** These plots show the relationship between γ (horizontal axis) and ŷ (vertical axis).
>     *   Plot D: Contains multiple lines in shades of red, similar to plot A, representing different sample sizes. The lines curve upwards, deviating from the diagonal black line.
>     *   Plot E: Contains multiple lines in shades of blue, similar to plot B, representing different sample sizes (6.83, 13.42, 19.94, 32.85, 65.06, and 129.26 samples). These lines are closer to the diagonal black line compared to plot D.
>     *   Plot F: Contains a single green line labeled "exact". This line closely follows the diagonal black line, indicating a more accurate relationship between γ and ŷ.
> 
> The axes in all plots are labeled with numerical values. The x-axes (η and γ) range from approximately 1.5 to 4 (Plots A, B, C) and 0 to 0.5 (Plots D, E, F). The y-axes (ῆ and ŷ) range from approximately 1 to 5 (Plots A, B, C) and 0 to 0.5 (Plots D, E, F).

Figure S7: Same as Figure S6, for the change localization model. Fixed sampling is substantially biased for both the measurement noise $\eta$ and the lapse rate $\gamma$, whereas IBS is accurate for $\eta$ and biased for $\gamma$, but still much less biased than fixed sampling.

---

#### Page 87

Table S3: Parameters and bounds of the four-in-a-row game model.

| Parameter | Description | LB | UB | PLB | PUB |
| :--: | :--: | :--: | :--: | :--: | :--: |
| $\eta \equiv \log \sigma$ | Value noise | $\log 0.01$ | $\log 5$ | $\log 0.2$ | $\log 3$ |
| $\xi$ | Pruning threshold | 0.01 | 10 | 1 | 10 |
| $\delta$ | Feature dropping rate | 0 | 1 | 0 | 0.5 |

spaced in the plausible range for that parameter (as per Table S3), while keeping the other two parameters at their baseline value. Again, we generate 100 data sets for each such parameter combination.

We fixed the other parameters of the model to typical values found in the previous study (van Opheusden et al., 2016), namely $w_{\text {center }}=0.60913, w_{\text {connected } 2 \text {-in-a-row }}=$ $0.90444, w_{\text {unconnected } 2 \text {-in-a-row }}=0.45076, w_{3 \text {-in-a-row }}=3.4272, w_{3 \text {-in-a-row }}=6.1728, C_{\text {act }}=$ $0.92498, \gamma_{\text {tree }}=0.02, \lambda=0.05$. The $w_{i}$ are the weights of features $f_{i}$ in the value function, briefly $f_{\text {center }}$ values pieces near the center of the board, the other features count the number of times certain patterns occur on the board (see van Opheusden et al., 2016 for the specific patterns). $C_{\text {act }}$ is a parameter which scales the value of features belonging to the active or passive player. The parameter $\gamma_{\text {tree }}$ is inversely proportional to the size of the tree built by the algorithm, and $\lambda$ is the lapse rate, that is the probability of a uniformly random move among the available squares (note that for the other models we denoted lapse rate as $\gamma$; here we use the variable naming from van Opheusden et al., 2016). See van Opheusden et al. (2016) for more details about the model and its parameters.

---

#### Page 88

# Complete parameter recovery results 

We report in Figure S8 the parameter recovery results for fixed sampling and inverse binomial sampling for the 4-in-a-row task, for all tested number of samples $M$ and IBS repeats $R$. For this task, there is no 'exact' method to evaluate the log-likelihood.

## C Improvements of IBS and further applications

## C. 1 Early stopping threshold

One downside of inverse binomial sampling is that the computational time it uses to estimate the log-likelihood is of the order of $\frac{1}{p}$, which is equal to $\exp (-\log p)=\exp (-\mathcal{L})$. In other words, IBS spends exponentially more time on estimating log-likelihoods of poorly-fitting models or bad parameters. This implies that an optimization algorithm that uses IBS allocates more computational resources to estimating the objective function $\mathcal{L}(\boldsymbol{\theta})$ for parameter vectors $\boldsymbol{\theta}$ where the objective is low. However, the value of the objective at such poor parameter vectors are unlikely to affect its estimate of the location or value of the maximum, so the optimizer (BADS in our case) is wasting time. It may be possible to develop optimization algorithms that take into account the exponentially large cost of probing points where the objective function is low, but we can circumvent the problem by amending IBS with a criterion that stops sampling when it realizes that $\dot{\mathcal{L}}(\boldsymbol{\theta})$ will be low.

In Section 4.1, we introduced a basic implementation of IBS for estimating the loglikelihood of multiple trials, by sequentially computing the log-likelihood of each trial. However, another way to implement multi-trial IBS (a 'parallel' implementation) is to

---

#### Page 89

> **Image description.** This image contains six scatter plots arranged in a 2x3 grid, labeled A through F. Each plot shows the relationship between an estimated parameter and the true parameter value.
> 
> *   **General Layout:** The plots are organized with A, B in the first row, C, D in the second row, and E, F in the third row. Each plot has axes labeled with parameter symbols. A black diagonal line is present in each plot, representing the ideal estimation (where estimated value equals true value).
> 
> *   **Plot A:** Shows the estimated parameter "$\hat{\eta}$" on the y-axis versus the true parameter "$\eta$" on the x-axis. Multiple lines are plotted, each representing a different number of samples. The lines are colored in shades of red, with darker red indicating a larger number of samples. A legend in the upper right corner indicates the number of samples for each line: 1, 2, 3, 5, 10, 15, 20, 35, 50, and 100 samples.
> 
> *   **Plot B:** Shows the estimated parameter "$\hat{\eta}$" on the y-axis versus the true parameter "$\eta$" on the x-axis. Three lines are plotted, colored in shades of blue. A legend in the upper left corner indicates the number of samples for each line: 25.70, 50.76, and 75.84 samples.
> 
> *   **Plot C:** Shows the estimated parameter "$\hat{\xi}$" on the y-axis versus the true parameter "$\xi$" on the x-axis. Multiple lines are plotted, each representing a different number of samples. The lines are colored in shades of red, with darker red indicating a larger number of samples.
> 
> *   **Plot D:** Shows the estimated parameter "$\hat{\xi}$" on the y-axis versus the true parameter "$\xi$" on the x-axis. Three lines are plotted, colored in shades of blue. A legend in the upper left corner indicates the number of samples for each line: 27.50, 53.80, and 80.20 samples.
> 
> *   **Plot E:** Shows the estimated parameter "$\hat{\delta}$" on the y-axis versus the true parameter "$\delta$" on the x-axis. Multiple lines are plotted, each representing a different number of samples. The lines are colored in shades of red, with darker red indicating a larger number of samples.
> 
> *   **Plot F:** Shows the estimated parameter "$\hat{\delta}$" on the y-axis versus the true parameter "$\delta$" on the x-axis. Three lines are plotted, colored in shades of blue. A legend in the upper left corner indicates the number of samples for each line: 27.92, 54.59, and 81.25 samples.
> 
> The plots on the left (A, C, E) use a red color scheme and show more sample number variations, while the plots on the right (B, D, F) use a blue color scheme and show fewer sample number variations. The x and y axes are scaled differently in each plot to accommodate the range of values for each parameter.

Figure S8: Same as Figure S6, for the four-in-a-row task. For this model, we do not have an exact log-likelihood formula or numerical approximation, so we only show fixed sampling and IBS. Overall, fixed sampling has substantial biases in its estimation of $\eta$ and $\delta$ and a smaller bias in estimating $\xi$. IBS has almost no bias for $\eta$ and only a small bias for $\xi$ and $\delta$.

---

#### Page 90

draw one sample from the simulator model for each trial, then set $K_{i}=1$ for each trial where the sample matches the participant's response. For all other trials, draw a second sample from the model, and if that matches the response, set $K_{i}=2$. Finally, repeat this process until no more trials remain. We illustrate this sampling scheme graphically in Figure S9.

After each iteration, we then compute

$$
\hat{\mathcal{L}}_{K}=-\sum_{i \in \mathcal{I}_{\text {match }}} \sum_{k=1}^{K_{i}-1} \frac{1}{k}-N_{\text {remaining }} \sum_{k=1}^{K-1} \frac{1}{k}
$$

where $K$ is the iteration number, $\mathcal{I}_{\text {match }}$ is the set of trials where we found a matching sample and $N_{\text {remaining }}$ is the number of remaining trials. This value $\hat{\mathcal{L}}_{K}$ is decreasing and by construction converges to $\sum_{i=1}^{N} \hat{\mathcal{L}}_{i, \text { IBS }}$ as $K \rightarrow \infty$. Therefore, whenever $\hat{\mathcal{L}}_{K}$ falls below a lower bound $\mathcal{L}_{\text {lower }}$, we are guaranteed that $\hat{\mathcal{L}}_{\text {IBS }}$ will be below that bound too. When it does, we stop sampling and return $\mathcal{L}_{\text {lower }}$ as estimate of $\mathcal{L}(\boldsymbol{\theta})$. This does introduce bias into the estimate, but since we bound the total log-likelihood, the bias will be exponentially small in $N$ as long as the true value $\mathcal{L}(\boldsymbol{\theta})$ is adequately larger than $\mathcal{L}_{\text {lower }}$.

In practice, we recommend using as lower bound the log-probability of the data under a 'chance' model, which assigns uniform probability to each possible response on each trial, and should be a poor model of the data. In the orientation discrimination and change localization examples from Sections 5.2 and 5.3, the log-likelihood of a chance model is $-N \log 2$ and $-N \log 6$, respectively. For the 4 -in-a-row game presented in Section 5.4 the log-likelihood of chance depends on the number of pieces on each board position; we chose an average value such that $\mathcal{L}_{\text {lower }}=-N \log 20$. This new sampling scheme has an additional advantage: since on each iteration we independently sample

---

#### Page 91

> **Image description.** This is a diagram illustrating a method for implementing IBS (Iterative Bayesian Sequential sampling) with multiple trials.
> 
> The diagram consists of a grid-like structure with six columns labeled "Trial" 1 through 6 along the bottom horizontal axis. The vertical axis is implicitly labeled with 'k' values from k=1 to k=6 on the left side. Each column represents a trial, and each row represents a successive sample from the model.
> 
> Each cell in the grid contains either a red "X" (representing a miss) or a green checkmark (representing a hit). Above each column is a "K" value representing the number of samples until a hit is achieved for that trial.
> 
> Specifically:
> *   Trial 1 has K=3, with a red X in the first two rows and a green checkmark in the third row.
> *   Trial 2 has K=1, with a green checkmark in the first row.
> *   Trial 3 has K=2, with a red X in the first row and a green checkmark in the second row.
> *   Trial 4 has K=1, with a green checkmark in the first row.
> *   Trial 5 has K=6, with red X's in the first five rows and a green checkmark in the sixth row.
> *   Trial 6 has K=4, with red X's in the first three rows and a green checkmark in the fourth row.

Figure S9: Graphical illustration of the two methods to implement IBS with multiple trials, in this case $N=6$. In this figure, each column represents a trial, each box above the trial number represents a successive sample from the model from that trial, with red crosses for samples that do not match the participant's response ('misses') and green checkmarks for ones that do ('hits'). Above each column, we indicate $K$, the number of samples until a hit. For trials 2 and $4, K=1$ so $\hat{\mathcal{L}}_{\text {IBS }}=0$. The most obvious implementation of multi-trial IBS is 'columns-first', to sample model responses for each trial until a hit, and only then move to the next trial. However, a more convenient sampling method is 'rows-first', and sample one response for each trial with $k=1$, then one response for each trial with $k=2$, excluding trials 2 and 4 since the first sample was a hit, and continue increasing $k$ until all trials reach a hit. This method allows for early stopping and a parallel processing.

---

#### Page 92

from the generative model on multiple trials, we can potentially run these computations in parallel.

# C. 2 Reducing variance by trial-dependent repeated sampling 

As we saw in Section 4.4, a simple method to improve the estimate of IBS is to run the estimator multiple times and average the results. Repeated sampling will preserve the zero bias but reduce variance inversely proportional to the number of repeats $R$.

We can further improve the estimator by varying the number of repeats $R_{i}$ between trials, for $1 \leq i \leq N$, and define

$$
\hat{\mathcal{L}}_{\text {IBS } \cdot \boldsymbol{R}}=\sum_{i=1}^{N} \frac{1}{R_{i}} \sum_{r=1}^{R_{i}} \hat{\mathcal{L}}_{i}^{(r)}
$$

where $\boldsymbol{R}$ is a vector of positive integers with elements $R_{i}$, and $\hat{\mathcal{L}}_{i}^{(r)}$ denotes the outcome of the $r$-th run of IBS on trial $i$. This estimator is unbiased regardless of the choice of $\boldsymbol{R}$ (as long as $R_{i}>0$ for all trials), and we can analytically compute both its variance and expected number of samples (see Equation S14).

We can then ask what is the best allocation of repeats $R_{i}$ that minimizes the variance of the estimator in Equation S13 such that the expected total number of samples does not exceed a fixed budget $S$. This defines the following constrained optimization problem,

$$
\boldsymbol{R}^{*}=\arg \min _{R_{i}, R_{2} \ldots, R_{N}}\left\{\left.\frac{1}{N} \sum_{i=1}^{N} \frac{\operatorname{Li}_{2}\left(1-p_{i}\right)}{R_{i}} \right\rvert\, \sum_{i=1}^{N} \frac{R_{i}}{p_{i}} \leq S\right\}
$$

where we used Equation 15 for the variance of the IBS estimator.

Assuming that the $R_{i}$ take continuous values, we can solve the optimization problem in Equation S14 exactly using a Lagrange multiplier, and find the following closed-form

---

#### Page 93

expression for the optimal number of repeats per trial,

$$
R_{i}^{*}=S\left(\sum_{j=1}^{N} \sqrt{\frac{\overline{\mathrm{Li}_{2}\left(1-p_{j}\right)}}{p_{j}}}\right)^{-1} \sqrt{p_{i} \overline{\mathrm{Li}_{2}\left(1-p_{i}\right)}}
$$

According to Equation S15, the optimal choice of repeats entails dividing the budget $S$ across trials, where trial $i$ is allocated repeats proportional to $\sqrt{p_{i} \overline{\mathrm{Li}_{2}\left(1-p_{i}\right)}}$. We plot this function in Figure S10 and see that, to minimize variance, we should allocate resources primarily to trials where $p_{i}$ is close to $\frac{1}{2}$ and avoid trials where $p_{i} \approx 1$ (since the variance of IBS is already small for those trials) or where $p_{i} \approx 0$ (since those utilize a larger share of the budget).

We can also calculate exactly the fractional increase in precision (inverse variance) when using the optimal choice of repeats vector $\boldsymbol{R}^{*}$, compared to a constant $R$ which divides the budget equally across trials,

$$
\frac{\operatorname{Var}\left[\hat{\mathcal{L}}_{\text {IBS-R }}\right]}{\operatorname{Var}\left[\hat{\mathcal{L}}_{\text {IBS-R }}\right]}=\left(\sum_{i=1}^{N} \sqrt{\frac{\overline{\mathrm{Li}_{2}\left(1-p_{i}\right)}}{p_{i}}}\right)^{2} \times\left(\sum_{i=1}^{N} \mathrm{Li}_{2}\left(1-p_{i}\right)\right)^{-1} \times\left(\sum_{i=1}^{N} \frac{1}{p_{i}}\right)^{-1}
$$

This equation implies that the gain in precision from this method depends on the distribution of $p_{i}$ across trials. If $p_{i} \sim$ Uniform[0,1] and $N=500$, the median precision gain is 1.584 and the inter-quartile range (IQR) is 1.375 to 2.090 . Note that the gain is always greater than 1 , unless $p_{i}$ is constant across trials.

# Practical implementation of trial-dependent repeated sampling 

In practice, Equation S15 cannot be applied directly, as we have treated $R_{i}$ as continuous variables, but the number of times to repeat IBS on a given trial has to be an integer. Additionally, the method is only unbiased if $R_{i}$ is at least 1 for each trial $i$. Therefore,

---

#### Page 94

> **Image description.** The image is a line graph showing the relationship between two variables.
> 
> *   **Axes:** The graph has two axes. The horizontal axis is labeled "p" and ranges from 0 to 1 in increments of 0.2. The vertical axis is labeled "$\sqrt{p \mathrm{Li}_{2}(1-p)}$" and ranges from 0 to 0.6 in increments of 0.2.
> *   **Data:** A single blue line is plotted on the graph. It starts at (0,0), rises to a peak around (0.5, 0.55), and then descends back to (1,0). The line is smooth and symmetrical.
> *   **Overall Shape:** The line forms a curve that resembles an inverted parabola or a bell curve that has been cut off at the x-axis.

Figure S10: Graph of $\sqrt{p \mathrm{Li}_{2}(1-p)}$, which is proportional to the optimal number of repeats for a trial with likelihood $p$ (see equation S15). We observe that the optimal allocation of computational resources entails repeated sampling for trials with $p \approx \frac{1}{2}$ and to avoid $p \approx 0$ or $p \approx 1$.

we can convert $R_{i}^{*}$ to integers by rounding up to the nearest integer. This method will make our solution approximate, and reduce the gain in precision, but it is still better than uniform repeats for uniformly distributed $p_{i}$ (median: 1.567, IQR: 1.374-2.002).

The derivation above has another, more fundamental problem. Computing $R_{i}^{*}$ requires knowledge of $p_{i}$ on each trial, which we do not have. While we could try and learn the allocation of $\boldsymbol{R}^{*}$ as a function of $\boldsymbol{\theta}$ in some adaptive way, in practice we recommend the following simple scheme:

1. Choose a default parameter vector $\boldsymbol{\theta}_{0}$, and run IBS with a large number of repeats (e.g, $R=100$ ) to estimate the (log)-likelihood of the model on each trial.
2. Compute the optimal repeats $R_{i}^{*}$ given the estimated likelihoods $\hat{p}_{i}$ and a total budget of expected samples $S$ per likelihood evaluation, and round up.
3. Run IBS with those fixed repeats per trial on each iteration of the optimization algorithm.

---

#### Page 95

This approach implicitly assumes that the log-likelihood will be correlated across trials between the generative model with parameter vector $\boldsymbol{\theta}_{0}$ and any other vector $\boldsymbol{\theta}$ probed by the optimization algorithm. This is usually the case, since low-probability trials are often those where something unexpected occurred (e.g., the participant of a behavioral experiment lapsed or otherwise made an error). In our experience, this scheme considerably reduces the variance of IBS for a given computational time budget.

# C. 3 Bayesian inference with IBS 

While the main text focused on maximum-likelihood estimation, the unbiased loglikelihood estimates provided by IBS can also be used to perform Bayesian inference of posterior distributions over model parameters. We describe here a few possible approaches to approximate Bayesian inference with IBS.

## Markov Chain Monte Carlo

Markov Chain Monte Carlo (MCMC; see e.g. Brooks et al., 2011) is a powerful class of algorithms that allows one to sequentially sample from a target probability density which is known up to a normalization constant (e.g., the joint distribution). A popular form of MCMC is known as Metropolis-Hastings (MH; Hastings, 1970), which explores the target distribution by drawing a sample from a 'proposal distribution' centered on the last sample (e.g., a multi-variate Gaussian). MH 'accepts' or 'rejects' the new sample with an acceptance probability that depends on the value of the target density at the proposed and at the last point. In case of acceptance, the new point is added the sequence of samples; otherwise, the last sample is repeated in the sequence. Under

---

#### Page 96

some conditions, the MH algorithm produces a (correlated) sequence of samples that are equivalent to draws from the target density. Crucially, and somewhat surprisingly, the MH algorithm is still valid (that is, produces a valid sequence) if one performs the comparison with a noisy but unbiased estimate of the target density as opposed to using the exact density (Andrieu et al., 2009).

One problem here is that IBS provides an unbiased estimate of the log-likelihood (and thus of the log target density); not of the likelihood. However, since the IBS estimates of the log-likelihood are nearly-exactly normally distributed (see Section 4.3), the distribution of the likelihood is log-normal. Thus, we can apply what is known as a 'convexity correction' and compute a (nearly) unbiased estimate of the likelihood $\hat{\ell}(\boldsymbol{\theta})$ by calculating the expected value of a log-normal variable, that is

$$
\hat{\ell}(\boldsymbol{\theta})=\exp \left(\hat{\mathcal{L}}(\boldsymbol{\theta})+\frac{1}{2} \operatorname{Var}[\hat{\mathcal{L}}(\boldsymbol{\theta})]\right)
$$

Equation S17 can be easily evaluated with IBS, using the expression for the variance of the IBS estimator (Equation 16).

# Variational inference 

An alternative class of approximate inference methods is based on variational inference (VI; Jordan et al., 1999). The goal of VI is to approximate the intractable posterior distribution with a simpler distribution $q(\boldsymbol{\theta})$ belonging to a chosen parametric family. A common choice is a multivariate normal with diagonal covariance (known as mean field approximation); but other choices are possible too. VI selects the 'best' approximation $q(\cdot)$ that minimizes the Kullback-Leibler divergence with the true posterior, or

---

#### Page 97

equivalently maximizes the following variational objective,

$$
\mathcal{E}[q]=\mathbb{E}_{\boldsymbol{\theta} \sim q(\cdot)}[\mathcal{L}(\boldsymbol{\theta})]+\mathcal{H}[q]
$$

where $\mathcal{H}[q]$ is the entropy of $q(\cdot)$, which we assume can be computed analytically or numerically. Crucially, we can obtain an unbiased estimate of the first term in Equation S18 (the expected log joint) with IBS, as we have seen in Section 6.1. The optimization of the variational objective can then be performed directly with derivative-free optimization methods (such as BADS), or via a technique that produces unbiased estimates of the gradient combined with variance-reduction tricks, called black-box variational inference (Ranganath et al., 2014).

# Gaussian process surrogate methods 

One issue with the approximate inference methods described above is that they require a large (possibly, very large) number of likelihood evaluations to converge. Thus, these approaches are unfeasible if the generative model is somewhat computationally expensive, as it is often the case. An alternative family of methods designed to deal with expensive likelihoods builds a Gaussian process approximation (a surrogate) of the log joint distribution, and uses it to actively acquire further points in a smart way, similarly to the approach of Bayesian optimization (Kandasamy et al., 2015; Acerbi, 2018; Järvenpää et al., 2019). However, unlike Bayesian optimization, the goal here is not to optimize the target function, but instead to build an accurate approximation of the posterior distribution, with as few likelihood evaluations as possible.

IBS is particularly suited to be used in combination with Gaussian process surrogate methods as it provides both an unbiased estimate of the log-likelihood, and a

---

#### Page 98

calibrated estimate of the uncertainty in each measurement, which can be used to inform the Gaussian process observation model. The development of Gaussian process surrogate methods is an active and very promising area of research. A recent example is Variational Bayesian Monte Carlo (VBMC; Acerbi, 2018, 2019), a technique that naturally combines Gaussian process surrogate modeling with variational inference thanks to Bayesian quadrature (O'Hagan, 1991). Conveniently, VBMC returns both an approximate posterior distribution and an estimate of the model evidence, which can be used for model comparison. Acerbi (2020) showed that VBMC, combined with IBS and modified to deal with noisy log-likelihood evaluations, performs very well on a variety of models from computational and cognitive neuroscience.