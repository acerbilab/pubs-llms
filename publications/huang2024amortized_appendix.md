# Amortized Bayesian Experimental Design for Decision-Making - Appendix

---

#### Page 15

# Appendix

The appendix is organized as follows:

- In Appendix A, we provide a brief introduction to conditional neural processes (CNPs) and Transformer neural processes (TNPs).
- In Appendix B, we describe the details of our model architecture and the training setups.
- In Appendix C, we present the full algorithm for training our TNDP architecture.
- In Appendix D, we compare the inference time with other methods and show the overall training time of TNDP.
- In Appendix E, we describe the details of our toy example.
- In Appendix F, we describe the details of the decision-aware active learning example.
- In Appendix G, we describe the details of the top- $k$ hyperparameter optimization task, along with additional results on the retrosynthesis planning task.

## A Conditional neural processes

CNPs (Garnelo et al., 2018) are designed to model complex stochastic processes through a flexible architecture that utilizes a context set and a target set. The context set consists of observed data points that the model uses to form its understanding, while the target set includes the points to be predicted. The traditional CNP architecture includes an encoder and a decoder. The encoder is a DeepSet architecture to ensure permutation invariance, it transforms each context point individually and then aggregates these transformations into a single representation that captures the overall context. The decoder then uses this representation to generate predictions for the target set, typically employing a Gaussian likelihood for approximation of the true predictive distributions. Due to the analytically tractable likelihood, CNPs can be efficiently trained through maximum likelihood estimation.

## A. 1 Transformer neural processes

Transformer Neural Processes (TNPs), introduced by Nguyen and Grover (2022), improve the flexibility and expressiveness of CNPs by incorporating the Transformer's attention mechanism (Vaswani et al., 2017). In TNPs, the transformer architecture uses self-attention to process the context set, dynamically weighting the importance of each point. This allows the model to create a rich representation of the context, which is then used by the decoder to generate predictions for the target set. The attention mechanism in TNPs facilitates the handling of large and variable-sized context sets, improving the model's performance on tasks with complex input-output relationships. The Transformer architecture is also useful in our setups where certain designs may have a more significant impact on the decision-making process than others. For more details about TNPs, please refer to Nguyen and Grover (2022).

## B Implementation details

## B. 1 Embedders

The embedder $f_{\text {emb }}$ is responsible for mapping the raw data to a space of the same dimension. For the toy example and the top- $k$ hyperparameter task, we use three embedders: a design embedder $f_{\text {emb }}^{(k)}$, an outcome embedder $f_{\text {emb }}^{(u)}$, and a time step embedder $f_{\text {emb }}^{(t)}$. Both $f_{\text {emb }}^{(k)}$ and $f_{\text {emb }}^{(u)}$ are multi-layer perceptions (MLPs) with the following architecture:

- Hidden dimension: the dimension of the hidden layers, set to 32.
- Output dimension: the dimension of the output space, set to 32 .
- Depth: the number of layers in the neural network, set to 4.
- Activation function: ReLU is used as the activation function for the hidden layers.

The time step embedder $f_{\text {emb }}^{(t)}$ is a discrete embedding layer that maps time steps to a continuous embedding space of dimension 32 .

---

#### Page 16

For the decision-aware active learning task, since the design space contains both the covariates and the decision, we use four embedders: a covariate embedder $f_{\text {emb }}^{(x)}$, a decision embedder $f_{\text {emb }}^{(\text {d }}$, an outcome embedder $f_{\text {emb }}^{(\text {a })}$, and a time step embedder $f_{\text {emb }}^{(t)} \cdot f_{\text {emb }}^{(x)} \cdot f_{\text {emb }}^{(y)}$ and $f_{\text {emb }}^{(t)}$ are MLPs which use the same settings as described above. The decision embedder $f_{\text {emb }}^{(\text {d }}$ is another discrete embedding layer.
For context embedding $\boldsymbol{E}^{(\mathrm{c})}$, we first map each $\xi_{i}^{(\mathrm{c})}$ and $y_{i}^{(\mathrm{c})}$ to the same dimension using their respective embedders, and then sum them to obtain the final embedding. For prediction embedding $\boldsymbol{E}^{(\mathrm{p})}$ and query embedding $\boldsymbol{E}^{(\mathrm{q})}$, we only encode the designs. For $\boldsymbol{E}^{(\mathrm{d})}$, except the embeddings of the time step, we also encode the global contextual information $\lambda$ using $f_{\text {emb }}^{(\mathrm{c})}$ in the toy example and the decision-aware active learning task. All the embeddings are then concatenated together to form our final embedding $\boldsymbol{E}$.

# B. 2 Transformer blocks

We utilize the official TransformerEncoder layer of PyTorch (Paszke et al., 2019) (https:// pytorch.org) for our transformer architecture. For all experiments, we use the same configuration: the model has 6 Transformer layers, with 8 heads per layer, the MLP block has a hidden dimension of 128 , and the embedding dimension size is set to 32 .

## B. 3 Output heads

The prediction head, $f_{\mathrm{p}}$ is an MLP that maps the Transformer's output embeddings of the query set to the predicted outcomes. It consists of an input layer with 32 hidden units, a ReLU activation function, and an output layer. The output layer predicts the mean and variance of a Gaussian likelihood, similar to CNPs.

For the query head $f_{\mathrm{q}}$, all candidate experimental designs are first mapped to embeddings $\boldsymbol{\lambda}^{(\mathrm{q})}$ by the Transformer, and these embeddings are then passed through $f_{\mathrm{q}}$ to obtain individual outputs. We then apply a Softmax function to these outputs to ensure a proper probability distribution. $f_{\mathrm{q}}$ is an MLP consisting of an input layer with 32 hidden units, a ReLU activation function, and an output layer.

## B. 4 Training details

For all experiments, we use the same configuration to train our model. We set the initial learning rate to $5 \mathrm{e}-4$ and employ the cosine annealing learning rate scheduler. The number of training epochs is set to 50,000 for top- $k$ tasks and 100,000 for other tasks, and the batch size is 16 . For the REINFORCE, we use a discount factor of $\alpha=0.99$.

## C Full algorithm for training TNDP

```
Algorithm 1 Transformer Neural Decision Processes (TNDP)
    Input: Utility function \(u\left(y_{\Xi}, a\right)\), prior \(p(\theta)\), likelihood \(p(y \mid \theta, \xi)\), query horizon \(T\)
    Output: Trained TNDP
    while within the training budget do
        Sample \(\theta \sim p(\theta)\) and initialize \(D\)
        for \(t=1\) to \(T\) do
            \(\xi_{t}^{(\mathrm{q})} \sim \pi_{t}\left(\cdot \mid h_{1: t-1}\right) \quad \triangleright\) sample next design from policy
            Sample \(y_{t} \sim p(y \mid \theta, \xi) \quad \triangleright\) observe outcome
            Set \(h_{1: t}=h_{1: t-1} \cup\left\{\left(\xi_{t}^{(\mathrm{q})}, y_{t}\right)\right\} \quad \triangleright\) update history
            Set \(D^{(\mathrm{c})}=h_{1: t}, D^{(\mathrm{q})}=D^{(\mathrm{q})} \backslash\left\{\xi_{t}^{(\mathrm{q})}\right\} \quad \triangleright\) update \(D\)
            Calculate \(r_{t}\left(\xi_{t}^{(\mathrm{q})}\right)\) with \(u\left(y_{\Xi}, a\right)\) using Eq. (9) \(\triangleright\) calculate reward
        end for
            \(R_{t}=\sum_{k=t}^{T} \alpha^{k-t} r_{k}\left(\xi_{k}^{(\mathrm{q})}\right) \quad \triangleright\) calculate cumulative reward
        Update TNDP using \(\mathcal{L}^{(p)}\) (Eq. (7)) and \(\mathcal{L}^{(\mathrm{q})}\) (Eq. (10))
    end while
    At deployment, we can use \(f^{(\mathrm{q})}\) to sequentially query \(T\) designs. Afterward, based on the queried
        experiments, we perform one-step final decision-making using the prediction from \(f^{(p)}\).
```

---

#### Page 17

# D Computational cost analysis

## D. 1 Inference time analysis

We evaluate the inference time of our algorithm during the deployment stage. We select decisionaware active learning as the experiment for our time comparison. All experiments are evaluated on an Intel Core i7-12700K CPU. We measure both the acquisition time and the total time. The acquisition time refers to the time required to compute one next design, while the total time refers to the time required to complete 10 rounds of design. The final results are presented in Table A1, with the mean and standard deviation calculated over 10 runs.

Traditional methods rely on updating the GP and optimizing the acquisition function, which is computationally expensive. D-EIG and T-EIG require many model retraining steps to get the next design, which is not tolerable in applications requiring fast decision-making. However, since our model is fully amortized, once it is trained, it only requires a single forward pass to design the experiments, resulting in significantly faster inference times.

| Method                         | Acquisition time (s) | Total time (s) |
| :----------------------------- | :------------------- | :------------- |
| GP-RS                          | $0.00002(0.00001)$   | $28(7)$        |
| GP-US                          | $0.07(0.01)$         | $29(7)$        |
| GP-DUS                         | $0.38(0.02)$         | $44(5)$        |
| T-EIG (Sundin et al., 2018)    | $1558(376)$          | $15613(3627)$  |
| D-EIG (Filstroff et al., 2024) | $572(105)$           | $5746(1002)$   |
| TDNP (ours)                    | $0.015(0.004)$       | $0.31(0.06)$   |

Table A1: Comparison of computational costs across different methods. We report the mean value and (standard deviation) derived from 10 runs with different seeds.

## D. 2 Overall training time

Throughout this paper, we carried out all experiments, including baseline model computations and preliminary experiments not included in the final paper, on a GPU cluster featuring a combination of Tesla P100, Tesla V100, and Tesla A100 GPUs. We estimate the total computational usage to be roughly 5000 GPU hours. For each experiment, it takes around 10 GPU hours on a Tesla V100 GPU with 32 GB memory to reproduce the result, with an average memory consumption of 8 GB .

## E Details of toy example

## E. 1 Data generation

In our toy example, we generate data using a GP with the Squared Exponential (SE) kernel, which is defined as:

$$
k\left(x, x^{\prime}\right)=v \exp \left(-\frac{\left(x-x^{\prime}\right)^{2}}{2 \ell^{2}}\right)
$$

where $v$ is the variance, and $\ell$ is the lengthscale. Specifically, in each training iteration, we draw a random lengthscale $\ell \sim 0.25+0.75 \times U(0,1)$ and the variance $v \sim 0.1+U(0,1)$, where $U(0,1)$ denotes a uniform random variable between 0 and 1 .

## F Details of decision-aware active learning experiments

## F. 1 Data generation

For this experiment, we use a GP with a Squared Exponential (SE) kernel to generate our data. The covariates $x$ are drawn from a standard normal distribution. For each decision, we use an independent GP to simulate different outcomes. In each training iteration, the lengthscale for each GP is randomly sampled as $\ell \sim 0.25+0.75 \times U(0,1)$ and the variance as $v \sim 0.1+U(0,1)$, where $U(0,1)$ denotes a uniform random variable between 0 and 1 .

---

#### Page 18

# F. 2 Other methods description

We compare our method with other non-amortized approaches, all of which use GPs as the functional prior. Each model is equipped with an SE kernel with automatic relevance determination. GP hyperparameters are estimated with maximum marginal likelihood.

Our method is compared with the following methods:

- Random sampling (GP-RS): randomly choose the next design $\xi_{t}$ from the query set.
- Uncertainty sampling (GP-US): choose the next design $\xi_{t}$ for which the predictive distribution $p\left(y_{t} \mid \xi_{t}, h_{t-1}\right)$ has the largest variance.
- Decision uncertainty sampling (GP-DUS): choose the next design $\xi_{t}$ such that the predictive distribution of the optimal decision corresponding to this design is the most uncertain.
- Targeted information (GP-TEIG) (Sundin et al., 2018): a targeted active learning criterion, introduced by (Sundin et al., 2018), selects the next design $\xi_{t}$ that provides the highest EIG on $p\left(y^{*} \mid x^{*}, h_{t-1} \cup\left\{\left(\xi_{t}, y_{t}\right)\right\}\right)$.
- Decision EIG (GP-DEIG) (Filstroff et al., 2024): choose the next design $\xi_{t}$ which directly aims at reducing the uncertainty on the posterior distribution of the optimal decision. See Filstroff et al. (2024) for a detailed explanation.

## F. 3 Ablation study

We also carry out an ablation study to verify the effectiveness of our query head and the non-myopic objective function. We first compare TNDP with TNDP using random sampling (TNDP-RS), and the results are shown in Fig. A1(a). We observe that the designs proposed by the query head significantly improve accuracy, demonstrating that the query head can propose informative designs based on downstream decisions.

We also evaluate the impact of the non-myopic objective by comparing TNDP with a myopic version that only optimizes for immediate utility rather than long-term gains $(\alpha=0)$. The results, presented in Fig. A1(b), show that TNDP with the non-myopic objective function achieves higher accuracy across iterations compared to using the myopic objective. This indicates that our non-myopic objective effectively captures the long-term benefits of each design choice, leading to improved overall performance.

> **Image description.** This is a line graph comparing the performance of two methods, TNDP and TNDP-RS, over 10 design steps.
>
> The graph has the following elements:
>
> - **Axes:** The x-axis is labeled "Design steps t" and ranges from 1 to 10. The y-axis is labeled "Proportion of correct decisions (%)" and ranges from 0.5 to 0.9. Grid lines are visible in the background of the plot.
> - **Data:** There are two lines plotted on the graph:
>   - A green line with triangle markers, labeled "TNDP-RS". This line starts around 0.59 at design step 1 and gradually increases to approximately 0.72 at design step 10. A shaded green area surrounds the line, representing the uncertainty or variability in the data.
>   - An orange line with star markers, labeled "TNDP". This line starts around 0.63 at design step 1, increases more rapidly than the green line, and plateaus around 0.85 at design step 10. A shaded orange area surrounds the line, representing the uncertainty or variability in the data.
> - **Title:** Below the graph, there is a title: "(a) Effect query head".
> - **Legend:** A legend is present at the bottom right of the plot, associating the green line with "TNDP-RS" and the orange line with "TNDP".
>
> The graph visually demonstrates that TNDP generally achieves a higher proportion of correct decisions compared to TNDP-RS across the design steps.

(a) Effect query head

> **Image description.** This is a line graph comparing the "Proportion of correct decisions (%)" against "Design steps t".
>
> The graph has the following characteristics:
>
> - **Axes:** The x-axis is labeled "Design steps t" and ranges from 1 to 10. The y-axis is labeled "Proportion of correct decisions (%)" and ranges from 0.5 to 0.9.
> - **Data:** There are two lines plotted on the graph:
>   - A red line with square markers, labeled "myopic".
>   - An orange line with star markers, labeled "non-myopic".
> - **Shaded Regions:** Each line has a shaded region around it, indicating some measure of variance or uncertainty. The "myopic" line has a red shaded region, and the "non-myopic" line has an orange shaded region.
> - **Trends:** The "non-myopic" line generally shows a higher proportion of correct decisions than the "myopic" line. Both lines show an increasing trend as the number of design steps increases.

(b) Impact of non-myopic objective

Figure A1: Comparison of TNDP variants on the decision-aware active learning task. (a) Shows the effect of the query head, where TNDP outperforms TNDP-RS, demonstrating its ability to generate informative designs. (b) Illustrates the impact of the non-myopic objective, with TNDP achieving higher accuracy than the myopic version.

---

#### Page 19

# G Details of top- $k$ hyperparameter optimization experiments

## G. 1 Data

In this task, we use HPO-B benchmark datasets (Arango et al., 2021). The HPO-B dataset is a large-scale benchmark for HPO tasks, derived from the OpenML repository. It consists of 176 search spaces (algorithms) evaluated on 196 datasets, with a total of 6.4 million hyperparameter evaluations. This dataset is designed to facilitate reproducible and fair comparisons of HPO methods by providing explicit experimental protocols, splits, and evaluation measures.
We extract four meta-datasets from the HPOB dataset: ranger ( $\mathrm{id}=7609, d_{x}=9$ ), svm ( $\mathrm{id}=5891, d_{x}=8$ ), rpart ( $\mathrm{id}=5859, d_{x}=6$ ), and xgboost ( $\mathrm{id}=5971, d_{x}=16$ ). In the test stage, the initial context set is chosen based on their pre-defined indices. For detailed information on the datasets, please refer to https://github.com/releaunifreiburg/HPO-B.

## G. 2 Other methods description

In our experiments, we compare our method with several common acquisition functions used in HPO. We use GPs as surrogate models for these acquisition functions. All the implementations are based on BoTorch (Balandat et al., 2020) (https://botorch.org/). The acquisition functions compared are as follows:

- Random Sampling (RS): This method selects hyperparameters randomly from the search space, without using any surrogate model or acquisition function.
- Upper Confidence Bound (UCB): This acquisition function balances exploration and exploitation by selecting points that maximize the upper confidence bound. The UCB is defined as:

$$
\alpha_{\mathrm{UCB}}(\mathbf{x})=\mu(\mathbf{x})+\kappa \sigma(\mathbf{x})
$$

where $\mu(\mathbf{x})$ is the predicted mean, $\sigma(\mathbf{x})$ is the predicted standard deviation, and $\kappa$ is a parameter that controls the trade-off between exploration and exploitation.

- Expected Improvement (EI): This acquisition function selects points that are expected to yield the greatest improvement over the current best observation. The EI is defined as:

$$
\alpha_{\mathrm{EI}}(\mathbf{x})=\mathbb{E}\left[\max \left(0, f(\mathbf{x})-f\left(\mathbf{x}^{+}\right)\right)\right]
$$

where $f\left(\mathbf{x}^{+}\right)$is the current best value observed, and the expectation is taken over the predictive distribution of $f(\mathbf{x})$.

- Probability of Improvement (PI): This acquisition function selects points that have the highest probability of improving over the current best observation. The PI is defined as:

$$
\alpha_{\mathrm{PI}}(\mathbf{x})=\Phi\left(\frac{\mu(\mathbf{x})-f\left(\mathbf{x}^{+}\right)-\omega}{\sigma(\mathbf{x})}\right)
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution, $f\left(\mathbf{x}^{+}\right)$ is the current best value observed, and $\omega$ is a parameter that encourages exploration.

In addition to those non-amortized GP-based methods, we also compare our method with an amortized surrogate model PFNs4BO (Müller et al., 2023). It is a Transformer-based model designed for hyperparameter optimization which does not consider the downstream task. We use the pre-trained PFNs4BO-BNN model which is trained on HPO-B datasets and choose PI as the acquisition function, the model and the training details can be found in their official implementation (https://github. com/automl/PFNs4BO).

## G. 3 Additional experiment on retrosynthesis planning

We now show a real-world experiment on retrosynthesis planning (Blacker et al., 2011). Specifically, our task is to help chemists identify the top- $k$ synthetic routes for a novel molecule (Mo et al., 2021), as it can be challenging to select the most practical routes from many random options generated by the retrosynthesis software (Stevens, 2011; Szymkuć et al., 2016). In this task, the design space for each molecule $m$ is a finite set of routes that can synthesize the molecule. The sequential experimental

---

#### Page 20

design is to select a route for a specific molecule to query its score $y$, which is calculated based on the tree edit distance (Bille, 2005) from the best route. The downstream task is to recommend the top- $k$ routes with the highest target-specific scores based on the collected information.

In this experiment, we choose $k=3$ and $T=10$, and set the global information $\gamma=m$. We train our TNDP on a novel non-public metadataset, including 1500 molecules with 70 synthetic routes for each molecule. The representation dimension of the molecule is 64 and that of the route is 264 , both of which are learned through a neural network. Given the highdimensional nature of the data representation, it is challenging to directly compare TNDP with other GP-based methods, as GPs typically struggle with scalability and performance in such high-dimensional settings. Therefore, we only compare TNDP with TNDP using random sampling. The final results are evaluated on 50 test molecules that are not seen during training, as shown in Fig. A2.

> **Image description.** This is a line graph comparing the performance of two methods, TNDP and Random search, over a series of design steps.
>
> The graph has the following characteristics:
>
> - **Axes:** The x-axis is labeled "Design steps t" and ranges from 0 to 10. The y-axis is labeled "Utility" and ranges from 6 to 18.
> - **Data:** There are two lines plotted on the graph:
>   - One line, in orange, represents the TNDP method. This line is marked with star-shaped data points.
>   - The other line, in red, represents the Random search method. This line is marked with square-shaped data points.
> - **Error Bands:** Each line has a shaded area around it, representing the standard deviation or confidence interval. The TNDP line has a light orange shaded area, while the Random search line has a light red shaded area.
> - **Legend:** A legend is present in the lower part of the graph, labeling the orange line as "TNDP" and the red line as "Random search".
> - **Overall Trend:** Both lines show an increasing trend, indicating that utility increases with the number of design steps. The TNDP method generally outperforms the Random search method, as its line is consistently higher on the graph.

Figure A2: Results of retrosynthesis planning experiment. The utility is the sum of the quality scores of top- $k$ routes and is calculated with 50 molecules. Our TNDP outperforms the random search baseline.
