# Amortized Decision-Aware Bayesian Experimental Design - Appendix

---

#### Page 8

# Appendix

## A Conditional neural processes

CNPs [11] are designed to model complex stochastic processes through a flexible architecture that utilizes a context set and a target set. The context set consists of observed data points that the model uses to form its understanding, while the target set includes the points to be predicted. The traditional CNP architecture includes an encoder and a decoder. The encoder is a DeepSet architecture to ensure permutation invariance, it transforms each context point individually and then aggregates these transformations into a single representation that captures the overall context. The decoder then uses this representation to generate predictions for the target set, typically employing a Gaussian likelihood for approximation of the true predictive distributions. Due to the analytically tractable likelihood, CNPs can be efficiently trained through maximum likelihood estimation.

## A. 1 Transformer neural processes

Transformer Neural Processes (TNPs), introduced by [20], enhance the flexibility and expressiveness of CNPs by incorporating the Transformer's attention mechanism [29]. In TNPs, the transformer architecture uses self-attention to process the context set, dynamically weighting the importance of each point. This allows the model to create a rich representation of the context, which is then used by the decoder to generate predictions for the target set. The attention mechanism in TNPs facilitates the handling of large and variable-sized context sets, improving the model's performance on tasks with complex input-output relationships. The Transformer architecture is also useful in our setups where certain designs may have a more significant impact on the decision-making process than others. For more details about TNPs, please refer to [20].

## B Additional details of TNDP

## B. 1 Full algorithm for training TNDP

```
Algorithm 1 Transformer Neural Decision Processes (TNDP)
    Input: Utility function \(u\left(y_{\Xi}, a\right)\), prior \(p(\theta)\), likelihood \(p(y \mid \theta, \xi)\), query horizon \(T\)
    Output: Trained TNDP
    while within the training budget do
        Sample \(\theta \sim p(\theta)\) and initialize \(D\)
        for \(t=1\) to \(T\) do
            \(\xi_{t}^{(q)} \sim \pi_{t}\left(\cdot \mid h_{1: t-1}\right) \quad \triangleright\) sample next design from policy
            Sample \(y_{t} \sim p(y \mid \theta, \xi) \quad \triangleright\) observe outcome
            Set \(h_{t}=h_{t-1} \cup\left\{\left(\xi_{t}^{(q)}, y_{t}\right)\right\} \quad \triangleright\) update history
            Set \(D^{(\mathrm{c})}=h_{1: t}, D^{(\mathrm{q})}=D^{(\mathrm{q})} \backslash\left\{\xi_{t}^{(\mathrm{q})}\right\} \quad \triangleright\) update \(D\)
            Calculate \(r_{t}\left(\xi_{t}^{(q)}\right)\) with \(u\left(y_{\Xi}, a\right)\) using Eq. (4)
            end for
            \(R_{t}=\sum_{k=t}^{T} \alpha^{k-t} r_{k}\left(\xi_{k}^{(q)}\right) \quad \triangleright\) calculate cumulative reward
        Update TNDP using \(\mathcal{L}^{(p)}(\mathrm{Eq} .(3))\) and \(\mathcal{L}^{(\mathrm{q})}\) (Eq. (5))
    end while
    At deployment, we can use \(f^{(\mathrm{q})}\) to sequentially query \(T\) designs. Afterward, based on the queried
        experiments, we perform one-step final decision-making using the prediction from \(f^{(p)}\).
```

## B. 2 Embedders

The embedder $f_{\text {emb }}$ is responsible for mapping the raw data to a space of the same dimension. For the toy example and the top- $k$ hyperparameter task, we use three embedders: a design embedder $f_{\text {emb }}^{(\xi)}$, an outcome embedder $f_{\text {emb }}^{(y)}$, and a time step embedder $f_{\text {emb }}^{(t)}$. Both $f_{\text {emb }}^{(\xi)}$ and $f_{\text {emb }}^{(y)}$ are multi-layer perceptions (MLPs) with the following architecture:

---

#### Page 9

- Hidden dimension: the dimension of the hidden layers, set to 32.
- Output dimension: the dimension of the output space, set to 32 .
- Depth: the number of layers in the neural network, set to 4.
- Activation function: ReLU is used as the activation function for the hidden layers.

The time step embedder $f_{\text {emb }}^{(t)}$ is a discrete embedding layer that maps time steps to a continuous embedding space of dimension 32 .

For the decision-aware active learning task, since the design space contains both the covariates and the decision, we use four embedders: a covariate embedder $f_{\text {emb }}^{(x)}$, a decision embedder $f_{\text {emb }}^{(d)}$, an outcome embedder $f_{\text {emb }}^{(y)}$, and a time step embedder $f_{\text {emb }}^{(t)} \cdot f_{\text {emb }}^{(x)}, f_{\text {emb }}^{(y)}$ and $f_{\text {emb }}^{(t)}$ are MLPs which use the same settings as described above. The decision embedder $f_{\text {emb }}^{(d)}$ is another discrete embedding layer.
For context embedding $\boldsymbol{E}^{(c)}$, we first map each $\xi_{i}^{(c)}$ and $y_{i}^{(c)}$ to the same dimension using their respective embedders, and then sum them to obtain the final embedding. For prediction embedding $\boldsymbol{E}^{(p)}$ and query embedding $\boldsymbol{E}^{(q)}$, we only encode the designs. For $\boldsymbol{E}^{(\mathrm{d})}$, except the embeddings of the time step, we also encode the global contextual information $\lambda$ using $f_{\text {emb }}^{(x)}$ in the toy example and the decision-aware active learning task. All the embeddings are then concatenated together to form our final embedding $\boldsymbol{E}$.

# B. 3 Transformer blocks

We utilize the official TransformerEncoder layer of PyTorch [21] (https://pytorch.org) for our transformer architecture. For all experiments, we use the same configuration, which is as follows:

- Number of layers: 6
- Number of heads: 8
- Dimension of feedforward layer: 128
- Dropout rate: 0.0
- Dimension of embedding: 32

## B. 4 Output heads

The prediction head, $f_{\mathrm{p}}$ is an MLP that maps the Transformer's output embeddings of the query set to the predicted outcomes. It consists of an input layer with 32 hidden units, a ReLU activation function, and an output layer. The output layer predicts the mean and variance of a Gaussian likelihood, similar to CNPs.

For the query head $f_{\mathrm{q}}$, all candidate experimental designs are first mapped to embeddings $\boldsymbol{\lambda}^{(\mathrm{q})}$ by the Transformer, and these embeddings are then passed through $f_{\mathrm{q}}$ to obtain individual outputs. We then apply a Softmax function to these outputs to ensure a proper probability distribution. $f_{\mathrm{q}}$ is an MLP consisting of an input layer with 32 hidden units, a ReLU activation function, and an output layer.

## B. 5 Training details

For all experiments, we use the same configuration to train our model. We set the initial learning rate to $5 \mathrm{e}-4$ and employ the cosine annealing learning rate scheduler. The number of training epochs is set to 50,000 . For the REINFORCE algorithm, we select a discount factor of $\alpha=0.99$.

## C Details of toy example

## C. 1 Data generation

In our toy example, we generate data using a GP with the Squared Exponential (SE) kernel, which is defined as:

---

#### Page 10

$$
k\left(x, x^{\prime}\right)=v \exp \left(-\frac{\left(x-x^{\prime}\right)^{2}}{2 \ell^{2}}\right)
$$

where $v$ is the variance, and $\ell$ is the lengthscale. Specifically, in each training iteration, we draw a random lengthscale $\ell \sim 0.25+0.75 \times U(0,1)$ and the variance $v \sim 0.1+U(0,1)$, where $U(0,1)$ denotes a uniform random variable between 0 and 1 .

# D Details of top- $k$ hyperparameter optimization experiments

## D. 1 Data

In this task, we use HPO-B benchmark datasets [1]. The HPO-B dataset is a large-scale benchmark for HPO tasks, derived from the OpenML repository. It consists of 176 search spaces (algorithms) evaluated on 196 datasets, with a total of 6.4 million hyperparameter evaluations. This dataset is designed to facilitate reproducible and fair comparisons of HPO methods by providing explicit experimental protocols, splits, and evaluation measures.
We extracted four meta-datasets from the HPOB dataset: ranger (7609), svm (5891), rpart (5859), and xgboost (5971). For detailed information on the datasets, please refer to https://github.com/ releaunifreiburg/HPO-B.

## D. 2 Other methods description

In our experiments, we compare our method with several common acquisition functions used in HPO. We use GPs as surrogate models for these acquisition functions. All the implementations are based on BoTorch [2] (https://botorch.org/). The acquisition functions compared are as follows:

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
\alpha_{\mathrm{PI}}(\mathbf{x})=\Phi\left(\frac{\mu(\mathbf{x})-f\left(\mathbf{x}^{+}\right)-\xi}{\sigma(\mathbf{x})}\right)
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution, $f\left(\mathbf{x}^{+}\right)$ is the current best value observed, and $\xi$ is a parameter that encourages exploration.

We also compared our method with an amortized method PFNs4BO [19]. It is a Transformer-based model designed for hyperparameter optimization which does not consider the downstream task. We used the pre-trained PFNs4BO-BNN model and chose PI as the acquisition function. We used the PFNs4BO's official implementation (https://github.com/automl/PFNs4BO).
