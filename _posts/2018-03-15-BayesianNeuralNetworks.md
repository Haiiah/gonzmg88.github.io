---
layout: post
title: Bayesian Neural Networks
date: 2018-03-15
author: Gonzalo Mateo-García
---

<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
      //TeX: { equationNumbers: { autoNumber: "False" } }
    });
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

* TOC
{:toc}

## Introduction
Dropout is a regularization method widely applied in Neural Networks (NN) training. It consists of setting to zero activations of your NN randomly while training. The heuristic behind this is that it can be thought as building an *ensemble of Neural Networks* where each component of this ensemble is a NN with the same structure as the original one but with some activations *dropped* (i.e. set to zero). For example, if we have a two layer Neural Network, dropout is nothing but the following formula:

$$
f^\omega(x) = h_2(h_1(x^t \operatorname{diag}(\epsilon_1) M_1+b_1)\operatorname{diag}(\epsilon_2)M_2+b_2)
$$

Where $$\omega = \{M_{1,2},b_{1,2}\}$$ and $\epsilon_{1,2}$ are 0,1 random vectors drawn with some probability $p_{1,2}$ of being 0 (formally $\epsilon_{i}\sim B(1-p_i)$).

On test time we could a) apply the formula several times with different $\epsilon$ values and compute the empirical mean or b) we could do the *rescale trick* which consists on replacing the $\epsilon_i$ with its expectation ($1-p_i$) and just run the $f^\omega(x)$ formula once. This is the more common approach and it is implemented by default in most NN frameworks.

Interestingly the (a) approach can be seen as having a *probabilistic* prediction which means we can get uncertainty estimates out of the NN. This connexion can be made explicit through Bayesian Neural Networks (BNN).  This post is about this connexion, how can we get from standard NN to BNN to rediscover dropout. This work is mainly due to [Yarin Gal](http://www.cs.ox.ac.uk/people/yarin.gal/). Most of the content is explained in greater detail (probably better too) in [[Yarin Gal Thesis]](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html#thesis).

## Neural Networks basics
In Neural Networks we assume we want to _learn_ a function $f^\omega(x)$ that maps points $x$ from an input space to an output space. This function depends on some *free* parameters $\omega$. To learn this function we minimize a *risk* function $J(\omega)$ (over the free parameters). If we call $$\mathcal{D}=\{x_i,y_i\}_{i=1}^N$$ to our training set, a risk function looks like this:

$$
J(\omega) = \sum_{i=1}^N \text{err}(y_i,f^\omega(x_i)) + \Omega(\omega)
$$

Where $\Omega(\omega)$ is the regularization term. To minimize $J$ we use **stochastic gradient descent** on $\omega$, that is we randomly initialize $\omega$ and then update this value iteratively using the *stochastic estimates* of the gradient, i.e. if we consider $S$ a random subset of indexes of size $M$ the stochastic gradient is:

$$
\nabla_{\omega}\hat{J}(\omega) = \frac{N}{M}\sum_{i\in S} \nabla_{\omega}[\text{err}(y_i,f^\omega(x_i))] + \nabla_{\omega}\Omega(\omega) \quad \text{(SG Eq)}
$$

Stochastic gradient descent will eventually give us some *optimal* weights $$\omega^\star$$ that we will use to make predictions for new $$x^\star$$ points (i.e. computing  $$f^{\omega^\star}(x^\star)$$).

This is useful and it maybe enough in many applications. However, sometimes we might want to give some *uncertainty information* to our predictions. One of the most *principled* way to do this is by using Bayesian statistics. Bayesian statistics + neural networks = Bayesian Neural Networks! (BNNs)

## Bayesian Neural Networks basics

In Bayesian Neural Networks we start by assuming some probabilistic dependency between the outputs of the Neural Network ($f^\omega(x)$) and the true values: $p(y_i \mid f^\omega(x_i))$.
In the case of regression we usually use the normal distribution: $p(y_i \mid f^\omega(x_i))\sim \mathcal{N}(f^\omega(x_i),\sigma^2)$ (which is equivalent to say that $y = f^\omega(x_i) + \epsilon$ with $\epsilon\sim \mathcal{N}(0,\sigma^2)$).

In binary classification we could set for example $p(y_i \mid f^\omega(x_i))\sim B(\text{sigmoid}(f^\omega(x_i)))$. ($B$ is the bernuilli distribution)

The goal of BNN is to find a probability distribution for the weights $\omega$ instead of some bare values for $\omega$. This probability distribution over the weights $\omega$ is called the **posterior distribution**: $p(\omega \mid y,X)$. Bayes Theorem gives us the way to compute it:

$$
p(\omega \mid y,X) = \frac{p(y\mid X,\omega)p(\omega)}{p(y\mid X)}\quad \text{(Bayes Theorem)}
$$

The terms on the $\text{(Bayes Theorem)}$ have their own names:

* The first term on the denominator is called the **likelihood**.  We will assume that each term on this *joint probability* is independent of each other *given* the prediction vale $f^\omega(x_i)$ which mathematically means:
$$
p(y\mid \omega, X) = \prod_{i=1}^N p(y_i \mid f^\omega(x_i))\quad \text{(Likelihood)}
$$

* The second term on the denominator is called the **prior** over the weights $\omega$. It is the controversial one since we have to choose it by hand and it will affect the posterior distribution we find out. If we have enough data we can just use a *non informative prior* like $\mathcal{N}(0,\beta)$ with $\beta$ large.

* The term on the denominator is called the **marginal likelihood**. This value is *constant* for a fixed dataset and it is usually hard to find. If we knew it we would have the **posterior** since we can evaluate the likelihood and the prior. This term can be computed doing the following integral:

$$
p(y\mid X) = \int p(y\mid X, \omega)p(\omega)d\omega
$$

## Variational Inference Approach

The Variational Inference (VI) approach arises because the marginal likelihood term is *difficult* to obtain. In VI we want to find a distribution over $\omega$ that depends on a set of parameters $\theta$ that **approximates the posterior**: $q_\theta(\omega)\approx p(\omega \mid y,X)$.


Here, in VI, *"approximates"* means **minimizes the Kullback-Leibler divergence** between the two distributions:

$$
\arg\min_\theta KL[q_\theta(\omega) || p(\omega \mid y,X)] = \arg\min_\theta -\mathbb{E}_{q_\theta(\omega)}\left[\log \frac{p(\omega \mid y, X)}{q_\theta(\omega)}\right]
$$

For example, we can try to find the normal distribution that best approximates the posterior: in that case we can set: $q_\theta(\omega) = \mathcal{N}(\omega \mid \mu,  \Sigma)$ and $$\theta=\{\mu,\Sigma\}$$. Then we minimize w.r.t. $\theta$ this divergence and we are done!

Nice, but, if we look at the KL divergence expression it depends on the posterior that it is what we are looking for... So it is not very useful... Fortunately we have the following expression that is central on the VI literature.

$$
\log p(y \mid X) = -\overbrace{-\mathbb{E}_{q_\theta(\omega)}\left[\log \frac{p(y\mid\omega,X)p(\omega)}{q_\theta(\omega)} \right]}^{\mathcal{L}(q_\theta)} \overbrace{-\mathbb{E}_{q_\theta(\omega)}\left[\log \frac{p(\omega \mid y,X)}{q_\theta(\omega)}\right]}^{KL[q_\theta(\omega) || p(\omega \mid y,X)]}
$$

This expression says that the KL divergence + the $\mathcal{L}$ term is **constant** for whatever $q_\theta$ distribution we have!

The term $-\mathcal{L}$ is called ELBO which stands for **evidence lower bound** since this term is a lower bound of the marginal likelihood (or evidence) $p(y\mid X)$. So we see that minimizing the KL divergence is equivalent to maximizing the ELBO which is equivalent to minimizing $\mathcal{L}$:

$$
\arg \min_\theta KL[q_\theta(\omega) || p(\omega \mid y,X)] = \arg \min_\theta \mathcal{L}(q_\theta)
$$

**Therefore our approach will be to minimize the $\mathcal{L}$ term w.r.t. $\theta$ using (stochastic) gradient descent**. Sometimes it happens that the integral $\mathcal{L}$ can be solved analytically for the chosen $q_\theta(\omega)$ distribution (this is a common criteria for choosing a $q_\theta$ distribution).  In that cases we can compute the integral and we will just have an expression depending on $\theta$ to optimize. That explains why VI approaches said that *"VI changes integration with optimization which is much easier"*.

<!--We will assume however that the integral $\mathcal{L}$ cannot be solved analytically. The way we plan to approach the problem is to minimize $\mathcal{L}$ w.r.t. $\theta$ using gradient descent (preferably **stochastic** gradient descent). We will have to compute then the gradient of $\mathcal{L}$ w.r.t. $\theta$: $\nabla_\theta \mathcal{L(q_\theta)}$-->

We will now expand the $\mathcal{L}$ term so that we can see how we will proceed to its minimization:

$$
\begin{aligned}
\mathcal{L}(q_\theta) &= \mathbb{E}_{q_\theta(\omega)}\left[-\log p(y\mid\omega,X) - \log \frac{p(\omega)}{q_\theta(\omega)} \right] \\
&= -\mathbb{E}_{q_\theta(\omega)}\left[\log p(y\mid\omega,X)\right] -\mathbb{E}_{q_\theta(\omega)}\left[ \log \frac{p(\omega)}{q_\theta(\omega)} \right]\\
&= -\mathbb{E}_{q_\theta(\omega)}\left[\log p(y\mid\omega,X)\right] + KL[q_\theta(\omega) || p(\omega)]
\end{aligned}
$$

<div class="alert alert-warning" role="alert" markdown="1">
**Watch out** that the $KL$ divergence of this equation is the divergence between the **prior** and our approximate distribution $q_\theta$ whereas before we had the divergence between the **posterior** and the approximate distribution $q_\theta$.
</div>

This expression is a **trade-off between the prior and the likelihood**: it has the interpretation of do not diverge too much from the prior unless you can reduce significantly the first expectation!

If we now plug in the likelihood definition of the BNN $(\text{Likelihood})$ we get:

$$
\begin{aligned}
\mathcal{L}(q_\theta) &=-\mathbb{E}_{q_\theta(\omega)}\left[ \log \left( \prod_{i=1}^N p(y_i \mid f^\omega(x_i))\right)\right] + KL[q_\theta(\omega) || p(\omega)] \\
&= -\sum_{i=1}^N \mathbb{E}_{q_\theta(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] + KL[q_\theta(\omega) || p(\omega)] \quad \text{(ELBO BNN)}
\end{aligned}
$$

This expression starts to resemble the risk equation of the *standard* Neural Networks. Actually, if we choose the approximate distribution $q_\theta(\omega)$ to be a $\delta$ distribution we will obtain the maximum a-posteriori (MAP) solution which is the same as the risk function of neural networks with $\mathcal{l}_2$ regularization. Click <a onclick="$('#map_relation').slideToggle();">unfold</a> if you are interested in this relation.

### MAP Variational inference relation

<a onclick="$('#map_relation').slideToggle();"> Unfold explanation </a>

<div id="map_relation" class="input_hidden" markdown="1">
Suppose we have a two layer NN let $$\omega=\{W_1, W_2, \beta_1, \beta_2 \}$$ be the parameters we want the posterior. If we set the approximate posterior to be a delta centered on some $\theta$ parameters:

$$q_\theta(\omega) = \delta(\omega-\theta) = \delta(W_1-M_1) \delta(W_2-M_2)\delta(\beta_1-b_1) \delta(\beta_2-b_2)$$ with $$\theta = \{M_1,M_2,b_1,b_2\}$$

Then the expectation within the $\text{(ELBO BNN)}$ equation is:

$$
 \mathbb{E}_{q_\theta(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] = \log  p(y_i \mid f^\theta(x_i))
$$

And the Kullback-Leibler divergence if we set a prior $$\{W_{1,2},\beta_{1,2}\}\sim \mathcal{N}(0,\lambda^{-1}I)$$:

$$
\begin{aligned}
KL[q_\theta(\omega) || p(\omega)] &= \int q_\theta(\omega)\Big(\log q_\theta(\omega) - \log(\mathcal{N}(\omega \mid 0,\lambda^{-1}I)) \Big) d\omega \\
KL[q_\theta(\omega) || p(\omega)] &= - \log(\mathcal{N}(\theta \mid 0,\lambda^{-1}I))\\
&= \tfrac{\lambda}{2}\|\theta\|^2 +\tfrac{K}{2}\log{2\pi} +\tfrac{K}{2}\log{\lambda^{-1}}
\end{aligned}
$$

Therefore the $\text{(ELBO BNN)}$ equation that we seek to minimize is:

$$
\mathcal{L}(q_\theta) = -\sum_{i=1}^N \log  p(y_i \mid f^\theta(x_i)) + \tfrac{\lambda}{2}\|\theta\|^2 +\tfrac{K}{2}\log{2\pi} +\tfrac{K}{2}\log{\lambda^{-1}}
$$

Which is the same function to minimize as the MAP estimator and the same function to minimize as the standard risk in Neural Networks with $\mathcal{l}_2$ regularization (weight decay).

<a onclick="$('#map_relation').slideToggle();"> Collapse </a>
</div>

The expression above $\text{(ELBO BNN)}$ has also the nice property that is a sum over all the training data $\mathcal{D}$. This is cool since it means we can optimize it by **stochastic gradient descent**. Formally that means that if we consider $S$ a random subset of indexes of size $M$ and the following estimator:

$$
\hat{\mathcal{L}}(q_\theta) = -\frac{N}{M} \sum_{i \in S} \mathbb{E}_{q_\theta(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] + KL[q_\theta(\omega) || p(\omega)]
$$

We have that: $$\nabla_\theta \mathcal{L(q_\theta)} = \mathbb{E}_S[\nabla_\theta \hat{\mathcal{L}}(q_\theta)]$$. (The expectation here is taken over all the subsets of $\mathcal{D}$ of size $M$)

So now we can minimize $\mathcal{L}$ by stochastic gradient descent to find the optimal $\theta$ and we will be done! But... wait... *how do we compute the derivative w.r.t. the density function of an expectation?*

$$
\nabla_\theta  \hat{\mathcal{L}}(q_\theta) = -\frac{N}{M} \sum_{i=1}^M \nabla_{\color{red}{\theta}}\mathbb{E}_{q_\color{red}{\theta}(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] + \nabla_\color{red}{\theta} KL[q_\color{red}{\theta}(\omega) || p(\omega)]
$$

This is where we can use the so called *tricks* that are very well explained in Shakir Mohamed blog: [[reparametrisation trick]](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/) [[Log-derivative trick]](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/). In the following section we will use the reparametrisation trick for the specific approximated distribution using in dropout networks.

## Dropout as Bayesian Neural networks

In the case of **dropout networks** we will use the *reparametrization trick*, this means that it is not necessary to bother by explicitly defining an approximate distribution $q_\theta$ but only its relation with a parameter free random variable (free here means *not depending on $\theta$*):

$$
\begin{aligned}
\omega &= g(\theta,\epsilon) = \{\operatorname{diag}(\epsilon_i)M_i, b_i\}_{i=1}^L \quad \epsilon_i \sim B(0,p_i)\\
\theta &= \{M_i,b_i\}_{i=1}^L
\end{aligned}
$$

Where $L$ is the number of layers of the NN. We could also explicitly define the distribution as we did before for the case of the MAP, click <a onclick="$('#dropout_relation').slideToggle();">unfold</a> below if you want to see the details.

### Approximate distribution dropout

<a onclick="$('#dropout_relation').slideToggle();"> Unfold explanation </a>

<div id="dropout_relation" class="input_hidden" markdown="1">

If we have a two layer fully connected NN, using the notation of <a href="#MAP-Variational-inference-relation"> the MAP-Variational inference relation of above</a> the approximate distribution defined by dropout is:

$$
\begin{aligned}
q_\theta(\omega) &= q_{M_1}(W_1) q_{M_2}(W_2) \delta(\beta_1 - b_1)\delta(\beta_2 - b_2) \\
&=  \prod_{i=1}^{n_{inputs}} \overbrace{\Big((1-p_1)\delta(W_1(i,:) - M_1(i,:))  + p_1\delta(W_1(i,:))\Big)}^{q_{M_1(i,:)}} \\ &\quad\quad \prod_{i=1}^{n_{hidden}} \overbrace{\Big((1-p_2)\delta(W_2(i,:) - M_2(i,:))  + p_2\delta(W_2(i,:))\Big)}^{q_{M_2(i,:)}}  \delta(\beta_1 - b_1)\delta(\beta_2 - b_2)
\end{aligned}
$$

Computing the expectation within the $\text{(ELBO BNN)}$ equation is not as easy as it was in the case of the all delta distributions of the MAP. For example if we assume the dropout is only applied on the first layer ($p_2=0$).


$$
\begin{aligned}
 \mathbb{E}_{q_\theta(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] &= \int \log  p(y_i \mid f^{W_1,M_2,b_1,b_2}(x_i)) q_{M_1}(W_1) dW_1 \\
 &= \int ... \left(\int \log  p(y_i \mid f^{W_1,M_2,b_1,b_2}(x_i)) q_{M_1(1,:)}(W_1(1,:))dW_1(1,:)\right)\\
 &\quad \quad q_{M_1(2,:)}(W_1(2,:))dW_1(2,:)q_{M_1(2,:)} ... (W_1(n_{inputs},:))dW_1(n_{inputs},:) \\
 &= \int ...\int \left((1-p_1)\log  p(y_i \mid f^{M_1(1,:),W_1\setminus W_1(1,:) ,M_2,b_1,b_2}(x_i)) + p_1 \log  p(y_i \mid f^{0,W_1\setminus W_1(1,:) ,M_2,b_1,b_2}(x_i)) \right)\\
 &\quad \quad q_{M_1(2,:)}(W_1(2,:))dW_1(2,:)q_{M_1(2,:)} ... (W_1(n_{inputs},:))dW_1(n_{inputs},:) \\
 \end{aligned}
$$

Which if we keep solving the integrals will be a sum with $2^{n_{inputs}}$ terms each of the terms corresponding to the matrix $M_1$ with a subset of rows set to zero. The weight of each of these terms will be $(1-p_1)^{(n_{inputs}-k)}(p_1)^k$ (where $k$ is the number of rows set to zero).

Following the same reasoning, if dropout is applied on both layers the number of terms will be $2^{n_{inputs}+n_{hidden}}$ with all the possibilities of the matrices $M_1$ and $M_2$ having a subset of rows set to zero!

Recall now that this will be just the evaluation of the ELBO for one of the inputs but the goal is to minimize the ELBO w.r.t. $\theta=\{M_1,M_2,b_1,b_2\}$ so multiple evaluations of the term must be made and we must also compute the derivative which will be also a sum over such number of terms!

<a onclick="$('#dropout_relation').slideToggle();"> Collapse </a>

</div>

Using the reparametrization trick the expectation of the $\text{(ELBO BNN)}$ term can be re-written as:

$$
\mathbb{E}_{q_\theta(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] = \mathbb{E}_{p(\epsilon)} \left[ \log  p(y_i \mid f^{g(\theta,\epsilon)}(x_i))\right]
$$

(see Gal's Thesis for an easy yet formal proof). We can interchange now the $\nabla_\theta$ operator with the expectation to get:

$$
\nabla_\theta  \hat{\mathcal{L}}(q_\theta) = -\frac{N}{M} \sum_{i \in S} \mathbb{E}_{p(\epsilon)}\left[\nabla_{\theta} \log  p(y_i \mid f^{g(\theta,\epsilon)}(x_i))\right] + \nabla_{\theta} KL[q_{\theta}(\omega) || p(\omega)]
$$

If we sample $S$ and $\epsilon\sim B(1-p_i)$ on the following expression we will have a *doubly-stochastic* estimator of the gradient:

$$
\nabla_\theta  \hat{\hat{\mathcal{L}}}(q_\theta) = -\frac{N}{M} \sum_{i \in S} \left[\nabla_{\theta} \log  p(y_i \mid f^{g(\theta,\epsilon)}(x_i))\right] + \nabla_{\theta} KL[q_{\theta}(\omega) || p(\omega)]
$$

Since $$\nabla_\theta \mathcal{L(q_\theta)} = \mathbb{E}_{S,p(\epsilon)}[\nabla_\theta \hat{\hat{\mathcal{L}}}(q_\theta)]$$.

Now it is where the magic happens since this stochastic gradient is suspiciously similar to $\text{(SG Eq)}$! We *just* have to choose an error function where $\text{err}(f^\omega(x),y) \propto \log(p(y_i\mid f^\omega(x_i)))$ and a prior and a regularization $\Omega(\omega)$ that is similar to the divergence! [[Yarin Gal Thesis]](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html#thesis) have the details regarding the *implicit* prior that leads to the same regularization as $\mathcal{l}_2$ regularization for example.

So if you are training your NN using dropout you are probably doing variational inference for some prior, therefore you could take advantage of not only the weights you found by the optimization procedure ($\theta^\star$) but also of the approximate posterior distribution $q_\theta^\star$. Using the approximate distribution you can compute uncertainty measures of your predictions!

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
