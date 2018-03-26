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
f^\omega(x) = h_2(h_1(x^t \text{diag}(\epsilon_1) M_1+b_1)\text{diag}(\epsilon_2)M_2+b2)
$$

Where $\epsilon_{1,2}$ are 0,1 random vectors drawn with some probability $p_{1,2}$ of being 0 (formally $\epsilon_{i,j}\sim B(1-p_i)$).

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

This expression says that the KL divergence + the $\mathcal{L}$ term is *constant* for whatever $q_\theta$ distribution we have!

The term $-\mathcal{L}$ is called ELBO which stands for **estimated lower bound** since this term is a lower bound of the marginal likelihood $p(y\mid X)$. So we see that minimizing the KL divergence is equivalent to maximizing the ELBO which is equivalent to minimizing $\mathcal{L}$:

$$
\arg \min_\theta KL[q_\theta(\omega) || p(\omega \mid y,X)] = \arg \min_\theta \mathcal{L}(q_\theta)
$$

Therefore our approach will be to minimize the $\mathcal{L}$ term w.r.t. $\theta$. Sometimes it happens that the integral $\mathcal{L}$ can be solved analytically for the chosen $q_\theta(\omega)$ distribution (this is a common criteria for choosing a $q_\theta$ distribution).  In that cases we can compute the integral and we will just have an expression depending on $\theta$ to optimize. That explains why VI approaches said that *"VI changes integration with optimization which is much easier"*.

We will assume however that the integral $\mathcal{L}$ cannot be solved analytically. The way we plan to approach the problem is to minimize $\mathcal{L}$ w.r.t. $\theta$ using gradient descent (preferably **stochastic** gradient descent). We will have to compute then the gradient of $\mathcal{L}$ w.r.t. $\theta$: $\nabla_\theta \mathcal{L(q_\theta)}$

We can write $\mathcal{L}$ in the following handy way:

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

If we now plug in the likelihood definition $(\text{Likelihood})$ we get:

$$
\begin{aligned}
\mathcal{L}(q_\theta) &=-\mathbb{E}_{q_\theta(\omega)}\left[ \log \left( \prod_{i=1}^N p(y_i \mid f^\omega(x_i))\right)\right] + KL[q_\theta(\omega) || p(\omega)] \\
&= -\sum_{i=1}^N \mathbb{E}_{q_\theta(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] + KL[q_\theta(\omega) || p(\omega)]
\end{aligned}
$$

This expression starts to resemble the risk equation of the *standard* Neural Networks. It also has the nice property that is a sum over all the training data $\mathcal{D}$. This is cool since it means we can optimize it by **stochastic gradient descent**. Formally that means that if we consider $S$ a random subset of indexes of size $M$ and the following estimator:

$$
\hat{\mathcal{L}}(q_\theta) = -\frac{N}{M} \sum_{i \in S} \mathbb{E}_{q_\theta(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] + KL[q_\theta(\omega) || p(\omega)]
$$

We have that: $$\nabla_\theta \mathcal{L(q_\theta)} = \mathbb{E}_S[\nabla_\theta \hat{\mathcal{L}}(q_\theta)]$$. (The expectation here is taken over all the subsets of $\mathcal{D}$ of size $M$)

So now we can optimize $\mathcal{L}$ by stochastic gradient descent to find the optimal $\theta$ and we will be done! But... wait... *how do we compute the derivative w.r.t. the density function of an expectation?*

$$
\nabla_\theta  \hat{\mathcal{L}}(q_\theta) = -\frac{N}{M} \sum_{i=1}^M \nabla_{\color{red}{\theta}}\mathbb{E}_{q_\color{red}{\theta}(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] + \nabla_\color{red}{\theta} KL[q_\color{red}{\theta}(\omega) || p(\omega)]
$$

This is where we can use the so called *tricks* that are very well explained in Shakir Mohamed blog: [[reparametrisation trick]](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/) [[Log-derivative trick]](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)

## Dropout as Bayesian Neural networks

In the case of **dropout networks** we will use the *reparametrization trick*, this means that the $q_\theta$ function in this case is not manually chosen but it is rather defined through its re-parametrization:

$$
\begin{aligned}
\omega &= g(\theta,\epsilon) = \{diag(\epsilon_i)M_i, b_i\}_{i=1}^L \quad \epsilon_i \sim B(0,p_i)\\
\theta &= \{M_i,b_i\}_{i=1}^L
\end{aligned}
$$

Where $L$ is the number of layers of the NN. The expectation of above can be re-written as:

$$
\mathbb{E}_{q_\theta(\omega)}\left[ \log  p(y_i \mid f^\omega(x_i))\right] = \mathbb{E}_{p(\epsilon)} \left[ \log  p(y_i \mid f^{g(\theta,\epsilon)}(x_i))\right]
$$

(see Gal's Thesis for an easy yet formal proof). We can interchange now the $\nabla_\theta$ operator with the expectation to get:

$$
\nabla_\theta  \hat{\mathcal{L}}(q_\theta) = -\frac{N}{M} \sum_{i \in S} \mathbb{E}_{p(\epsilon)}\left[\nabla_{\theta} \log  p(y_i \mid f^{g(\theta,\epsilon)}(x_i))\right] + \nabla_{\theta} KL[q_{\theta}(\omega) || p(\omega)]
$$

If we sample $S$ and $\epsilon\sim B(p_i)$ on the following expression we will have a *doubly stochastic* estimator of the gradient:

$$
\nabla_\theta  \hat{\hat{\mathcal{L}}}(q_\theta) = -\frac{N}{M} \sum_{i \in S} \left[\nabla_{\theta} \log  p(y_i \mid f^{g(\theta,\epsilon)}(x_i))\right] + \nabla_{\theta} KL[q_{\theta}(\omega) || p(\omega)]
$$

Since $$\nabla_\theta \mathcal{L(q_\theta)} = \mathbb{E}_{S,p(\epsilon)}[\nabla_\theta \hat{\hat{\mathcal{L}}}(q_\theta)]$$.

Now it is where the magic happens since this stochastic gradient is suspiciously similar to $\text{(SG Eq)}$! We *just* have to choose an error function where $\text{err}(f^\omega(x),y) \propto \log(p(y_i\mid f^\omega(x_i)))$ and a regularization $\Omega(\omega)$ that is similar to the divergence!

So if you are training your NN using dropout you are probably doing variational inference for some prior and some approximate distribution $q_\theta$.
