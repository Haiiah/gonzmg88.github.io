---
layout: post
title: Variational Free Energy for Sparse GPs
date: 2018-04-19
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

Let $$f=\{f^\star,\mathrm{f}\}$$ be the noise free latent variables of a $\mathcal{GP}$. We write $\mathrm{f}$ to be the latent variables where we have noisy observations $y$: $$y_i = \mathrm{f}_i + \epsilon_i$$ ($$\epsilon_i \sim \mathcal{N}(0,\sigma^2) $$) and $f^\star$ the points where we want to have predictions. The purpose of this post is to understand the Variational Free Energy (VFE) approach to $\mathcal{GP}$s of [[Titsias 2009]](https://pdfs.semanticscholar.org/db7b/e492a629a98db7f9d77d552fd3568ff42189.pdf){:target="_blank"} expanded later by [[Hensman et. al 2013]](http://arxiv.org/abs/1309.6835){:target="_blank"}. For this purpose we start  considering the **joint posterior** $p(f^\star,\mathrm{f} \mid y)$.  This joint posterior is a _strange_ object we normally do not consider if we are working with $\mathcal{GP}$s. When we work with $\mathcal{GP}$s we normally care about the **predictive posterior** $p(y^\star\mid y)$ (to make predictions) or the **marginal likelihood** $p(y)$ (to fit the hyperparameters of the kernel and the $\sigma^2$).

For the _standard_ $\mathcal{GP}$ this **joint posterior** is:

$$
\begin{aligned}
\require{cancel}
p(f^\star,\mathrm{f} \mid y) &= p(f^\star \mid \mathrm{f},\cancel{y})p(\mathrm{f} \mid y) \\
 &= \mathcal{N}\left(f^\star \mid K_{\star \mathrm{f}}K_{\mathrm{f}\mathrm{f}}^{-1}\mathrm{f}, K_{\star\star} - K_{\star \mathrm{f}}K_{\mathrm{f}\mathrm{f}}^{-1}K_{\mathrm{f}\star}\right)\frac{p(y \mid \mathrm{f}) p(\mathrm{f})}{p(y)} \\
 &= \mathcal{N}\left(f^\star \mid K_{\star \mathrm{f}}K_{\mathrm{f}\mathrm{f}}^{-1}\mathrm{f}, K_{\star\star} - K_{\star \mathrm{f}}K_{\mathrm{f}\mathrm{f}}^{-1}K_{\mathrm{f}\star}\right)\frac{\mathcal{N}(y \mid \mathrm{f},\sigma^2 I) \mathcal{N}(\mathrm{f} \mid 0, K_{\mathrm{f}\mathrm{f}})}{\mathcal{N}(y \mid 0, K_{\mathrm{f}\mathrm{f}}+\sigma^2 I)}
\end{aligned}
$$

If we integrate out the noise free observations $\mathrm{f}$ of the above equation we retrieve the **noise free predictive posterior $p(f^\star \mid y)$**.

When we do this we should get back the well known formula:

$$
\begin{aligned}
p(f^\star \mid y ) &= \int p(f^\star , \mathrm{f} \mid y) d\mathrm{f} \\
&= \mathcal{N}(f^\star \mid K_{\star \mathrm{f}}(K_{\mathrm{f}\mathrm{f}}+\sigma^2 I)^{-1}y, \\
&\quad \quad \quad \quad K_{\star\star}- K_{\star \mathrm{f}}(K_{\mathrm{f}\mathrm{f}}+\sigma^2 I)^{-1}K_{\mathrm{f}\star})
\end{aligned}
$$

* _If you manage to do this integral without driving you crazy let me know_
* _The only difference between this noise free predictive posterior and the "standard" predictive posterior ($$p(y^\star \mid y) $$ is that we have to add $\sigma^2 I $ to the variance of the later)_. <a onclick="$('#predictive_posterior').slideToggle();">[Click here to see why]</a>

<div id="predictive_posterior" class="input_hidden" markdown="1">

$$
\begin{aligned}
p(y^\star \mid y) &= \int p(y^\star \mid f^\star, \cancel{y}) p(f^\star \mid y) df^\star \\
    &= \int \mathcal{N}(y^\star \mid f^\star, \sigma^2 I ) p(f^\star \mid y) df^\star
\end{aligned}
$$

This integral of two Gaussians can be done using Bishop's book: Machine Learning and Pattern Recognition. in this case we will have to move to *section 2.3.3. Bayes' theorem for Gaussian variables*. This integral also came up in the [previous blog post on Nyström method]({{site.baseurl}}/blog/2017/10/24/NystromRBFNN#fully-bayesian-approach){:target="_blank"} and we showed there an alternative way to compute it. Anyway, the result is as expected the predictive distribution of the $\mathcal{GP}$:

$$
\begin{aligned}
p(y^\star \mid y) &= \mathcal{N}(y^\star \mid K_{\star \mathrm{f}}(K_{\mathrm{f}\mathrm{f}}+\sigma^2 I)^{-1}y, \\
&\quad \quad \quad \quad K_{\star\star} + \sigma^2 I - K_{\star \mathrm{f}}(K_{\mathrm{f}\mathrm{f}}+\sigma^2 I)^{-1}K_{\mathrm{f}\star})
\end{aligned}
$$

<a onclick="$('#predictive_posterior').slideToggle();"> Collapse </a>
</div>

## Variational sparse approximation

The purpose to introduce the joint posterior is that this is the object we want to approximate with variational inference. We call $q(f)$ to the approximate posterior distribution: $q(f)\approx p(f \mid y)$. To do this approximation we use the standard bound used in variational inference:

$$
\begin{aligned}
\log p(y) &= \overbrace{\mathbb{E}_{q(f)}\left[\log \frac{p(y,f)}{q(f)} \right]}^{\mathcal{L}(q)} \overbrace{-\mathbb{E}_{q(f)}\left[\log \frac{p(f \mid y)}{q(f)}\right]}^{KL\left[q \mid\mid p\right]}\\
&\geq \mathcal{L}(q)
\end{aligned}
$$

Since $\log p(y)$ is constant, maximizing $\mathcal{L}(q)$ is equivalent to minimize the Kullback-Leibler divergence between $q$ and the true joint posterior $p(f\mid y)$.

We now introduce the $u$ variables, called in the sparse $\mathcal{GP}$ literature pseudo outputs. $u$ are latent variables that are within $f$, so $f$ is now: $$f = \{f^\star,\mathrm{f}, u\}$$. We can decompose the $\mathcal{L}$ bound as:

$$
\require{cancel}
\begin{aligned}
\mathcal{L}(q) &= \mathbb{E}_{q(f)}\left[\log \frac{p(y\mid f)p(f)}{q(f)} \right] \\
&= \mathbb{E}_{q(f)}\left[\log \frac{p(y\mid \mathrm{f}, \cancel{f^\star}, \cancel{u})p(f^\star, \mathrm{f} \mid u) p(u)}{q(f)} \right]
\end{aligned}
$$

Since this is a $\mathcal{GP}$ $p(f) = \mathcal{N}(f \mid 0,K_{ff})$ so $p(f^\star, \mathrm{f} \mid u)$ can be easily computed using Gaussian identities ([wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions){:target="_blank"}). In this case we see that $p(f^\star, \mathrm{f} \mid u)$ is the standard posterior of the $\mathcal{GP}$ with noise free variables $u$:

$$
\begin{aligned}
p(f^\star, \mathrm{f} \mid u) = \mathcal{N}\Big(\begin{pmatrix} \mathrm{f} \\ f^\star  \end{pmatrix} \mid &\begin{pmatrix} K_{\mathrm{f}u} \\ K_{\star u}\end{pmatrix}K_{uu}^{-1}u,\\
&\begin{pmatrix} K_{\mathrm{f}\mathrm{f}} & K_{\mathrm{f}\star} \\
 K_{\star \mathrm{f}} & K_{\star\star}\end{pmatrix} - \begin{pmatrix} Q_{\mathrm{f}\mathrm{f}} & Q_{\mathrm{f}\star}\\
 Q_{\star \mathrm{f}} & Q_{\star\star}\end{pmatrix} \Big) \quad \text{(CD)}
 \end{aligned}
$$

Where $Q_{\mathrm{f}\star} := K_{\mathrm{f}u}K_{uu}^{-1}K_{u\star}$

We will write here for reference the marginal distributions of the former equation which are: $p(\mathrm{f} \mid u)=\mathcal{N}(\mathrm{f} \mid K_{\mathrm{f}u}K_{uu}^{-1}u , K_{\mathrm{f}\mathrm{f}} - Q_{\mathrm{f}\mathrm{f}})$ and $$p(f^\star \mid u)=\mathcal{N}(f^\star \mid K_{\star u}K_{uu}^{-1}u , K_{\star\star} - Q_{\star\star})$$.

The starting point of Titsias VFE approximation is that it chooses $q$ as:

$$
q(f) = q(f^\star,\mathrm{f},u) = p(f^\star,\mathrm{f} \mid u) q(u)
$$

Notice that:

* We could also rewrite the true joint posterior $p(f\mid y)$ using the subset $u$ of $f$:

$$
p(f\mid y) = p(f^\star,\mathrm{f},u \mid y) = p(f^\star, \mathrm{f} \mid u,y)p(u \mid y)
$$

* If we compare this two equations we see that **The Titsias approximation removes the dependency on the data ($y$) in the first term**. (While the second term $q(u)$ is let free).

* Again, in order to make predictions, we will need the **approximate (noise free) predictive distribution** $q(f^\star)$, thus we have to integrate out $\mathrm{f}$ and $u$ from $q(f)$:

$$
\begin{aligned}
q(f^\star) &= \int \int p(f^\star,\mathrm{f} \mid u) q(u) d\mathrm{f}du \\
 &= \int q(u) \underbrace{\int p(f^\star,\mathrm{f} \mid u)  d\mathrm{f}}_{p(f^\star \mid u)}du \\
 &= \int q(u) \mathcal{N}(f^\star \mid K_{\star u}K_{uu}^{-1}u, K_{\star\star}-Q_{\star\star})du
\end{aligned}
$$

* If we are *"lucky"* and $q$ is normal, $q(u)=\mathcal{N}(u \mid m, S)$, then the **approximate predictive distribution** (APD) can be computed using the Bishop Chapter 2 pág 93 trick of Marginal and Conditional Gaussians:

$$
\begin{aligned}
q(f^\star) = \mathcal{N}(f^\star &\mid K_{\star u}K_{uu}^{-1}m, \quad \text{(APD)}\\
&K_{\star\star}-Q_{\star\star}+ K_{\star u}K_{uu}^{-1}S K_{uu}^{-1}K_{u\star})
\end{aligned}
$$

* Substituting this $q$ in the bound $\mathcal{L}(q)$ leads to:

$$
\begin{aligned}
\require{cancel}
\mathcal{L}(q) &= \mathbb{E}_{q(f)}\left[\log \frac{p(y\mid \mathrm{f})\cancel{p(f^\star, \mathrm{f} \mid u)} p(u)}{\cancel{p(f^\star, \mathrm{f} \mid u)}q(u)}\right] \\
&= \mathbb{E}_{q(f)}\left[\log p(y\mid \mathrm{f})\right] + \mathbb{E}_{q(f)}\left[\log \frac{p(u)}{q(u)}\right] \\
&= \mathbb{E}_{p(f^\star, \mathrm{f} \mid u)q(u)}\left[\log p(y\mid \mathrm{f})\right] + \mathbb{E}_{p(f^\star, \mathrm{f} \mid u)q(u)}\left[\log \frac{p(u)}{q(u)}\right] \\
&= \mathbb{E}_{p(f^\star \mid \mathrm{f} , u)p(\mathrm{f} \mid u)q(u)}\left[\log p(y\mid \mathrm{f})\right] + \cancel{\mathbb{E}_{p(f^\star, \mathrm{f} \mid u)}}\mathbb{E}_{q(u)}\left[\log \frac{p(u)}{q(u)}\right] \\
&= \cancel{\mathbb{E}_{p(f^\star \mid \mathrm{f} , u)}}\mathbb{E}_{q(u)}\left[\mathbb{E}_{p(\mathrm{f} \mid u)}\log p(y\mid \mathrm{f})\right] + \mathbb{E}_{q(u)}\left[\log \frac{p(u)}{q(u)}\right] \\
&= \mathbb{E}_{q(u)}\left[\mathbb{E}_{p(\mathrm{f} \mid u)}\log p(y\mid \mathrm{f})\right] - KL\left[q(u) \mid\mid p(u) \right] \\
\end{aligned}
$$

* We will call $$\mathcal{L}_1$$ to the integral $$\mathbb{E}_{p(\mathrm{f} \mid u)}\left[\log p(y \mid \mathrm{f})\right]$$. Since $p(\mathrm{f} \mid u)$ is Gaussian this integral can be computed exactly. _(We'll have the derivation bellow)_.

* The equation $\mathcal{L}(q)$ of above **shows a trade-off that normally appears in Variational Inference**: we must find a $q(u)$ distribution that on one hand has high _expected likelihood_  $$\mathbb{E}_{q(u)}\left[\mathcal{L}_1\right]$$ but is not very divergent to the prior $p(u)=\mathcal{N}(u\mid 0, K_{uu})$. We saw a similar trade-off in the previous [Bayesian Neural Network blog post]({{site.baseurl}}/blog/2017/10/24/BayesianNeuralNetworks#variational-inference-approach){:target="_blank"}.

The roadwork is now to (1) compute the $\mathcal{L}_1$ integral, (2) plug-in $\mathcal{L}_1$ into $\mathcal{L}(q)$ (3) assume that $q(u)$ is normally distributed $q(u) = \mathcal{N}(u \mid m, S)$ (4) maximize $\mathcal{L}(q)$ w.r.t. the variational parameters $m$ and $S$.

### Compute $\mathcal{L}_1$ integral

We show below the value of the $\mathcal{L}_1$ integral. <a onclick="$('#l1_integral').slideToggle();">[Click here to see the full derivation]</a>.

<div id="l1_integral" class="input_hidden" markdown="1">

Let's try to compute it:

$$
\begin{aligned}
\mathcal{L}_1 &= \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\log \mathcal{N}(y \mid \mathrm{f}, \sigma^2 I)\right] \\
&= \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \exp\left(\frac{-1}{2\sigma^{2}} \|y -\mathrm{f}\|^2\right)\right)\right] \\
&= \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}} \|y -\mathrm{f}\|^2\right] \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\frac{-1}{2\sigma^{2}} \|y -\mathrm{f}\|^2\right] \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\mathbb{E}_{p(\mathrm{f} \mid u)}\left[ \left(\|y\|^2 + \|\mathrm{f}\|^2   -2 y^t \mathrm{f}\right)\right] \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( \|y\|^2 + \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\|\mathrm{f}\|^2\right]   -2 y^t \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\mathrm{f}\right]\right) \\
\end{aligned}
$$

Now the trick is to use that $p(\mathrm{f} \mid u)$ is normal, if we write: $\mathcal{N}(\mathrm{f} \mid \mu_{\mathrm{f}\mid u}, \operatorname{Cov}_{\mathrm{f}\mid u})$

$$
\begin{aligned}
\mathcal{L}_1 &= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( y^t y + \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\mathrm{f}^t \mathrm{f}\right]   -2 y^t \mu_{\mathrm{f}\mid u}\right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( y^t y + \mathbb{E}_{p(\mathrm{f} \mid u)}\left[(\mathrm{f}-\mu_{\mathrm{f}\mid u})^t (\mathrm{f}-\mu_{\mathrm{f}\mid u}) +2\mu_{\mathrm{f}\mid u}^t f - \mu_{\mathrm{f}\mid u}^t\mu_{\mathrm{f}\mid u}\right]   -2 y^t \mu_{\mathrm{f}\mid u}\right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( y^t y + \mathbb{E}_{p(\mathrm{f} \mid u)}\left[(\mathrm{f}-\mu_{\mathrm{f}\mid u})^t (\mathrm{f}-\mu_{\mathrm{f}\mid u}) \right] +2\mu_{\mathrm{f}\mid u}^t \mathbb{E}_{p(\mathrm{f} \mid u)}\left[f\right] - \mu_{\mathrm{f}\mid u}^t\mu_{\mathrm{f}\mid u}   -2 y^t \mu_{\mathrm{f}\mid u}\right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( y^t y + \mathrm{trace}(\operatorname{Cov}_{\mathrm{f}\mid u})  + \mu_{\mathrm{f}\mid u}^t\mu_{\mathrm{f}\mid u}   -2 y^t \mu_{\mathrm{f}\mid u}\right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left\| y - \mu_{\mathrm{f}\mid u}\right\|^2 + \frac{-1}{2\sigma^{2}}\left(\mathrm{trace}(\operatorname{Cov}_{\mathrm{f}\mid u}) \right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \exp\left(\frac{-1}{2\sigma^{2}}\left\| y - \mu_{\mathrm{f}\mid u}\right\|^2\right)\right) + \frac{-1}{2\sigma^{2}}\left(\mathrm{trace}(\operatorname{Cov}_{\mathrm{f}\mid u}) \right) \\
&= \log \left(\mathcal{N}\left(y \mid \mu_{\mathrm{f}\mid u}, \sigma^2 I\right)\right) - \frac{1}{2\sigma^{2}}\left(\mathrm{trace}(\operatorname{Cov}_{\mathrm{f}\mid u}) \right)
\end{aligned}
$$

If we now substitute the value of the mean and covariance of $p(\mathrm{f} \mid u)$ (This is the marginal distribution of equation $\text{(CD)}$). We have:

$$
\mathcal{L}_1 =\log \left(\mathcal{N}\left(y \mid K_{\mathrm{f}u}K_{uu}^{-1}u, \sigma^2 I\right)\right) - \tfrac{1}{2\sigma^{2}}\left(\mathrm{trace}(K_{\mathrm{f}\mathrm{f}}-Q_{\mathrm{f}\mathrm{f}}) \right)

$$

<a onclick="$('#l1_integral').slideToggle();"> Collapse </a>

</div>

$$
\begin{aligned}
\mathcal{L}_1 &=\log \left(\mathcal{N}\left(y \mid K_{\mathrm{f}u}K_{uu}^{-1}u, \sigma^2 I\right)\right) - \tfrac{1}{2\sigma^{2}}\left(\mathrm{trace}(K_{\mathrm{f}\mathrm{f}}-Q_{\mathrm{f}\mathrm{f}}) \right) \\
&= \sum_{i=1}^N \left[\log\mathcal{N}\left(y_i \mid K_{\mathrm{f}_iu}K_{uu}^{-1}u, \sigma^2\right) - \tfrac{1}{2\sigma^{2}}\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right)\right] \\
&= \sum_{i=1}^N \left[-\tfrac{1}{2\sigma^2}(y_i - K_{\mathrm{f}_iu}K_{uu}^{-1}u)^2  -\tfrac{M}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right)\right] \\
\end{aligned}
$$

It's worth to look at this $$\mathcal{L}_1$$ equation. First this equiation is a sum over the training data, this indicates that we could probably optimize $$\mathcal{L}(q)$$ using  **stochastic** gradient descent. Then this equation looks like a likelihood equation: for each point $y_i$ we want it to be close to its natural prediction given $u$ ($$K_{\mathrm{f}_iu}K_{uu}^{-1}u$$) but also, at the same time, to have low expected variance given $u$ ($$K_{\mathrm{f}_i\mathrm{f}_i}-K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}$$).

### Plug in $\mathcal{L}_1$ into $\mathcal{L}(q)$

$$
\begin{aligned}
\mathcal{L}(q) &= \mathbb{E}_{q(u)}[\mathcal{L}_1] - KL\left[q(u)\mid\mid p(u)\right] \\
&= \mathbb{E}_{q(u)}\left[\sum_{i=1}^N \left[\log\mathcal{N}\left(y_i \mid K_{\mathrm{f}_iu}K_{uu}^{-1}u, \sigma^2\right) - \tfrac{1}{2\sigma^{2}}\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right)\right]\right] - KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N \mathbb{E}_{q(u)}\left[\log\mathcal{N}\left(y_i \mid K_{\mathrm{f}_iu}K_{uu}^{-1}u, \sigma^2\right) - \tfrac{1}{2\sigma^{2}}\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right)\right] - KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N \mathbb{E}_{q(u)}\left[-\tfrac{1}{2\sigma^2}(y_i - K_{\mathrm{f}_iu}K_{uu}^{-1}u)^2  -\tfrac{M}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right)\right] - KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\mathbb{E}_{q(u)}\left[(y_i - K_{\mathrm{f}_iu}K_{uu}^{-1}u)^2\right]  -\tfrac{NM}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right) - KL\left[q(u)\mid\mid p(u)\right]
\end{aligned}
$$

So this shows that the $\mathcal{L}(q)$ bound is a sum over the training data!! Therefore it is possible that it can be optimized by **stochastic** gradient descent.

### Assume $q(u)$ is normally distributed

If we assume now that $q(u)$ is normal: $q(u)=\mathcal{N}(u \mid m, S)$, the bound $\mathcal{L}(q)$ can now be written as $\mathcal{L}(m,S)$. If use the change of variables $u = m + R\epsilon$ (where $S=RR^t$) with $\epsilon \sim \mathcal{N(0,I)}$ M-dimensional vector we can compute all the integrals in $\mathcal{L}(q)$ to have

$$
\begin{aligned}
\mathcal{L}(m,S) &= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\mathbb{E}_{\epsilon}\left[(y_i - K_{\mathrm{f}_iu}K_{uu}^{-1}(m + R\epsilon))^2\right]-\tfrac{NM}{2}\log(2\pi\sigma^2)+ \\ &\quad\quad\quad - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right) - KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\left(\mathbb{E}_{\epsilon}\left[(y_i - K_{\mathrm{f}_iu}K_{uu}^{-1}m)^2\right] + \mathbb{E}_{\epsilon}\left[(K_{\mathrm{f}_iu}K_{uu}^{-1}R\epsilon)^2\right] -\cancel{\mathbb{E}_{\epsilon}\left[2(y_i - K_{\mathrm{f}_iu}K_{uu}^{-1}m)R\epsilon \right]}\right)+ \\ &\quad\quad\quad -\tfrac{NM}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right) - KL\left[q(u)\mid\mid p(u)\right]\\
&= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\left((y_i - K_{\mathrm{f}_iu}K_{uu}^{-1}m)^2 + \mathbb{E}_{\epsilon}\left[\epsilon^tR^tK_{uu}^{-1}K_{u\mathrm{f}_i}K_{\mathrm{f}_iu}K_{uu}^{-1}R\epsilon\right] \right)+\\ &\quad\quad\quad-\tfrac{NM}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right) - KL\left[q(u)\mid\mid p(u)\right]\\
&= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\left((y_i - K_{\mathrm{f}_iu}K_{uu}^{-1}m)^2 + \operatorname{trace}(SK_{uu}^{-1}K_{u\mathrm{f}_i}K_{\mathrm{f}_iu}K_{uu}^{-1}) \right)+\\ &\quad\quad\quad-\tfrac{NM}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{\mathrm{f}_i\mathrm{f}_i}- K_{\mathrm{f}_iu}K_{uu}^{-1}K_{u\mathrm{f}_i}\right) - KL\left[q(u)\mid\mid p(u)\right]\\
\end{aligned}
$$

This equation is the $\mathcal{L}_3$ equation that appears in [[Hensman et. al 2013]](http://arxiv.org/abs/1309.6835){:target="_blank"}. We can fully expand it adding the value of the Kullback-Leibler divergence between the approximating posterior and the prior over $u$ since both of them are multivariate normal distributions [the wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions){:target="_blank"} says it is:

$$
\begin{aligned}
KL[q(u) \mid\mid p(u)] &= KL[\mathcal{N}(m,S)\mid\mid \mathcal{N}(0,K_{uu})] \\
&= \tfrac{1}{2}m^t K_{uu}^{-1}m - \tfrac{k}{2} + \tfrac{1}{2} \operatorname{trace}(K_{uu}^{-1}S) + \tfrac{1}{2}\log\left(\frac{\operatorname{det}K_{uu}}{\operatorname{det}S}\right)
\end{aligned}
$$

Some comments about the full $\mathcal{L}(m,S)$ equation:

1. $\mathcal{L}(m,S)$ has as variational parameters $m$, $S$, the pseudo-inputs $X_u$, $\sigma^2$ and the parameters of the kernel! (I don't feel very comfortable with the three later ($X_u$,$\sigma^2$ and the params of the kernel).)
1. The proposal of [[Hensman et. al 2013]](http://arxiv.org/abs/1309.6835){:target="_blank"} is to minimize this equation w.r.t. all this parameters using stochastic gradient descent. (Actually I think they do not optimize the pseudo-inputs $X_u$ in their experiments).
1. The mean of the $q(u)$ distribution is all we need to compute the mean of the approximate predictive distribution.
1. The dependencies of $m$ are only in two terms: in the first term of $\mathcal{L}(m,S)$ and in the first term of the Kullback-Leibler divergence: $\tfrac{-1}{2} m^t K_{uu}^{-1}m$.
1. The dependencies on $S$ are in the second term of $\mathcal{L}(m,S)$ and in the two latest terms of the Kullback-Leibler divergence.

### Maximize $\mathcal{L}(m,S)$ w.r.t. the variational parameters $m$ and $S$

#### Maximize w.r.t. $m$

Let's do the change of variables $\alpha = K_{uu}^{-1}m$. The $\mathcal{L}(m,S)$ bound only depends on $\alpha$ on two terms:

$$
\mathcal{L}(m,S) = \mathcal{L}(\alpha,S) = \overbrace{\tfrac{-1}{2\sigma^2}\|y-K_{\mathrm{f}u}\alpha\|^2 - \tfrac{1}{2}\alpha^t K_{uu} \alpha}^{J(\alpha)} + J(S) + \mathrm{const.}
$$

Therefore the derivative of $\mathcal{L}(\alpha,S)$ w.r.t. $\alpha$ is:

$$
\nabla_{\alpha} \mathcal{L}(\alpha,S) = \tfrac{1}{\sigma^2} K_{u\mathrm{f}}\left(y - K_{\mathrm{f}u}\alpha\right)-K_{uu}\alpha
$$

Seting $\nabla_\alpha \mathcal{L}(\alpha,S) = 0$ leads to the optimal $\alpha$ that we will call $\alpha^\star$:

$$
\alpha^\star =  \left(K_{u\mathrm{f}}K_{\mathrm{f}u} + \sigma^2 K_{uu}\right)^{-1}K_{u\mathrm{f}}y
$$

But... wait.. This equation is our beloved Nyström solution that we talk about in [this blog post]({{site.baseurl}}/blog/2017/10/24/NystromRBFNN){:target="_blank"}. (We just have to apply matrix inversion lemma to the latest equation). In addition, if we plug this $\alpha^\star$ in the approximate prediction distribution equation $\text{(APD)}$ of above we see that the mean of the approximate predictive distribution is $K_{\star u}\alpha^\star$!

Now we want to see what happen if we plug the optimal $\alpha^\star$ the bound: $\mathcal{L}(\alpha^\star,S)$. Since the bound only dependence on $\alpha$ is on $J(\alpha)$ we just have to compute $J(\alpha^\star)$. If you want to see the full derivation of $J(\alpha)$ <a onclick="$('#jalfa').slideToggle();"> Click here </a>

<div id="jalfa" class="input_hidden" markdown="1">
Let's split $J(\alpha)$ in the following two terms:

$$
J(\alpha^\star) = \tfrac{-1}{2\sigma^2}\overbrace{\|y-K_{\mathrm{f}u}\alpha\|^2}^{\operatorname{Err}(\alpha)} - \tfrac{1}{2}\overbrace{\alpha^t K_{uu} \alpha}^{\operatorname{Reg}(\alpha)}
$$

Then first the error:

$$
\begin{aligned}
\operatorname{Err}(\alpha^\star)  &= \|K_{\mathrm{f}u} \alpha^\star - y \|^2 \\
 &= \Big\|\overbrace{K_{\mathrm{f}u}K_{uu}^{-1}K_{u\mathrm{f}}}^{Q_{\mathrm{ff}}} \Big(Q_{\mathrm{ff}} + \sigma^2  I\Big)^{-1}y - y \Big\|^2  \\
 &=\Big\|\Big(Q_{\mathrm{ff}}+\sigma^2I -\sigma^2I\Big) \Big(Q_{\mathrm{ff}} + \sigma^2  I\Big)^{-1}y - y \Big\|^2  \text{ Trick add and substract }\sigma^2 I\\
  &=\Big\|\cancel{\Big(Q_{\mathrm{ff}}+\sigma^2I\Big)\Big(Q_{\mathrm{ff}} + \sigma^2  I\Big)^{-1}y}  -\sigma^2I \Big(Q_{\mathrm{ff}} + \sigma^2  I\Big)^{-1}y - \cancel{y} \Big\|^2  \text{ Trick add and substract }\sigma^2 I\\
 &= \sigma^4\left\| \left(Q_{\mathrm{ff}} + \sigma^2  I\right)^{-1}y  \right\|^2 \\
\end{aligned}
$$

Then the regularizer:

$$
\begin{aligned}
\operatorname{Reg}(\alpha^\star) &= \alpha^{\star t} K_{uu}\alpha^\star \\
&=  y^t\left(Q_{\mathrm{ff}} + \sigma^2 I\right)^{-1}K_{\mathrm{f}u}K_{uu}^{-1}\cancel{K_{uu}}\cancel{K_{uu}^{-1}}K_{u\mathrm{f}}\left(Q_{\mathrm{ff}} + \sigma^2 I\right)^{-1} y \\
&=  y^t\left(Q_{\mathrm{ff}} + \sigma^2 I\right)^{-1}\left(Q_{\mathrm{ff}}+\sigma^2I-\sigma^2I\right)\left(Q_{\mathrm{ff}} + \sigma^2 I\right)^{-1} y \\
&=  y^t\cancel{\left(Q_{\mathrm{ff}} + \sigma^2I\right)^{-1} \left(Q_{\mathrm{ff}}+\sigma^2I\right)} \left(Q_{\mathrm{ff}} + \sigma^2 I\right)^{-1} y - y^t\left(Q_{\mathrm{ff}} + \sigma^2I\right)^{-1}(\sigma^2I)\left(Q_{\mathrm{ff}} + \sigma^2 I\right)^{-1} y \\
&= y^t(Q_{\mathrm{ff}}+\sigma^2 I)^{-1} y -\sigma^2\|(Q_{ff}+\sigma^2I)^{-1}y\|^2
\end{aligned}
$$

So summing it up we get:

$$
\begin{aligned}
J(\alpha^\star) &= \cancel{\tfrac{-1}{2\sigma^2}\sigma^4\left\| \left(Q_{\mathrm{ff}} + \sigma^2  I\right)^{-1}y  \right\|^2} -\tfrac{1}{2}\left(y^t(Q_{\mathrm{ff}}+\sigma^2 I)^{-1} y -\cancel{\sigma^2\|(Q_{ff}+\sigma^2I)^{-1}y\|^2}\right)\\
&=-\tfrac{1}{2}y^t(Q_{\mathrm{ff}}+\sigma^2 I)^{-1} y
\end{aligned}
$$

<a onclick="$('#jalfa').slideToggle();"> Collapse </a>
</div>

The final equation we get is:

$$
J(\alpha^\star) = \tfrac{-1}{2} y^t \left(Q_{\mathrm{f}\mathrm{f}} + \sigma^2 I\right)^{-1} y
$$


<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
