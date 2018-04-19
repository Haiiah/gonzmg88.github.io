---
layout: post
title: Variational Free Energy
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

Let $$f=\{f^\star,\mathrm{f}\}$$ be the noise free latent variables of a $\mathcal{GP}$. We write $\mathrm{f}$ to be the latent variables where we have noisy observations $y$: $$y_i = \mathrm{f}_i + \epsilon_i$$ ($$\epsilon_i \sim \mathcal{N}(0,\sigma^2) $$) and $f^\star$ the points where we want to have predictions. For the purpose of understanding the Variational Free Energy (VFE) approach to $\mathcal{GP}$s of [[Titsias 2009]](https://pdfs.semanticscholar.org/db7b/e492a629a98db7f9d77d552fd3568ff42189.pdf) we will consider the **joint posterior** $p(f^\star,\mathrm{f} \mid y)$.  This joint posterior is a _strange_ object we normally do not consider if we are working with $\mathcal{GP}$s. When we work with $\mathcal{GP}$s we normally care about the **predictive posterior** $p(y^\star\mid y)$ (to make predictions) or the **marginal likelihood** $p(y)$ (to fit the hyperparameters of the kernel and the $\sigma^2$).

For the _normal_ $\mathcal{GP}$ this **joint posterior** is:

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
* _The only difference between this noise free predictive posterior and the "standard" predictive posterior ($$p(y^\star \mid y) $$ is that we have to add $\sigma^2 I $ to the variance of the later)_

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

Since this is a $\mathcal{GP}$ $p(f) = \mathcal{N}(f \mid 0,K_{ff})$ so $p(f^\star, \mathrm{f} \mid u)$ can be easily computed using Gaussian identities ([wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions)). In this case we see that $p(f^\star, \mathrm{f} \mid u)$ is the standard posterior of the $\mathcal{GP}$ with noise free variables $u$:

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

1. We could also rewrite the true posterior $p(f\mid y)$ using the subset $u$ of $f$:

$$
p(f\mid y) = p(f^\star,\mathrm{f},u \mid y) = p(f^\star, \mathrm{f} \mid u,y)p(u \mid y)
$$

1. If we compare this two equations we see that **The Titsias approximation removes the dependency on the data ($y$) in the first term**. (While the second term $q(u)$ is let free).

1. Again, in order to make predictions, we will need the **approximate (noise free) predictive distribution** $q(f^\star)$, thus we have to integrate out $\mathrm{f}$ and $u$ from $q(f)$:

$$
\begin{aligned}
q(f^\star) &= \int \int p(f^\star,\mathrm{f} \mid u) q(u) d\mathrm{f}du \\
 &= \int q(u) \underbrace{\int p(f^\star,\mathrm{f} \mid u)  d\mathrm{f}}_{p(f^\star \mid u)}du \\
 &= \int q(u) \mathcal{N}(f^\star \mid K_{\star u}K_{uu}^{-1}u, K_{\star\star}-Q_{\star\star})du
\end{aligned}
$$

1. If we are *"lucky"* and $q$ is normal, $q(u)=\mathcal{N}(u \mid m, S)$, then the **approximate predictive distribution** can be computed using the Bishop Chapter 2 pág 93 trick of Marginal and Conditional Gaussians:

$$
\begin{aligned}
q(f^\star) = \mathcal{N}(f^\star &\mid K_{\star u}K_{uu}^{-1}m, \\
&K_{\star\star}-Q_{\star\star}+ K_{\star u}K_{uu}^{-1}S K_{uu}^{-1}K_{u\star})
\end{aligned}
$$

1. Substituting this $q$ in the bound $\mathcal{L}(q)$ leads to:

$$
\begin{aligned}
\require{cancel}
\mathcal{L}(q) &= \mathbb{E}_{q(f)}\left[\log \frac{p(y\mid \mathrm{f})\cancel{p(f^\star, \mathrm{f} \mid u)} p(u)}{\cancel{p(f^\star, \mathrm{f} \mid u)}q(u)}\right] \\
&= \mathbb{E}_{q(f)}\left[\log p(y\mid \mathrm{f})\right] + \mathbb{E}_{q(f)}\left[\log \frac{p(u)}{q(u)}\right] \\
&= \mathbb{E}_{p(f^\star, \mathrm{f} \mid u)q(u)}\left[\log p(y\mid \mathrm{f})\right] + \mathbb{E}_{p(f^\star, \mathrm{f} \mid u)q(u)}\left[\log \frac{p(u)}{q(u)}\right] \\
&= \mathbb{E}_{p(f^\star \mid \mathrm{f} , u)p(\mathrm{f} \mid u)q(u)}\left[\log p(y\mid \mathrm{f})\right] + \cancel{\mathbb{E}_{p(f^\star, \mathrm{f} \mid u)}}\mathbb{E}_{q(u)}\left[\log \frac{p(u)}{q(u)}\right] \\
&= \cancel{\mathbb{E}_{p(f^\star \mid \mathrm{f} , u)}}\mathbb{E}_{q(u)}\left[\mathbb{E}_{p(\mathrm{f} \mid u)}\log p(y\mid \mathrm{f})\right] + \mathbb{E}_{q(u)}\left[\log \frac{p(u)}{q(u)}\right] \\
&= \mathbb{E}_{q(u)}\left[\mathbb{E}_{p(\mathrm{f} \mid u)}\log p(y\mid \mathrm{f})\right] + KL\left[q(u) \mid\mid p(u) \right] \\
\end{aligned}
$$

The integral $\mathbb{E}_{p(\mathrm{f} \mid u)}\left[\log p(y \mid \mathrm{f})\right]$ plays a central role in the derivation. We will call it $\mathcal{L}_1$. Let's try to compute it:

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

Now the trick is to use that $p(\mathrm{f} \mid u)$ is normal, if we write: $\mathcal{N}(\mathrm{f} \mid \mu_{\mathrm{f}\mid u}, \operatorname{Cov}_{u\mid \mathrm{f}})$

$$
\begin{aligned}
\mathcal{L}_1 &= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( y^t y + \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\mathrm{f}^t \mathrm{f}\right]   -2 y^t \mu_{\mathrm{f}\mid u}\right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( y^t y + \mathbb{E}_{p(\mathrm{f} \mid u)}\left[(\mathrm{f}-\mu_{\mathrm{f}\mid u})^t (\mathrm{f}-\mu_{\mathrm{f}\mid u}) +2\mu_{\mathrm{f}\mid u}^t f - \mu_{\mathrm{f}\mid u}^t\mu_{\mathrm{f}\mid u}\right]   -2 y^t \mu_{\mathrm{f}\mid u}\right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( y^t y + \mathbb{E}_{p(\mathrm{f} \mid u)}\left[(\mathrm{f}-\mu_{\mathrm{f}\mid u})^t (\mathrm{f}-\mu_{\mathrm{f}\mid u}) \right] +2\mu_{\mathrm{f}\mid u}^t \mathbb{E}_{p(\mathrm{f} \mid u)}\left[f\right] - \mu_{\mathrm{f}\mid u}^t\mu_{\mathrm{f}\mid u}   -2 y^t \mu_{\mathrm{f}\mid u}\right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left( y^t y + \mathrm{trace}(\operatorname{Cov}_{u\mid \mathrm{f}})  + \mu_{\mathrm{f}\mid u}^t\mu_{\mathrm{f}\mid u}   -2 y^t \mu_{\mathrm{f}\mid u}\right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \right) + \frac{-1}{2\sigma^{2}}\left\| y - \mu_{\mathrm{f}\mid u}\right\|^2 + \frac{-1}{2\sigma^{2}}\left(\mathrm{trace}(\operatorname{Cov}_{u\mid \mathrm{f}}) \right) \\
&= \log \left( \left(2\pi \sigma^{2N}\right)^{-1/2} \exp\left(\frac{-1}{2\sigma^{2}}\left\| y - \mu_{\mathrm{f}\mid u}\right\|^2\right)\right) + \frac{-1}{2\sigma^{2}}\left(\mathrm{trace}(\operatorname{Cov}_{u\mid \mathrm{f}}) \right) \\
&= \log \left(\mathcal{N}\left(y \mid \mu_{\mathrm{f}\mid u}, \sigma^2 I\right)\right) - \frac{1}{2\sigma^{2}}\left(\mathrm{trace}(\operatorname{Cov}_{u\mid \mathrm{f}}) \right)
\end{aligned}
$$

If we now substitute the value of the mean and covariance of $p(\mathrm{f} \mid u)$ (This is the marginal distribution of equation $\text{(CD)}$).

$$
\begin{aligned}
\mathcal{L}_1 &=\log \left(\mathcal{N}\left(y \mid K_{fu}K_{uu}^{-1}u, \sigma^2 I\right)\right) - \tfrac{1}{2\sigma^{2}}\left(\mathrm{trace}(K_{\mathrm{f}\mathrm{f}}-Q_{\mathrm{f}\mathrm{f}}) \right) \\
&= \sum_{i=1}^N \left[\log\mathcal{N}\left(y_i \mid K_{f_iu}K_{uu}^{-1}u, \sigma^2\right) - \tfrac{1}{2\sigma^{2}}\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right)\right] \\
&= \sum_{i=1}^N \left[-\tfrac{1}{2\sigma^2}(y_i - K_{f_iu}K_{uu}^{-1}u)^2  -\tfrac{M}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right)\right] \\
\end{aligned}
$$

Great! So now it we get back the $\mathcal{L}(q)$ bound we will see it is a sum over the training data!! Therefore it is possible that it can be optimized by **stochastic** gradient descent:

$$
\begin{aligned}
\mathcal{L}(q) &= \mathbb{E}_{q(u)}[\mathcal{L}_1] + KL\left[q(u)\mid\mid p(u)\right] \\
&= \mathbb{E}_{q(u)}\left[\sum_{i=1}^N \left[\log\mathcal{N}\left(y_i \mid K_{f_iu}K_{uu}^{-1}u, \sigma^2\right) - \tfrac{1}{2\sigma^{2}}\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right)\right]\right] + KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N \mathbb{E}_{q(u)}\left[\log\mathcal{N}\left(y_i \mid K_{f_iu}K_{uu}^{-1}u, \sigma^2\right) - \tfrac{1}{2\sigma^{2}}\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right)\right] + KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N \mathbb{E}_{q(u)}\left[-\tfrac{1}{2\sigma^2}(y_i - K_{f_iu}K_{uu}^{-1}u)^2  -\tfrac{M}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right)\right] + KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\mathbb{E}_{q(u)}\left[(y_i - K_{f_iu}K_{uu}^{-1}u)^2\right]  -\tfrac{NM}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right) + KL\left[q(u)\mid\mid p(u)\right]
\end{aligned}
$$

If we assume now that $q(u)$ is normal: $q(u)=\mathcal{N}(u \mid m, S)$ we can use the change of variables $u = m + R\epsilon$ (where $S=RR^t$) with $\epsilon \sim \mathcal{N(0,I)}$ M-dimensional vector to have:

$$
\begin{aligned}
\mathcal{L}(q) &= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\mathbb{E}_{\epsilon}\left[(y_i - K_{f_iu}K_{uu}^{-1}(m + R\epsilon))^2\right]-\tfrac{NM}{2}\log(2\pi\sigma^2)+ \\ &\quad\quad\quad - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right) + KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\left(\mathbb{E}_{\epsilon}\left[(y_i - K_{f_iu}K_{uu}^{-1}m)^2\right] + \mathbb{E}_{\epsilon}\left[(K_{f_iu}K_{uu}^{-1}R\epsilon)^2\right] -\cancel{\mathbb{E}_{\epsilon}\left[2(y_i - K_{f_iu}K_{uu}^{-1}m)R\epsilon \right]}\right)+ \\ &\quad\quad\quad -\tfrac{NM}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right) + KL\left[q(u)\mid\mid p(u)\right]\\
&= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\left((y_i - K_{f_iu}K_{uu}^{-1}m)^2 + \mathbb{E}_{\epsilon}\left[\epsilon^tR^tK_{uu}^{-1}K_{uf_i}K_{f_iu}K_{uu}^{-1}R\epsilon\right] \right)+\\ &\quad\quad\quad-\tfrac{NM}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right) + KL\left[q(u)\mid\mid p(u)\right]\\
&= \sum_{i=1}^N -\tfrac{1}{2\sigma^2}\left((y_i - K_{f_iu}K_{uu}^{-1}m)^2 + \operatorname{trace}(SK_{uu}^{-1}K_{uf_i}K_{f_iu}K_{uu}^{-1}) \right)+\\ &\quad\quad\quad-\tfrac{NM}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^{2}}\sum_{i=1}^N\left(K_{f_if_i}- K_{f_iu}K_{uu}^{-1}K_{uf_i}\right) + KL\left[q(u)\mid\mid p(u)\right]\\
\end{aligned}
$$

This equation is the $\mathcal{L}_3$ equation that appears in [[Hensman et. al 2013]](http://arxiv.org/abs/1309.6835). We can fully expand it adding the value of the Kullback-Leibler divergence between the approximating posterior and the prior over $u$ since both of them are multivariate normal distributions [the wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions) says it is:

$$
\begin{aligned}
KL[q(u) \mid\mid p(u)] &= KL[\mathcal{N}(m,S)\mid\mid \mathcal{N}(0,K_{uu})] \\
&= \tfrac{1}{2}m^t K_{uu}^{-1}m - \tfrac{k}{2} + \tfrac{1}{2} \operatorname{trace}(K_{uu}^{-1}S) + \tfrac{1}{2}\log\left(\frac{\operatorname{det}K_{uu}}{\operatorname{det}S}\right)
\end{aligned}
$$

Finally it is worth to have a look at the full equation $\mathcal{L}(q)$ and see its dependencies:

1. $\mathcal{L}(q)$ has as variational parameters $m$, $S$, the pseudo-inputs $X_u$, $\sigma^2$ and the parameters of the kernel! (I don't feel very comfortable with the three later ($X_u$,$\sigma^2$ and the params of the kernel).  )
1. The mean of the $q(u)$ distribution is all we need to compute the mean of the approximate predictive distribution.
1. The dependencies of $m$ are only in the first and last term.
1. I still want to write down the optimal $m$ to see why we reach back the ubiquitous Nyström solution for the predictive mean.

<!--
### Derivation from [[Hensman et. al 2013]](http://arxiv.org/abs/1309.6835)

Let's go back again  and see the things using [[Hensman et. al 2013]](http://arxiv.org/abs/1309.6835) derivation:
$$
\begin{aligned}
\log p(y) &= \log \mathbb{E}_{p(u)}\left[p(y \mid u)\right] \\
% &= \log \mathbb{E}_{p(u)}\left[\mathbb{E}_{p(\mathrm{f} \mid u)}\left[p(y\mid \mathrm{f},\cancel{u})\right]\right] \\
\end{aligned}
$$

We can decompose $\log p(y)$ using also the standard trick of variational inference:
$$
\log p(y) = \mathbb{E}_{p(u)}\left[\log p(y\mid u)\right] -  \mathbb{E}_{p(u)}\left[\log \frac{p(u \mid y)}{p(u)}\right]
$$

Instead of computing the bound on $p(y)$, we can compute the bound on $p(y \mid u)$ which is:

$$
\log p(y \mid u) = \overbrace{\mathbb{E}_{q(f^\star,\mathrm{f} \mid u)}\left[\log \frac{p(y,f^\star,\mathrm{f} \mid u)}{q(f^\star,\mathrm{f} \mid u)}\right]}^{\mathcal{L}\left(q(f^\star,\mathrm{f} \mid u)\right)}  \overbrace{-\mathbb{E}_{q(f^\star,\mathrm{f} \mid u)}\left[\log \frac{p(f^\star,\mathrm{f} \mid y,u)}{q(f^\star,\mathrm{f} \mid u)}\right]}^{KL\left(q(f^\star,\mathrm{f} \mid u) \mid \mid p(f^\star,\mathrm{f} \mid u,y)\right)}
$$

If we use here the Titsias approximation $q(f^\star,\mathrm{f} \mid u) = p(f^\star,\mathrm{f} \mid u)$. We find that:

1. The first term is our beloved $\mathcal{L}_1$ integral:
$$
\mathcal{L}\left(p(f^\star,\mathrm{f} \mid u)\right) = \mathbb{E}_{p(\mathrm{f} \mid u)}\left[\log p(y \mid \mathrm{f})\right]= \mathcal{L}_1
$$
1. The second term is just the divergence between the prior and and the posterior:
$$
KL\left(p(f^\star,\mathrm{f} \mid u) \mid \mid p(f^\star,\mathrm{f} \mid u,y)\right)
$$

We can apply this decomposition to $\log p(y)$:
$$
\begin{aligned}
\log p(y) &= \log \mathbb{E}_{p(u)}\left[p(y \mid u)\right] \\
&= \log \mathbb{E}_{p(u)}\left[\exp\left(\mathcal{L}_1 + KL\left(p(f^\star,\mathrm{f} \mid u) \mid \mid p(f^\star,\mathrm{f} \mid u,y)\right)\right)\right] \\
&= \log \mathbb{E}_{p(u)}\left[\exp\left(\mathcal{L}_1\right)\exp\left(KL\left(p(f^\star,\mathrm{f} \mid u) \mid \mid p(f^\star,\mathrm{f} \mid u,y)\right)\right)\right] \\
\end{aligned}
$$

We can also apply this decomposition in the variational inference decomposition of $\log p(y)$:
$$
\begin{aligned}
\log p(y) &= \mathbb{E}_{p(u)}\left[\log p(y\mid u)\right] -  \mathbb{E}_{p(u)}\left[\log \frac{p(u \mid y)}{p(u)}\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1 + KL\left(p(f^\star,\mathrm{f} \mid u) \mid \mid p(f^\star,\mathrm{f} \mid u,y)\right) \right] -  \mathbb{E}_{p(u)}\left[\log \frac{p(u \mid y)}{p(u)}\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] + \mathbb{E}_{p(u)}\left[KL\left(p(f^\star,\mathrm{f} \mid u) \mid \mid p(f^\star,\mathrm{f} \mid u,y)\right) \right] -  \mathbb{E}_{p(u)}\left[\log \frac{p(u \mid y)}{p(u)}\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] -\mathbb{E}_{p(u)}\left[\mathbb{E}_{p(f^\star,\mathrm{f} \mid u)}\left[\log \frac{p(f^\star,\mathrm{f} \mid y,u)}{p(f^\star,\mathrm{f} \mid u)}\right]\right] - \mathbb{E}_{p(u)}\left[\log \frac{p(u \mid y)}{p(u)}\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] -\mathbb{E}_{p(u)}\left[\mathbb{E}_{p(f^\star,\mathrm{f} \mid u)}\left[\log \frac{p(f^\star,\mathrm{f} \mid y,u)}{p(f^\star,\mathrm{f} \mid u)}\right] +\log \frac{p(u \mid y)}{p(u)}\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] -\mathbb{E}_{p(u)}\left[\mathbb{E}_{p(f^\star,\mathrm{f} \mid u)}\left[\log \frac{p(f^\star,\mathrm{f} \mid y,u)}{p(f^\star,\mathrm{f} \mid u)} +\log \frac{p(u \mid y)}{p(u)}\right]\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] -\mathbb{E}_{p(u)}\left[\mathbb{E}_{p(f^\star,\mathrm{f} \mid u)}\left[\log \frac{p(f^\star,\mathrm{f} \mid y,u)}{p(f^\star,\mathrm{f} \mid u)} \frac{p(u \mid y)}{p(u)}\right]\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] -\mathbb{E}_{p(u)}\left[\mathbb{E}_{p(f^\star,\mathrm{f} \mid u)}\left[\log \frac{p(f^\star,\mathrm{f},u \mid y)}{p(f^\star,\mathrm{f} \mid u)\cancel{p(u \mid y)}} \frac{\cancel{p(u \mid y)}}{p(u)}\right]\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] -\mathbb{E}_{p(u)}\left[\mathbb{E}_{p(f^\star,\mathrm{f} \mid u)}\left[\log \frac{p(f^\star,\mathrm{f},u \mid y)}{p(f^\star,\mathrm{f}, u)}\right]\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] -\mathbb{E}_{p(f^\star,\mathrm{f}, u)}\left[\log \frac{p(f^\star,\mathrm{f},u \mid y)}{p(f^\star,\mathrm{f}, u)}\right] \\
&= \mathbb{E}_{p(u)}\left[\mathcal{L}_1\right] -KL\left(p(f^\star,\mathrm{f}, u) \mid\mid p(f^\star,\mathrm{f},u \mid y)\right)
\end{aligned}
$$

Things left: compute $\mathbb{E}_{p(u)}\left[\mathcal{L}_1\right]$. Get back the idea of substituting in the first expresion $q(u)=\mathcal{N}(m,S)$.


OLD STUFF:
$$
\begin{aligned}
\mathcal{L}(q) &= \mathbb{E}_{q(f)}\left[\log \prod_{i=1}^N p(y_i\mid \mathrm{f}_i)\right] + KL\left[q(u)\mid\mid p(u)\right] \quad \text{(step 6)} \\
&= \mathbb{E}_{q(\mathrm{f})}\left[\sum_{i=1}^N \log  p(y_i\mid \mathrm{f}_i)\right] + KL\left[q(u)\mid\mid p(u)\right]\\)
&= \sum_{i=1}^N \mathbb{E}_{q(\mathrm{f})}\left[\log  p(y_i\mid \mathrm{f}_i)\right] + KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N \mathbb{E}_{q(\mathrm{f})}\left[\frac{-1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(y_i- f_i)^2\right] + KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N \int \left[\frac{-1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(y_i- f_i)^2\right]q(\mathrm{f}_i)\overbrace{\int q(f_1,..,f_{i-1},f_{i+1},..,f_N \mid \mathrm{f}_i) d(\mathrm{f}_1,..,\mathrm{f}_{i-1},\mathrm{f}_{i+1},..,\mathrm{f}_N)}^{=1}d\mathrm{f}_i + KL\left[q(u)\mid\mid p(u)\right] \\
&= \sum_{i=1}^N \mathbb{E}_{q(\mathrm{f}_i)}\left[\frac{-1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(y_i- \mathrm{f}_i)^2\right]  + KL\left[q(u)\mid\mid p(u)\right]\\
&= \frac{-N}{2}\log(2\pi\sigma^2)+\sum_{i=1}^N \mathbb{E}_{q(\mathrm{f}_i)}\left[ - \frac{1}{2\sigma^2}(y_i- \mathrm{f}_i)^2\right]  + KL\left[q(u)\mid\mid p(u)\right] \\
&= \frac{-N}{2}\log(2\pi\sigma^2)+\sum_{i=1}^N \mathbb{E}_{q(u)}\left[\mathbb{E}_{q(\mathrm{f}_i \mid u)}\left[ - \frac{1}{2\sigma^2}(y_i- \mathrm{f}_i)^2\right]\right]  + KL\left[q(u)\mid\mid p(u)\right] \\
&= \frac{-N}{2}\log(2\pi\sigma^2)+\sum_{i=1}^N \mathbb{E}_{q(u)}\left[\mathbb{E}_{p(\mathrm{f}_i \mid u)}\left[ - \frac{1}{2\sigma^2}(y_i- \mathrm{f}_i)^2\right]\right]  + KL\left[q(u)\mid\mid p(u)\right] \\
&= \frac{-N}{2}\log(2\pi\sigma^2)+\sum_{i=1}^N \mathbb{E}_{q(u)}\left[\mathbb{E}_{p(\mathrm{f}_i \mid u)}\left[ - \frac{1}{2\sigma^2}(y_i^2+ \mathrm{f}_i^2 -2y_i \mathrm{f}_i)\right]\right]  + KL\left[q(u)\mid\mid p(u)\right] \\
&= \frac{-N}{2}\log(2\pi\sigma^2)+\sum_{i=1}^N \mathbb{E}_{q(u)}\left[- \frac{1}{2\sigma^2}\left(y_i^2+ \mathbb{E}_{p(\mathrm{f}_i \mid u)}[\mathrm{f}_i^2] -2y_i \mathbb{E}_{p(\mathrm{f}_i \mid u)}[\mathrm{f}_i]\right)\right]  + KL\left[q(u)\mid\mid p(u)\right] \quad \text{Using }p(\mathrm{f}_i \mid u ) = \mathcal{N}(K_{\mathrm{f}_iu}K_{uu}^{-1}u, K_{\mathrm{f}_i\mathrm{f}_i} - Q_{\mathrm{f}_i\mathrm{f}_i}) \\
&= \frac{-N}{2}\log(2\pi\sigma^2)+\sum_{i=1}^N \mathbb{E}_{q(u)}\left[- \frac{1}{2\sigma^2}\left(y_i^2+ K_{\mathrm{f}_i\mathrm{f}_i} - Q_{\mathrm{f}_i\mathrm{f}_i} + u^tK_{uu}^{-1}K_{u\mathrm{f}_i}K_{\mathrm{f}_iu}K_{uu}^{-1}u -2y_i K_{\mathrm{f}_iu}K_{uu}^{-1}u\right)\right]  + KL\left[q(u)\mid\mid p(u)\right]   \\
\end{aligned}
$$-->
