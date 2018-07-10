---
layout: post
title: Accuracy measures
date: 2017-12-22
author: Gonzalo Mateo-García
---
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
      //TeX: { equationNumbers: { autoNumber: "False" } }
    });
  /*  MathJax.Hub.Queue(
  ["resetEquationNumbers",MathJax.InputJax.TeX],
  ["PreProcess",MathJax.Hub],
  ["Reprocess",MathJax.Hub]
);*/
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>


The confusion matrix $(i,j)$-element is the number of samples known to be $i$ that are predicted $j$. When the problem is binary the confusion matrix is $2 \times 2$ and its elements are normally represented in a table like the following:

| | | Predicted | | Total |
|- |-| ----|----- | |
|**Real** | | 0 | 1 |  |
| | 0 | TN | FP | $N$ |
| | 1 | FN | TP | $P$ |
|Total |  | $\hat{N}$  | $\hat{P}$ | total |

### Basic values

* $TN$ - True negatives (pred negative real negative)
* $FP$ - False positives (pred positive real negative)
* $FN$ - False negatives (pred negative real positive)
* $TP$ - True positives (pred positive real positive)

### Added values

We compute them by adding by rows or cols in the confusion matrix:

* $N = TN + FP$ number of **real** negatives.
* $P = FN + TP$ number of **real** positives.
* $\hat{N} =TN + FN$ number of **predicted** negatives.
* $\hat{P} =FP + TP$ number of **predicted** positives.
* $\text{total} = N + P = \hat{N}+ \hat{P} = TN+FP+FN+TP$ number of samples.

### Rates

* **PR** positive rate.

$$
PR = \frac{P}{\text{total}} = \frac{P}{P+N}
$$

* **NR** negative rate.

$$
NR = \frac{N}{\text{total}} = \frac{N}{P+N} = 1-PR

$$

* **TPR** - true positive rate. Also called **recall** or **sensitivity**.

$$
TPR = \frac{TP}{P} = \frac{TP}{TP + FN}
$$

* **FPR** - false positive rate. Also called **commission error** or **type I error**

$$
FPR = \frac{FP}{N} = \frac{FP}{FP + TN}
$$

* __FNR__ - false negative rate. Also called **omission error** or **type II error**

$$
FNR = \frac{FN}{P} = \frac{FN}{TP + FN} = 1 - TPR
$$

* __TNR__ - true negative rate. Also called **specificity**

$$
TNR = \frac{TN}{N} = \frac{TN}{TN + FP} = 1 - FPR
$$

* **Precision**. fraction of correct predicted positive:

$$
\text{Precision} = \frac{TP}{\hat{P}} = \frac{TP}{FP + TP}
$$

* **Accuracy**: correct predicted (it is the mean of __TPR__ and __TNR__ if the set is balanced which means __PR__=0.5 )

$$
\begin{aligned}
\text{Accuracy} &= \frac{TP+TN}{total} = \frac{TP+TN}{TN+FP+FN+TP} \\
&= \frac{TP}{P}\frac{P}{total}+\frac{TN}{N}\frac{N}{total} = TPR \cdot PR + TNR \cdot (1-PR) = TPR \cdot PR + (1-FPR) \cdot (1-PR)
\end{aligned}
$$

* $F_1$ __score__. Harmonic mean of precision and recall.

$$
F_1 = \frac{2}{\tfrac{1}{\text{recall}} + \tfrac{1}{\text{precision}}}
$$
