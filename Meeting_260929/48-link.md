---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat method">Method</span> The decision function: a stochastic threshold (probit)

<style scoped>.columns .col:first-child { flex: 0 0 560px; } p { margin: 0.3em 0; } ul { margin-top: 0.4em; }</style>

<div class="columns">
<div class="col">

$$g=a\,m_i+b\,m_j+c\,m_im_j+d$$

$$P(s{=}1)=\pi=\Phi\Big(g\big/\sqrt{\sigma_g^2+\epsilon^2}\Big)$$

<figure class="figure">

![w:560](media/link-cdf.png)

</figure>

</div>
<div class="col">

<figure class="figure">

<video src="media/link-rule.mp4" poster="media/link-rule.png" width="580" controls autoplay loop muted playsinline preload="none"></video>

</figure>

- $m_i,m_j$: outcome-$1$ probabilities from $K$ shots; shot noise $\sigma_g\propto K^{-1/2}$
- $\Phi$: standard normal CDF; $s=1$ applies $U_+$, else $U_-$
- Corners $\{0,1\}^2$: the sign of $g$ is a Boolean function (AND, OR, XOR)

</div>
</div>
