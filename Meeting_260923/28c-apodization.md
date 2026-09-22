---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
section { font-size: 20px; }
p { margin: 0.25em 0; }
figure.figure { margin: 0.1em auto; }
</style>

# <span class="cat strategy">Strategy</span> Apodization

The side lobes come from the sharp edges of the boxcar. If we try $w_k=\sin^{2}\!\big(\pi(k+1)/(N+1)\big)$, a Hann window, the kernel becomes smooth, which gives lobes that fall away fast. For $N=8$ the largest drops from $5.3\%$ to $0.07\%$ of the peak. 

<figure class="figure">

![w:1000](media/taps-apodization.png)

</figure>
