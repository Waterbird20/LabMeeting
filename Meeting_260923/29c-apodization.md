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

The side lobes come from the sharp edges of the boxcar, since the transform of a step decays only as $1/\omega$. So taper the taps instead, $w_k=\sin^{2}\!\big(\pi(k+1)/(N+1)\big)$, a Hann window. The kernel is the transform of the taps, so a smooth window gives lobes that fall away fast: for $N=8$ the largest drops from $5.3\%$ to $0.07\%$ of the peak. The price comes from the same transform. A tapered list is effectively shorter, so the main lobe is $1.4$ times wider. Apodization trades resolution for a clean baseline. It never makes a line narrower, and only more units can.

<figure class="figure">

![w:900](media/taps-apodization.png)

*Boxcar and Hann taps on the same eight units, their kernels in dB, and a strong line with a weak neighbour at $5\%$ of its strength. With the boxcar the side lobes of the strong line are as tall as the weak line, and with Hann only the real line remains. This is the first way to design the kernel, excluding what we do not want. Chapter 3 keeps the picture, with the pulses as the taps, and asks for the response we do want, exactly.*

</figure>
