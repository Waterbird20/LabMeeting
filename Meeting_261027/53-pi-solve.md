---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

<style scoped>
.columns { gap: 1.0rem; margin-top: 0.1em; align-items: flex-end; }
.columns .col:first-child { flex: 0 0 700px; }
figure.figure { margin: 0; }
</style>

# <span class="cat strategy">Strategy</span> DQ $\pi$ pulse: proper $\tau$

At short $\tau$ the fixed-$\tau$ sequences are the shortest and most accurate; Newton iteration finds them in about ten steps.

<div class="columns">
<div class="col">

<figure class="figure">

![w:700](media/ch3-pi-solutions.png)

</figure>

</div>
<div class="col">

<figure class="figure">

![w:450](media/ch3-pi-newton.png)

</figure>

</div>
</div>

- Isolated solutions fix $\tau$ themselves and are longer: palindromic $N=4$ at 20 MHz takes $236$ ns ($\tau=20.1$ ns).
- Below $\tau\approx20$ ns the duration $T$ hardly depends on $\tau$; we keep $\tau=10$ ns.
