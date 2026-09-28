---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat results">Results</span> The product is right more often than any QPU alone

<style scoped>figure.figure { margin: 0.2em auto 0.5em; } li { margin-bottom: 0.3em; }</style>

<figure class="figure">

![w:1120](media/readout-trained.png)

</figure>

- **Rescues:** correct signals with at least two QPUs wrong alone: $196$ of $354$ (NC), $82$ of $372$ (CC)
- **With communication,** QPUs 0 and 2 improve most: the message bits bring them a summary of feature block 1
