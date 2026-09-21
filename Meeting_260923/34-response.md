---
marp: true
theme: serif
math: mathjax
---

<!-- _class: dense -->

# <span class="cat method">Method</span> The response is the Fourier transform of the taps

The signal variable lives on the unit circle, $z=e^{i\delta\tau}$, so the polynomial evaluated along the experiment's sweep axis is a Fourier series in the detuning,
$$
\hat P(\delta)=\sum_{n=0}^{d}p_n\,e^{i\left(n-\frac{d}{2}\right)\delta\tau},
\qquad
|\hat P(\delta)|^{2}=\sum_{m=-d}^{d} r_m\,e^{im\delta\tau},
\qquad r_m=\sum_n p_{n+m}\,p_n^{*} .
$$
The measured curve is the transform of the **autocorrelation** of the tap list, which is the pointwise side of the one identity of Section 2: convolving the taps and multiplying the responses are the same statement read in two places. Two numbers follow immediately.

- The wait $\tau$ is the sampling interval, so the response repeats with period $2\pi/\tau$ in $\delta$. Choosing $\tau$ is choosing where the lines of the spectrum land on the circle.
- The total time $T=d\tau$ is the record length, so the narrowest feature the response can have is $\Delta\delta\sim 2\pi/T$. **Resolution is bought with sequence length**, exactly as in Section 2.
