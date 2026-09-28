---
marp: true
theme: serif
math: mathjax
---

<!-- _class: tight -->

# <span class="cat intro">Intro</span> Convolution extracts local features

<style scoped>.defn { align-items: center; } .defn .col:first-child { flex: 0 0 330px; } .defn p { margin: 0; } .clip-label p { text-align: center; font-weight: 600; margin: 0.1em 0 0.25em; }</style>

<div class="columns defn">
<div class="col">

$$(x*k)[i,j]=\sum_{m,n}k_{m,n}\,x_{i+m,j+n}$$

</div>
<div class="col">

Each output pixel weights the patch under the kernel $k$; the same $k$ at every $(i,j)$ gives translation equivariance.

</div>
</div>

<div class="columns">
<div class="col">

<div class="clip-label">

Box blur, $k_{m,n}=1/9$

</div>

<figure class="figure">

<video src="media/image-conv-kirby.mp4" poster="media/image-conv-kirby.png" width="540" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
<div class="col">

<div class="clip-label">

Sobel kernel: vertical edges

</div>

<figure class="figure">

<video src="media/sobel-kirby.mp4" poster="media/sobel-kirby.png" width="540" controls autoplay loop muted playsinline preload="none"></video>

</figure>

</div>
</div>

A CNN learns its kernels. MNIST-1D, ten digits (published): CNN $\approx0.94$, MLP $\approx0.68$.

<div class="src">Clips: 3Blue1Brown, <i>But what is a convolution?</i> (2022), re-rendered with Kirby; see also <i>But what is a neural network?</i> (2017).</div>

<!-- Speaker note: the averaging kernel, k_{m,n} = 1/9, blurs the image. The Sobel kernel is left minus right, so vertical edges light up by sign (red negative, cyan positive). Deep learning calls this a convolution; strictly the formula above is a cross-correlation (a true convolution flips k, i.e. uses x_{i-m, j-n}). The clips compute a true convolution with scipy.signal.convolve and display the flipped kernel, so the numbers shown on the moving frame are exactly the weights of the formula above. A CNN stacks layers so that edges combine into strokes and strokes into digits (3Blue1Brown, But what is a neural network?). MNIST-1D digits sit at a random shift, which is what the CNN's weight sharing handles. -->
<!-- src: clips rendered 2026-09-28 from 3Blue1Brown's manim source (_2022/convolutions/discrete.py, github.com/3b1b/videos; that repository's scene code is CC BY-NC-SA 4.0 per its README and LICENSE.txt, checked 2026-09-29 in ~/.cache/3b1b-videos; the manim library itself is MIT) with manimgl 1.7.2 via 3. wiki/code/conv-intro-animations/render.py --deck 260929; scenes BoxBlurKirby and SobelFilterKirbyPlain are BoxBlurCat and SobelFilterCat with only the image changed to KirbySmall, 3Blue1Brown's own 40 x 37 Kirby sprite (not in the public repo), traced from a frame of the video by code/conv-intro-animations/trace_kirby.py (frame supplied by the speaker, 2026-09-28; re-rendered the same day, replacing a first hand-drawn stand-in). Box blur: 3 x 3 kernel of 1/9 on each colour channel. Sobel: kernel (-0.25, 0, 0.25; -0.5, 0, 0.5; -0.25, 0, 0.25) convolved with the pixel mean, displayed flipped as (0.25, 0, -0.25; 0.5, 0, -0.5; 0.25, 0, -0.25), so the output is left minus right: red = negative (Kirby's left outline, dark to bright), cyan = positive (right outline). The eyes and the curved outline also respond, and the outermost output columns show a faint response from the zero padding at the image border. Video URLs: youtube.com/watch?v=KuXjwB4LzSA (convolution), youtube.com/watch?v=aircAruvnKk (neural network); full entries on the references slide. -->
<!-- src: random shift: mnist1d-eda.md §2.1. CNN about 94% and MLP about 68%: the paper's reference accuracies on all ten digits (Greydanus and Kobak, Table 1), quoted in mnist1d-eda.md §2 and mnist1d-repro.md; our reproduction with the authors' code and protocol gives 93.8 +- 0.4 and 64.6 +- 0.7 (mnist1d-repro.md, 3 seeds); under our own training setup ConvBase 0.855 and MLPBase 0.632 (mnist1d-eda.md §4.4). -->
<!-- EDIT-FORWARD: the MLP figure of about 68% is the paper's value; our reproduction of the authors' code gives 64.6%. Say "about 0.65 to 0.68" aloud if asked. Resolved 2026-09-29 (integrator): the slide now quotes the published values as decimals (0.94, 0.68), like every other accuracy in the deck. -->
<!-- src: posters: image-conv-kirby.png at 0.97 of the clip (march done, 1 s hold); sobel-kirby.png at 0.985 of the clip, inside a 2.5 s hold on the finished output that SobelFilterKirbyPlain adds after the march (the upstream scene ends the moment the march finishes; re-rendered 2026-09-28, 28.5 s). -->
