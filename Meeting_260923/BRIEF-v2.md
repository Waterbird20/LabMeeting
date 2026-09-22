# 260923 chapter-1 revision brief (2026-09-21): addendum to BRIEF.md

Not a slide (no leading digits). Read `BRIEF.md` first (talk, writing rules, clip embedding
form, verify procedure), then this file. This revision acts on the speaker's `<!-- TODO -->`
comments in chapter 1 (files `10-` to `19z-`). Every rule of `BRIEF.md` still holds.

## Machine and environment (changed since the first build)

- We are on **WSL Linux** now, not macOS. The manim venv is rebuilt at
  `3. wiki/code/env-manim/.venv` (manim community 0.21.0, numpy, scipy, matplotlib; python
  3.13). latex, dvisvgm, ffmpeg, pangocairo are system packages. Verified working.
- Render from `3. wiki/code/lm-260923-animations/`:
  `../env-manim/.venv/bin/python render.py <scenes_file.py> <SceneName> -q l` (preview,
  ~3 min per 20 s clip) and `-q h` (final; expect 15-40 min). `render.py` installs
  `media/<clip>.mp4` + `media/<clip>.png` in the deck folder. Preview to the scratchpad
  when you only want to look: `../env-manim/.venv/bin/manim render -ql --disable_caching
  --media_dir <scratch> <file> <Scene>` and grab frames with
  `ffmpeg -ss <t> -i <mp4> -frames:v 1 f.png`. The machine has 16 cores; run at most two
  `-q h` renders of your own at a time.
- Build the deck privately from the deck folder:
  `make THEME=serif BUILD=.build-<tag>.md BASE=<tag> pdf < /dev/null` (the `< /dev/null` is
  mandatory, marp otherwise waits on stdin forever), render your pages with
  `pdftoppm -png -r 60 -f A -l B <tag>.pdf <scratch>/pg`, Read the PNGs, then delete
  `.build-<tag>.md`, `<tag>.pdf`, `<tag>.html`. Never run the un-tagged `make` (it is the
  integrator's build and concurrent builds collide).

## Camera (already changed; do not change it again)

`bloch.py` now defines `CAM_PHI, CAM_THETA = 68 deg, 42 deg` and `setup_camera` defaults to
them; `scenes_seq.py`, `scenes_bell.py`, `scenes_dd.py` take `PHI, THETA` from there. The
view is now: $x$ front-left, $y$ front-right, $z$ up (the speaker's "x-axis should lie
on -y axis" TODO). Use `setup_camera(self)` or `PHI, THETA` and nothing else. If you must
edit `bloch.py`, add; never change existing behaviour, other agents import it live.

## Drive convention (decided by the speaker: switch the whole chapter to a $y$ drive)

$$H=\tfrac{\delta}{2}Z+\tfrac{\Omega}{2}Y,\qquad \hat n=\frac{(0,\Omega,\delta)}{\Omega_R},\qquad
\Omega_R=\sqrt{\Omega^2+\delta^2}.$$

- Rabi: $\delta=0$, rotation about $y$ by $\Omega t$; $\ket{0}$ sweeps toward $\ket{+}$ ($+x$)
  in the $x$-$z$ plane; $P(0)=\cos^2(\Omega t/2)$.
- Ramsey: $\pi/2$ about $y$ lands on $\ket{+}$; free precession about $z$ by $\delta\tau$;
  second $\pi/2$ about $-y$ gives $P(0)=\tfrac12(1+\cos\delta\tau)$ (about $+y$ it would be
  $\tfrac12(1-\cos\delta\tau)$; keep that remark in the EDIT-FORWARD, reworded to $y$).
- ODMR: $t=\pi/\Omega$, axis $\hat n=(0,\Omega,\delta)/\Omega_R$ tilts toward $z$,
  $P(0)=1-\frac{\Omega^2}{\Omega_R^2}\sin^2(\Omega_R t/2)$. (The slide currently has a wrong
  argument $\sqrt{\Omega_R^2+\delta^2}\,t/2$; fix it.)
- Any slide or clip that says "about $x$", "$\tfrac{\Omega}{2}X$" or "drive phase along
  $x$" in the sequence context changes accordingly. `16b-rotation.md` and its new clip
  use the same examples ($\tfrac{\Omega}{2}Y$ resonant pulse, $\tfrac{\delta}{2}Z$ precession).

## Shared visual vocabulary (so the clips look alike)

- **Rotation angle**: draw the swept angle as an arc on the sphere (a `ParametricFunction`
  or `Arc` in the plane perpendicular to the axis, radius about $0.45$ of the sphere radius,
  in the axis colour `C_AXIS`), labelled with the angle symbol ($\Omega t$, $\delta\tau$,
  $\theta$, $\phi$, ...). Compute every point from the matrices (`rot`, `bloch_vector`).
- **Measurement**: at read-out time, a **dashed line** (`DashedLine`, `GREY_B`) drops from
  the arrow tip perpendicularly onto the $z$ axis; a small dot marks the foot on the $z$
  axis; only then does the dot appear on the read-out plot (right panel). The sweep
  continues from there. All three sequence clips and any other read-out clip use this.
- Left half of the frame is the sphere, right half the read-out panel (`readout_axes`),
  title lines on top, the prepare / pulse / read strip at the bottom (`scenes_seq.py`
  `stage`, `title`, `steps` helpers are the pattern).
- Dark background, `C_STATE` crimson for the state, `C_0` blue / `C_1` orange for
  branches, `C_AXIS` for rotation axes. Clip length 10-40 s, looping.

## File ownership (one agent per row; touch nothing else)

| agent | scene file | clips (fixed names) | slide files |
|---|---|---|---|
| `bloch` | `scenes_bloch_plus.py` | `bloch-plus` (recut), `bloch-angles` (new) | `14-bloch.md`, `14a-angles.md` |
| `seq` | `scenes_seq.py` | `seq-rabi`, `seq-ramsey`, `seq-odmr` (all recut) | `17-examples.md`, `17a-rabi.md`, `17b-ramsey.md`, `17c-odmr.md` |
| `bell` | `scenes_bell.py` | `bell-mixed` (recut), `bell-decay` (new) | `18a-mixed.md`, `18b-decay.md`, new `18c-times.md` |
| `dd` | `scenes_dd.py` | `dd-unit` (new), `dd-cpmg` (recut) | `19-dd.md`, `19b-axes.md` |
| `unitary` | new `scenes_unitary.py` | `unitary-rotation` (new) | `16b-rotation.md` |

The integrator owns every other file (`11`, `12`, `13`, `13a`, `15`, `16`, `16a`, `17d`,
`18`, `19a`, `19z`, `README.md`, `BRIEF*.md`, the Makefile, `_index_wiki.md`, `log.md`).

## Slide-editing rules for this revision

- Act on each `<!-- TODO ... -->` in your files, then delete that TODO comment. Keep the
  `<!-- EDIT-FORWARD ... -->` comments that are still open (reword if the convention
  changed); delete the ones you resolved.
- The speaker rewrote the prose today; keep their sentences and meaning, fix only grammar,
  typos, and the physics errors named above. Full sentences, no em-dashes, all symbols in
  math mode, ~12 lines at 26 px, no overflow past the footer.
- **One consolidated write per file.** The vault is mirrored by a sync bridge that can
  echo a stale snapshot over a file during a burst of rapid edits. Compose the whole file,
  write it once, and at the end of your work `grep` each of your files to confirm the
  content is what you wrote.
- Keep the `_verify()` pattern: every state, axis, angle, and curve in a clip comes from
  the actual matrices, and a `_verify()` block checks the printed formulas numerically.
- Put the clip on the slide with the mandatory `<figure class="figure">` form from
  `BRIEF.md` (`preload="none"`, blank lines around the `<video>` tag, italic markdown caption).

## Report back (final message)

Clip names, durations, poster fractions, which pages you rendered and checked, what you
changed in the slide text beyond the TODOs, and anything the speaker still has to decide
(as EDIT-FORWARD comments in the file plus one line in your report).
