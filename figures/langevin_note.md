# Langevin dynamics on flat landscapes: an end-to-end walkthrough

This note develops the theory of Langevin dynamics on a degenerate fitness manifold with varying transverse curvature, from the general setup to two specific landscapes. For each setting we derive the analysis in both the continuous-time (idealized SDE) and discrete-time (Euler–Maruyama, finite step size) regimes.

The result is a unified framework: there is one general formula for the stationary distribution, parameterized by the curvature profile $g(y)$ and the step size $\eta$, and two examples show the qualitatively different regimes it captures.

-----

## 1. Setup

### 1.1 The landscape

We work on a two-dimensional landscape of the form
$$
\mathcal{F}(x,y) \;=\; \mathcal{F}^{*} \;-\; \tfrac{1}{2}\,g(y)\,x^{2},
$$
where $g(y) > 0$ is the curvature of the fast (transverse) direction at slow position $y$. The optimal manifold is $x=0$, on which the fitness equals $\mathcal{F}^{*}$ uniformly: it is exactly degenerate. The curvature profile $g(y)$ describes how the *transverse* sharpness varies along the manifold.

The question we want to answer: under noisy gradient ascent on $\mathcal{F}$, does the population stay on the manifold? Where does it sit? Does it equilibrate to a stationary distribution? How does the answer depend on $g(y)$ and on the discretization step size $\eta$?

### 1.2 The dynamics

Continuous-time Langevin dynamics (gradient ascent on $\mathcal{F}$ with white Gaussian noise):
$$
\dot{x} \;=\; \partial_{x}\mathcal{F} + \sqrt{2}\,\sigma\,\eta_{x}(t),
\qquad
\dot{y} \;=\; \partial_{y}\mathcal{F} + \sqrt{2}\,\sigma\,\eta_{y}(t),
$$
with $\langle\eta_{i}(t)\eta_{j}(t')\rangle = \delta_{ij}\delta(t-t')$. Computing the gradients,
$$
\dot{x} \;=\; -\,g(y)\,x \;+\; \sqrt{2}\,\sigma\,\eta_{x},
\qquad
\dot{y} \;=\; -\,\tfrac{1}{2}\,g'(y)\,x^{2} \;+\; \sqrt{2}\,\sigma\,\eta_{y}.
$$

The Euler–Maruyama discretization with step size $\eta$ is
$$
x_{t+1} \;=\; x_{t} - \eta\,g(y_{t})\,x_{t} + \sqrt{2\eta}\,\sigma\,\xi_{t,x},
$$
$$
y_{t+1} \;=\; y_{t} - \tfrac{1}{2}\,\eta\,g'(y_{t})\,x_{t}^{2} + \sqrt{2\eta}\,\sigma\,\xi_{t,y},
$$
with $\xi$ standard Gaussian.

### 1.3 The strategy: timescale separation

The fast direction $x$ feels a restoring force $-g(y)x$, while the slow direction $y$ feels a force $-\tfrac{1}{2}g'(y)x^{2}$ that depends on $x$ quadratically. Whenever the relaxation time of $x$ at fixed $y$ is much shorter than the timescale on which $y$ evolves, we can integrate out the fast direction (adiabatic elimination), replacing $x^{2}$ in the slow equation by its conditional equilibrium value $\langle x^{2}\rangle\big|_{y}$. The resulting effective dynamics for $y$ is a one-dimensional Langevin equation with a curvature-induced "entropic" drift.

We carry out this program twice: once in the continuous-time limit, then more carefully at finite $\eta$.

-----

## 2. General framework, continuous-time

### 2.1 Fast mode at fixed $y$

At fixed $y$, the $x$-equation is an Ornstein–Uhlenbeck process. Its stationary distribution is Gaussian with mean zero and variance
$$
\langle x^{2}\rangle\big|_{y} \;=\; \frac{\sigma^{2}}{g(y)},
$$
reached on the timescale $\tau_{x}(y) = 1/g(y)$. The variance is *large* where $g$ is small (transverse direction is soft) and *small* where $g$ is large (transverse direction is sharp). This position-dependence of the equilibrium fluctuations is the seed of the entropic mechanism.

### 2.2 Effective drift on the slow direction

Average the deterministic part of the $\dot{y}$ equation over the conditional distribution of $x$ at fixed $y$:
$$
\langle\dot{y}\rangle \;=\; -\,\tfrac{1}{2}\,g'(y)\,\langle x^{2}\rangle\big|_{y} \;=\; -\,\tfrac{1}{2}\,g'(y)\cdot\frac{\sigma^{2}}{g(y)} \;=\; -\,\frac{\sigma^{2}}{2}\,\frac{g'(y)}{g(y)}.
$$
This drift is proportional to the logarithmic gradient of $g$. It is negative where $g$ increases (toward sharper regions) and positive where $g$ decreases (toward flatter regions), so it always pushes the slow coordinate toward smaller $g$ — i.e., toward flatter transverse curvature.

### 2.3 Effective potential

Write $\langle\dot{y}\rangle = -V_{\rm eff}'(y)$. Integrating:
$$
\boxed{\;V_{\rm eff}(y) \;=\; \frac{\sigma^{2}}{2}\,\ln g(y).\;}
$$
This is the **entropic effective potential**. It is the free energy of the fast direction treated as a function of the slow coordinate: low where $g$ is small (lots of room for fluctuations, high entropy) and high where $g$ is large (constrained fluctuations, low entropy).

### 2.4 Effective SDE and Fokker–Planck

After adiabatic elimination, the slow direction obeys the one-dimensional Langevin equation
$$
dy \;=\; -V_{\rm eff}'(y)\,dt + \sqrt{2}\,\sigma\,dW_{y}.
$$
The corresponding Fokker–Planck equation has stationary solution (when normalizable)
$$
P^{*}(y) \;\propto\; \exp\!\Bigl(-\,\frac{V_{\rm eff}(y)}{\sigma^{2}}\Bigr) \;=\; \frac{1}{\sqrt{g(y)}}.
$$
$$
\boxed{\;P^{*}(y) \;\propto\; \frac{1}{\sqrt{g(y)}}.\;}
$$
**Three regimes for normalizability** of $P^{*}$:

1. $g(y)$ bounded above and below and the manifold is compact → always normalizable.
2. $g(y)$ bounded away from zero and growing faster than $y^{2}$ at infinity → $1/\sqrt{g}$ decays faster than $1/|y|$ at infinity, integrable.
3. $g(y)$ vanishes somewhere or grows too slowly at infinity → integrand diverges, no stationary distribution.

### 2.5 General Itô identity

For the effective SDE $dy = b(y)dt + \sqrt{2D}\,dW$ with $b(y) = -V_{\rm eff}'(y)$ and $D = \sigma^{2}$, Itô's lemma applied to a smooth function $f(y)$ gives
$$
df \;=\; \bigl[\,f'(y)\,b(y) + D\,f''(y)\,\bigr]\,dt + \sqrt{2D}\,f'(y)\,dW.
$$
Taking expectations:
$$
\frac{d\langle f(y)\rangle}{dt} \;=\; \bigl\langle f'(y)\,b(y)\bigr\rangle + D\,\bigl\langle f''(y)\bigr\rangle.
$$
Specializing to $f(y) = y^{2}$:
$$
\frac{d\langle y^{2}\rangle}{dt} \;=\; 2\,\bigl\langle y\,b(y)\bigr\rangle + 2D \;=\; -2\,\bigl\langle y\,V_{\rm eff}'(y)\bigr\rangle + 2\sigma^{2}.
$$
At stationarity this requires $\langle y\,V_{\rm eff}'(y)\rangle = \sigma^{2}$. Whether this can be satisfied as a pointwise identity (martingale) or only as an integrated balance against $P^{*}$ (true equilibration) depends on the form of $V_{\rm eff}'$, which depends on $g(y)$.

-----

## 3. General framework, discrete-time (finite $\eta$)

### 3.1 Fast mode at fixed $y$: discrete OU

At fixed $y$, the discrete $x$-update is the autoregressive process
$$
x_{t+1} \;=\; \bigl(1 - \eta\,g(y)\bigr)\,x_{t} + \sqrt{2\eta}\,\sigma\,\xi_{t,x}.
$$
This is stable iff $|1-\eta g(y)| < 1$, i.e.,
$$
0 \;<\; \eta\,g(y) \;<\; 2.
$$
The upper bound defines a **stability cutoff**:
$$
\boxed{\;|y| \;<\; Y_{\rm cut}\,, \quad \text{where}\quad \eta\,g(Y_{\rm cut}) = 2.\;}
$$
Beyond $Y_{\rm cut}$ the fast direction is dynamically unstable: trajectories there diverge within a few steps. The population is dynamically excluded from $|y|>Y_{\rm cut}$ even though the landscape extends to infinity. **This is a purely discrete-time effect that vanishes as $\eta\to 0$.**

The stationary variance of the stable AR(1) is
$$
V_{x}^{\rm disc}(y) \;=\; \frac{(\sqrt{2\eta}\sigma)^{2}}{1-(1-\eta g(y))^{2}} \;=\; \frac{2\sigma^{2}}{g(y)\bigl[\,2 - \eta\,g(y)\,\bigr]}.
$$
Compared to the continuous value $V_{x}^{\rm cont}(y) = \sigma^{2}/g(y)$, this is enhanced by
$$
K(y) \;\equiv\; \frac{V_{x}^{\rm disc}(y)}{V_{x}^{\rm cont}(y)} \;=\; \frac{2}{2-\eta\,g(y)} \;\geq\; 1,
$$
with $K \to 1$ as $\eta\to 0$ and $K \to \infty$ as $|y| \to Y_{\rm cut}$.

### 3.2 Effective drift, finite $\eta$

Substituting $\langle x^{2}\rangle = V_{x}^{\rm disc}(y)$ into the discrete $y$-update and averaging:
$$
\langle y_{t+1} - y_{t}\rangle \;=\; -\tfrac{1}{2}\,\eta\,g'(y)\,V_{x}^{\rm disc}(y).
$$
Per unit physical time,
$$
b_{\rm eff,\eta}(y) \;=\; -\,\frac{\sigma^{2}\,g'(y)}{g(y)\bigl[2-\eta g(y)\bigr]} \;=\; b_{\rm cont}(y)\cdot K(y).
$$
The finite-$\eta$ correction is to multiply the continuous-time drift by $K(y)$, which diverges at $Y_{\rm cut}$.

### 3.3 Effective potential, finite $\eta$

Write $b_{\rm eff,\eta}(y) = -V_{\rm eff,\eta}'(y)$ and integrate. Substituting $u = g(y)$, so $du = g'(y)\,dy$:
$$
V_{\rm eff,\eta}'(y)\,dy \;=\; \frac{\sigma^{2}\,du}{u\,(2-\eta u)}.
$$
Partial fractions:
$$
\frac{1}{u(2-\eta u)} \;=\; \frac{1}{2u} + \frac{\eta}{2(2-\eta u)}.
$$
Integrating term by term:
$$
\int\frac{du}{u} = \ln u, \qquad \int\frac{\eta\,du}{2-\eta u} = -\ln|2-\eta u|.
$$
Therefore
$$
V_{\rm eff,\eta}(y) \;=\; \frac{\sigma^{2}}{2}\,\bigl[\,\ln g(y) - \ln\bigl(2-\eta g(y)\bigr)\,\bigr] \;=\; \frac{\sigma^{2}}{2}\,\ln\!\frac{g(y)}{2-\eta\,g(y)}.
$$
$$
\boxed{\;V_{\rm eff,\eta}(y) \;=\; \frac{\sigma^{2}}{2}\,\ln\!\frac{g(y)}{2-\eta\,g(y)}.\;}
$$
Sanity check: as $\eta\to 0$, $V_{\rm eff,\eta} \to (\sigma^{2}/2)\ln[g(y)/2] = (\sigma^{2}/2)\ln g(y) + \text{const}$, recovering the continuous-time formula.

### 3.4 Stationary distribution, finite $\eta$

Applying the Boltzmann formula $P^{*} \propto \exp(-V_{\rm eff,\eta}/\sigma^{2})$:
$$
\boxed{\;P^{*}(y) \;\propto\; \frac{\sqrt{2-\eta\,g(y)}}{\sqrt{g(y)}}\quad\text{for }|y| < Y_{\rm cut}.\;}
$$
Compared to the continuous result $P^{*} \propto 1/\sqrt{g(y)}$, the finite-$\eta$ distribution carries the extra factor $\sqrt{2-\eta g(y)}$, which:

- Equals $\sqrt{2}$ (a constant) when $\eta g(y) \ll 1$ — the continuous limit.
- Drops to zero like a square root as $|y|\to Y_{\rm cut}^{-}$ — providing automatic compact support.

**Two ways the discrete distribution can fail to be normalizable:**

1. **Outer**: if $g(y)$ grows too slowly at $\infty$ AND the formal stationary on $|y|>Y_{\rm cut}$ would have unbounded mass. *Not a problem*: discretization always provides $Y_{\rm cut}$.
2. **Inner**: if $g(y) \to 0$ somewhere on the manifold, then $1/\sqrt{g(y)}$ diverges at that point. The factor $\sqrt{2-\eta g(y)} \to \sqrt{2}$ at such a point, **does not** regulate the divergence.

So: **discretization automatically regulates outer tails but never regulates inner singularities of $g$**.

### 3.5 Self-consistency: when is timescale separation valid?

The adiabatic elimination requires that $x$ relaxes much faster than $y$ evolves: $\tau_{x}(y) = 1/g(y) \ll \tau_{y}(y)$. The $y$-dynamics has drift rate $|V_{\rm eff}'(y)| = \sigma^{2}|g'|/(2g)$ and diffusion $\sigma^{2}$. The timescale on which $y$ changes by an amount comparable to itself is $\tau_{y} \sim y^{2}\,g/(|g'|\sigma^{2})$ or so; the precise form is landscape-dependent. The criterion is that $g(y)$ must not be too small, which fails at points where $g(y)\to 0$.

-----

## 4. Example 1: $g(y) = y^{2}$ (singular landscape)

The original landscape we used in the paper: $\mathcal{F} = \mathcal{F}^{*} - \tfrac{1}{2}x^{2}y^{2}$, so $g(y) = y^{2}$.

The key feature of this $g$ is that **it vanishes at $y=0$** — the flattest point on the manifold has *zero* curvature in the transverse direction. This makes timescale separation fail at exactly the most interesting place, and propagates a divergence into the stationary distribution.

### 4.1 Continuous-time analysis

Apply the general formulas with $g(y) = y^{2}$, $g'(y) = 2y$:

**Drift on slow direction:**
$$
\langle\dot{y}\rangle \;=\; -\,\frac{\sigma^{2}}{2}\,\frac{2y}{y^{2}} \;=\; -\,\frac{\sigma^{2}}{y}.
$$

**Effective potential:**
$$
V_{\rm eff}(y) \;=\; \frac{\sigma^{2}}{2}\,\ln y^{2} \;=\; \sigma^{2}\,\ln|y|.
$$

**Effective SDE:**
$$
dy \;=\; -\,\frac{\sigma^{2}}{y}\,dt + \sqrt{2}\,\sigma\,dW.
$$

**Formal stationary distribution:**
$$
P^{*}(y) \;\propto\; \frac{1}{|y|}.
$$
This is **not normalizable**: $\int dy/|y|$ diverges at both $y=0$ (logarithmically) and $|y|\to\infty$ (logarithmically). The dynamics has no stationary state.

**Itô analysis of the second moment.** Take $f(y) = y^{2}$ in the general identity. With $b(y) = -\sigma^{2}/y$ and $D = \sigma^{2}$:
$$
\frac{d\langle y^{2}\rangle}{dt} \;=\; 2\bigl\langle y\cdot(-\sigma^{2}/y)\bigr\rangle + 2\sigma^{2} \;=\; -2\sigma^{2} + 2\sigma^{2} \;=\; 0.
$$
The cancellation holds **pointwise**: at every $y$, $y\cdot b(y) = -\sigma^{2}$ exactly cancels $D = \sigma^{2}$. So $y^{2}$ is a martingale; $\langle y^{2}\rangle$ is conserved at its initial value, *exactly*, for all time.

This pointwise cancellation is special. For a general $V_{\rm eff}'(y) = c/y$ (drift $-c/y$, diffusion $D$), the second-moment evolution is $-2c + 2D$, which vanishes pointwise iff $c = D$. For our landscape this is automatic because $c = \sigma^{2} = D$.

**Physical picture.** The drift toward zero is real and detectable at the single-trajectory level, but at the population level $\langle y^{2}\rangle$ is *exactly* preserved — drift and diffusion balance pointwise. The drift collapses the *median* of the population while the heavy tails grow to compensate, conserving $\langle y^{2}\rangle$. The population gets simultaneously more peaked at the origin and more heavy-tailed, with no equilibration to any normalizable form.

### 4.2 Discrete-time analysis (finite $\eta$)

Apply the general formulas with $g(y) = y^{2}$:

**Stability cutoff:**
$$
\eta\,y^{2} \;<\; 2 \quad\Longleftrightarrow\quad |y| \;<\; Y_{\rm cut} \;=\; \sqrt{2/\eta}.
$$

**Discrete OU stationary variance:**
$$
V_{x}^{\rm disc}(y) \;=\; \frac{2\sigma^{2}}{y^{2}(2-\eta y^{2})}.
$$
Note that this *diverges* at $y=0$: the fast mode does not equilibrate at the singular point. The breakdown of TSS at $y=0$ is even more dramatic than in the continuous case.

**Effective drift:**
$$
b_{\rm eff,\eta}(y) \;=\; -\,\frac{2\sigma^{2}}{y\,(2-\eta y^{2})}.
$$

**Effective potential:**
$$
V_{\rm eff,\eta}(y) \;=\; \frac{\sigma^{2}}{2}\,\ln\!\frac{y^{2}}{2-\eta y^{2}} \;=\; \sigma^{2}\,\ln\!\frac{|y|}{\sqrt{2-\eta y^{2}}}.
$$

**Stationary distribution:**
$$
P^{*}(y) \;\propto\; \frac{\sqrt{2-\eta y^{2}}}{|y|}\quad\text{for }|y| < \sqrt{2/\eta}.
$$

**Normalizability check.**

- *Outer edge* $|y|\to Y_{\rm cut}^{-}$: numerator $\sqrt{2-\eta y^{2}}\to 0$, so $P^{*}(Y_{\rm cut}) = 0$. The outer edge is regulated. ✓
- *Inner singularity* $|y|\to 0$: numerator $\sqrt{2-\eta y^{2}}\to\sqrt{2}$ (finite), denominator $|y|\to 0$. So $P^{*}\sim\sqrt{2}/|y|$, and the integral diverges logarithmically. **Not regulated by finite $\eta$.** ✗

**Conclusion:** discretization regulates the outer tail but not the inner singularity of $g(y) = y^{2}$ at $y=0$. The distribution is still non-normalizable; the dynamics still has no stationary state, even at finite $\eta$. The non-normalizability is a structural feature of the singular landscape, not a continuum artifact.

**Itô at finite $\eta$.** The discrete-time analog of the Itô identity gives
$$
\frac{\langle y_{t+1}^{2}\rangle - \langle y_{t}^{2}\rangle}{\eta} \;=\; 2\sigma^{2}\Bigl[1 - 2\,\Bigl\langle\frac{1}{2-\eta y^{2}}\Bigr\rangle\Bigr] + O(\eta).
$$
In the continuous limit ($\eta y^{2}\to 0$, $1/(2-\eta y^{2})\to 1/2$), the bracket vanishes and the martingale is recovered. At finite $\eta$ the bracket is negative (since $\eta y^{2}>0$ for any nonzero $y$ gives $1/(2-\eta y^{2})>1/2$), so $\langle y^{2}\rangle$ slowly *decreases*. The discretization breaks the exact martingale conservation, introducing a slow inward drift on the second moment. But since the stationary distribution is non-normalizable, this slow decrease never converges to a finite value — it asymptotes toward whatever fraction of mass remains in the TSS-valid band.

-----

## 5. Example 2: $g(y) = (1+y^{2})^{2}$ (smooth landscape)

A regularized version of the landscape: $\mathcal{F} = \mathcal{F}^{*} - \tfrac{1}{2}x^{2}(1+y^{2})^{2}$, so $g(y) = (1+y^{2})^{2}$.

The key features of this $g$:

- $g(y) \geq 1$ everywhere — bounded away from zero.
- $g(0) = 1$: the flattest point has finite curvature, not zero.
- $g(y) \sim y^{4}$ at large $|y|$: grows faster than $y^{2}$.

Timescale separation holds everywhere, and $P^{*}\propto 1/\sqrt{g(y)}$ decays as $1/y^{2}$ at infinity — fast enough to be normalizable.

### 5.1 Continuous-time analysis

Apply the general formulas with $g(y) = (1+y^{2})^{2}$, $g'(y) = 4y(1+y^{2})$:

**Drift on slow direction:**
$$
\langle\dot{y}\rangle \;=\; -\,\frac{\sigma^{2}}{2}\,\frac{4y(1+y^{2})}{(1+y^{2})^{2}} \;=\; -\,\frac{2\sigma^{2}\,y}{1+y^{2}}.
$$

**Effective potential:**
$$
V_{\rm eff}(y) \;=\; \frac{\sigma^{2}}{2}\,\ln(1+y^{2})^{2} \;=\; \sigma^{2}\,\ln(1+y^{2}).
$$

**Effective SDE:**
$$
dy \;=\; -\,\frac{2\sigma^{2}\,y}{1+y^{2}}\,dt + \sqrt{2}\,\sigma\,dW.
$$

**Stationary distribution:**
$$
P^{*}(y) \;\propto\; \frac{1}{1+y^{2}}, \qquad \int_{-\infty}^{\infty}\frac{dy}{1+y^{2}} = \pi, \qquad P^{*}(y) \;=\; \frac{1/\pi}{1+y^{2}}.
$$
This is the **Cauchy distribution** (Lorentzian), centered at $y=0$ with scale parameter 1.

**Itô on second moment.** With $b(y) = -2\sigma^{2}y/(1+y^{2})$ and $D = \sigma^{2}$:
$$
\frac{d\langle y^{2}\rangle}{dt} \;=\; -2\,\Bigl\langle\frac{2\sigma^{2}y^{2}}{1+y^{2}}\Bigr\rangle + 2\sigma^{2} \;=\; 2\sigma^{2}\,\Bigl\langle\frac{1-y^{2}}{1+y^{2}}\Bigr\rangle.
$$
This is *not* zero pointwise — the integrand $(1-y^{2})/(1+y^{2})$ depends on $y$. The expected value vanishes only when averaged against the Cauchy stationary $P^{*}$, where one computes $\langle 1/(1+y^{2})\rangle = \langle y^{2}/(1+y^{2})\rangle = 1/2$.

**Physical picture.** The population genuinely equilibrates to the Cauchy distribution. The peak is at $y=0$ (entropic confinement at the flattest point), but the tails are heavy: $P^{*}\sim 1/y^{2}$ at large $|y|$, so $\langle y^{2}\rangle_{P^{*}} = \infty$. About half the mass lies within $|y|<1$; the 95th percentile of $|y|$ is at $\tan(0.475\pi)\approx 12.7$. Particles at $|y|\sim 2$ sit in the bulk of $P^{*}$, not in any tail.

### 5.2 Discrete-time analysis (finite $\eta$)

Apply the general formulas with $g(y) = (1+y^{2})^{2}$:

**Stability cutoff:**
$$
\eta(1+y^{2})^{2} < 2 \quad\Longleftrightarrow\quad |y| < Y_{\rm cut} = \sqrt{\sqrt{2/\eta}-1}.
$$
For $\eta = 0.01$: $Y_{\rm cut} \approx 3.625$.

**Discrete OU stationary variance:**
$$
V_{x}^{\rm disc}(y) \;=\; \frac{2\sigma^{2}}{(1+y^{2})^{2}\bigl[2-\eta(1+y^{2})^{2}\bigr]}.
$$
Bounded at all $y$ in the stability range — no inner singularity.

**Effective drift:**
$$
b_{\rm eff,\eta}(y) \;=\; -\,\frac{4\sigma^{2}\,y}{(1+y^{2})\bigl[2-\eta(1+y^{2})^{2}\bigr]}.
$$

**Effective potential:**
$$
V_{\rm eff,\eta}(y) \;=\; \frac{\sigma^{2}}{2}\,\ln\!\frac{(1+y^{2})^{2}}{2-\eta(1+y^{2})^{2}} \;=\; \sigma^{2}\,\ln\!\frac{1+y^{2}}{\sqrt{2-\eta(1+y^{2})^{2}}}.
$$

**Stationary distribution:**
$$
P^{*}(y) \;\propto\; \frac{\sqrt{2-\eta(1+y^{2})^{2}}}{1+y^{2}}\quad\text{for }|y| < Y_{\rm cut}.
$$

**Normalizability check.**

- *Outer edge* $|y|\to Y_{\rm cut}^{-}$: numerator $\to 0$, so $P^{*}(Y_{\rm cut}) = 0$. Regulated. ✓
- *Inner behavior* $|y|\to 0$: numerator $\to\sqrt{2-\eta}\approx\sqrt{2}$ (finite), denominator $\to 1$ (finite). $P^{*}(0)$ is finite. **No inner singularity to worry about.** ✓

The distribution is normalizable on compact support $[-Y_{\rm cut}, Y_{\rm cut}]$ and therefore has **finite variance**. As $\eta\to 0$, $Y_{\rm cut}\to\infty$ and the Cauchy distribution is recovered.

**Empirical check.** With $\sigma=0.05$, $\eta=0.01$, $N=1500$ particles run for $5\times 10^{5}$ iterations:

| quantile | empirical | continuous Cauchy | finite-$\eta$ theory |
|----------|-----------|-------------------|----------------------|
| p50      | 0.721     | 1.000             | 0.720                |
| p75      | 1.374     | 2.414             | 1.365                |
| p90      | 2.161     | 6.314             | 2.139                |
| p95      | 2.600     | 12.71             | 2.601                |
| p99      | 3.201     | 63.66             | 3.238                |

The finite-$\eta$ theory matches the empirical quantiles to within 1% across the full range. Cauchy fails dramatically beyond p75. **Running longer cannot push the empirical distribution toward Cauchy — only shrinking $\eta$ can.**

-----

## 6. Comparative summary

Putting all four cases on the same footing:

| Case                 | $g(y)$          | $P^{*}(y)\propto$                      | Outer cutoff             | Inner behavior | Normalizable?    | $\langle y^{2}\rangle$       |
|----------------------|-----------------|----------------------------------------|--------------------------|----------------|------------------|------------------------------|
| **1, continuous**    | $y^{2}$         | $1/\lvert y\rvert$                     | none                     | divergent      | **No**           | conserved (Itô martingale)   |
| **1, finite $\eta$** | $y^{2}$         | $\sqrt{2-\eta y^{2}}/\lvert y\rvert$   | $\sqrt{2/\eta}$          | divergent      | **No**           | slowly decreasing            |
| **2, continuous**    | $(1+y^{2})^{2}$ | $1/(1+y^{2})$                          | none                     | finite         | **Yes (Cauchy)** | infinite                     |
| **2, finite $\eta$** | $(1+y^{2})^{2}$ | $\sqrt{2-\eta(1+y^{2})^{2}}/(1+y^{2})$ | $\sqrt{\sqrt{2/\eta}-1}$ | finite         | **Yes**          | finite                       |

### Two orthogonal axes of comparison

**Continuous $\leftrightarrow$ Finite $\eta$.** The discretization adds an outer cutoff $Y_{\rm cut}$ defined by $\eta g(Y_{\rm cut}) = 2$ and a factor $\sqrt{2-\eta g(y)}$ in $P^{*}$ that vanishes at the cutoff. This always regulates outer tails. It never regulates inner singularities of $g$.

**$g$ singular at the flat point $\leftrightarrow$ $g$ smooth.** Whether the flat point itself has zero or finite transverse curvature controls whether $P^{*}\propto 1/\sqrt{g}$ has an integrable inner behavior. Singular $g$ (Case 1) is structurally non-normalizable; smooth $g$ (Case 2) is structurally normalizable.

The two effects multiply: finite-$\eta$ on smooth $g$ gives a fully confined, finite-variance distribution. Finite-$\eta$ on singular $g$ gives an outer-cutoff but inner-divergent distribution — still non-normalizable. The continuous limit on smooth $g$ is the marginal Cauchy: normalizable but heavy-tailed. The continuous limit on singular $g$ is the doubly-divergent $1/|y|$.

-----

## 7. Three takeaways

1. **Langevin dynamics on a degenerate manifold has a generic entropic drift toward flatter regions, given by $\langle\dot{y}\rangle = -(\sigma^{2}/2)\,g'(y)/g(y)$ regardless of the global shape of $g$.** This drift is real and detectable at the single-trajectory level. It is the analog of Di Carlo's curvature-induced entropic force for the more general curved-manifold geometry. It does not by itself produce a confined population.
2. **Whether the population reaches a stationary distribution depends on the inner behavior of $g$ at the flat point.** If $g$ vanishes there (as in our singular landscape $g=y^{2}$), the formal stationary $P^{*}\propto 1/\sqrt{g}$ is non-integrable at the flat point, and no stationary state exists. The continuous-time idealization has $\langle y^{2}\rangle$ exactly conserved (Itô martingale); the discrete dynamics has $\langle y^{2}\rangle$ slowly decreasing without convergence. If $g$ is bounded away from zero (the smooth landscape $g=(1+y^{2})^{2}$), the population does reach a stationary distribution.
3. **At finite step size $\eta$, the Euler–Maruyama discretization automatically truncates the manifold at $|y| < Y_{\rm cut}$ defined by $\eta g(Y_{\rm cut})=2$, replacing $P^{*}\propto 1/\sqrt{g}$ by $P^{*}\propto\sqrt{2-\eta g}/\sqrt{g}$. The truncation regulates outer tails (a Cauchy stationary distribution becomes a finite-variance one) but does not regulate inner singularities (a non-normalizable distribution stays non-normalizable). The "effective stationary state" your simulation reaches is governed by the finite-$\eta$ formula, not the continuous-time one.**

The qualitative picture in all four cases is the same: entropic drift toward flatness, plus diffusion, plus possibly a stability cutoff. The quantitative features — what the long-time distribution looks like, whether it exists at all, what its variance is — are sensitive to the curvature profile and the step size in ways that the unified formula $P^{*}\propto \sqrt{2-\eta g(y)}/\sqrt{g(y)}$ makes explicit.
