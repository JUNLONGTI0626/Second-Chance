# 4. Methodology

This paper examines time-varying connectedness among **AI, Electricity, Coal, GAS, WTI, and Gold** in three nested steps. First, we estimate a DCC-GARCH(4,4)-t system to establish the baseline pattern of dynamic comovement. Second, we use a TVP-VAR-SV framework as the main model to identify **directional** connectedness and net transmission roles. Third, we extend the same TVP-VAR connectedness setup into the frequency domain via TVP-VAR-BK, so that spillovers can be separated by investment horizon. This design keeps the empirical logic sequential: **correlation -> direction -> horizon decomposition**.

---

## 4.1 DCC-GARCH

### Why this method is used
We begin with DCC-GARCH because it provides a transparent benchmark for time-varying correlations across markets. In this study, DCC-GARCH(4,4)-t is treated as auxiliary evidence: it documents whether comovement intensifies or weakens over time, but it does not identify who transmits shocks to whom.

### Core formulas

#### (A) Return transformation
**Standard display**

\[
r_{i,t} = 100\,\ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right), \quad i=1,\ldots,N,\; N=6.
\]

**MathType linear input**

`r_{i,t} = 100 \ln(P_{i,t}/P_{i,t-1}), \; i=1,\ldots,N, \; N=6`

#### (B) Univariate mean and variance equations
**Standard display**

\[
r_{i,t} = \mu_i + \varepsilon_{i,t}, \quad \varepsilon_{i,t} = \sqrt{h_{i,t}}\,z_{i,t}, \quad z_{i,t}\sim t_{\nu}(0,1),
\]

\[
h_{i,t} = \omega_i + \sum_{p=1}^{4}\alpha_{i,p}\varepsilon_{i,t-p}^2 + \sum_{q=1}^{4}\beta_{i,q}h_{i,t-q}.
\]

**MathType linear input**

`r_{i,t} = \mu_i + \varepsilon_{i,t}, \; \varepsilon_{i,t}=\sqrt{h_{i,t}}z_{i,t}, \; z_{i,t}\sim t_{\nu}(0,1)`

`h_{i,t} = \omega_i + \sum_{p=1}^{4}\alpha_{i,p}\varepsilon_{i,t-p}^2 + \sum_{q=1}^{4}\beta_{i,q}h_{i,t-q}`

#### (C) DCC conditional correlation construction
Let \(\boldsymbol{\varepsilon}_t = \mathbf{D}_t\mathbf{z}_t\), where \(\mathbf{D}_t=\mathrm{diag}(\sqrt{h_{1,t}},\ldots,\sqrt{h_{N,t}})\). Then:

**Standard display**

\[
\mathbf{H}_t = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t,
\]

\[
\mathbf{Q}_t = (1-a-b)\bar{\mathbf{Q}} + a\,\mathbf{z}_{t-1}\mathbf{z}_{t-1}' + b\,\mathbf{Q}_{t-1},
\]

\[
\mathbf{R}_t = \mathrm{diag}(\mathbf{Q}_t)^{-1/2}\,\mathbf{Q}_t\,\mathrm{diag}(\mathbf{Q}_t)^{-1/2}.
\]

**MathType linear input**

`\mathbf{H}_t = \mathbf{D}_t\mathbf{R}_t\mathbf{D}_t`

`\mathbf{Q}_t = (1-a-b)\bar{\mathbf{Q}} + a\mathbf{z}_{t-1}\mathbf{z}_{t-1}' + b\mathbf{Q}_{t-1}`

`\mathbf{R}_t = diag(\mathbf{Q}_t)^{-1/2}\mathbf{Q}_t diag(\mathbf{Q}_t)^{-1/2}`

### Symbol notes
- \(P_{i,t}\): level price/index of variable \(i\) at time \(t\).
- \(r_{i,t}\): daily log return.
- \(h_{i,t}\): conditional variance of series \(i\).
- \(\mathbf{H}_t\): conditional covariance matrix.
- \(\mathbf{R}_t\): time-varying conditional correlation matrix.
- \(a,b\): DCC persistence parameters with \(a,b\ge0\), typically \(a+b<1\).

### Method statement for the paper
All six variables are transformed into daily log returns before estimation. We implement a DCC-GARCH(4,4)-t benchmark to capture heavy tails and persistence in volatility. The model delivers dynamic conditional correlations that summarize evolving comovement across AI and energy-related markets. However, correlation is symmetric and cannot recover directional transmission or net shock contribution, so this step is interpreted as baseline evidence only. In the main text, DCC-GARCH(4,4)-t is the reference DCC specification, while alternative marginal settings are deferred to robustness checks. **In this paper, the role of Section 4.1 is to establish dynamic comovement, not directional connectedness.**

---

## 4.2 TVP-VAR-SV

### Why this method is used
To identify directional spillovers, we move from correlation to a structural connectedness framework. TVP-VAR-SV is the core model because it allows coefficients and shock variances to evolve smoothly over time, which is essential when linkages between AI and commodity/energy markets are regime-dependent.

### Core formulas

#### (A) TVP-VAR representation
**Standard display**

\[
\mathbf{y}_t = \mathbf{c}_t + \sum_{\ell=1}^{p}\mathbf{A}_{\ell,t}\mathbf{y}_{t-\ell} + \mathbf{u}_t,
\]

\[
\mathbf{u}_t \sim \mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_t), \quad \mathbf{y}_t=(AI_t, ELEC_t, COAL_t, GAS_t, WTI_t, GOLD_t)'.
\]

**MathType linear input**

`\mathbf{y}_t = \mathbf{c}_t + \sum_{\ell=1}^{p}\mathbf{A}_{\ell,t}\mathbf{y}_{t-\ell} + \mathbf{u}_t`

`\mathbf{u}_t \sim \mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_t)`

#### (B) Time-varying states
**Standard display**

\[
\mathrm{vec}(\mathbf{A}_t) = \mathrm{vec}(\mathbf{A}_{t-1}) + \boldsymbol{\eta}_t,
\]

\[
\mathbf{c}_t = \mathbf{c}_{t-1} + \boldsymbol{\zeta}_t.
\]

**MathType linear input**

`vec(\mathbf{A}_t) = vec(\mathbf{A}_{t-1}) + \boldsymbol{\eta}_t`

`\mathbf{c}_t = \mathbf{c}_{t-1} + \boldsymbol{\zeta}_t`

#### (C) Stochastic volatility / time-varying covariance
A convenient decomposition is:

**Standard display**

\[
\boldsymbol{\Sigma}_t = \mathbf{L}_t^{-1}\mathbf{D}_t\mathbf{L}_t^{-\prime}, \quad
\log \mathbf{d}_t = \log \mathbf{d}_{t-1} + \boldsymbol{\xi}_t,
\]

where \(\mathbf{D}_t=\mathrm{diag}(d_{1,t},\ldots,d_{N,t})\).

**MathType linear input**

`\boldsymbol{\Sigma}_t = \mathbf{L}_t^{-1}\mathbf{D}_t\mathbf{L}_t^{-\prime}`

`\log \mathbf{d}_t = \log \mathbf{d}_{t-1} + \boldsymbol{\xi}_t`

#### (D) GFEVD and connectedness measures
Let \(\theta_{ij,t}^{g}(H)\) denote the generalized FEVD share from shocks in \(j\) to forecast-error variance of \(i\) at horizon \(H\), row-normalized as

**Standard display**

\[
\tilde{\theta}_{ij,t}^{g}(H) = \frac{\theta_{ij,t}^{g}(H)}{\sum_{j=1}^{N}\theta_{ij,t}^{g}(H)},
\quad \sum_{j=1}^{N}\tilde{\theta}_{ij,t}^{g}(H)=1.
\]

\[
TCI_t(H) = 100\times \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ij,t}^{g}(H).
\]

\[
TO_{i,t}(H)=100\times\sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ji,t}^{g}(H),
\]

\[
FROM_{i,t}(H)=100\times\sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ij,t}^{g}(H),
\]

\[
NET_{i,t}(H)=TO_{i,t}(H)-FROM_{i,t}(H).
\]

\[
NPDC_{ij,t}(H)=100\times\left(\tilde{\theta}_{ji,t}^{g}(H)-\tilde{\theta}_{ij,t}^{g}(H)\right).
\]

**MathType linear input**

`\tilde{\theta}_{ij,t}^{g}(H) = \theta_{ij,t}^{g}(H) / \sum_{j=1}^{N}\theta_{ij,t}^{g}(H)`

`TCI_t(H) = 100 \times (1/N) \sum_{i=1}^{N}\sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ij,t}^{g}(H)`

`TO_{i,t}(H)=100 \times \sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ji,t}^{g}(H)`

`FROM_{i,t}(H)=100 \times \sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ij,t}^{g}(H)`

`NET_{i,t}(H)=TO_{i,t}(H)-FROM_{i,t}(H)`

`NPDC_{ij,t}(H)=100 \times (\tilde{\theta}_{ji,t}^{g}(H)-\tilde{\theta}_{ij,t}^{g}(H))`

### Symbol notes
- \(\mathbf{A}_{\ell,t}\): time-varying autoregressive coefficient matrices.
- \(\boldsymbol{\Sigma}_t\): time-varying innovation covariance matrix.
- \(\tilde{\theta}_{ij,t}^{g}(H)\): normalized generalized FEVD share.
- \(TCI_t\): system-wide total connectedness.
- \(TO, FROM, NET\): directional and net transmitter/receiver metrics.
- \(NPDC_{ij,t}\): pairwise net directional connectedness from \(j\) to \(i\).

### Method statement for the paper
The empirical core is TVP-VAR-SV, estimated on the six-dimensional return vector. Time-varying coefficients absorb evolving propagation channels, while stochastic volatility captures heteroskedastic and episodic uncertainty. Based on generalized FEVD, we compute total, directional, net, and pairwise net connectedness in each period, which allows us to track both system-wide stress transmission and asset-level transmitter/receiver switching. This framework is the main identification device of the paper because it moves beyond comovement and recovers directional spillover structure over time. **In this paper, the role of Section 4.2 is to deliver the baseline time-varying directional connectedness results.**

---

## 4.3 TVP-VAR-BK

### Why this method is used
Directional connectedness can differ across horizons. To distinguish short-lived reactions from persistent transmission, we extend the TVP-VAR connectedness framework into the frequency domain using the BK decomposition. This is an extension layer built on the same TVP-VAR backbone, not an alternative main model.

### Core formulas

#### (A) Frequency-domain FEVD
Let \(\omega\in[-\pi,\pi]\) denote frequency and \(\mathbf{\Psi}_t(e^{-i\omega})\) the time-varying transfer function. The generalized causation spectrum is written as:

**Standard display**

\[
\phi_{ij,t}(\omega)=\frac{\sigma_{jj,t}^{-1}\left|\left(\mathbf{\Psi}_t(e^{-i\omega})\boldsymbol{\Sigma}_t\right)_{ij}\right|^2}{\left(\mathbf{\Psi}_t(e^{-i\omega})\boldsymbol{\Sigma}_t\mathbf{\Psi}_t(e^{+i\omega})'\right)_{ii}}.
\]

For a frequency band \(d=(a,b)\subset(0,\pi)\), the band-specific FEVD share is

**Standard display**

\[
\theta_{ij,t}^{g}(d)=\frac{\int_{a}^{b}\phi_{ij,t}(\omega)\,d\omega}{\sum_{j=1}^{N}\int_{0}^{\pi}\phi_{ij,t}(\omega)\,d\omega},
\quad
\tilde{\theta}_{ij,t}^{g}(d)=\frac{\theta_{ij,t}^{g}(d)}{\sum_{j=1}^{N}\theta_{ij,t}^{g}(d)}.
\]

**MathType linear input**

`\phi_{ij,t}(\omega)= \frac{\sigma_{jj,t}^{-1}|(\mathbf{\Psi}_t(e^{-i\omega})\boldsymbol{\Sigma}_t)_{ij}|^2}{(\mathbf{\Psi}_t(e^{-i\omega})\boldsymbol{\Sigma}_t\mathbf{\Psi}_t(e^{+i\omega})')_{ii}}`

`\theta_{ij,t}^{g}(d)= \frac{\int_{a}^{b}\phi_{ij,t}(\omega)d\omega}{\sum_{j=1}^{N}\int_{0}^{\pi}\phi_{ij,t}(\omega)d\omega}`

`\tilde{\theta}_{ij,t}^{g}(d)= \theta_{ij,t}^{g}(d) / \sum_{j=1}^{N}\theta_{ij,t}^{g}(d)`

#### (B) Band connectedness
**Standard display**

\[
TCI_t(d)=100\times\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ij,t}^{g}(d),
\]

\[
TO_{i,t}(d)=100\times\sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ji,t}^{g}(d),
\quad
FROM_{i,t}(d)=100\times\sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ij,t}^{g}(d),
\]

\[
NET_{i,t}(d)=TO_{i,t}(d)-FROM_{i,t}(d).
\]

**MathType linear input**

`TCI_t(d)=100 \times (1/N)\sum_{i=1}^{N}\sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ij,t}^{g}(d)`

`TO_{i,t}(d)=100 \times \sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ji,t}^{g}(d)`

`FROM_{i,t}(d)=100 \times \sum_{j=1,j\ne i}^{N}\tilde{\theta}_{ij,t}^{g}(d)`

`NET_{i,t}(d)=TO_{i,t}(d)-FROM_{i,t}(d)`

### Symbol notes
- \(\phi_{ij,t}(\omega)\): generalized causation spectrum at frequency \(\omega\).
- \(d=(a,b)\): pre-defined frequency band.
- \(\theta_{ij,t}^{g}(d)\): band-specific generalized FEVD contribution.
- \(TCI_t(d), TO_{i,t}(d), FROM_{i,t}(d), NET_{i,t}(d)\): frequency-specific connectedness measures.

### Horizon interpretation and implementation logic
We partition the spectrum into short-term, medium-term, and long-term bands, where short-term captures high-frequency (transitory) reactions, medium-term captures intermediate adjustment, and long-term captures low-frequency (persistent) transmission. In implementation, TVP-VAR-BK inherits the same baseline lag length, forecast horizon, and core identification choices used in TVP-VAR-SV, so that differences are attributable to horizon decomposition rather than model inconsistency. Hence, TVP-VAR-BK is explicitly treated as a frequency extension of TVP-VAR connectedness, not a substitute for the baseline SV framework. **In this paper, the role of Section 4.3 is to decompose the baseline directional connectedness into horizon-specific channels.**

---

## 4.4 Robustness strategy

Robustness checks are designed to test whether the main economic message is stable, rather than to redefine the main model. First, under DCC, we keep DCC-GARCH(4,4)-t as the benchmark and evaluate alternative marginal volatility specifications as sensitivity checks. Second, for TVP-VAR-SV, we conduct burn-in robustness by varying initial training windows and state initialization lengths, and then verify whether directional rankings and net transmitter/receiver roles remain qualitatively unchanged. Third, for TVP-VAR-BK, we run parameter robustness experiments (e.g., alternative lag/horizon settings and reasonable band partitions) while preserving comparability with the baseline TVP-VAR-SV setup. Across all checks, the assessment criterion is whether the dominant connectedness direction and core spillover topology are stable through time.

**In this paper, the role of Section 4.4 is to confirm the stability of baseline conclusions, not to replace the baseline framework.**
