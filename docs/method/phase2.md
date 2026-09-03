# Phase 2. w_S Decay Evidence

Phase 2는 핵심 질문에 답한다: **유전 분산몫 $V_G$와 공유환경 분산몫 $V_S$가 data에서 분리될 수 있는가?**

분리 가능성은 공유환경 세대간 감쇠율 $w_S$에 달려 있다(유전 감쇠율은 $0.5$로 고정). $w_S$가 0.5에서 충분히 멀어야 genetic decay와 shared-env decay가 다르게 보이고, $V_G$와 $V_S$를 따로 추정할 수 있다.

v1은 slope test로 $w_S$를 fast / slow / similar-decay 3-class로 구분해 분리 가능 여부를 판단했다. v2는 3-class를 넘어 $w_S$의 연속 추정값 **하나**를 반환한다. 그 추정을 얼마나 믿을지는 Phase 2가 라벨로 달지 않는다 — $w_S$ 불확실성은 Phase 3의 profile $w$-CI가 정량화한다(맨 끝 브릿지 참고).

## Input & Output

$$
\text{Phase 2}\left[(\hat{\boldsymbol{\rho}},\hat{\Sigma})\right]
\rightarrow
\hat{w}_{S,\text{cal}}
$$

- $\hat{\boldsymbol{\rho}}$ : DOR(친척 촌수 $d=1,2,3$)별 상대상관(relative correlation, phenotype similarity) vector ($D \times 1$, $D=3$ 고정)
- $\hat{\Sigma}$ : $\hat{\boldsymbol{\rho}}$의 covariance matrix ($D \times D$)
- $\hat{w}_{S,\text{cal}}$ : feature-calibrated shared environment decay rate — **Phase 2의 유일한 출력** (clip 범위 $(0.01, 0.99)$)

Phase 2는 $V_G, V_S$를 추정하지 않고 $w_S$ 점추정 하나만 낸다. 24개 진단 feature vector $\mathbf{x}$는 이 값을 만드는 ridge 입력으로 내부에서만 쓰이고(Step 1), Phase 2 출력이 아니다. $V_G, V_S$는 Phase 3가 $\hat{w}_{S,\text{cal}}$를 고정값으로 받아 refit한다. 자세한 handoff는 맨 끝 **Output → Phase 3** 섹션 참고.

## 왜 w_S 추정이 어려운가

### Identifiability Problem

공통 모델은 DOR $d$의 상관을 유전·공유환경 두 감쇠항의 합으로 쓴다:

$$
\rho_d = 0.5^{\,d}\,V_G + w_S^{\,d-1}\,V_S,\qquad d = 1, 2, 3
$$

유전 계수 $0.5^d$는 **genetic 감쇠율이 $0.5$로 고정**(친척이 한 촌수 멀어질 때 유전 공유는 정확히 ½)이라 나온다. 공유환경 계수 $w_S^{d-1}$은 $d=1$에서 $V_S$ 전량, 이후 세대마다 $w_S$ 배로 감쇠한다.

가장 어려운 경우는 $w_S = 0.5$다. 이때 shared environment도 genetic처럼 DOR마다 $0.5$씩 감소한다. 모델에 $w_S = 0.5$를 넣으면:

$$
\rho_d = 0.5^d V_G + 0.5^{d-1} V_S
$$

$0.5^{d-1} = 2 \cdot 0.5^d$이므로:

$$
\rho_d = 0.5^d (V_G + 2V_S)
$$

data는 $V_G + 2V_S$만 본다. $V_G$와 $V_S$를 따로 볼 수 없다. 즉 여러 조합이 같은 $\rho_d$를 만든다:

$$
V_G + 2V_S = C \;\Rightarrow\; \text{same fitted curve}
$$

예:

$$
(V_G, V_S) = (0.4,\, 0.1)
\quad\text{과}\quad
(V_G, V_S) = (0.2,\, 0.2)
$$

는 둘 다 $V_G + 2V_S = 0.6$이므로 동일한 $\rho_d$ pattern을 만든다.

수학적으로는 design matrix의 rank가 떨어지는 문제다. $w_S$가 0.5와 충분히 달라야 genetic decay curve $0.5^d$와 shared-env decay curve $w_S^{d-1}$이 서로 다르게 보이고, $V_G$와 $V_S$를 분리할 수 있다.

### Contrast Structure

$w_S$를 추정하려면 contrast structure를 봐야 한다. 두 contrast를 정의한다:

$$
D_2 = \rho_2 - \frac{\rho_1}{2} = (w_S - 0.5)\, V_S
$$

$$
D_3 = \rho_3 - \frac{\rho_2}{2} = w_S(w_S - 0.5)\, V_S
$$

유도:

$$
\rho_1 = \frac{1}{2}V_G + V_S
\;\Rightarrow\;
\frac{\rho_1}{2} = \frac{1}{4}V_G + \frac{1}{2}V_S
$$

$$
D_2 = \rho_2 - \frac{\rho_1}{2}
     = \frac{1}{4}V_G + w_S V_S
     - \frac{1}{4}V_G - \frac{1}{2}V_S
     = \left(w_S - \frac{1}{2}\right)V_S
$$

$$
\frac{\rho_2}{2} = \frac{1}{8}V_G + \frac{w_S}{2}V_S
$$

$$
D_3 = \rho_3 - \frac{\rho_2}{2}
     = \frac{1}{8}V_G + w_S^2 V_S
     - \frac{1}{8}V_G - \frac{w_S}{2}V_S
     = w_S\left(w_S - \frac{1}{2}\right)V_S
$$

noise가 없으면:

$$
w_S = \frac{D_3}{D_2}
$$

### 세 가지 실패 상황

문제는 $D_2$가 분모로 들어간다는 것이다.

$$
D_2 = (w_S - 0.5)\,V_S
$$

다음 세 상황에서 $D_2$가 0에 가까워진다.

```text
w_S ≈ 0.5:
  genetic decay (0.5^d)와 shared-env decay (w_S^d)가 거의 같음
  -> identifiability 문제. design matrix rank 하락.

V_S 작음:
  shared environment signal 자체가 약함
  -> D_2 = (w_S - 0.5) * V_S 자체가 작아짐

SE가 큼:
  rho_hat에 noise가 많음
  -> D_2, D_3 모두 불안정
```

따라서:

$$
\text{small error in } \hat{\rho}
\;\rightarrow\;
\text{large error in } D_2,D_3
\;\rightarrow\;
\text{larger error in } D_3/D_2
\;\rightarrow\;
\text{biased } \hat{w}_S
\;\rightarrow\;
\text{biased } \hat{V}_G, \hat{V}_S
$$

v1과 v2는 이 문제를 다르게 다룬다.

## BIGFAM.v1 — Slope Test

v1은 $\hat{\boldsymbol{\rho}}$에 slope test를 적용해서 $w_S$의 방향(fast vs slow)을 분류하고, 그 range 안에서 joint optimization을 한다.

흐름:

```text
rho_hat
-> log₂ 변환 → slope 추정 (OLS, equal weights)
-> slope CI로 fast / slow / similar-decay 분류
-> class별 w_S search range 제한
-> range 안에서 log-scale BIGFAM loss 최소화
-> best (V_G, V_S, w_S)
```

### Slope Test

slope의 직관:

```text
공통 모델에서 genetic component만 있다면:
  rho_d = 0.5^d V_G → log₂(rho_d) = -d + log₂(V_G)
  slope = 1

shared env도 genetic과 같은 rate로 decay (w_S = 0.5)하면:
  rho_d = (V_G + 2V_S) 0.5^d → log₂(rho_d) = -d + const
  slope = 1 (identifiability ridge)

w_S < 0.5 (공유환경이 genetic보다 빨리 사라짐):
  DOR1에서 rho가 상대적으로 크고, 고 DOR에서 빨리 떨어짐
  → slope > 1 (faster than genetic decay)

w_S > 0.5 (공유환경이 genetic보다 느리게 사라짐):
  고 DOR에서도 rho가 천천히 감소
  → slope < 1 (slower than genetic decay)
```

CI는 $\hat{\Sigma}$를 이용한 parametric bootstrap으로 구한다.

**Step 1 — Parametric bootstrap**

$$
\hat{\boldsymbol{\rho}}^{(b)} \sim \mathcal{N}(\hat{\boldsymbol{\rho}},\, \hat{\Sigma}), \quad b = 1, \ldots, B
$$

**Step 2 — OLS slope per bootstrap sample**

$$
y_d^{(b)} = \log_2\!\left(\hat{\rho}_d^{(b)}\right)
\qquad
\hat{\beta}_1^{(b)} = \text{OLS slope of } \mathbf{y}^{(b)} \text{ on } -d
$$

**Step 3 — Bootstrap CI**

$$
\text{CI}_{\text{lo}} = Q_{0.025}\!\left(\hat{\beta}_1^{(1:B)}\right),
\qquad
\text{CI}_{\text{hi}} = Q_{0.975}\!\left(\hat{\beta}_1^{(1:B)}\right)
$$

### Slope Class 분류

bootstrap CI를 기준으로 class를 결정한다.

```text
CI_lo > 1  → fast          (w_S < 0.5)
CI_hi < 1  → slow          (w_S > 0.5)
CI 1 포함  → similar-decay (w_S ≈ 0.5, 분리 어려움)
```

### Class별 w_S Range + Joint Optimization

Step 3에서는, 이 결과를 기반으로, slope class에 따라 $w_S$ search range를 제한한다.

```text
fast     → w_S ∈ [0.01, 0.49]
slow     → w_S ∈ [0.51, 0.99]
similar-decay → w_S ∈ [0.40, 0.60]
```

해당 range 안에서 log-scale BIGFAM loss를 최소화한다.

$$
(\hat{V}_G^{(v1)}, \hat{V}_S^{(v1)}, \hat{w}_S^{(v1)})
=
\arg\min_{\substack{V_G,V_S,w_S \\ w_S \in \text{range}}}
\mathcal{L}(V_G, V_S, w_S \mid \hat{\boldsymbol{\rho}})
$$

### v1의 구조적 한계

**OLS slope: log space SE hetero 무시.**

bootstrap CI의 폭은 log-space에서의 slope 분산에 달려 있다. 그런데 OLS slope point estimate 자체가 equal-weight 평균이라 log space에서 SE가 큰 고 DOR에 과도하게 끌린다.

```text
log₂(rho_hat_d)의 SD ≈ sigma_d / (rho_d · ln2)

rho가 작은 고 DOR일수록 log space에서 noise 큼.
OLS는 각 DOR에 equal weight → 고 DOR noise가 slope estimate를 흔듦.
bootstrap CI가 넓어지거나, 잘못된 방향으로 기울어짐.
```

**Jensen's inequality bias (2차 항).**

$\log_2(\hat{\rho}_d)$에 sampling noise가 있으면:

$$
E\!\left[\log_2(\hat{\rho}_d)\right] < \log_2\!\left(E[\hat{\rho}_d]\right)
$$

→ slope가 체계적으로 fast 방향으로 약간 밀린다.

**Coarse classification.**

fast / slow / similar-decay 3개뿐이다. 같은 "fast" 안에서도 $w_S = 0.01$과 $w_S = 0.45$는 range가 다른데, v1은 이를 같은 범위로 처리한다.

**$D_2$ denominator issue 유지.**

range를 정한 뒤에도 $V_G, V_S, w_S$를 jointly 최적화 → $D_2$가 약할 때 $w_S$ 오차가 $V_G, V_S$에 그대로 전파된다.

**불확실성 미표기.**

어떤 상황에서도 숫자 하나를 낸다. weak data에서 그 estimate를 얼마나 믿어야 할지 알려 주는 수단이 없다. v2는 이 불확실성을 Phase 2에서 라벨로 달지 않고, Phase 3의 profile $w$-CI로 넘겨 정량화한다(맨 끝 브릿지).

## BIGFAM.v2

v2는 $w_S$를 feature calibration으로 먼저 추정한다. v1처럼 $(V_G, V_S, w_S)$를 한꺼번에 joint optimization하지 않는다 — $w_S$만 먼저 풀고 $V_G/V_S$ 분해와 $w_S$ 불확실성 정량화는 Phase 3로 미룬다.

```text
v1: (V_G, V_S, w_S) 동시 joint optimization
v2: w_S 먼저 (feature calibration) → V_G,V_S 분해와 w_S 불확실성은 Phase 3
```

### Pipeline 한눈에

$$
(\hat{\boldsymbol\rho}, \hat\Sigma)
\;\xrightarrow{\text{Step 1}}\;
\mathbf{x}\ (24\ \text{features})
\;\xrightarrow{\text{Step 2}}\;
\hat{w}_{S,\text{cal}}
$$

| step | 입력 → 출력 | 한 줄 요약 |
|---|---|---|
| 1. Feature 추출 | (ρ̂, Σ) → x | $w_S$ 신호를 4각도(slope·profile·contrast·raw)에서 측정 |
| 2. w_S calibration | x → ŵ_S,cal | ridge regression (24 feature, 표준화) |

두 step 모두 **observed $(\hat{\boldsymbol\rho}, \hat\Sigma)$만** 쓴다. 학습(simulation) 때도 true $V_G, V_S, w_S$를 feature에 넣지 않는다 — inference에서 그 값이 없기 때문. simulation은 오직 calibration의 **계수를 학습**하는 데만 truth를 쓴다.

---

### Step 1 — Feature Extraction (24 features)

비율 하나에 기대지 않고 $(\hat{\boldsymbol\rho}, \hat\Sigma)$를 네 각도에서 요약한다. 전부 $(\hat{\boldsymbol\rho}, \hat\Sigma)$의 닫힌형 함수라 실데이터에서 추가 계산 없이 나온다.

| 그룹 | 개수 | 무엇을 보나 |
|---|---:|---|
| Slope (1a) | 3 | 닮음이 유전보다 빨리 식나 느리게 식나 |
| Profile (1b) | 7 | 어떤 $w_S$가 데이터와 맞나, 그리고 얼마나 뾰족한가 |
| Contrast (1c) | 7 | 유전 성분을 뺀 환경 신호를 직접 ($D_2$·$D_3$) |
| Raw / SE (1d) | 7 | 관측값과 오차 규모 그대로, 건전성 플래그 포함 |

$w_S$를 **직접** 재는 것은 이 중 4개뿐이다(`w_map`·`w_mean`·`w_median`·`ratio_naive`). 나머지 20개는 그 측정을 얼마나 믿을지를 보는 보조 단서다.

$(\hat{\boldsymbol{\rho}}, \hat{\Sigma})$에서 네 그룹의 feature를 만든다. 각 그룹은 $w_S$ 신호를 다른 각도에서 본다.

| group | 무엇을 본다 | features (개수) |
|---|---|---|
| Slope | decay가 genetic보다 빠른가/느린가 | slope_hat, slope_se, slope_z (3) |
| Profile | 어떤 $w_S$가 data와 맞는가 + 그 확신 | w_map, w_mean, w_median, profile_width, middle_mass, effective_count, map_mean_gap (7) |
| Contrast | genetic 제거 후 $w_S$ 방향·크기·안정성 | D_2_hat, D_3_hat, I_D2, I_D3, ratio_naive, fieller_bounded, fieller_width (7) |
| Raw+flag | 관측 rho·SE level | rho_hat_1,2,3, se_max, se_mean, signal_rms_z, any_nonpos (7) |

> 24개 중 ridge 계수는 **`w_median`에 집중**($\text{coef} \approx 0.229$)되고, 나머지 23개 feature는 모두 $|\text{coef}| < 0.025$의 보정 tail이다. `w_median`(profile 중심의 robust 추정)이 점추정을 이끌고, slope·contrast·raw feature가 그 값을 미세 조정한다.

#### 1a. Slope Features (Jensen-corrected GLS)

v1은 $\log_2(\hat{\rho}_d)$에 OLS를 적용해 slope를 추정했다. 세 가지 문제가 있었다.

```text
문제 1 — OLS: SE hetero 무시 (주된 문제):
  DOR1 (pair 많음, SE 작음) = DOR3 (pair 적음, SE 큼) 동일 weight
  → 노이즈 큰 DOR3가 slope를 흔듦
  → FPR ↑, sensitivity ↓
  simulation: OLS FPR_fast ≈ 15–18%, WLS FPR_fast ≈ 2%

문제 2 — Jensen bias (2차 항):
  E[log₂(rho_hat_d)] < log₂(rho_d)
  → slope가 fast 방향으로 약간 밀림

문제 3 — hard classifier:
  slope_z 부호 → fast/slow/similar-decay 3-class로 끊음
  → 연속 정보 손실
```

v2는 이 세 문제를 모두 고쳐서 slope를 **soft feature**로 포함한다.

**① Jensen correction**

$\hat{\rho}_d$에 sampling noise가 있으면 log 변환 후 bias가 생긴다.

$$
E\!\left[\log_2(\hat{\rho}_d)\right]
\approx
\log_2(\rho_d) - \frac{\hat{\Sigma}_{dd}}{2\rho_d^2 \ln 2}
$$

보정 후 transformed observation:

$$
y_d^*
=
\log_2(\hat{\rho}_d) + \frac{\hat{\Sigma}_{dd}}{2\hat{\rho}_d^2 \ln 2}
$$

**② Delta method로 full $\hat{\Sigma}$ 전파**

$y_d^*$는 $\hat{\rho}_d$만의 함수이므로 Jacobian은 대각이다.

$$
J_{dd}
=
\frac{\partial y_d^*}{\partial \hat{\rho}_d}
=
\frac{1}{\hat{\rho}_d \ln 2}
\left(1 - \frac{\hat{\Sigma}_{dd}}{\hat{\rho}_d^2}\right)
$$

$\mathbf{J} = \mathrm{diag}(J_{11}, J_{22}, J_{33})$으로 두면, $\mathbf{y}^*$의 covariance:

$$
\mathbf{V}^* = \mathbf{J}\,\hat{\Sigma}\,\mathbf{J}
$$

$\mathbf{V}^*$는 full $\hat{\Sigma}$의 off-diagonal도 반영한다. $\hat{\Sigma}$가 diagonal이면 $V^*_{dd} = J_{dd}^2 \hat{\Sigma}_{dd}$가 되어 delta-method WLS와 동일하다.

**③ GLS slope 추정**

design matrix (intercept + slope):

$$
\mathbf{H} =
\begin{bmatrix}
1 & -1 \\
1 & -2 \\
1 & -3
\end{bmatrix}
$$

GLS estimate:

$$
\hat{\boldsymbol{\beta}}^*
=
(\mathbf{H}^T [\mathbf{V}^*]^{-1} \mathbf{H})^{-1}
\mathbf{H}^T [\mathbf{V}^*]^{-1} \mathbf{y}^*
$$

slope와 SE:

$$
\hat{\beta}_1^* = [\hat{\boldsymbol{\beta}}^*]_2,
\qquad
\widehat{\mathrm{SE}}(\hat{\beta}_1^*)
=
\sqrt{\bigl[(\mathbf{H}^T [\mathbf{V}^*]^{-1} \mathbf{H})^{-1}\bigr]_{22}}
$$

**Features**

```text
slope_hat:  hat_beta_1*  — genetic decay (slope=1) 대비 실제 감소 속도
slope_se:   SE(hat_beta_1*)
slope_z:    (hat_beta_1* - 1) / SE(hat_beta_1*)

  slope_z > 0: fast (w_S < 0.5)
  slope_z < 0: slow (w_S > 0.5)
  slope_z ≈ 0: identifiability ridge 근처
```

v1에서 slope_z 부호를 hard classifier로 썼던 것과 달리, v2는 calibration regression의 soft feature로만 쓴다. 아래 profile/contrast feature와 함께 calibration model이 weight를 결정한다.

---

#### 1b. Profile Features (NNLS profile-out)

slope는 decay 방향을 요약하지만 $w_S$ 값 자체는 주지 않는다. profile features는 $w_S$ grid 전체를 model에 넣어 fit quality를 비교함으로써 $w_S$의 위치와 불확실성을 직접 측정한다.

$w_S$ 후보를 grid로 잡는다.

$$
w_S^{(k)} \in \{0.01, 0.02, \ldots, 0.99\}
$$

각 후보마다 design matrix를 만든다. ($D=3$ 기준)

$$
X(w_S^{(k)})
=
\begin{bmatrix}
0.5 & 1 \\
0.25 & w_S^{(k)} \\
0.125 & (w_S^{(k)})^2
\end{bmatrix}
$$

$X(w_S^{(k)})$는 $D \times 2$이다.

고정된 $w_S^{(k)}$에서 model은 $\boldsymbol{\beta} = (V_G, V_S)^T$에 대해 linear하다.

GLS loss:

$$
\ell(w_S^{(k)})
=
\min_{\boldsymbol{\beta} \geq 0}
\left(
\hat{\boldsymbol{\rho}} - X(w_S^{(k)})\boldsymbol{\beta}
\right)^T
\hat{\Sigma}^{-1}
\left(
\hat{\boldsymbol{\rho}} - X(w_S^{(k)})\boldsymbol{\beta}
\right)
$$

해석:

```text
ℓ(w_S^(k)) 작다:
  그 w_S가 observed rho curve를 잘 설명함

ℓ(w_S^(k)) 크다:
  그 w_S는 data와 안 맞음
```

여기서 $\boldsymbol\beta = (V_G, V_S)^T \geq 0$ 제약은 **proper NNLS**(KKT active-set, 2변수)로 푼다. unconstrained GLS 해가 음수 분산을 주면 그 성분을 0으로 고정하고 **나머지를 다시 최적화**한다 (단순 clip은 profile loss를 왜곡). 각 후보 $w_S^{(k)}$에서 $\ell$을 계산하면 profile curve가 생긴다.

profile을 확률처럼 보기 위해 softmax weight를 정의한다:

$$
p_k = \frac{\exp(-\ell(w_S^{(k)})/2)}{\sum_j \exp(-\ell(w_S^{(j)})/2)}
$$

$-\ell/2$는 Gaussian log-likelihood에 해당 → $p_k$는 후보 $w_S^{(k)}$의 "그럴듯함". 이 curve에서 7개 feature를 뽑는다:

$$
w_{\text{map}} = \arg\min_k \ell(w_S^{(k)}),
\qquad
w_{\text{mean}} = \sum_k p_k\, w_S^{(k)},
\qquad
w_{\text{median}} = \big\{w_S^{(k)} : \textstyle\sum_{j\le k} p_j = 0.5\big\}
$$

$$
\text{profile\_width} = \big|\{w_S : \ell(w_S) \le \ell_{\min} + 3.84\}\big|,
\qquad
\text{middle\_mass} = \sum_{k:\,w_S^{(k)} \in [0.35,\,0.65]} p_k
$$

$$
\text{effective\_count} = \exp\!\Big(-\sum_k p_k \log p_k\Big),
\qquad
\text{map\_mean\_gap} = |w_{\text{map}} - w_{\text{mean}}|
$$

각 항의 의미:

```text
w_map / w_mean / w_median:  profile 중심 추정 (3가지 방식)
  - w_map: 최고점. profile 평평하면 튐
  - w_median: 누적 50% 지점. 분포 전체를 봐서 robust (ridge가 가장 무겁게 쓰는 feature)

profile_width:  Δℓ < 3.84(χ²_1,0.95) 구간 폭
  - 좁다 → 특정 w_S만 잘 맞음 (확실)
  - 넓다 → 여러 w_S가 비슷하게 설명 (불확실)

middle_mass:    [0.35,0.65]에 몰린 확률
  - 크면 identifiability ridge(0.5) 근처에 mass 많음

effective_count:  몇 개 후보가 살아있나 (perplexity). 클수록 불확실
map_mean_gap:     봉우리와 평균의 어긋남. 비대칭/불안정 신호
```

profile_width·middle_mass·effective_count·map_mean_gap은 $w_S$ 값 자체보다 **그 추정의 확신도**를 잰다. ridge에서는 이들이 `w_median` 점추정을 미세 보정하는 tail($|\text{coef}| < 0.025$)로 들어간다.

#### 1c. Contrast Features (D₂/D₃ + Fieller)

profile은 model fit 관점이다.

contrast는 $w_S$ ratio structure를 직접 본다.

$$
\hat{D}_2 = \hat{\rho}_2 - \frac{\hat{\rho}_1}{2}
\qquad
\hat{D}_3 = \hat{\rho}_3 - \frac{\hat{\rho}_2}{2}
$$

$\hat{D}_2$와 $\hat{D}_3$는 $\hat{\boldsymbol{\rho}}$의 linear combination이므로 uncertainty는 $\hat{\Sigma}$에서 exact하게 propagation된다.

$\mathbf{c}_2 = (-\tfrac{1}{2},\ 1,\ 0)^T$라 하면 $\hat{D}_2 = \mathbf{c}_2^T\hat{\boldsymbol{\rho}}$이므로:

$$
\widehat{\mathrm{Var}}(\hat{D}_2)
=
\mathbf{c}_2^T \hat{\Sigma} \mathbf{c}_2
=
\tfrac{1}{4}\hat{\Sigma}_{11}
+ \hat{\Sigma}_{22}
- \hat{\Sigma}_{12}
$$

마찬가지로 $\mathbf{c}_3 = (0,\ -\tfrac{1}{2},\ 1)^T$:

$$
\widehat{\mathrm{Var}}(\hat{D}_3)
=
\tfrac{1}{4}\hat{\Sigma}_{22}
+ \hat{\Sigma}_{33}
- \hat{\Sigma}_{23}
$$

information score:

$$
I_{D_2} = \frac{|\hat{D}_2|}{\widehat{\mathrm{SE}}(\hat{D}_2)}
\qquad
I_{D_3} = \frac{|\hat{D}_3|}{\widehat{\mathrm{SE}}(\hat{D}_3)}
$$

해석:

```text
I_D2 크다:
  denominator D_2가 noise보다 충분히 큼
  w_S ratio 추정이 비교적 안전

I_D2 작다:
  denominator가 noise에 묻힘
  ratio를 그대로 믿으면 위험
```

ratio estimate:

$$
\text{ratio\_naive} = \frac{\hat{D}_3}{\hat{D}_2}
$$

이것은 $w_S$의 직접 추정이 아니라 **feature**다.

```text
ratio_naive:
  w_S가 어느 방향인지 알려주는 signal로만 쓴다.
  noisy denominator 때문에 그대로 최종값으로 쓰지 않는다.
```

Fieller feature:

$$
H_0:\ D_3 - w_S D_2 = 0
$$

이 가설을 pivot으로 ratio $D_3/D_2$의 신뢰구간을 만들고, 그 구간이 유한한지(bounded)와 폭(width)을 feature로 뽑는다 — `fieller_bounded`, `fieller_width`.

```text
Fieller bounded (fieller_bounded = 1):
  denominator D_2가 충분히 강함
  -> ratio가 통계적으로 유한한 구간에 갇힘

Fieller unbounded (fieller_bounded = 0):
  denominator가 너무 약함
  -> 어떤 w_S도 데이터와 불일치한다고 배제하기 어려움 (fieller_width 큼)
```

이 둘은 ridge에 들어가는 **입력 feature**이지, 별도의 판정 규칙(예전 Fieller confidence set)이 아니다. denominator 강도를 ridge가 다른 feature와 함께 저울질하는 재료로만 쓴다.

#### 1d. Raw / SE Features

model을 거치지 않은 관측값 그대로다. 다른 세 그룹의 feature가 놓친 level 정보나 noise 구조를 calibration model이 직접 활용할 수 있게 한다.

$$
\hat{\rho}_1,\ \hat{\rho}_2,\ \hat{\rho}_3,
\qquad
\text{se\_mean} = \frac{1}{D}\sum_d \hat{\sigma}_d,
\qquad
\text{se\_max} = \max_d \hat{\sigma}_d
$$

전체 signal/noise 비와 음수 발생 플래그:

$$
\text{signal\_rms\_z} = \sqrt{\frac{1}{D}\sum_{d=1}^{D} \frac{\hat\rho_d^2}{\hat\Sigma_{dd}}},
\qquad
\text{any\_nonpos} = \mathbb{1}\big[\exists d:\ \hat\rho_d \le 0\big]
$$

`signal_rms_z`는 각 DOR의 (signal/SE)²를 평균 후 root — 작을수록 rho_hat이 noise에 묻혀 모든 추정이 불안정. `any_nonpos`는 noise가 커서 음의 상관이 관측됐다는 약신호 플래그.

이유:

```text
같은 w_map이라도 SE가 크면 믿음의 강도가 다르다.
같은 ratio라도 rho scale이 다르면 bias pattern이 다를 수 있다.
signal_rms_z / any_nonpos는 "이 데이터 자체가 약한가"를 직접 잰다 → ridge 보정 tail.
```

### Step 2 — w_S Calibration (ridge)

24개 feature를 합쳐 $\mathbf{x}$를 만든 뒤, simulation으로 학습한 ridge regression으로 $w_S$를 추정한다:

$$
\hat{\mathbf{b}}
=
\arg\min_{\mathbf{b}}
\sum_i \left(w_{S,\text{true},i} - b_0 - \mathbf{x}_i^T \mathbf{b}\right)^2
+ \lambda \|\mathbf{b}\|^2,
\qquad
\hat{w}_{S,\text{cal}} = \mathrm{clip}(b_0 + \mathbf{x}^T \hat{\mathbf{b}},\ 0.01,\ 0.99)
$$

feature는 StandardScaler로 표준화해 입력하고, penalty $\lambda$는 RidgeCV(KFold 5)로 고른다($\lambda=1$). penalty $\lambda\|\mathbf{b}\|^2$는 feature collinearity(예: w_map ≈ w_mean)에서 계수를 안정화한다. intercept $b_0 = 0.501$은 학습셋 $w_S$ 평균이라, feature에 정보가 없으면 추정이 0.5(식별 불가 지점)로 수렴한다.

```text
학습(simulation):  truth w_S로 ridge b 적합
inference(real):   observed (ρ̂,Σ) → x → ŵ_S,cal  (truth 불필요)
```

#### 학습된 계수 — 사실상 `w_median` 하나

24개를 다 쓰는 블랙박스가 아니다. 출하 아티팩트(`src/artifacts/ws_calibration.json`)의 표준화 feature 계수를 절댓값 순으로 보면 하나가 지배한다.

```text
w_median         +0.2285    ← 지배 (profile 우도의 중앙값)
w_mean           -0.0245    2등이자 최대 보정항
D_2_hat          +0.0141    contrast 분모 신호
fieller_bounded  +0.0071    비율 CI가 유계인가
w_map            +0.0065    profile mode
```

`w_median` 계수가 2등의 9.3배다. 즉 이 회귀의 실체는 **profile 우도 중앙값을 살짝 보정한 것**이다. $w_S$를 직접 재는 feature는 4개(`w_map`·`w_mean`·`w_median`·`ratio_naive`)뿐인데, ridge는 그중 분산이 낮고 강건한 `w_median`을 주축으로 삼는다. `w_mean`은 `w_median`과 거의 공선이라 독립 정보가 아니라 **차이(= median − mean = skew) 보정**으로 들어가고, 그래서 직접 추정치인데 부호가 음수다. `ratio_naive`는 분모가 0에 가까울 때 폭발해 사실상 무시된다. 나머지는 전부 $|\text{coef}| < 0.025$의 보정 tail이다.

#### 학습 데이터 — wide DGP

ridge 계수(`ws_calibration.json`)가 학습되는 공간이다. 한 sample을 그리는 **draw 순서는 재현성을 위해 동결된 계약**이다:

```text
1. w_S ~ U(0.01, 0.99)               연속 — v1의 이산 3-class가 아님
2. (V_G, V_S) ~ Dirichlet(1,1,1)     심플렉스 균일
3. 상관행렬 R: 비대각 r₁₂,r₁₃,r₂₃ ~ U(0, 0.9) free
             PSD 아니면 rejection 재추출
4. sd ~ U(0.001, 0.10) (정렬 안 함), Σ = diag(sd) R diag(sd)
5. ρ_true = 공통 모델, ρ̂ = ρ_true + chol(Σ) z,  z ~ N(0, I)

N_SIM = 40000,  SEED = 42,  전 행 split = "train"
```

**"wide"인 이유**: $w_S$를 전 구간에서 연속으로 뽑고 $(V_G, V_S)$·상관·SE까지 넓게 흩뿌려, real data가 마주칠 다양한 상황(강신호·약신호·0.5 근처·off-diagonal 상관)을 모두 학습셋에 담기 때문. 좁은 격자로 학습하면 그 사이 값에서 계수가 편향된다. 이 표가 곧 ridge가 외우는 세상이라, 실데이터가 이 support 밖이면 calibration은 근거 없는 외삽이다.

학습은 offline 1회다. `PYTHONPATH=src .venv/bin/python src/scripts/train_phase2.py` → `src/artifacts/ws_calibration.json`. inference는 그 파일을 읽기만 한다.

## Output → Phase 3

Phase 2 출력은 숫자 하나다:

$$
\hat{w}_{S,\text{cal}}
$$

- calibrated shared-env decay rate (Step 2), clip 범위 $(0.01, 0.99)$.
- 24 feature $\mathbf{x}$는 이 값을 만든 ridge 입력일 뿐, Phase 2 출력이 아니다.
- 신뢰도 라벨도, per-point $w_S$ SE도 없다. $w_S$ 불확실성은 Phase 3의 profile $w$-CI가 전부 짊어진다.

### Phase 3 handoff

Phase 3는 $\hat{w}_{S,\text{cal}}$을 고정하고 Step 1b profile과 **동일한 NNLS**로 $V_G, V_S$를 refit한다:

$$
(\hat{V}_G, \hat{V}_S)
=
\arg\min_{\boldsymbol\beta \ge 0}
\bigl\|\hat{\boldsymbol\rho} - X(\hat{w}_{S,\text{cal}})\,\boldsymbol\beta\bigr\|^2_{\hat\Sigma^{-1}}
$$

**$w_S$ 불확실성은 어디로 갔나 (브릿지).** v2 Phase 2는 $w_S$ 점추정 하나만 넘긴다. 그 추정을 얼마나 믿을지는 Phase 3의 **profile $w$-CI**가 담당한다 — $w$ grid 전체에서 joint GLS 우도를 훑어 신뢰구간을 잡는다. 그 구간이 0.5를 덮으면 분해가 식별되지 않는다 → [phase3](phase3.md).

### Phase 간 I/O 계약

```text
Phase 1 output:  rho_hat, Sigma          (phenotype type 몰라도 됨)
Phase 2 output:  w_S,cal                  (V_G, V_S 몰라도 됨)
Phase 3 input:   rho_hat, Sigma, w_S,cal
Phase 3 output:  V_G, V_S (conditional SE), w_S CI
```

## References

- `src/artifacts/ws_calibration.json` — Step 2 계수·intercept의 실값 출처.
- `src/bigfam/phase2/dgm.py`·`train.py` — wide DGP 구현과 학습 경로. draw 순서 계약의 코드 출처.
- `research/v2-fieller-refactor/explain/step2b-ridge.md` — narrow → wide 재기준선 근거(corr 0.996, RMSE 0.158 → 0.154, intercept 0.500 → 0.501).
- `research/method-validation/explain/00-figures.md` — Step 1·2가 옳게 도는지의 시뮬레이션 검증(Fig 3·4).
- `docs/method/phase3.md` — 고정 $\hat{w}_{S,\text{cal}}$에서 $V_G/V_S$ 분해와 profile $w$-CI. 본 문서가 $w_S$ 불확실성을 넘기는 곳.
