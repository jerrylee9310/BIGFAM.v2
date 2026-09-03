# Phase 3. V_G / V_S Decomposition

**핵심 아이디어**: Phase 2가 준 $\hat w_S$를 **고정**하고 같은 NNLS로 $(V_G, V_S)$를 읽는다. 분해 정확도는 $w_S$가 0.5에서 얼마나 먼지(= design matrix conditioning)가 좌우한다. 불확실성은 두 축으로 갈라 그대로 보고한다 — $\hat{\boldsymbol\rho}$ 잡음이 남기는 conditional SE와, $w_S$ 자체의 profile 95% CI다.

## Input & Output

$$
\text{Phase 3}\left[(\hat{\boldsymbol\rho},\ \hat\Sigma,\ \hat w_S)\right]
\rightarrow
(\hat V_G,\ \hat V_S,\ \widehat{\mathrm{SE}}_{\text{cond}},\ z,\ \text{w-CI})
$$

- $\hat{\boldsymbol\rho},\ \hat\Sigma$ : Phase 1 출력 — DOR별 상대상관 $(\hat\rho_1,\hat\rho_2,\hat\rho_3)$와 그 $3\times3$ cluster-sandwich 공분산.
- $\hat w_S$ ($=\hat w_{S,\text{cal}}$) : Phase 2 출력 (calibrated decay rate 점추정) — **고정값으로 받음**. clip 범위 $(0.01, 0.99)$.
- 출력 : 성분 $(\hat V_G, \hat V_S)$ + conditional SE + 조건부 $z$ + profile w-CI.

Phase 3는 $w_S$를 다시 추정하지 않는다. $\hat w_S$에서 분산 성분을 분리하고(Step 1–2), 그 분리가 언제 불가능한지 짚은 뒤(Step 3), $w_S$ 자체의 신뢰구간을 세운다(Step 4).

**신뢰 등급 라벨은 내지 않는다.** 분해가 식별되는지는 Step 4의 w-CI가 0.5를 덮는지로 그대로 읽힌다 — 덮으면 $X(w)$가 rank 1인 지점이 데이터와 양립하므로 $(V_G, V_S)$가 안 갈린다. 어디서 자를지는 추정기가 아니라 그걸 쓰는 분석이 정한다.

---

## Step 1 — Refit (fixed-$w_S$ GLS / NNLS)

$w_S$를 고정하면 공통 모델은 분산몫 $\boldsymbol\beta = (V_G, V_S)^T$(유전 $V_G$·공유환경 $V_S$)에 대해 **linear**하다:

$$
\hat{\boldsymbol\rho} = X(w_S)\,\boldsymbol\beta + \boldsymbol\varepsilon,
\qquad
X(w_S) =
\begin{bmatrix}
0.5 & 1 \\
0.25 & w_S \\
0.125 & w_S^2
\end{bmatrix}
$$

각 열의 의미: 1열 $c_G = (0.5^d)_d$ = genetic decay, 2열 $c_S = (w_S^{d-1})_d$ = shared-env decay. $\boldsymbol\varepsilon \sim \mathcal{N}(0, \Sigma)$.

$w_S = \hat w_S$에서의 GLS 해 (= 제약 없는 interior NNLS):

$$
A = X^\top \hat\Sigma^{-1} X,
\qquad
\hat{\boldsymbol\beta}
=
A^{-1}\,X^\top \hat\Sigma^{-1} \hat{\boldsymbol\rho}
$$

여기서 $A$는 **고정-$w$ Fisher 정보**로 SE의 출처다(Step 2). 분산이 음수가 되면 ($V_G < 0$ 또는 $V_S < 0$) $\boldsymbol\beta \ge 0$ 제약을 **proper NNLS**(KKT active-set)로 처리한다 — `core/nnls` 공용 solver로, Step 4 profile이 격자 각 점에서 푸는 것과 동일하다. (0.5 특이점에서 수치가 죽지 않게 refit은 $A$에 상대 ridge $10^{-9}\,\mathrm{tr}(A)\,I$만 더한다 — 0.5에서 먼 곳엔 영향이 무시할 크기.)

**경계해 주의**: NNLS가 어느 성분을 정확히 0에 고정하면 그 성분은 경계에 놓인 것이라 대칭 Wald SE가 부정확해진다. $\hat V_G = 0$은 "유전 기여가 없다"는 유효한 답일 수 있으므로 결과에서 배제하지 않는다. 다만 그 형질의 $z$는 Wald 가정 밖이라 그대로 읽으면 안 된다.

> 일관성: 이 $\hat{\boldsymbol\beta}$는 Step 4 profile이 만드는 $\boldsymbol\beta(w)$ 곡선을 $w = \hat w_S$에서 평가한 값과 정확히 같다. Phase 3는 새 estimator가 아니라 "확정된 $w_S$에서 분산 읽기(Step 1–2) + 그 주변에서 흔들어 보기(Step 4)"다.

---

## Step 2 — Conditional SE ($\hat{\boldsymbol\rho}$ 잡음)

$w_S = \hat w_S$ 고정·$\hat{\boldsymbol\rho} \sim \mathcal{N}(\boldsymbol\rho, \Sigma)$일 때, $\hat{\boldsymbol\beta}$는 $\hat{\boldsymbol\rho}$의 linear map이므로 covariance가 exact하다:

$$
\mathrm{Cov}(\hat{\boldsymbol\beta} \mid \hat w_S)
=
(X^\top \hat\Sigma^{-1} X)^{-1}
= A^{-1}
$$

$$
\widehat{\mathrm{SE}}_{\text{cond}}(\hat V_G) = \sqrt{[A^{-1}]_{11}},
\qquad
\widehat{\mathrm{SE}}_{\text{cond}}(\hat V_S) = \sqrt{[A^{-1}]_{22}}
$$

이 **conditional SE**가 Phase 3가 보고하는 SE다. "conditional"은 $\hat w_S$를 고정했을 때라는 뜻 — $\hat{\boldsymbol\rho}$의 측정오차만 $V$에 남긴 sampling SE이고, $w_S$ **자체의** 불확실성은 여기 들어가지 않는다.

**왜 $w_S$ 불확실성을 여기 안 더하나.** $w_S$ 오차를 delta method로 이 SE에 접어 넣던 옛 방식(full SE)은 0.5 근처에서 잔차 항이 $\hat\Sigma^{-1}$(작은 $\sigma$라 큼)에 곱해져 폭발했고, CI가 과대해져 over-coverage가 났다. 그래서 v2는 $w_S$ 불확실성을 SE에 섞지 않고 **별도 축**(Step 4 profile w-CI)으로 분리해 보고한다.

같이 보고하는 조건부 유의도:

$$
z_{V_G} = \frac{\hat V_G}{\widehat{\mathrm{SE}}_{\text{cond}}(\hat V_G)},
\qquad
z_{V_S} = \frac{\hat V_S}{\widehat{\mathrm{SE}}_{\text{cond}}(\hat V_S)}
$$

$\mathrm{se} = 0$이면 $z$는 NaN이다. 이 $z$도 "$\hat w_S$가 정답이라면"이라는 조건부 양이라, $w_S$ 불확실성은 Step 4의 w-CI와 같이 읽어야 한다.

---

## Step 3 — Decomposition Identifiability (왜 0.5가 치명적인가)

분해가 가능하려면 $A = X^\top\hat\Sigma^{-1}X$가 invertible — 즉 $X$의 두 열 $c_G, c_S$가 **선형독립**이어야 한다.

$w_S = 0.5$를 대입하면:

$$
c_S = (1,\ 0.5,\ 0.25)^\top = 2\,(0.5,\ 0.25,\ 0.125)^\top = 2\,c_G
$$

두 열이 **정확히 비례** → $X$ rank 1 → $A$ singular → $V_G$와 $V_S$를 분리할 수 없다. data는 $V_G + 2V_S$만 본다:

$$
\rho_d = 0.5^d V_G + 0.5^{d-1} V_S = 0.5^d(V_G + 2V_S)
$$

이것은 Phase 2 맨 위 identifiability 문제가 **분해 단계에서 그대로 재현**된 것이다. $w_S$가 0.5에서 멀수록 $c_S$가 $c_G$에서 벌어져 $A$가 well-conditioned해진다.

**수치 예시** ($\hat\Sigma = (0.01)^2 I$, $\kappa(A)$는 $\sigma$ 무관):

| $w_S$ | $\kappa(A)$ | $\widehat{\mathrm{SE}}_{\text{cond}}(V_G)$ | $\widehat{\mathrm{SE}}_{\text{cond}}(V_S)$ |
|---|---|---|---|
| 0.10 | 31 | 0.043 | 0.025 |
| 0.30 | 120 | 0.081 | 0.044 |
| 0.40 | 493 | 0.160 | 0.084 |
| 0.45 | 2021 | 0.319 | 0.164 |
| **0.50** | **≈ 2×10¹⁶ (singular)** | **∞** | **∞** |
| 0.55 | 2177 | 0.320 | 0.155 |
| 0.70 | 163 | 0.082 | 0.036 |
| 0.90 | 60 | 0.044 | 0.016 |

0.5에서 멀어질수록 SE가 급감 (0.45→0.10에서 $\widehat{\mathrm{SE}}_{\text{cond}}(V_G)$ 0.32 → 0.04, 8배). conditioning이 분해 정밀도를 직접 결정한다. 이 발산은 버그가 아니라 "$V_G$·$V_S$를 분리할 수 없다"는 정직한 신호다. 별도 라벨을 붙이지 않아도 SE가 커지고 $z$가 내려가는 것으로 그대로 드러나며, Step 4의 w-CI가 0.5를 덮는 것이 같은 사실의 구간판이다.

---

## Step 4 — Profile w-CI ($w_S$ 불확실성)

conditional SE는 $\hat w_S$를 **정답으로 믿었을 때**의 오차만 잰다. 하지만 $\hat w_S$ 자체가 틀릴 수 있다. 이 불확실성을 SE에 접는 대신(Step 2에서 봤듯 0.5 근처에서 폭발한다), $w_S$를 **격자 위에서 훑어** 데이터가 허용하는 $w$ 범위를 직접 세운다. 두 단계다 — **(1)** 각 $w$에서 다시 적합해 그 $w$가 데이터와 얼마나 맞는지 재고 → **(2)** 그중 충분히 잘 맞는 것만 남겨 신뢰구간을 만든다.

**(1) 각 $w$에서 재적합 + misfit.** 격자 `WS_PROFILE`:

$$
\text{WS\_PROFILE} = \{\,w \in \text{linspace}(0.01, 0.99, 99) : |w - 0.5| > 10^{-9}\,\}
$$

(0.5는 특이라 제외). 각 $w$를 참이라 치고 고정-$w$ NNLS GLS로 분해를 다시 푼다:

$$
\boldsymbol\beta(w) = \bigl(V_G(w),\ V_S(w)\bigr) = \arg\min_{\boldsymbol\beta \ge 0}\ \bigl(\hat{\boldsymbol\rho} - X(w)\boldsymbol\beta\bigr)^\top \hat\Sigma^{-1} \bigl(\hat{\boldsymbol\rho} - X(w)\boldsymbol\beta\bigr).
$$

그 최적점에서 두 양을 읽는다:

$$
e(w) = \hat{\boldsymbol\rho} - X(w)\boldsymbol\beta(w),
\qquad
\ell(w) = e(w)^\top \hat\Sigma^{-1} e(w).
$$

- $e(w)$ = **잔차** — 데이터 $\hat{\boldsymbol\rho}$에서 그 $w$의 모델 예측 $X(w)\boldsymbol\beta(w)$을 뺀 나머지. 그 $w$의 모델이 설명하지 못한 부분이다.
- $\ell(w)$ = **misfit** — 그 잔차를 $\hat\Sigma^{-1}$로 가중한 제곱합, 스칼라 하나. "그 $w$에서 모델이 데이터를 얼마나 못 맞췄나"의 총점으로 **작을수록 그 $w$가 데이터와 잘 맞는다** (3-DOR joint 우도의 편차 deviance이기도 하다). $w$를 훑으면 제일 잘 맞는 $w$에서 골짜기(최소 $\ell_{\min}$)를 이루고, 멀어질수록 올라간다.

**왜 misfit을 $\ell(w)$로 재나.** $\ell$은 행렬 역이 필요 없는 스칼라라, $A$가 특이해지는 0.5 근처에서도 **항상 유한하게 정의된다**. conditional SE(Step 2)는 0.5에서 터지지만 $\ell$은 안 터진다 — $w_S$의 신뢰집합을 세울 안정된 재료가 된다.

**(2) 데이터가 허용하는 $w$ 범위 → profile w-CI.** 골짜기 바닥 $\ell_{\min}$에서 $\chi^2_{1,0.95} = 3.84$ 이상 나빠지지 않는 $w$들만 남긴다:

$$
\text{w-CI} = \{\,w \in \text{WS\_PROFILE} : \ell(w) \le \ell_{\min} + 3.84\,\},
\qquad \ell_{\min} = \min_w \ell(w).
$$

이 구간의 양끝 $(\text{wci\_lo}, \text{wci\_hi})$이 3-DOR 적합 자체가 말해 주는 $w_S$의 95% 신뢰구간이다. (컷 $3.84 = \chi^2_{1,0.95}$의 근거는 Wilks 정리 — misfit 차 $\ell(w) - \ell_{\min}$이 참 $w$에서 $\chi^2_1$을 따른다.)

w-CI가 0.5를 포함하면 그 형질은 분해가 식별되지 않는다. Step 3에서 본 대로 $w_S=0.5$는 $X(w)$가 rank 1이 되는 지점이고, 데이터가 그 지점을 배제하지 못했다는 뜻이기 때문이다. 반대로 구간이 0.5를 제외하면 $H_0: w_S = 0.5$를 $\chi^2_{1,0.95}$에서 기각한 것과 같다 — 문턱이 검정에서 나오지 튜닝에서 나오지 않는다.

Phase 3는 여기서 멈추고 $(\text{wci\_lo}, \text{wci\_hi})$를 그대로 내보낸다. 이 구간을 어디서 자를지, 그 컷으로 형질을 어떻게 묶을지는 분석마다 다르므로 추정기가 정하지 않는다.

---

## 왜 이 방식인가 (vs v1 joint optimization)

```text
v1:  (V_G, V_S, w_S)를 동시에 최소화
     → w_S 오차가 V_G/V_S에 nonlinear하게 얽혀 분리 불가
     → 어디서 오차가 왔는지 추적 못 함

v2:  w_S 먼저 (Phase 2 ridge)
     → 고정 후 linear refit (Phase 3 Step 1)
     → ρ̂ 잡음 = conditional SE (A⁻¹), w_S 불확실성 = profile w-CI로 분리
     → 두 축을 뭉치지 않고 그대로 보고
```

v2는 분해 불확실성을 **두 독립 축**($A^{-1}$ conditional SE + profile w-CI)으로 분리한다. 한 숫자로 합치지 않으므로 "무엇이 분해를 망쳤는가"를 읽을 수 있다. $\hat{\boldsymbol\rho}$가 시끄러우면 conditional SE가 커지고, $w_S$가 안 잡히면 w-CI가 넓어지며 0.5를 덮는다. 두 원인은 대처가 다르다 — 앞은 표본을 늘리면 줄고, 뒤는 모델 구조상 그렇지 않다(Step 3).

---

## 왜 raw 대신 요약 $(\hat{\boldsymbol\rho}, \hat\Sigma)$로 refit하나 (two-stage)

자연스러운 의문: $V_G, V_S$를 raw individual pair 데이터로 직접 적합하면 estimate·SE가 더 정확하지 않을까? 답은 **거의 같다** — Phase 1이 이미 individual-level 정보를 $(\hat{\boldsymbol\rho}, \hat\Sigma)$에 담았기 때문이다.

```text
비유:  Phase 1 = raw 재료를 손질해 반찬(ρ̂) + 품질표(Σ)로 만듦
       Phase 3 = 그 반찬만 보면 됨. raw로 다시 요리 = 같은 답, 시간만 더 씀
```

Phase 1이 한 일:

- **continuous**: pair residual OLS + **cluster sandwich** $\hat\Sigma = B^{-1}MB^{-1}$, $M = M_1 + M_2 - M_{12}$ → 한 개인이 여러 DOR pair에 겹쳐 등장하는 **overlap을 이미 보정**. 그 cross-DOR 상관이 $\hat\Sigma$의 off-diagonal에 들어 있다.
- **binary**: 모든 DOR를 **joint bivariate probit MLE**로 적합 → 이미 individual-level likelihood.

따라서 $(\hat{\boldsymbol\rho}, \hat\Sigma)$는 단순 요약이 아니라 **raw에서 짜낼 정보를 담은 sufficient·robust 통계량**이다.

고정 $w_S$에서 $\hat{\boldsymbol\rho}$를 $X(w_S)\boldsymbol\beta$에 맞추는 structured GLS는, efficient weight $\hat\Sigma^{-1}$을 쓰면 **raw에 직접 적합한 constrained MLE와 점근 등가**다 (minimum-distance / minimum-$\chi^2$ 정리). 즉 individual-level refit은 같은 답을 더 비싸게 구하는 redundant 작업이다.

**전제 (off-diagonal 필수)**: 이 등가는 $\hat\Sigma$가 **full sandwich covariance**일 때만 성립한다. weight를 diagonal로 근사하면 overlap 정보를 버려 SE가 틀어진다 → Step 1·2·4의 weight는 항상 full $\hat\Sigma^{-1}$.

**예외**: cluster(개인) 수가 적으면 sandwich $\hat\Sigma$ 자체가 noisy → GLS weight 불안정. 이때만 weight를 diagonal 쪽으로 shrinkage하는 정칙화를 고려한다.

단, raw로도 **못 고치는 것**: $w_S \approx 0.5$의 $V_G/V_S$ 분리 불가는 모델 구조($c_S = 2c_G$)라 데이터를 더 봐도 해결되지 않는다 (Step 3).

---

**다음**: 이 $(\hat V_G, \hat V_S, \widehat{\mathrm{SE}}_{\text{cond}}, \text{w-CI})$는 형질당 `.decomp`로 직렬화된다. 여기서 w-CI가 0.5를 제외하는 형질을 모으면 공유환경 신호를 집단 수준에서 보는 스크리닝이 되는데, 그 컷과 후속 분석은 이 문서 밖이다.

---
## References

- Aitken (1935), generalized least squares — 고정 $w_S$에서 $\hat{\boldsymbol\beta} = A^{-1}X^\top\hat\Sigma^{-1}\hat{\boldsymbol\rho}$와 $\mathrm{Cov} = A^{-1}$의 출처.
- Wilks (1938), likelihood-ratio $\to \chi^2$ — profile w-CI의 컷 $\ell \le \ell_{\min} + 3.84$ ($=\chi^2_{1,0.95}$)의 근거.
- [phase2.md](phase2.md) (calibrated ridge, identifiability) — $\hat w_S$의 출처이자 본 문서 $A$ conditioning과 동일 구조.
- `bigfam/phase3/{refit, robust}.py` — 본 문서 Step 1–2 / Step 4 수식의 구현.
- `research/method-validation/explain/00-figures.md` — Step 3 conditioning 표(Fig 5)와 conditional SE 검증(Fig 7)의 시뮬레이션 근거.
