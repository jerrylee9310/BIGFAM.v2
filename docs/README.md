# docs/ — 프로젝트 지식

BIGFAM의 **모델과 유도** 단일출처. src 코드 docstring이 여기를 참조한다.

여기엔 지금 참인 것만 둔다. 실험 기록·결과 수치·폐기된 설계의 경위는 `research/`, 형질 단위 결과값은 `db/`, 구현 세부는 `src/`가 각각 단일출처다. 같은 사실을 두 층에 쓰지 않는다 — 사본은 반드시 한쪽이 먼저 썩는다.

## 무엇을 푸는가

형질 $y$는 세 성분의 합이다. $y_i = g_i + s_i + \epsilon_i$, $V_G + V_S + V_E = 1$. 질문은 "형질이 닮은 이유가 유전인가 같이 살아서인가", 즉 $V_G$ 대 $V_S$다. 둘 다 직접 관측되지 않는다. 대신 **친척 쌍**을 보고 촌수(DOR) $d$가 멀어질 때 닮음이 어떻게 식는지를 본다.

결정적 단서는 유전과 환경이 **식는 속도가 다르다**는 것이다. 유전 기여는 촌수 한 단계마다 정확히 절반이고($w_G = 0.5$, 생물학으로 고정), 공유환경 기여는 미지의 속도 $w_S$로 식는다. 합치면 공통 모델이다.

$$
\rho_d = w_G^{\,d}\,V_G + w_S^{\,d-1}\,V_S, \qquad w_G = 0.5
$$

미지수는 $V_G, V_S, w_S$ 셋이고 DOR도 셋($d = 1,2,3$)이라 풀릴 것 같지만, 노이즈가 이 분해를 매우 불안정하게 만든다. 그래서 단계를 나눈다.

## 왜 $w_S = 0.5$가 중심인가

$w_S = 0.5$이면 공유환경도 유전과 똑같은 속도로 식는다.

$$
\rho_d = 0.5^d V_G + 0.5^{d-1} V_S = 0.5^d\,(V_G + 2V_S)
$$

데이터가 $V_G + 2V_S$ **한 숫자만** 본다. $(V_G, V_S) = (0.40, 0.10)$과 $(0.20, 0.20)$이 완전히 같은 곡선을 만든다. 표본을 아무리 늘려도 안 갈린다. 설계행렬 $X(w)$의 두 열이 비례해 rank 1이 되기 때문이다.

$w_S$가 0.5에서 멀수록 두 곡선이 다르게 보여 분해가 정밀해진다. 이 "0.5로부터의 거리"가 Phase 3의 conditioning과 profile $w$-CI를 하나로 꿰뚫는 양이다.

## 왜 3단계인가

BIGFAM.v1은 $(V_G, V_S, w_S)$ 셋을 한꺼번에 log-scale에서 최적화했다. $w_S$ 오차가 나머지 둘에 비선형으로 얽혀 오차의 출처를 추적할 수 없고, log-OLS가 측정 공분산 $\hat\Sigma$를 완전히 무시한다. v2는 분리한다. $w_S$를 먼저 점추정하고(Phase 2), 그 값을 고정한 채 깔끔한 선형 분해를 한다(Phase 3).

```text
친척 쌍 (phenotype · 나이·성별 covariate · 촌수 d)
    │
 [Phase 1]  flip&concat → covariate 제거 → DOR별 similarity → cluster sandwich
    ▼
 ρ̂ = (ρ̂₁, ρ̂₂, ρ̂₃)  +  Σ̂ (3×3)          ← 여기서 형질 종류가 사라진다
    │
 [Phase 2]  24 features → w_S calibration (ridge)
    ▼
 ŵ_S,cal  (점추정 하나)
    │
 [Phase 3]  w_S 고정 → 조건부 분해 (NNLS GLS) → profile w-CI
    ▼
 V̂_G, V̂_S  +  conditional SE  +  w_S의 95% CI
```

경계는 타입으로 강제된다. 각 단계는 앞 단계의 출력 객체만 본다.

| Phase | 입력 | 출력 | 모르는 것 |
|---|---|---|---|
| 1 | 친척 쌍 (phenotype, covariate, $d$) | $\hat{\boldsymbol\rho},\ \hat\Sigma$ | — |
| 2 | $\hat{\boldsymbol\rho},\ \hat\Sigma$ | $\hat w_{S,\text{cal}}$ | 형질 종류 |
| 3 | $\hat{\boldsymbol\rho},\ \hat\Sigma,\ \hat w_{S,\text{cal}}$ | $\hat V_G,\ \hat V_S$ (conditional SE), $w$-CI | 형질 종류 |

이 계약 덕에 Phase 2·3은 연속형이든 이진형이든 같은 코드로 돈다. Phase 1만 형질 종류를 안다.

불확실성은 두 갈래로 갈린다. $w_S$를 고정했을 때 $\hat{\boldsymbol\rho}$ 노이즈가 $V$에 남기는 **conditional SE**, 그리고 $w_S$ 자체의 불확실성을 담는 **profile $w$-CI**다. 한 숫자로 뭉치지 않는다. 신뢰 등급 라벨은 내지 않는다 — $w$-CI가 0.5를 덮는지가 곧 식별 여부이고, 어디서 자를지는 추정기가 아니라 분석의 몫이다.

## 파일

| 파일 | 내용 |
|---|---|
| `method/phase0.md` | 공통 모델·기호·연속/이진 phenotype scale·component covariance |
| `method/phase1.md` | 친척 쌍 → $(\hat\rho, \hat\Sigma)$. residualization, cluster sandwich |
| `method/phase2.md` | $(\hat\rho, \hat\Sigma)$ → $\hat w_S$. 24 features, ridge calibration, wide DGP |
| `method/phase3.md` | $w_S$ 고정 조건부 분해. conditioning, conditional SE, profile $w$-CI |
| `pkg/pkg_structure.md` | 패키지 설계 원칙과 배치 |

각 주제는 한 파일에만 있다. 같은 내용을 쉬운 문체로 다시 쓴 사본을 두지 않는다 — 직관 설명이 필요하면 그 주제를 다루는 파일의 도입부에 넣는다.

문서가 지금 코드와 어긋나는지는 `.venv/bin/python tools/check_stale.py`로 확인한다. `src/` 이력에 있다가 사라진 이름이 여기 남아 있으면 잡아낸다.

## 여기 없는 것

| 찾는 것 | 어디 |
|---|---|
| phase 메커니즘이 실제로 도는지의 시뮬 검증 | `research/method-validation/` |
| v1 대 v2 정량 비교 | `research/bigfam-benchmark/` |
| 형질별 분해 결과값 | `db/02_bigfam_results/` · `research/curated-bigfam/` |
| 외부 기준(SNP-h²) 대조 | `db/05_comparison/` · `research/snph2-method-compare/` |
| 폐기된 설계의 경위 | 해당 `research/<slug>/explain/` |

> LLM 운영 규칙(환경·경계·브랜치·문서 스타일)은 `../rules/`.
