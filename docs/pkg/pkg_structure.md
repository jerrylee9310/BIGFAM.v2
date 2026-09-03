# bigfam 패키지 — 설계 원칙과 배치

`src/bigfam`가 왜 이 모양인지만 적는다. 함수 시그니처·상수값·스키마는 **코드가 단일출처**다. 여기 옮겨 적으면 사본이 되고, 사본은 코드가 움직일 때 조용히 썩는다.

## 설계 원칙 5개

**1. Phase 경계 = 모듈 경계.** Phase 1은 형질 종류(연속/이진)를 알지만 Phase 2·3은 $(\hat\rho, \hat\Sigma)$만 본다. 이 I/O 계약을 문서가 아니라 코드에서 강제한다.

**2. 순수 함수 우선.** 각 계산 단계는 입력에서 출력으로 가는 순수 함수다. 파일 IO·print 같은 side effect는 `io/`와 CLI에만 둔다.

**3. Phase 연결은 dataclass로.** dict가 아니라 `types.py`의 frozen dataclass가 경계를 건넌다. 타입이 곧 계약이라, 필드가 사라지면 호출부가 즉시 깨진다.

**4. `docs/method/*.md`와 1:1.** 각 함수 docstring이 대응 수식 절을 명시한다(예: `phase2.md Step 1b`). 수식이 바뀌면 어느 함수를 고칠지가 문서에서 바로 나온다.

**5. 학습과 추론의 분리.** Phase 2 ridge 계수는 시뮬레이션으로 offline 학습해 `artifacts/`에 저장하고, 추론은 로드만 한다. 추론 아티팩트는 `ws_calibration.json` 하나뿐이다.

## 배치

```text
src/bigfam/
├── types.py            Phase 간 데이터 객체 (RhoEstimate·WsEstimate·Decomposition·CalibrationCoef)
├── config.py           전 상수 단일출처 (D, 격자, clip, seed, eps)
│
├── phase1/             친척 쌍 → (ρ̂, Σ̂)          [형질 종류를 아는 유일한 층]
│   ├── api.py            estimate_rho()
│   ├── pairs.py          flip & concat, pair table
│   ├── continuous.py     residualize + OLS
│   ├── binary.py         bivariate probit MLE
│   └── sandwich.py       cluster sandwich covariance (연속·이진 공통)
│
├── phase2/             (ρ̂, Σ̂) → ŵ_S
│   ├── api.py            estimate_ws() — 점추정 하나만 낸다
│   ├── features/         24 features. __init__의 FEAT_ALL이 순서 단일출처
│   │                     slope · profile · contrast · raw
│   ├── calibrate.py      표준화 → ridge
│   ├── dgm.py            wide DGP (offline 전용)
│   └── train.py          dgm → features → ridge 계수 (offline 전용)
│
├── phase3/             (ρ̂, Σ̂, ŵ_S) → (V_G, V_S, conditional SE, w-CI)
│   ├── api.py            decompose()
│   ├── refit.py          고정-w NNLS GLS + A⁻¹ conditional SE
│   └── robust.py         profile w-CI
│
├── core/               Phase 공유 수치 유틸 (nnls · design · linalg)
└── io/                 파일 입출력 (load · save) — 순수 계산과 분리
```

`artifacts/ws_calibration.json`이 유일한 추론 입력이고, CLI는 `src/scripts/{run_pipeline, train_phase2}.py` 둘이다. 테스트는 `src/tests/`에 phase별로 있다.

## 여기 없는 것

| 찾는 것 | 어디 |
|---|---|
| 함수 시그니처·인자 | 해당 모듈의 docstring |
| 상수 실값 | `src/bigfam/config.py` |
| 24 feature의 이름·순서 | `src/bigfam/phase2/features/__init__.py`의 `FEAT_ALL` |
| 출력 파일 스키마 | `src/bigfam/io/save.py` |
| 의존성 | `pyproject.toml` |
| 수식 유도 | `docs/method/phase0-3.md` |
| 왜 이 설계로 갔나의 경위 | `research/v2-fieller-refactor/`·`research/phase2-weight-recal/` |

## References
- `docs/README.md` — 모델·3-phase 분할 근거.
- `src/README.md` — 패키지 quickstart.
