# Phase 1. DOR-Level Evidence

Phase 1은 relative pair data를 DOR-level evidence로 바꾼다.

## Notation

flip & concat 이후 전 section에서 공통으로 쓰는 기호를 한 곳에 정의한다.

**Data**

- $N$ : flip & concat 이후 전체 row 수
- $D$ : DOR level 수
- $p$ : covariate 수 (intercept 제외)
- $\mathbf{Y}_1, \mathbf{Y}_2$ : 표현형 column, $N \times 1$
- $X_1, X_2$ : covariate matrix, $N \times p$. $X_{kn}$은 $k$번째 member의 $n$번째 row ($1 \times p$)
- $\mathbf{g}$ : DOR label vector, $N \times 1$. $g_n \in \{1,\ldots,D\}$
- $id_{1n},\, id_{2n}$ : row $n$에서 첫 번째, 두 번째 member의 individual ID

**Design**

- $W_k = [\mathbf{1},\, X_k]$ : member $k$의 covariate design matrix, $N \times (1+p)$. $W_{kn}$은 $n$번째 row ($1 \times (1+p)$)
- $A$ : DOR one-hot matrix, $N \times D$. $A_{nd} = 1$ iff $g_n = d$

**Parameters**

- $\boldsymbol{\gamma}$ : covariate effect (intercept 포함), $(1+p) \times 1$
- $\boldsymbol{\rho}$ : DOR-level phenotype similarity vector, $D \times 1$. $\rho_d$는 DOR $d$의 similarity

**Estimates & Output**

- $\hat{\boldsymbol{\rho}}$ : $D \times 1$
- $\hat{\Sigma} = \widehat{\mathrm{Cov}}(\hat{\boldsymbol{\rho}})$ : $D \times D$
- $\hat{\sigma}_d = \sqrt{\hat{\Sigma}_{dd}}$ : DOR $d$의 standard error

## Input & Output

$$
\text{Step1}\bigl[(y_i,y_j),\,(\mathbf{X}_i,\mathbf{X}_j),\,g_{ij}\bigr]
\rightarrow
(\hat{\boldsymbol{\rho}},\,\hat{\Sigma})
$$

- $(y_i, y_j)$ : flip & concat 이전 individual-level 표현형 pair
- $(\mathbf{X}_i, \mathbf{X}_j)$ : individual-level covariate pair
- $g_{ij}$ : DOR label of pair $(i,j)$

## Undirected Pair Construction

pair 방향 선택에 따른 estimate 변동을 방지하기 위해 continuous/binary 모두 먼저 flip & concat한다.

```text
(i, j)  →  (i, j), (j, i)
```

flip & concat 이후 column 형태:

```text
original direction:
  Y_1 = phenotype of i,  X_1 = covariates of i
  Y_2 = phenotype of j,  X_2 = covariates of j

flipped direction:
  Y_1 = phenotype of j,  X_1 = covariates of j
  Y_2 = phenotype of i,  X_2 = covariates of i
```

이후 $n$번째 row를 subscript $n$으로 표시한다: $Y_{1n},\, Y_{2n},\, X_{1n},\, X_{2n},\, g_n$.

## Cluster Sandwich Covariance

한 사람이 여러 쌍에 중복으로 들어간다. 어떤 개인은 형제와도 사촌과도 쌍을 이룬다. 이 중복을 무시하고 쌍을 독립으로 세면 실제보다 표본이 많은 셈이 되어 **오차를 너무 작게 잡는다**. cluster sandwich는 같은 개인이 낀 쌍들을 한 cluster로 묶어 이를 보정한다. 이름은 분산 코어 $M$(meat)을 $B^{-1}$(bread) 두 장 사이에 끼우는 모양에서 왔다.

continuous와 binary가 공유하는 sandwich 구조를 먼저 정의한다.

각 section에서 row $n$의 score vector $\mathbf{s}_n$을 구체적으로 정의한다. 공통 구조는 아래와 같다.

overlap pair 때문에 row score를 individual cluster별로 합친다.

first-member cluster meat:

$$
M_1
=
\sum_i
\Bigl(\sum_{n:\,id_{1n}=i}\mathbf{s}_n\Bigr)
\Bigl(\sum_{n:\,id_{1n}=i}\mathbf{s}_n\Bigr)^T
$$

second-member cluster meat:

$$
M_2
=
\sum_j
\Bigl(\sum_{n:\,id_{2n}=j}\mathbf{s}_n\Bigr)
\Bigl(\sum_{n:\,id_{2n}=j}\mathbf{s}_n\Bigr)^T
$$

intersection cluster meat:

$$
M_{12}
=
\sum_{(i,j)}
\Bigl(\sum_{n:\,(id_{1n},id_{2n})=(i,j)}\mathbf{s}_n\Bigr)
\Bigl(\sum_{n:\,(id_{1n},id_{2n})=(i,j)}\mathbf{s}_n\Bigr)^T
$$

예:

```text
row 1: first=A, second=B, score=s_1
row 2: first=A, second=C, score=s_2
row 3: first=D, second=B, score=s_3

M_1:   A → s_1+s_2,  D → s_3
M_2:   B → s_1+s_3,  C → s_2
M_12:  (A,B)→s_1,  (A,C)→s_2,  (D,B)→s_3
```

$$
M = M_1 + M_2 - M_{12}
$$

```text
M_1:  first member overlap 보정
M_2:  second member overlap 보정
M_12: M_1과 M_2에서 중복으로 센 intersection 제거
```

bread $B$와 함께 sandwich covariance:

$$
\hat{\Sigma} = B^{-1}MB^{-1}
$$

## Continuous Phenotype

$$
\mathbf{Y}_1, \mathbf{Y}_2 \in \mathbb{R}^N
$$

covariate를 먼저 regress out한 뒤, residual 위에서 DOR similarity를 estimate한다.

### Stage 1 — Covariate Residualization

두 친척의 표현형이 닮은 이유가 친척이라서가 아니라 둘 다 남자여서, 또는 나이대가 비슷해서일 수 있다. 공변량으로 설명되는 부분을 먼저 제거하고 **남은 잔차끼리만** 닮음을 잰다.

pair에 등장하는 모든 개인에서 중복을 제거한 unique individual dataset:

- $I$ : unique individual 수
- $\mathbf{y}_\text{uniq}$ : 표현형 vector, $I \times 1$
- $W_\text{uniq} = [\mathbf{1},\, X_\text{uniq}]$ : $I \times (1+p)$

$$
\hat{\boldsymbol{\gamma}}
=
(W_\text{uniq}^T W_\text{uniq})^{-1} W_\text{uniq}^T \mathbf{y}_\text{uniq}
$$

$\hat{\boldsymbol{\gamma}}$를 pair data에 적용해 residual을 만든다:

$$
\tilde{\mathbf{Y}}_k = \mathbf{Y}_k - W_k\hat{\boldsymbol{\gamma}},
\quad k=1,2
$$

$\tilde{\mathbf{Y}}_1,\, \tilde{\mathbf{Y}}_2$는 각각 $N \times 1$이다.

### Stage 2 — DOR Similarity Estimation

residualized design matrix:

$$
\tilde{Z} = \mathrm{diag}(\tilde{\mathbf{Y}}_2)\,A
$$

$\tilde{Z}$는 $N \times D$이다. row $n$에서 $\tilde{Y}_{2n}$이 $g_n$번째 column에만 들어간다.

covariate가 제거된 residual 위에서 DOR별 similarity를 OLS로 estimate한다:

$$
\tilde{\mathbf{Y}}_1 = \tilde{Z}\boldsymbol{\rho} + \mathbf{e}
$$

$$
\hat{\boldsymbol{\rho}}
=
(\tilde{Z}^T\tilde{Z})^{-1}\tilde{Z}^T\tilde{\mathbf{Y}}_1
$$

### Score and Bread

$$
\hat{e}_n = \tilde{Y}_{1n} - \tilde{Z}_n\hat{\boldsymbol{\rho}},
\qquad
\mathbf{s}_n = \tilde{Z}_n^T\hat{e}_n
$$

$\tilde{Z}_n$은 $\tilde{Z}$의 $n$번째 row ($1 \times D$), $\mathbf{s}_n$은 $D \times 1$이다.

$$
B = \tilde{Z}^T\tilde{Z} \quad (D \times D)
$$

Cluster Sandwich Covariance 섹션 공식 적용 → $\hat{\Sigma}$는 $D \times D$.

## Binary Phenotype

$$
\mathbf{Y}_1, \mathbf{Y}_2 \in \{0,1\}^N
$$

observed 0/1에서 직접 correlation을 fit하지 않고 latent liability를 둔다.

### Liability Model

row $n$에서:

$$
Y_{kn} = \mathbf{1}(L_{kn} > 0),
\quad k=1,2
$$

$$
\begin{pmatrix}L_{1n}\\L_{2n}\end{pmatrix}
\sim
N\!\left(
\begin{pmatrix}W_{1n}\boldsymbol{\gamma}\\W_{2n}\boldsymbol{\gamma}\end{pmatrix},
\begin{pmatrix}1 & \rho_{g_n}\\\rho_{g_n} & 1\end{pmatrix}
\right)
$$

$W_{1n},\, W_{2n}$은 각각 $1 \times (1+p)$이다. liability mean은 intercept와 covariate effect $\boldsymbol{\gamma}$로 결정된다.

### MLE

모든 DOR를 jointly fit한다. parameter:

$$
\boldsymbol{\theta} = (\boldsymbol{\gamma},\, \boldsymbol{\rho}),
\quad (1+p+D) \times 1
$$

$$
(\hat{\boldsymbol{\gamma}},\hat{\boldsymbol{\rho}})
=
\arg\max_{\boldsymbol{\gamma},\boldsymbol{\rho}}
\sum_{n=1}^N
\log P\bigl(Y_{1n},Y_{2n}
\mid W_{1n},W_{2n},\boldsymbol{\gamma},\rho_{g_n}\bigr)
$$

### Score and Bread

$$
\mathbf{s}_n(\hat{\boldsymbol{\theta}})
=
\frac{\partial}{\partial\boldsymbol{\theta}}
\log P\bigl(Y_{1n},Y_{2n}
\mid W_{1n},W_{2n},\hat{\boldsymbol{\gamma}},\hat{\rho}_{g_n}\bigr)
$$

$\mathbf{s}_n$은 $(1+p+D) \times 1$이다.

$$
B
=
-\frac{\partial^2 \ell(\hat{\boldsymbol{\theta}})}{\partial\boldsymbol{\theta}\,\partial\boldsymbol{\theta}^T}
\quad \bigl((1+p+D)\times(1+p+D)\bigr)
$$

Cluster Sandwich Covariance 섹션 공식 적용 → $\widehat{\mathrm{Var}}(\hat{\boldsymbol{\theta}})$는 $(1+p+D)\times(1+p+D)$.

$\boldsymbol{\rho}$ block만 꺼낸다:

$$
\hat{\Sigma}
=
\bigl[\widehat{\mathrm{Var}}(\hat{\boldsymbol{\theta}})\bigr]_{\boldsymbol{\rho},\boldsymbol{\rho}}
$$

$\hat{\Sigma}$는 $D \times D$이다.

## Unified Evidence Object

continuous와 binary는 $\hat{\boldsymbol{\rho}}$를 만드는 model만 다르다. Phase 1이 끝나면 output은 같다:

$$
(\hat{\boldsymbol{\rho}},\,\hat{\Sigma})
$$

```text
continuous pair
-> flip & concat
-> pooled covariate regression → Y_tilde residuals
-> OLS on residuals over all DOR
-> cluster sandwich (D×D)
-> (rho_hat, Sigma_hat)

binary pair
-> flip & concat
-> joint bivariate probit over all DOR
-> cluster sandwich ((1+p+D)×(1+p+D)) → rho block
-> (rho_hat, Sigma_hat)
```

downstream step은 phenotype type을 직접 보지 않는다:

```text
downstream input:
  rho_hat (D×1) + Sigma_hat (D×D)
```
