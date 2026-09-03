# Phase 0. Common Model

## Core idea

```text
pair phenotype -> relative similarity rho_d
```

## Notation

- $i,j$: individuals.
- $d$: degree of relatedness (DOR) for pair $(i,j)$.
- $y_i$: phenotype value of individual $i$.
- $\ell_i$: latent liability of individual $i$ for binary phenotype.

- $g_i$: genetic component.
- $s_i$: shared-environment component.
- $\epsilon_i$: individual-specific residual component.

- $V_G$: genetic variance component.
- $V_S$: shared environmental variance component.
- $V_E$: individual-specific residual variance component.

- $w_G$: genetic decay rate. 기본값 $w_G=0.5$.
- $w_S$: shared-environment decay rate, $0 \le w_S \le 1$.

- $K$: binary phenotype prevalence.
- $t_K$: binary liability threshold.
- $\rho_d$: DOR $d$에서의 relative-pair phenotype similarity.

## Phenotype Scale

이 문서에서 $y_i$의 의미는 trait type에 따라 다르다.

```text
continuous: y_i = standardized continuous phenotype
binary:     y_i = observed binary phenotype, y_i in {0,1}
```

binary에서는 observed phenotype $y_i$ 뒤에 latent liability $\ell_i$가 있다고 본다.

## Continuous Phenotype Model

continuous phenotype ($y_i$)은 genetic component ($g_i$), shared-environment component ($s_i$), individual-specific residual ($\epsilon_i$)로 나눈다.

$$
y_i
=
g_i
+
s_i
+
\epsilon_i
$$

standardized phenotype scale에서:

$$
\mathrm{Var}(g_i)=V_G,
\qquad
\mathrm{Var}(s_i)=V_S,
\qquad
\mathrm{Var}(\epsilon_i)=V_E
$$

$$
V_G + V_S + V_E = 1
$$

continuous에서 DOR similarity ($\rho_d$)는 phenotype correlation이다.

$$
\rho_d
=
\mathrm{Corr}(y_i,y_j \mid d)
$$

## Binary Phenotype Model

binary phenotype ($y_i$)은 observed 0/1 phenotype이다.

$$
y_i \in \{0,1\}
$$

latent liability ($\ell_i$)는 genetic component ($g_i$), shared-environment component ($s_i$), individual-specific residual ($\epsilon_i$)로 나눈다.

$$
\ell_i
=
g_i
+
s_i
+
\epsilon_i
$$

standardized liability scale에서:

$$
\mathrm{Var}(g_i)=V_G,
\qquad
\mathrm{Var}(s_i)=V_S,
\qquad
\mathrm{Var}(\epsilon_i)=V_E
$$

$$
V_G + V_S + V_E = 1
$$

prevalence ($K$)가 주어지면 liability threshold ($t_K$)는:

$$
t_K = \Phi^{-1}(1-K)
$$

observed binary phenotype ($y_i$)은 thresholded liability로 정의한다.

$$
y_i
=
\mathbf{1}(\ell_i > t_K)
$$

binary에서 DOR similarity ($\rho_d$)는 liability correlation이다.

$$
\rho_d
=
\mathrm{Corr}(\ell_i,\ell_j \mid d)
$$

## DOR Component Covariance

DOR component covariance는 analysis scale에서 정의한다.

```text
continuous: analysis scale = y_i
binary:     analysis scale = ell_i
```

pair $(i,j)$가 DOR $d$이면 genetic component covariance는:

$$
\mathrm{Cov}(g_i,g_j \mid d)
=
w_G^d V_G,
\qquad
w_G=0.5
$$

shared-environment component covariance는:

$$
\mathrm{Cov}(s_i,s_j \mid d)
=
w_S^{d-1} V_S
$$

individual-specific residual은 pair 간 공유되지 않는다고 둔다.

$$
\mathrm{Cov}(\epsilon_i,\epsilon_j \mid d)=0
$$

따라서 analysis-scale covariance는:

$$
w_G^d V_G
+
w_S^{d-1}V_S
$$

## DOR Similarity Model

공통 BIGFAM model:

$$
\rho_d
=
w_G^d V_G + w_S^{d-1} V_S,
\qquad w_G = 0.5
$$

trait type별로 쓰면:

$$
\rho_d
=
\begin{cases}
\mathrm{Corr}(y_i,y_j \mid d), & \text{continuous} \\
\mathrm{Corr}(\ell_i,\ell_j \mid d), & \text{binary}
\end{cases}
$$

해석:

- $w_G^d V_G$: DOR가 한 단계 멀어질 때마다 genetic contribution은 $w_G=0.5$배가 된다.
- $w_S^{d-1} V_S$: DOR가 한 단계 멀어질 때마다 shared-environment contribution은 $w_S$배가 된다.
- DOR1에서는 $w_S^{0}=1$이라 shared-environment component가 $V_S$ 그대로 들어간다.
