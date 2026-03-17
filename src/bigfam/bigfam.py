"""
BIGFAM 메인 클래스

사용자 친화적인 API를 제공하여 데이터 로드, 전처리, FR-reg 분석을 통합적으로 수행합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Literal, Dict, Union
from . import io
from . import processing
from . import frreg
from . import estimation
from .utils import ensure_log_columns


class BIGFAM:
    def __init__(self, verbose: bool = False):
        """
        BIGFAM 분석 객체 초기화
        
        Args:
            verbose: True이면 모든 과정에서 상세 로그 출력,
                     False이면 간략한 진행 상황만 출력 (기본값)
        """
        self.verbose = verbose
        
        self.df_pheno = None
        self.df_cov = None
        self.df_rel = None
        self.df_pairs = None
        self.df_results = None
        self.bootstrap_slopes = None  # Bootstrap slope values for variance estimation
        
        self.trait_type = None
        self.trait_col = None
        self.prevalence = None
        
    # =================
    # Load Data
    # =================
    def load_data(
        self,
        pheno_file: str,
        rel_file: str,
        cov_file: Optional[str] = None,
        trait_col: str = 'trait',
        trait_type: Optional[Literal['continuous', 'binary']] = None,
        id_col: str = 'iid',
        verbose: Optional[bool] = None
        ) -> 'BIGFAM':
        """
        데이터 로드 및 전처리
        
        Args:
            pheno_file: 표현형 파일 경로 (필수)
            rel_file: 관계 파일 경로 (필수)
            cov_file: 공변량 파일 경로 (선택, None이면 공변량 없이 진행)
            trait_col: 표현형 컬럼명
            trait_type: 'continuous' 또는 'binary' (None이면 자동 감지)
            id_col: ID 컬럼명
            verbose: True이면 상세 로그, None이면 global verbose 사용
            
        Returns:
            self
        """
        # verbose 결정
        effective_verbose = verbose if verbose is not None else self.verbose
        
        print(f"[Load Data]")
        
        # 1. 파일 로드
        self.df_pheno = io.load_phenotype(pheno_file, trait_col=trait_col, id_col=id_col, verbose=effective_verbose)
        self.df_rel = io.load_relationship(rel_file, verbose=effective_verbose)
        
        # 공변량 (optional)
        if cov_file is not None:
            self.df_cov = io.load_covariate(cov_file, id_col=id_col, verbose=effective_verbose)
            self.covariate_cols = [f'vol_{c}' for c in self.df_cov.columns if c != id_col]
        else:
            # 공변량 없이 진행: ID만 있는 빈 dataframe
            self.df_cov = self.df_pheno[[id_col]].copy()
            self.covariate_cols = []
        
        if len(self.covariate_cols) > 0:
            cov_names = [c.replace('vol_', '').replace('rel_', '') for c in self.covariate_cols]
            print(f"  - covariates: {cov_names}")
        else:
            print("  - no covariate. only adjust by intercepts.")
        
        self.trait_col = trait_col
        self.id_col = id_col
        
        # 2. Trait Type 감지 (입력값과 다르면 데이터 기준으로 자동 전환)
        # =================
        trait_raw = self.df_pheno[trait_col]
        trait_numeric = pd.to_numeric(trait_raw, errors='coerce')
        non_numeric_mask = trait_raw.notna() & trait_numeric.isna()
        is_binary_values = (not non_numeric_mask.any()) and trait_numeric.dropna().isin([0.0, 1.0]).all()
        detected_trait_type = 'binary' if is_binary_values else 'continuous'

        if trait_type is None:
            self.trait_type = detected_trait_type
        else:
            requested_trait_type = str(trait_type).lower()
            if requested_trait_type not in {'continuous', 'binary'}:
                raise ValueError("trait_type must be either 'continuous' or 'binary'.")
            self.trait_type = detected_trait_type
            if requested_trait_type != detected_trait_type:
                print(
                    f"  - trait_type mismatch: requested '{requested_trait_type}', "
                    f"detected '{detected_trait_type}'. Switching automatically."
                )

        # Binary로 감지된 경우 문자열 '0'/'1'도 안전하게 숫자로 정규화
        if self.trait_type == 'binary':
            self.df_pheno[trait_col] = trait_numeric
        
        # 원본 데이터 요약 (항상 출력)
        n_pheno_orig = len(self.df_pheno)
        n_dor = self.df_rel['DOR'].nunique() if 'DOR' in self.df_rel.columns else 0
        n_rel = self.df_rel['relationship'].nunique() if 'relationship' in self.df_rel.columns else 0
        print(f"  - {n_pheno_orig:,} phenotypes, {n_dor} DORs, {n_rel} relationships")
            
        # 3. Trait-specific 정보 
        # ========================
        if self.trait_type == 'binary':
            n_total = len(self.df_pheno)
            n_cases = (self.df_pheno[trait_col] == 1).sum()
            self.prevalence = (n_cases / n_total) * 100
            
            print(f"  - In sample prevalence: {self.prevalence:.2f}% ({n_cases:,} cases / {n_total:,} total)")
            
        # 4. 데이터 병합
        # ============
        self.df_pairs = processing.merge_to_pairs(
            self.df_pheno,
            self.df_cov,
            self.df_rel,
            pheno_id_col=id_col,
            cov_id_col=id_col,
            trait_col=trait_col,
            verbose=effective_verbose
        )
        
        # 완료 요약 (항상 출력)
        n_pairs = len(self.df_pairs)
        n_unique_ids = self.df_pairs['volid'].nunique()
        
        print(f"  - {self.trait_type.upper()}, left {n_pairs:,} pairs, {n_unique_ids:,} unique IDs after merging")
        # if self.covariate_cols:
        #     cov_names = [c.replace('vol_', '') for c in self.covariate_cols]
        #     print(f"  - covariates: {cov_names}")
        
        return self

    def set_manual_results(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """
        사용자가 직접 생성한 요약 결과(Summary Results)를 설정합니다.
        
        필수 컬럼:
        - 'DOR' + ('slope', 'se')
        - 또는 'DOR' + ('rho', 'se')
        - 또는 'DOR' + ('log_rho', 'log_se')
        자동 계산: 'log_rho', 'log_rho_se'
        (내부 연산에서는 기존 파이프라인 호환을 위해 'log_slope', 'log_se'도 보존)
        
        Args:
            data: DataFrame 또는 Dictionary 리스트
            
        Returns:
            pd.DataFrame: 설정된 결과 데이터프레임
            
        Example:
            >>> model = BIGFAM()
            >>> data = pd.DataFrame({
            ...     'DOR': [1, 2, 3, 4],
            ...     'slope': [0.3, 0.15, 0.075, 0.0375],
            ...     'se': [0.02, 0.02, 0.02, 0.02]
            ... })
            >>> model.set_manual_results(data)
            >>> model.run_slope_test()
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        if 'DOR' not in df.columns:
            raise ValueError("Missing required columns: ['DOR']")

        has_slope_se = {'slope', 'se'}.issubset(df.columns)
        has_log_rho = {'log_rho', 'log_se'}.issubset(df.columns)
        has_log_slope = {'log_slope', 'log_se'}.issubset(df.columns)
        has_log = has_log_rho or has_log_slope
        has_rho_se = {'rho', 'se'}.issubset(df.columns)

        if not (has_slope_se or has_log or has_rho_se):
            raise ValueError("Missing required columns: expected one of ('slope','se'), ('rho','se'), or ('log_rho','log_se').")

        if not has_slope_se:
            if has_log_rho:
                df["slope"] = 2 ** pd.to_numeric(df["log_rho"], errors="coerce")
                df["se"] = pd.to_numeric(df["log_se"], errors="coerce") * df["slope"] * np.log(2)
            elif has_log_slope:
                df["slope"] = 2 ** pd.to_numeric(df["log_slope"], errors="coerce")
                df["se"] = pd.to_numeric(df["log_se"], errors="coerce") * df["slope"] * np.log(2)
            elif has_rho_se:
                df["slope"] = pd.to_numeric(df["rho"], errors="coerce")
                df["se"] = pd.to_numeric(df["se"], errors="coerce")

        if "slope" not in df.columns or "se" not in df.columns:
            missing_cols = [c for c in ["slope", "se"] if c not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = ensure_log_columns(df)
        if "rho" not in df.columns:
            df["rho"] = df["slope"]
        else:
            df["rho"] = pd.to_numeric(df["rho"], errors="coerce").fillna(df["slope"])

        if "log_rho" not in df.columns:
            df["log_rho"] = df["log_slope"]
        else:
            df["log_rho"] = df["log_rho"].fillna(df["log_slope"])
        df["log_rho"] = pd.to_numeric(df["log_rho"], errors="coerce")

        if "log_rho_se" not in df.columns:
            df["log_rho_se"] = pd.to_numeric(df["log_se"], errors="coerce")
        else:
            df["log_rho_se"] = (
                pd.to_numeric(df["log_rho_se"], errors="coerce").fillna(pd.to_numeric(df["log_se"], errors="coerce"))
            )
            
        self.df_results = df
        
        print(f"\n{'='*60}")
        print(f"BIGFAM manual result setup complete")
        print(f"{'='*60}")
        print(f"  - Input rows: {len(df)}")
        print(f"  - Derived automatically: log_rho(log₂(ρ)), log_se(log₂(SE))")
        print(f"\n[Configured Results]")
        display_cols = [
            "DOR",
            "rho",
            "se",
            "log_rho",
            "log_rho_se",
            "n_asym",
            "n_asym_sym",
            "n_vols",
            "stable",
            "note",
        ]
        existing_display = [c for c in display_cols if c in self.df_results.columns]
        print(self.df_results[existing_display].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"{'='*60}\n")
        
        return self.df_results

    @staticmethod
    def _coerce_bool_like(value: object) -> bool:
        """
        다양한 형태의 불안정 플래그를 bool로 정규화.
        """
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        if isinstance(value, (list, tuple, set)):
            return len(value) > 0
        if isinstance(value, str):
            return bool(value.strip())
        return bool(value)

    def _normalize_correlation_output(
        self,
        df: pd.DataFrame,
        group_by: str,
        method: Optional[str] = None
        ) -> pd.DataFrame:
        """
        사용자 요청 형식(`rho`, `stable`, `note`)으로 출력 컬럼을 보강.
        """
        if df.empty:
            return df.copy()

        out = df.copy()
        method = (method or "").lower()

        if "slope" in out.columns and "rho" not in out.columns:
            out["rho"] = out["slope"]

        if "rho" in out.columns:
            out["rho"] = pd.to_numeric(out["rho"], errors="coerce")

        out["slope"] = pd.to_numeric(out.get("slope", out.get("rho")), errors="coerce")
        out["se"] = pd.to_numeric(out["se"], errors="coerce")

        for col in ["n_asym", "n_vols", "n_asym_sym"]:
            if col not in out.columns:
                out[col] = pd.NA

        if "note" not in out.columns:
            out["note"] = ""
        out["note"] = out["note"].fillna("").astype(str)

        stable = pd.Series([True] * len(out), index=out.index)

        if "robust_unstable" in out.columns:
            robust_unstable = out["robust_unstable"].map(self._coerce_bool_like)
            unstable_mask = robust_unstable.astype(bool)
            stable &= ~unstable_mask
            out.loc[unstable_mask, "note"] = out.loc[unstable_mask, "note"].apply(
                lambda v: f"{v} | robust_unstable" if v else "robust_unstable"
            )

        if method in {"bootstrap", "volsummary"}:
            req = pd.to_numeric(out.get("n_bootstrap_requested"), errors="coerce")
            succ = pd.to_numeric(out.get("n_bootstrap_success"), errors="coerce")
            fail = pd.to_numeric(out.get("n_bootstrap_fail"), errors="coerce")
            rate = succ / req
            no_request = req <= 0
            low_rate = (req > 0) & (rate < 0.7)
            unstable_mask = no_request.fillna(False) | low_rate.fillna(False)
            stable &= ~unstable_mask

            out.loc[unstable_mask, "note"] = out.loc[unstable_mask, "note"].apply(
                lambda v: f"{v} | bootstrap unstable" if v else "bootstrap unstable"
            )

            if rate.notna().any():
                out["bootstrap_success_rate"] = rate
                out["n_bootstrap_requested"] = req.fillna(0).astype(int)
                out["n_bootstrap_success"] = succ.fillna(0).astype(int)
                out["n_bootstrap_fail"] = fail.fillna(0).astype(int)

        if "stable" not in out.columns:
            if method == "robust":
                rho = pd.to_numeric(out.get("rho"), errors="coerce")
                se = pd.to_numeric(out.get("se"), errors="coerce")
                stable &= np.isfinite(rho) & np.isfinite(se) & (se > 0)
            else:
                stable &= True

        if "stable" in out.columns:
            stable &= out["stable"].fillna(False).astype(bool)

        note_mask = out["note"].astype(str).str.strip().astype(bool)
        stable &= ~note_mask

        out["stable"] = stable.astype(bool)

        if "log_slope" not in out.columns:
            out["log_slope"] = np.where(
                out["slope"] > 0,
                np.log2(out["slope"]),
                np.nan
            )
        else:
            out["log_slope"] = pd.to_numeric(out["log_slope"], errors="coerce")

        if "log_se" not in out.columns:
            out["log_se"] = np.where(
                (out["slope"] > 0) & (out["se"] > 0),
                out["se"] / (out["slope"] * np.log(2)),
                0.0
            )
        else:
            out["log_se"] = pd.to_numeric(out["log_se"], errors="coerce")

        if "log_rho" not in out.columns:
            out["log_rho"] = out["log_slope"]
        else:
            out["log_rho"] = pd.to_numeric(out["log_rho"], errors="coerce").fillna(out["log_slope"])

        if "log_rho_se" not in out.columns:
            out["log_rho_se"] = out["log_se"]
        else:
            out["log_rho_se"] = pd.to_numeric(out["log_rho_se"], errors="coerce").fillna(out["log_se"])

        preferred = [
            group_by,
            'rho',
            'slope',
            'se',
            'n_asym',
            'n_asym_sym',
            'n_vols',
            'stable',
            'note',
            'log_slope',
            'log_rho',
            'log_se',
            'log_rho_se',
        ]
        existing = [c for c in preferred if c in out.columns]
        remaining = [c for c in out.columns if c not in existing]
        return out[existing + remaining]



    # =================
    # FR-reg
    # =================
    def estimate_correlations(
        self,
        group_by: str = 'DOR',
        method: Optional[str] = None,
        min_pairs: int = 100,
        n_bootstrap: int = 100,
        aggregation: str = 'mean',
        prevalence: Optional[float] = None,
        covariate_cols: Optional[List[str]] = None,
        two_way_cluster: bool = True,
        verbose: Optional[bool] = None
        ) -> pd.DataFrame:
        """
        가족 상관계수(Familial Correlation) 추정 실행
        
        Args:
            group_by: 그룹화 기준 ('DOR' 또는 'relationship')
            method: FR-reg 방법 선택
                Continuous: 'bootstrap', 'volsummary', 'robust' (default)
                Binary: 'bootstrap', 'liability', 'robust' (기본값)
            min_pairs: 최소 쌍 수
            n_bootstrap: Bootstrap 반복 횟수
            aggregation: Vol-summary 집계 방법 ('mean' or 'sum')
            prevalence: 유병률 (Binary only)
            covariate_cols: 공변량 컬럼
            two_way_cluster: Two-way clustering (method='robust'에서 사용)
            verbose: 상세 로그 출력
            
        Returns:
            결과 DataFrame
        """
        print(f"[Estimate Correlations]")
        if self.df_pairs is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        # verbose 결정
        effective_verbose = verbose if verbose is not None else self.verbose
        
        # method 기본값 설정
        if method is None:
            method = 'robust'
        
        # covariate_cols 기본값: load_data에서 저장된 공변량 사용
        if covariate_cols is None and hasattr(self, 'covariate_cols') and self.covariate_cols:
            covariate_cols = self.covariate_cols
        
        # 헤더 출력
        n_groups = self.df_pairs[group_by].nunique()
        print(f"  - group : {group_by}, method: {method}")
            
        # 1. 쌍 데이터 필터링
        df_ready = processing.filter_groups_by_size(
            self.df_pairs,
            trait_col=self.trait_col,
            group_by=group_by,
            min_pairs=min_pairs,
            verbose=False
        )

        # 2. 분석 실행
        if self.trait_type == 'continuous':
            result = frreg.fit_continuous_frreg(
                df_ready,
                group_by=group_by,
                method=method,
                n_bootstrap=n_bootstrap,
                aggregation=aggregation,
                covariate_cols=covariate_cols,
                two_way_cluster=two_way_cluster,
                min_pairs=min_pairs,
                verbose=effective_verbose,
                return_bootstrap=False
            )
            
        elif self.trait_type == 'binary':
            if prevalence is None and self.prevalence is not None:
                prevalence = self.prevalence
            
            if method == 'bootstrap':
                result = frreg.fit_binary_frreg(
                    df_ready,
                    group_by=group_by,
                    prevalence=prevalence,
                    covariate_cols=covariate_cols,
                    n_bootstrap=n_bootstrap,
                    min_pairs=min_pairs,
                    verbose=effective_verbose,
                    return_bootstrap=False
                )
                
            elif method == 'liability':
                result = frreg.fit_binary_frreg_liability(
                    df_ready,
                    group_by=group_by,
                    covariate_cols=covariate_cols,
                    n_bootstrap=n_bootstrap,
                    aggregation=aggregation,
                    min_pairs=min_pairs,
                    verbose=effective_verbose,
                    return_bootstrap=False
                )
                
            elif method == 'robust':
                result = frreg.fit_binary_frreg_robust(
                    df_ready,
                    group_by=group_by,
                    covariate_cols=covariate_cols,
                    two_way_cluster=two_way_cluster,
                    min_pairs=min_pairs,
                    verbose=effective_verbose
                )
            else:
                raise ValueError(f"Unknown method for binary: {method}. Use 'bootstrap', 'liability', or 'robust'.")
        
        normalized_result = self._normalize_correlation_output(result, group_by=group_by, method=method)
        self.df_results = normalized_result
        
        # 결과 요약
        print(f"  - analyzed: {len(result)} groups")
        if effective_verbose:
            for _, row in result.iterrows():
                print(f"  | {group_by}={row[group_by]}: slope={row['slope']:.4f} ± {row['se']:.4f}")

        return self.df_results[[group_by, "rho", "se", "n_asym", "n_vols", "stable", "note"]]

    # def summary(self):
    #     """결과 요약 출력"""
    #     if self.df_results is None:
    #         print("분석 결과가 없습니다.")
    #         return
            
    #     print(f"\n{'='*60}")
    #     print(f"FR-reg 분석 결과 요약")
    #     print(f"{'='*60}")
        
    #     # 출력 컬럼 선택
    #     cols = ['DOR', 'slope', 'se', 'log_slope', 'log_se', 'n_pairs', 'n_vols']
    #     cols = [c for c in cols if c in self.df_results.columns]
        
    #     print(self.df_results[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    #     print(f"{'='*60}\n")
        
    # def get_pairs(self) -> pd.DataFrame:
    #     """현재 로드된 쌍 데이터 반환"""
    #     return self.df_pairs
        
    # def get_results(self) -> pd.DataFrame:
    #     """분석 결과 반환"""
    #     return self.df_results


    # =================
    # Obj.1. Partitioning Vg, Vs
    # =================
    def run_slope_test(
        self,
        method: Literal['direct', 'resample', 'lognormal', 'known_var'] = 'known_var',
        n_bootstrap: int = 100,
        verbose: Optional[bool] = None
        ) -> Dict:
        """
        Slope Test를 수행하여 w_S = 2 가설을 검정합니다.
        
        이론적 배경:
        - H0 (w_S = 2) 하에서: log2(λ_d) = -d + const
        - 따라서 -DOR에 대한 회귀 기울기가 1이면 H0 채택
        
        Args:
            method: 'direct' (권장, WLS) 또는 'resample' (기존 방식)
            n_bootstrap: Bootstrap 반복 횟수
            
        Returns:
            Dict: slope test 결과 (significance, slope, se, CI 등)
        """
        if self.df_results is None:
            raise ValueError("No FR-reg results found. Run estimate_correlations() first.")
        
        # verbose 결정: 메서드 인자 > 전역 설정
        effective_verbose = verbose if verbose is not None else self.verbose
        
        sig, result = estimation.run_slope_test(
            self.df_results,
            method=method,
            n_bootstrap=n_bootstrap,
            verbose=effective_verbose
        )
        
        self.slope_significance = sig
        self.slope_result = result
        
        return {'significance': sig, **result}

    def estimate_variance_components(
        self,
        loss_method: Literal['log', 'log_bc', 'linear_wls'] = 'log',
        n_resample: int = 1000,
        n_repeat_cv: int = 10,
        n_block: int = 10,
        correlation_input: Optional[Union[pd.DataFrame, str, Path]] = None,
        slope_test_method: Literal['direct', 'resample', 'lognormal', 'known_var'] = 'known_var',
        n_bootstrap: int = 100,
        use_parallel: bool = True,
        slope_verbose: Optional[bool] = None
        ) -> Dict:
        """
        FR-reg 결과로부터 분산 성분(V_G, V_S, w)을 추정합니다.
        
        correlation_input이 주어지면:
        1) `DOR`, `rho`, `se` 컬럼 기반으로 FR-reg 결과를 새로 설정하고
        2) `slope_test`를 실행한 뒤
        3) variance_components를 계산합니다.
        
        correlation_input 미지정 시:
        기존 동작(사전 계산된 `run_slope_test` 결과 활용)을 그대로 따릅니다.

        Args:
            method: 추정 방법
                - 'direct': Bootstrap 또는 Delta Method
                - 'resample': Cross-Validation 기반 (기존 BIGFAM 방식)
            n_bootstrap: Bootstrap 반복 횟수 (method='direct')
                - 0: Delta Method 사용 (빠름)
                - >0: Bootstrap 사용
            n_resample: 각 DOR당 resample 수 (method='resample', bootstrap_slopes 없을 때만 사용)
            n_repeat_cv: CV 반복 횟수 (method='resample')
            n_block: CV fold 수 (method='resample')
            correlation_input: DataFrame 또는 파일 경로
                - DataFrame: DOR-rho/se 또는 DOR-log_rho/log_se 또는 DOR-log_slope/log_se 조합 중 하나
                - 파일: TSV/CSV 경로, 상기 컬럼 조합 중 하나 필요
            loss_method: 'resample' 방식에서 사용할 loss function ('log', 'log_bc', 'linear_wls')
            slope_test_method: slope test method
            n_bootstrap: slope_test 부트스트랩 반복 횟수
            slope_verbose: slope_test verbose override
            
        Returns:
            Dict: 분산 성분 추정 결과
        """
        # correlation_input이 있으면 입력을 normalize 후 slope test부터 다시 수행
        if correlation_input is not None:
            if isinstance(correlation_input, (str, Path)):
                path = Path(correlation_input)
                if not path.exists():
                    raise FileNotFoundError(f"Could not find correlation_input path: {path}")

                df_input = None
                for sep in ["\t", ","]:
                    try:
                        df_input = pd.read_csv(path, sep=sep)
                        if (
                            {"DOR", "rho", "se"}.issubset(df_input.columns)
                            or {"DOR", "log_rho", "log_se"}.issubset(df_input.columns)
                            or {"DOR", "log_slope", "log_se"}.issubset(df_input.columns)
                        ):
                            break
                    except Exception:
                        df_input = None

                if (
                    df_input is None
                    or not (
                        {"DOR", "rho", "se"}.issubset(df_input.columns)
                        or {"DOR", "log_rho", "log_se"}.issubset(df_input.columns)
                        or {"DOR", "log_slope", "log_se"}.issubset(df_input.columns)
                    )
                ):
                    raise ValueError(
                        f"correlation_input must contain DOR and one of (rho/se), "
                        f"(log_rho/log_se), or (log_slope/log_se). path={path}"
                    )
            else:
                if not isinstance(correlation_input, pd.DataFrame):
                    raise TypeError("correlation_input must be a DataFrame or a file path.")
                df_input = correlation_input.copy()

            if {"DOR", "rho", "se"}.issubset(df_input.columns):
                df_frreg = df_input.loc[:, ["DOR", "rho", "se"]].rename(columns={"rho": "slope"})
            elif {"DOR", "log_rho", "log_se"}.issubset(df_input.columns):
                df_frreg = df_input.loc[:, ["DOR", "log_rho", "log_se"]]
            elif {"DOR", "log_slope", "log_se"}.issubset(df_input.columns):
                df_frreg = df_input.loc[:, ["DOR", "log_slope", "log_se"]]
            else:
                raise ValueError(
                    "correlation_input does not contain a valid column combination: "
                    "(rho/se), (log_rho/log_se), or (log_slope/log_se)."
                )

            self.set_manual_results(df_frreg)
            self.run_slope_test(method=slope_test_method, n_bootstrap=n_bootstrap, verbose=slope_verbose)
        else:
            if self.df_results is None:
                raise ValueError("No FR-reg results found. Run estimate_correlations() or simulate_frreg_results() first.")
            # Slope test가 실행되지 않았으면 에러
            if not hasattr(self, 'slope_significance') or self.slope_significance is None:
                raise ValueError("slope_test has not been run.")

        if self.slope_significance is None:
            raise ValueError("slope_test has not been run.")
        
        result, df_pred = estimation.estimate_variance_components(
            self.df_results,
            slope_significance=self.slope_significance,
            loss_method=loss_method,
            n_resample=n_resample,
            n_repeat_cv=n_repeat_cv,
            n_block=n_block,
            use_parallel=use_parallel,
            verbose=True
        )

        slope_test_bundle = {
            "significance": self.slope_significance,
            **self.slope_result
        }

        if isinstance(df_pred, pd.DataFrame):
            for k, v in slope_test_bundle.items():
                df_pred[f"slope_test_{k}"] = v

        if result is None:
            result_with_slope = None
        else:
            result_with_slope = dict(result)
            for k, v in slope_test_bundle.items():
                result_with_slope[f"slope_test_{k}"] = v
        
        self.variance_components = result_with_slope
        self.df_pred = df_pred
        
        return result_with_slope

    def estimate_pairwise_variance_components(
        self,
        ws: Optional[float] = None,
        w: Optional[float] = None,
        stage4_mode: Literal['pairwise_sandwich', 'rho_gls'] = 'pairwise_sandwich',
        rho_df: Optional[pd.DataFrame] = None,
        min_component: float = 1e-4,
        two_way_cluster: bool = True,
        covariate_cols: Optional[List[str]] = None,
        verbose: Optional[bool] = None
        ) -> Dict:
        """
        Stage 4: 고정된 w(=1/w_S)를 사용해 모든 relative pair에서 V_G, V_S를 재추정합니다.
        
        w/ws 지정이 없으면:
        1) self.df_pred (step3 CV 결과)의 w 중앙값 사용 (V_G,V_S > min_component 필터 우선)
        2) 없으면 self.variance_components['w'] 사용
        
        Args:
            ws: 고정할 w_S 값 (예: 2.0). 지정 시 w=1/ws로 변환
            w: 고정할 w 값 (=1/w_S, 0~1)
            stage4_mode: 'pairwise_sandwich' 또는 'rho_gls'
            rho_df: stage4_mode='rho_gls'일 때 사용할 DOR별 요약치 (DOR, slope, se)
            min_component: step3 결과에서 안정적 추정치 필터 기준
            two_way_cluster: True면 (volid, relid) two-way clustering sandwich SE
            covariate_cols: 공변량 컬럼 목록 (None이면 load_data에서 저장된 공변량 사용)
            verbose: 상세 로그 출력
        
        Returns:
            Dict: pairwise 재추정 결과 (V_G, V_S, CI, 고정 w/ws 등)
        """
        if self.df_pairs is None:
            raise ValueError("No pair data found. Run load_data() first.")

        effective_verbose = verbose if verbose is not None else self.verbose

        if covariate_cols is None and hasattr(self, 'covariate_cols') and self.covariate_cols:
            covariate_cols = self.covariate_cols

        if ws is not None and w is not None:
            raise ValueError("You cannot specify ws and w at the same time. Choose one.")

        w_source = "manual"
        if ws is not None:
            if ws <= 0:
                raise ValueError(f"ws must be positive. Received: {ws}")
            w_fixed = 1.0 / float(ws)
        elif w is not None:
            w_fixed = float(w)
        else:
            # 우선순위 1: step3의 CV 결과(df_pred)에서 안정 필터 후 median w
            if hasattr(self, 'df_pred') and isinstance(self.df_pred, pd.DataFrame):
                required_cols = {'w', 'V_G', 'V_S'}
                if required_cols.issubset(set(self.df_pred.columns)):
                    df_w = self.df_pred.dropna(subset=['w', 'V_G', 'V_S']).copy()
                    if len(df_w) > 0:
                        df_stable = df_w[
                            (df_w['V_G'] > min_component) &
                            (df_w['V_S'] > min_component)
                        ].copy()
                        if len(df_stable) > 0:
                            w_fixed = float(df_stable['w'].median())
                            w_source = f"df_pred_stable_median(threshold={min_component})"
                        else:
                            w_fixed = float(df_w['w'].median())
                            w_source = "df_pred_median(no_stable_subset)"
                    else:
                        w_fixed = np.nan
                else:
                    w_fixed = np.nan
            else:
                w_fixed = np.nan

            # 우선순위 2: step3 summary
            if not np.isfinite(w_fixed):
                if hasattr(self, 'variance_components') and self.variance_components is not None and 'w' in self.variance_components:
                    w_fixed = float(self.variance_components['w'])
                    w_source = "variance_components"
                else:
                    raise ValueError(
                        "Could not determine a fixed w. "
                        "Run estimate_variance_components() first or specify ws/w directly."
                    )

        if not (0 < w_fixed < 1):
            raise ValueError(f"Fixed w must be in the range (0, 1). Received: {w_fixed}")

        if effective_verbose:
            print(f"[Pairwise Refit]")
            print(f"  - fixed w: {w_fixed:.4f} (w_S={1/w_fixed:.3f})")
            print(f"  - source: {w_source}")
            print(f"  - mode: {stage4_mode}")

        df_rho_use = None
        if stage4_mode == 'rho_gls':
            if rho_df is not None:
                df_rho_use = rho_df.copy()
            elif self.df_results is not None:
                df_rho_use = self.df_results.copy()
            else:
                raise ValueError(
                    "rho_gls mode requires DOR-level summaries. "
                    "Run estimate_correlations(group_by='DOR') first or pass rho_df."
                )

        result, df_diag = estimation.estimate_pairwise_variance_components(
            self.df_pairs,
            trait_type=self.trait_type,
            w=w_fixed,
            stage4_mode=stage4_mode,
            df_rho=df_rho_use,
            covariate_cols=covariate_cols,
            two_way_cluster=two_way_cluster,
            verbose=effective_verbose
        )

        result['w_source'] = w_source

        self.pairwise_variance_components = result
        self.df_pairwise_pred = df_diag

        return result


    # =================
    # Obj.2. Predict Vx
    # =================
    def estimate_x_variance(
        self,
        n_resample: int = 1000,
        regout_bin: Optional[List[str]] = None,
        use_weights: bool = False,
        alpha_weight: float = 2.0,
        alpha_type: Literal['lambda', 'fixed'] = 'lambda'
        ) -> Dict:
        """
        환경 분산 (V_X) 추정.
        
        FR-reg 결과로부터 환경 요인의 분산을 추정합니다.
        이 기능을 사용하려면 estimate_correlations(group_by='relationship')을 
        먼저 실행해야 합니다.
        
        Args:
            n_resample: Bootstrap 샘플 수
            regout_bin: 잔차 계산 시 그룹화 기준 (기본값: ["DOR"])
            alpha_weight: L2 정규화 가중치
            alpha_type: 'lambda' (람다에 의존) 또는 'fixed' (고정값)
            
        Returns:
            Dict: V_X, V_X_lower, V_X_upper 포함
        """
        if self.df_results is None:
            raise ValueError("No FR-reg results found. Run estimate_correlations() first.")

        if 'relationship' not in self.df_results.columns:
            raise ValueError(
                "The 'relationship' column is missing. "
                "Run estimate_correlations(group_by='relationship') first."
            )
        
        if regout_bin is None:
            regout_bin = ["DOR"]
        
        result = estimation.estimate_x_variance(
            self.df_results,
            n_resample=n_resample,
            regout_bin=regout_bin,
            use_weights=use_weights,
            alpha_weight=alpha_weight,
            alpha_type=alpha_type,
            verbose=True
        )
        
        self.x_variance = result
        return result

    def estimate_sex_specific_x(
        self,
        n_resample: int = 1000,
        regout_bin: Optional[List[str]] = None,
        alpha_weight: float = 2.0,
        alpha_type: Literal['lambda', 'fixed'] = 'lambda',
        n_r_grid: int = 11
        ) -> Dict:
        """
        성별별 환경 분산 (V_X_male, V_X_female) + 상관계수 (r) 추정.
        
        FR-reg 결과로부터 남성/여성별 환경 요인의 분산과 상관계수를 추정합니다.
        이 기능을 사용하려면 estimate_correlations(group_by='relationship')을 
        먼저 실행해야 하며, sex_type 컬럼이 있어야 합니다.
        
        Args:
            n_resample: Bootstrap 샘플 수
            regout_bin: 잔차 계산 시 그룹화 기준 (기본값: ["DOR", "sex_type"])
            alpha_weight: L2 정규화 가중치
            alpha_type: 'lambda' 또는 'fixed'
            n_r_grid: r 그리드 수 (-1 ~ 1)
            
        Returns:
            Dict: V_X_male, V_X_female, r 포함
        """
        if self.df_results is None:
            raise ValueError("No FR-reg results found. Run estimate_correlations() first.")

        if 'relationship' not in self.df_results.columns:
            raise ValueError(
                "The 'relationship' column is missing. "
                "Run estimate_correlations(group_by='relationship') first."
            )

        if 'sex_type' not in self.df_results.columns:
            raise ValueError(
                "The 'sex_type' column is missing. "
                "Check that your data includes sex information."
            )
        
        if regout_bin is None:
            regout_bin = ["DOR", "sex_type"]
        
        result = estimation.estimate_sex_specific_x(
            self.df_results,
            n_resample=n_resample,
            regout_bin=regout_bin,
            alpha_weight=alpha_weight,
            alpha_type=alpha_type,
            n_r_grid=n_r_grid,
            verbose=True
        )
        
        self.sex_specific_x = result
        return result




    # =================
    # Simulation
    # =================
    def simulate_frreg_results(
        self,
        Vg: float,
        Vs: float,
        ws: float = 2.0,
        slope_se: Union[float, List[float]] = 0.0,
        dor_list: Optional[List[int]] = None,
        trait_type: Literal['continuous', 'binary'] = 'continuous',
        random_seed: Optional[int] = None
        ) -> pd.DataFrame:
        """
        이론적 FR-reg 공식에 기반한 요약 결과(Summary Results)를 시뮬레이션합니다.
        
        이 메서드는 개별 데이터를 생성하지 않고, 이론적 모델로부터 기대되는 
        FR-reg 계수(slope/rho)와 그에 따른 통계량을 직접 생성하여 self.df_results에 저장합니다.
        
        이론적 모델:
            E[λ_d] = 2^{-d} * V_G + w_S^{-d+1} * V_S
        
        Args:
            Vg: 유전 분산 (V_G, heritability에 해당)
            Vs: 공유 환경 분산 (V_S)
            ws: 환경 감쇠율 (w_S), 기본값 2.0
            slope_se: slope에 추가할 Gaussian noise의 표준편차 (SE 시뮬레이션용)
                      float인 경우 모든 DOR에 동일한 SE 적용
                      List[float]인 경우 각 DOR별 SE 적용 (dor_list와 길이 동일해야 함)
            dor_list: DOR 값 리스트, 기본값 [1, 2, 3, 4]
            trait_type: 'continuous' 또는 'binary'
            random_seed: 재현성을 위한 랜덤 시드
            
        Returns:
            pd.DataFrame: DOR별 FR-reg 결과
                - DOR: Degree of Relatedness
                - slope: E[λ_d] 값 (+ optional noise)
                - log_slope: log2(slope) 값
                - se: slope의 SE (slope_se로 설정)
                - log_se: log_slope의 SE (Delta Method 근사)
                
        Example:
            >>> model = BIGFAM()
            >>> # 동일 SE 적용
            >>> model.simulate_frreg_results(Vg=0.4, Vs=0.1, ws=2.0, slope_se=0.02)
            >>> # DOR별 다른 SE 적용 (WLS 테스트용)
            >>> model.simulate_frreg_results(Vg=0.4, Vs=0.1, ws=2.0, slope_se=[0.01, 0.02, 0.05])
            >>> result = model.run_slope_test()
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if dor_list is None:
            dor_list = [1, 2, 3, 4]
        
        # slope_se를 리스트로 변환 (단일 값인 경우 모든 DOR에 동일 적용)
        if isinstance(slope_se, (int, float)):
            se_list = [float(slope_se)] * len(dor_list)
        else:
            se_list = list(slope_se)
            if len(se_list) != len(dor_list):
                raise ValueError(f"slope_se list length ({len(se_list)}) must match dor_list length ({len(dor_list)})")
            
        self.trait_type = trait_type
        self.prevalence = None  # 요약 통계량 생성 시에는 사용되지 않음
            
        print(f"\n{'='*60}")
        print(f"BIGFAM theoretical FR-reg simulation ({trait_type.upper()})")
        print(f"{'='*60}")
        print(f"\n[Parameters]")
        print(f"  - Vg (genetic variance): {Vg}")
        print(f"  - Vs (shared environmental variance): {Vs}")
        print(f"  - ws (environmental decay rate): {ws}")
        print(f"  - slope_se (noise SE): {se_list}")
        print(f"  - DOR list: {dor_list}")
        
        results = []
        for i, d in enumerate(dor_list):
            current_se = se_list[i]
            
            # 이론적 lambda 계산: E[λ_d] = 2^{-d} * V_G + w_S^{-d+1} * V_S
            lambda_d = (2 ** (-d)) * Vg + (ws ** (-d + 1)) * Vs
            
            # 노이즈 추가 (옵션)
            if current_se > 0:
                noise = np.random.normal(0, current_se)
                lambda_d_noisy = lambda_d + noise
            else:
                lambda_d_noisy = lambda_d
                
            # log2(lambda) 계산
            if lambda_d_noisy > 0:
                log_slope = np.log2(lambda_d_noisy)
            else:
                log_slope = np.nan
                
            # log_se 계산 (Delta Method: se_log = se / (lambda * ln(2)))
            if lambda_d_noisy > 0 and current_se > 0:
                log_se = current_se / (lambda_d_noisy * np.log(2))
            else:
                log_se = 0.0
                
            results.append({
                'DOR': d,
                'slope': lambda_d_noisy,
                'se': current_se,
                'log_slope': log_slope,
                'log_se': log_se,
                'theoretical_slope': lambda_d,  # 노이즈 없는 이론값
            })
            
        self.df_results = pd.DataFrame(results)
        
        print(f"\n[Generated simulation results]")
        slope_label = "rho (ρ)" if trait_type == 'binary' else "slope (λ)"
        print(f"  * The {slope_label} column contains the estimated correlation.")
        print(self.df_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"{'='*60}\n")
        
        return self.df_results
