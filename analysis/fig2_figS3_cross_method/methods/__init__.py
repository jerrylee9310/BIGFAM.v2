"""Native (Python) method estimators -- falconer, he, pcgc, bigfam, bigfam_v1.

Each module exposes a (rho) -> {V_G, ...} callable (or a factory that closes
over the artifacts/seed to make one). The method x condition manifest is
assembled in run.py's build_specs(). The fixed-decay engines (SEM, LDAK
QuantHer/TetraHer) live in engines/ instead.
"""
