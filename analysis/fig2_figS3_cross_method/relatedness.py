"""Mendelian relatedness constants -- universal, not simulation knobs.

w_G^d = genetic correlation of a DOR-d relative pair (1/2 per meiosis). Every
estimator (simulation and real alike) shares these; truth/grid the simulation
*chooses* lives in config.py instead.
"""
from __future__ import annotations

W_G = 0.5                                 # genetic decay per degree of relatedness
DORS = [1, 2, 3]
REL = {1: 0.5, 2: 0.25, 3: 0.125}          # w_G^d
