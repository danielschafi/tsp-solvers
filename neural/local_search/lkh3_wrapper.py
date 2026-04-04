


import torch 
from torch import Tensor
import numpy as np
from pathlib import Path

def _check_binary_exists() -> None:
    if not Path("bin/lkh3/LKH-3.0.13/LKH").exists():
        raise FileNotFoundError("LKH3 Binary not found, please download the binary acoording to the instructions in the readme.md")


def _prepare_for_lkh3():
    pass


def _solve():
    pass




def lkh_solve(heatmap:Tensor, top_k:int 10):
    _check_binary_exists()
    _prepare_for_lkh3()
    _solve()