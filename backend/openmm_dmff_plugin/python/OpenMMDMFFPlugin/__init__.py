"""OpenMM is a toolkit for molecular simulation. It can be used either as a
stand-alone application for running simulations, or as a library you call
from your own code. It provides a combination of extreme flexibility
(through custom forces and integrators), openness, and high performance
(especially on recent GPUs) that make it truly unique among simulation codes.
"""
from __future__ import absolute_import
__author__ = "Ye Ding"
__mail__ = "dingye@westlake.edu.cn"
__version__ = "@GIT_HASH@"

import os, os.path
import sys
from .tools import DMFFModel
from .OpenMMDMFFPlugin import DMFFForce
