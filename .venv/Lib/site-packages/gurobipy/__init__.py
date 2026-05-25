import os
os.add_dll_directory(os.path.join(os.getenv('GUROBI_HOME'),'bin'))
from .gurobipy import *
