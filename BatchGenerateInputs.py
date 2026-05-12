import math 
import numpy as np 
import sklearn 

velocity = []
angle = []

JOURNAL_TEMPLATE = """\
/file/read-case "{base_case}"
/define/boundary-conditions/set/velocity-inlet inlet 

/solve/initialize/hyb-initialization
/solve/iterate 500

/file/write-case-data "{output}"
/exit yes
"""



for v in velocity: 
    