from __future__ import print_function
import sys, math, numpy as N, matplotlib.pyplot as P
from libstempo.libstempo import *
import libstempo
import numpy as np
import libstempo as T
import matplotlib.pyplot as plt

psr=T.tempopulsar(parfile="J1810-197_glitch.par", timfile="withpn.tim")

GLEPs=np.linspace(60094,60108,100)
chisq=np.zeros(len(GLEPs))
psr["GLF0_1"].fit=True
psr["GLF1_1"].fit=True
psr["GLPH_1"].fit=True
for i in range(len(GLEPs)):
  psr["GLEP_1"].val=GLEPs[i]
  psr.fit()
  psr.fit()
  chi=psr.chisq()
  chisq[i]=chi

plt.plot(GLEPs, chisq, label="chisq", color="r")
plt.show()
