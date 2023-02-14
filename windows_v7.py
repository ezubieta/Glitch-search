#!/usr/bin/env python
import argparse
import os
import astropy.units as u
import matplotlib.pyplot as plt
import pint.fitter
import pint.residuals
import pint.toa
from pint.models import get_model, get_model_and_toas
import pint.logging
import pint.config
import numpy as np
import pint.residuals as res
import copy
from pint.models import BinaryELL1, BinaryDD, PhaseJump, parameter, get_model
from pint.simulation import make_fake_toas_uniform as mft
from astropy import units as u, constants as c
from uncertainties import ufloat
from uncertainties.umath import *
import pandas as pd



#Archivos a usar:
#parfile = "J0835-4510_2020-2022.par"
#timfile = "J0835-4510_A1_2020-2022.tim"
#parfile = "J1048-5832.par"
#timfile = "J1048-5832.tim"
#parfile = "J1048-5832_noise.par"
parfile = "J1644-4559.par"
timfile_v1 = "J1644-4559_short.tim"





#Filtrar ToAs:
def dot(l1,l2):
    return np.array([v1 and v2 for v1,v2 in zip(l1,l2)])
def inv(l):
    return np.array([not i for i in l])

def mask_toas(toas,before=None,after=None,on=None,window=None):
    cnd=np.array([True for t in toas.get_mjds()])
    if before is not None:
        cnd = dot(cnd,toas.get_mjds().value >= before)
    if after is not None:
        cnd = dot(cnd,toas.get_mjds().value < after)
    if on is not None:
        on=np.array(on)
        for i,m in enumerate(on):
            m=m*u.day
            if type(m) is int:
                cnd = dot(cnd,inv(np.abs(toas.get_mjds()-m).astype(int) == np.abs((toas.get_mjds()-m)).min().astype(int)))
            else:
                cnd = dot(cnd,inv(np.abs(toas.get_mjds()-m) == np.abs((toas.get_mjds()-m)).min()))
    if window is not None:
        if len(window)!=2:
            raise ValueError("window must be a 2 element list/array")
        window = window*u.day
        lower = window[0]
        upper = window[1]
        cnd = dot(cnd,toas.get_mjds() < lower)+dot(cnd,toas.get_mjds() > upper)
    print(f'{sum(cnd)}/{len(cnd)} TOAs selected')
    return toas[cnd]


def delta_nu(nu0, nu1):
        return (nu1-nu0)/nu0







def windows(fitf1=False,n_obs_dense=10, thresh=1e-8):
	parfile = "J1644-4559.par"
	timfile_v1 = "J1644-4559_short.tim"
	file=pd.read_csv(timfile_v1, skiprows=1, comment='C', sep=' ', usecols=[0,1,2,3,4], skipinitialspace=True, names=['FORMAT 1', 'Freq','MJD','TOA', 'Antenna'])
	file_v2 = file.sort_values(by=["MJD"])
	delete = []
	for i in range(len(file_v2)-1):
     		if file_v2["MJD"][i+1]==file_v2["MJD"][i]:
         		delete.append(i+1)
	file_v3=file_v2.drop(delete, axis=0)
	file_v3.to_csv("J1644-4559_short.tim", sep=' ',header=False, index=False, index_label=None)
	timfile="J1644-4559_short.tim"
	print("I saved a cute and clean file.tim in your folder")
	m, t_all = get_model_and_toas(parfile, timfile)
	t=t_all[t_all.get_errors() < 0.1 * u.ms]
	m.F0.frozen=False
	mjds=t.get_mjds().value
	windows=mjds
	print(fitf1)
	print(not fitf1)
	print(f'{fitf1=}')
	print(f'{m.F1.frozen=}')
	m.F1.frozen = not fitf1
	print(f'{m.F1.frozen=}')
	F0=[]
	F0_par=m.F0.value
	F0_par_error=m.F0.uncertainty_value
	F1=[]
	F0_error=[]
	F1_error=[]
	F0_jump = []
	F0_par_jump=[]
	F1_jump=[]
	F1_par=m.F1.value
	F1_par_error=m.F1.uncertainty_value
	F1_par_jump=[]
	for i in range(len(windows)-(n_obs_dense+1)):
		window=mask_toas(t, before=windows[i], after=windows[i+n_obs_dense])
		f=pint.fitter.DownhillWLSFitter(window,m)
		f.fit_toas()
		F0_window=f.model["F0"].value
		F1_window=f.model["F1"].value
		F0_error_window=f.model["F0"].uncertainty_value
		F1_error_window=f.model["F1"].uncertainty_value
		F0.append(F0_window)
		F1.append(F1_window)
		F0_error.append(F0_error_window)
		F1_error.append(F1_error_window)
		print(F0_window, F0_error_window)
	for i in range(len(F0)-(n_obs_dense+1)):
		nu0=ufloat(F0[i], F0_error[i])
		nu0_par=ufloat(F0_par, F0_par_error)
		nudot0_par=ufloat(F1_par, F1_par_error)
		nudot0=ufloat(F1[i],F1_error[i])
		nu1=ufloat(F0[i+n_obs_dense], F0_error[i+n_obs_dense])
		nudot1=ufloat(F1[i+n_obs_dense], F1_error[i+n_obs_dense])
		F0_jump.append(delta_nu(nu0,nu1))
		F0_par_jump.append(delta_nu(nu0_par, nu1))
		F1_jump.append(delta_nu(nudot0, nudot1))
		F1_par_jump.append(delta_nu(nudot0_par,nudot1))
	no_glitches=0
	no_glitches_par=0
	possible_detections=0
	possible_detections_par=0
	day_of_detection=[]
	day_of_detection_par=[]
	F1_detection=[]
	F1_detection_par=[]
	possible_glitches=[]
	possible_glitches_par=[]
	hist=[]
	hist_par=[]
	print("Results comparing consecutive data spans")
	for i in range(len(F0_jump)):
		hist.append(F0_jump[i].nominal_value)
		if np.abs(F0_jump[i].nominal_value) < thresh:
			no_glitches= no_glitches + 1
		else:
			possible_glitches.append(F0_jump[i])
			possible_detections+=1
			day_of_detection.append(windows[i+n_obs_dense])
			F1_detection.append(F1_jump[i])
	print("No detections= " + str(no_glitches))
	print("Possible detections=  "+str(possible_detections))
	for i in range(possible_detections):
		print("Possible detection on " + str(day_of_detection[i]) + " of " + str(possible_glitches[i]) + " and " + "F1 jump= " + str(F1_detection[i]))
	plt.figure()
	plt.hist(np.array(hist), bins=np.arange(np.min(hist),np.max(hist),1e-10),label="F0 jump consecutive")
	#plt.xscale("log")
	plt.legend()
	plt.grid()
	plt.title("n= " + str(n_obs_dense) + ", fitf1=" + str(fitf1) + ": consecutive data")
	plt.savefig("n=" + str(n_obs_dense) + "_fitf1=" + str(fitf1) + "_consecutive data")
	plt.show()

	print("Results comparing with parfile")
	for i in range(len(F0_par_jump)):
		hist_par.append(F0_par_jump[i].nominal_value)
		if np.abs(F0_par_jump[i].nominal_value) < thresh:
			no_glitches_par= no_glitches_par + 1
		else:
			possible_glitches_par.append(F0_par_jump[i])
			possible_detections_par+=1
			day_of_detection_par.append(windows[i+n_obs_dense])
			F1_detection_par.append(F1_par_jump[i])
	print("No detections= " + str(no_glitches_par))
	print("Possible detections=  "+str(possible_detections_par))
	for i in range(possible_detections_par):
		print("Possible detection on " + str(day_of_detection_par[i]) + " of " + str(possible_glitches_par[i]) + " and " + "F1 jump= " + str(F1_detection_par[i]))
	plt.figure()
	plt.hist(np.array(hist_par), bins=np.arange(np.min(hist_par),np.max(hist_par),1e-10), label="F0 jump consecutive")
	#plt.xscale("log")
	plt.legend()
	plt.grid()
	plt.title("n= " + str(n_obs_dense) + ", fitf1=" + str(fitf1) + ": respect to parfile")
	plt.savefig("n=" + str(n_obs_dense) + "_fitf1=" + str(fitf1) + "_respect to parfile")
	plt.show()



def set_argparse():
   # add arguments
   parser = argparse.ArgumentParser(prog='windows_v7.py',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description='Main pipeline for mini glitch detections')
   parser.add_argument('--n_obs_dense', default=50, type=int,
      help='number of observations in each window')
   parser.add_argument('--thresh', default=1e-8, type=float,
      help='relative jump of glitch alert')
   parser.add_argument('--fitf1', default=False, type=lambda x: (str(x).lower() == 'true'), help='Fit F1? False or True')
   return parser.parse_args()


if __name__ == '__main__':

   # get cli-arguments
	args = set_argparse()
	print(args.fitf1)
	windows(args.fitf1, args.n_obs_dense, args.thresh)

