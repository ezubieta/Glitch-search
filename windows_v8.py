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
import astropy.stats as st
import logging
import sys

#Archivos a usar:
#parfile = "J0835-4510_2020-2022.par"
#timfile = "J0835-4510_A1_2020-2022.tim"
#parfile = "J1048-5832.par"
#timfile = "J1048-5832.tim"
#parfile = "J1048-5832_noise.par"




#Filtrar ToAs: dot e inv are used inside mask_toas function
def dot(l1,l2):
    return np.array([v1 and v2 for v1,v2 in zip(l1,l2)])
def inv(l):
    return np.array([not i for i in l])

def mask_toas(toas,before=None,after=None,on=None,window=None): #This function filters MJD between befora and after
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


def delta_nu(nu0, nu1): #relative jump definition
        return (nu1-nu0)/nu0







def windows(parfile,timfile,fitf1=False,n_obs_dense=10, thresh=1e-8): #Everything happens inside this function
	parfile = parfile #from parser
	timfile_v1 = timfile #from parser
	file=pd.read_csv(timfile_v1, skiprows=1, comment='C', sep=' ', usecols=[0,1,2,3,4,5,6],  skipinitialspace=True, names=['FORMAT 1', 'Freq','MJD','TOA', 'Antenna', 'pn', 'pulse_number']) #read timfile and select useful columns
	print(file)
	file_v2 = file.sort_values(by=["MJD"]) #order timfile
	print(file_v2)
	delete = []
	for i in range(len(file_v2)-1): #choosing repeated MJDS
     		if file_v2["MJD"][i+1]==file_v2["MJD"][i]:
         		delete.append(i+1)
	file_v3=file_v2.drop(delete, axis=0) #delete
	print(file_v3)
	file_v3.to_csv("clean" + timfile_v1, sep=' ',header=False, index=False, index_label=None) #save clean timfile
	timfile="clean"+timfile_v1
	print(timfile)
	print("I saved a cute and clean file.tim in your folder")
	m, t_all = get_model_and_toas(parfile, timfile) #import model and toas
	t=t_all[t_all.get_errors() < 3 * u.ms] #filtering toas with big errors
	m.F0.frozen=False #fit F0 = yes
	windows=t.get_mjds().value
	m.F1.frozen = not fitf1 #from parser we decide if we fit F1
	#We first fit for all the toas and get F0 and F1:
	f_all=pint.fitter.DownhillWLSFitter(t,m)
	#f_all.fit_toas()
	F0=[]
	F0_par=f_all.model["F0"].value
	F0_par_error=f_all.model["F0"].uncertainty_value
	F1=[]
	F0_error=[]
	F1_error=[]
	F0_jump = []
	F0_par_jump=[]
	F1_jump=[]
	F1_par=f_all.model["F1"].value
	F1_par_error=f_all.model["F1"].uncertainty_value
	F1_par_jump=[]
	window_first_MJD=[]
	for i in range(len(windows)-(n_obs_dense)): #we skip the last n mjds because they cant complete a window to get f0.
		window=mask_toas(t, before=windows[i], after=windows[i+n_obs_dense]) #choose first window and move one to the right
		f=pint.fitter.DownhillWLSFitter(window,m) #import fitter
		f.fit_toas(maxiter=50) #fit
		F0_window=f.model["F0"].value #keep f0 and f1 and errors for the window
		F1_window=f.model["F1"].value
		F0_error_window=f.model["F0"].uncertainty_value
		F1_error_window=f.model["F1"].uncertainty_value
		F0.append(F0_window)
		F1.append(F1_window)
		F0_error.append(F0_error_window)
		F1_error.append(F1_error_window)
		print(F0_window, F0_error_window)
		window_first_MJD.append(windows[i+n_obs_dense]) #save the beginnings of the windows
	if fitf1==False:
		F1_error=np.zeros(len(F0_error))
		F1_par_error=0
	for i in range(len(F0)-(n_obs_dense)): #comparing two consecutive windows, we have #mjds - 2n jumps.
		nu0=ufloat(F0[i], F0_error[i])
		print(nu0)
		nudot0=ufloat(F1[i],F1_error[i])
		print(nudot0)
		nu1=ufloat(F0[i+n_obs_dense], F0_error[i+n_obs_dense])
		nudot1=ufloat(F1[i+n_obs_dense], F1_error[i+n_obs_dense])
		F0_jump.append(delta_nu(nu0,nu1))
		F1_jump.append(delta_nu(nudot0, nudot1))
	for i in range(len(F0)): #comparing with parfile, we have #mjds - n jumps
		nu0_par=ufloat(F0_par, F0_par_error)
		nudot0_par=ufloat(F1_par, F1_par_error)
		nu0=ufloat(F0[i],F0_error[i])
		nudot0=ufloat(F1[i], F1_error[i])
		F0_par_jump.append(delta_nu(nu0_par, nu0))
		F1_par_jump.append(delta_nu(nudot0_par, nudot0))
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
	hist_err=[]
	hist_par=[]
	hist_par_err=[]
	windows_plot=window_first_MJD
	print("Results comparing consecutive data spans")
	for i in range(len(F0_jump)):
		hist.append(F0_jump[i].nominal_value) #for the histogram
		hist_err.append(F0_jump[i].s)
		if np.abs(F0_jump[i].nominal_value) < thresh: #counting detections and no detections
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

	print("Results comparing with parfile")
	for i in range(len(F0_par_jump)):
		hist_par.append(F0_par_jump[i].nominal_value)
		hist_par_err.append(F0_par_jump[i].s)
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
	rs=pint.residuals.Residuals(t, m)
	fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharex='col')
	plt.text(0.4, 0.98,"PSR" + str(args.timfile)+"; n= " + str(n_obs_dense) + "; fitf1=" + str(fitf1) + "; thresh=" + str(thresh), transform=fig.transFigure)
	plt.text(0.39, 0.94,"Possible detections:" + str(possible_detections) + " respect last window - " + str(possible_detections_par) + " respect parfile", transform=fig.transFigure)
	ax1.hist(np.array(np.abs(hist)), bins=np.arange(np.min(hist),np.max(hist),st.knuth_bin_width(np.array(hist))),label="F0 jump consecutive")
	ax1.legend()
	ax1.grid()
	ax2.errorbar(windows, rs.time_resids.to(u.ms), rs.toas.get_errors().to(u.ms), label="Original residuals", fmt="X", markersize=1)
	ax2.legend()
	ax2.grid()
	ax2.set_ylabel("Residuals (ms)")
	ax5=ax2.twinx()
	ax5.errorbar([windows_plot[i+n_obs_dense] for i in range(len(windows_plot) - n_obs_dense)], hist, hist_err, fmt=".", label="F0 jump consecutive", color="red")
	ax5.set_ylabel("F0 jump")
	ax5.legend()
	ax3.hist(np.array(np.abs(hist_par)), bins=np.arange(np.min(hist_par),np.max(hist_par),st.knuth_bin_width(np.array(hist_par))),label="F0 jump respect parfile")
	ax3.set_xlabel("Relative F0 jump")
	ax3.legend()
	ax3.grid()
	ax4.errorbar(windows, rs.time_resids.to(u.ms), rs.toas.get_errors().to(u.ms), label="Original residuals", fmt="X", markersize=1)
	ax4.legend()
	ax4.grid()
	ax4.set_xlabel("MJD")
	ax4.set_ylabel("Residuals (ms)")
	ax6=ax4.twinx()
	ax6.errorbar(windows_plot, hist_par, hist_par_err, fmt=".", label="F0 jump respect parfile", color="red")
	ax6.set_ylabel("F0 jump")
	ax6.legend()
	#np.savetxt("n" + str(n_obs_dense) + "_consecutive", hist)
	#np.savetxt("n" + str(n_obs_dense) + "_par", hist_par)
	plt.show()



def set_argparse():
   # add arguments
   parser = argparse.ArgumentParser(prog='windows_v8.py',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description='Main pipeline for mini glitch detections')
   parser.add_argument('--n', default=50, type=int,
      help='number of observations in each window')
   parser.add_argument('--thresh', default=1e-8, type=float,
      help='relative jump of glitch alert')
   parser.add_argument('--fitf1', default=False, type=lambda x: (str(x).lower() == 'true'), help='Fit F1? False or True')
   parser.add_argument('--parfile', help='name of the file.par')
   parser.add_argument('--timfile', help='name of the file.tim')
   return parser.parse_args()


if __name__ == '__main__':
	# get cli-arguments
	pint.logging.setup(level="ERROR")
	args = set_argparse()
	windows(args.parfile,args.timfile,args.fitf1, args.n, args.thresh)

