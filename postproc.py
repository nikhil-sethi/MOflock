#%%
import pickle

opt = pickle.load(open("sims/29-01-21/opt_pickle", "rb"))
res = opt["res"]

#%%
import numpy as np

data = np.concatenate(opt["data"])

# get actual fitness ops
ops = []
for pop in opt["ops"]:
    for env in pop:
        ops.append(env[-1])
ops = np.array(ops)
pops = np.concatenate(opt["data"])[:, :11]
fits = np.concatenate(opt["data"])[:, -6:]
# get all data
data = np.hstack((pops, ops, fits))
#%%
from pymoo.visualization.scatter import Scatter
import sklearn.preprocessing as p
scatter = Scatter(legend=True)
k = 10
g = 40
for i in range(k):
    h = p.minmax_scale(opt["res_history"][-i][1])
    h = h[np.argsort(h[:, 0])]
    scatter.add(h, label=f'{g-i}', plot_type='line')
    i -= 1
scatter.show()


# Convergence graph
import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
n_evals = np.array([e.evaluator.n_eval for e in res.history])
opts = np.array([e.opt[0].F for e in res.history])
ax1.title("Convergence")
ax1.plot(n_evals, opts, "--")
ax1.yscale("log")
ax1.show()

# nondom sorting on all
#%%
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

nds = NonDominatedSorting(method="fast_non_dominated_sort").do(np.array(fits), only_non_dominated_front=False)
#%%
nds_flat = np.concatenate(nds)
data_sorted = np.hstack((data[np.concatenate(nds)]))

#%%
# plot k fronts without saved history
numfronts = 40
import matplotlib.pyplot as plt
#
fig, ax = plt.subplots()
fronts = []
for i in range(0,numfronts,4):
    f = fits[nds[i]]
    fronts.append(f)
    f = f[np.argsort(f[:, 1])][::-1]
    ax.plot(f[:, 0], f[:, 1])
ax.legend(range(0, numfronts, 4), fontsize=13, title='Ranks')
ax.tick_params(axis='both', which='major', labelsize=15)
#%%

# run pca on order params
import sklearn.preprocessing as p
from sklearn.decomposition import PCA
pca_all = PCA(n_components=6)
pca_all.fit(p.scale(fits))

comp = pca.components_

comp[0]

#%% plot comparisons for ops\
import matplotlib.pyplot as plt
fig, (ax3, ax4) = plt.subplots(1,2)
op1=op_all[0][0]/np.array([5,6,1,0.0001,2,30])
op2=op_all[1][0]/np.array([5,6,1,0.0001,2,30])
ax3.plot(op1)
ax4.plot(op2)
ax3.grid()
ax4.grid()
ax3.legend([r'$\phi_{wall}/r_{tol}$',r'$\phi_{vel}/v_{flock}$',r'$\phi_{corr}$',r'$\phi_{coll}/a_{tol}/3$',r'$\phi_{disc}/2$',r'$N_{min}/N_{agents}$'], prop={'size':13})
ax3.set_xlabel('Time(s)', fontsize=15)
ax4.set_xlabel('Time(s)', fontsize=18)
ax3.set_ylabel('Order Parameter', fontsize=18)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax4.tick_params(axis='both', which='major', labelsize=12)
fig.savefig('./Media/op_comparison2.png', dpi=300)

# create comparison for waypoint op
# first run the simulation with the relevant targets and flock parametersto \
# get the order parameters
import scipy
def fit_sin(tt, yy):
    # courtesy of https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return A, w
    # return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

len_t = 363 # length of the time series 
A1, w1=fit_sin(np.arange(100, len_t), op_all[1][1][100:,-1])
A2, w2=fit_sin(np.arange(100, len_t), op_all[1][1][100:,-1])
phi_target_A= w1/A1
phi_target_B = w2/A2