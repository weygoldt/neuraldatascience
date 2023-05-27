import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import scipy.optimize as opt
from IPython import embed

#matplotlib inline

plt.style.use("../matplotlib_style.txt")

def gen_gauss_rf(D, width, center=(0,0)):
    
    sz = (D-1)/2
    x, y = np.meshgrid(np.arange(-sz, sz + 1), np.arange(-sz, sz + 1))
    x = x + center[0]
    y = y + center[1]
    w = np.exp(- (x ** 2/width + y ** 2 / width))
    w = w / np.sum(w.flatten())
    
    return w

def gen_gauss_rf_multiD(D,F, width, center=(0,0)):
    
    sz = (D-1)/2
    sy = (F-1)/2
    x, y = np.meshgrid(np.arange(-sy, sy + 1), np.arange(-sz, sz + 1))
    x = x + center[0]
    y = y + center[1]
    w = np.exp(- (x ** 2/width + y ** 2 / width))
    w = w / np.sum(w.flatten())
    
    return w


w = gen_gauss_rf(15,7,(1,1))

vlim = np.max(np.abs(w))
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.imshow(w, cmap='bwr', vmin=-vlim, vmax=vlim)
ax.set_title('Gaussian RF')

def sample_lnp(w, nT, dt, R, v):
    """Generate samples from an instantaneous LNP model neuron with
    receptive field kernel w.

    Parameters
    ----------

    w: np.array, (Dx * Dy, )
        (flattened) receptive field kernel.

    nT: int
        number of time steps

    dt: float
        duration of a frame in s

    R: float
        rate parameter

    v: float
        variance of the stimulus

    Returns
    -------

    c: np.array, (nT, )
        sampled spike counts in time bins

    r: np.array, (nT, )
        mean rate in time bins

    s: np.array, (Dx * Dy, nT)
        stimulus frames used

    Note
    ----

    See equations in task description above for a precise definition
    of the individual parameters.
    $$
    c_t \sim Poisson(r_t)\\
    r_t = \exp(w^T s_t) \cdot \Delta t \cdot R
    $$

    """

    np.random.seed(10) 
    stimulus = np.random.choice([0,1], (w.shape[0], nT))
    stimulus = stimulus * np.sqrt(v)
    
    mean_rate = np.exp(w.T @ stimulus) * dt * R
    spike_counts = np.random.poisson(mean_rate)  

    return spike_counts, mean_rate , stimulus

D = 15     # number of pixels
nT = 100000  # number of time bins
dt = 0.1   # bins of 100 ms
R = 50     # firing rate in Hz 
v = 5      # stimulus variance

w = gen_gauss_rf(D,7,(1,1))
w = w.flatten()
print(w.shape)

c, r, s = sample_lnp(w, nT, dt, R, v)
print(c.shape, r.shape, s.shape,)
sreshape = s.reshape((D,D,nT))
print(sreshape.shape)

#%matplotlib qt6

mosaic = "ABC"

fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(15,4))
ax["A"].plot(c)
ax["A"].set_title("Spike counts")
ax["B"].plot(r)
ax["B"].set_title("Mean rate")

ax["C"].imshow(sreshape[:,:,0], cmap='bwr')
ax["C"].set_title("Stimulus frames")

# plot the stimulus grid only for one frame

def L_omega(x, c, s, dt=0.1, R=50):
  '''Implements the negative (!) log-likelihood of the LNP model and its
  gradient with respect to the receptive field w.

  $$ L(\omega) = \sum_t \log(\exp(w^T s_t) \cdot \Delta t \cdot R)^{c_t}-\log(c_t!)-\exp(w^T s_t) \cdot \Delta t \cdot R$$

  Parameters
  ----------

  x: np.array, (Dx * Dy, )
    current receptive field 

  c: np.array, (nT, )
    spike counts 

  s: np.array, (Dx * Dy, nT)
    stimulus matrix


  Returns
  -------

  f: float
    function value of the negative log likelihood at x

  '''
  if x.shape[0] != s.shape[0]:
    raise ValueError("x and s must have the same dimension")
  if c.shape[0] == 0:
    return - np.sum(np.log((np.exp(x.T @ s) * dt * R) ** c) - np.log(float(np.math.factorial(int(c)))) - np.exp(x.T @ s) * dt * R)

  #print(np.log((np.exp(x.T @ s) * dt * R) ** c))
  #print(np.log([float(np.math.factorial(int(c_single))) for c_single in c]))
  #print(np.exp(x.T @ s) * dt * R)
  return - np.sum(np.log((np.exp(x.T @ s) * dt * R) ** c) - np.log([float(np.math.factorial(int(c_single))) for c_single in c]) - np.exp(x.T @ s) * dt * R)

def dL_omega(x, c, s, dt=0.1, R=50):
  '''Implements the negative (!) log-likelihood of the LNP model and its
  gradient with respect to the receptive field w.

  $$ L(\omega) = \sum_t \log(\exp(w^T s_t) \cdot \Delta t \cdot R)^{c_t}-\log(c_t!)-\exp(w^T s_t) \cdot \Delta t \cdot R$$

  Parameters
  ----------

  x: np.array, (Dx * Dy, )
    current receptive field 

  c: np.array, (nT, )
    spike counts 

  s: np.array, (Dx * Dy, nT)
    stimulus matrix


  Returns
  -------
  df: np.array, (Dx * Dy, )
    gradient of the negative log likelihood with respect to x 
  '''
  return - np.sum(s * (c - np.exp(x.T @ s) * dt * R), axis=1)

f = L_omega(w, c, s, dt, R)
df = dL_omega(w, c, s, dt, R)
print(f, df.shape)


err = opt.check_grad(L_omega, dL_omega, w, c, s, dt, R)
print(err)

# insert your code here 
res = opt.minimize(L_omega, x0=w, args=(c, s, dt, R), jac=dL_omega, method='SLSQP', options={'disp': True, 'maxiter': 1000})
print(np.shape(res))
print(res.x.shape)
# ------------------------------------------
# Estimate the receptive field by maximizing
# the log-likelihood (or more commonly, 
# minimizing the negative log-likelihood).
# 
# Tips: use scipy.optimize.minimize(). (1 pt)
# ------------------------------------------




# insert your code here 

# ------------------------------------
# Plot the ground truth and estimated 
# `w` side by side. (0.5 pts)
# ------------------------------------
vlim = np.max(np.abs(w))
mosaic = [["True", "Estimated"]]
fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(12,5), sharey=True,)
# plot grid
ax["True"].set_xticks(np.arange(0, D, 1))
ax["True"].set_yticks(np.arange(0, D, 1))
ax["Estimated"].set_xticks(np.arange(0, D, 1))
ax["Estimated"].set_yticks(np.arange(0, D, 1))
ax["True"].set_xlabel("x")
ax["True"].set_ylabel("y")
ax["Estimated"].set_xlabel("x")
ax["Estimated"].set_ylabel("y")
# plot vlines and hlines
ax["True"].vlines(np.arange(0, D, 1), ymin=0, ymax=D, color='k', linewidth=0.5)
ax["True"].hlines(np.arange(0, D, 1), xmin=0, xmax=D, color='k', linewidth=0.5)
ax["Estimated"].vlines(np.arange(0, D, 1), ymin=0, ymax=D, color='k', linewidth=0.5)
ax["Estimated"].hlines(np.arange(0, D, 1), xmin=0, xmax=D, color='k', linewidth=0.5)

ax["True"].imshow(w.reshape((D,D)), cmap='bwr', vmin=-vlim, vmax=vlim)
ax["True"].set_title("True RF") 
ax["Estimated"].imshow(res.x.reshape((D,D)), cmap='bwr', vmin=-vlim, vmax=vlim)
ax["Estimated"].set_title("Estimated RF")
# make sure to add a colorbar. 'bwr' is a reasonable choice for the cmap.

var = io.loadmat('../data/nds_cl_5_data.mat')

# t contains the spike times of the neuron
t = var['DN_spiketimes'].flatten()    

# trigger contains the times at which the stimulus flipped
trigger = var['DN_triggertimes'].flatten()

# contains the stimulus movie with black and white pixels
s = var['DN_stim']
print(s.shape)
s = s.reshape((300,1500)) # the shape of each frame is (20, 15)
s = s[:,1:len(trigger)]
print(s.shape) 
print(t.shape)
print(trigger.shape)


# insert your code here 

# ------------------------------------------
# Bin the spike counts at the same temporal
# resolution as the stimulus (0.5 pts)
# ------------------------------------------

spikes = np.zeros((len(trigger)-1)) 
for i in range(len(trigger)-1):
        spikes[i] = len(t[(t>=trigger[i]) & (t<trigger[i+1])])
print(spikes.shape) 
print(spikes[0:10])


# insert your code here 

# ------------------------------------------
# Fit the receptive field with time lags of
# 0 to 4 frames separately (1 pt)
# 
# The final receptive field (`w_hat`) should
# be in the shape of (Dx * Dy, 5)
# ------------------------------------------

# specify the time lags
delta = [0, 1, 2, 3, 4] 
# intial guess for the receptive field
w0 = gen_gauss_rf(20, 15, (1,1))
w0 = w0.flatten()   
print(w0.shape) 

# first time lag is 0 so we don't need to shift the stimulus 

#ll = L_omega(w0, spikes, s)
w0 = np.zeros(300)

print(w0.shape)
print(spikes[0].shape[0])
print(s[:, 0])

ll = L_omega(w0, spikes[0], s[:, 0])
#res = opt.minimize(L_omega, x0=w0, args=(spikes[0], s[:, 0]), jac=dL_omega, method='Newton-CG', options={'disp': True, 'maxiter': 4000})

 

# fit for each delay





# insert your code here 

# --------------------------------------------
# Plot all 5 frames of the fitted RFs (0.5 pt)
# --------------------------------------------

fig, ax = plt.subplot_mosaic(mosaic=[delta], figsize=(10,4), constrained_layout=True)
ax[0].imshow(res.x.reshape((20,15)), cmap='bwr', vmin=-vlim, vmax=vlim)

# insert your code here 

# --------------------------------------------
# Apply SVD to the fitted receptive field,
# you can use either numpy or sklearn (0.5 pt)
# --------------------------------------------

# shape of w_hat: (300,5). 
# subtract mean along axis=1 (time) since for every timelag the RF is estimated independently

# -------------------------------------------------
# Plot the spatial and temporal components (0.5 pt)
# -------------------------------------------------

fig, ax = plt.subplot_mosaic(mosaic=[['Spatial', 'Temporal']], figsize=(10,4), constrained_layout=True)
# add plot



# insert your code here 

# ------------------------------------------
# Fit the receptive field with time lags of
# 0 to 4 frames separately (the same as before) 
# with sklern or pyglmnet (1 pt)
# ------------------------------------------

delta = [0, 1, 2, 3, 4]


# ------------------------------------------
# plot the estimated receptive fields (1 pt)
# ------------------------------------------

fig, ax = plt.subplot_mosaic(mosaic=[delta], figsize=(10,4), constrained_layout=True)
# add plot
