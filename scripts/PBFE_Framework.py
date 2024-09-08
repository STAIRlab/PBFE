import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
%matplotlib inline
from scipy.stats import lognorm
import seaborn as sns
sns.set_theme(style='darkgrid', font_scale = 1.0)
rc={'figure.figsize':(12,8)}
plt.rc('figure', dpi=100, figsize=(7, 5))
plt.rc('font', size=12)
rng = np.random.default_rng()
import itertools


# Assume p_IM(D1) = 1.0, p_IM(Others) = 0
p_IM = np.zeros(5)
p_IM[1] = 1.0 


# Develop the probability distributions of LAI for each IM
dr_levels = np.arange(5) + 1

mean_SM = 2 - dr_levels * 0.2   # mean_LAI is assumed to decrease linearly with the drought
cov_SM = 0.2                    # cov_LAI is assumed to be the same for all levels
                                # Actual mean and COV values to be found from literature or using our own dataset
var_SM = mean_SM * cov_SM
incr_SM = 0.02
max_SM = 6

# Find the parameters to be used with the lognorm function 
mu_SM = np.log(mean_SM ** 2 / np.sqrt(var_SM + mean_SM ** 2))
sd_SM = np.sqrt(np.log(1 + var_SM / mean_SM ** 2) )

# Plot the PDF
vals_SM = np.linspace(incr_SM, max_SM, num = 300)

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
fig.suptitle('Probability Density Distributions of LAI for each IM', fontsize=13)
colors = pl.cm.YlOrRd(np.linspace(0,1,7))

p_SM = np.zeros([len(dr_levels), len(vals_SM)])
sumprob_SM = np.zeros(len(dr_levels))

for i in np.arange(len(dr_levels)):
    rv = lognorm(sd_SM[i], scale=np.exp(mu_SM[i]))    
    ax.plot(vals_SM, rv.pdf(vals_SM), lw=2, alpha=0.8, color=colors[i+2])
    p_SM[i, :] = rv.pdf(vals_SM) * incr_SM;   # SM probability
    sumprob_SM[i] = sum(p_SM[i, :])

ax.legend(['D0', 'D1', 'D2', 'D3', 'D4']);
ax.set(xlabel='LAI', ylabel='PDF');

# Develop the probability distributions of TFP for each threshold LAI value
mean_YP = 0.2 + vals_SM / 6   # mean_TFP is assumed to be equal to 0.2 + mean_LAI / 6
cov_YP = 0.2                  # cov_TFP is assumed to be the same for all levels
                              # Actual mean and COV values are to be found from literature
var_YP = mean_YP * cov_YP
incr_YP = 0.005
max_YP = 1.8                  # This is larger than 1 considering the variance

# Find the parameters to be used with the lognorm function 
mu_YP = np.log(mean_YP ** 2 / np.sqrt(var_YP + mean_YP ** 2))
sd_YP = np.sqrt(np.log(1 + var_YP / mean_YP ** 2) )

# Plot the PDF 
vals_YP = np.linspace(incr_YP, max_YP, num = 360)

p_YP = np.zeros([len(vals_SM), len(vals_YP)])
sumprob_YP = np.zeros(len(vals_SM))

plt.figure(figsize = [8, 6])
colors = pl.cm.YlGn(np.linspace(0,1,4))

for i in np.arange(len(vals_SM)):
    rv = lognorm(sd_YP[i], scale=np.exp(mu_YP[i]))
    pdf_YP = rv.pdf(vals_YP)
    p_YP[i, :] = pdf_YP * incr_YP   # YP probability
    sumprob_YP[i] = sum(p_YP[i, :])
    if vals_SM[i] in [0.1, 2, 4]:
        sns.lineplot(x=vals_YP, y=pdf_YP, lw=2, alpha=0.8, color=colors[[0.1, 2, 4].index(vals_SM[i])+1], label='LAI = %1.2f' % vals_SM[i])
plt.legend()
plt.xlabel('TFP')
plt.ylabel('PDF');

  
# Develop the probability distributions of DV for each TFP value
mean_DV = 1 - vals_YP / 2   # mean_DV is assumed to be equal to 1 - mean_TFP / 2
cov_DV = 0.2                # cov_DV is assumed to be the same for all levels
                            # Actual mean and COV values are to be found from literature
var_DV = mean_DV * cov_DV
incr_DV = 0.005
max_DV = 2                  # This is larger than 1 considering the variance

# Find the parameters to be used with the lognorm function 
mu_DV = np.log(mean_DV ** 2 / np.sqrt(var_DV + mean_DV ** 2))
sd_DV = np.sqrt(np.log(1 + var_DV / mean_DV ** 2) )

# Plot the PDF 
vals_DV = np.linspace(incr_DV, max_DV, num = 400)

p_DV = np.zeros([len(vals_YP), len(vals_DV)])
sumprob_DV = np.zeros(len(vals_YP))
POE_DV = np.zeros([len(vals_YP), len(vals_DV)])

plt.figure(figsize = [8, 6])
colors = pl.cm.Accent(np.linspace(0,1,5))

for i in np.arange(len(vals_YP)):
    rv = lognorm(sd_DV[i], scale=np.exp(mu_DV[i]))
    pdf_DV = rv.pdf(vals_DV)
    p_DV[i, :] = pdf_DV * incr_DV   # DV probability
    sumprob_DV[i] = sum(p_DV[i, :])
    
    if vals_YP[i] in [0.5, 1, 1.5]:
        sns.lineplot(x=vals_DV, y=pdf_DV, lw=2, alpha=1.0, color=colors[[0.5, 1, 1.5].index(vals_YP[i])+2], label='TFP = %1.1f' % vals_YP[i])
    
    for j in np.arange(len(vals_DV)):
        if j == 0: 
            POE_DV[i, j] = sumprob_DV[i]
        else:
            POE_DV[i, j] = POE_DV[i, j-1] - p_DV[i,j]

plt.xlabel('DV')
plt.ylabel('PDF');


# Compute and Plot the Loss Curve
POE = np.zeros(len(vals_DV))

for n, k, i, m in itertools.product(np.arange(len(vals_DV)), np.arange(len(vals_YP)), np.arange(len(vals_SM)), np.arange(len(p_IM))):
    POE[n] = POE[n] + POE_DV[k, n] * p_YP[i,k] * p_SM[m,i] * p_IM[m]

plt.figure(figsize = [8, 6])
sns.lineplot(x=vals_DV, y=POE, lw=2, alpha=0.8, color='b')
plt.xlabel('DV (normalized revenue loss)')
plt.ylabel('Probability of Exceedance')
plt.title("Loss Curve");
