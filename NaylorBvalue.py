import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf



# rawdat = pd.read_table("2009GL040460-text03.txt", sep="\s+", header=None)
# dat = rawdat.iloc[:, 0].values

wd = "C:/WIWIN/UNHAS/PSHA/BuatPAPERDewi/segmenYopi/b-valueMap/Palukoro-Matano/"
#cat_file = "(WardMethod)Cluster_8_5.5.xlsx"#
cat_file = "(WardMethod)Cluster_25_2.5.xlsx"
#sheet_name = "Decluster"
# cat_file = "DataNaylor.xlsx"
sheet_name = "Setelah" 
# READ CATALOG
cat = pd.read_excel(wd + cat_file, sheet_name=sheet_name)
mag = cat['magnitude']

minmag = 4.0
maxmag = 6
binwidth = 0.1

mag = mag[(mag >= minmag) & (mag <= maxmag)]
mag, edges = np.histogram(mag, bins=np.arange(minmag, maxmag, binwidth))
x = (edges[:-1] + edges[1:]) / 2
y = mag

pois = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson()).fit()
link_instance = sm.families.links.log()
glin = sm.GLM(y[y > 0], sm.add_constant(x[y > 0]), family=sm.families.Gaussian(link=link_instance)).fit()
glog = sm.OLS(np.log10(y[y > 0]), sm.add_constant(x[y > 0])).fit()

qhi = 0.975
qlo = 0.025

cipois = pois.conf_int(alpha=0.05) / np.log(10)
ciglin = glin.conf_int(alpha=0.05) / np.log(10)
ciglog = glog.conf_int(alpha=0.05)

plt.figure(figsize=(18, 6))

# Poisson plot
plt.subplot(1, 3, 1)
plt.scatter(x[y > 0], np.log10(y[y > 0]), label='Data')
plt.plot(x, np.log10(pois.predict(sm.add_constant(x))), 'r-', label=f'Poisson fit (b={-pois.params[1]:.2f} ± {pois.bse[1]:.2f})')
plt.plot(x, np.log10(poisson.ppf(qlo, pois.predict(sm.add_constant(x)))), '--r', label='95% CI (Poisson)', alpha=0.5)
plt.plot(x, np.log10(poisson.ppf(qhi, pois.predict(sm.add_constant(x)))), '--r', alpha=0.5)


#plt.fill_between(x, np.log10(poisson.ppf(qlo, pois.predict(sm.add_constant(x)))), np.log10(poisson.ppf(qhi, pois.predict(sm.add_constant(x)))), color='r', alpha=0.1)
plt.xlabel('Magnitude',fontsize=20)
plt.ylabel('log10(Frequency)',fontsize=20)
plt.xticks(fontsize=20)  # Memperbesar ukuran angka di sumbu x
plt.yticks(fontsize=20)  # Memperbesar ukuran angka di sumbu y
plt.legend()

# Gaussian (log link) plot
plt.subplot(1, 3, 2)
plt.scatter(x[y > 0], np.log10(y[y > 0]), label='Data')
plt.plot(x[y > 0], np.log10(glin.predict(sm.add_constant(x[y > 0]))), 'g-', label=f'Gaussian (log) fit (b={-glin.params[1]:.2f} ± {glin.bse[1]:.2f})')
plt.plot(x[y > 0], np.log10(norm.ppf(qlo, glin.predict(sm.add_constant(x[y > 0])), scale=np.std(np.log10(y[y > 0]) - glin.predict(sm.add_constant(x[y > 0]))))), '--g', label='95% CI (Gaussian (log) fit)', alpha=0.5)
plt.plot(x[y > 0], np.log10(norm.ppf(qhi, glin.predict(sm.add_constant(x[y > 0])), scale=np.std(np.log10(y[y > 0]) - glin.predict(sm.add_constant(x[y > 0]))))), '--g', alpha=0.5)

#plt.fill_between(x[y > 0], np.log10(norm.ppf(qlo, glin.predict(sm.add_constant(x[y > 0])), scale=np.std(np.log10(y[y > 0]) - glin.predict(sm.add_constant(x[y > 0]))))), np.log10(norm.ppf(qhi, glin.predict(sm.add_constant(x[y > 0])), scale=np.std(np.log10(y[y > 0]) - glin.predict(sm.add_constant(x[y > 0]))))), color='g', alpha=0.1)
plt.xlabel('Magnitude',fontsize=20)
plt.ylabel('log10(Frequency)',fontsize=20)
plt.legend()
plt.title('B-value for cluster 25',fontsize=20)
plt.xticks(fontsize=20)  # Memperbesar ukuran angka di sumbu x
plt.yticks(fontsize=20)  # Memperbesar ukuran angka di sumbu y
#plt.title('Naylor et al. 2009 (Modified)')
# OLS (log10) plot
plt.subplot(1, 3, 3)
plt.scatter(x[y > 0], np.log10(y[y > 0]), label='Data')
plt.plot(x[y > 0], glog.predict(sm.add_constant(x[y > 0])), 'b-', label=f'OLS (log10) fit (b={-glog.params[1]:.2f} ± {glog.bse[1]:.2f})')
plt.plot(x[y > 0], norm.ppf(qlo, glog.predict(sm.add_constant(x[y > 0])), scale=np.std(np.log10(y[y > 0]) - glog.predict(sm.add_constant(x[y > 0])))), '--b', label='95% CI (OLS)', alpha=0.5)
plt.plot(x[y > 0], norm.ppf(qhi, glog.predict(sm.add_constant(x[y > 0])), scale=np.std(np.log10(y[y > 0]) - glog.predict(sm.add_constant(x[y > 0])))), '--b', alpha=0.5)

#plt.fill_between(x[y > 0], norm.ppf(qlo, glog.predict(sm.add_constant(x[y > 0])), scale=np.std(np.log10(y[y > 0]) - glog.predict(sm.add_constant(x[y > 0])))), norm.ppf(qhi, glog.predict(sm.add_constant(x[y > 0])), scale=np.std(np.log10(y[y > 0]) - glog.predict(sm.add_constant(x[y > 0])))), color='b', alpha=0.1)
plt.xlabel('Magnitude',fontsize=20)
plt.ylabel('log10(Frequency)',fontsize=20)
plt.legend()
plt.xticks(fontsize=20)  # Memperbesar ukuran angka di sumbu x
plt.yticks(fontsize=20)  # Memperbesar ukuran angka di sumbu y
plt.tight_layout()


plt.show()




