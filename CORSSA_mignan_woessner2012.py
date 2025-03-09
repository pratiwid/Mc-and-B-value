import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# FUNCTIONS
def fmd(mag, mbin):
    """Calculate Frequency Magnitude Distribution (FMD)."""
    min_mi = np.round(np.min(mag/mbin)*mbin)
    max_mi = np.round(np.max(mag/mbin)*mbin)
    mi = np.arange(min_mi, max_mi + mbin, mbin)
    nbm = len(mi)
    cumnbmag = np.zeros(nbm)
    nbmag = np.zeros(nbm)
    for i in range(nbm):
        cumnbmag[i] = len(np.where(mag > mi[i]-mbin/2)[0])
    cumnbmagtmp = np.append(cumnbmag, 0)
    nbmag = np.abs(np.diff(cumnbmagtmp))
    res = {'m': mi, 'cum': cumnbmag, 'noncum': nbmag}
    return res

# def fmd(mag, mbin):
#     # Membuat deret nilai bin magnitudo
#     mi = np.arange(np.floor(min(mag)/mbin)*mbin, np.ceil(max(mag)/mbin)*mbin, mbin)
#     nbm = len(mi)
#     cumnbmag = np.zeros(nbm)
#     nbmag = np.zeros(nbm)
    
#     # Menghitung jumlah data magnitudo yang lebih besar dari setiap bin
#     for i in range(nbm):
#         cumnbmag[i] = np.sum(mag > mi[i] - mbin/2)
    
#     # Menghitung jumlah data magnitudo yang jatuh dalam setiap bin
#     cumnbmagtmp = np.concatenate((cumnbmag, [0]))
#     nbmag = np.abs(np.diff(cumnbmagtmp))
    
#     # Menyusun hasil dalam bentuk dictionary
#     res = {'m': mi, 'cum': cumnbmag, 'noncum': nbmag}
#     return res

def maxc(mag, mbin):
    """Determine Mc using Maximum Curvature (Maxc)."""
    FMD = fmd(mag, mbin)
    Mc = FMD['m'][np.where(FMD['noncum'] == np.max(FMD['noncum']))[0][0]]
    return {'Mc': Mc}

# def maxc(mag, mbin):
#     fmd_result = fmd(mag, mbin)
#     Mc = fmd_result['m'][np.argmax(fmd_result['noncum'])]
#     return {'Mc': Mc}

def gft(mag, mbin):
    FMD = fmd(mag, mbin)
    McBound = maxc(mag, mbin)['Mc']
    Mco = McBound-0.4+(np.arange(15)-1)/10
    R = np.zeros(15)
    for i in range(15):
        indmag = np.where(mag > Mco[i]-mbin/2)[0]
        b = np.log10(np.exp(1))/(np.mean(mag[indmag])-(Mco[i]-mbin/2))
        a = np.log10(len(indmag))+b*Mco[i]
        FMDcum_model = 10**(a-b*FMD['m'])
        indmi = np.where(FMD['m'] >= Mco[i])[0]
        R[i] = np.sum(np.abs(FMD['cum'][indmi]-FMDcum_model[indmi]))/np.sum(FMD['cum'][indmi])*100
    indGFT = np.where(R <= 5)[0]
    if len(indGFT) != 0:
        Mc = Mco[indGFT[0]]
        best = "95%"
    else:
        indGFT = np.where(R <= 10)[0]
        if len(indGFT) != 0:
            Mc = Mco[indGFT[0]]
            best = "90%"
        else:
            Mc = McBound
            best = "MAXC"
    return {'Mc': Mc, 'best': best, 'Mco': Mco, 'R': R, 'FMDcum_model':FMDcum_model}
# def gft(mag, mbin):
#     fmd_result = fmd(mag, mbin)
#     McBound = maxc(mag, mbin)['Mc']
#     Mco = McBound - 0.4 + (np.arange(15) - 1) / 10
#     R = np.zeros(15)
    
#     for i in range(15):
#         indmag = np.where(mag > Mco[i] - mbin/2)
#         b = np.log10(np.exp(1)) / (np.mean(mag[indmag]) - (Mco[i] - mbin/2))
#         a = np.log10(len(indmag)) + b * Mco[i]
#         FMDcum_model = 10**(a - b * fmd_result['m'])
#         indmi = np.where(fmd_result['m'] >= Mco[i])
#         R[i] = np.sum(np.abs(fmd_result['cum'][indmi] - FMDcum_model[indmi])) / np.sum(fmd_result['cum'][indmi]) * 100
    
#     indGFT = np.where(R <= 5)[0]  # 95% confidence
#     if len(indGFT) != 0:
#         Mc = Mco[indGFT[0]]
#         best = "95%"
#     else:
#         indGFT = np.where(R <= 10)[0]  # 90% confidence
#         if len(indGFT) != 0:
#             Mc = Mco[indGFT[0]]
#             best = "90%"
#         else:
#             Mc = McBound
#             best = "MAXC"
    
#     return {'Mc': Mc, 'best': best, 'Mco': Mco, 'R': R}


def mbs(mag, mbin):
    McBound = maxc(mag, mbin)['Mc']
    Mco = McBound - 0.7 + np.arange(20) / 10
    bi = np.zeros(20)
    unc = np.zeros(20)
    
    for i in range(20):
        indmag = np.where(mag > Mco[i] - mbin / 2)[0]
        nbev = len(indmag)
        if nbev > 0:
            bi[i] = np.log10(np.exp(1)) / (np.mean(mag[indmag]) - (Mco[i] - mbin / 2))
            unc[i] = 2.3 * bi[i]**2 * np.sqrt(np.sum((mag[indmag] - np.mean(mag[indmag]))**2) / (nbev * (nbev - 1)))
    
    bave = np.zeros(15)
    for i in range(15):
        bave[i] = np.mean(bi[i:i+6])
    
    dbi_old = np.abs(np.diff(bi))
    indMBS_old = np.where(dbi_old <= 0.03)[0]
    dbi = np.abs(bave[:15] - bi[:15])
    indMBS = np.where(dbi <= unc[:15])[0]
    
    if len(indMBS) > 0:
        Mc = Mco[indMBS[0]]
    else:
        Mc = None
    
    return {'Mc': Mc, 'Mco': Mco, 'bi': bi, 'unc': unc, 'bave': bave, 'indMBS_old':indMBS_old, 'dbi_old':dbi_old,'dbi':dbi,'indMBS':indMBS}

# def mbs(mag, mbin):
#     McBound = maxc(mag, mbin)['Mc']
#     Mco = McBound-0.7+(np.arange(20)-1)/10
#     bi = np.zeros(20)
#     unc = np.zeros(20)
#     for i in range(20):
#         indmag = np.where(mag > Mco[i]-mbin/2)[0]
#         nbev = len(indmag)
#         bi[i] = np.log10(np.exp(1))/(np.mean(mag[indmag])-(Mco[i]-mbin/2))
#         unc[i] = 2.3*bi[i]**2*np.sqrt(np.sum((mag[indmag]-np.mean(mag[indmag]))**2)/(nbev*(nbev-1)))
#     bave = np.zeros(15)
#     for i in range(15):
#         bave[i] = np.mean(bi[i:i+5])
#     dbi_old = np.abs(np.diff(bi))
#     indMBS_old = np.where(dbi_old <= 0.03)[0]
#     dbi = np.abs(bave[0:15]-bi[0:15])
#     indMBS = np.where(dbi <= unc[0:15])[0]
#     #Mc = None  # Atau nilai default yang sesuai dengan kasus Anda

#     if len(indMBS) > 0:
#         Mc = Mco[indMBS[0]]
#     else:
#     # Tindakan yang diperlukan ketika indMBS kosong
#     # Misalnya, Mc diberi nilai default atau NaN
#         Mc = np.nan
    
#     #Mc = Mco[indMBS[0]]
#     return {'Mc': Mc, 'Mco': Mco, 'bi': bi, 'unc': unc, 'bave': bave}


# def emr(mag, mbin):
#     FMD = fmd(mag, mbin)
#     nbm = len(FMD['m'])
#     McMAXC = maxc(mag, mbin)['Mc']
#     mu = np.abs(McMAXC/2)
#     sig = np.abs(McMAXC/4)
#     if mu > 1:
#         mu = np.abs(McMAXC/10)
#         sig = np.abs(McMAXC/20)
#     McBound = McMAXC
#     Mco = McBound-0.3+(np.arange(9)-1)/10
#     params = np.zeros((9, 4))
#     prob = np.zeros(9)
#     savedmodel = np.zeros((9, nbm))
#     for i in range(9):
#         indmag = np.where(mag > Mco[i]-mbin/2)[0]
#         nbev = len(indmag)
#         b = np.log10(np.exp(1))/(np.mean(mag[indmag])-(Mco[i]-mbin/2))
#         a = np.log10(len(indmag))+b*Mco[i]
#         cumN = 10**(a-b*FMD['m'])
#         params[i, 0] = a
#         params[i, 1] = b
#         cumNtmp = 10**(a-b*(np.max(FMD['m'])+mbin))
#         cumNtmp = np.append(cumN, cumNtmp)
#         N = np.abs(np.diff(cumNtmp))
#         data = pd.DataFrame({'N': N, 'm': FMD['m'], 'Nd': FMD['noncum']})
#         indLow = np.where(FMD['m'] < Mco[i])[0]
#         indHigh = np.where(FMD['m'] >= Mco[i])[0]
#         dataTest = pd.DataFrame({'N': data['N'][indLow], 'm': data['m'][indLow], 'Nd': data['Nd'][indLow]})
#         dataTmp = pd.DataFrame({'N': data['N'][indHigh], 'm': data['m'][indHigh], 'Nd': data['Nd'][indHigh]})
#         checkNo0 = np.where(dataTest['Nd'] != 0)[0]
#         dataTest = pd.DataFrame({'N': dataTest['N'][checkNo0], 'm': dataTest['m'][checkNo0], 'Nd': dataTest['Nd'][checkNo0]})
#         Nmax = np.max(dataTest['Nd'])
#         Mmintmp = np.min(dataTest['m'])
#         dataTest['Nd'] = dataTest['Nd']/Nmax
#         dataTest['m'] = dataTest['m']-Mmintmp
#         data4fit = pd.DataFrame({'N': dataTest['Nd'], 'm': dataTest['m']})
#         def pnorm(m, mean, sd):
#             return np.exp(-(m-mean)**2/(2*sd**2))/(sd*np.sqrt(2*np.pi))
#         nlsfit, _ = curve_fit(pnorm, data4fit['m'], data4fit['N'], p0=[mu, sig])
#         params[i, 2] = nlsfit[0]
#         params[i, 3] = nlsfit[1]
#         dataTest['N'] = pnorm(dataTest['m'], nlsfit[0], nlsfit[1])*Nmax
#         dataTest['m'] = dataTest['m']+Mmintmp
#         dataTest['Nd'] = dataTest['Nd']*Nmax
#         dataPred = pd.concat([dataTest, dataTmp])
#         dataPred['N'] = np.round(dataPred['N'])
#         savedmodel[i, np.append(checkNo0, indHigh)] = dataPred['N']
#         ylab = "Cumulative Number"
#         plt.plot(FMD['m'], FMD['cum'], 'b-', label='Cum. FMD')
#         plt.plot(FMD['m'], FMD['noncum'], 'r.', label='Non Cum. FMD')
#         plt.plot(FMD['m'], cumN, 'g-', label='Model')
#         plt.xlabel('Magnitude')
#         plt.ylabel(ylab)
#         plt.legend()
#         plt.show()
        
#     from scipy.special import gammaln
#     probtmp = np.zeros(nbm)
#     CheckNo0 = np.where(dataPred['N'] != 0)[0]
#     Pmodel = dataPred['N'][CheckNo0]
#     Pdata = dataPred['Nd'][CheckNo0]
#     probtmp[CheckNo0] = 1/np.log(10)*(-Pmodel+Pdata*np.log(Pmodel)-gammaln(Pdata+1))
#     #probtmp[CheckNo0] = 1/np.log(10)*(-Pmodel+Pdata*np.log(Pmodel)-np.loggamma(Pdata+1))
#     prob = -np.sum(probtmp)
#     indbestfit = np.where(prob == np.min(prob))[0]
#     res = {'Mc': Mco[indbestfit], 'a': params[indbestfit, 0], 'b': params[indbestfit, 1], 'mu': params[indbestfit, 2], 'sigma': params[indbestfit, 3], 'model': savedmodel[indbestfit, :], 'Mco': Mco, 'prob': prob}
#     return res

#__________________________________________________________________________________________

# def power_law_cdf(magnitudes, Mc, b_value):
#     """Power-law CDF for theoretical distribution."""
#     return 1 - np.exp(-b_value * (magnitudes - Mc))

# def calculate_ks_distance(mag, Mc_candidate, b_value=1.0):
#     """Calculate KS-Distance for Mc candidate."""
#     mag_cut = mag[mag >= Mc_candidate]
#     if len(mag_cut) < 2:  # Ensure sufficient data
#         return np.inf
#     empirical_cdf = np.arange(1, len(mag_cut) + 1) / len(mag_cut)
#     theoretical_cdf = power_law_cdf(mag_cut, Mc_candidate, b_value)
#     return np.max(np.abs(empirical_cdf - theoretical_cdf))


# INPUT PARAMETERS
#wd = "C:/WIWIN/UNHAS/PSHA/"
wd1= "C:/WIWIN/UNHAS/PSHA/BuatPAPERDewi/segmenYopi/b-valueMap/Palukoro-Matano/"
cat_file ="(WardMethod)Cluster_25_2.5.xlsx" #jangan lupa diubah
# "(WardMethod)Cluster_24_2.5.xlsx",
# "(WardMethod)Cluster_25_2.5.xlsx",
#cat_file = "(WardMethod)Cluster_3_7.xlsx"  # Ganti dengan nama file Excel yang diinginkan
sheet_name = "Setelah"
#sheet_name = "Sheet1"
# READ CATALOG
cat = pd.read_excel(wd1 + cat_file, sheet_name=sheet_name)
#cat = cat[pd.to_numeric(cat['magnitude'], errors='coerce').notnull()]
mag = cat['magnitude']
mbin = 0.1  # Magnitude bin
nbsample = 200  # Bootstrapping


# COMPUTE Mc dengan teknik maxc
Mc_bootstrap = np.zeros(nbsample)

np.random.seed(42) #(43)

for i in range(nbsample):
    Mc_bootstrap[i] = maxc(np.random.choice(mag, replace=True, size=len(mag)), mbin)['Mc']


# for i in range(nbsample):
#     try:
#         Mc_bootstrap[i] = emr(np.random.choice(mag, replace=True), mbin)['Mc']
#     except:
#         pass
    
McMean_maxc = np.nanmean(Mc_bootstrap)
Mc_sd = np.nanstd(Mc_bootstrap)
print("Mc(mean)_maxc:", McMean_maxc)
print("Sigma0 (std. dev.):", Mc_sd)


# PLOT FMD
FMD = fmd(mag, mbin)
plt.figure()
plt.plot(FMD['m'], FMD['cum'], 'o', label='Cum. FMD')
plt.plot(FMD['m'], FMD['noncum'], '^', label='Non Cum. FMD')

plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Magnitude')
plt.ylabel('Number of events')
plt.legend()
plt.axvline(x=McMean_maxc, linestyle='-', label=f'McMean_maxc= {round(McMean_maxc, 3)},$\sigma$={round(Mc_sd, 3)}', color='red')


# plt.axvline(x=Mc_sd, linestyle='--', label='McMean_sd(maxc)', color='red')

# COMPUTE Mc dengan teknik gft
Mc_bootstrap1 = np.zeros(nbsample)
for i in range(nbsample):
    Mc_bootstrap1[i] = gft(np.random.choice(mag, replace=True, size=len(mag)), mbin)['Mc']

# for i in range(nbsample):
#     try:
#         Mc_bootstrap1[i] = emr(np.random.choice(mag, replace=True), mbin)['Mc']
#     except:
#         pass    
    
McMean_gft = np.nanmean(Mc_bootstrap1)
Mc_sd1 = np.nanstd(Mc_bootstrap1)
print("Mc (gft):", McMean_gft)
print("Sigma0 (std. dev.):", Mc_sd1)

# Tampilkan garis putus-putus vertikal dengan teknik gft
plt.axvline(x=McMean_gft, linestyle='-', label=f'McMean_gft= {round(McMean_gft, 3)},$\sigma$={round(Mc_sd1, 3)}', color='blue')
#plt.axvline(x=Mc_sd1, linestyle='--', label='McMean_sd(gft)', color='blue', dashes=(3,5))

# COMPUTE Mc dengan teknik mbs
Mc_bootstrap2 = np.zeros(nbsample)
for i in range(nbsample):
    Mc_bootstrap2[i] = mbs(np.random.choice(mag, replace=True, size=len(mag)), mbin)['Mc']

# for i in range(nbsample):
#     try:
#         Mc_bootstrap2[i] = emr(np.random.choice(mag, replace=True), mbin)['Mc']
#     except:
#         pass    
    
McMean_mbs = np.nanmean(Mc_bootstrap2)
Mc_sd2 = np.nanstd(Mc_bootstrap2)
print("McMean_mbs:", McMean_mbs)
print("Sigma0 (std. dev.):", Mc_sd2)


# Tampilkan garis putus-putus vertikal dengan teknik mbs
plt.axvline(x=McMean_mbs, linestyle='-', label=f'McMean_mbs= {round(McMean_mbs, 3)},$\sigma$={round(Mc_sd2, 3)}', color='green')
# plt.axvline(x=Mc_sd2, linestyle='--', label='McMean_sd(mbs)', color='green', dashes=(2,5))

# Fit Gutenberg-Richter law above McMean_maxc
def gutenberg_richter(M, a, b):
    return 10**(a - b*M)

# Data points for fitting GR (magnitudes greater than McMean_maxc)
# mask = FMD['m'] >= McMean_maxc
# M_fit = FMD['m'][mask]
# cum_fit = FMD['cum'][mask]

# Estimasi parameter a dan b menggunakan data di atas McMean_maxc
# popt, pcov = curve_fit(lambda M, a, b: np.log10(gutenberg_richter(M, a, b)), M_fit, np.log10(cum_fit))
# a_est, b_est = popt
# print(f"Estimated a: {a_est}, Estimated b: {b_est}")

# Plot Gutenberg-Richter line
# M_values = np.linspace(min(FMD['m']), max(FMD['m']), 100)
# GR_values = gutenberg_richter(M_values, a_est, b_est)
# plt.plot(M_values, GR_values, label=f'Gutenberg-Richter: a={round(a_est, 2)}, b={round(b_est, 2)}', color='black')


# Letakkan legenda di bawah plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Magnitude')
plt.ylabel('Number of events')
plt.title("Magnitude of completeness for cluster 25")
# plt.savefig(wd1 + f"{cat_file}.png")
plt.show()

#_________________ Kolmogorov-Smirnov Method ____________________________________________________
# KS-Distance Minimization Technique
# Mc_candidates = np.arange(np.min(mag), np.max(mag), 0.1)
# ks_distances = [calculate_ks_distance(mag, Mc) for Mc in Mc_candidates]

# # Use McMean_maxc as the optimal Mc
# Mc_optimal = McMean_maxc

# # Find the index of Mc_optimal in Mc_candidates
# Mc_optimal_idx = np.argmin(np.abs(Mc_candidates - Mc_optimal))

# # Get the corresponding KS-Distance for Mc_optimal
# ks = ks_distances[Mc_optimal_idx]

#=======================================================================================
# Use McMean_maxc as the optimal Mc
# Mc_optimal1 = McMean_gft

# # Find the index of Mc_optimal in Mc_candidates
# Mc_optimal_idx1 = np.argmin(np.abs(Mc_candidates - Mc_optimal1))

# # Get the corresponding KS-Distance for Mc_optimal
# ks1 = ks_distances[Mc_optimal_idx1]

#=========================================================================================
# Use McMean_maxc as the optimal Mc
# Mc_optimal2 = McMean_mbs

# # Find the index of Mc_optimal in Mc_candidates
# Mc_optimal_idx2 = np.argmin(np.abs(Mc_candidates - Mc_optimal2))

# # Get the corresponding KS-Distance for Mc_optimal
# ks2 = ks_distances[Mc_optimal_idx2]

# # Plot KS-Distance
# plt.figure(figsize=(8, 6))
# plt.plot(Mc_candidates, ks_distances, label="KS-Distance", color='gray')
# plt.axvline(x=Mc_optimal, color='red', linestyle='--', label=f'Mc = {Mc_optimal:.3f}')
# plt.axvline(x=Mc_optimal1, color='blue', linestyle='--', label=f'Mc = {Mc_optimal1:.3f}')
# plt.axvline(x=Mc_optimal2, color='green', linestyle='--', label=f'Mc = {Mc_optimal2:.3f}')
# plt.scatter(Mc_optimal, ks, color='red', label=f'KS-D(Maxc) = {ks:.3f}', zorder=2)
# plt.scatter(Mc_optimal1, ks1, color='blue', label=f'KS-D(GFT) = {ks1:.3f}', zorder=2)
# plt.scatter(Mc_optimal2, ks2, color='green', label=f'KS-D(MBS) = {ks2:.3f}', zorder=2)
# plt.xlabel("Magnitude (Mc)")
# plt.ylabel("KS-Distance")
# plt.title("Magnitude of Completeness (Mc) Vs KS-Distance Minimization")
# plt.legend()
# plt.grid()
# plt.show()

#__________________________________________________________________________________

#-----------------------------------------------------------------------------------------
# FMD_result = fmd(mag,mbin)
# cumnbmag= FMD_result['cum']
# mi=FMD_result['m']

# resultmaxc = maxc(mag, mbin)
# Mcmaxc = resultmaxc['Mc']
# # Hitung nilai menggunakan fungsi MBS
# result = mbs(mag, mbin)
# Mco1 = result['Mco']
# Mcmbs = result['Mc']
# bi = result['bi']
# unc = result['unc']
# bave = result['bave']
# indMBS= result['indMBS']
# indMBS_old= result['indMBS_old']
# dbi_old = result ['dbi_old']
# dbi = result ['dbi']

#tes=indMBS/bi


# Plotting dengan error bars
# plt.figure(figsize=(10, 6))
#plt.plot(Mco, dbi_old,linestyle='-', label= "b(est)")
# plt.plot(Mco, bi,linestyle='-', label= "b(est)")
# plt.plot(Mco, unc+Mco ,linestyle='-', label= "unc")
# plt.plot(Mco, unc-Mco ,linestyle='-')
# plt.plot(Mco, bave ,marker='o', linestyle='-', label= "bave")

#plt.errorbar(Mco, bi, yerr=unc, fmt='o', color='b', ecolor='r', capsize=5, label='b-value with error')
# plt.xlabel('Mco')
# plt.ylabel('b-value')
# plt.title('Variation of Mco vs b-value with Error Bars')
# plt.grid(True)
# plt.legend()
# #plt.xlim(3.45,4.5)
# plt.show()


# COMPUTE Mc dengan teknik gft
gft_result = gft(mag, mbin)
Mco2 = gft_result['Mco']
R = gft_result['R']
FMDcum_model = gft_result ['FMDcum_model']
Mcgft = gft_result ['Mc']

# PLOT Residual % R-Value vs Mc
# plt.figure(figsize=(10, 6))
#plt.plot(Mco, R, 'o-', color='b', label='Residual in % R-Value')

#plt.axhline(y=550, color='r', linestyle='--', label='90% Threshold')
#plt.axhline(y=600, color='g', linestyle='--', label='95% Threshold')
# plt.xlabel('Mc')
# plt.ylabel('Residual % R-Value')
# plt.legend()
# plt.title('Residual % R-Value vs Mc (GFT Technique)')
# plt.grid(True)
