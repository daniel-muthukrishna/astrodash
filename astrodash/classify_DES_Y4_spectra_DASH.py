import os
import astrodash

directory = r'/Users/danmuth/Documents/DES_Y4_Spectra/'

# # LIST ALL SPECTRAL FILENAMES FROM DROPBOX
# filenames = []
# extensions = ('.flm', '.dat', '.txt', '.fits')
# for path, subdirs, files in os.walk(directory):
#     for name in files:
#         if name.lower().endswith(extensions):
#             filenames.append(os.path.join(path, name).replace(directory, ''))
# filenames.sort()
# for f in filenames:
#     print("('%s', 0)," % f)

atels = [
    ('Gemini/des16c1dgn-20161127-gmos.flm', 0.58),
    ('Gemini/des16c1fgm-20161230-gmos.flm', 0.363),
    ('Gemini/des16c2cva-20161109-gmos.flm', 0.404),
    # ('Gemini/des16c2dil-20161130-gmos.flm', 0.),
    ('Gemini/des16c3esw-20161229-gmos.flm', 0.67),
    ('Gemini/des16e1dcx-20161127-gmos.flm', 0.45),
    ('Gemini/des16e2cqq-20161105-gmos.flm', 0.426),
    ('Gemini/des16s1byw-20161012-gmos.flm', 0.358),
    ('Gemini/des16s2drt-20161128-gmos.flm', 0.331),
    ('Gemini/des16s2fgg-20170104-gmos.flm', 0.613),
    ('Gemini/des16x1cds-20161012-gmos.flm', 0.294),
    ('Gemini/des16x1cdu-20161013-gmos.flm', 0.289),
    ('Gemini/des16x1cpf-20161029-gmos.flm', 0.436),
    ('Gemini/des16x1der-20161128-gmos.flm', 0.453),
    ('Gemini/des16x1ewu-20161222-gmos.flm', 0.8),
    ('Gemini/des16x2cpp-20161029-gmos.flm', 0.518),
    ('Gemini/des16x2epa-20161222-gmos.flm', 0.650),
    ('Gemini/des16x3cry-20161105-gmos.flm', 0.612),
    ('Gemini/des16x3dlw-20161129-gmos.flm', 0.71),
    ('Gemini/des16x3dvb-20161223-gmos.flm', 0.329),
    # ('Keck/DES16C2nm.ms.dat', 0.),
    # ('Keck/lrisDES16C3eco.spec.txt', 0.),
    # ('Keck/lrisDES16S1ahw.spec.txt', 0.),
    ('Keck/lrisDES16S2ahx.spec.txt', 0.717),
    # ('Keck/lrisDES16S2aif.spec.txt', 0.),
    ('Keck/lrisDES16S2aif_obs2.spec.txt', 1.473),
    ('Keck/lrisDES16X1kx.spec.txt', 0.546),
    ('Keck/lrisDES16X3bji.spec.txt', 0.65),
    ('MMT/DES16S2cpe-20161102.dat', 0.384),
    ('MMT/DES16X1cph-20161102.dat', 0.49),
    ('MMT/DES16X2crr-20161102.dat', 0.50),
    ('MMT/DES16X2crt-20161102.dat', 0.567),
    # ('MMT/DES16X2csj-20161102.dat', 0.),
    # ('MMT/DES16X3crw-20161102.dat', 0.),
    ('Magellan_Chile/DES16C1cbg.txt', 0.33),
    ('Magellan_Chile/DES16C1mm.txt', 0.367),
    ('Magellan_Chile/DES16C1mm_2.txt', 0.367),
    # ('Magellan_Chile/DES16C3at.txt', 0.),
    ('Magellan_Chile/DES16E1bkp.txt', 0.541),
    ('Magellan_Chile/DES16E1bkp_2.txt', 0.541),
    ('Magellan_Chile/DES16E1bkp_3.txt', 0.541),
    ('Magellan_Chile/DES16S1bzz.txt', 0.556),
    ('Magellan_Chile/DES16S1bzz_2.txt', 0.556),
    ('Magellan_Chile/DES16S1bzz_3.txt', 0.556),
    ('Magellan_Chile/DES16S1bzz_4.txt', 0.556),
    ('Magellan_Chile/DES16S1eq.txt', 0.16),
    ('Magellan_Chile/DES16X2uj.txt', 0.322),
    ('Magellan_Chile/DES16X2uj_2.txt', 0.322),
    ('Magellan_Chile/DES16X2uj_s.txt', 0.322),
    ('Magellan_Chile/DES16X3bkz.txt', 0.07),
    ('Magellan_Chile/DES16X3bkz_2.txt', 0.07),
    ('Magellan_Chile/DES16X3gc.txt', 0.17),
    ('Magellan_RAISINS2/DES16C3boj-20161005.dat', 0.442),
    ('Magellan_RAISINS2/DES16E2bjz-20161005.dat', 0.278),
    ('Magellan_RAISINS2/DES16E2bkb-20161004.dat', 0.396), #0.41
    ('Magellan_RAISINS2/DES16S1bno-20161005.dat', 0.46),
    # ('Magellan_RAISINS2/DES16X1bky-20161004.dat', 0.),
    # ('Magellan_RAISINS2/DES16X1buy-20161005.dat', 0.),
    ('Magellan_RAISINS2/DES16X2bkq-20161005.dat', 0.334),
    # ('Magellan_UoC/DES16C1bntSpectrum.fits', 0.),
    # ('Magellan_UoC/DES16C2bnvSpectrum.fits', 0.),
    # ('Magellan_UoC/DES16C3benSpectrum.fits', 0.),
    # ('Magellan_UoC/DES16E1bssSpectrum.fits', 0.),
    # ('Magellan_UoC/DES16E1byySpectrum.fits', 0.),
    # ('Magellan_UoC/DES16E2bhtSpectrum.fits', 0.),
    # ('Magellan_UoC/DES16E2bllSpectrum.fits', 0.),
    # ('Magellan_UoC/DES16E2blzSpectrum.fits', 0.),
    # ('Magellan_UoC/DES16S1bnjSpectrum.fits', 0.),
    # ('Magellan_UoC/DES16S1bysSpectrum.fits', 0.),
    # ('VLT/DES16C1ccb_all.txt', 0.),
    ('VLT/DES16C2aiy_all.txt', 0.18),
    # ('VLT/DES16C2ayx_2016nov21_all.txt', 0.),
    # ('VLT/DES16C2ayx_2016sep24_all.txt', 0.),
    # ('VLT/DES16C2cbi_all.txt', 0.),
    # ('VLT/DES16C2dcv_all.txt', 0.),
    # ('VLT/DES16C2nm_2016nov21_all.txt', 0.),
    # ('VLT/DES16C2nm_2016oct23_all.txt', 0.),
    ('VLT/DES16C3byv_all.txt', 0.8),
    ('VLT/DES16C3bzu_all.txt', 0.62),
    ('VLT/DES16C3cv_all.txt', 0.727),
    ('VLT/DES16C3ecv_all.txt', 0.244),
    # ('VLT/DES16C3edf_all.txt', 0.),
    # ('VLT/DES16E1bss_all.txt', 0.),
    ('VLT/DES16E1ciy_all.txt', 0.173),
    ('VLT/DES16E1dcj_all.txt', 0.49),
    ('VLT/DES16E1emk_all.txt', 0.44),
    ('VLT/DES16E1oq_all.txt', 0.55),
    ('VLT/DES16E2bll_all.txt', 0.63),
    ('VLT/DES16E2cjg_all.txt', 0.48),
    ('VLT/DES16E2dci_all.txt', 0.601),
    ('VLT/DES16E2dck_all.txt', 0.34),
    # ('VLT/DES16E2of_all.txt', 0.),
    ('VLT/DES16E2yo_all.txt', 0.321),
    ('VLT/DES16E2zk_all.txt', 0.80),
    # ('VLT/DES16S1afy_all.txt', 0.),
    ('VLT/DES16S1cnk_all.txt', 0.8),
    ('VLT/DES16S2bdu_all.txt', 0.69),
    # ('VLT/DES16S2bpf_all.txt', 0.),
    ('VLT/DES16S2dgb_all.txt', 0.57),
    ('VLT/DES16S2ejs_all.txt', 0.38),
    # ('VLT/DES16X1ept_all.txt', 0.),
    ('VLT/DES16X2aey_all.txt', 0.210),
    # ('VLT/DES16X2afc_all.txt', 0.),
    ('VLT/DES16X3bdj_all.txt', 0.200),
    ('VLT/DES16X3bfv_all.txt', 0.60),
    ('VLT/DES16X3cdv_all.txt', 0.824),
    ('VLT/DES16X3cer_all.txt', 1.16),
    ('VLT/DES16X3cxy_all.txt', 0.688), #0.50
    # ('VLT/DES16X3eyo_all.txt', 0.)
    ]


filenames = [os.path.join(directory, i[0]) for i in atels]
knownRedshifts = [i[1] for i in atels]

classification = astrodash.Classify(filenames, knownRedshifts, classifyHost=False, rlapScores=True)
bestFits, redshifts, bestTypes, rejectionLabels, reliableFlags = classification.list_best_matches(n=5)

# SAVE BEST MATCHES
print(bestFits)
print("Finished classifying %d spectra!" % len(atels))

# PLOT SPECTRUM ON GUI
classification.plot_with_gui(indexToPlot=1)

