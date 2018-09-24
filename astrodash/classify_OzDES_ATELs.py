import os
import astrodash

directoryPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../templates/OzDES_data/')

atels = [
    ('ATEL_9504_Run24/DES16E1de_E1_combined_160825_v10_b00.dat', 0.292),
    ('ATEL_9504_Run24/DES16E2dd_E2_combined_160826_v10_b00.dat', 0.0746),
    ('ATEL_9504_Run24/DES16X3km_X3_combined_160827_v10_b00.dat', 0.06),
    ('ATEL_9504_Run24/DES16X3er_X3_combined_160827_v10_b00.dat', 0.167),
    ('ATEL_9504_Run24/DES16X3hj_X3_combined_160827_v10_b00.dat', 0.308),
    ('ATEL_9504_Run24/DES16X3es_X3_combined_160827_v10_b00.dat', 0.554),
    ('ATEL_9504_Run24/DES16X3jj_X3_combined_160827_v10_b00.dat', 0.238),
    ('ATEL_9504_Run24/DES16C3fv_C3_combined_160829_v10_b00.dat', 0.322),
    ('ATEL_9504_Run24/DES16C3bq_C3_combined_160829_v10_b00.dat', 0.241),
    ('ATEL_9504_Run24/DES16E1md_E1_combined_160829_v10_b00.dat', 0.178),
    ('ATEL_9504_Run24/DES16E1ah_E1_combined_160829_v10_b00.dat', 0.149),
    ('ATEL_9504_Run24/DES16C3ea_C3_combined_160829_v10_b00.dat', 0.217),
    ('ATEL_9504_Run24/DES16X1ey_X1_combined_160829_v10_b00.dat', 0.076),

    ('ATEL_9570_Run25/DES16C3bq_C3_combined_160925_v10_b00.dat', 0.237),
    ('ATEL_9570_Run25/DES16E2aoh_E2_combined_160925_v10_b00.dat', 0.403),
    ('ATEL_9570_Run25/DES16X3aqd_X3_combined_160925_v10_b00.dat', 0.033),
    ('ATEL_9570_Run25/DES16X3biz_X3_combined_160925_v10_b00.dat', 0.24),
    ('ATEL_9570_Run25/DES16C2aiy_C2_combined_160926_v10_b00.dat', 0.182),
    ('ATEL_9570_Run25/DES16C2ma_C2_combined_160926_v10_b00.dat', 0.24),
    ('ATEL_9570_Run25/DES16X1ge_X1_combined_160926_v10_b00.dat', 0.25),
    ('ATEL_9570_Run25/DES16X2auj_X2_combined_160927_v10_b00.dat', 0.144),
    ('ATEL_9570_Run25/DES16E2bkg_E2_combined_161005_v10_b00.dat', 0.478),
    ('ATEL_9570_Run25/DES16E2bht_E2_combined_161005_v10_b00.dat', 0.392),

    ('ATEL_9742_Run26/DES16E1ciy_E1_combined_161101_v10_b00.dat', 0.174),
    ('ATEL_9742_Run26/DES16S1cps_S1_combined_161101_v10_b00.dat', 0.274),
    ('ATEL_9742_Run26/DES16E2crb_E2_combined_161102_v10_b00.dat', 0.229),
    ('ATEL_9742_Run26/DES16E2clk_E2_combined_161102_v10_b00.dat', 0.367),
    ('ATEL_9742_Run26/DES16E2cqq_E2_combined_161102_v10_b00.dat', 0.426),
    ('ATEL_9742_Run26/DES16X2ceg_X2_combined_161103_v10_b00.dat', 0.335),
    ('ATEL_9742_Run26/DES16X2bkr_X2_combined_161103_v10_b00.dat', 0.159),
    ('ATEL_9742_Run26/DES16X2crr_X2_combined_161103_v10_b00.dat', 0.312),
    ('ATEL_9742_Run26/DES16X2cpn_X2_combined_161103_v10_b00.dat', 0.28),
    ('ATEL_9742_Run26/DES16X2bvf_X2_combined_161103_v10_b00.dat', 0.135),
    ('ATEL_9742_Run26/DES16C1cbg_C1_combined_161103_v10_b00.dat', 0.111),
    ('ATEL_9742_Run26/DES16C2cbv_C2_combined_161103_v10_b00.dat', 0.109),
    ('ATEL_9742_Run26/DES16C1bnt_C1_combined_161103_v10_b00.dat', 0.351),
    ('ATEL_9742_Run26/DES16C3at_C3_combined_161031_v10_b00.dat', 0.217),
    ('ATEL_9742_Run26/DES16X3cpl_X3_combined_161031_v10_b00.dat', 0.205),
    ('ATEL_9742_Run26/DES16E2cjg_E2_combined_161102_v10_b00.dat', 0.48),
    ('ATEL_9742_Run26/DES16X2crt_X2_combined_161103_v10_b00.dat', 0.57),

    ('ATEL_9855_Run27/DES16E1dcx_E1_combined_161125_v10_b00.dat', 0.453),
    ('ATEL_9855_Run27/DES16E1dcx_E2_combined_161126_v10_b00.dat', 0.453),
    ('ATEL_9855_Run27/DES16E1dic_E1_combined_161125_v10_b00.dat', 0.207),
    ('ATEL_9855_Run27/DES16X3dfk_X3_combined_161125_v10_b00.dat', 0.1495),
    ('ATEL_9855_Run27/DES16C3dhv_C3_combined_161125_v10_b00.dat', 0.300),
    ('ATEL_9855_Run27/DES16E2cxw_E2_combined_161126_v10_b00.dat', 0.293),
    ('ATEL_9855_Run27/DES16E2drd_E2_combined_161126_v10_b00.dat', 0.270),
    ('ATEL_9855_Run27/DES16X1drk_X1_combined_161127_v10_b00.dat', 0.463),
    ('ATEL_9855_Run27/DES16X1dbw_X1_combined_161127_v10_b00.dat', 0.336),
    ('ATEL_9855_Run27/DES16S2ean_S2_combined_161127_v10_b00.dat', 0.161),
    ('ATEL_9855_Run27/DES16S2dfm_S2_combined_161127_v10_b00.dat', 0.30),
    ('ATEL_9855_Run27/DES16X1dbx_X1_combined_161127_v10_b00.dat', 0.345),
    ('ATEL_9855_Run27/DES16E1eae_E1_combined_161129_v10_b00.dat', 0.534),
    ('ATEL_9855_Run27/DES16E1eef_E1_combined_161129_v10_b00.dat', 0.32),
    ('ATEL_9855_Run27/DES16S2drt_S2_combined_161127_v10_b00.dat', 0.331),
    ('ATEL_9855_Run27/DES16X1der_X1_combined_161127_v10_b00.dat', 0.453),
    ('ATEL_9855_Run27/DES16C3dhy_C3_combined_161128_v10_b00.dat', 0.276),
    ('ATEL_9855_Run27/DES16X2dqz_X2_combined_161128_v10_b00.dat', 0.204),

    ('ATEL_9961_Run28/DES16C3elb_C3_combined_161225_v10_b00.dat', 0.429),
    ('ATEL_9961_Run28/DES16X3dvb_X3_combined_161225_v10_b00.dat', 0.329),
    ('ATEL_9961_Run28/DES16C2ege_C2_combined_161225_v10_b00.dat', 0.348),
    ('ATEL_9961_Run28/DES16X3eww_X3_combined_161225_v10_b00.dat', 0.445),
    ('ATEL_9961_Run28/DES16X3enk_X3_combined_161225_v10_b00.dat', 0.331),
    ('ATEL_9961_Run28/DES16S1ffb_S1_combined_161226_v10_b00.dat', 0.164),
    ('ATEL_9961_Run28/DES16C1fgm_C1_combined_161226_v10_b00.dat', 0.361),
    ('ATEL_9961_Run28/DES16X2dzz_X2_combined_161226_v10_b00.dat', 0.325),
    ('ATEL_9961_Run28/DES16X1few_X1_combined_161227_v10_b00.dat', 0.311),
    ('ATEL_9961_Run28/DES16X1chc_X1_combined_161227_v10_b00.dat', 0.043),
    ('ATEL_9961_Run28/DES16S2ffk_S2_combined_161227_v10_b00.dat', 0.373)]

filenames = [os.path.join(directoryPath, i[0]) for i in atels]
knownRedshifts = [i[1] for i in atels]

classification = astrodash.Classify(filenames, knownRedshifts, classifyHost=False, smooth=5, knownZ=False)
bestFits, redshifts, bestTypes, rejectionLabels, reliableFlags, redshiftErrs = classification.list_best_matches(n=5)

# SAVE BEST MATCHES
print(bestFits)
f = open('classification_results.txt', 'w')
for i in range(len(filenames)):
    f.write("%s   z=%s     %s      %s     %s\n %s\n\n" % (filenames[i].strip(directoryPath), redshifts[i], bestTypes[i], reliableFlags[i], rejectionLabels[i], bestFits[i]))
f.close()
print("Finished classifying %d spectra!" % len(filenames))

# PLOT SPECTRUM ON GUI
classification.plot_with_gui(indexToPlot=18)

