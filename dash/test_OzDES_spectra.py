import os
import numpy as np
import dash

directoryPath = '/Users/dmuthukrishna/Documents/OzDES_data/ATEL_9742_Run26'

atel9742 = [
    ('DES16E1ciy_E1_combined_161101_v10_b00.dat', 0.174),
    ('DES16S1cps_S1_combined_161101_v10_b00.dat', 0.274),
    ('DES16E2crb_E2_combined_161102_v10_b00.dat', 0.229),
    ('DES16E2clk_E2_combined_161102_v10_b00.dat', 0.367),
    ('DES16E2cqq_E2_combined_161102_v10_b00.dat', 0.426),
    ('DES16X2ceg_X2_combined_161103_v10_b00.dat', 0.335),
    ('DES16X2bkr_X2_combined_161103_v10_b00.dat', 0.159),
    ('DES16X2crr_X2_combined_161103_v10_b00.dat', 0.312),
    ('DES16X2cpn_X2_combined_161103_v10_b00.dat', 0.28),
    ('DES16X2bvf_X2_combined_161103_v10_b00.dat', 0.135),
    ('DES16C1cbg_C1_combined_161103_v10_b00.dat', 0.111),
    ('DES16C2cbv_C2_combined_161103_v10_b00.dat', 0.109),
    ('DES16C1bnt_C1_combined_161103_v10_b00.dat', 0.351),
    ('DES16C3at_C3_combined_161031_v10_b00.dat', 0.217),
    ('DES16X3cpl_X3_combined_161031_v10_b00.dat', 0.205),
    ('DES16E2cjg_E2_combined_161102_v10_b00.dat', 0.48),
    ('DES16X2crt_X2_combined_161103_v10_b00.dat', 0.57)]

filenames = [os.path.join(directoryPath, i[0]) for i in atel9742]
knownRedshifts = [i[1] for i in atel9742]

classification = dash.Classify(filenames, knownRedshifts)
bestFits, bestTypes = classification.list_best_matches(n=1)
print(bestFits)
np.savetxt('Run26_fits.txt', bestFits, fmt='%s')
# classification.plot_with_gui(indexToPlot=1)

