import dash

filenames = []
filenames.append('DES16C3elb_C3_combined_161227_v10_b00.dat')
filenames.append('DES16X3dvb_X3_combined_161225_v10_b00.dat')
filenames.append('DES16C2ege_C2_combined_161225_v10_b00.dat')
filenames.append('DES16X3eww_X3_combined_161225_v10_b00.dat')
filenames.append('DES16X3enk_X3_combined_161225_v10_b00.dat')
filenames.append('DES16S1ffb_S1_combined_161226_v10_b00.dat')
filenames.append('DES16C1fgm_C1_combined_161226_v10_b00.dat')
filenames.append('DES16X2dzz_X2_combined_161226_v10_b00.dat')
filenames.append('DES16X1few_X1_combined_161227_v10_b00.dat')
filenames.append('DES16X1chc_X1_combined_161227_v10_b00.dat')
filenames.append('DES16S2ffk_S2_combined_161227_v10_b00.dat')


knownRedshifts = []
knownRedshifts.append(0.429)
knownRedshifts.append(0.329)
knownRedshifts.append(0.348)
knownRedshifts.append(0.445)
knownRedshifts.append(0.331)
knownRedshifts.append(0.164)
knownRedshifts.append(0.361)
knownRedshifts.append(0.325)
knownRedshifts.append(0.311)
knownRedshifts.append(0.043)
knownRedshifts.append(0.373)

classification = dash.Classify(filenames, knownRedshifts)
print classification.list_best_matches(n=3)
classification.plot_with_gui(indexToPlot=1)