import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rootDir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(rootDir)
fileList = []
for dir_, _, files in os.walk(rootDir):
    for fileName in files:
        relDir = os.path.relpath(dir_, rootDir)
        if relDir == '.':
            relFile = fileName
        else:
            relFile = os.path.join(relDir, fileName)
        print(relFile)
        fileList.append(relFile)

def snid_template_spectra_all(filename):
    """lnw file"""
    with open(filename, 'r') as FileObj:
        for lineNum, line in enumerate(FileObj):
            # Read Header Info
            if lineNum == 0:
                header = (line.strip('\n')).split(' ')
                header = [x for x in header if x != '']
                numAges, nwx, w0x, w1x, mostknots, tname, dta, ttype, ittype, itstype = header
                numAges, mostknots = map(int, (numAges, mostknots))
                nk = np.zeros(numAges)
                fmean = np.zeros(numAges)
                xk = np.zeros((mostknots, numAges))
                yk = np.zeros((mostknots, numAges))

            # Read Spline Info
            elif lineNum == 1:
                splineInfo = (line.strip('\n')).split(' ')
                splineInfo = [x for x in splineInfo if x != '']
                for j in range(numAges):
                    nk[j], fmean[j] = (splineInfo[2 * j + 1], splineInfo[2 * j + 2])
            elif lineNum in range(2, mostknots + 2):
                splineInfo = (line.strip('\n')).split(' ')
                splineInfo = [x for x in splineInfo if x != '']
                for j in range(numAges):
                    xk[lineNum - 2, j], yk[lineNum - 2, j] = (splineInfo[2 * j + 1], splineInfo[2 * j + 2])

            elif lineNum == mostknots + 2:
                break

    splineInfo = (nk, fmean, xk, yk)

    # Read Normalized spectra
    arr = np.loadtxt(filename, skiprows=mostknots + 2)
    ages = arr[0]
    ages = np.delete(ages, 0)
    arr = np.delete(arr, 0, 0)

    wave = arr[:, 0]
    fluxes = np.zeros(shape=(numAges, len(arr)))  # initialise 2D array

    for i in range(0, len(arr[0]) - 1):
        fluxes[i] = arr[:, i + 1]

    if ttype == 'Ia-99aa':
        ttype = 'Ia-91T'
    elif ttype == 'Ia-02cx':
        ttype = 'Iax'

    return wave, fluxes, numAges, ages, ttype, splineInfo


def read_dat_file(filename):
    data = pd.read_csv(filename, header=None, delim_whitespace=True).values
    wave = data[:, 0]
    flux = data[:, 1]

    return wave, flux


def read_superfit_template(filename):
    wave, flux = read_dat_file(filename)
    tType = os.path.split(os.path.split(filename)[0])[-1]  # Name of directory is the type name
    filename = os.path.basename(filename)
    snName, ageInfo = os.path.basename(filename).strip('.dat').split('.')
    if ageInfo == 'max':
        age = 0
    elif ageInfo[0] == 'm':
        age = -float(ageInfo[1:])
    elif ageInfo[0] == 'p':
        age = float(ageInfo[1:])
    else:
        raise Exception("Invalid Superfit file: {0}".format(filename))

    nCols = 1

    # plt.plot(wave, flux, label=filename)
    # plt.legend()
    # plt.show()

    return wave, [flux], nCols, [age], tType


def delete_files(fileList):
    for fname in fileList:
        if not fname.endswith('.lnw'):
            fname = "{0}.lnw".format(fname)
        if os.path.isfile(fname):
            os.remove(fname)
            print("Deleted: {0}".format(fname))


def main():
    # Delete Files from templates-2.0 and Liu & Modjaz
    NO_MAX_SNID_LIU_MODJAZ = ["sn1997X", "sn2001ai", "sn2001ej", "sn2001gd", "sn2001ig", "sn2002ji", "sn2004ao", "sn2004eu", "sn2004gk", "sn2005ar", "sn2005da", "sn2005kf", "sn2005nb", "sn2005U", "sn2006ck", "sn2006fo", "sn2006lc", "sn2006lv", "sn2006ld", "sn2007ce", "sn2007I", "sn2007rz", "sn2008an", "sn2008aq", "sn2008cw", "sn1988L", "sn1990K", "sn1990aa", "sn1991A", "sn1991N", "sn1991ar", "sn1995F", "sn1997cy", "sn1997dc", "sn1997dd", "sn1997dq", "sn1997ei", "sn1998T", "sn1999di", "sn1999dn", "sn2004dj"]
    BAD_SPECTRA = ['sn2010bh']
    delete_files(NO_MAX_SNID_LIU_MODJAZ)
    delete_files(BAD_SPECTRA)
    # Delete files from bsnip
    NO_MAX_BSNIP_AGE_999 = ['sn00ev_bsnip.lnw', 'sn00fe_bsnip.lnw', 'sn01ad_bsnip.lnw', 'sn01cm_bsnip.lnw', 'sn01cy_bsnip.lnw', 'sn01dk_bsnip.lnw', 'sn01do_bsnip.lnw', 'sn01ef_bsnip.lnw', 'sn01ey_bsnip.lnw', 'sn01gd_bsnip.lnw', 'sn01hg_bsnip.lnw', 'sn01ir_bsnip.lnw', 'sn01K_bsnip.lnw', 'sn01M_bsnip.lnw', 'sn01X_bsnip.lnw', 'sn02A_bsnip.lnw', 'sn02an_bsnip.lnw', 'sn02ap_bsnip.lnw', 'sn02bu_bsnip.lnw', 'sn02bx_bsnip.lnw', 'sn02ca_bsnip.lnw', 'sn02dq_bsnip.lnw', 'sn02eg_bsnip.lnw', 'sn02ei_bsnip.lnw', 'sn02eo_bsnip.lnw', 'sn02hk_bsnip.lnw', 'sn02hn_bsnip.lnw', 'sn02J_bsnip.lnw', 'sn02kg_bsnip.lnw', 'sn03ab_bsnip.lnw', 'sn03B_bsnip.lnw', 'sn03ei_bsnip.lnw', 'sn03G_bsnip.lnw', 'sn03gd_bsnip.lnw', 'sn03gg_bsnip.lnw', 'sn03gu_bsnip.lnw', 'sn03hl_bsnip.lnw', 'sn03ip_bsnip.lnw', 'sn03iq_bsnip.lnw', 'sn03kb_bsnip.lnw', 'sn04aq_bsnip.lnw', 'sn04bi_bsnip.lnw', 'sn04cz_bsnip.lnw', 'sn04dd_bsnip.lnw', 'sn04dj_bsnip.lnw', 'sn04du_bsnip.lnw', 'sn04et_bsnip.lnw', 'sn04eu_bsnip.lnw', 'sn04ez_bsnip.lnw', 'sn04fc_bsnip.lnw', 'sn04fx_bsnip.lnw', 'sn04gd_bsnip.lnw', 'sn04gr_bsnip.lnw', 'sn05ad_bsnip.lnw', 'sn05af_bsnip.lnw', 'sn05aq_bsnip.lnw', 'sn05ay_bsnip.lnw', 'sn05bi_bsnip.lnw', 'sn05bx_bsnip.lnw', 'sn05cs_bsnip.lnw', 'sn05ip_bsnip.lnw', 'sn05kd_bsnip.lnw', 'sn06ab_bsnip.lnw', 'sn06be_bsnip.lnw', 'sn06bp_bsnip.lnw', 'sn06by_bsnip.lnw', 'sn06ca_bsnip.lnw', 'sn06cx_bsnip.lnw', 'sn06gy_bsnip.lnw', 'sn06my_bsnip.lnw', 'sn06ov_bsnip.lnw', 'sn06T_bsnip.lnw', 'sn06tf_bsnip.lnw', 'sn07aa_bsnip.lnw', 'sn07ag_bsnip.lnw', 'sn07av_bsnip.lnw', 'sn07ay_bsnip.lnw', 'sn07bb_bsnip.lnw', 'sn07be_bsnip.lnw', 'sn07C_bsnip.lnw', 'sn07ck_bsnip.lnw', 'sn07cl_bsnip.lnw', 'sn07K_bsnip.lnw', 'sn07oc_bsnip.lnw', 'sn07od_bsnip.lnw', 'sn08aq_bsnip.lnw', 'sn08aw_bsnip.lnw', 'sn08be_bsnip.lnw', 'sn08bj_bsnip.lnw', 'sn08bl_bsnip.lnw', 'sn08D_bsnip.lnw', 'sn08es_bsnip.lnw', 'sn08fq_bsnip.lnw', 'sn08gf_bsnip.lnw', 'sn08gj_bsnip.lnw', 'sn08ht_bsnip.lnw', 'sn08in_bsnip.lnw', 'sn08iy_bsnip.lnw', 'sn88Z_bsnip.lnw', 'sn90H_bsnip.lnw', 'sn90Q_bsnip.lnw', 'sn91ao_bsnip.lnw', 'sn91av_bsnip.lnw', 'sn91C_bsnip.lnw', 'sn92ad_bsnip.lnw', 'sn92H_bsnip.lnw', 'sn93ad_bsnip.lnw', 'sn93E_bsnip.lnw', 'sn93G_bsnip.lnw', 'sn93J_bsnip.lnw', 'sn93W_bsnip.lnw', 'sn94ak_bsnip.lnw', 'sn94I_bsnip.lnw', 'sn94W_bsnip.lnw', 'sn94Y_bsnip.lnw', 'sn95G_bsnip.lnw', 'sn95J_bsnip.lnw', 'sn95V_bsnip.lnw', 'sn95X_bsnip.lnw', 'sn96ae_bsnip.lnw', 'sn96an_bsnip.lnw', 'sn96cc_bsnip.lnw', 'sn97ab_bsnip.lnw', 'sn97da_bsnip.lnw', 'sn97dd_bsnip.lnw', 'sn97ef_bsnip.lnw', 'sn97eg_bsnip.lnw', 'sn98A_bsnip.lnw', 'sn98dl_bsnip.lnw', 'sn98dt_bsnip.lnw', 'sn98E_bsnip.lnw', 'sn98S_bsnip.lnw', 'sn99eb_bsnip.lnw', 'sn99ed_bsnip.lnw', 'sn99el_bsnip.lnw', 'sn99em_bsnip.lnw', 'sn99gb_bsnip.lnw', 'sn99gi_bsnip.lnw', 'sn99Z_bsnip.lnw']
    SAME_SN_WITH_SAME_AGES_AS_SNID = ['sn02ic.lnw', 'sn04dj.lnw', 'sn05gj.lnw', 'sn05hk.lnw', 'sn96L.lnw', 'sn99ex.lnw', 'sn02ap.lnw', 'sn02bo.lnw', 'sn04aw_bsnip.lnw', 'sn04et.lnw', 'sn05cs.lnw', 'sn90N.lnw', 'sn92A.lnw', 'sn93J.lnw', 'sn97br.lnw', 'sn97ef.lnw', 'sn98S.lnw', 'sn99aa.lnw', 'sn99em.lnw']
    delete_files(NO_MAX_BSNIP_AGE_999)
    delete_files(SAME_SN_WITH_SAME_AGES_AS_SNID)
    print(len(SAME_SN_WITH_SAME_AGES_AS_SNID), len(NO_MAX_BSNIP_AGE_999))

    minWaves, maxWaves = [], []
    countBelow3000 = 0
    snTypes = {'Ia':[], 'Ib':[], 'Ic':[], 'II':[], 'SLSN':[]}
    for fname in fileList:
        if fname.endswith('.lnw'):
            continue # wave, fluxes, numAges, ages, ttype, splineInfo = snid_template_spectra_all(fname)
        elif fname.endswith('.dat'):
            wave, fluxes, nCols, ages, ttype = read_superfit_template(fname)
        else:
            continue
        # print(ttype, fname)
        for broadType in snTypes.keys():
            if broadType in ttype and 'IIb' not in ttype:
                snTypes[broadType].append((ttype, fname))
            elif broadType == 'Ib' and 'IIb' in ttype:
                snTypes['Ib'].append((ttype, fname))
        for ageIdx in range(len(ages)):
            flux = fluxes[ageIdx]
            nonZeros = np.where(flux != 0)[0]
            minIndex, maxIndex = min(nonZeros), max(nonZeros)
            minWaves.append(wave[minIndex])
            if wave[minIndex] < 2900:
                countBelow3000 += 1
                print("Min Below 3000A: ", ttype, fname, ages[ageIdx], wave[minIndex])
            maxWaves.append(wave[maxIndex])
        
    import pprint
    pprint.pprint(snTypes)
    print([("{0}: {1}".format(broadType, len(snTypes[broadType]))) for broadType in snTypes.keys()])

            # # Check if age is -999
            # if -999 in ages:
            #     noMax.append(fname)

    minWaves, maxWaves = np.array(minWaves), np.array(maxWaves)
    print(minWaves)
    print(maxWaves)
    print("MinWaves: Min:{0}, Max:{1}, Mean:{2}, Median:{3}, Std:{4}".format(min(minWaves), max(minWaves), np.mean(minWaves), np.median(minWaves), np.std(minWaves)))
    print("MaxWaves: Min:{0}, Max:{1}, Mean:{2}, Median:{3}, Std:{4}".format(min(maxWaves), max(maxWaves), np.mean(maxWaves), np.median(maxWaves), np.std(maxWaves)))
    print(countBelow3000, len(minWaves))
    plt.figure("MinWaves")
    plt.hist(minWaves, bins=75)
    plt.figure("MaxWaves")
    plt.hist(maxWaves, bins=75)
    plt.show()


if __name__ == '__main__':
    main()



    # # Create Template list
    # f = open('templist.txt', 'w')
    # for fname in files:
    #     if fname.endswith('.lnw'):
    #         f.write(fname)
    #         f.write('\n')
    # f.close()




    # # Compare templates-2.0/liumodjaz with bsnip
    # bsnipDir = '/Users/danmuth/PycharmProjects/DASH/templates/bsnip_v7_snid_templates/'
    # bsnipFiles = os.listdir(bsnipDir)
    # snidModjazLiuDir = '/Users/danmuth/PycharmProjects/DASH/templates/training_set/'
    # snidModjazLiuFiles = os.listdir(snidModjazLiuDir)

    # common, new = [], []
    # identicalFiles = []
    # for bsnipFile in bsnipFiles:
    #     if bsnipFile[0:2] == 'sn':
    #         name = bsnipFile[2:6].strip('_')
    #     else:
    #         continue

    #     commonName = False
    #     wave, fluxes, numAges, ages, ttype, splineInfo = snid_template_spectra_all(bsnipDir+bsnipFile)
    #     for snidFile in snidModjazLiuFiles:
    #         if name in snidFile:
    #             print(name, bsnipFile, snidFile)
    #             common.append((ttype, name))
    #             commonName = True
    #             wave2, fluxes2, numAges2, ages2, ttype2, splineInfo2 = snid_template_spectra_all(snidModjazLiuDir+snidFile)
    #             for age in ages:
    #                 if age in ages2:
    #                     identicalFiles.append((ttype, bsnipFile, snidFile, len(ages), len(ages2), np.intersect1d(ages, ages2))) # checks if the files have just some common ages
    #                     break
    #     if not commonName:
    #         new.append((ttype, name))

    # print(len(common))
    # pprint.pprint(common)
    # print(len(new))
    # pprint.pprint(new)
    # print("IdenticalFiles")
    # pprint.pprint(identicalFiles)
