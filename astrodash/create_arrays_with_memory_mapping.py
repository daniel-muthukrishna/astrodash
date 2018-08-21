import os
import glob
import numpy as np
from random import shuffle
import multiprocessing as mp
import itertools

from astrodash.helpers import temp_list, div0
from astrodash.sn_processing import PreProcessing
from astrodash.combine_sn_and_host import training_template_data
from astrodash.preprocessing import ProcessingTools
from astrodash.array_tools import zero_non_overlap_part, normalise_spectrum

try:
    from imblearn import over_sampling
    IMBLEARN_EXISTS = True
except ImportError:
    print("Need to install imblearn `pip install imblearn` to use smote.")
    IMBLEARN_EXISTS = False


class AgeBinning(object):
    def __init__(self, minAge, maxAge, ageBinSize):
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize

    def age_bin(self, age):
        ageBin = int(round(age / self.ageBinSize)) - int(round(self.minAge / self.ageBinSize))

        return ageBin

    def age_labels(self):
        ageLabels = []

        ageBinPrev = 0
        ageLabelMin = self.minAge
        for age in np.arange(self.minAge, self.maxAge, 0.5):
            ageBin = self.age_bin(age)

            if ageBin != ageBinPrev:
                ageLabelMax = int(round(age))
                ageLabels.append(str(int(ageLabelMin)) + " to " + str(ageLabelMax))
                ageLabelMin = ageLabelMax

            ageBinPrev = ageBin

        ageLabels.append(str(int(ageLabelMin)) + " to " + str(int(self.maxAge)))

        return ageLabels


class CreateLabels(object):

    def __init__(self, nTypes, minAge, maxAge, ageBinSize, typeList, hostList, nHostTypes):
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge-0.1) + 1
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.ageLabels = self.ageBinning.age_labels()
        self.hostList = hostList
        self.nHostTypes = nHostTypes

    def label_array(self, ttype, age, host=None):
        ageBin = self.ageBinning.age_bin(age)

        try:
            typeIndex = self.typeList.index(ttype)
        except ValueError as err:
            raise Exception("INVALID TYPE: {0}".format(err))

        if host is None:
            labelArray = np.zeros((self.nTypes, self.numOfAgeBins))
            labelArray[typeIndex][ageBin] = 1
            labelArray = labelArray.flatten()
            typeName = ttype + ": " + self.ageLabels[ageBin]
        else:
            hostIndex = self.hostList.index(host)
            labelArray = np.zeros((self.nHostTypes, self.nTypes, self.numOfAgeBins))
            labelArray[hostIndex][typeIndex][ageBin] = 1
            labelArray = labelArray.flatten()
            typeName = "{}: {}: {}".format(host, ttype, self.ageLabels[ageBin])

        labelIndex = np.argmax(labelArray)

        return labelIndex, typeName

    def type_names_list(self):
        typeNamesList = []
        if self.hostList is None:
            for tType in self.typeList:
                for ageLabel in self.ageBinning.age_labels():
                    typeNamesList.append("{}: {}".format(tType, ageLabel))
        else:
            for host in self.hostList:
                for tType in self.typeList:
                    for ageLabel in self.ageBinning.age_labels():
                        typeNamesList.append("{}: {}: {}".format(host, tType, ageLabel))

        return np.array(typeNamesList)


class ReadSpectra(object):

    def __init__(self, w0, w1, nw, snFilename, galFilename=None):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.snFilename = snFilename
        if galFilename is None:
            self.data = PreProcessing(snFilename, w0, w1, nw)
        else:
            self.galFilename = galFilename

    def sn_plus_gal_template(self, snAgeIdx, snCoeff, galCoeff, z):
        wave, flux, minIndex, maxIndex, nCols, ages, tType = training_template_data(snAgeIdx, snCoeff, galCoeff, z, self.snFilename, self.galFilename, self.w0, self.w1, self.nw)

        return wave, flux, nCols, ages, tType, minIndex, maxIndex

    def input_spectrum(self, z, smooth, minWave, maxWave):
        wave, flux, minIndex, maxIndex, z = self.data.two_column_data(z, smooth, minWave, maxWave)

        return wave, flux, int(minIndex), int(maxIndex), z


class ArrayTools(object):

    def __init__(self, nLabels, nw):
        self.nLabels = nLabels
        self.nw = nw

    def shuffle_arrays(self, memmapName='', **kwargs):
        """ Must take images and labels as arguments with the keyword specified.
        Can optionally take filenames and typeNames as arguments """
        arraySize = len(kwargs['labels'])
        kwargShuf = {}
        self.randnum = np.random.randint(10000)
        for key in kwargs:
            if key == 'images':
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=np.float16, mode='w+', shape=(arraySize, int(self.nw)))
            elif key == 'labels':
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=np.uint16, mode='w+', shape=arraySize)
            else:
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=object, mode='w+', shape=arraySize)
            kwargShuf[key] = arrayShuf

        print("Shuffling...")
        # Randomise order
        p = np.random.permutation(len(kwargs['labels']))
        for key in kwargs:
            assert len(kwargs[key]) == arraySize
            print(key, "shuffling...")
            print(len(p))
            kwargShuf[key] = kwargs[key][p]

        return kwargShuf

    def count_labels(self, labels):
        counts = np.zeros(self.nLabels)

        for i in range(len(labels)):
            counts[labels[i]] += 1

        return counts

    def augment_data(self, flux, stdDevMean=0.05, stdDevStdDev=0.05):
        minIndex, maxIndex = ProcessingTools().min_max_index(flux, outerVal=0.5)
        noise = np.zeros(self.nw)
        stdDev = abs(np.random.normal(stdDevMean, stdDevStdDev)) # randomised standard deviation
        noise[minIndex:maxIndex] = np.random.normal(0, stdDev, maxIndex - minIndex)
        # # Add white noise to regions outside minIndex to maxIndex
        # noise[0:minIndex] = np.random.uniform(0.0, 1.0, minIndex)
        # noise[maxIndex:] = np.random.uniform(0.0, 1.0, self.nw-maxIndex)

        augmentedFlux = flux + noise
        augmentedFlux = normalise_spectrum(augmentedFlux)
        augmentedFlux = zero_non_overlap_part(augmentedFlux, minIndex, maxIndex, outerVal=0.5)

        return augmentedFlux


class OverSampling(ArrayTools):
    def __init__(self, nLabels, nw, **kwargs):
        """ Must take images and labels as arguments with the keyword specified.
        Can optionally take filenames and typeNames as arguments """
        ArrayTools.__init__(self, nLabels, nw)
        self.kwargs = kwargs

        counts = self.count_labels(self.kwargs['labels'])
        print("Before OverSample")  #
        print(counts)  #

        self.overSampleAmount = np.rint(div0(1 * max(counts), counts))  # ignore zeros in counts
        self.overSampleArraySize = int(sum(np.array(self.overSampleAmount, int) * counts))
        print(np.array(self.overSampleAmount, int) * counts)
        print(np.array(self.overSampleAmount, int))
        print(self.overSampleArraySize, len(self.kwargs['labels']))
        self.kwargOverSampled = {}
        self.randnum = np.random.randint(10000)
        for key in self.kwargs:
            if key == 'images':
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=np.float16, mode='w+',
                                             shape=(self.overSampleArraySize, int(self.nw)))
            elif key == 'labels':
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=np.uint16, mode='w+',
                                             shape=self.overSampleArraySize)
            else:
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=object, mode='w+',
                                             shape=self.overSampleArraySize)
            self.kwargOverSampled[key] = arrayOverSampled

        self.kwargShuf = self.shuffle_arrays(memmapName='pre-oversample_{}'.format(self.randnum), **self.kwargs)
        print(len(self.kwargShuf['labels']))

    def oversample_mp(self, i_in, offset_in, std_in, labelIndex_in):
        print('oversampling', i_in, len(self.kwargShuf['labels']))
        oversampled = {key: [] for key in self.kwargs}
        repeatAmount = int(self.overSampleAmount[labelIndex_in])
        for r in range(repeatAmount):
            for key in self.kwargs:
                if key == 'images':
                    oversampled[key].append(self.augment_data(self.kwargShuf[key][i_in], stdDevMean=0.05, stdDevStdDev=std_in))
                else:
                    oversampled[key].append(self.kwargShuf[key][i_in])
        return oversampled, offset_in, repeatAmount

    def collect_results(self, result):
        """Uses apply_async's callback to setup up a separate Queue for each process"""
        oversampled_in, offset_in, repeatAmount = result
        for key in self.kwargs:
            rlength_array = np.array(oversampled_in[key])
            self.kwargOverSampled[key][offset_in:repeatAmount+offset_in] = rlength_array[:]

    def over_sample_arrays(self, smote=False):
        if smote:
            return self.smote_oversample()
        else:
            return self.minority_oversample_with_noise()

    def minority_oversample_with_noise(self):
        offset = 0
        # pool = mp.Pool()
        for i in range(len(self.kwargShuf['labels'])):
            labelIndex = self.kwargShuf['labels'][i]
            if self.overSampleAmount[labelIndex] < 10:
                std = 0.03
            else:
                std = 0.05
            # pool.apply_async(self.oversample_mp, args=(i, offset, std, labelIndex), callback=self.collect_results)
            self.collect_results(self.oversample_mp(i, offset, std, labelIndex))
            offset += int(self.overSampleAmount[labelIndex])
        # pool.close()
        # pool.join()

        # for i, output in enumerate(outputs):
        #     self.collect_results(output)
        #     print('combining results...', i, len(outputs))

        print("Before Shuffling")
        self.kwargOverSampledShuf = self.shuffle_arrays(memmapName='oversampled_{}'.format(self.randnum), **self.kwargOverSampled)
        print("After Shuffling")

        return self.kwargOverSampledShuf

    def smote_oversample(self):
        sm = over_sampling.SMOTE(random_state=42, n_jobs=30)
        images, labels = sm.fit_sample(X=self.kwargShuf['images'], y=self.kwargShuf['labels'])

        self.kwargOverSampledShuf = self.shuffle_arrays(memmapName='oversampled_smote_{}'.format(self.randnum), images=images, labels=labels)

        return self.kwargOverSampledShuf


class CreateArrays(object):
    def __init__(self, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts, hostTypes=None, nHostTypes=None):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.minZ = minZ
        self.maxZ = maxZ
        self.numOfRedshifts = numOfRedshifts
        self.ageBinning = AgeBinning(minAge, maxAge, ageBinSize)
        self.numOfAgeBins = self.ageBinning.age_bin(maxAge-0.1) + 1
        self.nLabels = nTypes * self.numOfAgeBins * nHostTypes
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize, self.typeList, hostTypes, nHostTypes)
        self.hostTypes = hostTypes

    def combined_sn_gal_templates_to_arrays(self, args):
        print("enter func")
        snTemplateLocation, snTempList, galTemplateLocation, galTempList, snFractions = args
        ncolsDict = {'PTF10bzf.lnw': 2, 'PTF10qts.lnw': 6, 'PTF10vgv.lnw': 4, 'PTF12gzk.lnw': 8, 'iPTF13bvn.lnw': 16, 'sn00cn_bsnip.lnw': 1, 'sn00cp_bsnip.lnw': 1, 'sn00cu_bsnip.lnw': 2, 'sn00cw_bsnip.lnw': 1, 'sn00cx.lnw': 20, 'sn00dg_bsnip.lnw': 2, 'sn00dk_bsnip.lnw': 4, 'sn00dm_bsnip.lnw': 2, 'sn00dn_bsnip.lnw': 2, 'sn00fa_bsnip.lnw': 2, 'sn01bg_bsnip.lnw': 2, 'sn01dw_bsnip.lnw': 1, 'sn01eh_bsnip.lnw': 4, 'sn01en_bsnip.lnw': 3, 'sn01ep_bsnip.lnw': 5, 'sn02bg_bsnip.lnw': 1, 'sn02bo_bsnip.lnw': 5, 'sn02bz_bsnip.lnw': 1, 'sn02cf_bsnip.lnw': 1, 'sn02cr_bsnip.lnw': 3, 'sn02cs_bsnip.lnw': 2, 'sn02cx.lnw': 7, 'sn02ef_bsnip.lnw': 2, 'sn02el_bsnip.lnw': 2, 'sn02er.lnw': 23, 'sn02er_bsnip.lnw': 3, 'sn02eu_bsnip.lnw': 2, 'sn02fb_bsnip.lnw': 2, 'sn02fk_bsnip.lnw': 3, 'sn02ha_bsnip.lnw': 4, 'sn02hd_bsnip.lnw': 1, 'sn02he_bsnip.lnw': 4, 'sn02jy_bsnip.lnw': 2, 'sn02kf_bsnip.lnw': 1, 'sn03U_bsnip.lnw': 1, 'sn03Y_bsnip.lnw': 2, 'sn03ai_bsnip.lnw': 1, 'sn03cq_bsnip.lnw': 1, 'sn03gq_bsnip.lnw': 0, 'sn03he_bsnip.lnw': 2, 'sn03iv_bsnip.lnw': 2, 'sn04as_bsnip.lnw': 1, 'sn04bg_bsnip.lnw': 1, 'sn04bl_bsnip.lnw': 1, 'sn04bw_bsnip.lnw': 1, 'sn04dt_bsnip.lnw': 3, 'sn04ef_bsnip.lnw': 3, 'sn04eo_bsnip.lnw': 3, 'sn04ey_bsnip.lnw': 2, 'sn04fu_bsnip.lnw': 3, 'sn04fz_bsnip.lnw': 3, 'sn04gs_bsnip.lnw': 1, 'sn05am_bsnip.lnw': 3, 'sn05bc_bsnip.lnw': 2, 'sn05be_bsnip.lnw': 2, 'sn05bl_bsnip.lnw': 1, 'sn05cf_bsnip.lnw': 5, 'sn05de_bsnip.lnw': 4, 'sn05dv_bsnip.lnw': 1, 'sn05el_bsnip.lnw': 3, 'sn05eq_bsnip.lnw': 2, 'sn05gj_bsnip.lnw': 2, 'sn05hk_bsnip.lnw': 3, 'sn05kc_bsnip.lnw': 2, 'sn05ke_bsnip.lnw': 3, 'sn05ki_bsnip.lnw': 1, 'sn05ms_bsnip.lnw': 2, 'sn06N_bsnip.lnw': 4, 'sn06ac_bsnip.lnw': 1, 'sn06bq_bsnip.lnw': 3, 'sn06bu_bsnip.lnw': 1, 'sn06bz_bsnip.lnw': 1, 'sn06cf_bsnip.lnw': 2, 'sn06cp_bsnip.lnw': 1, 'sn06cq_bsnip.lnw': 1, 'sn06cs_bsnip.lnw': 1, 'sn06cz_bsnip.lnw': 1, 'sn06dm_bsnip.lnw': 4, 'sn06dw_bsnip.lnw': 4, 'sn06ef_bsnip.lnw': 3, 'sn06ej_bsnip.lnw': 4, 'sn06em_bsnip.lnw': 2, 'sn06et_bsnip.lnw': 3, 'sn06ev_bsnip.lnw': 2, 'sn06gt_bsnip.lnw': 1, 'sn06ke_bsnip.lnw': 1, 'sn06kf_bsnip.lnw': 3, 'sn06lf_bsnip.lnw': 3, 'sn06or_bsnip.lnw': 2, 'sn06os_bsnip.lnw': 2, 'sn06sr_bsnip.lnw': 2, 'sn07A_bsnip.lnw': 2, 'sn07af_bsnip.lnw': 3, 'sn07al_bsnip.lnw': 1, 'sn07ba_bsnip.lnw': 3, 'sn07bc_bsnip.lnw': 2, 'sn07bd_bsnip.lnw': 1, 'sn07bm_bsnip.lnw': 4, 'sn07fb_bsnip.lnw': 4, 'sn07fr_bsnip.lnw': 1, 'sn07fs_bsnip.lnw': 4, 'sn07gk_bsnip.lnw': 2, 'sn07kk_bsnip.lnw': 1, 'sn07qe_bsnip.lnw': 3, 'sn08A_bsnip.lnw': 1, 'sn08bt_bsnip.lnw': 1, 'sn08ds_bsnip.lnw': 4, 'sn08ec_bsnip.lnw': 5, 'sn08hs_bsnip.lnw': 1, 'sn1979C.lnw': 4, 'sn1980K.lnw': 6, 'sn1981B.lnw': 6, 'sn1983N.lnw': 2, 'sn1983V.lnw': 12, 'sn1984A.lnw': 9, 'sn1984L.lnw': 9, 'sn1986G.lnw': 19, 'sn1987A.lnw': 130, 'sn1989B.lnw': 23, 'sn1990B.lnw': 15, 'sn1990I.lnw': 4, 'sn1990N.lnw': 7, 'sn1990O.lnw': 6, 'sn1990U.lnw': 4, 'sn1991M.lnw': 4, 'sn1991T.lnw': 13, 'sn1991bg.lnw': 17, 'sn1992A.lnw': 12, 'sn1992H.lnw': 3, 'sn1992ar.lnw': 1, 'sn1993J.lnw': 35, 'sn1993ac.lnw': 1, 'sn1994D.lnw': 29, 'sn1994I.lnw': 32, 'sn1994M.lnw': 10, 'sn1994Q.lnw': 6, 'sn1994S.lnw': 5, 'sn1994T.lnw': 5, 'sn1994ae.lnw': 13, 'sn1995D.lnw': 9, 'sn1995E.lnw': 7, 'sn1995ac.lnw': 2, 'sn1995ak.lnw': 4, 'sn1995al.lnw': 11, 'sn1995bd.lnw': 7, 'sn1996C.lnw': 3, 'sn1996L.lnw': 3, 'sn1996X.lnw': 22, 'sn1996Z.lnw': 6, 'sn1996ab.lnw': 1, 'sn1996ai.lnw': 6, 'sn1996bk.lnw': 1, 'sn1996bl.lnw': 4, 'sn1996bo.lnw': 5, 'sn1996bv.lnw': 1, 'sn1996cb.lnw': 13, 'sn1997E.lnw': 4, 'sn1997Y.lnw': 4, 'sn1997bp.lnw': 11, 'sn1997bq.lnw': 13, 'sn1997br.lnw': 19, 'sn1997cn.lnw': 13, 'sn1997do.lnw': 14, 'sn1997dt.lnw': 7, 'sn1997ef.lnw': 24, 'sn1998S.lnw': 19, 'sn1998V.lnw': 9, 'sn1998ab.lnw': 12, 'sn1998aq.lnw': 17, 'sn1998bp.lnw': 11, 'sn1998bu.lnw': 26, 'sn1998bw.lnw': 19, 'sn1998co.lnw': 7, 'sn1998de.lnw': 7, 'sn1998dh.lnw': 10, 'sn1998dk.lnw': 9, 'sn1998dm.lnw': 10, 'sn1998dt.lnw': 9, 'sn1998dx.lnw': 7, 'sn1998ec.lnw': 6, 'sn1998ef.lnw': 4, 'sn1998eg.lnw': 5, 'sn1998es.lnw': 21, 'sn1998fa.lnw': 3, 'sn1999X.lnw': 6, 'sn1999aa.lnw': 35, 'sn1999ac.lnw': 31, 'sn1999aw.lnw': 4, 'sn1999bh.lnw': 1, 'sn1999by.lnw': 14, 'sn1999cc.lnw': 7, 'sn1999cl.lnw': 11, 'sn1999cp.lnw': 3, 'sn1999cw.lnw': 2, 'sn1999da.lnw': 1, 'sn1999dq.lnw': 19, 'sn1999ee.lnw': 12, 'sn1999ef.lnw': 3, 'sn1999ej.lnw': 5, 'sn1999ek.lnw': 4, 'sn1999em.lnw': 24, 'sn1999ex.lnw': 3, 'sn1999gd.lnw': 5, 'sn1999gh.lnw': 12, 'sn1999gi.lnw': 6, 'sn1999gp.lnw': 9, 'sn2000B.lnw': 5, 'sn2000E.lnw': 4, 'sn2000H.lnw': 11, 'sn2000bh.lnw': 1, 'sn2000bk.lnw': 3, 'sn2000ce.lnw': 4, 'sn2000cf.lnw': 6, 'sn2000cn.lnw': 9, 'sn2000cp.lnw': 3, 'sn2000cu.lnw': 1, 'sn2000cw.lnw': 1, 'sn2000cx.lnw': 26, 'sn2000dg.lnw': 4, 'sn2000dk.lnw': 6, 'sn2000dm.lnw': 1, 'sn2000dn.lnw': 4, 'sn2000er.lnw': 5, 'sn2000fa.lnw': 14, 'sn2001E.lnw': 3, 'sn2001G.lnw': 8, 'sn2001N.lnw': 8, 'sn2001V.lnw': 30, 'sn2001ah.lnw': 5, 'sn2001ay.lnw': 18, 'sn2001az.lnw': 5, 'sn2001bf.lnw': 6, 'sn2001bg.lnw': 5, 'sn2001br.lnw': 3, 'sn2001cj.lnw': 1, 'sn2001ck.lnw': 4, 'sn2001cp.lnw': 8, 'sn2001da.lnw': 6, 'sn2001eh.lnw': 18, 'sn2001el.lnw': 5, 'sn2001en.lnw': 9, 'sn2001ep.lnw': 22, 'sn2001ex.lnw': 1, 'sn2001fe.lnw': 9, 'sn2001fh.lnw': 6, 'sn2001gc.lnw': 8, 'sn2002G.lnw': 2, 'sn2002ao.lnw': 4, 'sn2002ap.lnw': 22, 'sn2002aw.lnw': 3, 'sn2002bf.lnw': 13, 'sn2002bo.lnw': 45, 'sn2002cd.lnw': 13, 'sn2002cf.lnw': 1, 'sn2002ck.lnw': 8, 'sn2002cr.lnw': 7, 'sn2002cs.lnw': 5, 'sn2002cu.lnw': 3, 'sn2002cx.lnw': 7, 'sn2002de.lnw': 10, 'sn2002dj.lnw': 20, 'sn2002dl.lnw': 4, 'sn2002do.lnw': 4, 'sn2002dp.lnw': 7, 'sn2002ef.lnw': 1, 'sn2002er.lnw': 25, 'sn2002es.lnw': 6, 'sn2002eu.lnw': 5, 'sn2002fb.lnw': 6, 'sn2002fk.lnw': 13, 'sn2002ha.lnw': 4, 'sn2002hd.lnw': 3, 'sn2002he.lnw': 5, 'sn2002hu.lnw': 4, 'sn2002hw.lnw': 5, 'sn2002ic.lnw': 4, 'sn2002jg.lnw': 4, 'sn2002jy.lnw': 10, 'sn2002kf.lnw': 11, 'sn2003U.lnw': 3, 'sn2003W.lnw': 16, 'sn2003Y.lnw': 3, 'sn2003bg.lnw': 8, 'sn2003cg.lnw': 40, 'sn2003ch.lnw': 12, 'sn2003cq.lnw': 7, 'sn2003dh.lnw': 10, 'sn2003du.lnw': 43, 'sn2003fa.lnw': 19, 'sn2003gn.lnw': 1, 'sn2003hu.lnw': 4, 'sn2003hv.lnw': 4, 'sn2003ic.lnw': 8, 'sn2003it.lnw': 15, 'sn2003iv.lnw': 7, 'sn2003jd.lnw': 19, 'sn2003kc.lnw': 7, 'sn2003kf.lnw': 21, 'sn2003lw.lnw': 2, 'sn2004L.lnw': 11, 'sn2004S.lnw': 7, 'sn2004as.lnw': 18, 'sn2004at.lnw': 11, 'sn2004aw.lnw': 25, 'sn2004bd.lnw': 4, 'sn2004bg.lnw': 5, 'sn2004bk.lnw': 1, 'sn2004dk.lnw': 3, 'sn2004dn.lnw': 3, 'sn2004dt.lnw': 32, 'sn2004ef.lnw': 15, 'sn2004eo.lnw': 14, 'sn2004et.lnw': 24, 'sn2004fe.lnw': 9, 'sn2004ff.lnw': 1, 'sn2004fu.lnw': 9, 'sn2004fz.lnw': 1, 'sn2004gc.lnw': 1, 'sn2004ge.lnw': 1, 'sn2004gq.lnw': 15, 'sn2004gs.lnw': 10, 'sn2004gt.lnw': 6, 'sn2004gv.lnw': 4, 'sn2005A.lnw': 3, 'sn2005M.lnw': 9, 'sn2005am.lnw': 16, 'sn2005az.lnw': 18, 'sn2005bc.lnw': 1, 'sn2005be.lnw': 1, 'sn2005bf.lnw': 30, 'sn2005bl.lnw': 8, 'sn2005bo.lnw': 2, 'sn2005cc.lnw': 10, 'sn2005cf.lnw': 61, 'sn2005cg.lnw': 6, 'sn2005cs.lnw': 28, 'sn2005ek.lnw': 6, 'sn2005el.lnw': 7, 'sn2005eq.lnw': 8, 'sn2005eu.lnw': 6, 'sn2005gj.lnw': 10, 'sn2005hc.lnw': 3, 'sn2005hf.lnw': 7, 'sn2005hg.lnw': 17, 'sn2005hj.lnw': 10, 'sn2005hk.lnw': 30, 'sn2005iq.lnw': 1, 'sn2005kc.lnw': 6, 'sn2005ke.lnw': 10, 'sn2005ki.lnw': 1, 'sn2005kl.lnw': 2, 'sn2005la.lnw': 5, 'sn2005ls.lnw': 2, 'sn2005lu.lnw': 3, 'sn2005mc.lnw': 2, 'sn2005mf.lnw': 4, 'sn2005mz.lnw': 4, 'sn2005na.lnw': 6, 'sn2006D.lnw': 6, 'sn2006H.lnw': 14, 'sn2006N.lnw': 12, 'sn2006S.lnw': 9, 'sn2006T.lnw': 5, 'sn2006X.lnw': 21, 'sn2006ac.lnw': 11, 'sn2006aj.lnw': 25, 'sn2006ak.lnw': 1, 'sn2006al.lnw': 6, 'sn2006ax.lnw': 4, 'sn2006az.lnw': 1, 'sn2006bp.lnw': 19, 'sn2006bq.lnw': 1, 'sn2006br.lnw': 2, 'sn2006bt.lnw': 13, 'sn2006bw.lnw': 1, 'sn2006bz.lnw': 3, 'sn2006cc.lnw': 1, 'sn2006cf.lnw': 6, 'sn2006cj.lnw': 8, 'sn2006cm.lnw': 2, 'sn2006cp.lnw': 10, 'sn2006cq.lnw': 1, 'sn2006cz.lnw': 3, 'sn2006el.lnw': 7, 'sn2006em.lnw': 1, 'sn2006ep.lnw': 5, 'sn2006eq.lnw': 1, 'sn2006et.lnw': 2, 'sn2006eu.lnw': 1, 'sn2006ev.lnw': 1, 'sn2006gj.lnw': 2, 'sn2006gr.lnw': 17, 'sn2006gt.lnw': 3, 'sn2006gz.lnw': 17, 'sn2006hb.lnw': 8, 'sn2006jc.lnw': 18, 'sn2006kf.lnw': 1, 'sn2006le.lnw': 22, 'sn2006lf.lnw': 19, 'sn2006mo.lnw': 1, 'sn2006nz.lnw': 1, 'sn2006oa.lnw': 7, 'sn2006ot.lnw': 2, 'sn2006sr.lnw': 6, 'sn2006te.lnw': 1, 'sn2007A.lnw': 3, 'sn2007C.lnw': 11, 'sn2007D.lnw': 1, 'sn2007F.lnw': 11, 'sn2007S.lnw': 10, 'sn2007Y.lnw': 8, 'sn2007ae.lnw': 2, 'sn2007af.lnw': 20, 'sn2007ag.lnw': 3, 'sn2007al.lnw': 8, 'sn2007ap.lnw': 2, 'sn2007au.lnw': 4, 'sn2007ax.lnw': 9, 'sn2007ba.lnw': 3, 'sn2007bc.lnw': 7, 'sn2007bd.lnw': 6, 'sn2007bg.lnw': 4, 'sn2007bj.lnw': 5, 'sn2007bm.lnw': 11, 'sn2007bz.lnw': 1, 'sn2007ca.lnw': 9, 'sn2007cg.lnw': 3, 'sn2007ci.lnw': 11, 'sn2007cl.lnw': 2, 'sn2007co.lnw': 16, 'sn2007cq.lnw': 2, 'sn2007fb.lnw': 6, 'sn2007fs.lnw': 4, 'sn2007gr.lnw': 15, 'sn2007hj.lnw': 14, 'sn2007if.lnw': 32, 'sn2007jg.lnw': 2, 'sn2007kj.lnw': 2, 'sn2007kk.lnw': 14, 'sn2007le.lnw': 23, 'sn2007nq.lnw': 2, 'sn2007qe.lnw': 10, 'sn2007ru.lnw': 9, 'sn2007sr.lnw': 7, 'sn2007ux.lnw': 2, 'sn2007uy.lnw': 7, 'sn2008A.lnw': 14, 'sn2008C.lnw': 4, 'sn2008D.lnw': 18, 'sn2008L.lnw': 1, 'sn2008Q.lnw': 6, 'sn2008R.lnw': 1, 'sn2008Z.lnw': 14, 'sn2008ae.lnw': 5, 'sn2008af.lnw': 2, 'sn2008ar.lnw': 16, 'sn2008ax.lnw': 22, 'sn2008bf.lnw': 12, 'sn2008bo.lnw': 9, 'sn2009bb.lnw': 11, 'sn2009er.lnw': 8, 'sn2009iz.lnw': 9, 'sn2009jf.lnw': 33, 'sn2009mg.lnw': 10, 'sn2009nz.lnw': 1, 'sn2010ay.lnw': 2, 'sn2010ma.lnw': 3, 'sn2011bm.lnw': 8, 'sn2011dh.lnw': 28, 'sn2011ei.lnw': 11, 'sn2011fu.lnw': 7, 'sn2012ap.lnw': 11, 'sn2012bz.lnw': 3, 'sn2013cq.lnw': 1, 'sn2013dk.lnw': 1, 'sn2013dx.lnw': 16, 'sn89B.lnw': 23, 'sn91T.lnw': 13, 'sn91T_bsnip.lnw': 3, 'sn91bg.lnw': 16, 'sn91bg_bsnip.lnw': 5, 'sn94D.lnw': 36, 'sn94I.lnw': 18, 'sn94S_bsnip.lnw': 1, 'sn94ae.lnw': 13, 'sn94ae_bsnip.lnw': 0, 'sn95D.lnw': 9, 'sn95D_bsnip.lnw': 2, 'sn95E_bsnip.lnw': 1, 'sn95ac.lnw': 1, 'sn95ac_bsnip.lnw': 1, 'sn97Y_bsnip.lnw': 1, 'sn97br_bsnip.lnw': 2, 'sn98bu.lnw': 27, 'sn98bu_bsnip.lnw': 1, 'sn98bw.lnw': 16, 'sn98dx_bsnip.lnw': 1, 'sn98ec_bsnip.lnw': 1, 'sn98ef_bsnip.lnw': 2, 'sn98es_bsnip.lnw': 2, 'sn99aa_bsnip.lnw': 4, 'sn99by.lnw': 14, 'sn99by_bsnip.lnw': 0, 'sn99cp_bsnip.lnw': 2, 'sn99da_bsnip.lnw': 3, 'sn99dk_bsnip.lnw': 4, 'sn99dq_bsnip.lnw': 3, 'sn99gd_bsnip.lnw': 1, 'snls03D3bb.lnw': 1}

        print("defined dict")
        numSpectra = 0
        for name in snTempList:
            try:
                numSpectra += ncolsDict[name]
            except KeyError:
                print(name, "NOT IN DICTIONARY")

        print("memmory mapped in func")
        randnum = np.random.randint(10000)
        arraySize = numSpectra * len(galTempList) * len(snFractions) * self.numOfRedshifts
        images = np.memmap('images_{}_{}.dat'.format(snTempList[0], randnum), dtype=np.float16, mode='w+', shape=(arraySize, int(self.nw)))
        labelsIndexes = np.memmap('labels_{}_{}.dat'.format(snTempList[0], randnum), dtype=np.uint16, mode='w+', shape=arraySize)
        filenames = np.memmap('filenames_{}_{}.dat'.format(snTempList[0], randnum), dtype=object, mode='w+', shape=arraySize)
        typeNames = np.memmap('typeNames_{}_{}.dat'.format(snTempList[0], randnum), dtype=object, mode='w+', shape=arraySize)
        nRows = 0
        print("Create array loop begin...")
        for j in range(len(galTempList)):
            galFilename = galTemplateLocation + galTempList[j] if galTemplateLocation is not None else None
            for i in range(0, len(snTempList)):
                nCols = 15
                readSpectra = ReadSpectra(self.w0, self.w1, self.nw, snTemplateLocation + snTempList[i], galFilename)
                for ageidx in range(0, 1000):
                    if ageidx >= nCols:
                        break
                    for snCoeff in snFractions:
                        galCoeff = 1 - snCoeff
                        if self.numOfRedshifts == 1:
                            redshifts = [self.minZ]
                        else:
                            redshifts = np.random.uniform(low=self.minZ, high=self.maxZ, size=self.numOfRedshifts)
                        for z in redshifts:
                            tempWave, tempFlux, nCols, ages, tType, tMinIndex, tMaxIndex = readSpectra.sn_plus_gal_template(ageidx, snCoeff, galCoeff, z)
                            if tMinIndex == tMaxIndex or not tempFlux.any():
                                print("NO DATA for {} {} ageIdx:{} z>={}".format(galTempList[j], snTempList[i], ageidx, z))
                                break

                            if self.minAge < float(ages[ageidx]) < self.maxAge:
                                if self.hostTypes is None:  # Checks if we are classifying by host as well
                                    labelIndex, typeName = self.createLabels.label_array(tType, ages[ageidx], host=None)
                                else:
                                    labelIndex, typeName = self.createLabels.label_array(tType, ages[ageidx], host=galTempList[j])
                                if tMinIndex > (self.nw - 1):
                                    continue
                                nonzeroflux = tempFlux[tMinIndex:tMaxIndex + 1]
                                newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                                newflux2 = np.concatenate((tempFlux[0:tMinIndex], newflux, tempFlux[tMaxIndex + 1:]))
                                images[nRows] = np.array([newflux2])
                                labelsIndexes[nRows] = labelIndex
                                filenames[nRows] = "{0}_{1}_{2}_{3}_snCoeff{4}_z{5}".format(snTempList[i], tType, str(ages[ageidx]), galTempList[j], snCoeff, (z))
                                typeNames[nRows] = typeName
                                nRows += 1
                print(snTempList[i], nCols, galTempList[j])

        print("Done func")
        return images, labelsIndexes, filenames, typeNames, nRows

    def collect_results(self, result, wheretostartappending):
        """Uses apply_async's callback to setup up a separate Queue for each process"""
        imagesPart, labelsPart, filenamesPart, typeNamesPart, nRows = result

        self.images[wheretostartappending:wheretostartappending+nRows] = imagesPart[0:nRows]
        self.labelsIndexes[wheretostartappending:wheretostartappending+nRows] = labelsPart[0:nRows]
        self.filenames[wheretostartappending:wheretostartappending+nRows] = filenamesPart[0:nRows]
        self.typeNames[wheretostartappending:wheretostartappending+nRows] = typeNamesPart[0:nRows]

    def combined_sn_gal_arrays_multiprocessing(self, snTemplateLocation, snTempFileList, galTemplateLocation, galTempFileList):
        if galTemplateLocation is None or galTempFileList is None:
            galTempList = [None]
            galTemplateLocation = None
            snFractions = [1.0]
        else:
            galTempList = temp_list(galTempFileList)
            snFractions = [0.99, 0.98, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        snTempList = temp_list(snTempFileList)  # 514 files
        galAndSnTemps = list(itertools.product(galTempList, snTempList))[0:2]
        argsList = []
        for gal, sn in galAndSnTemps:
            argsList.append((snTemplateLocation, [sn], galTemplateLocation, [gal], snFractions))

        numSpectra = 3968  # sum nColsDict
        arraySize = numSpectra * len(galTempList) * len(snFractions) * self.numOfRedshifts
        print("Arraysize is:", arraySize)

        self.images = np.memmap('all_images.dat', dtype=np.float16, mode='w+', shape=(arraySize, int(self.nw)))
        self.labelsIndexes = np.memmap('all_labels.dat', dtype=np.uint16, mode='w+', shape=arraySize)
        self.filenames = np.memmap('all_filenames.dat', dtype=object, mode='w+', shape=arraySize)
        self.typeNames = np.memmap('all_typeNames.dat', dtype=object, mode='w+', shape=arraySize)

        print("images GiB:", self.images.nbytes / 2**30)
        print("labels GiB:", self.labelsIndexes.nbytes / 2**30)
        print("filenames GiB:", self.filenames.nbytes / 2**30)
        print("typeNames GiB:", self.typeNames.nbytes / 2**30)

        # #  Multiprocessing with map_async (faster)
        # pool = mp.Pool()
        # results = pool.map_async(self.combined_sn_gal_templates_to_arrays, argsList)
        # pool.close()
        # pool.join()
        # outputs = results.get()
        # for i, output in enumerate(outputs):
        #     self.collect_results(output)
        #     print('combining results...', output[-1], i, len(outputs))

        print("Begin pooling...")
        # #  Multiprocessing with apply_async (better when arrays are large - i.e. agnostic redshift)
        results = []
        pool = mp.Pool(processes=50)
        print("pool")
        for arg in argsList:
            print("argLoop")
            result = pool.apply_async(self.combined_sn_gal_templates_to_arrays, [arg])
            results.append(result)
        print("close pool")
        pool.close()
        pool.join()
        print("Finished Pooling")

        wheretostartappending = 0
        for i, p in enumerate(results):
            output = p.get()
            nRows = output[-1]
            self.collect_results(output, wheretostartappending)
            wheretostartappending += nRows
            print('combining results...', nRows, i, len(results))

        print("Completed Creating Arrays!")

        # Delete temporary memory mapping files
        for filename in glob.glob('images_*.dat'):
            os.remove(filename)
        for filename in glob.glob('labels_*.dat'):
            os.remove(filename)
        for filename in glob.glob('filenames_*.dat'):
            os.remove(filename)
        for filename in glob.glob('typeNames_*.dat'):
            os.remove(filename)

        return self.images, self.labelsIndexes, self.filenames, self.typeNames
