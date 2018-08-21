import numpy as np
import os
from astrodash.combine_sn_and_host import CombineSnAndHost

scriptDirectory = os.path.dirname(os.path.abspath(__file__))


def get_templates(snName, snAge, hostName, snTemplates, galTemplates, nw):
    snInfos = np.copy(snTemplates[snName][snAge]['snInfo'])
    snNames = np.copy(snTemplates[snName][snAge]['names'])
    if hostName != "No Host" and hostName != "":
        hostInfos = np.copy(galTemplates[hostName]['galInfo'])
        hostNames = np.copy(galTemplates[hostName]['names'])
    else:
        hostInfos = np.array([[np.zeros(nw), np.zeros(nw), 1, nw - 1]])
        hostNames = np.array(["No Host"])

    return snInfos, snNames, hostInfos, hostNames


def load_templates(templateFilename):
    loaded = np.load(os.path.join(scriptDirectory, templateFilename))
    snTemplates = loaded['snTemplates'][()]
    galTemplates = loaded['galTemplates'][()]

    return snTemplates, galTemplates


def combined_sn_and_host_data(snCoeff, galCoeff, z, snInfo, galInfo, w0, w1, nw):
    combineSnAndHost = CombineSnAndHost(snInfo, galInfo, w0, w1, nw)

    return combineSnAndHost.template_data(snCoeff, galCoeff, z)


if __name__ == "__main__":
    templateFilename1 = 'models/sn_and_host_templates.npz'
    snTemplates1, galTemplates1 = load_templates(templateFilename1)
    snInfoList = snTemplates1['Ia-norm']['-2 to 2']['snInfo']
    galInfoList = galTemplates1['S0']['galInfo']
    for i in range(len(snInfoList)):
        wave, flux, minMaxIndex = combined_sn_and_host_data(snCoeff=0.5, galCoeff=0.5, z=0, snInfo=snInfoList[i], galInfo=galInfoList[0], w0=3500, w1=10000, nw=1024)
        print(i)
