import numpy as np

from iblutil.numerical import ismember
from iblatlas.atlas import BrainRegions
import iblatlas.plots as anatomyplots
regions = BrainRegions()


def coarse_regions(acronyms):
    regions = acronyms.copy()
    regions[
        np.isin(acronyms, ['ACAd1', 'ACAd2/3', 'ACAd5', 'ACAd6a', 'ACAv1', 'ACAv2/3', 'ACAv5', 'ACAv6a','ACAv6b', 'ILA5', 'ILA6a',
                          'PL5', 'PL6a', 'PL6b', 'RSPagl1', 'RSPagl2/3', 'RSPagl5', 'RSPagl6a', 'RSPagl6b'])
    ] = 'Prefrontal Ctx'
    regions[
        np.isin(acronyms, ['AMd', 'CL', 'CM', 'IAD', 'IAM', 'IMD', 'LD', 'LP', 'MD', 'PCN', 'PO', 'PR', 'PVT', 'PVi', 'RE', 'RH', 'RT', 'SMT',
                          'TH', 'VAL', 'VM', 'VPL', 'VPM', 'VPMpc', 'Xi'])
    ] = 'Thalamus' 
    regions[
        np.isin(acronyms, ['AUDd5', 'AUDp5', 'AUDp6b', 'AUDv4', 'AUDv5', 'AUDv6a', 'AUDv6b'])
    ] = 'Auditory Ctx'
    regions[
        np.isin(acronyms, ['BLAa', 'BLAp', 'BMAa', 'CEAc', 'CEAl', 'CEAm', 'COApm', 'IA', 'LA', 'MEA', 'PA', 'PAA'])
    ] = 'Amygdala' 
    regions[
        np.isin(acronyms, ['CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg', 'HPF', 'IG'])
    ] = 'Hippocampus'
    regions[
        np.isin(acronyms, ['ACB', 'CP', 'BST', 'GPe', 'GPi', 'PAL', 'STR'])
    ] = 'Striatum'
    regions[
        np.isin(acronyms, ['DMH', 'HY', 'LPO', 'MEPO', 'PVHd', 'ZI'])
    ] = 'Hypothal'
    regions[
        np.isin(acronyms, ['ECT5', 'ECT6a', 'ECT6b', 'ENTl3', 'ENTl5', 'ENTl6a', 'TEa1', 'TEa2/3', 'TEa4', 'TEa5', 'TEa6a'])
    ] = 'Entorhinal Ctx'
    regions[
        np.isin(acronyms, ['LH'])
    ] = 'Habenula'
    regions[
        np.isin(acronyms, ['LSc', 'LSr', 'LSv', 'MS', 'SF', 'SH'])
    ] = 'Septum'
    regions[
        np.isin(acronyms, ['MOp1', 'MOp2/3', 'MOp6a', 'MOp6b'])
    ] = 'Motor Ctx'
    regions[
        np.isin(acronyms, ['MOs1', 'MOs2/3', 'MOs5', 'MOs6a', 'MOs6b'])
    ] = 'Supp. Motor Ctx'
    regions[
        np.isin(acronyms, ['NDB', 'SI'])
    ] = 'Basal Forebrain'
    regions[
        np.isin(acronyms, ['DP', 'OLF', 'PIR', 'TTd'])
    ] = 'Olfactory Ctx'
    regions[
        np.isin(acronyms, ['SSp-bfd2/3', 'SSp-bfd4', 'SSp-bfd5', 'SSp-bfd6a', 'SSp-bfd6b', 'SSp-ll6a', 'SSp-ll6b', 'SSs5', 'SSs6a', 'SSs6b'])
    ] = 'Somatosens. Ctx'
    regions[
        np.isin(acronyms, ['VISa1', 'VISa2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b', 'VISrl4', 'VISrl5', 'VISrl6a'])
    ] = 'Visual Ctx'
    regions[
        np.isin(acronyms, ['CTXsp', 'EDp','EPd', 'EPv'])
    ] = 'Claustrum'
    regions[
        np.isin(acronyms, ['SEZ', 'VL', 'alv', 'amc', 'ar', 'ccb', 'ccg', 'ccs', 'chpl', 'cing', 'ec', 'ee', 'em', 'fa', 'fi', 'fiber tracts',
                           'fp', 'fr', 'int', 'opt', 'or', 'scwm', 'sm', 'st'])
    ] = 'Fiber tract'
    regions[
        np.isin(acronyms, ['root', 'void', None])
    ] = 'None'
    return regions
    

def remap_names(acronyms, source='Allen', dest='Beryl'):
    br = BrainRegions()
    _, inds = ismember(br.acronym2id(acronyms), br.id[br.mappings[source]])
    remapped_acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    return remapped_acronyms

def combine_regions(allen_acronyms, split_thalamus=True, abbreviate=True):
    acronyms = remap_names(allen_acronyms)
    regions = np.array(['root'] * len(acronyms), dtype=object)
    if abbreviate:
        regions[np.isin(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'mPFC'
        regions[np.isin(acronyms, ['MOs'])] = 'M2'
        regions[np.isin(acronyms, ['ORBl', 'ORBm'])] = 'OFC'
        if split_thalamus:
            regions[np.isin(acronyms, ['PO'])] = 'PO'
            regions[np.isin(acronyms, ['LP'])] = 'LP'
            regions[np.isin(acronyms, ['LD'])] = 'LD'
            regions[np.isin(acronyms, ['RT'])] = 'RT'
            regions[np.isin(acronyms, ['VAL'])] = 'VAL'
        else:
            regions[np.isin(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thal'
        regions[np.isin(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'SC'
        regions[np.isin(acronyms, ['RSPv', 'RSPd'])] = 'RSP'
        regions[np.isin(acronyms, ['MRN'])] = 'MRN'
        regions[np.isin(acronyms, ['ZI'])] = 'ZI'
        regions[np.isin(acronyms, ['PAG'])] = 'PAG'
        regions[np.isin(acronyms, ['SSp-bfd'])] = 'BC'
        #regions[np.isin(acronyms, ['LGv', 'LGd'])] = 'LG'
        regions[np.isin(acronyms, ['PIR'])] = 'Pir'
        #regions[np.isin(acronyms, ['SNr', 'SNc', 'SNl'])] = 'SN'
        regions[np.isin(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'VIS'
        regions[np.isin(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amyg'
        regions[np.isin(acronyms, ['AON', 'TTd', 'DP'])] = 'OLF'
        regions[np.isin(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Str'
        regions[np.isin(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hipp'
    else:
        regions[np.isin(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'Medial prefrontal cortex'
        regions[np.isin(acronyms, ['MOs'])] = 'Secondary motor cortex'
        regions[np.isin(acronyms, ['ORBl', 'ORBm'])] = 'Orbitofrontal cortex'
        if split_thalamus:
            regions[np.isin(acronyms, ['PO'])] = 'Thalamus (PO)'
            regions[np.isin(acronyms, ['LP'])] = 'Thalamus (LP)'
            regions[np.isin(acronyms, ['LD'])] = 'Thalamus (LD)'
            regions[np.isin(acronyms, ['RT'])] = 'Thalamus (RT)'
            regions[np.isin(acronyms, ['VAL'])] = 'Thalamus (VAL)'
        else:
            regions[np.isin(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thalamus'
        regions[np.isin(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'Superior colliculus'
        regions[np.isin(acronyms, ['RSPv', 'RSPd'])] = 'Retrosplenial cortex'
        regions[np.isin(acronyms, ['MRN'])] = 'Midbrain reticular nucleus'
        regions[np.isin(acronyms, ['AON', 'TTd', 'DP'])] = 'Olfactory areas'
        regions[np.isin(acronyms, ['ZI'])] = 'Zona incerta'
        regions[np.isin(acronyms, ['PAG'])] = 'Periaqueductal gray'
        regions[np.isin(acronyms, ['SSp-bfd'])] = 'Barrel cortex'
        #regions[np.isin(acronyms, ['LGv', 'LGd'])] = 'Lateral geniculate'
        regions[np.isin(acronyms, ['PIR'])] = 'Piriform'
        #regions[np.isin(acronyms, ['SNr', 'SNc', 'SNl'])] = 'Substantia nigra'
        regions[np.isin(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'Visual cortex'
        regions[np.isin(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amygdala'
        regions[np.isin(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Tail of the striatum'
        regions[np.isin(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hippocampus'
    return regions