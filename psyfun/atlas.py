import numpy as np

from iblutil.numerical import ismember
from iblatlas.atlas import BrainRegions
import iblatlas.plots as anatomyplots
regions = BrainRegions()

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