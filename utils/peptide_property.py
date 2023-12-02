from typing import Dict, List

import modlamp.analysis as manalysis
import numpy as np

hydrophilic = ['R', 'N', 'D', 'Q', 'E', 'K']
aa_with_positive_charge = ['K', 'R', 'H']
aa_with_negative_charge = ['D', 'E']


def calculate_hydrophobicity(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalAnalysis(data)
    h.calc_H(scale='eisenberg')
    return h.H[0]


def calculate_hydrophobicmoment(data: List[str]) -> np.ndarray:
    h = manalysis.PeptideDescriptor(data, 'eisenberg')
    h.calculate_moment()
    return h.descriptor.flatten()


def calculate_charge(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalAnalysis(data)
    h.calc_charge()
    return h.charge[0]


def calculate_isoelectricpoint(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return h.descriptor.flatten()


def calculate_chargeDensity(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.charge_density()
    return h.descriptor.flatten()


def calculate_length(sequences: List[str]) -> np.ndarray:
    return np.array([len(seq) for seq in sequences])


def calculate_physchem_prop(sequences: List[str]) -> Dict[str, object]:
    return {
        "Length": calculate_length(sequences).tolist(),
        "Hydrophobicity": calculate_hydrophobicity(sequences).tolist(),
        "Hydrophobic moment": calculate_hydrophobicmoment(sequences).tolist(),
        "Charge": calculate_charge(sequences).tolist(),
        "Charge Density": calculate_chargeDensity(sequences).tolist(),
        "Isoelectric point": calculate_isoelectricpoint(sequences).tolist(),
    }