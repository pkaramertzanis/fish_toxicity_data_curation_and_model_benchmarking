import logger

log = logger.get_logger(__name__)

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat

from cheminformatics.rdkit_toolkit import Rdkit_operation

# from mordred import Calculator, descriptors


def compute_morgan_fingerprint(
    mol: Chem.Mol, radius: int = 2, nBits: int = 2048
) -> rdkit.DataStructs.cDataStructs.ExplicitBitVect:
    """
    Compute the Morgan fingerprint for a given molecule.
    :param mol: RDKit molecule object
    :param radius: Radius for the Morgan fingerprint
    :param nBits: Number of bits in the fingerprint
    :return: Morgan fingerprint as a bit vector
    """
    if mol is None:
        return None
    try:
        with Rdkit_operation() as sio:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
            fp = gen.GetFingerprint(mol)  # returns an ExplicitBitVect
            error_warning = sio.getvalue()
            if error_warning:
                # Log the warning if there is an RDKit warning
                log.warning(f"RDKit warning: {error_warning}")
            return fp
    except Exception as e:
        log.error(f"Error computing Morgan fingerprint: {e}")
        return None


def compute_mordred_descriptors(mol: Chem.Mol, ignore_3D: bool = True) -> dict:
    """
    Compute the mordred descriptors for a given molecule and return them as a dictionary.
    :param mol: RDKit molecule object
    :param ignore_3D: Whether to ignore 3D information
    :return: Dictionary with the Morgan descriptors
    """
    if mol is None:
        return {}
    try:
        # set up Mordred calculator
        calc = Calculator(descriptors, ignore_3D=True)
        # compute descriptors
        desc = calc(mol)
        # convert to dictionary
        desc_dict = desc.asdict()
        return desc_dict
    except Exception as e:
        log.error(f"Error computing Mordred descriptors: {e}")
        return {}
