# setup logging
import logger

log = logger.get_logger(__name__)

import json
import re
from collections import Counter

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from cheminformatics.rdkit_toolkit import Rdkit_operation, remove_fragments


# standardise the molecular structures
def standardise_check_molecule(smiles: str) -> dict():
    """
    Utility function to standardise and check a molecule. In case of error it returns immediately with the first error occurred.

    The function returns a dictionary with the
    - standardised molecule
    - the standardised smiles
    - a list of standarisation operations that were performed on the molecule
    - the first occurred error
    - a list of warnings
    :param smiles: input smiles
    :return: dictionary with standardised smiles, standardised mol, list of standardisation operations, any error and list of warnings
    """

    # initialise the result dictionary
    result = {
        "smiles": smiles,
        "smiles (standardised)": None,
        "mol (standardised)": None,
        "standardisation operations": [],
        "error": None,
        "warnings": [],
    }

    # conversion to mol with the standard sanitisation
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except:
        pass
    finally:
        if not mol:
            result["error"] = "Could not convert smiles to mol."
            return result

    # check for any atom in the structure (typically seen as "*" in smiles or as A)
    has_any_atom = any([atom.GetSymbol() == "*" for atom in mol.GetAtoms()])
    if has_any_atom:
        result["error"] = "Structure contains wildcard."
        return result

    # disconnect metals, this break bonds to alkali metals and other metals but not when connected to carbon
    # smis = ('CCO[Fe]', 'CCO[AlH2]', 'C[Hg]C', 'Br[Mg]c1ccccc1CCC(=O)O[Na]')
    # mol = Chem.MolFromSmiles(smis[0])
    with Rdkit_operation() as sio:
        md = rdMolStandardize.MetalDisconnector()
        mol = md.Disconnect(mol)
        message = sio.getvalue()
        match_expr = re.compile(r"(?i)(Removed .*?)\n")
        ops_applied = re.findall(match_expr, message)
        if ops_applied:
            result["standardisation operations"].append(
                {"disconnect metals": ops_applied}
            )

    # remove toxicologically irrelevant fragments
    mol, removed_fragments = remove_fragments(mol)
    if removed_fragments:
        removed_fragments = dict(Counter(removed_fragments))
        result["standardisation operations"].append(
            {"inert fragment removal": removed_fragments}
        )
    if not mol:
        result["error"] = "no fragment left after removing inert fragments"
        return result

    # make remaining fragments unique
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    unique_smiles = set()
    unique_fragments = []
    removed_fragments = []
    for fragment in fragments:
        smiles = Chem.MolToSmiles(fragment, canonical=True)  # Get canonical SMILES
        if smiles not in unique_smiles:
            unique_smiles.add(smiles)
            unique_fragments.append(fragment)
        else:
            removed_fragments.append(smiles)
    removed_fragments = dict(Counter(removed_fragments))
    if removed_fragments:
        removed_fragments = dict(Counter(removed_fragments))
        result["standardisation operations"].append(
            {"duplicate fragment removal": removed_fragments}
        )
    # check if there is more than one remaining fragment
    if len(unique_fragments) > 1:
        result["error"] = "more than one fragment in the structure"
        return result
    else:
        mol = unique_fragments[0]

    # normalise functional groups
    with Rdkit_operation() as sio:
        normaliser = rdMolStandardize.Normalizer()
        mol = normaliser.normalize(mol)
        message = sio.getvalue()
        match_expr = re.compile(r"(Rule applied: .*?)\n")
        ops_applied = match_expr.findall(message)
        if ops_applied:
            result["standardisation operations"].append(
                {"functional group normalisation": ops_applied}
            )

    # remove charges
    with Rdkit_operation() as sio:
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        message = sio.getvalue()
        match_expr = re.compile(r"(Removed .*?)\n")
        ops_applied = match_expr.findall(message)
        if ops_applied:
            result["standardisation operations"].append({"charge removal": ops_applied})

    # check if the structure contains atoms other than the allowed ones
    allowed_atoms = ["C", "O", "N", "Cl", "S", "F", "Br", "P", "B", "Si", "I", "H"]
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            result["error"] = (
                f"structure contains atoms other than allowed {json.dumps(allowed_atoms)}"
            )
            return result

    # check if the structure contains at least one carbon atom
    min_num_carbon_atoms = 1
    num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    if num_carbons < min_num_carbon_atoms:
        result["error"] = f"structure did not contain a carbon atom"
        return result

    # set the final structures
    smiles = Chem.MolToSmiles(mol, canonical=True)
    result["smiles (standardised)"] = smiles
    result["mol (standardised)"] = mol

    # generate the tautomers
    te = rdMolStandardize.TautomerEnumerator()
    # Set maximum number of tautomers
    te.SetMaxTautomers(100)
    parent_taut = te.Canonicalize(mol)
    tauts = te.Enumerate(parent_taut)
    scores = [f"{te.ScoreTautomer(x):.1f}" for x in tauts]
    if scores:
        result["tautomers"] = [
            {
                "tautomer ID": i,
                "tautomer": Chem.MolToSmiles(taut),
                "score": score,
                "canonical tautomer": Chem.MolToSmiles(taut)
                == Chem.MolToSmiles(parent_taut),
            }
            for i, (taut, score) in enumerate(zip(tauts, scores))
        ]
        print(result["tautomers"])
        result["canonical tautomer == smiles (standardised)"] = (
            Chem.MolToSmiles(parent_taut) == result["smiles (standardised)"]
        )

    # remove explicit hydrogens
    # mol = Chem.RemoveHs(mol)  # Remove explicit hydrogens
    #
    # enumerator = rdMolStandardize.TautomerEnumerator()
    # mol = enumerator.Canonicalize(mol)  # Normalize tautomers

    # convert lists to json
    result["standardisation operations"] = json.dumps(
        result["standardisation operations"]
    )
    result["warnings"] = json.dumps(result["warnings"])
    result["tautomers"] = json.dumps(result["tautomers"])

    return result
