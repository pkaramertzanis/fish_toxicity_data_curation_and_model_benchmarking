'''
Module to predict dissociation using the ACD Percepta kernel for all standardised structures.
'''
# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '08_ACD_percepta_dissociation', file_name ='logs/08_ACD_percepta_dissociation.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd

from rdkit import Chem

# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr',False)
# maximum number of columns
pd.set_option("display.max_columns",50)
# maximum number of rows
pd.set_option("display.max_rows",5000)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True


# read the standardised structures
smiles_uniq_set = pd.read_excel(r'data/structures/smiles.xlsx', sheet_name='smiles (standardised)')

# read the ACD Percepta dissociation predictions
pka_predictions = pd.read_csv(r'D:\myApplications\local\2025_10_10_acute_chronic_fish_QSAR\data\predictions\ACDPercepta/raw/results.txt', sep='\t', skiprows=2).dropna(axis=1, how="all")
pka_predictions = pka_predictions.rename({'ID': 'mol ID'}, axis='columns').drop('Smiles', axis='columns')
# .. convert to long format
pka_predictions = (
    pd.wide_to_long(
        pka_predictions,
        stubnames=[
            "ACD_pKa_Apparent",
            "ACD_pKa_DissAtom_Apparent",
        ],
        i=["mol ID"],
        j="pka ID",
        sep="_",
        suffix=r"\d+",
    )
    .reset_index()
)
# drop NaN values
pka_predictions = pka_predictions.dropna(
    subset=["ACD_pKa_Apparent", "ACD_pKa_DissAtom_Apparent"],
    how="all",
)
# pack the values back to a single row per molecule using a dict structure
pka_predictions = (pka_predictions
                   .groupby(['mol ID'], group_keys=False)
                   .apply(lambda df: df.sort_values(by='ACD_pKa_Apparent', ascending=True).to_dict(orient='records'), include_groups=False)
                   .reset_index())
log.info(f'Number of molecules with ACD Percepta pKa predictions: {pka_predictions["mol ID"].nunique()}')


# read the charge state from the sdf
sdf_file = r'D:\myApplications\local\2025_10_10_acute_chronic_fish_QSAR\data\predictions\ACDPercepta\raw/pka_ionic_form_file.sdf'
def get_charge_at_ph(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    molecular_charges = []
    for mol in suppl:
        mol_id = None
        mol_id = mol.GetProp('ID')
        if mol is None:
            continue
        try:
            pH = mol.GetProp('Dominant_Ionic_Form_at_pH')
            # obtain the charge from the mol structure
            molecular_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            molecular_charge = {'mol ID': mol_id,
                                'prediction status': 'succeeded',
                                f'pH': pH,
                                f'molecular charge': molecular_charge}
            molecular_charges.append(molecular_charge)
        except KeyError as ex:
            log.warning(f'Could not read molecule {mol_id} from sdf file')
            molecular_charge = {'mol ID': mol_id,
                                'prediction status': 'failed',
                                f'pH': None,
                                f'molecular charge': None}
            molecular_charges.append(molecular_charge)

    return molecular_charges
molecular_charges = get_charge_at_ph(sdf_file)
molecular_charges = pd.DataFrame(molecular_charges)
log.info(f'Number of molecules with ACD Percepta charge state predictions: {molecular_charges["mol ID"].nunique()}')
molecular_charges = (molecular_charges
                     .dropna(subset=['pH'])
                     .pivot(index=['mol ID', 'prediction status'], columns='pH', values='molecular charge').reset_index())
molecular_charges = molecular_charges.rename(columns=lambda x: f'charge at pH {int(float(x))}' if molecular_charges[x].dtype=='float' else x)
molecular_charges['mol ID'] = pd.to_numeric(molecular_charges['mol ID'], errors='coerce').astype('Int64')

# put everything together
res = (smiles_uniq_set
       .merge(pka_predictions, on='mol ID', how='left')
       .merge(molecular_charges, on='mol ID', how='left')
       .fillna({'prediction status': 'failed'})
       )
res.to_excel(r'data/predictions/ACDPercepta/processed/ACDPercepta_dissociation_predictions.xlsx', index=False)