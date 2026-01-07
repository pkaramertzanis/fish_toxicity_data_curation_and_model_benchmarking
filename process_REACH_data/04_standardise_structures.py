'''
Purpose: standardise molecular structures from REACH fish toxicity datasets and prepare inputs for various QSAR tools
For selected tools, structures that are likely to cause issues during predictions (e.g., high molecular weight, specific substructures) are replaced with 'XX' to make the prediction fail.
This may affect later on operations that rely on the number of structures, e.g., applicability domain assessments if we merge with the smiles (standardised) column.
'''

# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '04_standardise_structures', file_name ='logs/04_standardise_structures.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np

from cheminformatics.rdkit_toolkit import remove_fragments
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Draw
from cheminformatics.rdkit_toolkit import Rdkit_operation

import json
import re

from collections import Counter

from cheminformatics.standardise_check import standardise_check_molecule


# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr',False)
# maximum number of columns
pd.set_option("display.max_columns",50)
# maximum number of rows
pd.set_option("display.max_rows",500)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True

iteration = 'iteration1'


# read the molecular structures
REACH_short_term_fish_measurement = pd.read_excel(rf'data\fish_acute\processed\REACH_short_term_fish_measurement.xlsx')[['smiles', 'CAS number']].drop_duplicates()
REACH_short_term_fish_measurement.insert(1, 'source', 'short-term fish toxicity')
REACH_long_term_fish_measurement = pd.read_excel(rf'data\fish_chronic\processed\REACH_long_term_fish_measurement.xlsx')[['smiles', 'CAS number']].drop_duplicates()
REACH_long_term_fish_measurement.insert(1, 'source', 'long-term fish toxicity')
smiles_set = pd.concat([REACH_short_term_fish_measurement, REACH_long_term_fish_measurement], axis='index', sort=False, ignore_index=True).drop_duplicates()
smiles_set = smiles_set.groupby('smiles')[['source', 'CAS number']].aggregate(lambda x: ', '.join(sorted(set(x.dropna()))), ).reset_index()


# check and standardised the structures
tmp = smiles_set['smiles'].apply(standardise_check_molecule)
tmp = pd.json_normalize(tmp)
smiles_set = pd.concat([smiles_set, tmp.drop('smiles', axis='columns')], axis='columns')


# remove problematic structures
msk = smiles_set['smiles (standardised)'].notnull()
smiles_set = smiles_set.loc[msk]

# compute molecular weight, round to avoid issues with groupby below
smiles_set['molecular weight (standardised)'] = smiles_set['mol (standardised)'].apply(lambda mol: round(MolWt(mol), 4) if mol else None)

# compute the number of rings
smiles_set['number of rings (standardised)'] = smiles_set['mol (standardised)'].apply(lambda mol: mol.GetRingInfo().NumRings())

# create the output
smiles_uniq_set = (smiles_set.groupby(['smiles (standardised)', 'molecular weight (standardised)', 'number of rings (standardised)'])[['smiles', 'source', 'CAS number']].apply(lambda x: x.to_json(orient='records'))
                   .rename('source')
                   .reset_index().reset_index().rename(columns={'index': 'mol ID'}))

# create excel with two sheets for smiles_set and smiles_uniq_set
with pd.ExcelWriter(rf'data/structures/smiles.xlsx') as writer:
    smiles_set.drop('mol (standardised)', axis='columns').to_excel(writer, index=False, sheet_name='smiles (source)')
    smiles_uniq_set.to_excel(writer, index=False, sheet_name='smiles (standardised)')

# ACD batch input
smiles_uniq_set[['smiles (standardised)', 'mol ID']].to_csv(fr'data/structures/smiles_ACD_batch_input.txt', index=False, header=False, sep='\t')


# ecosar input (excel and sdf)
# .. excel input (in the output Number has the mol ID even though we did not specify it, it is necessary to have the mol ID reset)
tmp = smiles_uniq_set[['smiles (standardised)', 'mol ID']].rename({'smiles (standardised)': 'Smiles'}, axis='columns')
tmp = tmp.assign(**{'Log Kow': None, 'WS': None, 'MP': None})
# write in batches of 500 structures to avoid slow ecosar predictions
def create_ecosar_sdf(smiles_df, output_file):
    writer = Chem.SDWriter(output_file)
    for _, row in smiles_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles (standardised)'])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mol.SetProp('_MolFileV3000', '1')
        if mol:
            mol.SetProp('Log Kow', '')
            mol.SetProp('WS', '')
            mol.SetProp('MP', '')
            mol.SetProp('NAME', str(row['mol ID']))
            mol.SetProp('CAS', '')
            writer.write(mol)
    writer.close()
for range_start in range(0, len(tmp), 200):
    range_end = min(range_start + 200, len(tmp))
    batch_num = range_start // 200 + 1
    tmp.iloc[range_start:range_end][['Smiles']].to_csv(fr'data/structures/smiles_ecosar_input_part{batch_num}.txt', index=False, header=False, sep=' ')
    create_ecosar_sdf(smiles_uniq_set.iloc[range_start:range_end], fr'data/structures/smiles_ecosar_input_part{batch_num}.sdf')
    log.info(f'Wrote ecosar excel input for structures {range_start} to {range_end} in part {batch_num}')


# trident input
smiles_uniq_set['smiles (standardised)'].to_csv('data/structures/smiles_trident_input.txt', index=False, header=False, sep='\t')

# opera input
smiles_uniq_set[['smiles (standardised)', 'mol ID']].to_csv(fr'data/structures/smiles_opera_input.smi', index=False, header=False, sep='\t')

# vega input
smiles_uniq_set[['smiles (standardised)', 'mol ID']].to_csv(fr'data/structures/smiles_vega_input.smi', index=False, header=False, sep='\t')

# catalogic input, remove structures with molecular weight > 1100
tmp = smiles_uniq_set.copy()
msk = (smiles_uniq_set['molecular weight (standardised)'] > 1100)
print(f'number of structures with molecular weight > 1100: {msk.sum()}')
tmp['smiles (standardised)'] = np.where(msk, 'XX', smiles_uniq_set['smiles (standardised)'])
cols = ['mol ID', 'smiles (standardised)']
tmp = tmp[cols]
tmp.insert(loc=0,column='dummy',value=None)
tmp.columns = ['CAS NUM', 'NAME', 'SMILE']
tmp.to_csv(fr'data/structures/smiles_catalogic_input.smi', index=False, header=True, sep='\t')

# test input, remove structures with molecular weight > 1100 or more than 8 rings
tmp = smiles_uniq_set.copy()
msk = (smiles_uniq_set['molecular weight (standardised)'] > 1100) | (smiles_uniq_set['number of rings (standardised)'] > 8)
print(f'number of structures with molecular weight > 1100 or more than 8 rings: {msk.sum()}')
tmp['smiles (standardised)'] = np.where(msk, 'XX', smiles_uniq_set['smiles (standardised)'])
tmp = tmp[['smiles (standardised)']]
tmp.to_csv(fr'data/structures/smiles_test_input.smi', index=False, header=False, sep='\t')

# biodegradability app input (Kroener et al)
smiles_uniq_set[['mol ID', 'smiles (standardised)']].rename({'smiles (standardised)': 'smiles'}, axis='columns').to_csv(fr'data/structures/smiles_biodegradability_app_input.csv', index=False, header=True, sep=',')

# iSafeRat input
tmp = (smiles_uniq_set
       [['mol ID', 'smiles (standardised)']]
       .rename({'mol ID': 'Substance Number', 'smiles (standardised)': 'SMILES (mandatory)'}, axis='columns')
         .assign(**{'Melting Point (°C) (needed for Water Solubility, Ecotoxicity and Human Health predictions. If not given, it will be assumed to be less than 25°C (liquid substance at room temperature))': None,
                    'Boiling Point (°C) (needed for Vapour Pressure prediction, which in turn is needed for skin penetration and sensitisation predictions)': None,
                    'Density (mg/mL) (currently not needed by any models)': None,
                    'logKow input (optional)': None,
                    'Water Solubility input (mg/L) (optional)': None,
                    'Vapour Presure input (Pa) (optional)': None}))
tmp.insert(loc=1, column='Substance Name or other Identifier', value=None)
# make the column names the first row
tmp = pd.concat([pd.DataFrame([tmp.columns], columns=tmp.columns), tmp], axis='index', sort=False, ignore_index=True)
tmp.columns = range(len(tmp.columns))  # reset the column names to integers
tmp.to_excel(fr'data/structures/smiles_isaferat_input.xlsx', index=False, header=True)


# ISIDA input (sdf)
# .. excel input (in the output Number has the mol ID even though we did not specify it, it is necessary to have the mol ID reset)
tmp = smiles_uniq_set[['smiles (standardised)', 'mol ID']].rename({'smiles (standardised)': 'Smiles'}, axis='columns')
# write in batches of 500 structures to avoid slow ecosar predictions
def create_isida_sdf(smiles_df, output_file):
    writer = Chem.SDWriter(output_file)
    for _, row in smiles_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles (standardised)'])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mol.SetProp('_MolFileV3000', '1')
        writer.write(mol)
    writer.close()
for range_start in range(0, len(tmp), 100):
    range_end = min(range_start + 100, len(tmp))
    batch_num = range_start // 100 + 1
    create_ecosar_sdf(smiles_uniq_set.iloc[range_start:range_end], fr'data/structures/smiles_isida_input_part{batch_num}.sdf')
    log.info(f'Wrote ISIDA excel input for structures {range_start} to {range_end} in part {batch_num}')


# KATE input
tmp = smiles_uniq_set[['smiles (standardised)', 'mol ID']].rename({'smiles (standardised)': 'SMILES', 'mol ID': 'ID'}, axis='columns')
# remove stereochemistry for KATE as it fails otherwise
tmp['SMILES'] = tmp['SMILES'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False))
# replace structures that contain phosphonium, sulfonium, O+ or C+ with XX as KATE fails on them
exclude_smarts = Chem.MolFromSmarts("[#15+,#6+,#8+,#16+]")
excluded_structures = []
for smi in tmp['SMILES'].to_list():
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    matches = mol.GetSubstructMatches(exclude_smarts)
    if matches:
        excluded_structures.append(smi)
log.info(f'Number of structures containing phosphonium, sulfonium, O+ or C+: {len(excluded_structures)}')
tmp['SMILES'] = tmp['SMILES'].apply(lambda x: 'XX' if x in excluded_structures else x)
# write
for range_start in range(0, len(tmp), 250):
    range_end = min(range_start + 250, len(tmp))
    batch_num = range_start // 250 + 1
    tmp.iloc[range_start:range_end][['SMILES', 'ID']].to_csv(fr'data/structures/smiles_kate_input_part{batch_num}.txt', index=False, header=True, sep='\t')
    log.info(f'Wrote KATE excel input for structures {range_start} to {range_end} in part {batch_num}')



# TEST input
tmp = smiles_uniq_set[['smiles (standardised)', 'mol ID']].rename({'smiles (standardised)': 'Smiles'}, axis='columns')
for range_start in range(0, len(tmp), 500):
    range_end = min(range_start + 500, len(tmp))
    batch_num = range_start // 500 + 1
    tmp.iloc[range_start:range_end][['Smiles']].to_csv(fr'data/structures/smiles_test_input_part{batch_num}.txt', index=False, header=False, sep=' ')
    log.info(f'Wrote KATE excel input for structures {range_start} to {range_end} in part {batch_num}')



