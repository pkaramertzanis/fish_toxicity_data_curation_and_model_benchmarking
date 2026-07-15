# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '01_collect_predictions_isaferat', file_name ='logs/01_collect_predictions_isaferat.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np
import numpy as np
import re
import math
import chardet

from cheminformatics.standardise_check import standardise_check_molecule


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

def process_CI_domain(text):
    '''
    Process the confidence interval and applicability domain text from iSafeRat predictions
    :param text: input text
    :return: Series with the extracted values for the columns:
        '95% Pred Interval lower' (float)
        '95% Pred Interval upper' (float)
        'Structural domain' (str)
        'Overall AD' (str)
    '''
    if pd.isna(text):
        return pd.Series({'95% Pred Interval lower': None, '95% Pred Interval upper': None,
                          'Structural domain': None, 'Overall AD': None})
    pattern = re.compile(
        r'Pred Interval:\s*(?P<lower>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*-\s*'
        r'(?P<upper>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        r';\s*Structural domain:\s*(?P<struct_domain>\w+)'
        r';\s*Overall AD:\s*(?P<overall_ad>\w+)'
    )
    match = pattern.search(text)
    if match:
        return pd.Series({'95% Pred Interval lower': float(match.group('lower')),
                          '95% Pred Interval upper': float(match.group('upper')),
                          'Structural domain': match.group('struct_domain'),
                          'Overall AD': match.group('overall_ad')})
    else:
        return pd.Series({'95% Pred Interval lower': None, '95% Pred Interval upper': None,
                          'Structural domain': None, 'Overall AD': None})

# read the original and standardised smiles
smiles_isaferat = (pd.read_excel(rf'data/structures/smiles_isaferat_input.xlsx', skiprows=1)
                   .rename({'Substance Number': 'mol ID', 'SMILES (mandatory)': 'smiles (standardised)'}, axis='columns')
                   [['mol ID', 'smiles (standardised)']])


# mark the structures that need to be put out of domain because of a bug, these are phenyl esters
from rdkit import Chem
def filter_matching_smiles(smiles_list):
    patterns = [Chem.MolFromSmarts(pat) for pat in [f'[#1,#6]C(=O)Oc']]
    matches = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for patt in patterns:
            if mol.HasSubstructMatch(patt):
                matches.append(smi)
                break
    return matches
structures_to_be_put_out_of_domain = filter_matching_smiles(smiles_isaferat['smiles (standardised)'].to_list())
mol_IDs_to_be_put_out_of_domain = (smiles_isaferat
                                  .loc[smiles_isaferat['smiles (standardised)'].isin(structures_to_be_put_out_of_domain), 'mol ID']
                                  .to_list())


# read the training and validation sets
training_validation_set = pd.read_excel(rf'data/training_validation_sets/isaferat/isaferat_training_validation_sets.xlsx')
msk = training_validation_set['SMILES'].notnull() & (training_validation_set['taxonomic_group']=='fish')
training_validation_set = training_validation_set.loc[msk].reset_index(drop=True)
# .. check and standardise the training/validation set structures
tmp = training_validation_set['SMILES'].apply(standardise_check_molecule)
tmp = pd.json_normalize(tmp)
training_validation_set = pd.concat([training_validation_set, tmp.drop('smiles', axis='columns')], axis='columns')
training_validation_set = (training_validation_set
                          [['sheet_name', 'acute/chronic', 'smiles (standardised)', 'training/validation set']]
                          .drop_duplicates()
                          )


for option in ['AD_no_extrapolation', 'AD_extrapolation_all', 'AD_extrapolation_structural_in_domain']:


    # read the isaferat predictions, with and without the opera MP
    cols_common = ['Substance Number',
            'iSafeRat® Mechanisms of Action (MechoA Premium v1.3)',
            'CLASS Applicability Domain', 'Errors', 'Warnings'
            ]
    cols_acute = ['iSafeRat® 96h-LC50 to fish v2.0', '96h-LC50 to fish Confidence Interval and Applicability Domain']
    cols_chronic = ['iSafeRat® 32d-EC10 to fish v2.1', '32d-EC10 to fish Confidence Interval and Applicability Domain']
    predictions_acute_isaferat_without_mp = (pd.read_excel(rf'data/predictions/isaferat/raw/without_MP/smiles_isaferat_output.xlsx', skiprows=1, usecols=cols_common+cols_acute)
                                       .rename({'Substance Number': 'mol ID',
                                                'iSafeRat® 96h-LC50 to fish v2.0': 'prediction',
                                                '96h-LC50 to fish Confidence Interval and Applicability Domain': 'confidence interval and applicability domain'
                                                }, axis='columns')
                                       .assign(**{'model name': '96h-LC50 to fish (no MP)',
                                                  'model version': '2.0',
                                                  'predicted quantity': 'LC50 (mg/L)',
                                                  'study type': 'acute'})
                                       )
    predictions_acute_isaferat_with_mp = (pd.read_excel(rf'data/predictions/isaferat/raw/with_opera_MP/smiles_isaferat_output.xlsx', skiprows=1, usecols=cols_common+cols_acute)
                                       .rename({'Substance Number': 'mol ID',
                                                'iSafeRat® 96h-LC50 to fish v2.0': 'prediction',
                                                '96h-LC50 to fish Confidence Interval and Applicability Domain': 'confidence interval and applicability domain'
                                                }, axis='columns')
                                       .assign(**{'model name': '96h-LC50 to fish (Opera MP)',
                                                  'model version': '2.0',
                                                  'predicted quantity': 'LC50 (mg/L)',
                                                  'study type': 'acute'})
                                       )
    predictions_chronic_isaferat_without_mp = (pd.read_excel(rf'data/predictions/isaferat/raw/without_MP/smiles_isaferat_output.xlsx', skiprows=1, usecols=cols_common+cols_chronic)
                                       .rename({'Substance Number': 'mol ID',
                                                'iSafeRat® 32d-EC10 to fish v2.1': 'prediction',
                                                '32d-EC10 to fish Confidence Interval and Applicability Domain': 'confidence interval and applicability domain'
                                                }, axis='columns')
                                       .assign(**{'model name': '32d-EC10 to fish (no MP)',
                                                  'model version': '2.1',
                                                  'predicted quantity': 'EC10 (mg/L)',
                                                  'study type': 'chronic'})
                                       )
    predictions_chronic_isaferat_with_mp = (pd.read_excel(rf'data/predictions/isaferat/raw/with_opera_MP/smiles_isaferat_output.xlsx', skiprows=1, usecols=cols_common+cols_chronic)
                                       .rename({'Substance Number': 'mol ID',
                                                'iSafeRat® 32d-EC10 to fish v2.1': 'prediction',
                                                '32d-EC10 to fish Confidence Interval and Applicability Domain': 'confidence interval and applicability domain'
                                                }, axis='columns')
                                       .assign(**{'model name': '32d-EC10 to fish (Opera MP)',
                                                  'model version': '2.1',
                                                  'predicted quantity': 'EC10 (mg/L)',
                                                  'study type': 'chronic'})
                                       )
    predictions_isaferat = pd.concat([smiles_isaferat.merge(predictions_acute_isaferat_without_mp, on='mol ID', how='left'),
                             smiles_isaferat.merge(predictions_acute_isaferat_with_mp, on='mol ID', how='left'),
                             smiles_isaferat.merge(predictions_chronic_isaferat_without_mp, on='mol ID', how='left'),
                             smiles_isaferat.merge(predictions_chronic_isaferat_with_mp, on='mol ID', how='left')], axis='index', ignore_index=True, sort=False)

    # set the columns platform
    predictions_isaferat['platform'] = 'iSafeRat'

    # mark the failed predictions
    predictions_isaferat['prediction status'] = np.where(predictions_isaferat['prediction'] == 'N/D', 'failed', 'succeeded')

    # initiate the notes dictionary
    predictions_isaferat['notes'] = [dict() for _ in range(len(predictions_isaferat))]


    # mark the predictions that are in the training/validation set
    training_smiles_acute = (training_validation_set
                       .query('`acute/chronic`=="acute"')
                       .query('`training/validation set`=="training set"')['smiles (standardised)'])
    validation_smiles_acute = (training_validation_set
                       .query('`acute/chronic`=="acute"')
                       .query('`training/validation set`=="validation set"')['smiles (standardised)'])
    training_smiles_chronic = (training_validation_set
                       .query('`acute/chronic`=="chronic"')
                       .query('`training/validation set`=="training set"')['smiles (standardised)'])
    validation_smiles_chronic = (training_validation_set
                       .query('`acute/chronic`=="chronic"')
                       .query('`training/validation set`=="validation set"')['smiles (standardised)'])
    predictions_isaferat['training/validation set'] = np.select(condlist=[predictions_isaferat['smiles (standardised)'].isin(training_smiles_acute) & (predictions_isaferat['study type'] == 'acute'),
                                                                          predictions_isaferat['smiles (standardised)'].isin(validation_smiles_acute) & (predictions_isaferat['study type'] == 'acute'),
                                                                          predictions_isaferat['smiles (standardised)'].isin(training_smiles_chronic) & (predictions_isaferat['study type'] == 'chronic'),
                                                                          predictions_isaferat['smiles (standardised)'].isin(validation_smiles_chronic) & (predictions_isaferat['study type'] == 'chronic')
                                                                          ],
                                                             choicelist=['training set', 'validation set', 'training set', 'validation set'],
                                                             default='not in training/validation set')


    # mark the predictions for which there are no effects at saturation
    msk = predictions_isaferat['prediction'] == 'toxicity > solubility limit'
    predictions_isaferat['no effects at saturation'] = np.select(condlist=[~msk & (predictions_isaferat['prediction status'] == 'succeeded'),
                                                                            msk],
                                                                 choicelist=['no', 'yes'],
                                                                 default=None)
    predictions_isaferat['prediction'] = np.where(msk, None, predictions_isaferat['prediction'])


    # set the domain
    predictions_isaferat['overall AD'] = predictions_isaferat['confidence interval and applicability domain'].str.extract(r'Overall AD:\s*(\w+)', expand=False)
    predictions_isaferat['structural AD'] = predictions_isaferat['confidence interval and applicability domain'].str.extract(r'Structural domain:\s*(\w+)', expand=False)

    if option == 'AD_no_extrapolation':
        predictions_isaferat['AD'] = np.select(condlist=[(predictions_isaferat['prediction status'] == 'succeeded') & (predictions_isaferat['overall AD'] == 'Inside'),
                                                         (predictions_isaferat['prediction status'] == 'succeeded') & (predictions_isaferat['overall AD'] == 'Extrapolated')],
                                               choicelist=['in domain', 'out of domain'],
                                               default=None)
        predictions_isaferat['model name'] = predictions_isaferat['model name'] + ', extr. AD excluded'
    elif option  == 'AD_extrapolation_all':
        predictions_isaferat['AD'] = np.select(condlist=[(predictions_isaferat['prediction status'] == 'succeeded') & (predictions_isaferat['overall AD'] == 'Inside'),
                                                         (predictions_isaferat['prediction status'] == 'succeeded') & (predictions_isaferat['overall AD'] == 'Extrapolated')],
                                               choicelist=['in domain', 'in domain'],
                                               default=None)
        predictions_isaferat['model name'] = predictions_isaferat['model name'] + ', extr. AD included'
    elif option  == 'AD_extrapolation_structural_in_domain':
        predictions_isaferat['AD'] = np.select(condlist=[(predictions_isaferat['prediction status'] == 'succeeded') & (predictions_isaferat['overall AD'] == 'Inside'),
                                                         (predictions_isaferat['prediction status'] == 'succeeded') & ((predictions_isaferat['overall AD'] == 'Extrapolated') & (predictions_isaferat['structural AD'] == 'Inside')),
                                                         (predictions_isaferat['prediction status'] == 'succeeded') & ((predictions_isaferat['overall AD'] == 'Extrapolated') & (predictions_isaferat['structural AD'] != 'Inside'))
                                                         ],
                                               choicelist=['in domain', 'in domain', 'out of domain'],
                                               default=None)
        predictions_isaferat['model name'] = predictions_isaferat['model name'] + ', extr. AD included (str. AD in domain)'
    # .. overrule the prediction status for the structures that need to be put out of domain because of a bug, we indicate that these structures are failed
    msk = predictions_isaferat['mol ID'].isin(mol_IDs_to_be_put_out_of_domain) & (predictions_isaferat['prediction status'] == 'succeeded')
    log.info(f'Number of structures indicated as failed due to known iSafeRat bug for phenyl esters: {msk.sum()}')
    predictions_isaferat.loc[msk, 'prediction status'] = 'failed'
    predictions_isaferat.loc[msk, ['prediction', 'no effects at saturation', 'AD']] = None
    for idx in predictions_isaferat.loc[msk].index:
        predictions_isaferat.at[idx, 'notes']['AD adjustment'] = 'put out of domain due to known iSafeRat bug for phenyl esters'
    # put the important columns first
    first_cols = ['platform', 'model name', 'model version', 'study type', 'mol ID', 'smiles (standardised)',
                  'prediction status', 'training/validation set', 'AD', 'predicted quantity', 'prediction', 'no effects at saturation', 'notes']
    new_order = first_cols + [col for col in predictions_isaferat.columns if col not in first_cols]
    predictions_isaferat = predictions_isaferat[new_order]


    # store the processed predictions
    predictions_isaferat.to_excel(rf'data/predictions/isaferat/processed/predictions_isaferat_{option}.xlsx', index=False)



