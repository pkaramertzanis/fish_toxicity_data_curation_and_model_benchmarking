'''
Collect and process the T.E.S.T. model predictions for the test set.
Because some standaridised smiles in the input have been been set to XX to make predictions fail, we need to re-introduce the standardised smiles in the output using the mol ID.
'''
# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '01_collect_predictions_test', file_name ='logs/01_collect_predictions_test.log', level_stream=logging.INFO, level_file=logging.DEBUG)

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

# # read the original and standardised smiles
smiles_set = pd.read_excel(rf'data/structures/smiles.xlsx', sheet_name='smiles (standardised)')

for relax_fragment_constraint in ['relax_fragment_constraint_true', 'relax_fragment_constraint_false']:
    # read the training and validation sets
    training_validation_set = pd.read_excel(rf'data/training_validation_sets/test/test_training_validation_sets.xlsx')
    # .. check and standardise the training/validation set structures
    tmp = training_validation_set['smiles'].apply(standardise_check_molecule)
    tmp = pd.json_normalize(tmp)
    training_validation_set = pd.concat([training_validation_set, tmp.drop('smiles', axis='columns')], axis='columns')
    training_validation_set = (training_validation_set
                              [['model', 'study type', 'smiles (standardised)', 'training/validation set']]
                              .dropna(subset=['smiles (standardised)'])
                              .drop_duplicates()
                              )


    # read in the predictions
    predictions_consensus = (pd.read_excel(rf'data/predictions/test/{relax_fragment_constraint}/raw/Fathead_minnow_LC50_(96_hr)_Consensus.xlsx')
                             .assign(**{'mol ID': lambda df: df['Index'].astype(int)-1})
                             .merge(smiles_set[['mol ID', 'smiles (standardised)']], on='mol ID', how='left')
                             .rename({'Pred_Value:\nmg/L': 'prediction', 'Error': 'error'}, axis='columns')
                             .assign(**{'model name': f'''Fathead minnow LC50 (96 hr) consensus{' (RFC)' if relax_fragment_constraint == "relax_fragment_constraint_true" else ""}'''})
                             [['model name', 'mol ID', 'smiles (standardised)', 'prediction', 'error']])
    predictions_moa = (pd.read_excel(rf'data/predictions/test/{relax_fragment_constraint}/raw/Fathead_minnow_LC50_(96_hr)_Mode of action.xlsx')
                       .assign(**{'mol ID': lambda df: df['Index'].astype(int) - 1})
                       .merge(smiles_set[['mol ID', 'smiles (standardised)']], on='mol ID', how='left')
                       .rename({'Pred_Value:\nmg/L': 'prediction', 'Error': 'error', 'Experimental MOA': 'experimental MOA', 'Predicted MOA': 'predicted MOA', }, axis='columns')
                       .assign(**{'model name': f'''Fathead minnow LC50 (96 hr) MoA{' (RFC)' if relax_fragment_constraint == "relax_fragment_constraint_true" else ""}'''})
                       [[ 'model name', 'mol ID', 'smiles (standardised)', 'prediction', 'experimental MOA', 'predicted MOA', 'error']])
    predictions_test = pd.concat([predictions_consensus, predictions_moa])

    # set the columns platform
    predictions_test['platform'] = 'T.E.S.T.'

    # set the model version
    predictions_test['model version'] = '5.1.2'

    # mark the failed predictions
    predictions_test['prediction status'] = np.where(predictions_test['prediction'].isnull(), 'failed', 'succeeded')

    # set the study type
    predictions_test['study type'] = 'acute'

    # initiate the notes dictionary
    predictions_test['notes'] = [dict() for _ in range(len(predictions_test))]


    # mark the predictions that are in the training/validation set
    training_smiles_acute_consensus = (training_validation_set
                             .query(f'''`model`=="Fathead minnow LC50 (96 hr) consensus{' (RFC)' if relax_fragment_constraint == "relax_fragment_constraint_true" else ""}"''')
                             .query('`training/validation set`=="training set"')['smiles (standardised)'])
    validation_smiles_acute_consensus = (training_validation_set
                               .query(f'''`model`=="Fathead minnow LC50 (96 hr) consensus{' (RFC)' if relax_fragment_constraint == "relax_fragment_constraint_true" else ""}"''')
                               .query('`training/validation set`=="validation set"')['smiles (standardised)'])
    training_smiles_acute_moa = (training_validation_set
                                .query(f'''`model`=="Fathead minnow LC50 (96 hr) MoA{' (RFC)' if relax_fragment_constraint == "relax_fragment_constraint_true" else ""}"''')
                                .query('`training/validation set`=="training set"')['smiles (standardised)'])
    validation_smiles_acute_moa = (training_validation_set
                                  .query(f'''`model`=="Fathead minnow LC50 (96 hr) MoA{' (RFC)' if relax_fragment_constraint == "relax_fragment_constraint_true" else ""}"''')
                                  .query('`training/validation set`=="validation set"')['smiles (standardised)'])
    predictions_test['training/validation set'] = np.select(condlist=[predictions_test['smiles (standardised)'].isin(training_smiles_acute_consensus) & (predictions_test['model name'] == 'Fathead minnow LC50 (96 hr) consensus'),
                                                                      predictions_test['smiles (standardised)'].isin(validation_smiles_acute_consensus) & (predictions_test['model name'] == 'Fathead minnow LC50 (96 hr) consensus'),
                                                                      predictions_test['smiles (standardised)'].isin(training_smiles_acute_moa) & (predictions_test['model name'] == 'Fathead minnow LC50 (96 hr) MoA'),
                                                                      predictions_test['smiles (standardised)'].isin(validation_smiles_acute_moa) & (predictions_test['model name'] == 'Fathead minnow LC50 (96 hr) MoA')
                                                                     ],
                                                            choicelist=['training set', 'validation set', 'training set', 'validation set'],
                                                            default='not in training/validation set')

    # set the applicability domain
    predictions_test['AD'] = np.where(predictions_test['prediction status'] == 'succeeded', 'in domain', None)

    # set the predicted quantity
    predictions_test['predicted quantity'] = 'LC50 (mg/L)'

    # set the no effects at saturation column
    predictions_test['no effects at saturation'] = np.where(predictions_test['prediction status'] == 'succeeded', 'no', None)

    # put the important columns first
    first_cols = ['platform', 'model name', 'model version', 'study type', 'mol ID', 'smiles (standardised)',
                  'prediction status', 'training/validation set', 'AD', 'predicted quantity', 'prediction', 'no effects at saturation', 'notes']
    new_order = first_cols + [col for col in predictions_test.columns if col not in first_cols]
    predictions_test = predictions_test[new_order]


    # store the processed predictions
    predictions_test.to_excel(rf'data/predictions/test/{relax_fragment_constraint}/processed/predictions_test.xlsx', index=False)



