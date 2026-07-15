'''
Collect and process the KATE model predictions for the test set.
Because some standaridised smiles in the input have been been set to XX to make predictions fail, we need to re-introduce the standardised smiles in the output using the mol ID.
'''
# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '01_collect_predictions_kate', file_name ='logs/01_collect_predictions_kate.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np
import numpy as np
import re
import math
import chardet

from pathlib import Path

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


# smiles_test = (pd.read_csv(rf'data/structures/smiles_test_input.smi', header=None)
#                .reset_index()
#                .rename({'index': 'mol ID', 0: 'smiles (standardised)'}, axis='columns')
#                [['mol ID', 'smiles (standardised)']])



# read the training and validation sets
training_validation_set = pd.read_excel(rf'data/training_validation_sets/kate/kate_qsar_reference_chemicals.xlsx')
msk = training_validation_set['Organism'] == 'Fish'
training_validation_set = (training_validation_set.loc[msk]
                           .rename({'Acute or Chronic': 'study type', 'SMILES': 'smiles'}, axis='columns')
                           .assign(**{'training/validation set': 'training set'})
                           .dropna(subset='smiles')
                           .reset_index()
                           )
# .. check and standardise the training/validation set structures
tmp = training_validation_set['smiles'].apply(standardise_check_molecule)
tmp = pd.json_normalize(tmp)
training_validation_set = pd.concat([training_validation_set, tmp.drop('smiles', axis='columns')], axis='columns')
training_validation_set = (training_validation_set
                          [['study type', 'smiles (standardised)', 'training/validation set']]
                          .dropna(subset=['smiles (standardised)'])
                          .drop_duplicates()
                          )


# read the predictions
files = Path(r'data/predictions/kate/raw/').glob('*.tsv')
predictions_kate = []
for file in files:
    log.info(f'Reading prediction file: {file}')
    predictions_kate.append(pd.read_csv(file, sep='\t'))
predictions_kate = pd.concat(predictions_kate, axis='index', ignore_index=True, sort=False)
predictions_kate = predictions_kate.rename({'ID': 'mol ID', }, axis='columns')
# .. keep only fish predictions
msk = predictions_kate['Organism'] == 'Fish'
predictions_kate = predictions_kate.loc[msk].reset_index(drop=True)
predictions_kate['study type'] = predictions_kate['Acute or Chronic'].str.lower()
log.info(f'Number of predictions from Kate\'s model: {len(predictions_kate)}, {predictions_kate["mol ID"].nunique()} unique molecules')



# add the standardised smiles, this will ensure that standardised smiles with XX to make the predictions will now be restored
predictions_kate = predictions_kate.merge(smiles_set[['smiles (standardised)', 'mol ID']], on='mol ID')
# predictions_kate.to_excel('junk/debug_all_predictions_kate.xlsx', index=False)

# mark the failed predictions and add rows for which there was no prediction
predictions_kate = (smiles_set
                      .merge(pd.DataFrame({'study type': ['acute', 'chronic']}), how='cross')
                      .merge(predictions_kate, on=['mol ID', 'smiles (standardised)', 'study type'], how='left')
                      )
predictions_kate['prediction status'] = np.where(predictions_kate['Predicted Toxicity'].notnull(), 'succeeded', 'failed')



# set the prediction
predictions_kate['prediction'] = predictions_kate['Predicted Toxicity']

# add the columns platform, model name, version, predicted quantity
predictions_kate = (predictions_kate
                    .assign(**{'platform': 'KATE'})
                    .assign(**{'model version': '1.1'})
                    .assign(**{'model name': np.select(condlist=[predictions_kate['study type']=='acute',
                                                                  predictions_kate['study type']=='chronic'],
                                                       choicelist=['Fish Acute Toxicity model',
                                                                   'Fish Chronic Toxicity model'], default='other')})
                     .assign(**{'predicted quantity': np.select(condlist=[predictions_kate['study type']=='acute',
                                                                          predictions_kate['study type']=='chronic'],
                                                                 choicelist=['LC50 (mg/L)',
                                                                             'ChV (mg/L)'], default='other')})
                     )


# initiate the notes dictionary
predictions_kate['notes'] = [dict() for _ in range(len(predictions_kate))]

# mark the predictions in the applicability domain
msk = (predictions_kate['log P Judgement'] == 'in') & (predictions_kate['Structure Judgement'].isin(['in', 'in(p)'])) & (predictions_kate['criteria'] == 'yes') & (predictions_kate['prediction status'] == 'succeeded')
predictions_kate['AD'] = np.select(condlist=[msk, predictions_kate['prediction status'] == 'succeeded'],
                                   choicelist=['in domain', 'out of domain'],
                                   default=None)

# mark whether the molecule is in the training/validation set
training_smiles_acute = (training_validation_set
                   .query('`study type`=="Acute"')
                   .query('`training/validation set`=="training set"')['smiles (standardised)'])
training_smiles_chronic = (training_validation_set
                   .query('`study type`=="Chronic"')
                   .query('`training/validation set`=="training set"')['smiles (standardised)'])
predictions_kate['training/validation set'] = np.select(condlist=[predictions_kate['smiles (standardised)'].isin(training_smiles_acute) & (predictions_kate['study type'] == 'acute'),
                                                                  predictions_kate['smiles (standardised)'].isin(training_smiles_chronic) & (predictions_kate['study type'] == 'chronic'),
                                                                      ],
                                                         choicelist=['training set', 'training set',],
                                                         default='not in training/validation set')

# set the no effects at saturation
predictions_kate['no effects at saturation'] = np.where(predictions_kate['prediction status'] == 'succeeded', 'no', None)



# put the important columns first
first_cols = ['platform', 'model name', 'model version', 'study type', 'mol ID', 'smiles (standardised)',
              'prediction status', 'training/validation set', 'AD', 'predicted quantity', 'prediction', 'no effects at saturation', 'notes']
new_order = first_cols + [col for col in predictions_kate.columns if col not in first_cols]
predictions_kate = predictions_kate[new_order]


# store the processed predictions
predictions_kate.to_excel(rf'data/predictions/kate/processed/predictions_kate.xlsx', index=False)
