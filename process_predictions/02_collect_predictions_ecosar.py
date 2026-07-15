# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '02_collect_predictions_ecosar', file_name ='logs/02_collect_predictions_ecosar.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np
import math
import chardet
from pathlib import Path
import re
import itertools
import json

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
pd.set_option("max_colwidth", 120)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True

# read the original and standardised smiles input and predictions in order to set the mol ID correctly
files_input = sorted(list(Path(rf'data\structures').glob('smiles_ecosar_input*.txt')), key=lambda f: int(re.search(r'smiles_ecosar_input_part(\d+)\.txt', f.name).group(1)))
files_predictions = sorted(list(Path(rf'data\predictions\ecosar\raw').glob('ecosar_predictions_batch*.xlsx')), key=lambda f: int(re.search(r'ecosar_predictions_batch(\d+)\.xlsx', f.name).group(1)))
smiles_ecosar = []
predictions_ecosar = []
n_molecules = 0
for f_input, f_prediction in zip(files_input, files_predictions):
    log.info(f"Reading ecosar smiles from file: {f_input}")
    # read the input smiles
    tmp = pd.read_csv(f_input, sep=' ', header=None).rename({0: 'smiles (standardised)'}, axis='columns')
    smiles_ecosar.append(tmp)
    n_batch = len(tmp)
    # read the predictions
    log.info(f"Reading ecosar predictions from file: {f_prediction}")
    tmp = (pd.read_excel(f_prediction)
           .query('SMILES.notnull()')
           .assign(**{'mol ID': lambda df: df['Number'] + n_molecules})
           .drop(['CAS', 'Chemical', 'SMILES'], axis='columns')
           .drop_duplicates())
    n_molecules += n_batch
    predictions_ecosar.append(tmp)
smiles_ecosar = (pd.concat(smiles_ecosar, axis='index', ignore_index=True, sort=False)
                 .reset_index()
                 .rename({'index': 'mol ID'}, axis='columns')
                 [['mol ID', 'smiles (standardised)']])
predictions_ecosar = pd.concat(predictions_ecosar, axis='index', ignore_index=True, sort=False)
predictions_ecosar['ECOSAR Class'] = predictions_ecosar['ECOSAR Class'].str.strip()

# keep only the fish acute and chronic toxicity predictions
msk_fish_acute = (predictions_ecosar['Organism'] == 'Fish') & ((predictions_ecosar['End Point'] == 'LC50') & (predictions_ecosar['Duration'] == '96h'))
msk_fish_chronic = (predictions_ecosar['Organism'] == 'Fish') & (predictions_ecosar['End Point'] == 'ChV')
predictions_ecosar = (predictions_ecosar
                      .assign(**{'study type': lambda df: np.select(condlist=[msk_fish_acute, msk_fish_chronic],
                                                             choicelist=['acute', 'chronic'], default='other')})
                      .loc[msk_fish_acute | msk_fish_chronic]
                      .reset_index(drop=True)
                      )

# add the smiles (standardised) to the predictions
predictions_ecosar = (predictions_ecosar
                      .merge(smiles_ecosar, on='mol ID', how='left'))

# mark the failed predictions and add rows for which there was no prediction
predictions_ecosar = (smiles_ecosar
                      .merge(pd.DataFrame({'study type': ['acute', 'chronic']}), how='cross')
                      .merge(predictions_ecosar, on=['mol ID', 'smiles (standardised)', 'study type'], how='left')
                      )
predictions_ecosar['prediction status'] = np.where(predictions_ecosar['Number'].notnull(),
                                                   'succeeded', 'failed')
# predictions_ecosar.to_excel('data/predictions/ecosar/processed/debug_all_predictions_ecosar.xlsx', index=False)


# add the columns platform, model name, version, predicted quantity
predictions_ecosar = (predictions_ecosar
                     .assign(**{'platform': 'ECOSAR'})
                     .assign(**{'model version': '2.2'})
                     .assign(**{'model name': np.select(condlist=[predictions_ecosar['study type']=='acute',
                                                                  predictions_ecosar['study type']=='chronic'],
                                                       choicelist=['Fish Acute Toxicity model',
                                                                   'Fish Chronic Toxicity model'], default='other')})
                     .assign(**{'predicted quantity': np.select(condlist=[predictions_ecosar['study type']=='acute',
                                                                          predictions_ecosar['study type']=='chronic'],
                                                                 choicelist=['LC50 (mg/L)',
                                                                             'ChV (mg/L)'], default='other')})
                     )

# initiate the notes dictionary
predictions_ecosar['notes'] = [dict() for _ in range(len(predictions_ecosar))]

# create column ECOSAR class (only letters)
predictions_ecosar = (predictions_ecosar
                      .assign(**{'ECOSAR class (only letters)': lambda df: df['ECOSAR Class'].str.lower().str.replace(r'[^a-z]', '', regex=True)})
                      )

# read the training/validation set, training_validation_set in the end has the columns ['smiles (standardised)', 'training/validation set']
training_validation_set = pd.read_excel(r'data/training_validation_sets/ecosar/ecosar_substance_list_detailed_structures.xlsx')
training_validation_set['class definition name'] = training_validation_set['class definition name'].str.strip()
# .. keep only the acute and chronic fish toxicity training sets
msk_SAR_models_acute = (training_validation_set['SAR model'].str.contains(r'(?i)FISH.*(?:96.*h)', na=False)
                      & ~(training_validation_set['SAR model'].str.contains(r'SW', na=False)))
# training_validation_set['SAR model'].loc[msk_SAR_models_acute].drop_duplicates()
msk_SAR_models_chronic = (training_validation_set['SAR model'].str.contains(r'(?i)FISH.*(?:ChV)', na=False)
                      & ~(training_validation_set['SAR model'].str.contains(r'SW', na=False)))
training_validation_set['study type'] = np.select(condlist=[msk_SAR_models_acute, msk_SAR_models_chronic],
                                                 choicelist=['acute', 'chronic'], default='other')
# .. extract the number of chemicals per class definition to be used later for the applicability domain
number_of_chemicals_per_ecosar_class = (training_validation_set[['class definition name', 'study type', 'number of chemicals']]
                                        .loc[msk_SAR_models_acute | msk_SAR_models_chronic]
                                        .drop_duplicates()
                                        .assign(**{'number of chemicals (numerical)': lambda df: pd.to_numeric(df['number of chemicals'].fillna('').str.extract(r'^(\d+)')[0], errors='coerce')})
                                        .assign(**{'class definition name (only letters)': lambda df: df['class definition name'].str.lower().str.replace(r'[^a-z]', '', regex=True)})
                                        # correct some class definition names to match those in the predictions
                                        .replace({'class definition name (only letters)': {'vinyallylpropargylalcoholsunhindered': 'vinylallylpropargylalcoholsunhindered',
                                                                                           'vinylallylpropargyketones': 'vinylallylpropargylketones',
                                                                                           'vinylallylpropargyesters': 'vinylallylpropargylesters',
                                                                                           'estersphosphatesinertsubstitution': 'estersphosphatesinertsubstitutions',
                                                                                           'estersphosphateswithdrawingsubstitution': 'estersphosphatewithdrawingsubstitutions',
                                                                                           'thiocarbamatesdifreeacid': 'thiocarbamatesdifreeacids',
                                                                                           'thiocarbamatesdisubstit': 'thiocarbamatesdisubstituted'
                                                                                           }})
                                        )
# .. keep only rows for which the SMILES is present, of more than one SMILES, keep them all even though at least one structure will wrong
msk_smiles_present = training_validation_set['SMILES'].notnull()
training_validation_set = (training_validation_set
       .loc[msk_smiles_present & (msk_SAR_models_acute | msk_SAR_models_chronic), ['class definition name', 'study type', 'SMILES']]
       .assign(SMILES=lambda df: df['SMILES'].str.split(',')).explode('SMILES')
       .drop_duplicates()
       .reset_index(drop=True)
       .assign(**{'training/validation set': 'training set'})
      )
# .. check and standardise the training/validation set structures
tmp = training_validation_set['SMILES'].apply(standardise_check_molecule)
tmp = pd.json_normalize(tmp)
training_validation_set = pd.concat([training_validation_set, tmp.drop('smiles', axis='columns')], axis='columns')
training_validation_set = (training_validation_set
                          .dropna(subset=['smiles (standardised)'])
                          [['class definition name', 'study type', 'smiles (standardised)']]
                          .drop_duplicates()
                          )


# mark the predictions that are in the training/validation set
smiles_std_training_acute = training_validation_set.loc[training_validation_set['study type']=='acute', 'smiles (standardised)'].unique()
msk_fish_acute = (predictions_ecosar['study type'] == 'acute') & (predictions_ecosar['smiles (standardised)'].isin(smiles_std_training_acute))
smiles_std_training_chronic = training_validation_set.loc[training_validation_set['study type']=='chronic', 'smiles (standardised)'].unique()
msk_fish_chronic = (predictions_ecosar['study type'] == 'chronic') & (predictions_ecosar['smiles (standardised)'].isin(smiles_std_training_chronic))
predictions_ecosar['training/validation set'] = np.select(condlist=[msk_fish_acute, msk_fish_chronic],
                                                             choicelist=['training set', 'training set'], default='not in training/validation set')
# add the number of chemicals per ecosar class to the predictions
predictions_ecosar = (predictions_ecosar
                        .merge(number_of_chemicals_per_ecosar_class[['class definition name (only letters)', 'study type', 'number of chemicals (numerical)']],
                               left_on=['ECOSAR class (only letters)', 'study type'],
                               right_on=['class definition name (only letters)', 'study type'],
                               how='left')
                      .fillna({'number of chemicals (numerical)': 0})
                      .astype({'number of chemicals (numerical)': 'Int64'})
                     )


# mark the predictions for which there are no effects at saturation
predictions_ecosar['no effects at saturation'] = np.select(condlist=[predictions_ecosar['Alert'].str.contains(r'(?i)SaturateSolublity', na=False) & (predictions_ecosar['prediction status'] == 'succeeded'),
                                                                     (predictions_ecosar['prediction status'] == 'succeeded')],
                                                           choicelist=['yes', 'no'], default=None)

# set the prediction
predictions_ecosar['prediction'] = predictions_ecosar['Concentration (mg/L)']

# set the domain
msk_acute_to_chronic_ratio = predictions_ecosar['Alert'].str.contains(r'(?i)AcuteToChronicRatios', na=False) & (predictions_ecosar['prediction status'] == 'succeeded') & (predictions_ecosar['study type'] == 'chronic')
predictions_ecosar['notes'] = np.where(msk_acute_to_chronic_ratio,
                                [{**d, "AD reasoning 1": 'acute to chronic ratio'} for d in predictions_ecosar["notes"]],
                                predictions_ecosar["notes"])
msk_logKow_out_of_domain = predictions_ecosar['Alert'].str.contains(r'(?i)LogKowCutOff', na=False) & (predictions_ecosar['prediction status'] == 'succeeded')
predictions_ecosar['notes'] = np.where(msk_logKow_out_of_domain,
                                [{**d, "AD reasoning 2": 'exceeded Log Kow maximum value'} for d in predictions_ecosar["notes"]],
                                predictions_ecosar["notes"])
msk_model_with_5_or_more_chemicals = predictions_ecosar['number of chemicals (numerical)'] >= 5
predictions_ecosar['notes'] = np.where(~msk_model_with_5_or_more_chemicals & (predictions_ecosar['prediction status'] == 'succeeded'),
                                [{**d, "AD reasoning 3": 'model trained with less than 5 chemicals'} for d in predictions_ecosar["notes"]],
                                predictions_ecosar["notes"])
msk_domain_of_applicability = predictions_ecosar['Alert'].str.contains(r'(?i)DomainOfApplicability', na=False) & (predictions_ecosar['prediction status'] == 'succeeded')
predictions_ecosar['notes'] = np.where(msk_domain_of_applicability,
                                [{**d, "AD reasoning 4": 'domain of applicability, e.g. MW>1000'} for d in predictions_ecosar["notes"]],
                                predictions_ecosar["notes"])




predictions_ecosar['AD'] = np.select(condlist=[msk_model_with_5_or_more_chemicals & ~msk_acute_to_chronic_ratio & ~msk_logKow_out_of_domain & ~msk_domain_of_applicability,
                                              predictions_ecosar['prediction status'] == 'succeeded'],
                                     choicelist=['in domain', 'out of domain'], default=None)

# put the important columns first
first_cols = ['platform', 'model name', 'model version', 'study type', 'mol ID', 'smiles (standardised)',
              'prediction status', 'training/validation set', 'AD', 'predicted quantity', 'prediction', 'no effects at saturation', 'notes']
new_order = first_cols + [col for col in predictions_ecosar.columns if col not in first_cols]
predictions_ecosar = predictions_ecosar[new_order]


# dump the notes column separately as json
predictions_ecosar['notes'] = predictions_ecosar['notes'].apply(json.dumps)

# store the processed predictions
predictions_ecosar.to_excel(rf'data/predictions/ecosar/processed/predictions_ecosar.xlsx', index=False)

# predictions_ecosar.pivot_table(index=['model name'], columns=['prediction status', 'AD', 'training/validation set'], values='mol ID', aggfunc='nunique', fill_value=0, dropna=False).to_clipboard()




predictions_ecosar.groupby('smiles (standardised)')['ECOSAR Class'].agg(lambda x: ', '.join(sorted(set(x.dropna())))).value_counts()