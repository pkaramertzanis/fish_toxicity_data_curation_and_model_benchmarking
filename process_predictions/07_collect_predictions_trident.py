'''
Collect and process the Trident model predictions for the test set.

We did not use the model's training set but relied on the model output to identify structures that were in the training/validation set.

We used the combined EC10/50 model for fish and predicted:
- acute toxicity at 96h (EC50)
- chronic toxicity at 720h (30d) for mortality (EC10)
- chronic toxicity at 720h (30d) for growth (EC10)

For chronic predictions we took the minimum of the mortality and growth predictions.

'''
# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '07_collect_predictions_trident', file_name ='logs/07_collect_predictions_trident.log', level_stream=logging.INFO, level_file=logging.DEBUG)

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

# read the predictions
trident_acute = pd.read_csv(r'data/predictions/trident/raw/TRIDENT_prediction_results_fish_EC50_MOR_96h_combo_model.csv', sep=',')
log.info(f'Number of Trident acute predictions read: {len(trident_acute)}')
trident_chronic_mortality = pd.read_csv(r'data/predictions/trident/raw/TRIDENT_prediction_results_fish_EC10_MOR_720h_combo_model.csv', sep=',')
log.info(f'Number of Trident chronic mortality predictions read: {len(trident_chronic_mortality)}')
trident_chronic_growth = pd.read_csv(r'data/predictions/trident/raw/TRIDENT_prediction_results_fish_EC10_GRO_720h_combo_model.csv', sep=',')
log.info(f'Number of Trident chronic growth predictions read: {len(trident_chronic_growth)}')

all_predictions = []
for study_type, df in zip(['acute','chronic', 'chronic'], [trident_acute, trident_chronic_mortality, trident_chronic_growth]):
    # set the mol ID
    df['mol ID'] = df.index
    # set the platform
    df['platform'] = 'Trident'
    # set the model name and study type
    if df.equals(trident_acute):
        df['model name'] = 'fish acute EC50 (96 hr) combo model'
        df['study type'] = 'acute'
    elif df.equals(trident_chronic_mortality):
        df['model name'] = 'fish chronic EC10 (720 hr) mortality combo model'
        df['study type'] = 'chronic'
    elif df.equals(trident_chronic_growth):
        df['model name'] = 'fish chronic EC10 (720 hr) growth combo model'
        df['study type'] = 'chronic'

    # set the model version
    df['model version'] = '1.0'
    # add the standardised smiles
    df = (df
          .merge(smiles_set[['mol ID', 'smiles (standardised)']], on='mol ID', how='inner')
          )
    # set the prediction status
    df['prediction status'] = np.where(df['predictions (mg/L)'].isnull(), 'failed', 'succeeded')
    # set the applicability domain
    df['AD'] = np.select(condlist=[(df['prediction status']=='succeeded') & (df['mean cosine similarity'] > 0.3),
                                   (df['prediction status']=='succeeded') & (df['mean cosine similarity'] <= 0.3)],
                         choicelist=['in domain', 'out of domain'], default=None)
    # set the predictions and the predicted quantity
    df['prediction'] = np.where(df['prediction status']=='succeeded', df['predictions (mg/L)'], None)
    if study_type == 'acute':
        df['predicted quantity'] = np.where(df['prediction status']=='succeeded', 'LC50 (mg/L)', None)
    else:
        df['predicted quantity'] = np.where(df['prediction status']=='succeeded', 'EC10 (mg/L)', None)
    # set the training/validation/set
    df['training/validation set'] = np.select(condlist=[(df['prediction status']=='succeeded') & (df['endpoint match'] > 0.5),
                                                         (df['prediction status']=='succeeded') & (df['endpoint match'] < 0.5)],
                                              choicelist=['training set', 'not in training/validation set'], default=None)
    # set the no effects at saturation column
    df['no effects at saturation'] = np.where(df['prediction status']=='succeeded', 'no', None)
    # set the notes column
    df['notes'] = [dict() for _ in range(len(df))]

    # keep only the important columns
    first_cols = ['platform', 'model name', 'model version', 'mol ID', 'smiles (standardised)',
                  'prediction status', 'training/validation set', 'AD', 'predicted quantity', 'prediction', 'no effects at saturation', 'notes']
    new_order = first_cols + [col for col in df.columns if col not in first_cols]
    df = df[new_order]
    all_predictions.append(df)
all_predictions = pd.concat(all_predictions, ignore_index=True, axis='index', sort=False)

# keep the most conservative chronic  predictions (minimum of mortality and growth)
msk = all_predictions['study type'] == 'chronic'
tmp = (all_predictions
       .loc[msk]
       .sort_values(['mol ID', 'prediction'], ascending=True)
       .groupby('mol ID')
       .first()
       .reset_index()
       .assign(**{'model name': 'fish chronic EC10 (720 hr) combo model'})
       )
# report how many chronic predictions are based on growth and how many on mortality
log.info(f'Lowest chronic predictions are based on: {tmp['effect'].value_counts().to_dict()}')

# combine all acute and chronic predictions
all_predictions = pd.concat([all_predictions.loc[~msk], tmp], ignore_index=True, axis='index', sort=False)
# store the processed predictions
all_predictions.to_excel(rf'data/predictions/trident/processed/predictions_trident.xlsx', index=False)





