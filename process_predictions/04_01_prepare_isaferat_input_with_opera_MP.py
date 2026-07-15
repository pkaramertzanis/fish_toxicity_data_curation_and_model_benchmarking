# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '04_01_prepare_isaferat_input_with_opera_MP', file_name ='logs/04_01_prepare_isaferat_input_with_opera_MP.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np
import json


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
pd.set_option("max_colwidth", 120)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True


# read the standardised structures
smiles_uniq_set = pd.read_excel(r'data/structures/smiles.xlsx', sheet_name='smiles (standardised)')

# read the opera predicted melting points
opera_predictions = pd.read_excel(r'data/predictions/opera/with_standardisation/processed/opera_predictions.xlsx')
# keep only the melting point predictions that are in domain and only for solids
msk_solids = opera_predictions['MP_pred'].notnull() & opera_predictions['AD_MP'].notnull() & (opera_predictions['MP_pred'] >= 25)
msk_liquids = opera_predictions['MP_pred'].notnull() & opera_predictions['AD_MP'].notnull() & (opera_predictions['MP_pred'] < 25)
opera_predictions['melting point used by iSafeRat'] = np.select(condlist=[msk_solids, msk_liquids],
                                                              choicelist=[opera_predictions['MP_pred'], None],
                                                              default='no MP opera prediction available')
smiles_uniq_set = (smiles_uniq_set
                   .merge(opera_predictions[['mol ID', 'melting point used by iSafeRat']],
                          on='mol ID',
                          how='inner') # Opera returns a row for all input structures
                   )


# iSafeRat input
tmp = (smiles_uniq_set
       [['mol ID', 'smiles (standardised)']]
       .rename({'mol ID': 'Substance Number', 'smiles (standardised)': 'SMILES (mandatory)'}, axis='columns')
         .assign(**{'Melting Point (°C) (needed for Water Solubility, Ecotoxicity and Human Health predictions. If not given, it will be assumed to be less than 25°C (liquid substance at room temperature))': smiles_uniq_set['melting point used by iSafeRat'],
                    'Boiling Point (°C) (needed for Vapour Pressure prediction, which in turn is needed for skin penetration and sensitisation predictions)': None,
                    'Density (mg/mL) (currently not needed by any models)': None,
                    'logKow input (optional)': None,
                    'Water Solubility input (mg/L) (optional)': None,
                    'Vapour Presure input (Pa) (optional)': None}))
tmp.insert(loc=1, column='Substance Name or other Identifier', value=None)
# make the column names the first row
tmp = pd.concat([pd.DataFrame([tmp.columns], columns=tmp.columns), tmp], axis='index', sort=False, ignore_index=True)
tmp.columns = range(len(tmp.columns))  # reset the column names to integers
tmp.to_excel(fr'data/structures/smiles_isaferat_input_with_opera_MP.xlsx', index=False, header=True)