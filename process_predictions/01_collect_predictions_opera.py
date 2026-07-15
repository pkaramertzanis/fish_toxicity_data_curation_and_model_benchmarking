# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '01_collect_predictions_opera', file_name ='logs/01_collect_predictions_opera.log', level_stream=logging.INFO, level_file=logging.DEBUG)

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


# add the Opera predictions
smiles_opera = pd.read_csv(rf'data/structures/smiles_opera_input.smi', header=None, sep='\t', names=['smiles (standardised)', 'mol ID'])
log.info(f'number of structures for which Opera predictions were run: {len(smiles_opera)}')

# MP predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/MP/smiles_opera_input-smi_OPERA2.9Pred_MP.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'MP_exp', 'MP_pred', 'MP_predRange', 'AD_MP',
       'AD_index_MP', 'Conf_index_MP']
predictions_MP = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_MP = smiles_opera.set_index('mol ID').merge(predictions_MP.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# LogD predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/LogD/smiles_opera_input-smi_OPERA2.9Pred_LogD.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'LogD55_pred', 'LogD55_predRange', 'LogD74_pred',
       'LogD74_predRange', 'AD_LogD', 'AD_index_LogD', 'Conf_index_LogD']
predictions_LogD = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_LogD = smiles_opera.set_index('mol ID').merge(predictions_LogD.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# LogP predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/LogP/smiles_opera_input-smi_OPERA2.9Pred_LogP.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'LogP_exp', 'LogP_pred', 'LogP_predRange', 'AD_LogP',
       'AD_index_LogP', 'Conf_index_LogP']
predictions_LogP = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_LogP = smiles_opera.set_index('mol ID').merge(predictions_LogP.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# pKa predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/pKa/smiles_opera_input-smi_OPERA2.9Pred_pKa.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'pKa_a_exp', 'pKa_b_exp', 'ionization', 'pKa_a_pred',
       'pKa_a_predRange', 'pKa_b_pred', 'pKa_b_predRange', 'AD_pKa',
       'AD_index_pKa', 'Conf_index_pKa']
predictions_pKa = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_pKa = smiles_opera.set_index('mol ID').merge(predictions_pKa.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# WS predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/WS/smiles_opera_input-smi_OPERA2.9Pred_WS.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'LogWS_exp', 'LogWS_pred', 'WS_predRange', 'AD_WS',
       'AD_index_WS', 'Conf_index_WS']
predictions_WS = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_WS = smiles_opera.set_index('mol ID').merge(predictions_WS.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# VP predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/VP/smiles_opera_input-smi_OPERA2.9Pred_VP.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'LogVP_exp', 'LogVP_pred', 'VP_predRange', 'AD_VP',
       'AD_index_VP', 'Conf_index_VP']
predictions_VP = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_VP = smiles_opera.set_index('mol ID').merge(predictions_VP.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# HL predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/HL/smiles_opera_input-smi_OPERA2.9Pred_HL.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'LogHL_exp', 'LogHL_pred', 'HL_predRange', 'AD_HL',
       'AD_index_HL', 'Conf_index_HL']
predictions_HL = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_HL = smiles_opera.set_index('mol ID').merge(predictions_HL.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')


# CLint predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/CLint/smiles_opera_input-smi_OPERA2.9Pred_CLint.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'Clint_exp', 'Clint_pred', 'Clint_predRange', 'AD_Clint',
       'AD_index_Clint', 'Conf_index_Clint']
predictions_opera_CLint = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_opera_CLint = smiles_opera.set_index('mol ID').merge(predictions_opera_CLint.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# FUB predictions
predictions_fub = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/FUB/smiles_opera_input-smi_OPERA2.9Pred_FUB.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'FUB_exp', 'FUB_pred', 'FUB_predRange', 'AD_FUB',
       'AD_index_FUB', 'Conf_index_FUB']
predictions_fub = predictions_fub[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_fub = smiles_opera.set_index('mol ID').merge(predictions_fub.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# Caco2 predictions
predictions_opera = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/Caco2/smiles_opera_input-smi_OPERA2.9Pred_Caco2.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'CACO2_exp', 'CACO2_pred', 'CACO2_predRange', 'AD_CACO2',
       'AD_index_CACO2', 'Conf_index_CACO2']
predictions_opera_Caco2 = predictions_opera[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_opera_Caco2 = smiles_opera.set_index('mol ID').merge(predictions_opera_Caco2.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# R-Biodeg predictions
predictions_r_biodeg = pd.read_csv(rf'data/predictions/opera/with_standardisation/raw/R-Biodeg/smiles_opera_input-smi_OPERA2.9Pred_RBioDeg.csv')
# .. keep the needed columns and standardise the column names
cols = ['MoleculeID', 'ReadyBiodeg_exp', 'ReadyBiodeg_pred', 'AD_ReadyBiodeg',
       'AD_index_ReadyBiodeg', 'Conf_index_ReadyBiodeg']
predictions_r_biodeg = predictions_r_biodeg[cols].rename({'MoleculeID': 'mol ID'}, axis='columns')
# .. merge with the smiles
predictions_r_biodeg = smiles_opera.set_index('mol ID').merge(predictions_r_biodeg.set_index('mol ID'), how='left', left_index=True, right_index=True).reset_index().drop('mol ID', axis='columns')

# add everything together
all_predictions = (smiles_opera
                   .merge(predictions_MP, how='left', on='smiles (standardised)')
                   .merge(predictions_LogD, how='left', on='smiles (standardised)')
                   .merge(predictions_LogP, how='left', on='smiles (standardised)')
                   .merge(predictions_pKa, how='left', on='smiles (standardised)')
                   .merge(predictions_WS, how='left', on='smiles (standardised)')
                   .merge(predictions_VP, how='left', on='smiles (standardised)')
                   .merge(predictions_HL, how='left', on='smiles (standardised)')
                   .merge(predictions_opera_CLint, how='left', on='smiles (standardised)')
                   .merge(predictions_fub, how='left', on='smiles (standardised)')
                   .merge(predictions_opera_Caco2, how='left', on='smiles (standardised)')
                   .merge(predictions_r_biodeg, how='left', on='smiles (standardised)')
                   )

log.info(f'number of structures for which Opera predictions were produced: {len(all_predictions)}')


# merge with the inventory
all_predictions.to_excel(rf'data\predictions\opera\with_standardisation\processed\opera_predictions.xlsx', index=False)