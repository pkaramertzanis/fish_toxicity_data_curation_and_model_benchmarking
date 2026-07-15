# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '01_collect_predictions_vega', file_name ='logs/01_collect_predictions_vega.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np
import math
import chardet
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
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True

# read the original and standardised smiles
smiles_vega = pd.read_csv(rf'data\structures\smiles_vega_input.smi', sep='\t', header=None, names=['smiles (standardised)', 'mol ID'])

# list of models to collect predictions and training sets
models = [
    # Fish Acute (LC50) Toxicity model (IRFMN)
    {'platform': 'vega', 'model name': 'Fish Acute (LC50) Toxicity model (IRFMN)', 'version': '1.0.2',
     'predictions file': rf'data/predictions/vega/raw/report_FISH_LC50.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_FISH_LC50.txt',
     'study type': 'acute', 'species': 'unknown', 'endpoint': 'LC50 (mg/L)', 'prediction column': 'Predicted LC50 [mg/l]', 'domain': ('ADI',  math.nextafter(0.7, float('inf')))},
    # Fish Acute (LC50) Toxicity model (KNN-Read-Across)
    {'platform': 'vega', 'model name': 'Fish Acute (LC50) Toxicity model (KNN-Read-Across)', 'version': '1.0.1',
     'predictions file': rf'data/predictions/vega/raw/report_FISH_KNN.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_FISH_KNN.txt',
     'study type': 'acute', 'species': 'unknown', 'endpoint': 'LC50 (mg/L)', 'prediction column': 'Predicted toxicity [mg/L]', 'domain': ('ADI',  math.nextafter(0.7, float('inf')))},
    # Fish Acute (LC50) Toxicity model (KNN-Read-Across)
    {'platform': 'vega', 'model name': 'Fish Acute (LC50) Toxicity classification (SarPy-IRFMN)', 'version': '1.0.3',
     'predictions file': rf'data/predictions/vega/raw/report_FISH_IRFMN.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_FISH_IRFMN.txt',
     'study type': 'acute', 'species': 'unknown', 'endpoint': 'class', 'prediction column': 'Predicted toxicity class', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # Fathead Minnow LC50 96h (EPA)
    {'platform': 'vega', 'model name': 'Fathead Minnow LC50 96h (EPA)', 'version': '1.0.10',
     'predictions file': rf'data/predictions/vega/raw/report_FATHEAD_EPA.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_FATHEAD_EPA.txt',
     'study type': 'acute', 'species': 'fathead minnow', 'endpoint': 'LC50 (mg/L)', 'prediction column': 'Predicted toxicity [mg/l]',
     'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # Fish Acute (LC50) Toxicity model (IRFMN-Combase)
    {'platform': 'vega', 'model name': 'Fish Acute (LC50) Toxicity model (IRFMN-Combase)', 'version': '1.0.2',
     'predictions file': rf'data/predictions/vega/raw/report_FISH_COMBASE.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_FISH_COMBASE.txt',
     'study type': 'acute', 'species': 'unknown', 'endpoint': 'LC50 (mg/L)',
     'prediction column': 'Predicted LC50 [mg/l]', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # Fathead Minnow LC50 model (KNN-IRFMN)
    {'platform': 'vega', 'model name': 'Fathead Minnow LC50 model (KNN-IRFMN)', 'version': '1.1.2',
     'predictions file': rf'data/predictions/vega/raw/report_FATHEAD_KNN.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_FATHEAD_KNN.txt',
     'study type': 'acute', 'species': 'fathead minnow', 'endpoint': 'LC50 (mg/L)',
     'prediction column': 'Predicted toxicity [mg/l]', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # Guppy LC50 model (KNN-IRFMN)
    {'platform': 'vega', 'model name': 'Guppy LC50 model (KNN-IRFMN)', 'version': '1.1.2',
     'predictions file': rf'data/predictions/vega/raw/report_GUPPY_KNN.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_GUPPY_KNN.txt',
     'study type': 'acute', 'species': 'guppy', 'endpoint': 'LC50 (mg/L)',
     'prediction column': 'Predicted toxicity [mg/l]', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # Fish Acute (LC50) Toxicity model (NIC)
    {'platform': 'vega', 'model name': 'Fish Acute (LC50) Toxicity model (NIC)', 'version': '1.0.2',
     'predictions file': rf'data/predictions/vega/raw/report_FISH_NIC.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_FISH_NIC.txt',
     'study type': 'acute', 'species': 'unknown', 'endpoint': 'LC50 (mg/L)',
     'prediction column': 'Predicted fish toxicity [mg/l]', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # Fish Chronic (NOEC) Toxicity model (IRFMN)
    {'platform': 'vega', 'model name': 'Fish Chronic (NOEC) Toxicity model (IRFMN)', 'version': '1.0.2',
     'predictions file': rf'data/predictions/vega/raw/report_FISH_NOEC.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_FISH_NOEC.txt',
     'study type': 'chronic', 'species': 'unknown', 'endpoint': 'NOEC (mg/L)',
     'prediction column': 'Predicted NOEC [mg/l]', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # MOA fish toxicity classification (EPA T.E.S.T.)
    # {'platform': 'vega', 'model name': 'MOA fish toxicity classification (EPA T.E.S.T.)', 'version': '1.0.2',
    #  'predictions file': rf'output/{iteration}/predictions/vega/raw/report_MOA_EPA.txt',
    #  'training/validation set file': rf'output/{iteration}/predictions/training_validation_sets/vega/dataset_MOA_EPA.txt',
    #  'study type': 'not relevant', 'species': 'unknown', 'endpoint': 'mode of action',
    #  'prediction column': 'Predicted MOA toxicity', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # MOA pesticide classification (IRFMN)
    # {'platform': 'vega', 'model name': 'MOA pesticide classification (IRFMN)', 'version': '1.0.2',
    #  'predictions file': rf'output/{iteration}/predictions/vega/raw/report_MOA_IRFMN.txt',
    #  'training/validation set file': rf'output/{iteration}/predictions/training_validation_sets/vega/dataset_MOA_IRFMN.txt',
    #  'study type': 'not relevant', 'species': 'unknown', 'endpoint': 'mode of action',
    #  'prediction column': 'Predicted MoA', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # Zebrafish embryo AC50 (IRFMN-CORAL)
    {'platform': 'vega', 'model name': 'Zebrafish embryo AC50 (IRFMN-CORAL)', 'version': '1.0.1',
     'predictions file': rf'data/predictions/vega/raw/report_ZEBRAFISH_CORAL.txt',
     'training/validation set file': rf'data/training_validation_sets/vega/dataset_ZEBRAFISH_CORAL.txt',
     'study type': 'acute', 'species': 'zebra fish', 'endpoint': 'AC50 (ug/L)',
     'prediction column': 'Predicted AC50 [ug/l]', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
    # Verhaar classification (TOXTREE)
    # {'platform': 'vega', 'model name': 'Verhaar classification (TOXTREE)', 'version': '1.0.1',
    #  'predictions file': rf'output/{iteration}/predictions/vega/raw/report_VERHAAR_TOXTREE.txt',
    #  'training/validation set file': rf'output/{iteration}/predictions/training_validation_sets/vega/dataset_VERHAAR_TOXTREE.txt',
    #  'study type': 'not relevant', 'species': 'unknown', 'endpoint': 'mode of action',
    #  'prediction column': 'Predicted Verhaar class', 'domain': ('ADI', math.nextafter(0.7, float('inf')))},
]

predictions_all = []
for model in models:
    log.info(f"Collecting predictions for model: {model['model name']}")

    # check the encoding of the prediction
    with open(model['predictions file'], "rb") as f:
        raw = f.read()
        result = chardet.detect(raw)
        encoding = result['encoding']
        log.info(f"Detected encoding for {model['predictions file']}: {encoding}, confidence: {result['confidence']}")

    # read the predictions
    predictions = pd.read_csv(model['predictions file'], sep='\t', skiprows=4, encoding=encoding)
    predictions['mol ID'] = predictions['Id']
    # predictions[['ADI', 'Assessment']].sort_values('ADI')

    # convert the prediction column to numeric when possible, set to None if prediction is non positive for concentration-based endpoints
    if (model['endpoint'] == 'LC50 (mg/L)') or (model['endpoint'] == 'NOEC (mg/L)') or (model['endpoint'] == 'AC50 (ug/L)'):
        predictions[model['prediction column']] = pd.to_numeric(predictions[model['prediction column']], errors='coerce').apply(lambda x: x if x>0 else None)

    # set the prediction status
    msk = (# there is a prediction (even classification results contain numbers)
             predictions[model['prediction column']].notnull()
           & (predictions[model['prediction column']] != '-')
           & (predictions[model['prediction column']].astype(str).str.contains(r'\d', na=False, regex=True))
           #  there is a domain value
           & predictions[model['domain'][0]].notnull()
           & (predictions[model['domain'][0]] != '-')
           & (predictions[model['domain'][0]].astype(str).str.contains(r'\d', na=False, regex=True))
           )
    predictions['prediction status'] = np.select(condlist=[msk], choicelist=['succeeded'], default='failed')
    log.info(f"Predictions: {len(predictions)} entries, {predictions['mol ID'].nunique()} unique structures, {msk.sum()} successful predictions")

    # initiate the notes dictionary
    predictions['notes'] = [dict() for _ in range(len(predictions))]

    # add in the notes the reason for failed predictions by using the Vega Remarks column
    msk_failed = predictions['prediction status'] == 'failed'
    predictions['notes'] = np.where(msk_failed,
                                    [{**d, "failed prediction reasoning": v} for d, v in zip(predictions["notes"], predictions["Remarks"])],
                                    predictions["notes"])

    # set the no effects at saturation column
    predictions['no effects at saturation'] = np.where(predictions['prediction status'] == 'succeeded', 'no', None)

    # set the applicability domain
    predictions[model['domain'][0]] = pd.to_numeric(predictions[model['domain'][0]], errors='coerce')
    msk_in_domain = (predictions['prediction status'] == 'succeeded') & predictions[model['domain'][0]].notnull() & (predictions[model['domain'][0]] >= model['domain'][1])
    msk_out_domain = (predictions['prediction status'] == 'succeeded') & predictions[model['domain'][0]].notnull() & (predictions[model['domain'][0]] < model['domain'][1])
    predictions['AD'] = np.select(condlist=[msk_in_domain, msk_out_domain],
                                  choicelist=['in domain', 'out of domain'], default=None)

    # read the training/validation set, training_validation_set in the end has the columns ['smiles (standardised)', 'training/validation set']
    training_validation_set = pd.read_csv(model['training/validation set file'], sep='\t', header=0)
    training_validation_set['training/validation set'] = np.select(condlist=[training_validation_set['Status']=='Training',
                                                                             training_validation_set['Status']=='Test'],
                                                                   choicelist=['training set', 'validation set'],  default='unknown')
    if (training_validation_set['training/validation set']=='unknown').sum():
        raise ValueError("Some entries in the training/validation set have unknown set assignment")
    # .. check and standardise the training/validation set structures
    tmp = training_validation_set['SMILES'].apply(standardise_check_molecule)
    tmp = pd.json_normalize(tmp)
    training_validation_set = pd.concat([training_validation_set, tmp.drop('smiles', axis='columns')], axis='columns')
    log.info(f"Training/validation set: {len(training_validation_set)} entries, {training_validation_set['smiles (standardised)'].nunique()} unique standardised structures")
    training_validation_set = training_validation_set[['smiles (standardised)', 'training/validation set']].dropna(how='any', axis='index').drop_duplicates().reset_index(drop=True)

    # add the standardised smiles
    predictions = predictions.merge(smiles_vega, on='mol ID', how='left')

    # add the information about training/validation set to predictions
    predictions = predictions.merge(training_validation_set, on='smiles (standardised)', how='left').fillna({'training/validation set': 'not in training/validation set'})

    # rename the predicted column and add model information
    predictions = predictions.rename({model['prediction column']: model['endpoint']}, axis='columns')
    predictions['platform'] = model['platform']
    predictions['model name'] = model['model name']
    predictions['model version'] = model['version']
    predictions['study type'] = model['study type']
    predictions['predicted quantity'] = model['endpoint']
    predictions['prediction'] = predictions[model['endpoint']]

    # keep only the required columns
    cols = ['platform', 'model name', 'model version', 'study type', 'mol ID', 'smiles (standardised)', 'prediction status', 'training/validation set', 'AD', 'predicted quantity', 'prediction', 'no effects at saturation', 'notes']
    predictions = predictions[cols]

    predictions_all.append(predictions)


# put all predictions together
predictions_all = pd.concat(predictions_all, axis='index', ignore_index=True, sort=False)

# dump the notes column separately as json
predictions_all['notes'] = predictions_all['notes'].apply(json.dumps)

# store the processed predictions
predictions_all.to_excel(rf'data/predictions/vega/processed/predictions_vega.xlsx', index=False)

# predictions_all.pivot_table(index=['model name'], columns=['prediction status', 'AD', 'training/validation set'], values='mol ID', aggfunc='nunique', fill_value=0, dropna=False).to_clipboard()


