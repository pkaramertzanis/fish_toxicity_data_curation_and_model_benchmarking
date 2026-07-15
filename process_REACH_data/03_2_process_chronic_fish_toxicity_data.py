'''
Utility script to process the REACH fish long term studies to a form that is suitable for model comparison and for building QSAR models.

The output of this script is data/fish_chronic/processed/REACH_long_term_fish_measurement.xlsx
'''

# setup logging
import logger
import logging
log = logger.setup_applevel_logger(file_name ='logs/REACH_flatten.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
import json
import re

from process_REACH_data.process_nominal_measured_concentrations import parse_concentrations_from_text, convert_to_mg_per_L
from process_REACH_data.process_conclusions_executive_summary import parse_presence_of_effects_from_text

import math

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

tqdm.pandas()


endpoint_study_records = pd.read_pickle(rf'data/fish_chronic/processed/REACH_long_term_fish_raw_validation.pickle')


# produce the tabular dataset
datasets = endpoint_study_records.copy(deep=True)



# process the GLP information
datasets = pd.json_normalize(datasets['GLP compliance']).add_prefix('GLP_compliance.').join(datasets).drop('GLP compliance', axis='columns')
datasets['GLP compliance'] = np.select([datasets['GLP_compliance.code (mapped)'].str.contains(r'(?i)^yes', na=False), \
                                                            datasets['GLP_compliance.code (mapped)'].str.contains(r'(?i)^no$', na=False)], ['yes', 'no'], default='not specified')

# process analytical monitoring information
datasets = pd.json_normalize(datasets['analytical monitoring']).add_prefix('analytical_monitoring.').join(datasets).drop('analytical monitoring', axis='columns')
datasets['analytical monitoring'] = np.select([datasets['analytical_monitoring.code (mapped)'].str.contains(r'(?i)^yes', na=False), \
                                                            datasets['analytical_monitoring.code (mapped)'].str.contains(r'(?i)^no$', na=False)], ['yes', 'no'], default='not specified')

# .. structure the validity criteria (a repeatable block)
# .. if all yes, we set the validity criteria to yes, if one no we set the validity criteria to no, otherwise (including if empty we set the validity criteria to not specified
def aggregate_validity_criteria(val_criteria: list):
    val_criteria_codes = [val_criterion.get('ValidityCriteriaFulfilled', dict()).get('code (mapped)', None) for val_criterion in val_criteria]
    # .. empty list
    if not val_criteria_codes:
        return 'not specified'
    # .. all are yes
    if all([val_criteria_code=='yes' for val_criteria_code in val_criteria_codes]):
        return 'yes'
    # .. one is no
    if any([val_criteria_code=='no' for val_criteria_code in val_criteria_codes]):
        return 'no'
    else:
        return 'not specified'
datasets['validity criteria (study sponsor)'] = datasets['validity criteria'].apply(aggregate_validity_criteria)


# process test organism information
datasets = pd.json_normalize(datasets['test organisms']).add_prefix('test_organisms.').join(datasets).drop('test organisms', axis='columns')
datasets['test organisms'] = np.where(datasets['test_organisms.code (mapped)'].str.contains(r'(?i)^other', na=False), 'not specified', datasets['test_organisms.code (mapped)'])


# produce the molecular weight from the molecular structures (the original DSSTox structure)
def compute_molecular_weight(row: pd.Series) -> float:
    '''Computes the molecular weight from the molecular structure. The molecular weight is returned
    only if the test material identifiers array has a single element with a DSSTox structure,
    or if there are multiple elements all of them have the same DSSTox structure
    :param row: row of the dataset with the following columns
                        - test material identifiers
                        - test material with one DSSTox structure
    :return: the molecular weight
    '''
    if row['test material with one DSSTox structure'] == 'yes':
        test_material_identifiers = row['test material identifiers']
        structure = test_material_identifiers[0].get('DSSTox structures', [])[0]['SMILES']
        mol = Chem.MolFromSmiles(structure)
        molecular_weight = Descriptors.MolWt(mol) if mol else None
        return molecular_weight
    else:
        return None
datasets['molecular weight'] = datasets[['test material identifiers', 'test material with one DSSTox structure']].apply(compute_molecular_weight, axis='columns')



# expand and process the effect concentrations
datasets = datasets.explode('effect concentrations').reset_index(drop=True)
datasets = pd.json_normalize(datasets['effect concentrations']).add_prefix('effect_concentrations.').join(datasets).drop('effect concentrations', axis='columns')
# .. convert the duration to days
def convert_duration(row: pd.Series) -> float:
    '''Converts the duration to hours
    :param row: row of the dataset with the following columns
                        - effect_concentrations.Duration.value
                        - effect_concentrations.Duration.unit.code (mapped)
    :return: the duration in hours
    '''
    value = row['effect_concentrations.Duration.value']
    unit = row['effect_concentrations.Duration.unit.code (mapped)']
    if pd.notnull(value) and pd.notnull(unit):
        if unit == 'd':
            return value
        elif unit == 'wk':
            return value * 7
        elif unit == 'mo':
            return value * 30
        elif unit == 'h':
            return value/24
        elif unit == 'min':
            return value / (24*60)
        else:
            raise ValueError(f'Unknown unit: {unit}')
datasets['duration (d)'] = datasets[['effect_concentrations.Duration.value', 'effect_concentrations.Duration.unit.code (mapped)']].apply(convert_duration, axis='columns')
# .. produce ranges according to the lower bound, the upper bound and the qualifiers
def create_bound(row: pd.Series) -> dict:
    '''Creates the lower and upper bounds (can be inf or 0) of the effect concentration in both mg/L and mol/L
    using the molecular weight; if the molecular weight is not available or the unit is not available or other
    then a dict with None values is returned
    :param row: row of the dataset with the following columns
                        - effect_concentrations.EffectConc.lowerQualifier
                        - effect_concentrations.EffectConc.lowerValue
                        - effect_concentrations.EffectConc.upperQualifier
                        - effect_concentrations.EffectConc.upperValue
                        - effect_concentrations.EffectConc.unit.code (mapped)
                        - molecular weight (of the original DSSTox structure)
    :return: dict of the lower and upper bounds in mg/L and mol/L
    '''
    res = {'effect concentration (mg/L, lower bound)': None,
           'effect concentration (mg/L, upper bound)': None,
           'effect concentration (mol/L, lower bound)': None,
           'effect concentration (mol/L, upper bound)': None,
                }
    if pd.isnull(row['molecular weight']) or pd.isnull(row['effect_concentrations.EffectConc.unit.code (mapped)']):
        return res
    # conversion to mg/L
    if row['effect_concentrations.EffectConc.unit.code (mapped)'] == 'mg/L':
        conversion = 1.
    elif row['effect_concentrations.EffectConc.unit.code (mapped)'] == 'µg/L':
        conversion = 1.e-3
    elif row['effect_concentrations.EffectConc.unit.code (mapped)'] == 'g/L':
        conversion = 1.e+3
    elif row['effect_concentrations.EffectConc.unit.code (mapped)'] == 'ng/L':
        conversion = 1.e-6
    elif row['effect_concentrations.EffectConc.unit.code (mapped)'] == 'mol/L':
        conversion = row['molecular weight']*1.e3
    elif row['effect_concentrations.EffectConc.unit.code (mapped)'] == 'mmol/L':
        conversion = row['molecular weight']
    elif row['effect_concentrations.EffectConc.unit.code (mapped)'] == 'µmol/L':
        conversion = row['molecular weight']*1.e-3
    elif row['effect_concentrations.EffectConc.unit.code (mapped)'] == 'other:':
        conversion = None
        return res
    else:
        raise ValueError(f'Unknown unit: {row["effect_concentrations.EffectConc.unit.code (mapped)"]}')
    lower_bound_orig = row['effect_concentrations.EffectConc.lowerValue']
    upper_bound_orig = row['effect_concentrations.EffectConc.upperValue']
    lower_bound_mg_L = row['effect_concentrations.EffectConc.lowerValue']*conversion if pd.notnull(lower_bound_orig) else None
    upper_bound_mg_L = row['effect_concentrations.EffectConc.upperValue']*conversion if pd.notnull(upper_bound_orig) else None
    lower_bound_mol_L = lower_bound_mg_L*1e-3/row['molecular weight'] if pd.notnull(lower_bound_orig) else None
    upper_bound_mol_L = upper_bound_mg_L*1e-3/row['molecular weight'] if pd.notnull(upper_bound_orig) else None
    if pd.notnull(lower_bound_orig) and pd.notnull(upper_bound_orig):
        pass
    elif pd.notnull(lower_bound_orig) and pd.isnull(upper_bound_orig) and (pd.isnull(row['effect_concentrations.EffectConc.lowerQualifier']) or row['effect_concentrations.EffectConc.lowerQualifier'] in['ca.']):
        upper_bound_mg_L = lower_bound_mg_L
        upper_bound_mol_L = lower_bound_mol_L
    elif pd.isnull(lower_bound_orig) and pd.notnull(upper_bound_orig) and (pd.isnull(row['effect_concentrations.EffectConc.upperQualifier']) or row['effect_concentrations.EffectConc.upperQualifier'] in ['ca.']):
        lower_bound_mg_L = upper_bound_mg_L
        lower_bound_mol_L = upper_bound_mol_L
    elif pd.notnull(lower_bound_orig) and pd.isnull(upper_bound_orig) and row['effect_concentrations.EffectConc.lowerQualifier'] in ['>', '>=']:
        upper_bound_mg_L = np.inf
        upper_bound_mol_L = np.inf
    elif pd.isnull(lower_bound_orig) and pd.notnull(upper_bound_orig) and row['effect_concentrations.EffectConc.upperQualifier'] in ['<', '<=']:
        lower_bound_mg_L = 0.
        lower_bound_mol_L = 0.
    res = {'effect concentration (mg/L, lower bound)': lower_bound_mg_L,
           'effect concentration (mg/L, upper bound)': upper_bound_mg_L,
           'effect concentration (mol/L, lower bound)': lower_bound_mol_L,
           'effect concentration (mol/L, upper bound)': upper_bound_mol_L}
    return res
tmp = pd.json_normalize(datasets.apply(create_bound, axis='columns'))
datasets = pd.concat([datasets, tmp], axis='columns', sort=False, ignore_index=False)
# .. set the endpoint
datasets['endpoint'] = np.select([datasets['effect_concentrations.Endpoint.code (mapped)'].isnull(), datasets['effect_concentrations.Endpoint.code (mapped)']=='other:'], ['not specified', 'not specified'], default=datasets['effect_concentrations.Endpoint.code (mapped)'])
# .. set the basis for effect
datasets['basis for effect'] = np.select([datasets['effect_concentrations.BasisForEffect.code (mapped)'].isnull(), datasets['effect_concentrations.BasisForEffect.code (mapped)']=='other:'], ['not specified', 'not specified'], default=datasets['effect_concentrations.BasisForEffect.code (mapped)'])
# .. set the concentration type
datasets['concentration type'] = np.select([datasets['effect_concentrations.NominalMeasured.code (mapped)'].isnull(), datasets['effect_concentrations.NominalMeasured.code (mapped)']=='other:'], ['not specified', 'not specified'], default=datasets['effect_concentrations.NominalMeasured.code (mapped)'])
# .. set the concentration based on
datasets['concentration based on'] = np.select([datasets['effect_concentrations.ConcBasedOn.code (mapped)'].isnull(), datasets['effect_concentrations.ConcBasedOn.code (mapped)']=='other:'], ['not specified', 'not specified'], default=datasets['effect_concentrations.ConcBasedOn.code (mapped)'])


# set the endpoint
endpoint_mapping = {'NOEC': 'NOEC',
                    'NOAEC': 'NOAEC',
                    'LOEC': 'LOEC',
                    'EC10': 'EC10',
                    'LC10': 'LC10',
                    }
endpoint_other_mapping = {
                          r'(?i)NOEC': 'NOEC',
                          r'(?i)NOAEC': 'NOAEC',
                          r'(?i)EC(?:[0-1]\d|\d(?:[^\d]+|$))': 'EC10',
                          r'(?i)LC(?:[0-1]\d|\d(?:[^\d]+|$))': 'LC10',
                          r'(?i)MATC|(?:(max.*accept.*\s+toxic.*concentration))': 'MATC'
                          }
def standardise_endpoint(row: pd.Series) -> str:
    '''Standardises the endpoint
    :param row: row of the dataset with the following columns
                        - effect_concentrations.Endpoint.code (mapped)
                        - effect_concentrations.Endpoint.other
    :return:endpoint andendpoint (standardised)
    '''
    endpoint_standardised = []
    endpoint_code_mapped = row['effect_concentrations.Endpoint.code (mapped)']
    endpoint_other = row['effect_concentrations.Endpoint.other']
    endpoint_remarks = row['effect_concentrations.Endpoint.remarks']
    if endpoint_code_mapped:
        # map the pick list values
        if endpoint_code_mapped in endpoint_mapping:
            endpoint_standardised.append(endpoint_mapping[endpoint_code_mapped])
    if pd.notnull(endpoint_other):
        for pat, mapped_value in endpoint_other_mapping.items():
            if re.search(pat, endpoint_other):
                endpoint_standardised.append(mapped_value)
    if pd.notnull(endpoint_remarks):
        for pat, mapped_value in endpoint_other_mapping.items():
            if re.search(pat, endpoint_remarks):
                endpoint_standardised.append(mapped_value)
    endpoint_standardised = sorted(list(set(endpoint_standardised)))
    endpoint = {'picklist value': endpoint_code_mapped,
                'other': endpoint_other,
                'remarks': endpoint_remarks,
               }
    return endpoint, endpoint_standardised
datasets[['endpoint', 'endpoint (standardised)']] = datasets.apply(standardise_endpoint, axis='columns', result_type='expand')

# set the basis for effect
basis_for_effect_mapping = {'mortality': 'mortality',
                            'weight': 'growth',
                            'length': 'growth',
                            'growth rate': 'growth',
                            'number hatched': 'hatching',
                            'time to hatch': 'hatching',
                            'reproduction': 'reproduction',
                            'behaviour': 'behaviour',
                            'larval development': 'developmental',
                            'morphology': 'developmental',
                            'adult mortality': 'mortality',
                            'number of fertilised eggs per spawn': 'reproduction',
                            'fertility': 'reproduction',
                            'time to swim up': 'developmental',
                            'gonadal histology': 'reproduction',
                            'secondary sexual characteristics': 'reproduction',
                            'adult behaviour': 'behaviour',
                            'number of spawns': 'reproduction'}
basis_for_effect_other_mapping = {r'(?i)(?:biomass|weight|growth|length)': 'growth',
                                  r'(?i)(?:mortality|survival|death)': 'mortality',
                                  r'(?i)(?:development|abnormalit|larvae)': 'developmental',
                                  r'(?i)(?:number.*egg|spawn|hatch)': 'hatching',
                                 }
def standardise_basis_for_effect(row: pd.Series) -> list[str]:
    '''Standardises the basis for effect
    :param row: row of the dataset with the following columns
                        - effect_concentrations.BasisForEffect.code (mapped)
                        - effect_concentrations.BasisForEffect.other
    :return: basis for effect and basis for effect (standardised)
    '''
    basis_for_effect_standardised = []
    basis_for_effect_code_mapped = row['effect_concentrations.BasisForEffect.code (mapped)']
    basis_for_effect_other = row['effect_concentrations.BasisForEffect.other']
    basis_for_effect_remarks = row['effect_concentrations.BasisForEffect.remarks']
    if basis_for_effect_code_mapped:
        # map the pick list values
        if basis_for_effect_code_mapped in basis_for_effect_mapping:
            basis_for_effect_standardised.append(basis_for_effect_mapping[basis_for_effect_code_mapped])
    if  pd.notnull(basis_for_effect_other):
        for pat, mapped_value in basis_for_effect_other_mapping.items():
            if re.search(pat, basis_for_effect_other):
                basis_for_effect_standardised.append(mapped_value)
    if pd.notnull(basis_for_effect_remarks):
        for pat, mapped_value in basis_for_effect_other_mapping.items():
            if re.search(pat, basis_for_effect_remarks):
                basis_for_effect_standardised.append(mapped_value)
    basis_for_effect_standardised = sorted(list(set(basis_for_effect_standardised)))
    basis_for_effect = {'picklist value': basis_for_effect_code_mapped,
                        'other': basis_for_effect_other,
                        'remarks': basis_for_effect_remarks,
                        }
    return basis_for_effect, basis_for_effect_standardised
datasets[['basis for effect', 'basis for effect (standardised)']] = datasets.apply(standardise_basis_for_effect, axis='columns', result_type='expand')



# global filtering operations
#  .. keep the rows corresponding to experimental studies with reliability is 1 or 2
msk1 = datasets['experimental study with reliability 1 or 2'] == 'yes'
# .. keep the rows corresponding to the matched guidelines
msk2 = datasets['matched guidelines'].apply(lambda guids: len(guids) > 0)
# .. keep the rows corresponding to one study type (based on guideline) only
msk3 = datasets['study type (based on guideline)'].apply(lambda vals: len(vals) == 1)
# .. keep the rows corresponding to studies with test material composition containing multiple reference substances, but all point to one and only DSSTox structure
msk4 = datasets['test material with one DSSTox structure'] == 'yes'
# .. there is numerical data
msk5 = datasets['effect concentration (mg/L, lower bound)'].notnull()
msk6 = datasets['effect concentration (mg/L, upper bound)'].notnull()
msk7 = datasets['effect concentration (mol/L, lower bound)'].notnull()
msk8 = datasets['effect concentration (mol/L, upper bound)'].notnull()
# duration is available
msk9 = datasets['duration (d)'].notnull()
msk = msk1 & msk2 & msk3 & msk4 & msk5 & msk6 & msk7 & msk8 & msk9
datasets = datasets[msk]
# reset the index
datasets = datasets.reset_index(drop=True)


# add required columns
# .. set the test material identifiers
datasets['substance name'] = datasets['test material identifiers'].apply(lambda x: x[0].get('IUPAC name', None) or x[0].get('CAS name', None))
datasets['CAS number'] = datasets['test material identifiers'].apply(lambda x: x[0].get('CAS number', None))
datasets['smiles'] = datasets['test material identifiers'].apply(lambda x: x[0].get('DSSTox structures', [{}])[0].get('SMILES', None))
datasets['dtxsid'] = datasets['test material identifiers'].apply(lambda x: x[0].get('DSSTox structures', [{}])[0].get('dtxsid', None))
# .. set the source
datasets['source'] = 'REACH data'
datasets['raw input file'] = None
# .. set the reference
datasets['reference'] = datasets['RML ID'] + ' (' + datasets['UUID (endpoint study record) (echachem)']+')'
datasets['notes'] = None
datasets['additional source data'] = None
# .. make lists a string
datasets['matched guidelines'] = datasets['matched guidelines'].apply(lambda guids: ', '.join(guids))
datasets['study type (based on guideline)'] = datasets['study type (based on guideline)'].apply(lambda vals: ', '.join(vals))
datasets['measured quantity (based on guideline)'] = datasets['measured quantity (based on guideline)'].apply(lambda vals: ', '.join(vals))


# structure the nominal and measured concentrations using an LLM
merged_cols = ['nominal and measured concentrations', 'details on sampling', 'details on test conditions']
texts = (datasets[merged_cols].apply(lambda merged_cols: '\n'.join(merged_cols.dropna().tolist()), axis='columns')
       .dropna()
       .drop_duplicates()
       .to_list())
structured_tested_concentrations_studies = []
n_succeeded = 0
for idx, text in tqdm(enumerate(texts), desc='Parsing tested concentrations'):
    structured_tested_concentration_study = parse_concentrations_from_text(text, sleep_time=0.25, use_cache=True, read_cached_failed=False, cache_failed=True)
    n_succeeded += structured_tested_concentration_study['status'] == 'succeeded'
    log.info(f'Parsed {idx+1}/{len(texts)} texts, succeeded {n_succeeded} times')
    structured_tested_concentrations_studies.append(structured_tested_concentration_study)
structured_tested_concentrations_studies = pd.DataFrame(structured_tested_concentrations_studies)
structured_tested_concentrations_studies = structured_tested_concentrations_studies.rename({'raw concentration information': 'nominal and measured concentrations (raw text, combined columns)',
                                                                                                    'extracted concentrations tested': 'nominal and measured concentrations (structured)',
                                                                                                    'status': 'nominal and measured concentrations (extraction status)'}, axis='columns')
datasets = (datasets
            # assign merged columns to match with the structured tested concentrations studies
            .assign(merged_columns=lambda df: df[merged_cols].apply(lambda merged_cols: '\n'.join(merged_cols.dropna().tolist()), axis='columns'))
            # merged the datasets with the structured tested concentrations studies
            .merge(structured_tested_concentrations_studies, left_on='merged_columns', right_on='nominal and measured concentrations (raw text, combined columns)', how='left')
            .drop('merged_columns', axis='columns')
            )

# process the conclusions and executive summary to extract presence of effects information using an LLM
merged_cols = ['details on results', 'conclusions', 'executive summary']
texts = (datasets[merged_cols].apply(lambda merged_cols: '\n'.join(merged_cols.dropna().tolist()), axis='columns')
       .dropna()
       .drop_duplicates()
       .to_list())
effects_in_conclusions_executive_summary_all = []
n_succeeded = 0
for idx, text in tqdm(enumerate(texts), desc='Parsing conclusions and executive summary'):
    effects_in_conclusions_executive_summary = parse_presence_of_effects_from_text(text, sleep_time=0.25, use_cache=True, read_cached_failed=False, cache_failed=True)
    n_succeeded += effects_in_conclusions_executive_summary['status'] == 'succeeded'
    log.info(f'Parsed {idx+1}/{len(texts)} texts, succeeded {n_succeeded} times')
    effects_in_conclusions_executive_summary_all.append(effects_in_conclusions_executive_summary)
effects_in_conclusions_executive_summary_all = pd.DataFrame(effects_in_conclusions_executive_summary_all)
effects_in_conclusions_executive_summary_all['effects seen based on conclusions/executive summary (LLM, extracted information)'] = effects_in_conclusions_executive_summary_all['extracted information from conclusions and executive summary'].apply(json.loads).apply(lambda t: t.get('effects_at_highest_concentration', None))
effects_in_conclusions_executive_summary_all = effects_in_conclusions_executive_summary_all.rename({'status': 'effects seen based on conclusions/executive summary (LLM, status)'}, axis='columns')
effects_in_conclusions_executive_summary_all = effects_in_conclusions_executive_summary_all[['raw conclusion and executive summary information',
                                                                                             'effects seen based on conclusions/executive summary (LLM, extracted information)',
                                                                                             'effects seen based on conclusions/executive summary (LLM, status)']]
datasets = (datasets
            # assign merged columns to match with the structured tested concentrations studies
            .assign(merged_columns=lambda df: df[merged_cols].apply(lambda merged_cols: '\n'.join(merged_cols.dropna().tolist()), axis='columns'))
            # merged the datasets with the structured tested concentrations studies
            .merge(effects_in_conclusions_executive_summary_all, left_on='merged_columns', right_on='raw conclusion and executive summary information', how='left')
            .drop(['merged_columns', 'raw conclusion and executive summary information'], axis='columns')
            )




# convert the structured tested concentrations to mg/L, count the number of tested concentrations and set the nominal
# minimum and maximum, and the measured minimum and maximum
def process_structured_tested_concentrations(row: pd.Series) -> dict:
    '''Processes the structured tested concentrations to convert them to mg/L, count the number of tested concentrations
    the nominal minimum and maximum, and the measured minimum and maximum
    :param row: row of the dataset with the following columns
                        - nominal and measured concentrations (structured)
                        - molecular weight (the original DSSTox structure)
    :return: dict with the number of tested concentrations, nominal min and max, measured min and max
    '''
    res = {'number of concentrations tested': None,
           'nominal concentrations (mg/L)': set(),
           'measured concentrations (fresh, mg/L)': set(),
           'measured concentrations (aged, mg/L)': set(),
           'measured concentrations (mean, mg/L)': set(),
           'max nominal concentrations (mg/L)': None,
           'max measured concentrations (mean, mg/L)': None
           }
    if pd.isnull(row['nominal and measured concentrations (structured)']) or pd.isnull(row['molecular weight']):
        return res
    mw = row['molecular weight']
    conc_entries = json.loads(row['nominal and measured concentrations (structured)']).get('tested_concentrations', [])
    # from pprint import pprint
    # print('---------------------')
    # pprint(conc_entries)
    for conc_entry in conc_entries:
        try:
            unit = conc_entry.get('unit', None)
            if conc_entry.get('concentration_type', None) == 'nominal':
                res['nominal concentrations (mg/L)'].update([convert_to_mg_per_L(value=conc, unit=unit, mw=mw) for conc in conc_entry.get('treatment_concentrations', None)])
            elif conc_entry.get('concentration_type', None) == 'measured' and conc_entry.get('fresh_or_aged', None) == 'fresh':
                res['measured concentrations (fresh, mg/L)'].update([convert_to_mg_per_L(value=conc, unit=unit, mw=mw) for conc in conc_entry.get('treatment_concentrations', None)])
            elif conc_entry.get('concentration_type', None) == 'measured' and conc_entry.get('fresh_or_aged', None) == 'aged':
                res['measured concentrations (aged, mg/L)'].update([convert_to_mg_per_L(value=conc, unit=unit, mw=mw) for conc in conc_entry.get('treatment_concentrations', None)])
            elif conc_entry.get('concentration_type', None) == 'measured' and conc_entry.get('fresh_or_aged', None) == 'mean':
                res['measured concentrations (mean, mg/L)'].update([convert_to_mg_per_L(value=conc, unit=unit, mw=mw) for conc in conc_entry.get('treatment_concentrations', None)])
        except Exception as e:
            log.error(f'Error processing concentration entry: {e}')
            continue
    res['nominal concentrations (mg/L)'] = set(filter(lambda x: pd.notnull(x), res['nominal concentrations (mg/L)']))
    res['measured concentrations (fresh, mg/L)'] = set(filter(lambda x: pd.notnull(x), res['measured concentrations (fresh, mg/L)']))
    res['measured concentrations (aged, mg/L)'] = set(filter(lambda x: pd.notnull(x), res['measured concentrations (aged, mg/L)']))
    res['measured concentrations (mean, mg/L)'] = set(filter(lambda x: pd.notnull(x), res['measured concentrations (mean, mg/L)']))
    res['number of concentrations tested'] = (max(len(res['nominal concentrations (mg/L)']),
                                                  len(res['measured concentrations (fresh, mg/L)']),
                                                  len(res['measured concentrations (aged, mg/L)']),
                                                  len(res['measured concentrations (mean, mg/L)'])))
    # maximum nominal concentration
    res['max nominal concentrations (mg/L)'] = max(res['nominal concentrations (mg/L)']) if res['nominal concentrations (mg/L)'] else None
    # maximum mean concentration
    max_fresh = max(res['measured concentrations (fresh, mg/L)']) if res['measured concentrations (fresh, mg/L)'] else None
    max_aged = max(res['measured concentrations (aged, mg/L)']) if res['measured concentrations (aged, mg/L)'] else None
    max_mean = max(res['measured concentrations (mean, mg/L)']) if res['measured concentrations (mean, mg/L)'] else None
    candidates = []
    if max_mean is not None:
        candidates.append(max_mean)
    if max_fresh is not None and max_aged is not None:
        geom_mean = math.sqrt(max_fresh * max_aged)
        candidates.append(geom_mean)
    if not candidates:
        res['max measured concentrations (mean, mg/L)'] = None
    else:
        res['max measured concentrations (mean, mg/L)'] = max(candidates)
    res['max measured concentrations (aged, mg/L)'] = max_aged
    res['max measured concentrations (fresh, mg/L)'] = max_fresh
    return res
datasets = pd.concat([datasets, pd.json_normalize(datasets[['nominal and measured concentrations (structured)', 'molecular weight']].progress_apply(process_structured_tested_concentrations, axis='columns'))], axis='columns', sort=False, ignore_index=False)




# set the scenarios
scenarios = []

study_types = {'FLS': 'fish, early-life stage toxicity test',
               'JGT': 'fish, juvenile growth test',
               'LCT': 'fish life cycle toxicity',
               'ESF': 'short-term toxicity test on embryo and sac-fry stages',
               'SDT': 'fish sexual development test',
               }

for study_type_code, study_type_name in study_types.items():

    # scenario xx_1a: long-term toxicity to fish, EC10/NO(A)EC based on dose response and measured mean concentrations
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
            & ( ((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
               |((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
                )
            & (datasets['analytical monitoring'] == 'yes')
            & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
            & datasets['endpoint (standardised)'].apply(lambda s: len(s)>0)
            & datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0)
            & datasets['concentration based on'].isin(['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
            # two or more concentrations tested
            & (datasets['number of concentrations tested'] >= 3)
            # effect concentration is not a range or if it is a range the lower and upper bound are within 20%
            & ((~np.isinf(datasets['effect concentration (mg/L, upper bound)']) & (np.abs(datasets['effect concentration (mg/L, lower bound)'] - datasets['effect concentration (mg/L, upper bound)']) < 0.2*datasets['effect concentration (mg/L, upper bound)'])))

            # do not fire the scenario if the conclusions/executive summary indicate no significant effects seen
            & (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] != 'no significant effects seen')

            # & (
            #        (datasets['max measured concentrations (mean, mg/L)'].notnull() & (datasets['effect concentration (mg/L, lower bound)'] <= pd.to_numeric(datasets['max measured concentrations (mean, mg/L)'], errors='coerce').fillna(0.)))
            #       |(datasets['max measured concentrations (aged, mg/L)'].notnull() & (datasets['effect concentration (mg/L, lower bound)'] <= pd.to_numeric(datasets['max measured concentrations (aged, mg/L)'], errors='coerce').fillna(0.)))
            #       |(datasets['max measured concentrations (fresh, mg/L)'].notnull() & (datasets['effect concentration (mg/L, lower bound)'] <= pd.to_numeric(datasets['max measured concentrations (fresh, mg/L)'], errors='coerce').fillna(0.)))
            #       |(datasets['max nominal concentrations (mg/L)'].notnull() & (datasets['effect concentration (mg/L, lower bound)'] <= pd.to_numeric(datasets['max nominal concentrations (mg/L)'], errors='coerce').fillna(0.)))
            #    )
            )
    scenario = {'scenario number': study_type_code+'_1a',
                'scenario description': study_type_name + ', EC10/NO(A)EC based on dose response and measured mean concentrations', 'msk': msk.copy(deep=True)}
    scenarios.append(scenario)

    # scenario xx_1b: short-term toxicity to fish, EC10/NO(A)EC based on dose response and preserved nominal concentrations
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
            & ( ((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
               |((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
                )
            & (datasets['analytical monitoring'] == 'yes')
            # & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
            & datasets['endpoint (standardised)'].apply(lambda s: len(s)>0)
            & datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0)
            & datasets['concentration based on'].isin(['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
            # two or more concentrations tested
            & (datasets['number of concentrations tested'] >= 3)
            # effect concentration is not a range or if it is a range the lower and upper bound are within 20%
            & ((~np.isinf(datasets['effect concentration (mg/L, upper bound)']) & (np.abs(datasets['effect concentration (mg/L, lower bound)'] - datasets['effect concentration (mg/L, upper bound)']) < 0.2*datasets['effect concentration (mg/L, upper bound)'])))

           # do not fire the scenario if the conclusions/executive summary indicate no significant effects seen
            & (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] != 'no significant effects seen')

            & (datasets['max measured concentrations (aged, mg/L)'].notnull() | datasets['max measured concentrations (mean, mg/L)'].notnull()) & datasets['max nominal concentrations (mg/L)'].notnull()
            & (
               ((datasets['max measured concentrations (aged, mg/L)'] - datasets['max nominal concentrations (mg/L)']).abs() < 0.2 * datasets['max nominal concentrations (mg/L)'])
              |((datasets['max measured concentrations (mean, mg/L)'] - datasets['max nominal concentrations (mg/L)']).abs() < 0.2 * datasets['max nominal concentrations (mg/L)'])
           )

            )
    scenario = {'scenario number': study_type_code+'_1b',
                'scenario description': study_type_name + ', EC10/NO(A)EC based on dose response and preserved nominal concentrations', 'msk': msk.copy(deep=True)}
    scenarios.append(scenario)

    # scenario xx_1c: short-term toxicity to fish, EC10/NO(A)EC based on dose response but exposure concentration unclear
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
            & ( ((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
               |((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
                )
            & (datasets['analytical monitoring'] == 'yes')
            # & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
            & datasets['endpoint (standardised)'].apply(lambda s: len(s)>0)
            & datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0)
            & datasets['concentration based on'].isin(['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
            # two or more concentrations tested
            & (datasets['number of concentrations tested'] >= 3)
            # effect concentration is not a range or if it is a range the lower and upper bound are within 20%
            & ((~np.isinf(datasets['effect concentration (mg/L, upper bound)']) & (np.abs(datasets['effect concentration (mg/L, lower bound)'] - datasets['effect concentration (mg/L, upper bound)']) < 0.2*datasets['effect concentration (mg/L, upper bound)'])))

            # do not fire the scenario if the conclusions/executive summary indicate no significant effects seen
            & (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] != 'no significant effects seen')

            )
    scenario = {'scenario number': study_type_code+'_1c',
                'scenario description': study_type_name + ', EC10/NO(A)EC based on dose response but exposure concentration unclear', 'msk': msk.copy(deep=True)}
    scenarios.append(scenario)

    # scenario xx_2a: short-term toxicity to fish, EC10/NO(A)EC based on solubility limit and mean measured concentrations
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
            & (((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
               |((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
                )
            & (datasets['analytical monitoring'] == 'yes')
            & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
            & datasets['endpoint (standardised)'].apply(lambda s: len(s)>0)
            & (datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0) | (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen'))
            & datasets['concentration based on'].isin(['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
            # two or more concentrations tested
            # & (datasets['number of concentrations tested'] >= 2)
            # effect concentration is unbounded or no significant effects seen and effect concentration lower bound is below 10 mg/L
            & (
                   ((np.isinf(datasets['effect concentration (mg/L, upper bound)']) & ((datasets['effect concentration (mg/L, lower bound)'] < 10))))
                   |
                   ((datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen') & (datasets['effect concentration (mg/L, lower bound)'] < 10))
              )
            )
    scenario = {'scenario number': study_type_code+'_2a',
                'scenario description': study_type_name + ', unbounded EC10/NO(A)EC based on solubility limit and mean measured concentrations', 'msk': msk.copy(deep=True)}
    scenarios.append(scenario)

    # scenario xx_2b: short-term toxicity to fish, EC10/NO(A)EC based on solubility limit and preserved nominal concentrations
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
           & (((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
              | ((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
              )
           & (datasets['analytical monitoring'] == 'yes')
           # & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
           & datasets['endpoint (standardised)'].apply(lambda s: len(s) > 0)
           & (datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0) | (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen'))
           & datasets['concentration based on'].isin(
                ['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
           # two or more concentrations tested
           # & (datasets['number of concentrations tested'] >= 2)
           # effect concentration is unbounded or no significant effects seen and effect concentration lower bound is below 10 mg/L
           & (
                   ((np.isinf(datasets['effect concentration (mg/L, upper bound)']) & ((datasets['effect concentration (mg/L, lower bound)'] < 10))))
                   |
                   ((datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen') & (datasets['effect concentration (mg/L, lower bound)'] < 10))
           )

           & (datasets['max measured concentrations (aged, mg/L)'].notnull() | datasets['max measured concentrations (mean, mg/L)'].notnull()) & datasets['max nominal concentrations (mg/L)'].notnull()
           & (
                   ((datasets['max measured concentrations (aged, mg/L)'] - datasets['max nominal concentrations (mg/L)']).abs() < 0.2 * datasets['max nominal concentrations (mg/L)'])
                   | ((datasets['max measured concentrations (mean, mg/L)'] - datasets['max nominal concentrations (mg/L)']).abs() < 0.2 * datasets['max nominal concentrations (mg/L)'])
           )

           )
    scenario = {'scenario number': study_type_code + '_2b',
                'scenario description': study_type_name + ', unbounded EC10/NO(A)EC based on solubility limit and preserved nominal concentrations',
                'msk': msk.copy(deep=True)}
    scenarios.append(scenario)

    # scenario xx_2c: short-term toxicity to fish, LC(E)50 based on solubility limit and preserved nominal concentrations
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
           & (((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
              | ((datasets['duration (d)'] >=7 ) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
              )
           & (datasets['analytical monitoring'] == 'yes')
           # & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
           & datasets['endpoint (standardised)'].apply(lambda s: len(s) > 0)
           & (datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0) | (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen'))
           & datasets['concentration based on'].isin(
                ['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
           # two or more concentrations tested
           # & (datasets['number of concentrations tested'] >= 2)
           # effect concentration is unbounded or no significant effects seen and effect concentration lower bound is below 10 mg/L
           & (
                   ((np.isinf(datasets['effect concentration (mg/L, upper bound)']) & ((datasets['effect concentration (mg/L, lower bound)'] < 10))))
                   |
                   ((datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen') & (datasets['effect concentration (mg/L, lower bound)'] < 10))
           )

           )
    scenario = {'scenario number': study_type_code + '_2c',
                'scenario description': study_type_name + ', unbounded EC10/NO(A)EC based on solubility limit but exposure concentration unclear',
                'msk': msk.copy(deep=True)}
    scenarios.append(scenario)


    # scenario xx_4a: short-term toxicity to fish, EC10/NO(A)EC based on limit test and mean measured concentrations
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
            & (((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
               |((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
                )
            & (datasets['analytical monitoring'] == 'yes')
            & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
            & datasets['endpoint (standardised)'].apply(lambda s: len(s)>0)
            & (datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0) | (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen'))
            & datasets['concentration based on'].isin(['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
            # two or more concentrations tested
            # & (datasets['number of concentrations tested'] >= 2)
            # effect concentration is unbounded or no significant effects seen mentioned in the conclusions/executive summary
            & ((np.isinf(datasets['effect concentration (mg/L, upper bound)']) & ((datasets['effect concentration (mg/L, lower bound)'] >= 10)))
               |
               ((datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen') & (datasets['effect concentration (mg/L, lower bound)'] >= 10))
              )

            )
    scenario = {'scenario number': study_type_code+'_4a',
                'scenario description': study_type_name + ', unbounded EC10/NO(A)EC based on limit test and mean measured concentrations', 'msk': msk.copy(deep=True)}
    scenarios.append(scenario)

    # scenario xx_4b: short-term toxicity to fish, EC10/NO(A)EC based on limit test and preserved nominal concentrations
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
           & (((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
              | ((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
              )
           & (datasets['analytical monitoring'] == 'yes')
           # & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
           & datasets['endpoint (standardised)'].apply(lambda s: len(s) > 0)
           & (datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0) | (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen'))
           & datasets['concentration based on'].isin(
                ['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
           # two or more concentrations tested
           # & (datasets['number of concentrations tested'] >= 2)
           # effect concentration is unbounded or no significant effects seen mentioned in the conclusions/executive summary
           & ((np.isinf(datasets['effect concentration (mg/L, upper bound)']) & ((datasets['effect concentration (mg/L, lower bound)'] >= 10)))
              |
              ((datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen') & (datasets['effect concentration (mg/L, lower bound)'] >= 10))
              )

           & (datasets['max measured concentrations (aged, mg/L)'].notnull() | datasets['max measured concentrations (mean, mg/L)'].notnull()) & datasets['max nominal concentrations (mg/L)'].notnull()
           & (
                   ((datasets['max measured concentrations (aged, mg/L)'] - datasets['max nominal concentrations (mg/L)']).abs() < 0.2 * datasets['max nominal concentrations (mg/L)'])
                   | ((datasets['max measured concentrations (mean, mg/L)'] - datasets['max nominal concentrations (mg/L)']).abs() < 0.2 * datasets['max nominal concentrations (mg/L)'])
           )

           )
    scenario = {'scenario number': study_type_code + '_4b',
                'scenario description': study_type_name + ', unbounded EC10/NO(A)EC based on limit test and preserved nominal concentrations',
                'msk': msk.copy(deep=True)}
    scenarios.append(scenario)

    # scenario xx_4c: short-term toxicity to fish, EC10/NO(A)EC based on limit test and preserved nominal concentrations
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
           & (((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
              | ((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
              )
           & (datasets['analytical monitoring'] == 'yes')
           # & datasets['concentration type'].str.contains(r'(?i)^meas.*(?:mean|TWA)', na=False, regex=True)
           & datasets['endpoint (standardised)'].apply(lambda s: len(s) > 0)
           & (datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0) | (datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen'))
           & datasets['concentration based on'].isin(
                ['test mat.', 'act. ingr.', 'dissolved', 'test mat. (dissolved fraction)', 'act. ingr. (dissolved fraction)'])
           # two or more concentrations tested
           # & (datasets['number of concentrations tested'] >= 2)
           # effect concentration is unbounded or no significant effects seen mentioned in the conclusions/executive summary
           & ((np.isinf(datasets['effect concentration (mg/L, upper bound)']) & ((datasets['effect concentration (mg/L, lower bound)'] >= 10)))
              |
              ((datasets['effects seen based on conclusions/executive summary (LLM, extracted information)'] == 'no significant effects seen') & (datasets['effect concentration (mg/L, lower bound)'] >= 10))
              )

           )
    scenario = {'scenario number': study_type_code + '_4c',
                'scenario description': study_type_name + ', unbounded EC10/NO(A)EC based on limit test but exposure concentration unclear',
                'msk': msk.copy(deep=True)}
    scenarios.append(scenario)

    # scenario xx_5: short-term toxicity to fish, other effect level
    msk = ((datasets['study type (based on guideline)'] == study_type_name)
           & (((datasets['duration (d)'] >= 28) & (study_type_name in ['fish, early-life stage toxicity test', 'fish, juvenile growth test', 'fish life cycle toxicity', 'fish sexual development test']))
              | ((datasets['duration (d)'] >= 7) & (study_type_name in ['short-term toxicity test on embryo and sac-fry stages']))
              )
           & (datasets['analytical monitoring'] == 'yes')
           & datasets['endpoint (standardised)'].apply(lambda s: len(s) > 0)
           & datasets['basis for effect (standardised)'].apply(lambda s: len(s) > 0)

           )
    scenario = {'scenario number': study_type_code + '_5',
                'scenario description': study_type_name + ', other EC10/NO(A)EC',
                'msk': msk.copy(deep=True)}
    scenarios.append(scenario)


# attach the scenarios to the datasets
datasets['scenario'] = None
datasets['scenario description'] = None
for scenario_number, scenario in enumerate(scenarios, start=1):
    datasets.loc[scenario['msk'], 'scenario'] = np.where(datasets.loc[scenario['msk'], 'scenario'].isnull(), scenario['scenario number'], datasets.loc[scenario['msk'], 'scenario'])
    datasets.loc[scenario['msk'], 'scenario description'] = np.where(datasets.loc[scenario['msk'], 'scenario description'].isnull(), scenario['scenario description'], datasets.loc[scenario['msk'], 'scenario description'])


# ----
(datasets.loc[datasets['CAS number']=='149-44-0'].assign(**{'tmp': datasets['basis for effect (standardised)'].apply(lambda lst: ', '.join(sorted(lst)) if lst else None),})
       .groupby(['UUID (dossier) (initial)', 'UUID (endpoint study record) (echachem)', 'scenario', 'tmp'], dropna=False)
       .progress_apply(lambda df: df.assign(**{'exclusion reason (LOEC in the presence of NOEC)':
                                                np.where(df['endpoint (standardised)'].apply(lambda s: 'LOEC' in s)
                                                         & (df['endpoint (standardised)'].apply(lambda s: 'NOEC' in s).sum()>=1)
                                                         & df['scenario'].notnull(),
                                                         'LOEC and NOEC for the same effect and scenario', None)}))
       .drop('tmp', axis='columns')
       ).reset_index(drop=True)


# ----

# in case we have the situation NOEC = x and LOEC > x for the same basis of effect, exclude the NOEC
datasets = (datasets.assign(**{'tmp': datasets['basis for effect (standardised)'].apply(lambda lst: ', '.join(sorted(lst)) if lst else None),})
       .groupby(['UUID (dossier) (initial)', 'UUID (endpoint study record) (echachem)', 'tmp'], dropna=False)
       .progress_apply(lambda df: df.assign(**{'exclusion reason (NOEC=x, LOEC>x for the same effect)':
                                                np.where(df['endpoint (standardised)'].apply(lambda s: 'NOEC' in s)
                                                       & (df['endpoint (standardised)'].apply(lambda s: 'LOEC' in s).sum()>=1)
                                                       & (df['effect concentration (mg/L, lower bound)'].max() - df['effect concentration (mg/L, lower bound)'].min()< 1.e-6),
                                                      'NOEC=x, LOEC>x for the same effect', None)}))
       .drop('tmp', axis='columns')
       ).reset_index(drop=True)

# in case there is LOEC and a NOEC for the same effect for the same scenario, keep the NOEC (groupby removes empty scenarios)
datasets = (datasets.assign(**{'tmp': datasets['basis for effect (standardised)'].apply(lambda lst: ', '.join(sorted(lst)) if lst else None),})
       .groupby(['UUID (dossier) (initial)', 'UUID (endpoint study record) (echachem)', 'scenario', 'tmp'], dropna=False)
       .progress_apply(lambda df: df.assign(**{'exclusion reason (LOEC and NOEC for the same effect and scenario)':
                                                np.where(df['endpoint (standardised)'].apply(lambda s: 'LOEC' in s)
                                                       & (df['endpoint (standardised)'].apply(lambda s: 'NOEC' in s).sum()>=1)
                                                       & df['scenario'].notnull(),
                                                      'LOEC and NOEC for the same effect and scenario', None)}))
       .drop('tmp', axis='columns')
       ).reset_index(drop=True)




# combine all exclusion reasons
exclusion_columns = [col for col in datasets.columns if 'exclusion reason ' in col]
log.info(f'Exclusion reason columns: {exclusion_columns}')
datasets['exclusion reasons'] = datasets[exclusion_columns].apply(lambda row: '; '.join([str(reason) for reason in row if pd.notnull(reason)]) if row.notnull().sum()>0 else None, axis='columns')



# store the dataset as pickle
datasets.to_pickle(rf'data/fish_chronic/processed/REACH_long_term_fish_measurement.pickle')
# datasets = pd.read_pickle(rf'data/fish_chronic/processed/REACH_long_term_fish_measurement.pickle')

# remove unnecessary columns
columns_to_keep = [
    "RML ID",
    "UUID (endpoint study record) (echachem)",
    "name (echachem)",
    "test_organisms.code",
    "test_organisms.code (mapped)",
    "test_organisms.other",
    "analytical_monitoring.code",
    "analytical_monitoring.code (mapped)",
    "analytical_monitoring.remarks",
    "GLP_compliance.code",
    "GLP_compliance.code (mapped)",
    "GLP_compliance.remarks",
    "administrative",
    "guideline",
    "test material",
    "literature reference",
    "details on sampling",
    "details on analytical methods",
    "details on test organisms",
    "test type",
    "limit test",
    "total exposure duration",
    "test temperature",
    "pH",
    "dissolved oxygen",
    "nominal and measured concentrations",
    "details on test conditions",
    "details on results",
    "overall remarks",
    "validity criteria",
    "conclusions",
    "executive summary",
    "read-across source",
    "test material identifiers",
    "study year",
    "matched guidelines",
    "study type (based on guideline)",
    "experimental study with reliability 1 or 2",
    "test material with one DSSTox structure",
    "GLP compliance",
    "analytical monitoring",
    "validity criteria (study sponsor)",
    "test organisms",
    "molecular weight",
    "duration (d)",
    "effect concentration (mg/L, lower bound)",
    "effect concentration (mg/L, upper bound)",
    "effect concentration (mol/L, lower bound)",
    "effect concentration (mol/L, upper bound)",
    "endpoint",
    "basis for effect",
    "concentration type",
    "concentration based on",
    "endpoint (standardised)",
    "basis for effect (standardised)",
    "substance name",
    "CAS number",
    "smiles",
    "dtxsid",
    "nominal and measured concentrations (raw text, combined columns)",
    "nominal and measured concentrations (structured)",
    "nominal and measured concentrations (extraction status)",
    "effects seen based on conclusions/executive summary (LLM, extracted information)",
    "effects seen based on conclusions/executive summary (LLM, status)",
    "number of concentrations tested",
    "nominal concentrations (mg/L)",
    "measured concentrations (fresh, mg/L)",
    "measured concentrations (aged, mg/L)",
    "measured concentrations (mean, mg/L)",
    "max nominal concentrations (mg/L)",
    "max measured concentrations (mean, mg/L)",
    "max measured concentrations (aged, mg/L)",
    "max measured concentrations (fresh, mg/L)",
    "scenario",
    "scenario description",
    "exclusion reasons"
]
datasets = datasets.reindex(columns=columns_to_keep)
# convert some columns to json strings
json_columns = ['administrative', 'guideline', 'test material', 'literature reference', 'test type', 'limit test',
                'total exposure duration', 'test material identifiers', 'validity criteria',
                'endpoint', 'endpoint (standardised)', 'basis for effect', 'basis for effect (standardised)']
for col in json_columns:
    datasets[col] = datasets[col].apply(lambda x: json.dumps(x, default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else obj) if x is not None else None)





# store the dataset as excel
datasets.to_excel(rf'data/fish_chronic/processed/REACH_long_term_fish_measurement.xlsx', index=False)


# scenario co-occurrence
msk = datasets['exclusion reasons'].isnull()
scenario_occurrence = datasets.loc[msk].assign(dummy=range(len(datasets.loc[msk]))).pivot_table(index=['smiles', 'dtxsid'], columns='scenario', values='dummy', aggfunc=pd.Series.nunique).notnull().reset_index()
scenario_occurrence.to_excel(rf'data/fish_chronic/processed/REACH_long_term_fish_measurement_scenario_occurrence.xlsx', index=False)
scenario_occurrence.groupby([col for col in scenario_occurrence.columns if col != 'smiles' and col != 'dtxsid']).size().rename('number of substances').reset_index().sort_values(by='number of substances', ascending=False)



