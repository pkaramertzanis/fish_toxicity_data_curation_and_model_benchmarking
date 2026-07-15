'''
Utility script to flatten the REACH fish long term studies to a single excel file.

The script produces one row per experimental study (endpoint study record) with all the relevant information.

The script enhances the test material by adding molecular structures obtained from CCTE.

This scripts does not filter any rows, it just flattens the information. Filtering is done in subsequent scripts.

The output of this script is fish_chronic/processed/REACH_long_term_fish_raw_validation.xlsx
'''
# setup logging
import logger
import logging
log = logger.setup_applevel_logger(file_name ='logs/REACH_flatten.log', level_stream=logging.INFO, level_file=logging.DEBUG)


from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
import json
from collections import Counter
from io import StringIO, BytesIO
import re
import time
from glob import glob
import pickle
from copy import deepcopy
from tqdm import tqdm
from itertools import chain
from functools import partial

from bs4 import BeautifulSoup
import requests

from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Descriptors

from process_REACH_data.document_processing import match_guideline

from configparser import ConfigParser

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
pd.set_option("max_colwidth", 150)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True

tqdm.pandas()

# read the CCTE API key from the config file
parser = ConfigParser()
fs = parser.read('configuration_ccte.ini')
ccte_token = parser.get('CCTE', 'token')


docs = pd.read_parquet(r'data/fish_chronic/raw/ENDPOINT_STUDY_RECORD_LongTermToxToFish_esr.parquet')
# keep only the rows with endpoint study records
msk = docs['UUID (endpoint study record) (echachem)'].notnull()
docs = docs.loc[msk]
log.info(f'number of endpoint study records for fish chronic toxicity: {len(docs)}')


# extract information from the endpoint study record documents
endpoint_study_records = []
for idx, row in tqdm(docs.iterrows(), 'processing endpoint study records'):
    esr_dict = row['endpoint study record']

    endpoint_study_record = dict()
    endpoint_study_record.update(row[['RML ID',
                                      'UUID (dossier) (initial)',
                                      'UUID (dossier) (echachem)',
                                      'UUID (endpoint study record) (echachem)',
                                      'duplicated',
                                      'name (echachem)',
                                      ]].to_dict())

    # administrative information
    endpoint_study_record['administrative'] = esr_dict.get('AdministrativeData')

    # guideline information
    endpoint_study_record['guideline'] = esr_dict.get('MaterialsAndMethods', dict()).get('Guideline')

    # GLP compliance
    endpoint_study_record['GLP compliance'] = esr_dict.get('MaterialsAndMethods', dict()).get('GLPComplianceStatement')

    # test material information
    endpoint_study_record['test material'] = row['test material']

    # literature information
    endpoint_study_record['literature reference'] = row['literature reference']

    # MaterialsAndMethods.SamplingAndAnalysis
    # sampling and analysis
    endpoint_study_record['analytical monitoring'] = esr_dict.get('MaterialsAndMethods', dict()).get('SamplingAndAnalysis', dict()).get('AnalyticalMonitoring')
    # details on sampling
    endpoint_study_record['details on sampling'] = esr_dict.get('MaterialsAndMethods', dict()).get('SamplingAndAnalysis', dict()).get('DetailsOnSampling')
    # details on analytical methods
    endpoint_study_record['details on analytical methods'] = esr_dict.get('MaterialsAndMethods', dict()).get('SamplingAndAnalysis', dict()).get('DetailsOnAnalyticalMethods')

    # MaterialsAndMethods.TestOrganisms
    # test organisms (species)
    endpoint_study_record['test organisms'] = esr_dict.get('MaterialsAndMethods', dict()).get('TestOrganisms', dict()).get('TestOrganismsSpecies')
    # details on test organisms
    endpoint_study_record['details on test organisms'] = esr_dict.get('MaterialsAndMethods', dict()).get('TestOrganisms', dict()).get('DetailsOnTestOrganisms')

    # MaterialsAndMethods.StudyDesign
    # test type
    endpoint_study_record['test type'] = esr_dict.get('MaterialsAndMethods', dict()).get('StudyDesign', dict()).get('TestType')
    # limit test
    endpoint_study_record['limit test'] = esr_dict.get('MaterialsAndMethods', dict()).get('StudyDesign', dict()).get('LimitTest')
    # total exposure duration
    endpoint_study_record['total exposure duration'] = esr_dict.get('MaterialsAndMethods', dict()).get('StudyDesign', dict()).get('TotalExposureDuration')

    # MaterialsAndMethods.TestConditions
    # test temperature
    endpoint_study_record['test temperature'] = esr_dict.get('MaterialsAndMethods', dict()).get('TestConditions', dict()).get('TestTemperature')
    # pH
    endpoint_study_record['pH'] = esr_dict.get('MaterialsAndMethods', dict()).get('TestConditions', dict()).get('Ph')
    # dissolved oxygen
    endpoint_study_record['dissolved oxygen'] = esr_dict.get('MaterialsAndMethods', dict()).get('TestConditions', dict()).get('DissolvedOxygen')
    # nominal and measured concentrations
    endpoint_study_record['nominal and measured concentrations'] = esr_dict.get('MaterialsAndMethods', dict()).get('TestConditions', dict()).get('NominalAndMeasuredConcentrations')
    # details on test conditions
    endpoint_study_record['details on test conditions'] = esr_dict.get('MaterialsAndMethods', dict()).get('TestConditions', dict()).get('DetailsOnTestConditions')

    # ResultsAndDiscussion
    # effect concentrations
    endpoint_study_record['effect concentrations'] = esr_dict.get('ResultsAndDiscussion', dict()).get('EffectConcentrations')
    # details on results
    endpoint_study_record['details on results'] = esr_dict.get('ResultsAndDiscussion', dict()).get('ResultsDetails')

    # Overall remarks and atachments
    # overall remarks
    endpoint_study_record['overall remarks'] = esr_dict.get('OverallRemarksAttachments', dict()).get('RemarksOnResults')
    if pd.notnull(endpoint_study_record['overall remarks']):
        soup = BeautifulSoup(endpoint_study_record['overall remarks'], 'html.parser')
        endpoint_study_record['overall remarks'] = soup.get_text()

    # validity criteria fulfilled
    endpoint_study_record['validity criteria'] = esr_dict.get('ApplicantSummaryAndConclusion', dict()).get('ValidityCriteria')

    # conclusions
    endpoint_study_record['conclusions'] = esr_dict.get('ApplicantSummaryAndConclusion', dict).get('Conclusions')

    # executive summary
    endpoint_study_record['executive summary'] = esr_dict.get('ApplicantSummaryAndConclusion', dict).get('ExecutiveSummary')
    if pd.notnull(endpoint_study_record['executive summary']):
        soup = BeautifulSoup(endpoint_study_record['executive summary'], 'html.parser')
        endpoint_study_record['executive summary'] = soup.get_text()

    endpoint_study_records.append(endpoint_study_record)
endpoint_study_records = pd.DataFrame(endpoint_study_records)

endpoint_study_records.groupby('UUID (endpoint study record) (echachem)').size().idxmax()


# mark the endpoint study records that correspond to experimental studies that are sources to read-across
def extract_cross_referenced(administrative_data: list[dict]) -> list[str]:
    '''Retrieve the cross-referenced endpoint study record list and returns a list of endpoint study records that are read-across sources
    :param administrative_data: dictionary with endpoint study record administrative data
    It returns the list of endpoint study record UUIds that are read-across sources
    '''
    cross_references = administrative_data.get('CrossReference', [])
    read_across_source_uuids = []
    for cross_reference in cross_references:
        if isinstance(cross_reference.get('ReasonPurpose'), dict) and cross_reference.get('ReasonPurpose', dict()).get('code (mapped)') == 'read-across source' and cross_reference.get('RelatedInformation'):
            read_across_source_uuids.append(cross_reference.get('RelatedInformation'))
    return read_across_source_uuids
tmp = endpoint_study_records['administrative'].apply(extract_cross_referenced).explode().to_list()
endpoint_study_records['read-across source'] = endpoint_study_records['UUID (endpoint study record) (echachem)'].isin(tmp)



# extract the test material identifiers
def extract_test_material_identifiers(test_material: dict) -> list[dict]:
    '''
    Extracts the identifiers from the test material
    :param test_material: dictionary with the test material information
    :return: list of dictionaries with the test material identifiers, including the molecular structure
    '''
    reference_substances = []
    if pd.notnull(test_material) and isinstance(test_material.get('Composition'), dict):
        composition_list = test_material.get('Composition', dict()).get('CompositionList', [])
        for composition_entry in composition_list:
            reference_substance = dict()
            reference_substance_doc = composition_entry.get('reference substance', dict())
            if not isinstance(reference_substance_doc, dict):
                continue
            # obtain the component type
            if isinstance(composition_entry.get('Type'), dict):
                component_type = composition_entry.get('Type').get('code (mapped)', None)
                reference_substance['component type'] = component_type
            # obtain the CAS number and CAS name
            if isinstance(reference_substance_doc.get('Inventory'), dict):
                reference_substance['CAS number'] = reference_substance_doc.get('Inventory', dict()).get('CASNumber')
                reference_substance['CAS name'] = reference_substance_doc.get('Inventory', dict()).get('CASName')
                if reference_substance['CAS name']:
                    reference_substance['CAS name'] = re.sub(r'[\s\r\n]+', ' ', reference_substance['CAS name']).strip()
                    reference_substance['CAS name'] = reference_substance['CAS name'] if len(reference_substance['CAS name']) > 5 else None
            # obtain the IUPAC name
            reference_substance['IUPAC name'] = reference_substance_doc.get('IupacName')
            if reference_substance['IUPAC name']:
                reference_substance['IUPAC name'] = re.sub(r'[\s\r\n]+', ' ', reference_substance['IUPAC name']).strip()
                reference_substance['IUPAC name'] = reference_substance['IUPAC name'] if len(reference_substance['IUPAC name']) > 5 else None

            if isinstance(reference_substance_doc.get('Inventory'), dict):
                reference_substance['CAS number'] = reference_substance_doc.get('Inventory', dict()).get('CASNumber')
            # obtain the SMILES and InChI
            if isinstance(reference_substance_doc.get('MolecularStructuralInfo'), dict):
                reference_substance['SMILES'] = reference_substance_doc.get('MolecularStructuralInfo', dict()).get('SmilesNotation')
                if reference_substance['SMILES']:
                    reference_substance['SMILES'] = re.sub(r'[\s\r\n]+', ' ', reference_substance['SMILES']).strip()
                reference_substance['InChI'] = reference_substance_doc.get('MolecularStructuralInfo', dict()).get('InChl')
                if reference_substance['InChI']:
                    reference_substance['InChI'] = re.sub(r'[\s\r\n]+', ' ', reference_substance['InChI']).strip()
            reference_substances.append(reference_substance)
    return reference_substances
endpoint_study_records['test material identifiers'] = endpoint_study_records['test material'].apply(extract_test_material_identifiers).dropna()


# convert the IUPAC names, CAS names and CAS numbers to DSSTox structures
identifiers = (pd.DataFrame.from_records(endpoint_study_records['test material identifiers'].explode().dropna().reset_index(drop=True))
               .drop(['component type', 'SMILES', 'InChI'], axis='columns')
               .melt(var_name='identifier type', value_name='identifier value', ignore_index=False)['identifier value']
               .dropna().drop_duplicates().sample(frac=1.).to_list())
batch_size = 50
headers = {
    "accept": "application/json",
    "x-api-key": ccte_token
}
ccte_data = []
for i in tqdm(range(0, len(identifiers), batch_size)):
    try:
        # we attempt to run in a batch, but we raise an exception if the number of
        # responses does not equal the number of input identifiers
        identifiers_batch = identifiers[i:i + batch_size]
        ccd = f"https://comptox.epa.gov/ctx-api/chemical/search/equal/"
        response = requests.post(ccd, headers=headers, data='\n'.join(identifiers_batch))
        response = pd.DataFrame.from_records(response.json())
        if len(response) == len(identifiers_batch):
            ccte_data.append(response.assign(identifier=identifiers_batch))
        else:
            ex = Exception(f'batch {i}-{i + batch_size-1} returned {len(response)} records instead of the expected {len(identifiers_batch)}')
            log.error(ex)
            raise ex
    except:
        log.info('will submit batch record per record')
        response = []
        for identifier in identifiers_batch:
            response.append(requests.post(ccd, headers=headers, data=identifier).json())
        ccte_data.append(pd.DataFrame.from_records(chain.from_iterable(response)).assign(identifier=identifiers_batch))

ccte_data = pd.concat(ccte_data, axis='index', ignore_index=True, sort=False)
ccte_structural_information = (ccte_data[['identifier', 'smiles', 'dtxsid']].dropna(subset='smiles').drop_duplicates()
                               .rename({'identifier': 'identifier value', 'smiles': 'SMILES'}, axis='columns'))


# compute the molecular weight from the smiles
for idx, row in ccte_structural_information.iterrows():
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)
    # compute the molecular weight with rdkit
    if mol:
        ccte_structural_information.at[idx, 'molecular weight'] = Descriptors.MolWt(mol)
ccte_structural_information = ccte_structural_information.dropna(subset=['molecular weight']).reset_index(drop=True)


# .. add the DSTTox structures into the test material identifiers
def add_DSSTox_structures(test_material_identifiers: list[dict], ccte_structural_information: pd.DataFrame) -> list[dict]:
    '''
    Adds the DSSTox structures into the test material identifiers
    :param test_material_identifiers: test material identifiers as in the original endpoint study record
    :return: test material identifiers with the added DSSTox structures
    '''
    test_material_identifiers = deepcopy(test_material_identifiers)
    if isinstance(test_material_identifiers, list):
        for reference_substance_identifiers in test_material_identifiers:
            reference_substance_identifiers_filtered = {k: v for k, v in reference_substance_identifiers.items() if k in ['CAS number', 'CAS name', 'IUPAC name']}
            msk = ccte_structural_information['identifier value'].isin(pd.Series(reference_substance_identifiers_filtered).dropna().drop_duplicates().to_list())
            reference_substance_identifiers['DSSTox structures'] = ccte_structural_information.loc[msk, ['SMILES', 'dtxsid', 'molecular weight']].drop_duplicates().to_dict(orient='records')
    return test_material_identifiers
endpoint_study_records['test material identifiers'] = endpoint_study_records['test material identifiers'].progress_apply(add_DSSTox_structures, ccte_structural_information=ccte_structural_information)

#  derive a single study year for the whole endpoint study record
def extract_study_year_information(administrative: dict, references: list) -> dict:
    '''
    Extracts the study year information from:
    - study period start date (in administrative information)
    - study period end date (in administrative information)
    - study period remarks (in administrative information)
    - literature reference year (in literature references)
    - literature reference report date (in literature references)
    :param administrative: dictionary with administrative information
    :param references: list with reference information
    :return: dictionary with extracted year information
    '''
    # obtain the study period start year
    if (study_period_start_year := administrative.get('StudyPeriodStartDate', None)):
        study_period_start_year =  datetime.strptime(study_period_start_year, '%Y-%m-%d').year
    # obtain the study period end year
    if (study_period_end_year := administrative.get('StudyPeriodEndDate', None)):
        study_period_end_year =  datetime.strptime(study_period_end_year, '%Y-%m-%d').year
    # obtain the study period years from the non-migrated field
    study_period_years = list(map(int, re.findall(r'((?:19|20)\d{2})(?:[^\d]|$)', study_period if (study_period := administrative.get('StudyPeriod', None)) else '')))
    try:
        references = list(references)
        # obtain the literature years
        literature_years = list(map(int, [ref_year for lit in references if pd.notnull(ref_year := lit.get('GeneralInfo', dict()).get('ReferenceYear') )]))
        # obtain the literature report years
        literature_report_years = [datetime.strptime(rep_date, '%Y-%m-%d').year for lit in references if pd.notnull( rep_date := lit.get('GeneralInfo', dict()).get('ReportDate') )]
    except TypeError as ex:
        literature_years = []
        literature_report_years = []
    study_years = {'study period start year': study_period_start_year,
                   'study period end year': study_period_end_year,
                   'study period remarks': study_period_years,
                   'year (literature)': literature_years,
                   'year (report date)': literature_report_years,
                   'year (max)': max(years) if (years:=list(filter(pd.notnull, [study_period_start_year, study_period_end_year, *study_period_years, *literature_years, *literature_report_years]))) else None}
    return study_years
endpoint_study_records['study year'] = endpoint_study_records[['administrative', 'literature reference']].apply(lambda x: extract_study_year_information(x.iloc[0], x.iloc[1]).get('year (max)'), axis='columns').fillna('not specified')




guideline_tagging = [
    ('fish, early-life stage toxicity test', ['EC10', 'NOEC'], 'OECD Guideline 210', r'(?i)(?:OECD|TG|Guideline)[^\d]*210'),
    ('fish, early-life stage toxicity test', ['EC10', 'NOEC'], 'EPA OPPTS 850.1400', r'(?i)(?:EPA|OPPTS|TSCA)[^\d]*850[\.\s\-]+1400'),
    ('fish, early-life stage toxicity test', ['EC10', 'NOEC'], 'EPA OTS 797.1000', r'(?i)(?:EPA|OTS|TSCA)[^\d]*797[\.\s\-]+1000'),
    ('fish, early-life stage toxicity test', ['EC10', 'NOEC'], 'EPA OTS 797.1600', r'(?i)(?:EPA|OTS|TSCA)[^\d]*797[\.\s\-]+1600'),
    ('fish, early-life stage toxicity test', ['EC10', 'NOEC'], 'EPA OPP 72-4', r'(?i)(?:EPA|OPP|TSCA)[^\d]*72[\.\s\-]+4'),
    ('fish, early-life stage toxicity test', ['EC10', 'NOEC'], 'ASTM E1241', r'(?i)ASTM[^\d]*E[\.\s\-]*1241'),

    ('fish, juvenile growth test', ['EC10', 'NOEC'], 'OECD Guideline 215', r'(?i)(?:OECD|TG|Guideline)[^\d]*215'),
    ('fish, juvenile growth test', ['EC10', 'NOEC'],  'EU Method C.14',  r'(?i)EU[^\d]*C[\.\s\-]*14'),

    ('fish life cycle toxicity', ['EC10', 'NOEC'], 'EPA OPPTS 850.1500', r'(?i)(?:EPA|OPPTS|TSCA)[^\d]*850[\.\s\-]+1500'),
    ('fish life cycle toxicity', ['EC10', 'NOEC'],  'EPA OPP 72-5', r'(?i)(?:EPA|OPP)[^\d]*72[\.\s\-]+5'),

    ('short-term toxicity test on embryo and sac-fry stages', ['EC10', 'NOEC'], 'OECD Guideline 212', r'(?i)(?:OECD|TG|Guideline)[^\d]*212'),
    ('short-term toxicity test on embryo and sac-fry stages', ['EC10', 'NOEC'], 'EU Method C.15', r'(?i)EU[^\d]*C[\.\s\-]*15'),

    ('fish sexual development test', ['EC10', 'NOEC'], 'OECD Guideline 234', r'(?i)(?:OECD|TG|Guideline)[^\d]*234'),

]
guideline_tagging = pd.DataFrame.from_records(guideline_tagging, columns=['study type', 'measured quantity', 'guideline', 'guideline pattern'])
# .. match guidelines (both the picklist guideline values and not the other text)
target_guidelines = guideline_tagging[['guideline', 'guideline pattern']].set_index('guideline').squeeze().to_dict()
endpoint_study_records['matched guidelines'] = endpoint_study_records['guideline'].apply(partial(match_guideline, match_other=True), target_guidelines=target_guidelines)
endpoint_study_records['study type (based on guideline)'] = endpoint_study_records['matched guidelines'].apply(lambda matched_guidelines: guideline_tagging.loc[guideline_tagging['guideline'].isin(matched_guidelines), 'study type'].drop_duplicates().sort_values().to_list())
endpoint_study_records['measured quantity (based on guideline)'] = endpoint_study_records['matched guidelines'].apply(lambda matched_guidelines: guideline_tagging.loc[guideline_tagging['guideline'].isin(matched_guidelines), 'measured quantity'].explode().drop_duplicates().sort_values().to_list())

# extract key information for filtering the endpoint study records
# .. experimental study with reliability 1 or 2
msk1 = endpoint_study_records['administrative'].apply(lambda x: x.get('Reliability', dict()).get('code (mapped)') if x.get('Reliability') else None).isin(['1 (reliable without restriction)', '2 (reliable with restrictions)'])
msk2 = endpoint_study_records['administrative'].apply(lambda x: x.get('StudyResultType', dict()).get('code (mapped)') if x.get('StudyResultType') else None).isin(['experimental study'])
endpoint_study_records['experimental study with reliability 1 or 2'] = np.where(msk1 & msk2, 'yes', 'no')
# .. test material composition contains one or multiple reference substances, but all point to one and only DSSTox structure
msk3 = endpoint_study_records['test material identifiers'].apply(lambda ref_subs: (len(ref_subs) >= 1)
                                                                 and all([len(ref_sub.get('DSSTox structures', [])) == 1 for ref_sub in ref_subs])
                                                                 and all([ref_sub.get('DSSTox structures', [])[0]['dtxsid'] == ref_subs[0].get('DSSTox structures', [])[0]['dtxsid'] for ref_sub in ref_subs])
                                                                 )
# .. alternatively we can filter the studies, test material composition contains one reference substance with one DSSTox structure
# msk3 = endpoint_study_records['test material identifiers'].apply(lambda ref_subs: (len(ref_subs)==1) and (len(ref_subs[0].get('DSSTox structures', [])) == 1))
endpoint_study_records['test material with one DSSTox structure'] = np.where(msk3, 'yes', 'no')

# export the raw dataset for further processing
endpoint_study_records.to_pickle(rf'data/fish_chronic/processed/REACH_long_term_fish_raw_validation.pickle')

# export the raw dataet for validation
# convert dictionaries to json strings for easier export to excel
# class to handle numpy arrays in json serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
cols_with_structs = [
    col for col in endpoint_study_records.columns
    if endpoint_study_records[col].apply(lambda x: isinstance(x, (list, dict))).any()
]
for json_column in cols_with_structs:
    log.info(f'converting column {json_column} to json strings')
    endpoint_study_records[json_column] = endpoint_study_records[json_column].apply(lambda x: json.dumps(x, indent=2, cls=NumpyEncoder))
endpoint_study_records.to_excel(rf'data/fish_chronic/processed/REACH_long_term_fish_raw_validation.xlsx', index=False)



