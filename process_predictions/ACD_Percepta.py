'''
Module to collect the classic ACD Percepta dissociation predictions, including the charge at pH 6, 7, 8 and 9.
'''
# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '07_collect_ACD_percepta_dissociation', file_name ='logs/07_collect_ACD_percepta_dissociation.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd

from rdkit import Chem

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


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def request_with_retries(
    method,
    url,
    *,
    timeout=5,
    params=None,
    json=None,
    data=None,
    headers=None,):

    session = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.request(
            method=method.upper(),
            url=url,
            timeout=timeout,
            params=params,
            json=json,
            data=data,
            headers=headers,
        )
        print(response.status_code, response.text)
        response.raise_for_status()

        # return json if possible, otherwise text
        try:
            return response.json()
        except ValueError:
            return response.text

    except requests.exceptions.Timeout:
        log.warning("Request timed out, method=%s url=%s", method, url)

    except requests.exceptions.ConnectionError:
        log.warning("Connection error occurred, method=%s url=%s", method, url)

    except requests.exceptions.HTTPError as e:
        log.warning("HTTP error occurred: %s, method=%s url=%s", e, method, url)

    except requests.exceptions.RequestException as e:
        log.warning("An error occurred: %s, method=%s url=%s", e, method, url)

    return None




# set the base URL for ACD Percepta kernel
url_ACD = 'http://epcecha01724.echa.europa.local:8080/percepta-web-service'


# obtain the available compilation IDs
url = f'{url_ACD}/compilations'
compilation_ID = request_with_retries('GET', url, timeout=10)
log.info(f'Available compilation IDs: {compilation_ID}')


# compute the dissociation constant using the GALAS algorithm (this seems to be the only available algorithm in ACD Percepta kernel without adding compilation IDs on the server)
smiles = 'SC(C)(C)C(N)C(=O)O'
url = f'{url_ACD}/calculations/ionization/compilations/{compilation_ID}'
headers = {'content-type': 'application/json'}
data = {'format': 'SMILES', 'mol': smiles}
response = request_with_retries('POST', url, json=data, headers=headers, timeout=10)


# covert name to structure
name = 'toluene'
url = f'{url_ACD}/utilities/generate/acd_name_to_structure/compilations/{compilation_ID}'
headers = {'content-type': 'application/json'}
data = {'param': name}
response = request_with_retries('POST', url, json=data, headers=headers, timeout=10)



