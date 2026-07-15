# setup logging
import logger

log = logger.get_logger(__name__)

# process the comments with an LLM
import json
import pickle
import time
from configparser import ConfigParser
from pathlib import Path
from typing import Annotated, Dict, List, Literal

import pandas as pd
import tiktoken
from openai import AzureOpenAI
from pydantic import BaseModel, Field, ValidationError, confloat, conlist
from tqdm import tqdm

# pandas display options
# do not fold dataframes
pd.set_option("expand_frame_repr", False)
# maximum number of columns
pd.set_option("display.max_columns", 50)
# maximum number of rows
pd.set_option("display.max_rows", 500)
# precision of float numbers
pd.set_option("display.precision", 3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True

MAX_TOKENS = 12000

tqdm.pandas()

parser = ConfigParser()

# .. read in the LLM configuration
parser = ConfigParser()
fs = parser.read("configuration_llm.ini")
subscription_key = parser.get("ECHA-LLM", "subscription_key")
endpoint = parser.get("ECHA-LLM", "endpoint")
api_version = parser.get("ECHA-LLM", "api_version")
model_name = parser.get("ECHA-LLM", "model_name")
deployment = parser.get("ECHA-LLM", "deployment")
# .. set up the client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


# pydantic classes for the expected structured output
class ConcentrationData(BaseModel):
    concentration_type: Annotated[
        Literal["nominal", "measured"],
        Field(
            description="A nominal concentration is the concentration of a substance that is intended, set, or added in an experiment, rather than measured. "
            "It’s the value you target or prepare."
        ),
    ]
    fresh_or_aged: Annotated[
        Literal["fresh", "aged", "not relevant", "mean"],
        Field(
            description="Fresh concentrations are those that are freshly prepared when the exposure starts. They are often named concentrations at 0h. "
            "Aged concentrations, on the other hand, refer to those that have been measured after 24h or 48h hours and before the test medium is renewed in semi-static systems. "
            "The aged concentrations may be lower than the fresh due to factors like absorption, degradation, evaporation, precipitation or hydrolysis."
        ),
    ]
    treatment_concentrations: Annotated[
        conlist(confloat(ge=0), min_length=0, max_length=20),
        Field(
            description="A list of up to 20 treatment concentrations (numeric values only, no strings or other characters). "
            "If there are more than 20 concentrations, please select the 20 first ones in ascending order."
        ),
    ]
    unit: Annotated[
        str,
        Field(
            description="The unit of the concentrations that can be one of [g/L, mg/L, µg/L, ng/L, g/L, mg/kg, µg/kg, ng/kg, g/kg, mol/L, mmol/L, µmol/L, nmol/L, pmol/L, ppm, ppb]. If another unit is used, "
            "do not attempt to convert it to one of these units and just state 'unknown unit' and add the unknown unit in brackets, e.g. 'unknown unit (mg/m3)' or 'unknown unit (mg Al/L)'."
        ),
    ]


class ConcentrationList(BaseModel):
    tested_concentrations: Annotated[
        List[ConcentrationData],
        Field(
            description="A list of concentration data objects, each containing details about the concentration type, whether it's fresh or aged, the treatment concentrations, and the unit."
        ),
    ]


# cache functionality
CACHE_FILE = Path(
    "process_REACH_data/cached_LLM_results/parsed_tested_concentrations.pickle"
)


def _load_cache() -> list[dict]:
    if CACHE_FILE.exists():
        with open(
            rf"process_REACH_data/cached_LLM_results/parsed_tested_concentrations.pickle",
            "rb",
        ) as f:
            cached_results = pickle.load(f)
        log.info(f"Cache loaded from {CACHE_FILE} with {len(cached_results)} entries.")
        return cached_results
    else:
        return []


def _save_cache(cached_results: list[dict]) -> None:
    with open(
        rf"process_REACH_data/cached_LLM_results/parsed_tested_concentrations.pickle",
        "wb",
    ) as f:
        pickle.dump(cached_results, f)
        log.info(f"Cache saved to {CACHE_FILE} with {len(cached_results)} entries.")


# set the prompts
system_prompt = "You are a helpful regulatory ecotoxicologist interpreting ecotoxicity information in study reports."
with open(rf"prompts/prompt_treatment_concentrations_v2.txt", mode="r") as f:
    prompt_treatment_concentrations = f.read()


def parse_concentrations_from_text(
    text: str,
    sleep_time: int = 2,
    limit_tokens: int = MAX_TOKENS,
    use_cache: bool = True,
    read_cached_failed: bool = True,
    cache_failed: bool = True,
) -> Dict:
    """
    Parse a list of concentrations from a given text.
    The function looks for numeric values in the text and returns them as a list of tested concentration dictionaries.
    If no numeric values are found, it returns an empty list.
    The cache is used to avoid redundant API calls for previously processed texts but only if use_cache is True. The cache is updated with new results in any case.

    :param text: free text with an excerpt from a study report that is expected to contain information on the concentrations tested
    :param sleep_time: sleep time between API calls to avoid rate limiting
    :param limit_tokens: maximum number of tokens per call to avoid excessive costs
    :param use_cache: whether to use cached results to avoid redundant API calls; in any case the cache is updated with new results
    :param read_cached_failed: whether to read failed cached results; if False, failed cached results are ignored and the function will attempt to re-process the text
    :param cache_failed: whether to cache failed results; if False, failed results are not cached and the function will attempt to re-process the text next time
    :return: Dictionary with the raw text and the extracted tested concentrations found in the text
    """

    # load the cache
    cached_results = _load_cache()

    # try to retrieve from cache
    if use_cache:
        result = [
            cached_result
            for cached_result in cached_results
            if cached_result["raw concentration information"] == text
        ]
        # entry found in cache
        if result:
            # if the cached result is failed and we do not want to read failed results, ignore it and re-process the text
            if (result[0]["status"] == "failed") and (not read_cached_failed):
                pass
            else:
                log.info("Using cached result for concentration extraction.")
                return result[0]

    # .. set the prompt
    prompt = (
        f"""{prompt_treatment_concentrations}\n"""
        f"""Study report: "{text}" """
    )
    # .. count the number of tokens in the prompt
    encoding = tiktoken.encoding_for_model(model_name)

    # parse the study report with the LLM
    result = {"raw concentration information": text}
    if len(encoding.encode(prompt)) >= limit_tokens:
        result["error"] = (
            f"Comment exceeds {limit_tokens} token limit ({len(encoding.encode(prompt))} tokens)"
        )
        result["status"] = "failed"
    else:
        try:
            log.info("Calling LLM for concentration extraction.")
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "treatment_concentrations",
                        "schema": ConcentrationList.model_json_schema(),
                    },
                },
                # max_tokens=limit_tokens,
                # temperature=0.,
                # top_p=1.0,
                model=deployment,
            )
            try:
                # check that the output can be validated
                parsed = ConcentrationList.model_validate_json(
                    response.choices[0].message.content
                )
            except ValidationError as e:
                result["error"] = f"LLM output could not be validated: {e}"
                result["status"] = "failed"
            else:
                result["extracted concentrations tested"] = response.choices[
                    0
                ].message.content
                result["status"] = "succeeded"
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "failed"
        finally:
            time.sleep(sleep_time)

    # cache the result if it succeeded or if caching of failed results is enabled
    if (result["status"] == "succeeded") or (
        cache_failed and result["status"] == "failed"
    ):
        cached_results = [
            cached_result
            for cached_result in cached_results
            if cached_result["raw concentration information"] != text
        ] + [result]
        _save_cache(cached_results)

    return result


def convert_to_mg_per_L(value: float, unit: str, mw: float) -> float:
    """
    Convert a concentration to mg/L.
    Note that micro may be expressed with either:
    - "μ" (Greek small letter mu) → U+03BC
    - "µ" (micro sign) → U+00B5
    Parameters
    ----------
    value : float
        The concentration value.
    unit : str
        The unit of the input value (e.g., "g/L", "mg/L", "µg/L", "mmol/L", "ppm", "ppb").
    mw : float
        Molecular weight of the substance in g/mol (required for molar units).
    Returns
    -------
    float
        Concentration converted to mg/L.
    """
    # convert to lower case and strip whitespace
    unit = unit.strip().lower()
    # replace Greek mu with micro sign, using the unicode numbers
    unit = unit.replace("\u03bc", "u").replace("\u00b5", "u")

    # Mass/volume units
    if unit == "g/l":
        return value * 1000.0
    elif unit == "mg/l":
        return value
    elif unit == "ug/l":
        return value / 1000.0
    elif unit == "ng/l":
        return value / 1e6
    elif unit == "pg/l":
        return value / 1e9

    # Mass/mass units (assuming density ~1 g/cm³ → 1 kg/L)
    elif unit == "g/kg":
        return value * 1000.0
    elif unit == "mg/kg":
        return value
    elif unit == "ug/kg":
        return value / 1000.0
    elif unit == "ng/kg":
        return value / 1e6

    # Molar concentration units
    elif unit == "mol/l":
        return value * mw * 1000.0
    elif unit == "mmol/l":
        return value * mw
    elif unit == "umol/l":
        return value * mw / 1000.0
    elif unit == "nmol/l":
        return value * mw / 1e6
    elif unit == "pmol/l":
        return value * mw / 1e9

    # PPM and PPB (w/v, density = 1 g/cm³ → 1 ppm = 1 mg/L, 1 ppb = 1 µg/L)
    elif unit == "ppm":
        return value
    elif unit == "ppb":
        return value / 1000.0

    else:
        log.info(f"Unsupported unit: {unit}")
        return None
