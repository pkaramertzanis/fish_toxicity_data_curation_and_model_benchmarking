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
class ExecutiveSummary(BaseModel):
    effects_at_highest_concentration: Annotated[
        Literal[
            "significant effects seen",
            "no significant effects seen",
            "cannot determine",
        ],
        Field(
            description='This value is set to "no effects seen" if no significant effects were observed at the highest tested concentration. '
            'The value is set to "effects seen" if significant effects were observed at any tested concentration. '
            'If none of the above holds, the value is set to "cannot determine".'
        ),
    ]


# cache functionality
CACHE_FILE = Path(
    "process_REACH_data/cached_LLM_results/parsed_conclusions_executive_summary.pickle"
)


def _load_cache() -> list[dict]:
    if CACHE_FILE.exists():
        with open(
            rf"process_REACH_data/cached_LLM_results/parsed_conclusions_executive_summary.pickle",
            "rb",
        ) as f:
            cached_results = pickle.load(f)
        log.info(f"Cache loaded from {CACHE_FILE} with {len(cached_results)} entries.")
        return cached_results
    else:
        return []


def _save_cache(cached_results: list[dict]) -> None:
    with open(
        rf"process_REACH_data/cached_LLM_results/parsed_conclusions_executive_summary.pickle",
        "wb",
    ) as f:
        pickle.dump(cached_results, f)
        log.info(f"Cache saved to {CACHE_FILE} with {len(cached_results)} entries.")


# set the prompts
system_prompt = "You are a helpful regulatory ecotoxicologist interpreting ecotoxicity information in study reports."
with open(rf"prompts/prompt_conclusions_executive_summary_v1.txt", mode="r") as f:
    prompt_conclusions_executive_summary = f.read()


def parse_presence_of_effects_from_text(
    text: str,
    sleep_time: int = 2,
    limit_tokens: int = MAX_TOKENS,
    use_cache: bool = True,
    read_cached_failed: bool = True,
    cache_failed: bool = True,
) -> Dict:
    """
    Parse the conclusions and executive summary to determine if significant effects have occurred.
    The cache is used to avoid redundant API calls for previously processed texts but only if use_cache is True. The cache is updated with new results in any case.

    :param text: free text with an excerpt from a study report that is expected to contain information on effects presence, typically the conclusions and executive summary
    :param sleep_time: sleep time between API calls to avoid rate limiting
    :param limit_tokens: maximum number of tokens per call to avoid excessive costs
    :param use_cache: whether to use cached results to avoid redundant API calls; in any case the cache is updated with new results
    :param read_cached_failed: whether to read failed cached results; if False, failed cached results are ignored and the function will attempt to re-process the text
    :param cache_failed: whether to cache failed results; if False, failed results are not cached and the function will attempt to re-process the text next time
    :return: Dictionary with the extracted results
    """

    # load the cache
    cached_results = _load_cache()

    # try to retrieve from cache
    if use_cache:
        result = [
            cached_result
            for cached_result in cached_results
            if cached_result["raw conclusion and executive summary information"] == text
        ]
        # entry found in cache
        if result:
            # if the cached result is failed and we do not want to read failed results, ignore it and re-process the text
            if (result[0]["status"] == "failed") and (not read_cached_failed):
                pass
            else:
                log.info(
                    "Using cached result for processing conclusions and executive summary."
                )
                return result[0]

    # .. set the prompt
    prompt = (
        f"""{prompt_conclusions_executive_summary}\n"""
        f"""Study report: "{text}" """
    )
    # .. count the number of tokens in the prompt
    encoding = tiktoken.encoding_for_model(model_name)

    # parse the study report with the LLM
    result = {"raw conclusion and executive summary information": text}
    if len(encoding.encode(prompt)) >= limit_tokens:
        result["error"] = (
            f"Comment exceeds {limit_tokens} token limit ({len(encoding.encode(prompt))} tokens)"
        )
        result["status"] = "failed"
    else:
        try:
            log.info("Calling LLM for processing conclusions and executive summary.")
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "conclusions_executive_summary",
                        "schema": ExecutiveSummary.model_json_schema(),
                    },
                },
                # max_tokens=limit_tokens,
                # temperature=0.,
                # top_p=1.0,
                model=deployment,
            )
            try:
                # check that the output can be validated
                parsed = ExecutiveSummary.model_validate_json(
                    response.choices[0].message.content
                )
            except ValidationError as e:
                result["error"] = f"LLM output could not be validated: {e}"
                result["status"] = "failed"
            else:
                result[
                    "extracted information from conclusions and executive summary"
                ] = response.choices[0].message.content
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
            if cached_result["raw conclusion and executive summary information"] != text
        ] + [result]
        _save_cache(cached_results)

    return result
