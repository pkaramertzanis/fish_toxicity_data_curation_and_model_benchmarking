# setup logging
import logger

log = logger.get_logger(__name__)

import re


def match_guideline(
    guidelines: list, target_guidelines: dict, match_other=False
) -> bool:
    """Returns True if the guideline matches the target list
    :param guidelines: list of guidelines in the endpoint study record
    :param target_guidelines: dictionary with the target guidelines and the patterns to match them
    :param match_other: if True it also matches the guideline other field
    Returns the matched guidelines, can be an empty list
    """
    try:
        guidelines = list(guidelines)
    except:
        return []

    matched_guidelines = []
    for guideline in guidelines:
        guideline_block = guideline.get("Guideline", dict())
        if guideline_block:
            guideline_name = guideline_block.get("code (mapped)", "")
            guideline_name_other = guideline_block.get("other", "")
            for guid, pat in target_guidelines.items():
                if re.search(pat, guideline_name):
                    matched_guidelines.append(guid)
                elif (
                    match_other
                    and guideline_name_other
                    and re.search(pat, guideline_name_other)
                ):
                    matched_guidelines.append(guid)
    return sorted(matched_guidelines)
