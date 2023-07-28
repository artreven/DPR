import enum
import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
from torch import Tensor as T

from dpr.utils.data_utils import Tensorizer

logger = logging.getLogger()
# setup_logger(logger)
r_logger = logging.getLogger("root")
r_logger.setLevel(logging.ERROR)

# fixme this should be global?
cpt2toktensor = dict()


class ConceptExpansionStrategy(enum.Enum):
    CONCEPT_SUFFIX_POS_ON_TOP = "concept_suffix_pos_on_top"


# expand at end
#   ids_tensor = torch.cat((ids_tensor, cpt_ids), 0)
# expand at start
#   ids_tensor = torch.cat((cpt_ids, ids_tensor), 0)
# expand inline
#   ids_tensor = torch.cat((ids_tensor[:offset_map[start_end[1]] + 1],  # until end of the mention
#                           cpt_ids,
#                           ids_tensor[offset_map[start_end[1]] + 1:]),  # from end of the mention
#                          0)
# expand replace
#   ids_tensor = torch.cat((ids_tensor[:offset_map[start_end[0]]],  # until start of the mention
#                           cpt_ids,
#                           ids_tensor[offset_map[start_end[1]] + 1:]),  # from end of the mention
#                          0)

@dataclass()
class ConceptExpander:
    concept_str_transform: Callable
    tensorizer: Tensorizer
    maxlen: int = 512,


    def expand_concepts(self, all_matches, ids_tensor, pos_tensor, offset_map):
        raise NotImplementedError


    def cpt_encoder(self, cpt_label):
        if cpt_label in cpt2toktensor.keys():
            return cpt2toktensor[cpt_label]
        tens = self.tensorizer.text_to_tensor(cpt_label.lower())
        valid_idx = [x for x in tens.nonzero().squeeze().tolist()
                     if tens[x].greater(999)]
        embd = tens[valid_idx]
        cpt2toktensor[cpt_label] = embd
        return embd


    def __call__(self,
                 token_tensor: T,
                 offset_map: np.ndarray,
                 concepts: Dict[str, List[Tuple[int, int]]],
                 force_concepts=0,
                 *args,
                 **kwargs):

        default_pos_ids = torch.arange(self.maxlen).expand((1, -1))
        pos_tensor = default_pos_ids[:offset_map.max()]

        logger.debug(f"\n\n-------------------\n Got new document with sizes "
                     f"pos_tensor: {pos_tensor.shape} "
                     f"token_tensor: {token_tensor.shape}")

        if len(concepts) == 0:
            logger.warning(f"No concepts for this document {self.tensorizer.to_string(token_tensor)[:40]}")
            return token_tensor, pos_tensor

        # sort matches by start offset
        all_matches = []
        for cpt, matches in concepts.items():
            trans_cpt = self.concept_str_transform(cpt)
            for start_end in matches:
                cpt_ids_len = len(self.cpt_encoder(trans_cpt))
                all_matches.append((trans_cpt, start_end, cpt_ids_len))
        all_matches.sort(key=lambda x: x[1][0])

        #trimming by removing anything after the SEP
        sep_idx = token_tensor[-1]
        sep_positions = (token_tensor == token_tensor[-1]).nonzero().squeeze().tolist()
        if isinstance(sep_positions, int):
            sep_positions = [sep_positions, sep_positions]
        assert len(sep_positions) > 0
        ids_tensor = token_tensor[:sep_positions[0]]
        pos_tensor = pos_tensor[:, :len(ids_tensor)]
        logger.debug(f"ids_tensor: {ids_tensor.shape}"
                     f"pos_tensor: {pos_tensor.shape}\t<---trimming")

        # further (right-sided) trimming for forced concept inclusion
        matches_to_force = min([force_concepts, len(all_matches)])
        if matches_to_force > 0:
            forceable_ids_lens = [cpt_ids_len for _, _, cpt_ids_len in all_matches[:matches_to_force]]
            sizetotrim = sum(forceable_ids_lens)
            ids_tensor = ids_tensor[:-sizetotrim]
            pos_tensor = pos_tensor[:, :len(ids_tensor)]

        # adding concepts to tensors
        ids_tensor, pos_tensor = self.expand_concepts(all_matches=all_matches,
                                                      ids_tensor=ids_tensor,
                                                      pos_tensor=pos_tensor,
                                                      offset_map=offset_map)

        # re-adding SEP token
        ids_tensor = torch.cat((ids_tensor, sep_idx.unsqueeze(0)), 0).long()
        endpos = pos_tensor.max().unsqueeze(0).unsqueeze(0)
        pos_tensor = torch.cat((pos_tensor, endpos + 1), 1).int()

        logger.debug(f"pos_tensor: {pos_tensor.shape}"
                     f"ids_tensor: {ids_tensor.shape}\t<---end\n"
                     f"-------------------------------------")
        assert ids_tensor.shape[0] == pos_tensor.shape[1]

        return ids_tensor, pos_tensor


class CptSuffixPosOnTopConceptExpander(ConceptExpander):
    """
    Concept strings are placed *at the end* of the query.
    Positions of the added concept is *same as the mention* ("on top").
    """


    def expand_concepts(self, all_matches, ids_tensor, pos_tensor, offset_map):
        addedmatches = 0
        for i, (cpt, start_end, cpt_ids_lens) in enumerate(all_matches):
            cpt_ids = self.cpt_encoder(cpt)
            first_token_match = offset_map[start_end[0]]
            cpt_pos = T(range(first_token_match, first_token_match + len(cpt_ids)))
            if (first_token_match < 1 or
                    cpt_pos.max() > self.maxlen - 1):
                continue
            if ids_tensor.shape[0] + cpt_ids_lens > self.maxlen:
                logging.warning(f"Ran of out of space to add matches "
                                f"Only {addedmatches} were added"
                                )
                break
            cpt_pos = cpt_pos[None, :]  # add a dimension to it
            # Then we actually append it
            ids_tensor = torch.cat((ids_tensor, cpt_ids), 0)
            pos_tensor = torch.cat((pos_tensor, cpt_pos), 1)

            addedmatches += 1
            logger.debug(f"{cpt} |"
                         f"charoffset: {start_end} "
                         f"ids: {cpt_ids} "
                         f"tokenoffset: {offset_map[start_end[0]]}")

        logger.debug(f"Added {addedmatches} concept "
                     f"matches, out of a total of {len(all_matches)} ")

        return ids_tensor, pos_tensor


class CptSuffixPosAfterConceptExpander(ConceptExpander):
    """
    Concept strings are placed *at the end* of the query.
    Positions of the added concept is *after the mention*.
    """


    def expand_concepts(self, all_matches, ids_tensor, pos_tensor, offset_map):
        addedmatches = 0
        for i, (cpt, start_end, cpt_ids_lens) in enumerate(all_matches):
            cpt_ids = self.cpt_encoder(cpt)

            first_token_match = offset_map[start_end[1]] + 1
            cpt_pos = T(range(first_token_match, first_token_match + len(cpt_ids)))
            if (first_token_match < 1 or
                    cpt_pos.max() > self.maxlen - 1):
                continue
            if ids_tensor.shape[0] + cpt_ids.shape[0] > self.maxlen:
                logging.warning(f"Ran of out of space to add matches "
                                f"Only {addedmatches} were added"
                                )
                break
            cpt_pos = cpt_pos[None, :]  # add a dimension to it
            # Then we actually append it
            ids_tensor = torch.cat((ids_tensor, cpt_ids), 0)
            pos_tensor = torch.cat((pos_tensor, cpt_pos), 1)

            addedmatches += 1
            logger.debug(f"{cpt} |"
                         f"charoffset: {start_end} "
                         f"ids: {cpt_ids} "
                         f"tokenoffset: {offset_map[start_end[0]]}")

        logger.debug(f"Added {addedmatches} concept "
                     f"matches, out of a total of {len(all_matches)} ")

        return ids_tensor, pos_tensor


class CptInlineConceptExpander(ConceptExpander):
    """
    Concept strings are placed *inline* of the query, i.e., directly after the mention.
    Positions are not altered.
    """


    def expand_concepts(self, all_matches, ids_tensor, pos_tensor, offset_map):
        tokens_added = 0
        addedmatches = 0
        for i, (cpt, start_end, cpt_ids_lens) in enumerate(all_matches):  # important -> matches must be ordered!
            cpt_ids = self.cpt_encoder(cpt)
            first_token_match = offset_map[start_end[1]] + 1
            cpt_pos = T(range(first_token_match, first_token_match + len(cpt_ids)))
            if (first_token_match < 1 or
                    cpt_pos.max() > self.maxlen - 1):
                continue
            new_start = offset_map[start_end[1]] + 1 + tokens_added  # until end of the mention + new added tokens
            new_end = offset_map[start_end[1]] + 1 + tokens_added  # from end of the mention + new added tokens
            ids_tensor = torch.cat((ids_tensor[:new_start],
                                    cpt_ids,
                                    ids_tensor[new_end:]),
                                   0)

            addedmatches += 1
            tokens_added += cpt_ids_lens
            logger.debug(f"{cpt} |"
                         f"charoffset: {start_end} "
                         f"ids: {cpt_ids} "
                         f"tokenoffset: {offset_map[start_end[0]]}")

        logger.debug(f"Added {addedmatches} concept "
                     f"matches, out of a total of {len(all_matches)} ")
        # default positions
        pos_tensor = torch.arange(len(ids_tensor)).expand((1, -1))

        return ids_tensor, pos_tensor


def _cpt_translator(cpt_label, tensorizer):
    if cpt_label in cpt2toktensor.keys():
        return cpt2toktensor[cpt_label]
    tens = tensorizer.text_to_tensor(cpt_label.lower())
    valid_idx = [x for x in tens.nonzero().squeeze().tolist()
                 if tens[x].greater(999)]
    embd = tens[valid_idx]
    cpt2toktensor[cpt_label] = embd
    return embd


def _add_positions(text: str,
                   token_tensor: T,
                   offset_map: np.ndarray,
                   concepts: Dict[str, List[Tuple[int, int]]],
                   tensorizer: Tensorizer,
                   maxlen: int = 512,
                   force_concepts=0,
                   concepts_after_match=False) -> Tuple[T, T]:
    """

    :param text: the string that is being vectorized
    :param token_tensor: a 1 x num_tokens tensor. The i'th entry is the
    id of the i'th token, in the Bert vocabulary. Notice that Bert adds
    a special token at the start of every text.
    :param offset_map: a 1 x len(text) numpy array with values in
    [0,numtokens]. The j'th entry specifies which token does the j'th
    character belong to, or 0 if none (e.g. a space).
    :param concepts
    :param tensorizer
    :param maxlen
    :param force_concepts
    :param concepts_after_match
    :return:
    """
    expander = CptInlineConceptExpander(lambda s: f"({s})", tensorizer, maxlen)
    alt_ids_tensor, alt_pos_tensor = expander(token_tensor,offset_map, concepts,force_concepts)


    # note: this was originally not maxlen, i.e., tensorizer.max_length but rather maxposembed  with
    # maxposembed = config_dict.get("max_position_embeddings", 256)
    # if self.tensorizer.max_length > maxposembed:
    #     maxposembed = self.tensorizer.max_length
    # we hope that maxposembed and maxlen are the same, otherwise, start here to debug :)
    default_pos_ids = torch.arange(maxlen).expand((1, -1))

    pos_tensor = default_pos_ids[:offset_map.max()]
    # logger.debug(f"\npos_tensor {pos_tensor}"
    #              f"\ntoken_tensor {token_tensor}")
    # logger.debug(f"pos_tensor: {pos_tensor.shape}"
    #              f"token_tensor: {token_tensor.shape}"
    #              f"\t<-- start")
    logger.debug(f"\n\n-------------------\n Got new document with sizes "
                 f"pos_tensor: {pos_tensor.shape} "
                 f"token_tensor: {token_tensor.shape}")

    if len(concepts) == 0:
        logger.warning(f"No concepts for this document {text[:40]}")
        return token_tensor, pos_tensor

    # First we trim the ids an position tensors by removing those indices
    #  that are not actually used
    endid = token_tensor[-1]
    ends = (token_tensor == token_tensor[-1]).nonzero().squeeze().tolist()

    # The tolist() method doesn't actually return a list when the tensor
    # is 0-dimensional ;)

    if isinstance(ends, int):
        ends = [ends, ends]
    assert len(ends) > 0
    ids_tensor = token_tensor[:ends[0]]
    pos_tensor = pos_tensor[:, :len(ids_tensor)]
    logger.debug(f"ids_tensor: {ids_tensor.shape}"
                 f"pos_tensor: {pos_tensor.shape}\t<---trimming")
    trimmedshape = pos_tensor.shape

    # For every concept match, we append the concept's indices to the id's
    #  (there can be more than one index, think of multitoken labels)
    # and position of the concept match to the position tensor
    skip_concepts = False
    logger.debug(f"Will now add {len(concepts)} concepts ")
    addedmatches = 0
    all_matches = []
    # We collapse all matches into a single giant list, so that we can
    # sort them by start offset
    for cpt, matches in concepts.items():
        for start_end in matches:
            all_matches.append((cpt, start_end))
    all_matches.sort(key=lambda x: x[1][0])

    # If want to force some matches to appear in the tensorization,
    # even if all the positions are already occupied, we first compute
    # their ids, and then trim the ids and pos tensor to make space for
    # them
    matches_to_force = min([force_concepts,
                            len(all_matches)])
    if matches_to_force > 0:
        forceable_ids = [_cpt_translator(cpt, tensorizer)
                         for cpt, _ in all_matches[:matches_to_force]]
        sizetotrim = sum(fid.shape[0]
                         for fid in forceable_ids)
        ids_tensor = token_tensor[:-sizetotrim]
        pos_tensor = pos_tensor[:, :len(ids_tensor)]

    logger.debug(f"In total there are {len(all_matches)} cpt matches\n---")
    for cpt, start_end in all_matches:
        # First we prepare the vectors to append
        cpt_ids = _cpt_translator(cpt, tensorizer)
        # On top of the match
        first_token_match = offset_map[start_end[0]]
        if concepts_after_match:
            first_token_match = offset_map[start_end[1]] + 1
        cpt_pos = T([first_token_match + i  # cpt token start
                     for i in range(len(cpt_ids))])
        if (first_token_match < 1 or
                cpt_pos.max() > maxlen - 1):
            continue
        if ids_tensor.shape[0] + cpt_ids.shape[0] > maxlen:
            logging.warning(f"Ran of out of space to add matches "
                            f"Only {addedmatches} were added, with "
                            f"{matches_to_force} of them forced. "
                            f"Trimmed shape was {trimmedshape}"
                            )
            break
        cpt_pos = cpt_pos[None, :]  # add a dimension to it
        # Then we actually append it
        ids_tensor = torch.cat((ids_tensor, cpt_ids), 0)
        pos_tensor = torch.cat((pos_tensor, cpt_pos), 1)
        addedmatches += 1
        logger.debug(f"{cpt} |"
                     f"charoffset: {start_end} "
                     f"ids: {cpt_ids} "
                     f"tokenoffset: {offset_map[start_end[0]]}")

    logger.debug(f"Added {addedmatches} concept "
                 f"matches, out of a total of {len(all_matches)} ")
    # We put into the ids_tensor the end-token (102)
    ids_tensor = torch.cat((ids_tensor, endid.unsqueeze(0)), 0).long()
    # And add one more entry to the pos_tensor, equal to its max+1
    endpos = pos_tensor.max().unsqueeze(0).unsqueeze(0)
    pos_tensor = torch.cat((pos_tensor, endpos + 1), 1).int()

    # logger.debug(f"\npos_tensor {pos_tensor}"
    #              f"\ntoken_tensor {ids_tensor}")
    logger.debug(f"pos_tensor: {pos_tensor.shape}"
                 f"ids_tensor: {ids_tensor.shape}\t<---end\n"
                 f"-------------------------------------")
    assert ids_tensor.shape[0] == pos_tensor.shape[1]

    #assert torch.equal(ids_tensor, alt_ids_tensor)
    #assert torch.equal(pos_tensor, alt_pos_tensor)

    return ids_tensor, pos_tensor
