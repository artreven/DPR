import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor as T

from dpr.utils.data_utils import Tensorizer

logger = logging.getLogger()
# setup_logger(logger)
r_logger = logging.getLogger("root")
r_logger.setLevel(logging.ERROR)


class AbstractEntityExpander:
    def __init__(self, concept_str_transform: str):
        """

        :param concept_str_transform: string that dictates how the concepts are inserted in the context,
        where the concept needs to be represented as "{s}", e.g. "({s})" will put parentheses around the concept.
        """
        self.concept_str_transform = concept_str_transform
        self.ent2toktensor = dict()



    def expand_concepts(self, all_matches, ids_tensor, pos_tensor, tensorizer, offset_map):
        raise NotImplementedError


    def ent_encoder(self, ent_label, tensorizer):
        if ent_label in self.ent2toktensor.keys():
            return self.ent2toktensor[ent_label]
        tens = tensorizer.text_to_tensor(ent_label.lower())
        valid_idx = [x for x in tens.nonzero().squeeze().tolist()
                     if tens[x].greater(999)]
        embd = tens[valid_idx]
        self.ent2toktensor[ent_label] = embd
        return embd


    def __call__(self,
                 token_tensor: T,
                 tensorizer: Tensorizer,
                 offset_map: np.ndarray,
                 concepts: Dict[str, List[Tuple[int, int]]],
                 force_concepts=0,
                 *args,
                 **kwargs):

        default_pos_ids = torch.arange(tensorizer.max_length).expand((1, -1))
        pos_tensor = default_pos_ids[:offset_map.max()]

        logger.debug(f"\n\n-------------------\n Got new document with sizes "
                     f"pos_tensor: {pos_tensor.shape} "
                     f"token_tensor: {token_tensor.shape}")

        if len(concepts) == 0:
            logger.warning(f"No concepts for this document {tensorizer.to_string(token_tensor)[:40]}")
            return token_tensor, pos_tensor

        # sort matches by start offset
        all_matches = []
        for ent, matches in concepts.items():
            trans_ent = self.concept_str_transform.format(s=ent)
            for start_end in matches:
                ent_ids_len = len(self.ent_encoder(trans_ent, tensorizer))
                all_matches.append((trans_ent, start_end, ent_ids_len))
        all_matches.sort(key=lambda x: x[1][0])

        # trimming by removing anything after the SEP
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
            forceable_ids_lens = [ent_ids_len for _, _, ent_ids_len in all_matches[:matches_to_force]]
            sizetotrim = sum(forceable_ids_lens)
            ids_tensor = ids_tensor[:-sizetotrim]
            pos_tensor = pos_tensor[:, :len(ids_tensor)]

        # adding concepts to tensors
        ids_tensor, pos_tensor = self.expand_concepts(all_matches=all_matches,
                                                      ids_tensor=ids_tensor,
                                                      pos_tensor=pos_tensor,
                                                      offset_map=offset_map,
                                                      tensorizer=tensorizer)

        # re-adding SEP token
        ids_tensor = torch.cat((ids_tensor, sep_idx.unsqueeze(0)), 0).long()
        endpos = pos_tensor.max().unsqueeze(0).unsqueeze(0)
        pos_tensor = torch.cat((pos_tensor, endpos + 1), 1).int()

        logger.debug(f"pos_tensor: {pos_tensor.shape}"
                     f"ids_tensor: {ids_tensor.shape}\t<---end\n"
                     f"-------------------------------------")
        assert ids_tensor.shape[0] == pos_tensor.shape[1]

        return ids_tensor, pos_tensor


class EntSuffixPosOnTopEntityExpander(AbstractEntityExpander):
    """
    Concept strings are placed *at the end* of the query.
    Positions of the added concept is *same as the mention* ("on top").
    """


    def expand_concepts(self, all_matches, ids_tensor, pos_tensor, tensorizer, offset_map):
        addedmatches = 0
        for i, (ent, start_end, ent_ids_lens) in enumerate(all_matches):
            ent_ids = self.ent_encoder(ent, tensorizer)
            first_token_match = offset_map[start_end[0]]
            ent_pos = T(range(first_token_match, first_token_match + len(ent_ids)))
            if (first_token_match < 1 or
                    ent_pos.max() > tensorizer.max_length - 1):
                continue
            if ids_tensor.shape[0] + ent_ids_lens > tensorizer.max_length:
                logging.warning(f"Ran of out of space to add matches "
                                f"Only {addedmatches} were added"
                                )
                break
            ent_pos = ent_pos[None, :]  # add a dimension to it
            # Then we actually append it
            ids_tensor = torch.cat((ids_tensor, ent_ids), 0)
            pos_tensor = torch.cat((pos_tensor, ent_pos), 1)

            addedmatches += 1
            logger.debug(f"{ent} |"
                         f"charoffset: {start_end} "
                         f"ids: {ent_ids} "
                         f"tokenoffset: {offset_map[start_end[0]]}")

        logger.debug(f"Added {addedmatches} concept "
                     f"matches, out of a total of {len(all_matches)} ")

        return ids_tensor, pos_tensor


class EntSuffixPosAfterEntityExpander(AbstractEntityExpander):
    """
    Concept strings are placed *at the end* of the query.
    Positions of the added concept is *after the mention*.
    """


    def expand_concepts(self, all_matches, ids_tensor, pos_tensor, tensorizer, offset_map):
        addedmatches = 0
        for i, (ent, start_end, ent_ids_lens) in enumerate(all_matches):
            ent_ids = self.ent_encoder(ent, tensorizer)

            first_token_match = offset_map[start_end[1]] + 1
            ent_pos = T(range(first_token_match, first_token_match + len(ent_ids)))
            if (first_token_match < 1 or
                    ent_pos.max() > tensorizer.max_length - 1):
                continue
            if ids_tensor.shape[0] + ent_ids.shape[0] > tensorizer.max_length:
                logging.warning(f"Ran of out of space to add matches "
                                f"Only {addedmatches} were added"
                                )
                break
            ent_pos = ent_pos[None, :]  # add a dimension to it
            # Then we actually append it
            ids_tensor = torch.cat((ids_tensor, ent_ids), 0)
            pos_tensor = torch.cat((pos_tensor, ent_pos), 1)

            addedmatches += 1
            logger.debug(f"{ent} |"
                         f"charoffset: {start_end} "
                         f"ids: {ent_ids} "
                         f"tokenoffset: {offset_map[start_end[0]]}")

        logger.debug(f"Added {addedmatches} concept "
                     f"matches, out of a total of {len(all_matches)} ")

        return ids_tensor, pos_tensor


class EntInlineEntityExpander(AbstractEntityExpander):
    """
    Concept strings are placed *inline* of the query, i.e., directly after the mention.
    Positions are not altered.
    """


    def expand_concepts(self, all_matches, ids_tensor, pos_tensor, tensorizer, offset_map):
        tokens_added = 0
        addedmatches = 0
        for i, (ent, start_end, ent_ids_lens) in enumerate(all_matches):  # important -> matches must be ordered!
            ent_ids = self.ent_encoder(ent, tensorizer)
            first_token_match = offset_map[start_end[1]] + 1
            ent_pos = T(range(first_token_match, first_token_match + len(ent_ids)))
            if (first_token_match < 1 or
                    ent_pos.max() > tensorizer.max_length - 1):
                continue
            new_start = offset_map[start_end[1]] + 1 + tokens_added  # until end of the mention + new added tokens
            new_end = offset_map[start_end[1]] + 1 + tokens_added  # from end of the mention + new added tokens
            ids_tensor = torch.cat((ids_tensor[:new_start],
                                    ent_ids,
                                    ids_tensor[new_end:]),
                                   0)

            addedmatches += 1
            tokens_added += ent_ids_lens
            logger.debug(f"{ent} |"
                         f"charoffset: {start_end} "
                         f"ids: {ent_ids} "
                         f"tokenoffset: {offset_map[start_end[0]]}")

        logger.debug(f"Added {addedmatches} concept "
                     f"matches, out of a total of {len(all_matches)} ")
        # default positions
        pos_tensor = torch.arange(len(ids_tensor)).expand((1, -1))

        return ids_tensor, pos_tensor
