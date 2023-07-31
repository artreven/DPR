#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.knowledge_infusion.expanders import AbstractEntityExpander
from dpr.knowledge_infusion.extractors import AbstractEntityExtractor
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)

logger = logging.getLogger()
setup_logger(logger)


def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    extractor: AbstractEntityExtractor = None,
    expander: AbstractEntityExpander = None,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    additional_tens_ttt_parameter = {}
    if extractor is not None:
        additional_tens_ttt_parameter["return_offsets"] = True
        additional_tens_ttt_parameter["return_text"] = True
        additional_tens_ttt_parameter["title_concat_str"] = cfg.title_concat_string
    for j, batch_start in enumerate(range(0, n, bsz)):
        batch = ctx_rows[batch_start : batch_start + bsz]

        batch_token_tensors = []
        batch_offset_tensors = []

        for ctx in batch:
            output = tensorizer.text_to_tensor(ctx[1].text,
                                               title=ctx[1].title if (cfg.insert_title and ctx[1].title) else None,
                                          **additional_tens_ttt_parameter)

            if extractor is not None:
                assert (isinstance(output, dict))
                tensor = output['ids']
                offsets = output['offsets']
                ctx_text = output["text"]
                concepts = extractor.extract_no_overlap(ctx_text)

                exp_tensor, positions = expander(token_tensor=tensor,
                                             tensorizer=tensorizer,
                                             offset_map=offsets,
                                             concepts=concepts)
                batch_token_tensors.append(exp_tensor)
                batch_offset_tensors.append(torch.squeeze(positions))
            else:
                batch_token_tensors = output

        ctx_ids_batch = move_to_device(tensorizer.pad_tensor_list(batch_token_tensors), cfg.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch) , cfg.device)
        ctx_pos_batch = move_to_device(tensorizer.pad_tensor_list(batch_offset_tensors) , cfg.device) if extractor is not None else None

        additional_model_params = {"position_ids" : ctx_pos_batch} if ctx_pos_batch is not None else {}
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask, **additional_model_params)
        out = out.cpu()

        ctx_ids = [r[0] for r in batch]
        extra_info = []
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        # TODO: refactor to avoid 'if'
        if extra_info:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i]) for i in range(out.size(0))])
        else:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        if total % 10 == 0:
            logger.info("Encoded passages %d", total)
    return results


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    cfg = setup_cfg_gpu(cfg)

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))
    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    extractor = encoder.extractor if hasattr(encoder, "extractor") else None
    extractor = extractor if cfg.use_concepts else None
    expander = encoder.expander if hasattr(encoder, "expander") else None
    expander = expander if cfg.use_concepts else None
    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state, strict=False)

    logger.info("reading data source: %s", cfg.ctx_src)

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    shard_size = math.ceil(len(all_passages) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx]

    data = gen_ctx_vectors(cfg, shard_passages, encoder, tensorizer, extractor, expander)

    file = cfg.out_file + "_" + str(cfg.shard_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)


if __name__ == "__main__":
    main()
