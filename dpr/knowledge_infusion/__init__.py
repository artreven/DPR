import importlib

"""
 'Router'-like set of methods for component initialization with lazy imports 
"""

def init_pp_extractor(args, **kwargs):
    from dpr.knowledge_infusion.extractors import PoolPartyExtractor
    return PoolPartyExtractor(**args, **kwargs)

EXTRACTOR_INITIALIZERS = {
    "PoolPartyExtractor" : init_pp_extractor
}


def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError("unsupported model type: {}".format(type))


def init_extractor(extractor_type: str, extractor_args, **kwargs):
    return init_comp(EXTRACTOR_INITIALIZERS, extractor_type, extractor_args, **kwargs)