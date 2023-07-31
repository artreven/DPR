import importlib

"""
 'Router'-like set of methods for component initialization with lazy imports 
"""

#fixme extend here for other extractors
def init_pp_extractor(args, **kwargs):
    from dpr.knowledge_infusion.extractors import PoolPartyExtractor
    return PoolPartyExtractor(**args, **kwargs)

def init_EntSuffixPosOnTop_expander(args, **kwargs):
    from dpr.knowledge_infusion.expanders import EntSuffixPosOnTopEntityExpander
    return EntSuffixPosOnTopEntityExpander(**args, **kwargs)

def init_EntSuffixPosAfter_expander(args, **kwargs):
    from dpr.knowledge_infusion.expanders import EntSuffixPosAfterEntityExpander
    return EntSuffixPosAfterEntityExpander(**args, **kwargs)

def init_EntInline_expander(args, **kwargs):
    from dpr.knowledge_infusion.expanders import EntInlineEntityExpander
    return EntInlineEntityExpander(**args, **kwargs)


KNOWLEDGE_INFUSION_INITIALIZERS = {
    "PoolPartyExtractor": init_pp_extractor,
    "EntSuffixPosOnTopEntityExpander": init_EntSuffixPosOnTop_expander,
    "EntSuffixPosAfterEntityExpander": init_EntSuffixPosAfter_expander,
    "EntInlineEntityExpander": init_EntInline_expander
}


def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError("unsupported model type: {}".format(type))


def init_extractor(extractor_type: str, extractor_args, **kwargs):
    return init_comp(KNOWLEDGE_INFUSION_INITIALIZERS, extractor_type, extractor_args, **kwargs)


def init_expander(expander_type: str, expander_args, **kwargs):
    return init_comp(KNOWLEDGE_INFUSION_INITIALIZERS, expander_type, expander_args, **kwargs)
