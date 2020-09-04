from .pimages import execute as i_execute
from .ptext import execute as t_execute
from ailabs import config


def process(feature_extractor):
    if config.dataset == 'vqa2.0':
        i_execute(feature_extractor)
        t_execute()
