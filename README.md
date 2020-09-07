# CVML
Implementation of Paper [Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering][0] in [PyTorch][1].

This implementation is slgihtly focused on [VQA_v2.0 DataSet][2]

# Installation

```
pip install git+https://github.com/nikhilkarnwal/CVML.git 
```

# Usage

- Download data from [VQA_v2.0][2] for both training and validation and unzip it.
- Import ailabs library, and first preprocess images and text(question and answers) and then start training.

        import json
        import os
        import torch
        from ailabs.vqa import VQANet
        from ailabs import config
        from ailabs.vqa.src import train, FeatureExtractor
        from ailabs.vqa.src.dataset import process, ptext, pimages
        from ailabs.vqa.src.config import Config

        config.device = 'cuda'
        config.dir = './saved_models'
        if not os.path.exists(config.dir):
            os.mkdir(config.dir)
        config.train_path='./train2014'
        config.val_path='./val2014'
        config.preprocessed_path = os.path.join(config.dir, 'resnet-14x14.h5')
        config.vocabulary_path = os.path.join(config.dir, 'vocab.json')
        config.data_workers = 8
        vqa_v = 'v2'
        config.vqa_data['train_q'] = '{0}_{1}_{2}_{3}_{4}.json'.format(vqa_v, config.task, config.dataset, 'train2014',
                                                                       'questions')
        config.vqa_data['val_q'] = '{0}_{1}_{2}_{3}_{4}.json'.format(vqa_v, config.task, config.dataset, 'val2014',
                                                                     'questions')
        config.vqa_data['train_a'] = '{0}_{2}_{3}_{4}.json'.format(vqa_v, config.task, config.dataset, 'train2014',
                                                                   'annotations')
        config.vqa_data['val_a'] = '{0}_{2}_{3}_{4}.json'.format(vqa_v, config.task, config.dataset, 'val2014',
                                                                 'annotations')
        config.qa_path = './'
        config.preprocess_batch_size=6

        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(json.dumps(config.vqa_data))
        net = FeatureExtractor()
        pimages.execute(net)
        ptext.execute()

        # params for training
        config.batch_size = 64
        config.epochs = 50
        config.resume = None # change it to checkpoint in case you want to resume from there


        def getattr(obj):
            return {name: obj.__getattribute__(name) for name in dir(obj) if not name.startswith('_')}

        # Check all config
        print(json.dumps(getattr(config), indent=4))
        # start training
        train.execute()

# Dependencies

It is build and tested on python=3.7.4
- pytroch=1.6
- tqdm
- h5py

# Author
Nikhil Karnwal
nikhilkarnwal93@gmail.com

[0]: https://arxiv.org/abs/1704.03162
[1]: https://github.com/pytorch/pytorch
[2]: https://visualqa.org/download.html
