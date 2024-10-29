# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:18:04 2024

# Code related to zeroshot classifier is mainly ported from
# https://github.com/mlfoundations/wise-ft/blob/master/src/models/zeroshot.py
# https://github.com/EfficientTraining/LabelBench/blob/main/LabelBench/model/model_impl/clip.py

@author: wenzt
"""

import clip
import torch
import torch.nn as nn

# from tqdm import tqdm
# import templates as templates

# class ClassificationHead(torch.nn.Linear):
#     def __init__(self, normalize, weights, biases=None):
#         output_size, input_size = weights.shape
#         super().__init__(input_size, output_size)
#         self.normalize = normalize
#         if weights is not None:
#             self.weight = torch.nn.Parameter(weights.clone())
#         if biases is not None:
#             self.bias = torch.nn.Parameter(biases.clone())
#         else:
#             self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

#     def forward(self, inputs):
#         if self.normalize:
#             inputs = inputs / inputs.norm(dim=-1, keepdim=True)
#         return super().forward(inputs)

# def get_zeroshot_classifier(clip_model, tokenizer, classnames, template):
#     assert template is not None, 'template is required for zeroshot classifier.'
#     assert classnames is not None, 'classnames is required for zeroshot classifier.'
#     template = getattr(templates, template)
#     logit_scale = clip_model.logit_scale

#     clip_model.eval()
#     clip_model.cuda()

#     print('Getting zeroshot weights.')
#     with torch.no_grad():
#         zeroshot_weights = []
#         for classname in tqdm(classnames):
#             texts = []
#             for t in template:
#                 texts.append(t(classname))
#             texts = tokenizer(texts).cuda()  # Tokenize.
#             embeddings = clip_model.encode_text(texts)  # Embed with text encoder.
#             embeddings /= embeddings.norm(dim=-1, keepdim=True)

#             embeddings = embeddings.mean(dim=0, keepdim=True)
#             embeddings /= embeddings.norm()

#             zeroshot_weights.append(embeddings)

#         zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
#         zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

#         zeroshot_weights *= logit_scale.exp()

#         zeroshot_weights = zeroshot_weights.squeeze().float()
#         zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

#     classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

#     return classification_head


class CLIPBackbone(nn.Module):
    def __init__(self, args):

        super(CLIPBackbone, self).__init__()
        # assert pretrain, "CLIPVisionOnly only support pretrain model"
        model_name = args.network[5:]
        model, _ = clip.load(model_name, download_root=args.selfmodel_path)
        self.image_encoder_model = model.float()  # Convert to float to avoid NAN loss when using AdamW.
        self.embed_dim = model.state_dict()["text_projection"].shape[1]


    def forward(self, imgs):
        return self.image_encoder_model.encode_image(imgs)

    def get_embedding_dim(self):
        return self.embed_dim



# CLIPVisionOnly(model_config["num_output"],
#                           ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
#                           pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
#                           model_name="ViT-B/16")

# CLIPVisionOnly(model_config["num_output"],
#                           ret_emb=model_config["ret_emb"] if "ret_emb" in model_config else False,
#                           pretrain=model_config["pretrain"] if "pretrain" in model_config else True,
#                           model_name="ViT-B/32")