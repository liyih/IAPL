import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import pdb
from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from .dct import DCT_Condition_Module

_tokenizer = _Tokenizer()
       
def load_clip_to_cpu(model_path, n_ctx, adapter_list_vit, adapter_list_text, prompt_depth, gate):

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": n_ctx,
                      "vit_adapter_list": adapter_list_vit,
                      "text_adapter_list": adapter_list_text,
                      "prompt_depth": prompt_depth,
                      "gate": gate,}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    model.float()

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        n_ctx = cfg['N_CTX']

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg['SIZE'][0]
        vision_width = cfg['VISION_WIDTH']
        # Default is 1, which is compound shallow prompting
        self.compound_prompts_depth = cfg['PROMPT_DEPTH']  # max=12, but will create 11 such shared prompts
        
        if self.compound_prompts_depth == 0:
            self.ctx = None
            self.compound_prompts_vision = []
            
        else:
            self.ctx = nn.Parameter(torch.empty(n_ctx, vision_width))
            nn.init.normal_(self.ctx, std=0.02)

            self.compound_prompts_vision = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, vision_width))
                                                        for _ in range(self.compound_prompts_depth - 1)])
            for single_para in self.compound_prompts_vision:
                nn.init.normal_(single_para, std=0.02)


    def forward(self):
        
        return self.ctx, self.compound_prompts_vision   # pass here original, as for visual 768 is required

class LabelSmoothingBCE(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smooth_targets = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        return self.bce(logits, smooth_targets)
                
class CLIPModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        cfg = {'N_CTX': args.n_ctx,
               'PROMPT_DEPTH': args.prompt_depth,
               'SIZE': [args.image_size, args.image_size],
               'VISION_WIDTH': args.vision_width,}
        
        clip_model = load_clip_to_cpu('/Path/to/ViT-L-14.pt', cfg['N_CTX'], args.vit_adapter_list, args.text_adapter_list, args.prompt_depth, args.gate)

        # learnable prompts
        self.prompt_learner = MultiModalPromptLearner(cfg, clip_model)
        self.fc_binary = nn.Linear(768, 1)

        # freezen CLIP
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype

        # do conditional ctx
        if args.condition:
            self.conditional_ctx = DCT_Condition_Module()
        else:
            self.conditional_ctx = None
            
        # freeze unused params.
        if args.tta:
            self.freeze_tta()
        else:
            trained_clip = []
            for param_name, param in self.image_encoder.named_parameters():
                if "adapter" in param_name or "gamma" in param_name:
                    param.requires_grad = True
                    trained_clip.append(param_name)
                else:
                    param.requires_grad = False
            print(trained_clip)

        # loss weight
        self.criterion_weight_dict = {"loss_adapter": args.loss_adapter}
        if args.use_contrast:
            self.criterion_weight_dict["loss_contrast"] = args.loss_contrast
        if args.condition:
            self.criterion_weight_dict["loss_condition"] = args.loss_condition

        # loss function
        if args.smooth:
            self.loss_fn = LabelSmoothingBCE()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def get_criterion(self, outputs, targets):

        loss_dic = {}
        loss_dic["loss_adapter"] = self.loss_fn(outputs[0].squeeze(), targets.float())

        if "loss_contrast" in self.criterion_weight_dict.keys():
            loss_dic["loss_contrast"] = self.contrastive_loss(outputs[1], targets)
        if "loss_condition" in self.criterion_weight_dict.keys():
            loss_dic["loss_condition"] = self.loss_fn(outputs[2].squeeze(), targets.float())

        return loss_dic

    def contrastive_loss(self, embeddings, image_labels, margin=0):
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        fake_index = (image_labels == 1).nonzero(as_tuple=True)[0]
        fake_embeddings = embeddings[fake_index]  # f_N L D
        real_index = (image_labels == 0).nonzero(as_tuple=True)[0]
        real_embeddings = embeddings[real_index]  # N r_L D
        real_nums = len(real_index)
        fake_nums = len(fake_index)

        if (real_nums != 0) and  (fake_nums != 0):
            if fake_nums >= real_nums:
                random_fake_index = torch.randperm(fake_nums)[:real_nums]
                random_fake_embeddings = fake_embeddings[random_fake_index]
                real_neg = (real_embeddings @ random_fake_embeddings.permute(1, 0)) / 0.5
                real_pos = (real_embeddings @ real_embeddings.permute(1, 0) - margin) / 0.5

            else:
                random_real_index = torch.randperm(real_nums)[:fake_nums]
                random_real_embeddings = real_embeddings[random_real_index]
                real_neg = (random_real_embeddings @ fake_embeddings.permute(1, 0)) / 0.5
                real_pos = (random_real_embeddings @ random_real_embeddings.permute(1, 0) - margin) / 0.5
            loss_real_neg = torch.sum(torch.exp(real_neg))
            loss_real_pos = torch.sum(torch.exp(real_pos))
            loss_inter = -torch.log(loss_real_pos / (loss_real_pos + loss_real_neg))

        else: 
            loss_inter = 0

        return loss_inter
    def forward(self, image):

        image_vp = image.type(self.dtype)

        shared_ctx, deep_compound_prompts_vision = self.prompt_learner()
        
        if self.conditional_ctx is not None:
            bias, pred_bias = self.conditional_ctx(image_vp)
            shared_ctx = shared_ctx.expand(image_vp.shape[0], -1, -1)
            shared_ctx = shared_ctx + bias
        else:
            if shared_ctx is not None:
                shared_ctx = shared_ctx.expand(image_vp.shape[0], -1, -1)

        image_features, feat_bank = self.image_encoder(image_vp, shared_ctx, deep_compound_prompts_vision)
        
        logits = self.fc_binary(image_features)
        
        if self.training:
            if self.conditional_ctx is not None:
                return [logits, image_features, pred_bias]
            else:
                return [logits, image_features]
        else:
            return logits

    def freeze_tta(self):
        for param_name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        for param_name, param in self.prompt_learner.named_parameters():
            if "ctx" not in param_name:
                param.requires_grad = False
        for param_name, param in self.fc_binary.named_parameters():
            param.requires_grad = False
        for param_name, param in self.conditional_ctx.named_parameters():
            param.requires_grad = False
        print('-----------freezen TTA mode-----------')
        
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])