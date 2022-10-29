import logging
import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils

logger = logging.getLogger(__name__)

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # ===================== Downstream ===================== #

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]
        
        if self.hparams.config["loss_names"]["snli_ve"] > 0:
            self.ve_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.ve_classifier.apply(objectives.init_weights)

        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            load_from_distilled_model = False
            if self.hparams.config["loss_names"]["nlvr2"] > 0 and state_dict['token_type_embeddings.weight'].size(0) != self.token_type_embeddings.weight.size(0):
                old_emb_data = state_dict['token_type_embeddings.weight'].data
                new_emb_weight = nn.Embedding(3, hs).weight
                new_emb_weight.data[0, :] = old_emb_data[0, :]
                new_emb_weight.data[1, :] = old_emb_data[1, :]
                new_emb_weight.data[2, :] = old_emb_data[1, :]
                state_dict['token_type_embeddings.weight'] = new_emb_weight
                
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                logger.info("Weights not initialized from pretrained model: {}".format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used: {}".format(unexpected_keys))

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                logger.info("Weights not initialized from pretrained model: {}".format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used: {}".format(unexpected_keys))
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        pre_select=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
                select,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
                pre_select=pre_select,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        all_hidden_states = []
        all_attn_states = []
        all_q_states = []
        all_k_states = []
        all_v_states = []
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn, norm1_x, q, k, v = blk(x, mask=co_masks)
            all_hidden_states.append(norm1_x)
            all_attn_states.append(_attn)
            all_q_states.append(q)
            all_k_states.append(k)
            all_v_states.append(v)

        x = self.transformer.norm(x)
        all_hidden_states.append(x)

        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
            "co_masks": co_masks,
            "all_hidden_states": all_hidden_states,
            "all_attention_states": all_attn_states,
            "all_q_states": all_q_states,
            "all_k_states": all_k_states,
            "all_v_states": all_v_states,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Visual Entailment
        if "snli_ve" in self.current_tasks:
            ret.update(objectives.compute_ve(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        save_path = "/".join(self.hparams.config["load_path"].split("/")[:-1])
        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, save_path)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)

class TwoTowerViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        tower_config = copy.deepcopy(config)
        for k in tower_config["loss_names"].keys():
            if tower_config["loss_names"][k] > 0:
                tower_config["loss_names"][k] = 0
        self.image_encoder = ViLTransformerSS(tower_config)
        self.text_encoder = ViLTransformerSS(tower_config)      

        # ===================== Downstream ===================== #

        hs = self.hparams.config["hidden_size"]
        
        # fusion different modal input part
        self.fusion = nn.Sequential(
            nn.Linear(hs * 2, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, hs),
        )
        self.fusion.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.image_encoder.token_type_embeddings.weight.data
            self.image_encoder.token_type_embeddings = nn.Embedding(3, hs)
            self.image_encoder.token_type_embeddings.apply(objectives.init_weights)
            self.image_encoder.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.image_encoder.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.image_encoder.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["snli_ve"] > 0:
            self.ve_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.ve_classifier.apply(objectives.init_weights)

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                logger.info("Weights not initialized from pretrained model: {}".format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used: {}".format(unexpected_keys))

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        mode="multi_modal"
    ):
        if mode == "text":
            do_mlm = "_mlm" if mask_text else ""
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]
            text_embeds = self.text_encoder.text_embeddings(text_ids)

            text_embeds = text_embeds + self.text_encoder.token_type_embeddings(torch.zeros_like(text_masks))

            x = text_embeds
            all_hidden_states = []
            all_attn_states = []
            all_q_states = []
            all_k_states = []
            all_v_states = []
            for i, blk in enumerate(self.text_encoder.transformer.blocks):
                x, _attn, norm1_x, q, k, v = blk(x, mask=text_masks)
                all_hidden_states.append(norm1_x)
                all_attn_states.append(_attn)
                all_q_states.append(q)
                all_k_states.append(k)
                all_v_states.append(v)

            x = self.text_encoder.transformer.norm(x)
            all_hidden_states.append(x)
            cls_feats = self.text_encoder.pooler(x)


            ret = {
                "text_feats": x,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "all_hidden_states": all_hidden_states,
                "all_attention_states": all_attn_states,
                "all_q_states": all_q_states,
                "all_k_states": all_k_states,
                "all_v_states": all_v_states,
            }
            return ret

        elif mode == "image":
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            if image_embeds is None and image_masks is None:
                img = batch[imgkey][0]
                (
                    image_embeds,
                    image_masks,
                    patch_index,
                    image_labels,
                    select,
                ) = self.image_encoder.transformer.visual_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image,
                )
            else:
                patch_index, image_labels = (
                    None,
                    None,
                )
            
            image_embeds = image_embeds + self.image_encoder.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            x = image_embeds
            all_hidden_states = []
            all_attn_states = []
            all_q_states = []
            all_k_states = []
            all_v_states = []
            for i, blk in enumerate(self.image_encoder.transformer.blocks):
                x, _attn, norm1_x, q, k, v = blk(x, mask=image_masks)
                all_hidden_states.append(norm1_x)
                all_attn_states.append(_attn)
                all_q_states.append(q)
                all_k_states.append(k)
                all_v_states.append(v)

            x = self.image_encoder.transformer.norm(x)
            all_hidden_states.append(x)

            cls_feats = self.image_encoder.pooler(x)
            
            ret = {
                "image_feats": x,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "image_labels": image_labels,
                "image_masks": image_masks,
                "patch_index": patch_index,
                "all_hidden_states": all_hidden_states,
                "all_attention_states": all_attn_states,
                "all_q_states": all_q_states,
                "all_k_states": all_k_states,
                "all_v_states": all_v_states,
            }

            return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_twotower_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_twotower_nlvr2(self, batch))

        # Visual Entailment
        if "snli_ve" in self.current_tasks:
            ret.update(objectives.compute_twotower_ve(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        save_path = "/".join(self.hparams.config["load_path"].split("/")[:-1])
        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, save_path)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)

class DstTwoTowerViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        tower_config = copy.deepcopy(config)
        for k in tower_config["loss_names"].keys():
            if tower_config["loss_names"][k] > 0:
                tower_config["loss_names"][k] = 0
        logger.info("**** Image Encoder ****")
        self.image_encoder = ViLTransformerSS(tower_config)
        logger.info("**** Text Encoder ****")
        self.text_encoder = ViLTransformerSS(tower_config)

        # multimodal encoder as teacher
        # teacher config
        teacher_config = copy.deepcopy(config)
        teacher_config["vit"] = config["teacher_vit"]
        teacher_config["hidden_size"] = teacher_config["teacher_hidden_size"]
        teacher_config["load_path"] = teacher_config["teacher_load_path"]
        self.teacher_model = ViLTransformerSS(teacher_config)
        if config["fixed_teacher"]:
            for p in self.teacher_model.parameters():
                p.requires_grad = False 
    
        # ===================== Downstream ===================== #

        hs = self.hparams.config["hidden_size"]
        
        # fusion different modal input part
        self.fusion = nn.Sequential(
            nn.Linear(hs * 2, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, hs),
        )
        self.fusion.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.image_encoder.token_type_embeddings.weight.data
            self.image_encoder.token_type_embeddings = nn.Embedding(3, hs)
            self.image_encoder.token_type_embeddings.apply(objectives.init_weights)
            self.image_encoder.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.image_encoder.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.image_encoder.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["snli_ve"] > 0:
            self.ve_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.ve_classifier.apply(objectives.init_weights)

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                logger.info("Weights not initialized from pretrained model: {}".format(missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used: {}".format(unexpected_keys))

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        mode="multi_modal"
    ):
        if mode == "text":
            do_mlm = "_mlm" if mask_text else ""
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]
            text_embeds = self.text_encoder.text_embeddings(text_ids)

            text_embeds = text_embeds + self.text_encoder.token_type_embeddings(torch.zeros_like(text_masks))

            x = text_embeds
            all_hidden_states = []
            all_attn_states = []
            all_q_states = []
            all_k_states = []
            all_v_states = []
            for i, blk in enumerate(self.text_encoder.transformer.blocks):
                x, _attn, norm1_x, q, k, v = blk(x, mask=text_masks)
                all_hidden_states.append(norm1_x)
                all_attn_states.append(_attn)
                all_q_states.append(q)
                all_k_states.append(k)
                all_v_states.append(v)

            x = self.text_encoder.transformer.norm(x)
            all_hidden_states.append(x)
            cls_feats = self.text_encoder.pooler(x)
            
            ret = {
                "text_feats": x,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "all_hidden_states": all_hidden_states,
                "all_attention_states": all_attn_states,
                "all_q_states": all_q_states,
                "all_k_states": all_k_states,
                "all_v_states": all_v_states,
            }
            return ret

        elif mode == "image":
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            if image_embeds is None and image_masks is None:
                img = batch[imgkey][0]
                (
                    image_embeds,
                    image_masks,
                    patch_index,
                    image_labels,
                    select,
                ) = self.image_encoder.transformer.visual_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image,
                )
            else:
                patch_index, image_labels = (
                    None,
                    None,
                )
            
            image_embeds = image_embeds + self.image_encoder.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            x = image_embeds
            all_hidden_states = []
            all_attn_states = []
            all_q_states = []
            all_k_states = []
            all_v_states = []
            for i, blk in enumerate(self.image_encoder.transformer.blocks):
                x, _attn, norm1_x, q, k, v = blk(x, mask=image_masks)
                all_hidden_states.append(norm1_x)
                all_attn_states.append(_attn)
                all_q_states.append(q)
                all_k_states.append(k)
                all_v_states.append(v)

            x = self.image_encoder.transformer.norm(x)
            all_hidden_states.append(x)

            cls_feats = self.image_encoder.pooler(x)

            ret = {
                "image_feats": x,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "image_labels": image_labels,
                "image_masks": image_masks,
                "patch_index": patch_index,
                "all_hidden_states": all_hidden_states,
                "all_attention_states": all_attn_states,
                "all_q_states": all_q_states,
                "all_k_states": all_k_states,
                "all_v_states": all_v_states,
                "select": select,
            }

            return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            if self.training:   # teacher model is only used in training phase
                ret.update(objectives.compute_twotower_dst_vqa(self, batch))
            else:
                ret.update(objectives.compute_twotower_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            if self.training:
                ret.update(objectives.compute_twotower_dst_nlvr2(self, batch))
            else:
                ret.update(objectives.compute_twotower_nlvr2(self, batch))

        # Visual Entailment
        if "snli_ve" in self.current_tasks:
            if self.training:
                ret.update(objectives.compute_twotower_dst_ve(self, batch))
            else:
                ret.update(objectives.compute_twotower_ve(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        save_path = "/".join(self.hparams.config["load_path"].split("/")[:-1])
        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, save_path)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)