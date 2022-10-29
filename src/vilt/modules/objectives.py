import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from vilt.modules.dist_utils import all_gather

def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret

def compute_twotower_vqa(pl_module, batch):
    text_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="text")
    image_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="image")
    fusion_cls_feats = pl_module.fusion(torch.cat((text_infer["cls_feats"], image_infer["cls_feats"]), dim=-1))
    vqa_logits = pl_module.vqa_classifier(fusion_cls_feats)
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret

def compute_twotower_dst_vqa(pl_module, batch):
    text_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="text")
    image_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="image")
    fusion_cls_feats = pl_module.fusion(torch.cat((text_infer["cls_feats"], image_infer["cls_feats"]), dim=-1))
    student_vqa_logits = pl_module.vqa_classifier(fusion_cls_feats)
    # teacher part
    pl_module.teacher_model.eval()
    with torch.no_grad():
        teacher_infer = pl_module.teacher_model.infer(batch, mask_text=False, mask_image=False, pre_select=image_infer["select"])
        teacher_vqa_logits = pl_module.teacher_model.vqa_classifier(teacher_infer["cls_feats"])
    # hard label
    vqa_targets = torch.zeros(
        len(student_vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(student_vqa_logits.float(), vqa_targets.float())
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19
    vqa_loss = vqa_loss.data.new([0.0]).squeeze(-1)
    sl_vqa_loss = F.binary_cross_entropy_with_logits(student_vqa_logits.float(), torch.sigmoid(teacher_vqa_logits.float().detach())) * vqa_targets.shape[1]
    attention_loss = vqa_loss.data.new([0.0]).squeeze(-1)
    attention_loss += compute_twotower_cross_modal_attention_loss(text_infer, image_infer, teacher_infer, scale=pl_module.text_encoder.transformer.blocks[0].attn.scale, )

    ret = {
        "vqa_loss": vqa_loss,
        "soft_vqa_loss": sl_vqa_loss,
        "attention_loss": attention_loss,
        "vqa_logits": student_vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    soft_vqa_loss =  getattr(pl_module, f"{phase}_soft_vqa_loss")(ret["soft_vqa_loss"])
    attention_vqa_loss = getattr(pl_module, f"{phase}_attention_vqa_loss")(ret["attention_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/soft_vqa_loss", soft_vqa_loss)
    pl_module.log(f"vqa/{phase}/attention_vqa_loss", attention_vqa_loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret

def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
            batch, mask_text=False, mask_image=False, image_token_type_idx=1
        )
    infer2 = pl_module.infer(
            batch, mask_text=False, mask_image=False, image_token_type_idx=2
        )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret

def compute_twotower_nlvr2(pl_module, batch):
    text_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="text")
    image_infer1 = pl_module.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1, mode="image")
    image_infer2 = pl_module.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2, mode="image")
    fusion_infer1_cls_feats = pl_module.fusion(torch.cat((text_infer["cls_feats"], image_infer1["cls_feats"]), dim=-1))
    fusion_infer2_cls_feats = pl_module.fusion(torch.cat((text_infer["cls_feats"], image_infer2["cls_feats"]), dim=-1))
    cls_feats = torch.cat([fusion_infer1_cls_feats, fusion_infer2_cls_feats], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret

def compute_twotower_dst_nlvr2(pl_module, batch):
    # student part
    text_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="text")
    image_infer1 = pl_module.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1, mode="image")
    image_infer2 = pl_module.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2, mode="image")
    fusion_infer1_cls_feats = pl_module.fusion(torch.cat((text_infer["cls_feats"], image_infer1["cls_feats"]), dim=-1))
    fusion_infer2_cls_feats = pl_module.fusion(torch.cat((text_infer["cls_feats"], image_infer2["cls_feats"]), dim=-1))
    student_cls_feats = torch.cat([fusion_infer1_cls_feats, fusion_infer2_cls_feats], dim=-1)
    student_nlvr2_logits = pl_module.nlvr2_classifier(student_cls_feats)
    # teacher_part
    pl_module.teacher_model.eval()
    with torch.no_grad():
        teacher_infer1 = pl_module.teacher_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1, pre_select=image_infer1["select"])
        teacher_infer2 = pl_module.teacher_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2, pre_select=image_infer2["select"])

        teacher_cls_feats = torch.cat([teacher_infer1["cls_feats"], teacher_infer2["cls_feats"]], dim=-1)
        teacher_nlvr2_logits = pl_module.teacher_model.nlvr2_classifier(teacher_cls_feats)

    # hard label
    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(student_nlvr2_logits, nlvr2_labels)
    nlvr2_loss = nlvr2_loss.data.new([0.0]).squeeze(-1)
    sl_nlvr2_loss = F.kl_div(F.log_softmax(student_nlvr2_logits.float(), dim=-1) , F.softmax(teacher_nlvr2_logits.float(), dim=-1), reduction="batchmean")
    attention_loss = nlvr2_loss.data.new([0.0]).squeeze(-1)
    attention_loss += compute_twotower_cross_modal_attention_loss(text_infer, image_infer1, teacher_infer1, scale=pl_module.text_encoder.transformer.blocks[0].attn.scale, )
    attention_loss += compute_twotower_cross_modal_attention_loss(text_infer, image_infer2, teacher_infer2, scale=pl_module.text_encoder.transformer.blocks[0].attn.scale, )
        
    ret = {
        "nlvr2_loss": nlvr2_loss,
        "soft_nlvr2_loss": sl_nlvr2_loss,
        "attention_loss": attention_loss,
        "nlvr2_logits": student_nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        soft_nlvr2_loss =  getattr(pl_module, f"{phase}_soft_nlvr2_loss")(ret["soft_nlvr2_loss"])
        attention_nlvr2_loss = getattr(pl_module, f"{phase}_attention_nlvr2_loss")(ret["attention_loss"])
    
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/soft_nlvr2_loss", soft_nlvr2_loss)
        pl_module.log(f"nlvr2/{phase}/attention_nlvr2_loss", attention_nlvr2_loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret

def compute_ve(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    ve_logits = pl_module.ve_classifier(infer["cls_feats"])

    ve_labels = batch["answers"]
    ve_labels = torch.tensor(ve_labels).to(pl_module.device).long()
    ve_loss = F.cross_entropy(ve_logits, ve_labels)

    ret = {
        "snli_ve_loss": ve_loss,
        "snli_ve_logits": ve_logits,
        "snli_ve_labels": ve_labels,
    }

    phase = "train" if pl_module.training else "val"
    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_ve_loss")(ret["snli_ve_loss"])
        acc = getattr(pl_module, f"{phase}_snli_ve_accuracy")(
            ret["snli_ve_logits"], ret["snli_ve_labels"]
        )
        pl_module.log(f"snli_ve/{phase}/loss", loss)
        pl_module.log(f"snli_ve/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_ve_loss")(
                F.cross_entropy(
                    ret["snli_ve_logits"][dev_batches], ret["snli_ve_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_ve_accuracy")(
                ret["snli_ve_logits"][dev_batches], ret["snli_ve_labels"][dev_batches]
            )
            pl_module.log(f"snli_ve/dev/loss", dev_loss)
            pl_module.log(f"snli_ve/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_ve_loss")(
                F.cross_entropy(
                    ret["snli_ve_logits"][test_batches], ret["snli_ve_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_ve_accuracy")(
                ret["snli_ve_logits"][test_batches], ret["snli_ve_labels"][test_batches]
            )
            pl_module.log(f"snli_ve/test/loss", test_loss)
            pl_module.log(f"snli_ve/test/accuracy", test_acc)

    return ret

def compute_twotower_ve(pl_module, batch):
    text_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="text")
    image_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="image")
    fusion_cls_feats = pl_module.fusion(torch.cat((text_infer["cls_feats"], image_infer["cls_feats"]), dim=-1))
    ve_logits = pl_module.ve_classifier(fusion_cls_feats)

    ve_labels = batch["answers"]
    ve_labels = torch.tensor(ve_labels).to(pl_module.device).long()
    ve_loss = F.cross_entropy(ve_logits, ve_labels)

    ret = {
        "snli_ve_loss": ve_loss,
        "snli_ve_logits": ve_logits,
        "snli_ve_labels": ve_labels,
    }

    phase = "train" if pl_module.training else "val"
    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_ve_loss")(ret["snli_ve_loss"])
        acc = getattr(pl_module, f"{phase}_snli_ve_accuracy")(
            ret["snli_ve_logits"], ret["snli_ve_labels"]
        )
        pl_module.log(f"snli_ve/{phase}/loss", loss)
        pl_module.log(f"snli_ve/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_ve_loss")(
                F.cross_entropy(
                    ret["snli_ve_logits"][dev_batches], ret["snli_ve_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_ve_accuracy")(
                ret["snli_ve_logits"][dev_batches], ret["snli_ve_labels"][dev_batches]
            )
            pl_module.log(f"snli_ve/dev/loss", dev_loss)
            pl_module.log(f"snli_ve/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_ve_loss")(
                F.cross_entropy(
                    ret["snli_ve_logits"][test_batches], ret["snli_ve_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_ve_accuracy")(
                ret["snli_ve_logits"][test_batches], ret["snli_ve_labels"][test_batches]
            )
            pl_module.log(f"snli_ve/test/loss", test_loss)
            pl_module.log(f"snli_ve/test/accuracy", test_acc)

    return ret

def compute_twotower_dst_ve(pl_module, batch):
    text_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="text")
    image_infer = pl_module.infer(batch, mask_text=False, mask_image=False, mode="image")
    fusion_cls_feats = pl_module.fusion(torch.cat((text_infer["cls_feats"], image_infer["cls_feats"]), dim=-1))
    student_ve_logits = pl_module.ve_classifier(fusion_cls_feats)
    # teacher part
    pl_module.teacher_model.eval()
    with torch.no_grad():
        teacher_infer = pl_module.teacher_model.infer(batch, mask_text=False, mask_image=False, pre_select=image_infer["select"])
        teacher_ve_logits = pl_module.teacher_model.ve_classifier(teacher_infer["cls_feats"])
    # hard label
    ve_labels = batch["answers"]
    ve_labels = torch.tensor(ve_labels).to(pl_module.device).long()
    ve_loss = F.cross_entropy(student_ve_logits, ve_labels)
    ve_loss = ve_loss.data.new([0.0]).squeeze(-1)
    sl_ve_loss = F.kl_div(F.log_softmax(student_ve_logits.float(), dim=-1) , F.softmax(teacher_ve_logits.float(), dim=-1), reduction="batchmean")
    attention_loss = ve_loss.data.new([0.0]).squeeze(-1)
    attention_loss += compute_twotower_cross_modal_attention_loss(text_infer, image_infer, teacher_infer, scale=pl_module.text_encoder.transformer.blocks[0].attn.scale, )
        
    ret = {
        "snli_ve_loss": ve_loss,
        "soft_snli_ve_loss": sl_ve_loss,
        "attention_loss": attention_loss,
        "snli_ve_logits": student_ve_logits,
        "snli_ve_labels": ve_labels,
    }

    phase = "train" if pl_module.training else "val"
    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_ve_loss")(ret["snli_ve_loss"])
        soft_snli_ve_loss =  getattr(pl_module, f"{phase}_soft_snli_ve_loss")(ret["soft_snli_ve_loss"])
        attention_snli_ve_loss = getattr(pl_module, f"{phase}_attention_snli_ve_loss")(ret["attention_loss"])
        
        acc = getattr(pl_module, f"{phase}_snli_ve_accuracy")(
            ret["snli_ve_logits"], ret["snli_ve_labels"]
        )
        pl_module.log(f"snli_ve/{phase}/loss", loss)
        pl_module.log(f"snli_ve/{phase}/soft_snli_ve_loss", soft_snli_ve_loss)
        pl_module.log(f"snli_ve/{phase}/attention_snli_ve_loss", attention_snli_ve_loss)
        pl_module.log(f"snli_ve/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_ve_loss")(
                F.cross_entropy(
                    ret["snli_ve_logits"][dev_batches], ret["snli_ve_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_ve_accuracy")(
                ret["snli_ve_logits"][dev_batches], ret["snli_ve_labels"][dev_batches]
            )
            pl_module.log(f"snli_ve/dev/loss", dev_loss)
            pl_module.log(f"snli_ve/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_ve_loss")(
                F.cross_entropy(
                    ret["snli_ve_logits"][test_batches], ret["snli_ve_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_ve_accuracy")(
                ret["snli_ve_logits"][test_batches], ret["snli_ve_labels"][test_batches]
            )
            pl_module.log(f"snli_ve/test/loss", test_loss)
            pl_module.log(f"snli_ve/test/accuracy", test_acc)

    return ret

# text2image image2text
def compute_twotower_cross_modal_attention_loss(text_infer, image_infer, teacher_infer, scale, teacher_layer=11, student_layer=11):
    student_text_all_q_states = text_infer["all_q_states"][student_layer:(student_layer+1)]
    student_text_all_k_states = text_infer["all_k_states"][student_layer:(student_layer+1)]
    student_text_all_v_states = text_infer["all_v_states"][student_layer:(student_layer+1)]
    student_image_all_q_states = image_infer["all_q_states"][student_layer:(student_layer+1)]
    student_image_all_k_states = image_infer["all_k_states"][student_layer:(student_layer+1)]
    student_image_all_v_states = image_infer["all_v_states"][student_layer:(student_layer+1)]
    teacher_all_q_states = teacher_infer["all_q_states"][teacher_layer:(teacher_layer+1)]
    teacher_all_k_states = teacher_infer["all_k_states"][teacher_layer:(teacher_layer+1)]
    teacher_all_v_states = teacher_infer["all_v_states"][teacher_layer:(teacher_layer+1)]
    
    text_masks = text_infer["text_masks"]
    image_masks = image_infer["image_masks"]
    attn_masks = torch.cat([text_masks, image_masks], dim=1)
    assert torch.equal(attn_masks, teacher_infer["co_masks"])
    # bsz, num_heads, seq_len, head_dim
    teacher_text_all_q_states = [teacher_q_states[:, :, :text_masks.shape[1], :] for teacher_q_states in teacher_all_q_states]
    teacher_image_all_q_states = [teacher_q_states[:, :, text_masks.shape[1]:, :] for teacher_q_states in teacher_all_q_states]
    teacher_text_all_k_states = [teacher_k_states[:, :, :text_masks.shape[1], :] for teacher_k_states in teacher_all_k_states]
    teacher_image_all_k_states = [teacher_k_states[:, :, text_masks.shape[1]:, :] for teacher_k_states in teacher_all_k_states]
    teacher_text_all_v_states = [teacher_v_states[:, :, :text_masks.shape[1], :] for teacher_v_states in teacher_all_v_states]
    teacher_image_all_v_states = [teacher_v_states[:, :, text_masks.shape[1]:, :] for teacher_v_states in teacher_all_v_states]

    assert teacher_text_all_q_states[0].shape[1] == student_text_all_q_states[0].shape[1]
    
    # print("***t2i****")
    t2i_attention_loss = 0
    for student_text_q_states, student_image_k_states, teacher_text_q_states, teacher_image_k_states in zip(student_text_all_q_states, student_image_all_k_states, teacher_text_all_q_states, teacher_image_all_k_states):
        student_t2i_attn = (student_text_q_states @ student_image_k_states.transpose(-2, -1)) * scale
        student_t2i_attn = student_t2i_attn.masked_fill(~image_masks.bool()[:, None, None, :], float("-inf"))
        teacher_t2i_attn = (teacher_text_q_states @ teacher_image_k_states.transpose(-2, -1)) * scale
        teacher_t2i_attn = teacher_t2i_attn.masked_fill(~image_masks.bool()[:, None, None, :], float("-inf"))
        tmp_loss = F.kl_div(F.log_softmax(student_t2i_attn.float(), dim=-1), F.softmax(teacher_t2i_attn.float(), dim=-1), reduction='none')
        tmp_loss = tmp_loss * image_masks.type_as(tmp_loss)[:, None, None, :]
        tmp_loss = torch.mean(torch.sum(tmp_loss, dim=-1))
        t2i_attention_loss += tmp_loss

    i2t_attention_loss = 0
    for student_image_q_states, student_text_k_states, teacher_image_q_states, teacher_text_k_states in zip(student_image_all_q_states, student_text_all_k_states, teacher_image_all_q_states, teacher_text_all_k_states):
        student_i2t_attn = (student_image_q_states @ student_text_k_states.transpose(-2, -1)) * scale
        student_i2t_attn = student_i2t_attn.masked_fill(~text_masks.bool()[:, None, None, :], float("-inf"))
        teacher_i2t_attn = (teacher_image_q_states @ teacher_text_k_states.transpose(-2, -1)) * scale
        teacher_i2t_attn = teacher_i2t_attn.masked_fill(~text_masks.bool()[:, None, None, :], float("-inf"))
        tmp_loss = F.kl_div(F.log_softmax(student_i2t_attn.float(), dim=-1), F.softmax(teacher_i2t_attn.float(), dim=-1), reduction='none')
        tmp_loss = tmp_loss * text_masks.type_as(tmp_loss)[:, None, None, :]
        tmp_loss = torch.mean(torch.sum(tmp_loss, dim=-1))
        i2t_attention_loss += tmp_loss
    attention_loss = t2i_attention_loss + i2t_attention_loss
    return attention_loss

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name, save_path):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs(save_path, exist_ok=True)
        with open(save_path + f"/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
