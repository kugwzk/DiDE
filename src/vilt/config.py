from sacred import Experiment

ex = Experiment("ViLT")


def _loss_names(d):
    ret = {
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "snli_ve": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Two tower setting
    use_two_tower = False
    fixed_teacher = True

    # Distillation Setting
    use_dst = False
    teacher_vit = "vit_base_patch32_384"
    teacher_hidden_size = 768
    teacher_load_path = ""

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    check_val_every_n_epoch = 1.0
    test_only = False
    save_top_k = 1

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_finetune_snli_ve():
    exp_name = "finetune_snli_ve"
    datasets = ["snli_ve"]
    loss_names = _loss_names({"snli_ve": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4

@ex.named_config
def task_finetune_snli_ve_randaug():
    exp_name = "finetune_snli_ve"
    datasets = ["snli_ve"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"snli_ve": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4

@ex.named_config
def task_twotower_finetune_snli_ve_randaug():
    exp_name = "twotower_finetune_snli_ve_randaug"
    datasets = ["snli_ve"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"snli_ve": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    # 2 tower
    use_two_tower = True

@ex.named_config
def task_twotower_dst_finetune_snli_ve_randaug():
    exp_name = "twotower_dst_finetune_snli_ve_randaug"
    datasets = ["snli_ve"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"snli_ve": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    # 2 tower
    use_two_tower = True
    # dst part
    use_dst = True
    teacher_vit = "vit_base_patch32_384"
    teacher_hidden_size = 768

@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    # for debug
    # max_epoch = 2
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4

@ex.named_config
def task_twotower_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    # 2 tower
    use_two_tower = True

@ex.named_config
def task_twotower_dst_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    # 2 tower
    use_two_tower = True
    # dst part
    use_dst = True
    teacher_vit = "vit_base_patch32_384"
    teacher_hidden_size = 768

@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult = 10

@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult = 10

@ex.named_config
def task_twotower_finetune_vqa_randaug():
    exp_name = "finetune_twotower_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult = 10
    # 2 tower
    use_two_tower = True

@ex.named_config
def task_twotower_dst_finetune_vqa_randaug():
    exp_name = "finetune_twotower_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_mult = 10
    # 2 tower
    use_two_tower = True
    # dst part
    use_dst = True
    teacher_vit = "vit_base_patch32_384"
    teacher_hidden_size = 768
