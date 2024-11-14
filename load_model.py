import os.path
import json
import torch
import timm
from timm.models import create_model
from torchvision.models import get_model
from models import LinearProbeViT, IndividualLandmarkConvNext, IndividualLandmarkResNet, IndividualLandmarkViT, \
    AttnMaskViT, LateMaskViT, AttnMaskViTTeacher, LateMaskViTStudent, AttnMaskViTParallel, \
    FullTwoStageModelDoubleClassify, FullTwoStageModelAttributes, FullTwoStageModelDoubleClassifyHF
from models.selfpatch_vision_transformer import vit_small_sp, load_pretrained_weights_sp


def load_model_arch(args, num_cls):
    """
    Function to load the model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    if 'resnet' in args.model_arch:
        num_layers_split = [int(s) for s in args.model_arch if s.isdigit()]
        num_layers = int(''.join(map(str, num_layers_split)))
        if num_layers >= 100:
            timm_model_arch = args.model_arch + ".a1h_in1k"
        else:
            timm_model_arch = args.model_arch + ".a1_in1k"

    if "resnet" in args.model_arch and args.use_torchvision_resnet_model:
        weights = "DEFAULT" if args.pretrained_start_weights else None
        base_model = get_model(args.model_arch, weights=weights)
    elif "resnet" in args.model_arch and not args.use_torchvision_resnet_model:
        if args.eval_only:
            base_model = create_model(
                timm_model_arch,
                pretrained=args.pretrained_start_weights,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
        else:
            base_model = create_model(
                timm_model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )

    elif "convnext" in args.model_arch:
        if args.eval_only:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
        else:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
    elif "vit" in args.model_arch:
        if args.pretrained_model_path is not None:
            if args.eval_only:
                base_model = create_model(
                    args.model_arch,
                    pretrained=False,
                )
            else:
                base_model = create_model(
                    args.model_arch,
                    pretrained=False,
                    drop_path_rate=args.drop_path,
                )
            pretrained_state_dict = convert_pretrained_checkpoint(args.pretrained_model_path)
            img_size = timm.data.resolve_model_data_config(base_model)['input_size'][-1]
            if img_size != args.image_size:
                raise ValueError(f"Image size {args.image_size} must match the image size of the model {img_size}")
            base_model.load_state_dict(pretrained_state_dict, strict=False)
        else:
            if args.eval_only:
                base_model = create_model(
                    args.model_arch,
                    pretrained=args.pretrained_start_weights,
                    img_size=args.image_size,
                )
            else:
                base_model = create_model(
                    args.model_arch,
                    pretrained=args.pretrained_start_weights,
                    drop_path_rate=args.drop_path,
                    img_size=args.image_size,
                )
        vit_patch_size = base_model.patch_embed.proj.kernel_size[0]
        if args.image_size % vit_patch_size != 0:
            raise ValueError(f"Image size {args.image_size} must be divisible by patch size {vit_patch_size}")
    elif "clippy" in args.model_arch:
        if args.eval_only:
            base_model = create_model(
                'vit_base_patch16_224.dino',
                pretrained=args.pretrained_start_weights,
            )
        else:
            base_model = create_model(
                'vit_base_patch16_224.dino',
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
            )
        state_dict = torch.load(args.pretrained_model_path, map_location=torch.device('cpu'), weights_only=True)
        base_model.load_state_dict(state_dict, strict=False)
        img_size = timm.data.resolve_model_data_config(base_model)['input_size'][-1]
        if img_size != args.image_size:
            raise ValueError(f"Image size {args.image_size} must match the image size of the model {img_size}")
    elif "selfpatch" in args.model_arch:
        base_model = vit_small_sp()
        load_pretrained_weights_sp(base_model, args.pretrained_model_path)
    elif "ibot" in args.model_arch:
        base_model = load_model_arch_ibot(args.model_arch)
    else:
        raise ValueError('Model not supported.')

    return base_model


def load_model_arch_ibot(model_arch):
    base_url = "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/"
    ckpt_name = "checkpoint_teacher.pth"
    if "small" in model_arch:
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        model_url = "vits_16/"
        url_path = base_url + model_url + ckpt_name

    elif "base" in model_arch:
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        if "rand_mask" in model_arch:
            model_url = "vitb_16_rand_mask/"
        else:
            model_url = "vitb_16/"
        url_path = base_url + model_url + ckpt_name
    elif "large" in model_arch:
        model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
        if "rand_mask" in model_arch:
            model_url = "vitl_16_rand_mask/"
        else:
            model_url = "vitl_16/"
        url_path = base_url + model_url + ckpt_name
    else:
        raise ValueError('Model not supported.')
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, "ibot", model_url)
    state_dict = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')['state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def load_pdiscoformer_model_arch(args, num_cls):
    """
    Function to load the model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    if 'resnet' in args.pdiscoformer_model_arch:
        num_layers_split = [int(s) for s in args.model_arch if s.isdigit()]
        num_layers = int(''.join(map(str, num_layers_split)))
        if num_layers >= 100:
            timm_model_arch = args.pdiscoformer_model_arch + ".a1h_in1k"
        else:
            timm_model_arch = args.pdiscoformer_model_arch + ".a1_in1k"

    if "resnet" in args.pdiscoformer_model_arch and args.use_torchvision_resnet_model:
        weights = "DEFAULT" if args.pretrained_start_weights else None
        base_model = get_model(args.pdiscoformer_model_arch, weights=weights)
    elif "resnet" in args.pdiscoformer_model_arch and not args.use_torchvision_resnet_model:
        if args.eval_only:
            base_model = create_model(
                timm_model_arch,
                pretrained=args.pretrained_start_weights,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
        else:
            base_model = create_model(
                timm_model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )

    elif "convnext" in args.pdiscoformer_model_arch:
        if args.eval_only:
            base_model = create_model(
                args.pdiscoformer_model_arch,
                pretrained=args.pretrained_start_weights,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
        else:
            base_model = create_model(
                args.pdiscoformer_model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
    elif "vit" in args.pdiscoformer_model_arch:
        if args.eval_only:
            base_model = create_model(
                args.pdiscoformer_model_arch,
                pretrained=args.pretrained_start_weights,
                img_size=args.pdisco_image_size,
            )
        else:
            base_model = create_model(
                args.pdiscoformer_model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                img_size=args.pdisco_image_size,
            )
        vit_patch_size = base_model.patch_embed.proj.kernel_size[0]
        if args.pdisco_image_size % vit_patch_size != 0:
            raise ValueError(f"Image size {args.pdisco_image_size} must be divisible by patch size {vit_patch_size}")
    else:
        raise ValueError('Model not supported.')

    return base_model


def init_model_baseline(base_model, args, num_cls):
    """
    Function to initialize the baseline model
    :param base_model: Model loaded from the timm/torchvision library
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    # Initialize the network
    if 'convnext' in args.model_arch:
        torch.nn.init.trunc_normal_(base_model.fc.head.weight, std=.02)
        if base_model.fc.head.bias is not None:
            torch.nn.init.constant_(base_model.fc.head.bias, 0.)

    if 'resnet' in args.model_arch:
        torch.nn.init.trunc_normal_(base_model.fc.weight, std=.02)
        if base_model.fc.bias is not None:
            torch.nn.init.constant_(base_model.fc.bias, 0.)

    elif 'vit' in args.model_arch or 'clippy' in args.model_arch or 'selfpatch' or 'ibot' in args.model_arch:
        base_model = LinearProbeViT(base_model, num_classes=num_cls, pooling_type=args.pooling_type,
                                    reinit_fc_norm=args.reinit_fc_norm)
    else:
        raise ValueError('Model not supported.')
    return base_model


def load_model_baseline(args, num_cls):
    """
    Function to load the baseline model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    base_model = load_model_arch(args, num_cls)
    model = init_model_baseline(base_model, args, num_cls)

    return model


def load_starting_weights(args, model):
    """
    Function to load the starting weights
    :param args: Arguments from the command line
    :param model: Model object
    :return:
    """
    snapshot_data = torch.load(args.pdiscoformer_pretrained_path, map_location=torch.device('cpu'), weights_only=True)
    state_dict = snapshot_data['model_state']
    if args.classifier_type == 'none':
        state_dict.pop('fc_class_landmarks.weight', None)
        state_dict.pop('fc_class_landmarks.bias', None)
        keys_to_change = [key for key in state_dict.keys() if "modulation" in key]
        for key in keys_to_change:
            new_key = key.replace("modulation", "attribute_modulation")
            state_dict[new_key] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    return model


def convert_pretrained_checkpoint(ckpt_path):
    """
    Function to convert the checkpoint
    :param ckpt_path: Path to the checkpoint
    :return:
    """
    snapshot_data = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    # timm model
    if 'state_dict' in snapshot_data:
        state_dict = snapshot_data['state_dict']
    else:
        state_dict = snapshot_data

    state_dict.pop('head.weight', None)
    state_dict.pop('head.bias', None)

    return state_dict


def init_model_mask_vit(base_model, args, num_cls):
    if args.late_masking:
        base_model = LateMaskViT(base_model, num_classes=num_cls, pooling_type=args.pooling_type,
                                 reinit_fc_norm=args.reinit_fc_norm)
    else:
        base_model = AttnMaskViT(base_model, num_classes=num_cls, pooling_type=args.pooling_type,
                                 reinit_fc_norm=args.reinit_fc_norm)
    return base_model


def init_model_mask_vit_parallel(base_model, args, dataset):
    num_att_per_part = dataset.num_attributes_per_part_idx
    base_model = AttnMaskViTParallel(base_model, num_att_per_part=num_att_per_part, num_copies_cls=args.num_parts + 1)
    return base_model


def load_model_mask_vit(args, num_cls, dataset=None):
    base_model = load_model_arch(args, num_cls=0)
    if args.parallel_masking:
        model = init_model_mask_vit_parallel(base_model, args, dataset)
    else:
        model = init_model_mask_vit(base_model, args, num_cls)
    return model


def init_model_distillation(base_model, args):
    student_model = LateMaskViTStudent(base_model, pooling_type=args.pooling_type)
    teacher_model = AttnMaskViTTeacher(base_model)

    return student_model, teacher_model


def load_model_distillation(args, num_cls):
    base_model = load_model_arch(args, num_cls)
    student_model, teacher_model = init_model_distillation(base_model, args)
    return student_model, teacher_model


def init_pdisco_model(base_model, args, num_cls):
    """
    Function to initialize the model
    :param base_model: Base model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    # Initialize the network
    if 'convnext' in args.pdiscoformer_model_arch:
        sl_channels = base_model.stages[-1].downsample[-1].in_channels
        fl_channels = base_model.head.in_features
        model = IndividualLandmarkConvNext(base_model, args.num_parts, num_classes=num_cls,
                                           sl_channels=sl_channels, fl_channels=fl_channels,
                                           modulation_type=args.modulation_type,
                                           gumbel_softmax=args.gumbel_softmax,
                                           gumbel_softmax_temperature=args.gumbel_softmax_temperature,
                                           gumbel_softmax_hard=args.gumbel_softmax_hard,
                                           modulation_orth=args.modulation_orth,
                                           classifier_type=args.pdiscoformer_classifier_type)
    elif 'resnet' in args.pdiscoformer_model_arch:
        sl_channels = base_model.layer4[0].conv1.in_channels
        fl_channels = base_model.fc.in_features
        model = IndividualLandmarkResNet(base_model, args.num_parts, num_classes=num_cls,
                                         sl_channels=sl_channels, fl_channels=fl_channels,
                                         use_torchvision_model=args.use_torchvision_resnet_model,
                                         part_dropout=args.part_dropout, modulation_type=args.modulation_type,
                                         gumbel_softmax=args.gumbel_softmax,
                                         gumbel_softmax_temperature=args.gumbel_softmax_temperature,
                                         gumbel_softmax_hard=args.gumbel_softmax_hard,
                                         modulation_orth=args.modulation_orth,
                                         classifier_type=args.pdiscoformer_classifier_type)
    elif 'vit' in args.pdiscoformer_model_arch:
        model = IndividualLandmarkViT(base_model, num_landmarks=args.num_parts, num_classes=num_cls,
                                      modulation_type=args.modulation_type, gumbel_softmax=args.gumbel_softmax,
                                      gumbel_softmax_temperature=args.gumbel_softmax_temperature,
                                      gumbel_softmax_hard=args.gumbel_softmax_hard,
                                      modulation_orth=args.modulation_orth,
                                      classifier_type=args.pdiscoformer_classifier_type)
    else:
        raise ValueError('Model not supported.')

    return model


def load_starting_weights_pdiscoformer(args, model):
    """
    Function to load the starting weights
    :param args: Arguments from the command line
    :param model: Model object
    :return:
    """
    snapshot_data = torch.load(args.pdiscoformer_pretrained_path, map_location=torch.device('cpu'), weights_only=True)
    state_dict = snapshot_data['model_state']
    model.load_state_dict(state_dict, strict=False)
    return model


def load_model_pdisco(args, num_cls):
    """
    Function to load the model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    base_model = load_pdiscoformer_model_arch(args, num_cls)
    model = init_pdisco_model(base_model, args, num_cls)
    model = load_starting_weights_pdiscoformer(args, model)

    return model


def load_model_arch_2_stage(args, stage_num):
    """
    Function to load the model
    :param args: Arguments from the command line
    :param stage_num: Stage number
    :return:
    """
    if stage_num == 1:
        drop_path_rate = args.drop_path_stage_1
    else:
        drop_path_rate = args.drop_path_stage_2
    if args.eval_only:
        base_model = create_model(
            args.model_arch,
            pretrained=args.pretrained_start_weights,
            img_size=args.image_size,
        )
    else:
        base_model = create_model(
            args.model_arch,
            pretrained=args.pretrained_start_weights,
            drop_path_rate=drop_path_rate,
            img_size=args.image_size,
        )
    return base_model


def load_model_2_stage(args, dataset_test, num_cls):
    if args.part_logits_threshold_path:
        with open(args.part_logits_threshold_path, 'r') as f:
            part_logits_threshold = json.load(f)
    else:
        part_logits_threshold = None

    if args.use_hf_transformers:
        model = load_model_2_stage_hf(args, num_cls, part_logits_threshold)
    else:
        base_model_1 = load_model_arch_2_stage(args, 1)
        base_model_2 = load_model_arch_2_stage(args, 2)
        if not args.attribute_prediction:
            model = FullTwoStageModelDoubleClassify(base_model_1, base_model_2, num_classes=num_cls,
                                                    return_transformer_qkv=False,
                                                    num_landmarks=args.num_parts,
                                                    gumbel_softmax=args.gumbel_softmax,
                                                    softmax_temperature=args.softmax_temperature,
                                                    gumbel_softmax_hard=args.gumbel_softmax_hard,
                                                    part_dropout=args.part_dropout,
                                                    part_dropout_stage_2=args.part_dropout_stage_2,
                                                    part_logits_threshold=part_logits_threshold,
                                                    use_soft_masks=args.use_soft_masks)
        else:
            num_att_per_part = dataset_test.num_attributes_per_part_idx
            model = FullTwoStageModelAttributes(base_model_1, base_model_2, num_landmarks=args.num_parts,
                                                num_att_per_part=num_att_per_part,
                                                return_transformer_qkv=False, gumbel_softmax=args.gumbel_softmax,
                                                softmax_temperature=args.softmax_temperature,
                                                gumbel_softmax_hard=args.gumbel_softmax_hard,
                                                dropout_rate=args.part_dropout, use_part_attention=args.use_part_attention)
    return model


def load_model_2_stage_hf(args, num_cls, part_logits_threshold):
    from transformers import AutoModel
    init_model = AutoModel.from_pretrained(args.model_arch)
    model = FullTwoStageModelDoubleClassifyHF(init_model, init_model.config, num_classes=num_cls, num_landmarks=args.num_parts,
                                              softmax_temperature=args.softmax_temperature, part_dropout=args.part_dropout, part_dropout_stage_2=args.part_dropout_stage_2,
                                              use_soft_masks=args.use_soft_masks, part_logits_threshold=part_logits_threshold)
    return model
