import sys
sys.path.append("/lkk/new_comfyui/sam/ComfyUI")
sys.path.append("/lkk/new_comfyui/sam/ComfyUI-segment-anything-2")

import json
import yaml
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import torch
import math
import struct
import comfy.checkpoint_pickle
import comfy.model_management as mm
from comfy.utils import ProgressBar, common_upscale
import safetensors.torch
import numpy as np
from PIL import Image
import logging
import itertools
from torch.functional import F

from tqdm import tqdm
from contextlib import nullcontext

def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                logging.warning("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=comfy.checkpoint_pickle)
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def load_model(model_path, model_cfg_path, segmentor, dtype, device):
    # Load the YAML configuration
    with open(model_cfg_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract the model configuration
    model_config = config['model']

    # Instantiate the image encoder components
    trunk_config = model_config['image_encoder']['trunk']
    neck_config = model_config['image_encoder']['neck']
    position_encoding_config = neck_config['position_encoding']

    position_encoding = PositionEmbeddingSine(
        num_pos_feats=position_encoding_config['num_pos_feats'],
        normalize=position_encoding_config['normalize'],
        scale=position_encoding_config['scale'],
        temperature=position_encoding_config['temperature']
    )

    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=neck_config['d_model'],
        backbone_channel_list=neck_config['backbone_channel_list'],
        fpn_top_down_levels=neck_config['fpn_top_down_levels'],
        fpn_interp_model=neck_config['fpn_interp_model']
    )

    keys_to_include = ['embed_dim', 'num_heads', 'global_att_blocks', 'window_pos_embed_bkg_spatial_size', 'stages']
    trunk_kwargs = {key: trunk_config[key] for key in keys_to_include if key in trunk_config}
    trunk = Hiera(**trunk_kwargs)

    image_encoder = ImageEncoder(
        scalp=model_config['image_encoder']['scalp'],
        trunk=trunk,
        neck=neck
    )
    # Instantiate the memory attention components
    memory_attention_layer_config = config['model']['memory_attention']['layer']
    self_attention_config = memory_attention_layer_config['self_attention']
    cross_attention_config = memory_attention_layer_config['cross_attention']

    self_attention = RoPEAttention(
        rope_theta=self_attention_config['rope_theta'],
        feat_sizes=self_attention_config['feat_sizes'],
        embedding_dim=self_attention_config['embedding_dim'],
        num_heads=self_attention_config['num_heads'],
        downsample_rate=self_attention_config['downsample_rate'],
        dropout=self_attention_config['dropout']
    )

    cross_attention = RoPEAttention(
        rope_theta=cross_attention_config['rope_theta'],
        feat_sizes=cross_attention_config['feat_sizes'],
        rope_k_repeat=cross_attention_config['rope_k_repeat'],
        embedding_dim=cross_attention_config['embedding_dim'],
        num_heads=cross_attention_config['num_heads'],
        downsample_rate=cross_attention_config['downsample_rate'],
        dropout=cross_attention_config['dropout'],
        kv_in_dim=cross_attention_config['kv_in_dim']
    )

    memory_attention_layer = MemoryAttentionLayer(
        activation=memory_attention_layer_config['activation'],
        dim_feedforward=memory_attention_layer_config['dim_feedforward'],
        dropout=memory_attention_layer_config['dropout'],
        pos_enc_at_attn=memory_attention_layer_config['pos_enc_at_attn'],
        self_attention=self_attention,
        d_model=memory_attention_layer_config['d_model'],
        pos_enc_at_cross_attn_keys=memory_attention_layer_config['pos_enc_at_cross_attn_keys'],
        pos_enc_at_cross_attn_queries=memory_attention_layer_config['pos_enc_at_cross_attn_queries'],
        cross_attention=cross_attention
    )

    memory_attention = MemoryAttention(
        d_model=config['model']['memory_attention']['d_model'],
        pos_enc_at_input=config['model']['memory_attention']['pos_enc_at_input'],
        layer=memory_attention_layer,
        num_layers=config['model']['memory_attention']['num_layers']
    )

    # Instantiate the memory encoder components
    memory_encoder_config = config['model']['memory_encoder']
    position_encoding_mem_enc_config = memory_encoder_config['position_encoding']
    mask_downsampler_config = memory_encoder_config['mask_downsampler']
    fuser_layer_config = memory_encoder_config['fuser']['layer']

    position_encoding_mem_enc = PositionEmbeddingSine(
        num_pos_feats=position_encoding_mem_enc_config['num_pos_feats'],
        normalize=position_encoding_mem_enc_config['normalize'],
        scale=position_encoding_mem_enc_config['scale'],
        temperature=position_encoding_mem_enc_config['temperature']
    )

    mask_downsampler = MaskDownSampler(
        kernel_size=mask_downsampler_config['kernel_size'],
        stride=mask_downsampler_config['stride'],
        padding=mask_downsampler_config['padding']
    )

    fuser_layer = CXBlock(
        dim=fuser_layer_config['dim'],
        kernel_size=fuser_layer_config['kernel_size'],
        padding=fuser_layer_config['padding'],
        layer_scale_init_value=float(fuser_layer_config['layer_scale_init_value'])
    )
    fuser = Fuser(
        num_layers=memory_encoder_config['fuser']['num_layers'],
        layer=fuser_layer
    )

    memory_encoder = MemoryEncoder(
        position_encoding=position_encoding_mem_enc,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
        out_dim=memory_encoder_config['out_dim']
    )

    sam_mask_decoder_extra_args = {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    }

    def initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device):
        return model_class(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
            num_maskmem=model_config['num_maskmem'],
            image_size=model_config['image_size'],
            sigmoid_scale_for_mem_enc=model_config['sigmoid_scale_for_mem_enc'],
            sigmoid_bias_for_mem_enc=model_config['sigmoid_bias_for_mem_enc'],
            use_mask_input_as_output_without_sam=model_config['use_mask_input_as_output_without_sam'],
            directly_add_no_mem_embed=model_config['directly_add_no_mem_embed'],
            use_high_res_features_in_sam=model_config['use_high_res_features_in_sam'],
            multimask_output_in_sam=model_config['multimask_output_in_sam'],
            iou_prediction_use_sigmoid=model_config['iou_prediction_use_sigmoid'],
            use_obj_ptrs_in_encoder=model_config['use_obj_ptrs_in_encoder'],
            add_tpos_enc_to_obj_ptrs=model_config['add_tpos_enc_to_obj_ptrs'],
            only_obj_ptrs_in_the_past_for_eval=model_config['only_obj_ptrs_in_the_past_for_eval'],
            pred_obj_scores=model_config['pred_obj_scores'],
            pred_obj_scores_mlp=model_config['pred_obj_scores_mlp'],
            fixed_no_obj_ptr=model_config['fixed_no_obj_ptr'],
            multimask_output_for_tracking=model_config['multimask_output_for_tracking'],
            use_multimask_token_for_obj_ptr=model_config['use_multimask_token_for_obj_ptr'],
            compile_image_encoder=model_config['compile_image_encoder'],
            multimask_min_pt_num=model_config['multimask_min_pt_num'],
            multimask_max_pt_num=model_config['multimask_max_pt_num'],
            use_mlp_for_obj_ptr_proj=model_config['use_mlp_for_obj_ptr_proj'],
            proj_tpos_enc_in_obj_ptrs=model_config['proj_tpos_enc_in_obj_ptrs'],
            no_obj_embed_spatial=model_config['no_obj_embed_spatial'],
            use_signed_tpos_enc_to_obj_ptrs=model_config['use_signed_tpos_enc_to_obj_ptrs'],
            binarize_mask_from_pts_for_mem_enc=True if segmentor == 'video' else False,
        ).to(dtype).to(device).eval()

    # Load the state dictionary
    sd = load_torch_file(model_path)

    # Initialize model based on segmentor type
    if segmentor == 'single_image':
        model_class = SAM2Base
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
        model = SAM2ImagePredictor(model)
    elif segmentor == 'video':
        model_class = SAM2VideoPredictor
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
    elif segmentor == 'automaskgenerator':
        model_class = SAM2Base
        model = initialize_model(model_class, model_config, segmentor, image_encoder, memory_attention, memory_encoder, sam_mask_decoder_extra_args, dtype, device)
        model.load_state_dict(sd)
        model = SAM2AutomaticMaskGenerator(model)
    else:
        raise ValueError(f"Segmentor {segmentor} not supported")

    return model

model_path = "/lkk/new_comfyui/ComfyUI/models/sam2/sam2_hiera_large.safetensors"
model_cfg_path = "/lkk/new_comfyui/ComfyUI/custom_nodes/ComfyUI-segment-anything-2/sam2_configs/sam2_hiera_l.yaml"
segmentor = "single_image"
dtype = torch.bfloat16
device = "hpu"
version = "2.0"


class Sam2Segmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam2_model": ("SAM2MODEL", ),
                "image": ("IMAGE", ),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "coordinates_negative": ("STRING", {"forceInput": True}),
                "bboxes": ("BBOX", ),
                "individual_objects": ("BOOLEAN", {"default": False}),
                "mask": ("MASK", ),

            },
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES =("mask", )
    FUNCTION = "segment"
    CATEGORY = "SAM2"

    def segment(self, image, sam2_model, keep_model_loaded, coordinates_positive=None, coordinates_negative=None,
                individual_objects=False, bboxes=None, mask=None):

        print(image)
        print(sam2_model)
        print(keep_model_loaded)
        print(coordinates_positive)
        print(coordinates_negative)
        print(individual_objects)
        print(bboxes)
        print(mask)


        offload_device = mm.unet_offload_device()
        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]
        segmentor = sam2_model["segmentor"]
        B, H, W, C = image.shape

        if mask is not None:
            input_mask = mask.clone().unsqueeze(1)
            input_mask = F.interpolate(input_mask, size=(256, 256), mode="bilinear")
            input_mask = input_mask.squeeze(1)

        if segmentor == 'automaskgenerator':
            raise ValueError("For automaskgenerator use Sam2AutoMaskSegmentation -node")
        if segmentor == 'single_image' and B > 1:
            print("Segmenting batch of images with single_image segmentor")

        if segmentor == 'video' and bboxes is not None and "2.1" not in sam2_model["version"]:
            raise ValueError("2.0 model doesn't support bboxes with video segmentor")

        if segmentor == 'video': # video model needs images resized first thing
            model_input_image_size = model.image_size
            print("Resizing to model input image size: ", model_input_image_size)
            image = common_upscale(image.movedim(-1,1), model_input_image_size, model_input_image_size, "bilinear", "disabled").movedim(1,-1)

        #handle point coordinates
        if coordinates_positive is not None:
            try:
                coordinates_positive = json.loads(coordinates_positive.replace("'", '"'))
                coordinates_positive = [(coord['x'], coord['y']) for coord in coordinates_positive]
                if coordinates_negative is not None:
                    coordinates_negative = json.loads(coordinates_negative.replace("'", '"'))
                    coordinates_negative = [(coord['x'], coord['y']) for coord in coordinates_negative]
            except Exception as e:
                print(str(e))
                assert 1 == 2
                pass


            if not individual_objects:
                positive_point_coords = np.atleast_2d(np.array(coordinates_positive))
            else:
                positive_point_coords = np.array([np.atleast_2d(coord) for coord in coordinates_positive])

            if coordinates_negative is not None:
                negative_point_coords = np.array(coordinates_negative)
                # Ensure both positive and negative coords are lists of 2D arrays if individual_objects is True
                if individual_objects:
                    assert negative_point_coords.shape[0] <= positive_point_coords.shape[0], "Can't have more negative than positive points in individual_objects mode"
                    if negative_point_coords.ndim == 2:
                        negative_point_coords = negative_point_coords[:, np.newaxis, :]
                    # Extend negative coordinates to match the number of positive coordinates
                    while negative_point_coords.shape[0] < positive_point_coords.shape[0]:
                        negative_point_coords = np.concatenate((negative_point_coords, negative_point_coords[:1, :, :]), axis=0)
                    final_coords = np.concatenate((positive_point_coords, negative_point_coords), axis=1)
                else:
                    final_coords = np.concatenate((positive_point_coords, negative_point_coords), axis=0)
            else:
                final_coords = positive_point_coords

        # Handle possible bboxes
        if bboxes is not None:
            boxes_np_batch = []
            for bbox_list in bboxes:
                boxes_np = []
                for bbox in bbox_list:
                    boxes_np.append(bbox)
                boxes_np = np.array(boxes_np)
                boxes_np_batch.append(boxes_np)
            if individual_objects:
                final_box = np.array(boxes_np_batch)
            else:
                final_box = np.array(boxes_np)
            final_labels = None

        #handle labels
        if coordinates_positive is not None:
            if not individual_objects:
                positive_point_labels = np.ones(len(positive_point_coords))
            else:
                positive_labels = []
                for point in positive_point_coords:
                    positive_labels.append(np.array([1])) # 1)
                positive_point_labels = np.stack(positive_labels, axis=0)

            if coordinates_negative is not None:
                if not individual_objects:
                    negative_point_labels = np.zeros(len(negative_point_coords))  # 0 = negative
                    final_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=0)
                else:
                    negative_labels = []
                    for point in positive_point_coords:
                        negative_labels.append(np.array([0])) # 1)
                    negative_point_labels = np.stack(negative_labels, axis=0)
                    #combine labels
                    final_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=1)
            else:
                final_labels = positive_point_labels
            print("combined labels: ", final_labels)
            print("combined labels shape: ", final_labels.shape)

        mask_list = []
        try:
            model.to(device)
        except:
            model.model.to(device)

        autocast_condition = not mm.is_device_mps(device)
        print(f"Sam2Segmentation autocast device:{mm.get_autocast_device(device)} and dtype {dtype}  and autocast_condition:{autocast_condition}")
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            if segmentor == 'single_image':
                image_np = (image.contiguous() * 255).byte().numpy()
                comfy_pbar = ProgressBar(len(image_np))
                tqdm_pbar = tqdm(total=len(image_np), desc="Processing Images")
                for i in range(len(image_np)):
                    model.set_image(image_np[i])
                    if bboxes is None:
                        input_box = None
                    else:
                        if len(image_np) > 1:
                            input_box = final_box[i]
                        input_box = final_box

                    out_masks, scores, logits = model.predict(
                        point_coords=final_coords if coordinates_positive is not None else None,
                        point_labels=final_labels if coordinates_positive is not None else None,
                        box=input_box,
                        multimask_output=True if not individual_objects else False,
                        mask_input = input_mask[i].unsqueeze(0) if mask is not None else None,
                        )

                    if out_masks.ndim == 3:
                        sorted_ind = np.argsort(scores)[::-1]
                        out_masks = out_masks[sorted_ind][0] #choose only the best result for now
                        scores = scores[sorted_ind]
                        logits = logits[sorted_ind]
                        mask_list.append(np.expand_dims(out_masks, axis=0))
                    else:
                        _, _, H, W = out_masks.shape
                        # Combine masks for all object IDs in the frame
                        combined_mask = np.zeros((H, W), dtype=bool)
                        for out_mask in out_masks:
                            combined_mask = np.logical_or(combined_mask, out_mask)
                        combined_mask = combined_mask.astype(np.uint8)
                        mask_list.append(combined_mask)
                    comfy_pbar.update(1)
                    tqdm_pbar.update(1)

            elif segmentor == 'video':
                mask_list = []
                if hasattr(self, 'inference_state'):
                    model.reset_state(self.inference_state)
                self.inference_state = model.init_state(image.permute(0, 3, 1, 2).contiguous(), H, W, device=device)
                if bboxes is None:
                        input_box = None
                else:
                    input_box = bboxes[0]

                if individual_objects and bboxes is not None:
                    raise ValueError("bboxes not supported with individual_objects")


                if individual_objects:
                    for i, (coord, label) in enumerate(zip(final_coords, final_labels)):
                        _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=i,
                        points=final_coords[i],
                        labels=final_labels[i],
                        clear_old_points=True,
                        box=input_box
                        )
                else:
                    _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=final_coords if coordinates_positive is not None else None,
                        labels=final_labels if coordinates_positive is not None else None,
                        clear_old_points=True,
                        box=input_box
                    )

                pbar = ProgressBar(B)
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(self.inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    pbar.update(1)
                    if individual_objects:
                        _, _, H, W = out_mask_logits.shape
                        # Combine masks for all object IDs in the frame
                        combined_mask = np.zeros((H, W), dtype=np.uint8)
                        for i, out_obj_id in enumerate(out_obj_ids):
                            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                            combined_mask = np.logical_or(combined_mask, out_mask)
                        video_segments[out_frame_idx] = combined_mask

                if individual_objects:
                    for frame_idx, combined_mask in video_segments.items():
                        mask_list.append(combined_mask)
                else:
                    for frame_idx, obj_masks in video_segments.items():
                        for out_obj_id, out_mask in obj_masks.items():
                            mask_list.append(out_mask)

        if not keep_model_loaded:
            try:
                model.to(offload_device)
            except:
                model.model.to(offload_device)

        out_list = []
        for mask in mask_list:
            mask_tensor = torch.from_numpy(mask)
            mask_tensor = mask_tensor.permute(1, 2, 0)
            mask_tensor = mask_tensor[:, :, 0]
            out_list.append(mask_tensor)
        mask_tensor = torch.stack(out_list, dim=0).cpu().float()
        return (mask_tensor,)


def init_func():
    model = load_model(model_path, model_cfg_path, segmentor, dtype, device)
    sam2_seg = Sam2Segmentation()
    sam2_model = {
        'model': model,
        'dtype': dtype,
        'device': torch.device('hpu'),
        'segmentor' : segmentor,
        'version': version
    }
    return sam2_model, sam2_seg

