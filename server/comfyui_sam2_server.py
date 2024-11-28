import sys

import json
import yaml

import torch
import math
import struct
import safetensors.torch
import numpy as np
from PIL import Image
import logging
import itertools
from torch.functional import F

from tqdm import tqdm
from contextlib import nullcontext


sys.path.append("/lkk/new_comfyui/sam/GenAIComps/")

from comps import (
    CustomLogger,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
import base64

logger = CustomLogger("sam2-servers")

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union

from io import BytesIO

class Sam2Inputs(BaseModel):
    image: str = None
    keep_model_loaded: bool = None
    coordinates_positive: str = None
    coordinates_negative: str = None
    individual_objects: bool = None
    bboxes: str = None
    mask: str = None
    mask_shape: List[int] = None


initialized = False

sam2_model = None
sam2_seg = None

def init_model():
    global initialized, sam2_model, sam2_seg
    if not initialized:
        from sam2_loading import init_func
        sam2_model, sam2_seg = init_func()
        initialized = True

@register_microservice(
    name="opea_service@sam2seg",
    service_type=ServiceType.TEXT2IMAGE,
    endpoint="/v1/sam2seg",
    host="198.175.100.223",
    port=443,
)
def img2mask(request: Sam2Inputs):
    init_model()
    # print(request)
    image_str = request.image
    image_byte = base64.b64decode(image_str)
    image_io = BytesIO(image_byte)  # convert image to file-like object
    image = Image.open(image_io)   # img is now PIL Image object

    # https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py#L1577

    image = image.convert("RGB")

    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    print(image)
    print(image.shape)

    mask_str = request.mask
    mask_byte = base64.b64decode(mask_str)

    mask = np.frombuffer(mask_byte, dtype=np.float32).reshape(request.mask_shape)
    print(mask)
    mask = torch.from_numpy(mask)
    print(mask)
    print(mask.shape)

    output_mask = sam2_seg.segment(image,
        sam2_model,
        request.keep_model_loaded,
        coordinates_positive=request.coordinates_positive,
        coordinates_negative=request.coordinates_negative,
        individual_objects=request.individual_objects,
        bboxes=None,
        mask=mask)
    print(output_mask)
    print(output_mask[0].shape)

    output_mask_str = base64.b64encode(output_mask[0].numpy().tobytes()).decode('utf-8')

    return {"output_mask": output_mask_str, "mask_shape": output_mask[0].shape}


if __name__ == "__main__":
    opea_microservices["opea_service@sam2seg"].start()


