#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F


LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path: Path):
    with path.open("rb") as file:
        data = file.read()

    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(data)

    if engine is None:
        raise RuntimeError(f"failed to load engine: {path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"failed to create context: {path}")

    return runtime, engine, context


def preprocess(image: np.ndarray):
    height, width = image.shape[:2]

    scale = 512.0 / max(height, width)
    new_height = int(height * scale + 0.5)
    new_width = int(width * scale + 0.5)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = (
        torch.from_numpy(rgb)
        .cuda()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )

    tensor = F.interpolate(
        tensor,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )

    mean = torch.tensor(
        [123.675, 116.28, 103.53],
        device="cuda",
        dtype=torch.float32,
    ).view(1, 3, 1, 1) / 255.0

    std = torch.tensor(
        [58.395, 57.12, 57.375],
        device="cuda",
        dtype=torch.float32,
    ).view(1, 3, 1, 1) / 255.0

    tensor = (tensor - mean) / std
    tensor = F.pad(tensor, (0, 512 - new_width, 0, 512 - new_height))

    return tensor.contiguous(), (new_height, new_width)


def transform_box(box, original_size, input_size):
    height, width = original_size
    new_height, new_width = input_size

    x0, y0, x1, y1 = box

    coords = torch.tensor(
        [[[x0, y0], [x1, y1]]],
        dtype=torch.float32,
        device="cuda",
    )

    coords[..., 0] *= new_width / width
    coords[..., 1] *= new_height / height

    # SAM encodes box corners as labels 2 and 3.
    labels = torch.tensor(
        [[2.0, 3.0]],
        dtype=torch.float32,
        device="cuda",
    )

    return coords.contiguous(), labels


def run_engine(engine, context, tensors, stream):
    for name, tensor in tensors.items():
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = engine.get_tensor_shape(name)

            if any(dim == -1 for dim in shape):
                if not context.set_input_shape(name, tuple(tensor.shape)):
                    raise RuntimeError(f"failed to set input shape: {name}")

        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"failed to bind tensor: {name}")

    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT execution failed")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--box",
        required=True,
        help="XYXY box: x0,y0,x1,y1",
    )

    parser.add_argument(
        "--encoder",
        default="models/efficientvit-sam/engines/l0_encoder_fp16.engine",
    )

    parser.add_argument(
        "--decoder",
        default="models/efficientvit-sam/engines/l0_decoder_fp16.engine",
    )

    parser.add_argument(
        "--output",
        default="/tmp/efficientvit_sam_mask.png",
    )

    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"failed to read image: {args.image}")

    box = tuple(float(value) for value in args.box.split(","))
    if len(box) != 4:
        raise RuntimeError("--box requires x0,y0,x1,y1")

    height, width = image.shape[:2]
    input_tensor, input_size = preprocess(image)

    prompt_size = (
        int(height * (1024.0 / max(height, width)) + 0.5),
        int(width * (1024.0 / max(height, width)) + 0.5),
    )

    coords, labels = transform_box(
        box,
        (height, width),
        prompt_size,
    )

    encoder_runtime, encoder, encoder_context = load_engine(
        Path(args.encoder)
    )

    decoder_runtime, decoder, decoder_context = load_engine(
        Path(args.decoder)
    )

    embedding = torch.empty(
        (1, 256, 64, 64),
        dtype=torch.float32,
        device="cuda",
    )

    low_res_mask = torch.empty(
        (1, 1, 256, 256),
        dtype=torch.float32,
        device="cuda",
    )

    iou = torch.empty(
        (1, 1),
        dtype=torch.float32,
        device="cuda",
    )

    stream = torch.cuda.current_stream()

    encoder_start = torch.cuda.Event(enable_timing=True)
    encoder_end = torch.cuda.Event(enable_timing=True)

    decoder_start = torch.cuda.Event(enable_timing=True)
    decoder_end = torch.cuda.Event(enable_timing=True)

    encoder_start.record(stream)

    run_engine(
        encoder,
        encoder_context,
        {
            "input_image": input_tensor,
            "image_embeddings": embedding,
        },
        stream,
    )

    encoder_end.record(stream)
    decoder_start.record(stream)

    run_engine(
        decoder,
        decoder_context,
        {
            "image_embeddings": embedding,
            "point_coords": coords,
            "point_labels": labels,
            "masks": low_res_mask,
            "iou_predictions": iou,
        },
        stream,
    )

    decoder_end.record(stream)
    stream.synchronize()

    if not torch.isfinite(low_res_mask).all():
        raise RuntimeError("mask contains non-finite values")

    if not torch.isfinite(iou).all():
        raise RuntimeError("IoU prediction is non-finite")

    new_height, new_width = input_size

    mask = F.interpolate(
        low_res_mask,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    )

    prompt_height, prompt_width = prompt_size
    mask = mask[..., :prompt_height, :prompt_width]

    mask = F.interpolate(
        mask,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    binary = mask[0, 0] > 0.0
    coverage = binary.float().mean().item()

    if coverage <= 0.0:
        raise RuntimeError("mask is empty")

    output = binary.mul(255).byte().cpu().numpy()

    if not cv2.imwrite(args.output, output):
        raise RuntimeError(f"failed to write: {args.output}")

    print(f"image: {width}x{height}")
    print(f"encoder input: 512x512")
    print(f"resized content: {new_width}x{new_height}")
    print(f"box: {box}")
    print(f"iou prediction: {iou.item():.4f}")
    print(f"mask coverage: {coverage * 100.0:.2f}%")
    print(f"encoder: {encoder_start.elapsed_time(encoder_end):.3f} ms")
    print(f"decoder: {decoder_start.elapsed_time(decoder_end):.3f} ms")
    print(f"mask: {args.output}")
    print("PASS: TensorRT box-prompted EfficientViT-SAM inference")

if __name__ == "__main__":
    main()