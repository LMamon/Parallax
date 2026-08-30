from dataclasses import dataclass
from pathlib import Path

import torch

from nanoowl.owl_predictor import (
    OwlDecodeOutput,
    OwlEncodeTextOutput,
    OwlPredictor,
)


@dataclass(frozen=True)
class NanoOwlResult:
    query: str
    query_revision: int
    output: OwlDecodeOutput


class NanoOwlDetector:
    def __init__(self, engine_path: str, model_name: str = "google/owlvit-base-patch32"):
        engine = Path(engine_path)

        if not engine.is_file():
            raise FileNotFoundError(f"NanoOWL engine not found: {engine}")

        self._engine_path = str(engine)
        self._model_name = model_name

        # NanoOWL owns model, TensorRT engine, and decode behavior.
        self._predictor = OwlPredictor(model_name, image_encoder_engine=self._engine_path)

        self._query = ""
        self._query_revision = 0
        self._text_encoding: OwlEncodeTextOutput | None = None
        self._query_encoding_count = 0
        self._closed = False

    @property
    def image_size(self) -> tuple[int, int]:
        self._require_open()
        return self._predictor.get_image_size()

    @property
    def engine_path(self) -> str:
        return self._engine_path

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def query(self) -> str:
        return self._query

    @property
    def query_revision(self) -> int:
        return self._query_revision

    @property
    def query_encoding_count(self) -> int:
        return self._query_encoding_count

    @property
    def closed(self) -> bool:
        return self._closed

    def set_query(self, query: str, revision: int) -> None:
        self._require_open()

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        if revision <= 0:
            raise ValueError("query revision must be positive")

        # Keep one bounded cached encoding for the active query.
        if query != self._query:
            self._text_encoding = self._predictor.encode_text([query])
            self._query_encoding_count += 1

        self._query = query
        self._query_revision = revision

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, threshold: float = 0.1) -> NanoOwlResult:
        self._require_open()

        if self._text_encoding is None:
            raise RuntimeError("query is not set")

        if not image.is_cuda:
            raise ValueError("image must be CUDA-resident")

        if image.dtype != torch.float32:
            raise ValueError("image must be float32")

        if image.ndim != 4:
            raise ValueError("image must be NCHW")

        if image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("image must have shape [1, 3, H, W]")

        height, width = self.image_size

        if image.shape[2] != height or image.shape[3] != width:
            raise ValueError(f"image must be {width}x{height}")

        # Caller owns the active Torch/CUDA stream.
        image_output = self._predictor.encode_image(image)
        output = self._predictor.decode(image_output, self._text_encoding, threshold)

        return NanoOwlResult(query=self._query, query_revision=self._query_revision, output=output)

    def close(self) -> None:
        if self._closed:
            return

        # Drop backend references without a device-wide synchronize.
        self._text_encoding = None
        self._predictor = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("NanoOWL detector is closed")