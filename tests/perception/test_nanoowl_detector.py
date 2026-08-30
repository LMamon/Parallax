import os
import unittest

import PIL.Image
import torch
from torchvision.ops import roi_align

from nanoowl.image_preprocessor import ImagePreprocessor

from parallax.nanoowl_detector import NanoOwlDetector


ENGINE_PATH = os.environ.get("PARALLAX_NANOOWL_ENGINE",
                             "/workspace/Parallax/models/"
                             "owl_image_encoder_patch32_fp16.engine")

IMAGE_PATH = "/opt/nanoowl/assets/owl_glove_small.jpg"


class NanoOwlDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = NanoOwlDetector(ENGINE_PATH)

        image = PIL.Image.open(IMAGE_PATH).convert("RGB")

        preprocessor = ImagePreprocessor().cuda().eval()
        image_tensor = preprocessor.preprocess_pil_image(image)

        width, height = image.size

        rois = torch.tensor([[0, 0, width, height]], dtype=image_tensor.dtype, device=image_tensor.device)
        cls.image = roi_align(image_tensor, [rois], output_size=cls.detector.image_size)

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()

    def test_reference_detection(self):
        self.detector.set_query("an owl", 1)

        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            result = self.detector.predict(
                self.image,
                threshold=0.1,
            )

        stream.synchronize()

        self.assertEqual(result.query, "an owl")
        self.assertEqual(result.query_revision, 1)
        self.assertGreater(len(result.output.scores), 0)

        labels = result.output.labels.cpu()
        scores = result.output.scores.cpu()
        boxes = result.output.boxes.cpu()

        self.assertEqual(labels.ndim, 1)
        self.assertEqual(scores.ndim, 1)
        self.assertEqual(boxes.ndim, 2)

        self.assertEqual(len(labels), len(scores))
        self.assertEqual(len(scores), len(boxes))

    def test_query_encoding_is_cached(self):
        before = self.detector.query_encoding_count

        self.detector.set_query("person", 2)
        after_first = self.detector.query_encoding_count

        self.detector.set_query("person", 3)
        after_second = self.detector.query_encoding_count

        self.assertEqual(after_first, before + 1)
        self.assertEqual(after_second, after_first)
        self.assertEqual(self.detector.query_revision, 3)

    def test_query_replacement_reencodes(self):
        self.detector.set_query("person", 4)
        before = self.detector.query_encoding_count

        self.detector.set_query("cup", 5)
        after = self.detector.query_encoding_count

        self.assertEqual(after, before + 1)
        self.assertEqual(self.detector.query, "cup")
        self.assertEqual(self.detector.query_revision, 5)

    def test_missing_engine_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            NanoOwlDetector("/tmp/parallax-engine-does-not-exist.engine")

if __name__ == "__main__":
    unittest.main()