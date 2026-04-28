# src/anpr/deblurrer.py
"""
Plate image deblurring via DeblurGAN-v2 (Keras h5 model).

GPU Acceleration:
  • Automatically detects and uses GPU for TensorFlow inference.
  • Configures GPU memory growth to avoid OOM errors.
  • Uses tf.function-wrapped predict for faster repeated calls.

Falls back to returning the original image if TensorFlow is unavailable
or the model fails to load — this ensures the pipeline never crashes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BLUR_THRESHOLD = 90.0         # Laplacian variance below this → blurry
_MODEL_INPUT_SIZE = (256, 256)
_CACHE_MAXSIZE = 128           # max deblurred crops cached in memory


def _configure_tf_gpu() -> None:
    """
    Configure TensorFlow to use GPU with memory growth enabled.
    This prevents TF from grabbing all GPU memory at once,
    leaving room for PyTorch (YOLO) models.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("TensorFlow GPU(s) configured with memory growth: %s", gpus)
        else:
            logger.info("No TensorFlow GPU devices found — will use CPU.")
    except Exception as exc:
        logger.warning("TensorFlow GPU config failed: %s", exc)


class Deblurrer:
    """
    Load the DeblurGAN-v2 Keras model and enhance blurry plate crops.

    Parameters
    ----------
    weights_path : str
        Path to ``fpn_inception.h5``.
    blur_threshold : float
        Images with Laplacian variance below this are enhanced.
        Set to 0 to always enhance.
    use_gpu : bool
        If True, attempt to use GPU for TensorFlow inference.
    """

    def __init__(
        self,
        weights_path: str = "models/fpn_inception.h5",
        blur_threshold: float = _BLUR_THRESHOLD,
        use_gpu: bool = True,
    ) -> None:
        env_threshold = os.getenv("DEBLUR_BLUR_THRESHOLD", "").strip()
        if env_threshold:
            try:
                blur_threshold = float(env_threshold)
            except ValueError:
                logger.warning("Invalid DEBLUR_BLUR_THRESHOLD='%s' (using %.1f)", env_threshold, blur_threshold)

        self.blur_threshold = blur_threshold
        self.always_on = os.getenv("DEBLUR_ALWAYS", "0").strip().lower() in {"1", "true", "yes"}
        self._model = None
        self._predict_fn = None  # tf.function-wrapped predict for speed
        self._cache: dict[str, np.ndarray] = {}
        self._stats = {"calls": 0, "cache_hits": 0, "deblurred": 0, "skipped_sharp": 0, "total_ms": 0.0}

        if use_gpu:
            _configure_tf_gpu()

        self._load_model(weights_path, use_gpu)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, weights_path: str, use_gpu: bool = True) -> None:
        try:
            import tensorflow as tf  # noqa: F401 — soft dependency

            # Set device context for model loading
            device_str = "/GPU:0" if (use_gpu and tf.config.list_physical_devices("GPU")) else "/CPU:0"
            logger.info("Loading DeblurGAN model on %s …", device_str)

            with tf.device(device_str):
                self._model = tf.keras.models.load_model(weights_path, compile=False)

            # Create a tf.function-wrapped predict for faster repeated calls
            @tf.function(reduce_retracing=True)
            def _fast_predict(inp):
                return self._model(inp, training=False)

            self._predict_fn = _fast_predict

            # Warm up the model with a dummy input to pre-compile the graph
            dummy = np.zeros((1, *_MODEL_INPUT_SIZE, 3), dtype=np.float32)
            _ = self._predict_fn(tf.constant(dummy))

            logger.info("DeblurGAN model loaded from %s (device=%s, warm-up complete)",
                        weights_path, device_str)

        except ImportError:
            logger.warning(
                "TensorFlow not installed — deblurring disabled. "
                "Install tensorflow to enable DeblurGAN."
            )
        except Exception as exc:
            logger.warning("Failed to load DeblurGAN model: %s — deblurring disabled.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if the DeblurGAN model is loaded and ready."""
        return self._model is not None

    @property
    def stats(self) -> dict:
        """Return performance statistics."""
        return dict(self._stats)

    def is_blurry(self, image: np.ndarray) -> bool:
        """Return True if Laplacian variance is below the configured threshold."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.blur_threshold

    @staticmethod
    def _fallback_enhance(image: np.ndarray) -> np.ndarray:
        """Fast non-TensorFlow enhancement used when DeblurGAN is unavailable."""
        if image.ndim == 2:
            base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            base = image

        blurred = cv2.GaussianBlur(base, (0, 0), 1.0)
        sharpened = cv2.addWeighted(base, 1.45, blurred, -0.45, 0)

        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(l_chan)
        return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    def enhance(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        """
        Enhance a plate crop. Returns the enhanced image or the original
        if the model is not available or the image is already sharp.

        Parameters
        ----------
        force : bool
            If True, bypass blur gating and always apply enhancement.

        Uses image-hash caching to avoid re-processing identical crops.
        """
        self._stats["calls"] += 1

        if not force and not self.always_on and not self.is_blurry(image):
            self._stats["skipped_sharp"] += 1
            return image  # already sharp — no need to enhance

        # Cache lookup
        img_key = self._img_hash(image)
        if img_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[img_key]

        try:
            t0 = time.perf_counter()

            if self._model is None:
                out = self._fallback_enhance(image)
            else:
                import tensorflow as tf

                # Preprocess: resize → normalise to [-1, 1]
                resized   = cv2.resize(image, _MODEL_INPUT_SIZE)
                inp       = resized.astype(np.float32) / 127.5 - 1.0
                inp_batch = np.expand_dims(inp, axis=0)   # shape (1, 256, 256, 3)

                # Inference using tf.function for speed
                if self._predict_fn is not None:
                    out_batch = self._predict_fn(tf.constant(inp_batch))
                    out = out_batch.numpy()[0]
                else:
                    out_batch = self._model.predict(inp_batch, verbose=0)
                    out       = out_batch[0]

                # Post-process: de-normalise → resize back to original dimensions
                out = (out + 1.0) * 127.5
                out = np.clip(out, 0, 255).astype(np.uint8)
                out = cv2.resize(out, (image.shape[1], image.shape[0]))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._stats["deblurred"] += 1
            self._stats["total_ms"] += elapsed_ms

            if self._model is None:
                logger.info("Deblur fallback enhanced plate crop in %.1fms", elapsed_ms)
            else:
                logger.info("DeblurGAN enhanced plate crop in %.1fms", elapsed_ms)

            # Cache result
            if len(self._cache) >= _CACHE_MAXSIZE:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[img_key] = out

            return out

        except Exception as exc:
            logger.error("Deblurring failed: %s — using fallback enhancement.", exc)
            return self._fallback_enhance(image)

    @staticmethod
    def _img_hash(img: np.ndarray) -> str:
        """Fast perceptual hash for cache key."""
        thumb = cv2.resize(img, (16, 8), interpolation=cv2.INTER_AREA)
        if thumb.ndim == 3:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(thumb.tobytes()).hexdigest()