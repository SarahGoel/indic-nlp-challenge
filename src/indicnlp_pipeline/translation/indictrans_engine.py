
"""Local IndicTrans2 inference engine for the Indic NLP Challenge.

This module provides lazy model loading, language-pair validation,
batched translation, and explicit output handling.

Model weights and generated translations are never written to ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from loguru import logger


class IndicTransEngineError(RuntimeError):
    """Raised when IndicTrans2 configuration or inference fails."""


class IndicTransEngine:
    """Load and run a configured IndicTrans2 Indic-to-Indic checkpoint.

    Model and tokenizer loading is lazy. Constructing the engine does not
    download weights or allocate model memory.

    Args:
        matrix_path: Path to configs/translation_matrix.yaml.
        checkpoint: Optional checkpoint identifier or local checkpoint path.
        device: Explicit device ('cpu', 'cuda') or None for automatic selection.
        batch_size: Maximum number of sentences processed per batch.
        max_input_tokens: Maximum tokenized input length.
        max_new_tokens: Maximum number of generated tokens.
        local_files_only: If True, prohibit Hugging Face network downloads.
    """

    def __init__(
        self,
        matrix_path: Path | None = None,
        checkpoint: str | None = None,
        device: str | None = None,
        batch_size: int = 4,
        max_input_tokens: int = 512,
        max_new_tokens: int = 512,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("Token limits must be positive.")

        self.repository_root = self._find_repository_root()
        self.matrix_path = (
            matrix_path
            if matrix_path is not None
            else self.repository_root / "configs" / "translation_matrix.yaml"
        )
        self.matrix_path = self.matrix_path.resolve()

        self.matrix = self._load_matrix()
        backend = self.matrix.get("backends", {}).get("indictrans2", {})

        configured_checkpoint = backend.get("checkpoint")
        if not checkpoint and not configured_checkpoint:
            raise IndicTransEngineError(
                "No IndicTrans2 checkpoint is configured."
            )

        self.checkpoint = checkpoint or configured_checkpoint
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only

        if device is not None:
            if device not in {"cpu", "cuda"}:
                raise ValueError("device must be 'cpu', 'cuda', or None.")
            if device == "cuda" and not torch.cuda.is_available():
                raise IndicTransEngineError(
                    "CUDA was requested but is not available."
                )
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.local_model_dir = self.repository_root / "models" / "indictrans2"

        self._tokenizer: Any = None
        self._model: Any = None
        self._processor: Any = None

        logger.info(
            "IndicTransEngine initialized: checkpoint={}, device={}, batch_size={}",
            self.checkpoint,
            self.device,
            self.batch_size,
        )

    def _find_repository_root(self) -> Path:
        """Find the repository root by locating the project configuration."""
        for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
            if (candidate / "configs" / "config.yaml").is_file():
                return candidate

        raise IndicTransEngineError(
            "Could not locate repository root containing configs/config.yaml."
        )

    def _load_matrix(self) -> dict[str, Any]:
        """Load and validate the translation matrix YAML file."""
        if not self.matrix_path.is_file():
            raise FileNotFoundError(
                f"Translation matrix not found: {self.matrix_path}"
            )

        with self.matrix_path.open("r", encoding="utf-8") as file:
            matrix = yaml.safe_load(file)

        if not isinstance(matrix, dict):
            raise IndicTransEngineError(
                "Translation matrix must contain a YAML mapping."
            )

        if not isinstance(matrix.get("languages"), dict):
            raise IndicTransEngineError(
                "Translation matrix must define a languages mapping."
            )

        if not isinstance(matrix.get("translation_matrix"), dict):
            raise IndicTransEngineError(
                "Translation matrix must define translation_matrix."
            )

        return matrix

    def _validate_pair(self, source_lang: str, target_lang: str) -> tuple[str, str]:
        """Validate a configured pair and return its model language codes."""
        languages = self.matrix["languages"]
        pairs = self.matrix["translation_matrix"]

        if source_lang not in languages:
            raise ValueError(f"Unknown source language: {source_lang}")
        if target_lang not in languages:
            raise ValueError(f"Unknown target language: {target_lang}")

        pair = pairs.get(source_lang, {}).get(target_lang)
        if not isinstance(pair, dict):
            raise ValueError(
                f"Language pair {source_lang}->{target_lang} is not configured."
            )

        if pair.get("status") != "configured":
            raise ValueError(
                f"Language pair {source_lang}->{target_lang} is not enabled: "
                f"{pair.get('reason', pair.get('status', 'unknown status'))}"
            )

        source_code = languages[source_lang].get("model_code")
        target_code = languages[target_lang].get("model_code")

        if not source_code or not target_code:
            raise IndicTransEngineError(
                f"Missing model language code for {source_lang}->{target_lang}."
            )

        return source_code, target_code

    def load_model(self) -> None:
        """Load the tokenizer, model, and IndicProcessor once."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from IndicTransToolkit import IndicProcessor
        except ImportError as exc:
            raise IndicTransEngineError(
                "IndicTrans2 dependencies are missing or cannot be imported. "
                "Check the compatible transformers version and install "
                "IndicTransToolkit in the project virtual environment."
            ) from exc

        # Prefer a complete local checkpoint when one has been prepared.
        local_config = self.local_model_dir / "config.json"
        model_source = (
            str(self.local_model_dir)
            if local_config.is_file()
            else self.checkpoint
        )

        logger.info("Loading IndicTrans2 checkpoint from {}", model_source)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_source,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
            )

            model.to(self.device)
            model.eval()

            processor = IndicProcessor(inference=True)

        except Exception as exc:
            logger.exception("IndicTrans2 model initialization failed.")
            raise IndicTransEngineError(
                "Could not load the IndicTrans2 model. Check checkpoint "
                "availability, Hugging Face access requirements, local cache, "
                "and dependency compatibility."
            ) from exc

        self._tokenizer = tokenizer
        self._model = model
        self._processor = processor

        logger.info("IndicTrans2 model loaded successfully.")

    def translate_batch(
        self,
        texts: Sequence[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate an ordered batch of source sentences.

        Empty input returns an empty list. Blank strings remain blank and
        retain their original positions in the returned list.
        """
        source_code, target_code = self._validate_pair(
            source_lang, target_lang
        )

        if isinstance(texts, (str, bytes)):
            raise TypeError(
                "texts must be a sequence of strings, not a single string."
            )

        inputs = list(texts)
        if any(not isinstance(text, str) for text in inputs):
            raise TypeError("Every input segment must be a string.")

        if not inputs:
            return []

        results = [
            text if not text.strip() else ""
            for text in inputs
        ]
        pending_indices = [
            index for index, text in enumerate(inputs) if text.strip()
        ]

        if not pending_indices:
            return results

        self.load_model()

        assert self._processor is not None
        assert self._tokenizer is not None
        assert self._model is not None

        try:
            for start in range(0, len(pending_indices), self.batch_size):
                batch_indices = pending_indices[start : start + self.batch_size]
                batch_texts = [inputs[index] for index in batch_indices]

                prepared = self._processor.preprocess_batch(
                    batch_texts,
                    src_lang=source_code,
                    tgt_lang=target_code,
                )

                encoded = self._tokenizer(
                    prepared,
                    truncation=True,
                    max_length=self.max_input_tokens,
                    padding="longest",
                    return_tensors="pt",
                )

                encoded = {
                    name: tensor.to(self.device)
                    for name, tensor in encoded.items()
                }

                with torch.inference_mode():
                    generated = self._model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=False,
                    )

                decoded = self._tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                translations = self._processor.postprocess_batch(
                    decoded,
                    lang=target_code,
                )

                if len(translations) != len(batch_indices):
                    raise IndicTransEngineError(
                        "Model output count does not match input batch size."
                    )

                for index, translation in zip(
                    batch_indices, translations, strict=True
                ):
                    if not isinstance(translation, str):
                        raise IndicTransEngineError(
                            "Model returned a non-string translation."
                        )
                    results[index] = translation

        except IndicTransEngineError:
            raise
        except Exception as exc:
            logger.exception(
                "Translation failed for language pair {}->{}.",
                source_lang,
                target_lang,
            )
            raise IndicTransEngineError(
                f"Translation failed for {source_lang}->{target_lang}."
            ) from exc

        return results

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate one text segment."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        return self.translate_batch(
            [text],
            source_lang=source_lang,
            target_lang=target_lang,
        )[0]