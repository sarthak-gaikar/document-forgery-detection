"""
tests/test_data.py
Unit tests for document generation and tampering engines.
"""
from __future__ import annotations

import random
import numpy as np
import pytest

from src.data.synthetic_gen import DocumentGenerator, TamperingEngine, apply_jpeg_recompression


def test_document_generator_receipt():
    gen = DocumentGenerator(seed=42)
    doc = gen.generate_receipt()
    assert "image" in doc
    assert "text_regions" in doc
    assert doc["doc_type"] == "receipt"
    assert len(doc["text_regions"]) > 0
    assert doc["image"].size == (794, 1123)


def test_document_generator_all_types():
    gen = DocumentGenerator(seed=123)
    for method_name in ["generate_invoice", "generate_form", "generate_certificate", "generate_marksheet"]:
        method = getattr(gen, method_name)
        doc = method()
        assert doc["image"].size == (794, 1123)
        assert len(doc["text_regions"]) > 0


def test_tampering_engine_text_replacement():
    gen = DocumentGenerator(seed=42)
    engine = TamperingEngine()
    doc = gen.generate_receipt()
    rng = random.Random(42)

    tampered = engine.text_replacement(doc, rng)
    assert tampered["is_forged"] is True
    assert tampered["tamper_type"] == "text_replacement"

    mask_arr = np.array(tampered["mask"])
    assert mask_arr.max() == 255  # tampered pixels exist


def test_tampering_engine_copy_move():
    gen = DocumentGenerator(seed=42)
    engine = TamperingEngine()
    doc = gen.generate_invoice()
    rng = random.Random(42)

    tampered = engine.copy_move(doc, rng)
    assert tampered["is_forged"] is True
    assert tampered["tamper_type"] == "copy_move"
    mask_arr = np.array(tampered["mask"])
    assert mask_arr.max() == 255


def test_jpeg_recompression():
    gen = DocumentGenerator(seed=42)
    doc = gen.generate_form()
    recompressed = apply_jpeg_recompression(doc["image"], q1=90, q2=70)
    assert recompressed.size == doc["image"].size

