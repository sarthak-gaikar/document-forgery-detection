"""
tests/test_api.py
Unit tests for the FastAPI backend functions (offline-safe).
"""
from __future__ import annotations

import io
from PIL import Image
import pytest

from src.api.app import index, health


def test_index_endpoint():
    data = index()
    assert "Document Forgery Detection API" in data["service"]
    assert "disclaimer" in data


def test_health_endpoint():
    data = health()
    assert data["status"] == "healthy"
    assert "device" in data
