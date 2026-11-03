"""
src/data/synthetic_gen.py
Synthetic document forgery dataset generator.

Generates realistic A4 document images (receipts, invoices, forms, certificates,
marksheets) using Pillow + Faker, then applies one of 5 tampering operations
with binary pixel-level masks.
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import struct
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOC_W, DOC_H = 794, 1123          # A4 at 96 dpi
MARGIN = 40
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (10, 10, 10)
ACCENT_COLOR = (30, 60, 120)
LIGHT_GRAY = (220, 220, 220)

# font sizes (pixels) for PIL default bitmap font is fixed-size; we use
# ImageFont.load_default(size=N) which is available in Pillow ≥ 10.
# Fallback to load_default() (fixed 10px) if the size kwarg is not supported.

def _load_font(size: int) -> ImageFont.ImageFont:
    """Load PIL default font at the requested size, with a safe fallback."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow < 10 – load_default() has no size arg
        return ImageFont.load_default()


FONT_TITLE  = _load_font(24)
FONT_HEADER = _load_font(18)
FONT_BODY   = _load_font(14)
FONT_SMALL  = _load_font(11)

DOC_TYPES = ["receipt", "invoice", "form", "certificate", "marksheet"]

TAMPER_TYPES = ["text_replacement", "copy_move", "text_insertion",
                "text_removal", "splicing"]

# Weighted sampling of tamper types (splicing intentionally less frequent)
TAMPER_WEIGHTS = [0.30, 0.25, 0.20, 0.17, 0.08]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont,
               xy: Tuple[int, int]) -> List[int]:
    """Return [x1, y1, x2, y2] for text rendered at xy."""
    x, y = xy
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    except AttributeError:
        # Pillow < 8.0 fallback
        w, h = draw.textsize(text, font=font)
        left, top, right, bottom = x, y, x + w, y + h
    return [left, top, right, bottom]


def _draw_line(draw: ImageDraw.ImageDraw, y: int, color=LIGHT_GRAY, width: int = 1) -> None:
    draw.line([(MARGIN, y), (DOC_W - MARGIN, y)], fill=color, width=width)


def _draw_rect(draw: ImageDraw.ImageDraw, bbox: List[int],
               fill=None, outline=None, width: int = 1) -> None:
    draw.rectangle(bbox, fill=fill, outline=outline, width=width)


def _new_doc() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (DOC_W, DOC_H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    return img, draw


# ---------------------------------------------------------------------------
# DocumentGenerator
# ---------------------------------------------------------------------------

class DocumentGenerator:
    """Generates realistic synthetic A4 document images with text region metadata."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)

    # ------------------------------------------------------------------
    # Receipt
    # ------------------------------------------------------------------

    def generate_receipt(self) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        img, draw = _new_doc()
        text_regions: List[Dict] = []
        y = MARGIN

        # Header
        store = self.fake.company()
        bbox = _text_bbox(draw, store, FONT_TITLE, (MARGIN, y))
        draw.text((MARGIN, y), store, font=FONT_TITLE, fill=ACCENT_COLOR)
        text_regions.append({"bbox": bbox, "text": store, "field": "store_name"})
        y = bbox[3] + 6

        addr = self.fake.address().replace("\n", "  ")
        bbox = _text_bbox(draw, addr, FONT_SMALL, (MARGIN, y))
        draw.text((MARGIN, y), addr, font=FONT_SMALL, fill=TEXT_COLOR)
        text_regions.append({"bbox": bbox, "text": addr, "field": "address"})
        y = bbox[3] + 4

        phone = self.fake.phone_number()
        bbox = _text_bbox(draw, f"Tel: {phone}", FONT_SMALL, (MARGIN, y))
        draw.text((MARGIN, y), f"Tel: {phone}", font=FONT_SMALL, fill=TEXT_COLOR)
        text_regions.append({"bbox": bbox, "text": phone, "field": "phone"})
        y = bbox[3] + 10

        _draw_line(draw, y, color=ACCENT_COLOR, width=2)
        y += 12

        # Receipt title
        title = "RECEIPT"
        bbox = _text_bbox(draw, title, FONT_HEADER, (DOC_W // 2 - 40, y))
        draw.text((DOC_W // 2 - 40, y), title, font=FONT_HEADER, fill=ACCENT_COLOR)
        y = bbox[3] + 10

        # Receipt number & date
        receipt_no = f"RCP-{self.rng.randint(10000, 99999)}"
        date_str = self.fake.date_this_year().strftime("%d/%m/%Y")
        for label, val, field in [("Receipt No:", receipt_no, "receipt_no"),
                                   ("Date:", date_str, "date")]:
            row = f"{label:<14}{val}"
            bbox = _text_bbox(draw, row, FONT_BODY, (MARGIN, y))
            draw.text((MARGIN, y), row, font=FONT_BODY, fill=TEXT_COLOR)
            text_regions.append({"bbox": bbox, "text": val, "field": field})
            y = bbox[3] + 4

        y += 8
        _draw_line(draw, y)
        y += 10

        # Items
        n_items = self.rng.randint(3, 10)
        total = 0.0
        for _ in range(n_items):
            item_name = self.fake.word().capitalize() + " " + self.fake.word().capitalize()
            qty = self.rng.randint(1, 5)
            price = round(self.rng.uniform(1.0, 99.0), 2)
            subtotal = round(qty * price, 2)
            total += subtotal
            row = f"{item_name:<28}{qty:>3} x ${price:>7.2f}  ${subtotal:>8.2f}"
            bbox = _text_bbox(draw, row, FONT_BODY, (MARGIN, y))
            draw.text((MARGIN, y), row, font=FONT_BODY, fill=TEXT_COLOR)
            text_regions.append({"bbox": bbox, "text": row,
                                  "field": "line_item"})
            y = bbox[3] + 4
            if y > DOC_H - 200:
                break

        y += 6
        _draw_line(draw, y)
        y += 10

        # Totals
        tax = round(total * 0.1, 2)
        grand = round(total + tax, 2)
        for label, val, field in [("Subtotal:", f"${total:.2f}", "subtotal"),
                                   ("Tax (10%):", f"${tax:.2f}", "tax"),
                                   ("TOTAL:", f"${grand:.2f}", "total")]:
            font = FONT_HEADER if field == "total" else FONT_BODY
            row = f"{label:<16}{val:>12}"
            bbox = _text_bbox(draw, row, font, (MARGIN, y))
            draw.text((MARGIN, y), row, font=font, fill=TEXT_COLOR)
            text_regions.append({"bbox": bbox, "text": val, "field": field})
            y = bbox[3] + 4

        y += 14
        footer = "Thank you for your purchase!"
        bbox = _text_bbox(draw, footer, FONT_SMALL, (MARGIN, y))
        draw.text((MARGIN, y), footer, font=FONT_SMALL, fill=ACCENT_COLOR)
        text_regions.append({"bbox": bbox, "text": footer, "field": "footer"})

        return {"image": img, "text_regions": text_regions,
                "doc_type": "receipt", "source_doc_id": doc_id}

    # ------------------------------------------------------------------
    # Invoice
    # ------------------------------------------------------------------

    def generate_invoice(self) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        img, draw = _new_doc()
        text_regions: List[Dict] = []
        y = MARGIN

        # Company header block
        company = self.fake.company()
        bbox = _text_bbox(draw, company, FONT_TITLE, (MARGIN, y))
        draw.text((MARGIN, y), company, font=FONT_TITLE, fill=ACCENT_COLOR)
        text_regions.append({"bbox": bbox, "text": company, "field": "company_name"})
        y = bbox[3] + 6

        for line in self.fake.address().split("\n"):
            bbox = _text_bbox(draw, line.strip(), FONT_SMALL, (MARGIN, y))
            draw.text((MARGIN, y), line.strip(), font=FONT_SMALL, fill=TEXT_COLOR)
            y = bbox[3] + 3

        # INVOICE title on right
        inv_title = "INVOICE"
        bbox = _text_bbox(draw, inv_title, FONT_TITLE, (DOC_W - MARGIN - 120, MARGIN))
        draw.text((DOC_W - MARGIN - 120, MARGIN), inv_title,
                  font=FONT_TITLE, fill=ACCENT_COLOR)
        text_regions.append({"bbox": bbox, "text": inv_title, "field": "doc_title"})

        y = max(y, MARGIN + 80) + 14
        _draw_line(draw, y, color=ACCENT_COLOR, width=2)
        y += 12

        # Invoice metadata
        inv_no = f"INV-{self.rng.randint(1000, 9999)}"
        inv_date = self.fake.date_this_year().strftime("%d %b %Y")
        due_date = self.fake.date_this_year().strftime("%d %b %Y")
        for label, val, field in [("Invoice No:", inv_no, "invoice_no"),
                                   ("Invoice Date:", inv_date, "invoice_date"),
                                   ("Due Date:", due_date, "due_date")]:
            row = f"{label:<16}{val}"
            bbox = _text_bbox(draw, row, FONT_BODY, (MARGIN, y))
            draw.text((MARGIN, y), row, font=FONT_BODY, fill=TEXT_COLOR)
            text_regions.append({"bbox": bbox, "text": val, "field": field})
            y = bbox[3] + 4

        y += 10

        # Bill To
        bbox = _text_bbox(draw, "Bill To:", FONT_HEADER, (MARGIN, y))
        draw.text((MARGIN, y), "Bill To:", font=FONT_HEADER, fill=ACCENT_COLOR)
        y = bbox[3] + 6

        client = self.fake.name()
        client_company = self.fake.company()
        for val, field in [(client, "client_name"), (client_company, "client_company")]:
            bbox = _text_bbox(draw, val, FONT_BODY, (MARGIN, y))
            draw.text((MARGIN, y), val, font=FONT_BODY, fill=TEXT_COLOR)
            text_regions.append({"bbox": bbox, "text": val, "field": field})
            y = bbox[3] + 4

        y += 14
        # Table header
        _draw_rect(draw, [MARGIN, y, DOC_W - MARGIN, y + 22], fill=ACCENT_COLOR)
        headers = ["Description", "Qty", "Unit Price", "Amount"]
        col_xs = [MARGIN + 4, MARGIN + 310, MARGIN + 420, MARGIN + 530]
        for hdr, cx in zip(headers, col_xs):
            draw.text((cx, y + 4), hdr, font=FONT_BODY, fill=(255, 255, 255))
        y += 26

        # Line items
        total = 0.0
        n_items = self.rng.randint(3, 8)
        for i in range(n_items):
            desc = self.fake.bs().capitalize()[:40]
            qty = self.rng.randint(1, 20)
            unit_price = round(self.rng.uniform(10, 500), 2)
            amount = round(qty * unit_price, 2)
            total += amount
            bg = LIGHT_GRAY if i % 2 == 0 else BG_COLOR
            _draw_rect(draw, [MARGIN, y, DOC_W - MARGIN, y + 18], fill=bg)
            row_vals = [desc, str(qty), f"${unit_price:.2f}", f"${amount:.2f}"]
            for val, cx in zip(row_vals, col_xs):
                bbox = _text_bbox(draw, val, FONT_BODY, (cx, y + 2))
                draw.text((cx, y + 2), val, font=FONT_BODY, fill=TEXT_COLOR)
            text_regions.append({"bbox": [MARGIN, y, DOC_W - MARGIN, y + 18],
                                  "text": desc, "field": "line_item"})
            text_regions.append({"bbox": _text_bbox(draw, f"${amount:.2f}", FONT_BODY,
                                                     (col_xs[3], y + 2)),
                                  "text": f"${amount:.2f}", "field": "line_amount"})
            y += 20
            if y > DOC_H - 200:
                break

        y += 10
        tax = round(total * 0.15, 2)
        grand = round(total + tax, 2)
        for label, val, field in [("Subtotal:", f"${total:.2f}", "subtotal"),
                                   ("GST (15%):", f"${tax:.2f}", "tax"),
                                   ("Total Due:", f"${grand:.2f}", "total")]:
            font = FONT_HEADER if field == "total" else FONT_BODY
            x_label = DOC_W - MARGIN - 230
            bbox = _text_bbox(draw, label, font, (x_label, y))
            draw.text((x_label, y), label, font=font, fill=TEXT_COLOR)
            vbbox = _text_bbox(draw, val, font, (DOC_W - MARGIN - 80, y))
            draw.text((DOC_W - MARGIN - 80, y), val, font=font, fill=TEXT_COLOR)
            text_regions.append({"bbox": vbbox, "text": val, "field": field})
            y = max(bbox[3], vbbox[3]) + 4

        return {"image": img, "text_regions": text_regions,
                "doc_type": "invoice", "source_doc_id": doc_id}

    # ------------------------------------------------------------------
    # Form
    # ------------------------------------------------------------------

    def generate_form(self) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        img, draw = _new_doc()
        text_regions: List[Dict] = []
        y = MARGIN

        org = self.fake.company()
        bbox = _text_bbox(draw, org, FONT_HEADER, (MARGIN, y))
        draw.text((MARGIN, y), org, font=FONT_HEADER, fill=ACCENT_COLOR)
        text_regions.append({"bbox": bbox, "text": org, "field": "org_name"})
        y = bbox[3] + 8

        form_title = random.choice(["Application Form", "Registration Form",
                                    "Contact Form", "Feedback Form",
                                    "Enrollment Form"])
        bbox = _text_bbox(draw, form_title, FONT_TITLE, (MARGIN, y))
        draw.text((MARGIN, y), form_title, font=FONT_TITLE, fill=TEXT_COLOR)
        text_regions.append({"bbox": bbox, "text": form_title, "field": "form_title"})
        y = bbox[3] + 12

        _draw_line(draw, y, color=ACCENT_COLOR, width=2)
        y += 16

        fields = [
            ("Full Name", self.fake.name(), "full_name"),
            ("Date of Birth", self.fake.date_of_birth(minimum_age=18,
                                                       maximum_age=80).strftime("%d/%m/%Y"), "dob"),
            ("Email Address", self.fake.email(), "email"),
            ("Phone Number", self.fake.phone_number(), "phone"),
            ("Address", self.fake.street_address(), "address"),
            ("City", self.fake.city(), "city"),
            ("Postal Code", self.fake.postcode(), "postcode"),
            ("Country", self.fake.country(), "country"),
            ("Occupation", self.fake.job(), "occupation"),
            ("Employer", self.fake.company(), "employer"),
            ("ID Number", f"ID-{self.rng.randint(100000, 999999)}", "id_number"),
            ("Reference No", f"REF-{self.rng.randint(10000, 99999)}", "ref_no"),
        ]

        for label, value, field_name in fields:
            # Label
            lbbox = _text_bbox(draw, f"{label}:", FONT_BODY, (MARGIN, y))
            draw.text((MARGIN, y), f"{label}:", font=FONT_BODY, fill=ACCENT_COLOR)
            # Value
            vx = MARGIN + 200
            vbbox = _text_bbox(draw, value, FONT_BODY, (vx, y))
            draw.text((vx, y), value, font=FONT_BODY, fill=TEXT_COLOR)
            text_regions.append({"bbox": vbbox, "text": value, "field": field_name})
            # Underline
            line_y = max(lbbox[3], vbbox[3]) + 2
            _draw_line(draw, line_y, color=LIGHT_GRAY)
            y = line_y + 10
            if y > DOC_H - 80:
                break

        # Signature
        y += 10
        sig_label = "Signature: ____________________________"
        bbox = _text_bbox(draw, sig_label, FONT_BODY, (MARGIN, y))
        draw.text((MARGIN, y), sig_label, font=FONT_BODY, fill=TEXT_COLOR)
        y = bbox[3] + 8

        date_label = f"Date: {self.fake.date_this_year().strftime('%d/%m/%Y')}"
        bbox = _text_bbox(draw, date_label, FONT_BODY, (MARGIN, y))
        draw.text((MARGIN, y), date_label, font=FONT_BODY, fill=TEXT_COLOR)
        text_regions.append({"bbox": bbox, "text": date_label, "field": "form_date"})

        return {"image": img, "text_regions": text_regions,
                "doc_type": "form", "source_doc_id": doc_id}

    # ------------------------------------------------------------------
    # Certificate
    # ------------------------------------------------------------------

    def generate_certificate(self) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        img, draw = _new_doc()
        text_regions: List[Dict] = []

        # Decorative border
        border_margin = 20
        draw.rectangle([border_margin, border_margin,
                         DOC_W - border_margin, DOC_H - border_margin],
                        outline=ACCENT_COLOR, width=4)
        draw.rectangle([border_margin + 6, border_margin + 6,
                         DOC_W - border_margin - 6, DOC_H - border_margin - 6],
                        outline=LIGHT_GRAY, width=1)

        y = MARGIN + 30
        cert_type = random.choice(["Certificate of Achievement",
                                    "Certificate of Completion",
                                    "Certificate of Excellence",
                                    "Certificate of Participation",
                                    "Certificate of Merit"])
        cx = DOC_W // 2
        bbox = _text_bbox(draw, cert_type, FONT_TITLE, (MARGIN, y))
        title_x = cx - (bbox[2] - bbox[0]) // 2
        bbox = _text_bbox(draw, cert_type, FONT_TITLE, (title_x, y))
        draw.text((title_x, y), cert_type, font=FONT_TITLE, fill=ACCENT_COLOR)
        text_regions.append({"bbox": bbox, "text": cert_type, "field": "cert_type"})
        y = bbox[3] + 20

        # "This is to certify that"
        sub = "This is to certify that"
        sbbox = _text_bbox(draw, sub, FONT_BODY, (MARGIN, y))
        sub_x = cx - (sbbox[2] - sbbox[0]) // 2
        draw.text((sub_x, y), sub, font=FONT_BODY, fill=TEXT_COLOR)
        y = sbbox[3] + 14

        # Recipient name (large)
        recipient = self.fake.name()
        rbbox = _text_bbox(draw, recipient, FONT_TITLE, (MARGIN, y))
        rec_x = cx - (rbbox[2] - rbbox[0]) // 2
        rbbox = _text_bbox(draw, recipient, FONT_TITLE, (rec_x, y))
        draw.text((rec_x, y), recipient, font=FONT_TITLE, fill=TEXT_COLOR)
        text_regions.append({"bbox": rbbox, "text": recipient, "field": "recipient_name"})
        y = rbbox[3] + 16

        # Achievement description
        achievement = (f"has successfully completed the course in "
                       f"{self.fake.bs().title()} with distinction.")
        # Wrap text manually
        words = achievement.split()
        line, lines = [], []
        for word in words:
            test = " ".join(line + [word])
            tbbox = _text_bbox(draw, test, FONT_BODY, (0, 0))
            if tbbox[2] - tbbox[0] > DOC_W - 2 * MARGIN - 20:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))

        for ln in lines:
            lbbox = _text_bbox(draw, ln, FONT_BODY, (MARGIN, y))
            ln_x = cx - (lbbox[2] - lbbox[0]) // 2
            lbbox = _text_bbox(draw, ln, FONT_BODY, (ln_x, y))
            draw.text((ln_x, y), ln, font=FONT_BODY, fill=TEXT_COLOR)
            y = lbbox[3] + 6

        text_regions.append({"bbox": [MARGIN, y - len(lines) * 20, DOC_W - MARGIN, y],
                              "text": achievement, "field": "achievement_desc"})
        y += 20

        # Date and organization
        cert_date = self.fake.date_this_year().strftime("%d %B %Y")
        org = self.fake.company()
        for val, field in [(cert_date, "cert_date"), (org, "issuing_org")]:
            vbbox = _text_bbox(draw, val, FONT_BODY, (MARGIN, y))
            v_x = cx - (vbbox[2] - vbbox[0]) // 2
            vbbox = _text_bbox(draw, val, FONT_BODY, (v_x, y))
            draw.text((v_x, y), val, font=FONT_BODY, fill=TEXT_COLOR)
            text_regions.append({"bbox": vbbox, "text": val, "field": field})
            y = vbbox[3] + 8

        y += 20
        # Signature lines
        sig_y = DOC_H - 160
        for label, x_start in [("Authorized Signatory", MARGIN + 20),
                                 ("Director", DOC_W - MARGIN - 220)]:
            draw.line([(x_start, sig_y), (x_start + 180, sig_y)],
                       fill=TEXT_COLOR, width=1)
            lbbox = _text_bbox(draw, label, FONT_SMALL, (x_start, sig_y + 4))
            draw.text((x_start, sig_y + 4), label, font=FONT_SMALL, fill=TEXT_COLOR)

        # Certificate ID at bottom
        cert_id = f"Cert ID: CERT-{self.rng.randint(100000, 999999)}"
        cbbox = _text_bbox(draw, cert_id, FONT_SMALL, (MARGIN, DOC_H - 50))
        draw.text((MARGIN, DOC_H - 50), cert_id, font=FONT_SMALL, fill=LIGHT_GRAY)
        text_regions.append({"bbox": cbbox, "text": cert_id, "field": "cert_id"})

        return {"image": img, "text_regions": text_regions,
                "doc_type": "certificate", "source_doc_id": doc_id}

    # ------------------------------------------------------------------
    # Marksheet
    # ------------------------------------------------------------------

    def generate_marksheet(self) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        img, draw = _new_doc()
        text_regions: List[Dict] = []
        y = MARGIN

        institution = self.fake.company() + " University"
        bbox = _text_bbox(draw, institution, FONT_HEADER, (MARGIN, y))
        draw.text((MARGIN, y), institution, font=FONT_HEADER, fill=ACCENT_COLOR)
        text_regions.append({"bbox": bbox, "text": institution, "field": "institution"})
        y = bbox[3] + 6

        title = "ACADEMIC MARKSHEET / TRANSCRIPT"
        bbox = _text_bbox(draw, title, FONT_TITLE, (MARGIN, y))
        draw.text((MARGIN, y), title, font=FONT_TITLE, fill=TEXT_COLOR)
        text_regions.append({"bbox": bbox, "text": title, "field": "doc_title"})
        y = bbox[3] + 10

        _draw_line(draw, y, color=ACCENT_COLOR, width=2)
        y += 12

        student_name = self.fake.name()
        roll_no = f"ROLL-{self.rng.randint(10000, 99999)}"
        program = random.choice(["B.Sc. Computer Science", "B.Com Accounting",
                                  "B.A. Economics", "B.Eng. Electrical",
                                  "M.Sc. Data Science"])
        semester = f"Semester {self.rng.randint(1, 8)}, {self.rng.randint(2020, 2026)}"

        for label, val, field in [("Student Name:", student_name, "student_name"),
                                   ("Roll Number:", roll_no, "roll_no"),
                                   ("Program:", program, "program"),
                                   ("Semester:", semester, "semester")]:
            row = f"{label:<20}{val}"
            bbox = _text_bbox(draw, row, FONT_BODY, (MARGIN, y))
            draw.text((MARGIN, y), row, font=FONT_BODY, fill=TEXT_COLOR)
            text_regions.append({"bbox": bbox, "text": val, "field": field})
            y = bbox[3] + 4

        y += 12
        # Table header
        _draw_rect(draw, [MARGIN, y, DOC_W - MARGIN, y + 22], fill=ACCENT_COLOR)
        col_headers = ["Subject", "Code", "Max Marks", "Marks Obtained", "Grade"]
        col_xs = [MARGIN + 4, MARGIN + 240, MARGIN + 320, MARGIN + 420, MARGIN + 530]
        for hdr, cx in zip(col_headers, col_xs):
            draw.text((cx, y + 4), hdr, font=FONT_SMALL, fill=(255, 255, 255))
        y += 24

        subjects = [
            "Mathematics", "Physics", "Chemistry", "English",
            "Computer Science", "Statistics", "Biology", "History",
            "Economics", "Geography",
        ]
        chosen = self.rng.sample(subjects, min(self.rng.randint(5, 8), len(subjects)))
        total_obtained = 0
        total_max = 0

        grades = ["A+", "A", "B+", "B", "C+", "C", "D", "F"]
        grade_thresholds = [90, 80, 70, 60, 50, 45, 35, 0]

        for i, subject in enumerate(chosen):
            code = f"{subject[:3].upper()}{self.rng.randint(100, 499)}"
            max_marks = 100
            obtained = self.rng.randint(35, 100)
            grade = grades[next((j for j, t in enumerate(grade_thresholds) if obtained >= t), len(grades) - 1)]
            total_max += max_marks
            total_obtained += obtained

            bg = LIGHT_GRAY if i % 2 == 0 else BG_COLOR
            _draw_rect(draw, [MARGIN, y, DOC_W - MARGIN, y + 18], fill=bg)
            row_data = [subject, code, str(max_marks), str(obtained), grade]
            for val, cx in zip(row_data, col_xs):
                draw.text((cx, y + 2), val, font=FONT_SMALL, fill=TEXT_COLOR)
            marks_bbox = _text_bbox(draw, str(obtained), FONT_SMALL, (col_xs[3], y + 2))
            text_regions.append({"bbox": marks_bbox, "text": str(obtained),
                                  "field": "marks_obtained"})
            y += 20
            if y > DOC_H - 180:
                break

        # Total row
        _draw_line(draw, y)
        y += 6
        percentage = round(total_obtained / total_max * 100, 2) if total_max else 0
        overall_grade = grades[next((j for j, t in enumerate(grade_thresholds)
                                      if percentage >= t), len(grades) - 1)]
        total_row = f"TOTAL: {total_obtained}/{total_max}  |  Percentage: {percentage}%  |  Overall Grade: {overall_grade}"
        bbox = _text_bbox(draw, total_row, FONT_BODY, (MARGIN, y))
        draw.text((MARGIN, y), total_row, font=FONT_BODY, fill=ACCENT_COLOR)
        text_regions.append({"bbox": bbox, "text": total_row, "field": "total_summary"})
        y = bbox[3] + 14

        # Declaration
        decl = f"Certified that this is the official marksheet issued by {institution}."
        dbbox = _text_bbox(draw, decl, FONT_SMALL, (MARGIN, y))
        draw.text((MARGIN, y), decl, font=FONT_SMALL, fill=TEXT_COLOR)
        text_regions.append({"bbox": dbbox, "text": decl, "field": "declaration"})

        return {"image": img, "text_regions": text_regions,
                "doc_type": "marksheet", "source_doc_id": doc_id}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def generate(self, doc_type: Optional[str] = None) -> Dict[str, Any]:
        if doc_type is None:
            doc_type = self.rng.choice(DOC_TYPES)
        generators = {
            "receipt":     self.generate_receipt,
            "invoice":     self.generate_invoice,
            "form":        self.generate_form,
            "certificate": self.generate_certificate,
            "marksheet":   self.generate_marksheet,
        }
        return generators[doc_type]()


# ---------------------------------------------------------------------------
# JPEG recompression
# ---------------------------------------------------------------------------

def apply_jpeg_recompression(pil_img: Image.Image, q1: int = 90,
                              q2: int = 75) -> Image.Image:
    """Apply double JPEG compression to simulate editing artefacts."""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=q1)
    buf.seek(0)
    img1 = Image.open(buf).copy()

    buf2 = io.BytesIO()
    img1.save(buf2, format="JPEG", quality=q2)
    buf2.seek(0)
    img2 = Image.open(buf2).copy()
    return img2


# ---------------------------------------------------------------------------
# TamperingEngine
# ---------------------------------------------------------------------------

class TamperingEngine:
    """Applies one of 5 tampering operations to a document dict."""

    def _blank_mask(self, size: Tuple[int, int]) -> Image.Image:
        return Image.new("L", size, 0)

    def _mark_mask(self, mask: Image.Image,
                   bbox: List[int], value: int = 255) -> None:
        """Fill a rectangular region of the mask with value."""
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(mask.width, x2); y2 = min(mask.height, y2)
        if x2 > x1 and y2 > y1:
            region = Image.new("L", (x2 - x1, y2 - y1), value)
            mask.paste(region, (x1, y1))

    # ------------------------------------------------------------------

    def text_replacement(self, doc: Dict[str, Any],
                          rng: random.Random) -> Dict[str, Any]:
        """Replace the text in a random text region with different fake text."""
        regions = doc["text_regions"]
        if not regions:
            return self._fallback(doc, rng)

        img = doc["image"].copy()
        mask = self._blank_mask(img.size)
        draw = ImageDraw.Draw(img)
        fake = Faker()

        region = rng.choice(regions)
        bbox = region["bbox"]
        x1, y1, x2, y2 = bbox

        # Erase old text
        draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], fill=BG_COLOR)

        # Generate replacement text of similar length
        new_text = fake.bothify("??" * max(1, len(region["text"]) // 4))[:len(region["text"])]
        field = region.get("field", "")
        if "amount" in field or "total" in field or "price" in field:
            new_text = f"${rng.uniform(10, 9999):.2f}"
        elif "date" in field:
            new_text = fake.date_this_decade().strftime("%d/%m/%Y")
        elif "name" in field:
            new_text = fake.name()
        elif "no" in field or "id" in field:
            new_text = f"{rng.randint(10000, 99999)}"
        else:
            new_text = fake.word().capitalize()

        draw.text((x1, y1), new_text, font=FONT_BODY, fill=TEXT_COLOR)
        self._mark_mask(mask, [x1 - 2, y1 - 2, x2 + 2, y2 + 2])

        img = apply_jpeg_recompression(img)
        return {"image": img, "mask": mask, "tamper_type": "text_replacement",
                "tamper_bbox": bbox, "is_forged": True}

    # ------------------------------------------------------------------

    def copy_move(self, doc: Dict[str, Any],
                   rng: random.Random) -> Dict[str, Any]:
        """Copy a region and paste it to a different location."""
        img = doc["image"].copy()
        mask = self._blank_mask(img.size)
        w, h = img.size

        # Pick random source region
        rw = rng.randint(60, 200)
        rh = rng.randint(20, 60)
        sx = rng.randint(0, max(1, w - rw - 1))
        sy = rng.randint(0, max(1, h - rh - 1))
        src_bbox = (sx, sy, sx + rw, sy + rh)

        # Pick destination (different enough)
        for _ in range(20):
            dx = rng.randint(0, max(1, w - rw - 1))
            dy = rng.randint(0, max(1, h - rh - 1))
            if abs(dx - sx) > 40 or abs(dy - sy) > 40:
                break

        patch = img.crop(src_bbox)
        img.paste(patch, (dx, dy))
        dst_bbox = [dx, dy, dx + rw, dy + rh]
        self._mark_mask(mask, dst_bbox)

        img = apply_jpeg_recompression(img)
        return {"image": img, "mask": mask, "tamper_type": "copy_move",
                "tamper_bbox": dst_bbox, "is_forged": True}

    # ------------------------------------------------------------------

    def text_insertion(self, doc: Dict[str, Any],
                        rng: random.Random) -> Dict[str, Any]:
        """Insert new fake text into a blank area."""
        img = doc["image"].copy()
        mask = self._blank_mask(img.size)
        draw = ImageDraw.Draw(img)
        fake = Faker()
        w, h = img.size

        new_text = fake.sentence(nb_words=rng.randint(3, 8))
        x = rng.randint(MARGIN, w - MARGIN - 200)
        y = rng.randint(MARGIN, h - MARGIN - 30)

        tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = _text_bbox(tmp_draw, new_text, FONT_BODY, (x, y))

        draw.text((x, y), new_text, font=FONT_BODY, fill=TEXT_COLOR)
        self._mark_mask(mask, bbox)

        img = apply_jpeg_recompression(img)
        return {"image": img, "mask": mask, "tamper_type": "text_insertion",
                "tamper_bbox": bbox, "is_forged": True}

    # ------------------------------------------------------------------

    def text_removal(self, doc: Dict[str, Any],
                      rng: random.Random) -> Dict[str, Any]:
        """Erase text in a region by painting over it with white."""
        regions = doc["text_regions"]
        if not regions:
            return self._fallback(doc, rng)

        img = doc["image"].copy()
        mask = self._blank_mask(img.size)
        draw = ImageDraw.Draw(img)

        region = rng.choice(regions)
        bbox = region["bbox"]
        x1, y1, x2, y2 = bbox
        pad = 4
        erased_bbox = [x1 - pad, y1 - pad, x2 + pad, y2 + pad]
        draw.rectangle(erased_bbox, fill=BG_COLOR)
        self._mark_mask(mask, erased_bbox)

        img = apply_jpeg_recompression(img)
        return {"image": img, "mask": mask, "tamper_type": "text_removal",
                "tamper_bbox": erased_bbox, "is_forged": True}

    # ------------------------------------------------------------------

    def splicing(self, doc: Dict[str, Any], doc2: Dict[str, Any],
                  rng: random.Random) -> Dict[str, Any]:
        """Paste a text region from doc2 into doc."""
        regions2 = doc2["text_regions"]
        if not regions2:
            return self._fallback(doc, rng)

        img = doc["image"].copy()
        mask = self._blank_mask(img.size)
        w, h = img.size

        region2 = rng.choice(regions2)
        x1, y1, x2, y2 = region2["bbox"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(doc2["image"].width, x2), min(doc2["image"].height, y2)
        if x2 <= x1 or y2 <= y1:
            return self._fallback(doc, rng)

        patch = doc2["image"].crop((x1, y1, x2, y2))
        pw, ph = patch.size

        # Paste into doc at a random position
        dx = rng.randint(MARGIN, max(MARGIN + 1, w - pw - MARGIN))
        dy = rng.randint(MARGIN, max(MARGIN + 1, h - ph - MARGIN))
        img.paste(patch, (dx, dy))
        dst_bbox = [dx, dy, dx + pw, dy + ph]
        self._mark_mask(mask, dst_bbox)

        img = apply_jpeg_recompression(img)
        return {"image": img, "mask": mask, "tamper_type": "splicing",
                "tamper_bbox": dst_bbox, "is_forged": True}

    # ------------------------------------------------------------------

    def _fallback(self, doc: Dict[str, Any],
                   rng: random.Random) -> Dict[str, Any]:
        """Fallback when a tampering op can't proceed: use copy_move."""
        return self.copy_move(doc, rng)

    # ------------------------------------------------------------------

    def apply(self, doc: Dict[str, Any], tamper_type: str,
               rng: random.Random,
               doc2: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatch to the appropriate tampering method."""
        if tamper_type == "text_replacement":
            return self.text_replacement(doc, rng)
        elif tamper_type == "copy_move":
            return self.copy_move(doc, rng)
        elif tamper_type == "text_insertion":
            return self.text_insertion(doc, rng)
        elif tamper_type == "text_removal":
            return self.text_removal(doc, rng)
        elif tamper_type == "splicing":
            if doc2 is None:
                return self._fallback(doc, rng)
            return self.splicing(doc, doc2, rng)
        else:
            raise ValueError(f"Unknown tamper_type: {tamper_type!r}")


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

def generate_dataset(output_dir: Path, n_samples: int,
                     config: Any, seed: int = 42) -> None:
    """
    Generate a synthetic document forgery dataset.

    Args:
        output_dir: Directory in which images, masks, and metadata are saved.
        n_samples:  Total number of samples to generate.
        config:     OmegaConf / dict-like config (synthetic_gen.yaml).
        seed:       Global random seed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np.random.seed(seed)

    # Parse config
    try:
        cfg = config.synthetic if hasattr(config, "synthetic") else config
        holdout = cfg.get("holdout_tamper_type", "splicing")
        jpeg_enabled = cfg.get("jpeg_recompression", {}).get("enabled", True)
        doc_types = list(cfg.get("doc_types", DOC_TYPES))
    except Exception:
        holdout = "splicing"
        jpeg_enabled = True
        doc_types = DOC_TYPES

    # Split: 70% authentic, 30% forged
    n_forged = max(1, int(n_samples * 0.50))
    n_authentic = n_samples - n_forged

    doc_gen = DocumentGenerator(seed=seed)
    tamper_eng = TamperingEngine()

    # Progress
    try:
        from tqdm import tqdm
        progress = tqdm(total=n_samples, desc="Generating dataset", unit="sample")
        def _tick():
            progress.update(1)
        def _close():
            progress.close()
    except ImportError:
        _counter = [0]
        def _tick():
            _counter[0] += 1
            if _counter[0] % 50 == 0:
                print(f"  Generated {_counter[0]}/{n_samples} samples …")
        def _close():
            pass

    samples_meta: List[Dict] = []

    # -------------------------------------------------------------------
    # Generate authentic documents
    # -------------------------------------------------------------------
    for i in range(n_authentic):
        sample_id = f"auth_{i:06d}"
        doc_type = rng.choice(doc_types)
        doc = doc_gen.generate(doc_type)

        img = doc["image"]
        mask = Image.new("L", img.size, 0)

        img.save(output_dir / f"{sample_id}.png")
        mask.save(output_dir / f"{sample_id}_mask.png")

        meta: Dict[str, Any] = {
            "sample_id": sample_id,
            "source_doc_id": doc["source_doc_id"],
            "doc_type": doc["doc_type"],
            "is_forged": False,
            "tamper_type": None,
            "tamper_bbox": None,
            "split": None,          # assigned later
        }
        with open(output_dir / f"{sample_id}.json", "w") as f:
            json.dump(meta, f, indent=2)

        samples_meta.append(meta)
        _tick()

    # -------------------------------------------------------------------
    # Generate forged documents
    # -------------------------------------------------------------------
    # Adjusted tamper weights to make splicing less frequent
    tamper_types_local = list(TAMPER_TYPES)
    weights_local = list(TAMPER_WEIGHTS)

    # If there's a configured holdout, reduce its weight
    if holdout in tamper_types_local:
        idx = tamper_types_local.index(holdout)
        weights_local[idx] *= 0.4
    total_w = sum(weights_local)
    weights_local = [w / total_w for w in weights_local]

    for i in range(n_forged):
        sample_id = f"forged_{i:06d}"
        doc_type = rng.choice(doc_types)
        doc = doc_gen.generate(doc_type)

        # Sample tamper type
        tamper_type = rng.choices(tamper_types_local, weights=weights_local, k=1)[0]

        doc2 = None
        if tamper_type == "splicing":
            doc2_type = rng.choice(doc_types)
            doc2 = doc_gen.generate(doc2_type)

        result = tamper_eng.apply(doc, tamper_type, rng, doc2=doc2)

        img = result["image"]
        mask = result["mask"]

        img.save(output_dir / f"{sample_id}.png")
        mask.save(output_dir / f"{sample_id}_mask.png")

        meta = {
            "sample_id": sample_id,
            "source_doc_id": doc["source_doc_id"],
            "doc_type": doc["doc_type"],
            "is_forged": True,
            "tamper_type": result["tamper_type"],
            "tamper_bbox": result["tamper_bbox"],
            "split": None,
        }
        with open(output_dir / f"{sample_id}.json", "w") as f:
            json.dump(meta, f, indent=2)

        samples_meta.append(meta)
        _tick()

    _close()

    # -------------------------------------------------------------------
    # Assign splits by grouping source_doc_id
    # -------------------------------------------------------------------
    from collections import defaultdict

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples_meta:
        groups[s["source_doc_id"]].append(s)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = max(1, int(n_groups * 0.70))
    n_val   = max(1, int(n_groups * 0.15))
    # rest → test

    train_ids = set(group_ids[:n_train])
    val_ids   = set(group_ids[n_train: n_train + n_val])
    # test: everything else

    for s in samples_meta:
        gid = s["source_doc_id"]
        if gid in train_ids:
            s["split"] = "train"
        elif gid in val_ids:
            s["split"] = "val"
        else:
            s["split"] = "test"

        # Update the JSON on disk with the assigned split
        json_path = output_dir / f"{s['sample_id']}.json"
        with open(json_path, "w") as f:
            json.dump(s, f, indent=2)

    # -------------------------------------------------------------------
    # Write manifest
    # -------------------------------------------------------------------
    manifest = {
        "n_samples": len(samples_meta),
        "n_authentic": n_authentic,
        "n_forged": n_forged,
        "splits": {"train": 0.70, "val": 0.15, "test": 0.15},
        "samples": samples_meta,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Dataset written to {output_dir}  "
          f"({n_authentic} authentic, {n_forged} forged)")
    split_counts = {"train": 0, "val": 0, "test": 0}
    for s in samples_meta:
        split_counts[s["split"]] += 1
    print(f"  Splits -> train: {split_counts['train']}, "
          f"val: {split_counts['val']}, test: {split_counts['test']}")
