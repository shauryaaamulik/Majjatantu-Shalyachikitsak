"""Torch-free synthetic testbed with planted, known-ground-truth defects.

Purpose: exercise the whole discover -> localize -> repair -> verify loop on a
model small enough to inspect completely, so harness bugs are separated from
research questions. A success here shows the *harness* works; it says nothing
yet about real language models (README §14).

Reuses the shared records in :mod:`majjatantu.types` and :mod:`majjatantu.reporting`.
"""
