"""Shared helpers for building the simple notebook series."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = """from google.colab import drive
drive.mount('/content/drive')
folder = '/content/drive/MyDrive/DataSets/ThisTimeIsDifferent/'

import warnings
warnings.filterwarnings('ignore')     # hide warning messages"""

FOOTER = ("*Simple notebook series of the project 'This Time Is Different?'. Prepared with AI assistance "
          "(declared in `notes/ai_use.md`); a study notebook, not dissertation text.*")


class NB:
    def __init__(self, name, title, intro):
        self.name = name
        self.cells = [nbf.v4.new_markdown_cell(f"# {title}\n{intro.strip()}\n\n{FOOTER}")]

    def md(self, text):
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(self, text):
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n")))

    def save(self):
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"] = {"colab": {"provenance": []},
                          "kernelspec": {"name": "python3", "display_name": "Python 3"},
                          "language_info": {"name": "python"}}
        nbf.write(nb, HERE / f"{self.name}.ipynb")
        print("wrote", self.name, len(self.cells), "cells")
