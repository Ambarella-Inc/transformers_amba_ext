# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../../"))

readme_path = Path(__file__).parent.resolve().parent.parent / "README.md"
readme_target = Path(__file__).parent / "readme.md"

with readme_target.open("w") as outf:
    outf.write(
        "\n".join(
            [
                "======",
                "",
            ]
        )
    )
    lines = []
    for line in readme_path.read_text().split("\n"):
        lines.append(line)
    outf.write("\n".join(lines))


project = 'transformers_amba_ext'
copyright = '2025, Ambarella'
author = 'Ambarella'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "recommonmark",
    "sphinx_markdown_tables",
]

templates_path = ['_templates']
exclude_patterns = []

source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}
source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'alabaster'

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = ["_themes"]

try:
    import importlib

    theme = importlib.import_module("sphinx_book_theme")
    html_theme = "sphinx_book_theme"
    html_theme_path = [theme.get_html_theme_path()]
except ImportError:
    print(
        "**** WARNING ****: reverting to default theme, because "
        "sphinx_book_theme is not installed"
    )
    html_theme = "default"
print(f"html_theme='{html_theme}'")

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

if html_theme == "sphinx_rtd_theme":
    html_theme_options = {"logo_only": True}
else:
    html_theme_options = {}
print(f"html_theme_options={html_theme_options}")

