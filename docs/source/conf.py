# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyautoencoder'
copyright = '2025, Andrea Pollastro'
author = 'Andrea Pollastro'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # for NumPy/Google docstrings
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.mathjax",       # ← add this
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "../../assets/logo_nobackground.png"
html_static_path = ['_static']

html_theme_options = {
    "logo": {
        "text": "PyAutoencoder",
    },
    "navigation_depth": 3,
    "show_prev_next": False,
}

html_sidebars = {
    "index": [],
    "getting_started": [],
    "architecture": [],
}

import os
import sys

# Add the project root (the folder that contains "pyautoencoder/") to sys.path
sys.path.insert(0, os.path.abspath("../.."))