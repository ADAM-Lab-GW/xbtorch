# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../src/xbtorch/"))

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Optional, creates summary tables
]

# Optional: autosummary automatically generates summary pages
autosummary_generate = True
autodoc_default_options = {
    "members": True,            # include all class members
    "undoc-members": True,      # include undocumented members
    "show-inheritance": True,   # show base classes
    "noindex": True,  # prevents duplicate indexing
    "no-signatures": True,          # hide function/method signatures
}

modindex_common_prefix = ['xbtorch.'] # drop prefix in module index 

html_theme = "sphinx_rtd_theme"

project = 'XBTorch'
copyright = '2025, George Washington University'
author = 'Osama Yousuf'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

