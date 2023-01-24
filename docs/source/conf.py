# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from typing import Dict, List

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../"))
# -- Project information -----------------------------------------------------

project = "UserSimCRS"
copyright = "2022, IAI group, University of Stavanger"
author = "Jafar Afzali, Krisztian Balog, Aleksander Drzewiecki \
        and Shuo Zhang"

# The short X.Y version.
version = "0.0.1"
# The full version, including alpha/beta/rc tags.
release = "0.0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "autoapi.extension",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_sidebars = {
    "**": [
        "versions.html",
    ],
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ["_static"]
html_theme_options: Dict[str, str] = {}
html_favicon = "_static/favicon.png"

# Auto api
autoapi_type = "python"
autoapi_dirs = ["../../usersimcrs"]
autoapi_ignore = ["*tests/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "init"

# -- Options for versioning -------------------------------------------------
# See documentation at:
# https://holzhaus.github.io/sphinx-multiversion/master/configuration.html#
smv_tag_whitelist = r"^.*$"  # Include all tags
smv_branch_whitelist = r"^main$"  # Include only main branch
smv_remote_whitelist = (
    r"^(origin|upstream)$"  # Use branches from origin and upstream
)
smv_released_pattern = r"^tags/.*$"  # Tags only
smv_outputdir_format = "{ref.name}"  # Use the branch/tag name

# Determines whether remote or local git branches/tags are preferred if their
# output dirs conflict
smv_prefer_remote_refs = True
