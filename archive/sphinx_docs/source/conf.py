# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'The aMAZEing maze'
copyright = '2024, Andre Maia Chagas, Alejandra Carriero, Shahd Al Balushi, Moira Eley, Miguel Maravall'
author = 'Andre Maia Chagas, Alejandra Carriero, Shahd Al Balushi, Moira Eley,  Miguel Maravall'
#release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration', 
    "myst_parser",
    "sphinx_design"]

myst_enable_extensions = [
    "colon_fence",
    "linkify",
    "substitution",
    "html_admonition",
    "html_image",
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"


# 'pydata_sphinx_theme'
#"sphinx_rtd_theme"
#'furo'
html_static_path = ['_static']
