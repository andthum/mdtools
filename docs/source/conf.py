# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options.
# For a full list see the documentation:
# http://www.sphinx-doc.org/en/master/config


# Standard libraries
import os
import sys


# -- Path setup --------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here.  If the directory
# is relative to the  documentation root, use os.path.abspath to make it
# absolute, like shown here.

# Either install MDTools before building the docs or add MDTools to
# sys.path.
module_path = os.path.abspath("./../..")
sys.path.insert(0, module_path)
try:
    import mdtools as mdt
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Before building the docs, install MDTools or add the the"
        " MDTools directory to sys.path in Sphinx's configuration file"
    )

# Recursively import all directories containing scripts.
script_path = os.path.abspath("./../../scripts")
for script_dir in os.walk(script_path):
    sys.path.insert(0, script_dir[0])


# -- Project information -----------------------------------------------

project = mdt.__title__
authors = mdt.__author__
copyright = mdt.__copyright__[14:]

# The short X.Y version
version = mdt.__version__
# The full version, including alpha/beta/rc tags.  If you do not need
# the separation provided between version and release, just set them
# both to the same value.
release = mdt.__version__


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.1'

# The master toctree document.
master_doc = 'index'

# The suffix(es) of source filenames.
source_suffix = {'.rst': 'restructuredtext'}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.  This pattern
# also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the default domain.
primary_domain = 'py'

# The name of a reStructuredText role (builtin or Sphinx extension) to
# use as the default role, that is, for text marked up `like this`.
# default_role = None

# The default language to highlight source code in.
highlight_language = 'python3'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# Add any Sphinx extension module names here, as strings.  They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Include documentation from docstrings
    # 'sphinx.ext.autosectionlabel',  # Allow reference sections using its title
    'sphinx.ext.autosummary',  # Generate autodoc summaries
    'sphinx.ext.coverage',  # Collect doc coverage stats
    'sphinx.ext.doctest',  # Test snippets in the documentation
    'sphinx.ext.duration',  # Measure durations of Sphinx processing
    # 'sphinx.ext.githubpages',  # Publish HTML docs in GitHub Pages
    # 'sphinx.ext.inheritance_diagram',  # Include inheritance diagrams
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.mathjax',  # Render math via JavaScript for HTML output
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.todo',  # Support for todo items
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx_rtd_theme',  # Read the Docs Sphinx Theme
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# A string of reStructuredText that will be included at the beginning of
# every source file that is read.  This is a possible place to add
# substitutions that should be available in every file
rst_prolog = """
.. role:: bash(code)
    :language: bash
.. role:: raw-html(raw)
    :format: html

.. |GNU_GPLv3| replace::
    :raw-html:`<a href="https://www.gnu.org/licenses/gpl-3.0.html">GNU General Public License</a>`
.. |GCC| replace::
    :raw-html:`<a href="https://gcc.gnu.org/">GCC</a>`
.. |Lmod| replace::
    :raw-html:`<a href="https://lmod.readthedocs.io/en/latest/">Lmod</a>`
.. |Git| replace::
    :raw-html:`<a href="https://git-scm.com/">Git</a>`
.. |GitHub| replace::
    :raw-html:`<a href="https://github.com/">GitHub</a>`
.. |Gromacs| replace::
    :raw-html:`<a href="https://manual.gromacs.org/">Gromacs</a>`

.. |Python| replace::
    :raw-html:`<a href="https://www.python.org/">Python</a>`
.. |PyPI| replace::
    :raw-html:`<a href="https://pypi.org/">PyPI</a>`
.. |pip| replace::
    :raw-html:`<a href="https://pip.pypa.io/en/stable/">pip</a>`
.. |virtual_Python_environment| replace::
    :raw-html:`<a href="https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/">virtual Python environment</a>`
.. |venv| replace::
    :raw-html:`<a href="https://docs.python.org/3/library/venv.html">venv</a>`
.. |Virtualenv| replace::
    :raw-html:`<a href="https://virtualenv.pypa.io/en/latest/">Virtualenv</a>`
.. |pytest| replace::
    :raw-html:`<a href="https://docs.pytest.org/en/stable/">pytest</a>`
.. |NumPy| replace::
    :raw-html:`<a href="https://numpy.org/">NumPy</a>`
.. |SciPy| replace::
    :raw-html:`<a href="https://www.scipy.org/">SciPy</a>`
.. |matplotlib| replace::
    :raw-html:`<a href="https://matplotlib.org/">matplotlib</a>`
.. |MDAnalysis| replace::
    :raw-html:`<a href="https://www.mdanalysis.org/">MDAnalysis</a>`
.. |mda_user_guide| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/stable/index.html">MDAnalysis user guide</a>`
.. |PyEMMA| replace::
    :raw-html:`<a href="http://emma-project.org/latest/">PyEMMA</a>`

.. |RTD| replace::
    :raw-html:`<a href="https://readthedocs.org/">Read the Docs</a>`
.. |Sphinx| replace::
    :raw-html:`<a href="https://www.sphinx-doc.org">Sphinx</a>`
.. |RST| replace::
    :raw-html:`<a href="https://docutils.sourceforge.io/rst.html">reStructuredText</a>`
.. |rst_option_list| replace::
    :raw-html:`<a href="https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#option-lists">reStructuredText option list</a>`
.. |NumPy_docstring_convention| replace::
    :raw-html:`<a href="https://numpydoc.readthedocs.io/en/latest/format.html">NumPy docstring convention</a>`

.. |mda_trj| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/stable/trajectories/trajectories.html">MDAnalysis trajectory</a>`
.. |mda_trjs| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/stable/trajectories/trajectories.html">MDAnalysis trajectories</a>`
.. |trajectory_and_topology_formats| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/stable/formats/index.html">trajectory and topology formats</a>`
.. |supported_coordinate_formats| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/stable/formats/index.html#coordinates">supported coordinate formats</a>`
.. |supported_topology_formats| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/stable/formats/index.html#topology">supported topology formats</a>`
.. |selection_syntax| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/stable/selections.html">selection syntax</a>`
.. |explanation_of_these_terms| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/stable/groups_of_atoms.html">explanation of these terms</a>`
"""


# -- Options for internationalization ----------------------------------

# The language for content autogenerated by Sphinx.  Refer to
# documentation for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'


# -- Options for HTML output -------------------------------------------

# The "theme" that the HTML output should use.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see
# the documentation.
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# If given, this must be the name of an image file (path relative to the
# configuration directory) that is the logo of the docs.   It is placed
# at the top of the sidebar; its width should therefore not exceed 200
# pixels.
html_logo = "../logo/mdtools_logo_192x135.png"

# If given, this must be the name of an image file (path relative to the
# configuration directory) that is the favicon of the docs.   It should
# be a Windows-style icon file (.ico), which is 16x16 or 32x32 pixels
# large.
html_favicon = "../logo/mdtools_favicon_32x32.png"

# Add any paths that contain custom static files (such as style sheets)
# here, relative to this directory.  They are copied after the builtin
# static files, so a file named "default.css" will overwrite the builtin
# "default.css".
html_static_path = ['_static']


def setup(app):
    """
    Limit the line length of tables to avoid horizontal scrollbars.
    
    See
    https://stackoverflow.com/questions/40641252/how-can-i-avoid-the-horizontal-scrollbar-in-a-rest-table
    """
    app.add_stylesheet('custom.css')


# If this is not None, a "Last updated on:" timestamp is inserted at
# every page bottom, using the given strftime() format.  The empty
# string is equivalent to '%b %d, %Y' (or a locale-dependent equivalent).
# html_last_updated_fmt = ''

# If True, add an index to the HTML documents.
html_use_index = True

# If True, the reStructuredText sources are included in the HTML build
# as _sources/name.
html_copy_source = True

# If True (and html_copy_source is True as well), links to the
# reStructuredText sources will be added to the sidebar.
html_show_sourcelink = True

# If True, "(C) Copyright ..." is shown in the HTML footer.
html_show_copyright = True

# If True, "Created using Sphinx" is shown in the HTML footer.
html_show_sphinx = True


# -- Options for HTML help output --------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'mdtoolsdoc'


# -- Options for Epub output -------------------------------------------

# The HTML theme for the epub output.
epub_theme = 'epub'

# The description of the document. The default value is 'unknown'.
epub_description = "MDTools' Documentation"


# -- Options for LaTeX output ------------------------------------------

# The LaTeX engine to build the docs.
latex_engine = 'pdflatex'

# If given, this must be the name of an image file (relative to the
# configuration directory) that is the logo of the docs.
latex_logo = None

# If true, add page references after internal references.  This is very
# useful for printed copies of the manual.
latex_show_pagerefs = True

# Control whether to display URL addresses.  This is very useful for
# printed copies of the manual.
latex_show_urls = 'footnote'


# -- Extension configuration -------------------------------------------
# -- Options for autodoc extension -------------------------------------

# This value selects what content will be inserted into the main body of
# an autoclass directive.
# See also napoleon_include_init_with_doc
autoclass_content = 'both'  # class and __init__ docstring are concatenated

# This value selects how automatically documented members are sorted.
# See also autodoc_default_options
autodoc_member_order = 'groupwise'

# This value is a list of autodoc directive flags that should be
# automatically applied to all autodoc directives.
# See also autodoc_default_options
# See also napoleon_include_init_with_doc
# See also napoleon_include_private_with_doc
# See also napoleon_include_special_with_doc
autodoc_default_flags = ['members',
                         'undoc-members']

# The default options for autodoc directives.
# See also autodoc_default_flags
# See also autodoc_member_order
# See also napoleon_include_init_with_doc
# See also napoleon_include_private_with_doc
# See also napoleon_include_special_with_doc
# autodoc_default_options = {'members': True,
#                            'member-order': 'groupwise',
#                            'undoc-members': True,
#                            'private-members': False,
#                            'special-members': False,
#                            'inherited-members': False,
#                            'show-inheritance': False,
#                            'imported-members': False}

# If True, the default argument values of functions will be not
# evaluated on generating document.  It preserves them as is in the
# source code.
autodoc_preserve_defaults = True

# If set to True the docstring for classes or methods, if not explicitly
# set, is inherited from parents.
autodoc_inherit_docstrings = True


# -- Options for autosectionlabel extension ----------------------------

# Useful for avoiding ambiguity when the same section heading appears in
# different documents.
# autosectionlabel_prefix_document = True


# -- Options for autosummary extension ---------------------------------

# Boolean indicating whether to scan all found documents for autosummary
# directives, and to generate stub pages for each.
autosummary_generate = True


# -- Options for doctest extension -------------------------------------

# A list of directories that will be added to sys.path when the doctest
# builder is used.  Make sure it contains absolute paths.
doctest_path = [os.path.abspath("./../..")]

# Python code that is treated like it were put in a testsetup directive
# for every file that is tested.
doctest_global_setup = '''
import numpy as np
import mdtools as mdt
'''

# If this is a nonempty string (the default is 'default'), standard
# reStructuredText doctest blocks will be tested too.
doctest_test_doctest_blocks = 'default'


# -- Options for intersphinx extension ---------------------------------

# Locations and names of other projects that should be linked to in this
# documentation.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'psutil': ('https://psutil.readthedocs.io/en/latest/', None),
    # 'tqdm': ('https://tqdm.github.io/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org', None),
    'MDAnalysis': ('https://docs.mdanalysis.org/stable/', None),
    'pyemma': ('http://emma-project.org/latest/', None)
}


# -- Options for napoleon extension ------------------------------------

# Support for Google style docstrings.
napoleon_google_docstring = False

# Support for NumPy style docstrings.
napoleon_numpy_docstring = True

# True to list __init___ docstrings separately from the class docstring.
# False to fall back to Sphinx's default behavior, which considers the
# __init___ docstring as part of the class documentation.
# See also autoclass_content
# See also autodoc_default_flags
# See also autodoc_default_options
# napoleon_include_init_with_doc = False

# True to include private members (like _membername) with docstrings in
# the documentation.
# See also autodoc_default_flags
# See also autodoc_default_options
# napoleon_include_private_with_doc = False

# True to include special members (like __membername__) with docstrings
# in the documentation.
# See also autodoc_default_flags
# See also autodoc_default_options
# napoleon_include_special_with_doc = False

# True to use a :param: role for each function parameter.  False to use
# a single :parameters: role for all the parameters.
napoleon_use_param = False


# -- Options for todo extension ----------------------------------------

# If this is True, todo and todolist directives produce output, else
# they produce nothing.
todo_include_todos = True
