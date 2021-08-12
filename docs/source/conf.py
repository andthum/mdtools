# -*- coding: utf-8 -*-

# Sort configuration options in the same order as in the documentation
# of Spinx.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""
Configuration file for the Sphinx documentation builder.

For a full list of options see the documentation:
http://www.sphinx-doc.org/en/master/config
"""


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

# The documented project's name.
project = mdt.__title__

# The author name(s) of the document.
author = mdt.__author__

# A copyright statement in the style "2008, Author Name".
copyright = mdt.__copyright__[14:]

# The short X.Y version
version = mdt.__version__

# The full version, including alpha/beta/rc tags.  If you do not need
# the separation provided between version and release, just set them
# both to the same value.
release = mdt.__version__


# -- General configuration ---------------------------------------------

# Sphinx extension module names as strings.
extensions = [
    # Include documentation from docstrings
    "sphinx.ext.autodoc",
    # Allow reference sections using its title
    #  "sphinx.ext.autosectionlabel",
    # Generate autodoc summaries
    "sphinx.ext.autosummary",
    # Collect doc coverage stats
    "sphinx.ext.coverage",
    # Test snippets in the documentation
    "sphinx.ext.doctest",
    # Measure durations of Sphinx processing
    "sphinx.ext.duration",
    # Publish HTML docs in GitHub Pages
    #  "sphinx.ext.githubpages",
    # Include inheritance diagrams
    #  "sphinx.ext.inheritance_diagram",
    # Link to other projects' documentation
    "sphinx.ext.intersphinx",
    # Render math via JavaScript for HTML output
    "sphinx.ext.mathjax",
    # Support for NumPy and Google style docstrings
    "sphinx.ext.napoleon",
    # Support for todo items
    "sphinx.ext.todo",
    # Add links to highlighted source code
    "sphinx.ext.viewcode",
    # Read the Docs Sphinx Theme
    "sphinx_rtd_theme",
    # Enable "search-as-you-type" on Read the Docs
    "sphinx_search.extension",
]

# The file extensions of source files.
source_suffix = {".rst": "restructuredtext"}

# The document that contains the root toctree directive.
master_doc = "index"

# A list of glob-style patterns that should be excluded when looking for
# source files.  They are matched against the source file names relative
# to the source directory, using slashes as directory separators on all
# platforms.  exclude_patterns is also consulted when looking for static
# files in html_static_path and html_extra_path.
exclude_patterns = []

# A list of paths that contain extra templates (or templates that
# overwrite builtin/theme-specific templates).  Relative paths are taken
# as relative to the configuration directory.
templates_path = ["_templates"]

# A string of reStructuredText that will be included at the end of every
# source file that is read.  This is a possible place to add
# substitutions that should be available in every file
rst_epilog = """
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
.. |mda_always_guesses_atom_masses| replace::
    :raw-html:`<a href="https://userguide.mdanalysis.org/formats/guessing.html">MDAnalysis always guesses atom masses</a>`
"""  # noqa: E501

# The name of the default domain.
primary_domain = "py"

# The name of a reStructuredText role (builtin or Sphinx extension) to
# use as the default role, that is, for text marked up `like this`.
default_role = None

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.0"

# If true, Sphinx will warn about all references where the target cannot
# be found.
# Would be nice if we could activate this.  However, therefore we have
# to revise a lot of docstrings first.
nitpicky = False

# The default language to highlight source code in.
highlight_language = "python3"

# The style name to use for Pygments highlighting of source code.
pygments_style = None

# Whether parentheses are appended to function and method role texts.
add_function_parentheses = True

# Whether module names are prepended to all object names.
add_module_names = True

# Whether codeauthor and sectionauthor directives produce any output in
# the built files.
show_authors = True

# Trim spaces before footnote references that are necessary for the
# reStructuredText parser to recognize the footnote, but do not look too
# nice in the output.
trim_footnote_reference_space = True

# If true, doctest flags (comments looking like # doctest: FLAG, ...) at
# the ends of lines and <BLANKLINE> markers are removed for all code
# blocks showing interactive Python sessions (i.e. doctests).
trim_doctest_flags = True


# -- Options for internationalization ----------------------------------

# The language for content autogenerated by Sphinx.  Refer to
# documentation for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"


# -- Options for HTML output -------------------------------------------

# The theme that the HTML output should use.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see
# the documentation.
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "both",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
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

# A list of CSS files.  Filenams must be relative to html_static_path.
html_css_files = ["custom.css"]

# A list of paths that contain custom static files (such as style sheets
# or script files).  Relative paths are taken as relative to the
# configuration directory.
html_static_path = ["_static"]

# If this is not None, a "Last updated on:" timestamp is inserted at
# every page bottom, using the given strftime() format.  The empty
# string is equivalent to "%b %d, %Y" (or a locale-dependent equivalent)
# html_last_updated_fmt = ""

# Whether to add permalinks for each heading and description environment
html_permalinks = True

# If True, add an index to the HTML documents.
html_use_index = True

# If True, the index is generated twice: once as a single page with all
# the entries, and once as one page per starting letter.
html_split_index = False

# If True, the reStructuredText sources are included in the HTML build
# as _sources/name.
html_copy_source = True

# If True (and html_copy_source is True as well), links to the
# reStructuredText sources will be added to the sidebar.
html_show_sourcelink = True

# If nonempty, an OpenSearch description file will be output, and all
# pages will contain a <link> tag referring to it.   the value of this
# option must be the base URL from which these documents are served
# (without trailing slash)
html_use_opensearch = "https://mdtools.readthedocs.io"

# If True, "(C) Copyright ..." is shown in the HTML footer.
html_show_copyright = True

# If True, "Created using Sphinx" is shown in the HTML footer.
html_show_sphinx = True

# Suffix for section numbers.
html_secnumber_suffix = ") "


# -- Options for HTML help output --------------------------------------
# These options influence the epub output. As this builder derives from
# the HTML builder, the HTML options also apply where appropriate.

# Output file base name for HTML help builder.
htmlhelp_basename = "mdtoolsdoc"


# -- Options for epub output -------------------------------------------

# The HTML theme for the epub output.
epub_theme = "epub"

# The description of the document. The default value is 'unknown'.
epub_description = "MDTools' Documentation"

# An identifier for the document.
epub_identifier = "https://github.com/andthum/mdtools"

# The publication scheme for the epub_identifier.
epub_scheme = "URL"

# Control whether to display URL addresses.
epub_show_urls = "footnote"


# -- Options for LaTeX output ------------------------------------------

# The LaTeX engine to build the docs.
latex_engine = "xelatex"

# The theme that the LaTeX output should use.
latex_theme = "manual"

# latex_documents determines how to group the document tree into LaTeX
# source files.
_targetname = htmlhelp_basename + ".tex"
_startdocname = master_doc
_title = "MDTools Documentation"
_author = r" \and ".join(author.split(", "))
_theme = latex_theme
_toctree_only = False
latex_documents = [
    _startdocname,
    _targetname,
    _title,
    _author,
    _theme,
    _toctree_only,
]

# If given, this must be the name of an image file (relative to the
# configuration directory) that is the logo of the docs.
latex_logo = "../logo/mdtools_label_3900x760.png"

# This value determines the topmost sectioning unit.
# latex_toplevel_sectioning = None

# Whether to add page references after internal references.
latex_show_pagerefs = True

# Control whether to display URL addresses.
latex_show_urls = epub_show_urls

# If True, the PDF build from the LaTeX files created by Sphinx will use
# xindy rather than makeindex for preparing the index of general terms.
# latex_use_xindy = True

# A dictionary that contains LaTeX snippets overriding those Sphinx
# usually puts into the generated .tex files.
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "12pt",
    "extrapackages": r"\usepackage{unicode-math}",
}


# -- Options for text output -------------------------------------------

# Determines which end-of-line character(s) are used in text output.
text_newlines = "native"

# Suffix for section numbers in text output.
text_secnumber_suffix = html_secnumber_suffix


# -- Extension configuration -------------------------------------------
# -- Options for autodoc extension -------------------------------------

# This value selects what content will be inserted into the main body of
# an autoclass directive.
# See also napoleon_include_init_with_doc
autoclass_content = "both"  # class and __init__ docstring are concatenated

# This value selects how automatically documented members are sorted.
# See also autodoc_default_options
autodoc_member_order = "groupwise"

# This value is a list of autodoc directive flags that should be
# automatically applied to all autodoc directives.
# See also autodoc_default_options
# See also napoleon_include_init_with_doc
# See also napoleon_include_private_with_doc
# See also napoleon_include_special_with_doc
autodoc_default_flags = ["members", "undoc-members"]

# The default options for autodoc directives.
# See also autodoc_default_flags
# See also autodoc_member_order
# See also napoleon_include_init_with_doc
# See also napoleon_include_private_with_doc
# See also napoleon_include_special_with_doc
# autodoc_default_options = {"members": True,
#                            "member-order": "groupwise",
#                            "undoc-members": True,
#                            "private-members": False,
#                            "special-members": False,
#                            "inherited-members": False,
#                            "show-inheritance": False,
#                            "imported-members": False}

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

# Whether to scan all found documents for autosummary directives, and to
# generate stub pages for each.
autosummary_generate = True

# Whether to overwrites existing files by generated stub pages.
autosummary_generate_overwrite = True

# Whether to document classes and functions imported in modules.
autosummary_imported_members = False


# -- Options for coverage extension ------------------------------------

# Whether to write headlines.
coverage_write_headline = True

# Whether to skip objects that are not documented in the source with a
# docstring.
coverage_skip_undoc_in_source = False

# Whether to print objects that are missing to standard output.
coverage_show_missing_items = True


# -- Options for doctest extension -------------------------------------

# A list of directories that will be added to sys.path when the doctest
# builder is used.  Make sure it contains absolute paths.
doctest_path = [os.path.abspath("./../..")]

# Python code that is treated like it were put in a testsetup directive
# for every file that is tested.
doctest_global_setup = """
import numpy as np
import mdtools as mdt
"""

# If this is a nonempty string (the default is "default"), standard
# reStructuredText doctest blocks will be tested too.
doctest_test_doctest_blocks = "default"


# -- Options for intersphinx extension ---------------------------------

# Locations and names of other projects that should be linked to in this
# documentation.
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org", None),
    "MDAnalysis": ("https://docs.mdanalysis.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "psutil": ("https://psutil.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

# The maximum number of days to cache remote inventories (in days).
intersphinx_cache_limit = 7


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

# True to use the .. admonition:: directive for the Example and Examples
# sections.  False to use the .. rubric:: directive instead.
napoleon_use_admonition_for_examples = False

# True to use the .. admonition:: directive for Notes sections.  False
# to use the .. rubric:: directive instead.
napoleon_use_admonition_for_notes = napoleon_use_admonition_for_examples

# True to use the .. admonition:: directive for References sections.
# False to use the .. rubric:: directive instead.
napoleon_use_admonition_for_references = napoleon_use_admonition_for_examples

# True to use the :ivar: role for instance variables.  False to use the
# .. attribute:: directive instead.
napoleon_use_ivar = False

# True to use a :param: role for each function parameter.  False to use
# a single :parameters: role for all the parameters.
napoleon_use_param = False

# True to use a :keyword: role for each function keyword argument.
# False to use a single :keyword arguments: role for all the keywords.
napoleon_use_keyword = napoleon_use_param

# True to use the :rtype: role for the return type. False to output the
# return type inline with the description.
napoleon_use_rtype = False

# Whether to convert the type definitions in the docstrings as
# references.
napoleon_preprocess_types = True


# -- Options for todo extension ----------------------------------------

# If this is True, todo and todolist directives produce output, else
# they produce nothing.
todo_include_todos = True

# If True, todo emits a warning for each TODO entry.
todo_emit_warnings = False
