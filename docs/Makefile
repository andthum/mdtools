# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS         =
#SPHINXOPTS         = -W
SPHINXBUILD        = sphinx-build
SOURCEDIR          = source
BUILDDIR           = build
AUTOSUMDIR_PATTERN = _sphinx_autosummary*
AUTOSUMDIR         = $(shell find $(SOURCEDIR) -type d -name "$(AUTOSUMDIR_PATTERN)")

# Put this first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "  clean          to remove everything under '$(BUILDDIR)'"
	@echo "  clean_autosum  to remove all '$(AUTOSUMDIR_PATTERN)' directories in '$(SOURCEDIR)'"

.PHONY: help clean_autosum Makefile

# Custom clean target to remove autosummary generated files. Can be
# removed when https://github.com/sphinx-doc/sphinx/issues/1999 is
# fixed.
clean_autosum:
	@echo "Removing all '$(AUTOSUMDIR_PATTERN)' directories in '$(SOURCEDIR)' recursively..."
	@rm -r $(AUTOSUMDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
