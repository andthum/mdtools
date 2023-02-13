# markdownlint style file.  See
# https://github.com/markdownlint/markdownlint/blob/master/docs/creating_styles.md

all
rule "MD007", :indent => 4  # Unordered list indentation
rule "MD009", :br_spaces => 2  # Trailing spaces
rule "MD013", :line_length => 72  # Line length
exclude_rule "MD026"  # Trailing punctuation in header
