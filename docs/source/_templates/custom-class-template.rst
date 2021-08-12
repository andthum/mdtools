.. Custom CLASS template for Sphinx' autosummary extension
   https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
   from
   https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough/62613202#62613202
   Also see
   https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
   and
   https://sphinx-autosummary-recursion.readthedocs.io/en/latest/index.html

:github_url: hide

{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:

    {% block methods %}
    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
        :template: custom-method-template.rst
        :nosignatures:
    {% for item in methods %}
        {% if item != '__init__' %}
        ~{{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
        :template: custom-attribute-template.rst
        :nosignatures:
    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
