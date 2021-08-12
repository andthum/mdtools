.. Custom MODULE template for Sphinx' autosummary extension
   https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
   from
   https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough/62613202#62613202
   Also see
   https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
   and
   https://sphinx-autosummary-recursion.readthedocs.io/en/latest/index.html

:github_url: hide

{{ name | escape | underline}}

.. automodule:: {{ fullname }}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Module Attributes') }}

    .. autosummary::
        :toctree:
        :template: custom-attribute-template.rst
        :nosignatures:
    {% for item in attributes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Functions') }}

    .. autosummary::
        :toctree:
        :template: custom-function-template.rst
        :nosignatures:
    {% for item in functions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Classes') }}

    .. autosummary::
        :toctree:
        :template: custom-class-template.rst
        :nosignatures:
    {% for item in classes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: {{ _('Exceptions') }}

    .. autosummary::
        :toctree:
        :nosignatures:
    {% for item in exceptions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
    :toctree:
    :template: custom-module-template.rst
    :recursive:
    :nosignatures:
{% for item in modules %}
    {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
