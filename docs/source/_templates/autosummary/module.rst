{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module attributes

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{# Recurse into sub-modules, skipping test packages/modules. #}
{% set documented = [] %}
{% for item in modules %}
{% set leaf = item.split('.')[-1] %}
{% if leaf not in ['tests', 'conftest'] and not leaf.startswith('test_') %}
{{ documented.append(item) or '' }}
{% endif %}
{% endfor %}
{% if documented %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in documented %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
