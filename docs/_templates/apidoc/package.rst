{%- macro summary_section(typ, title, in_list='__all__') %}
{% set member_names = get_members(typ=typ, in_list=in_list, out_format='names', include_imported=True) -%}
{% if member_names -%}
.. rubric:: :h2-no-toc:`{{ title }}`

.. autosummary::
    {% for name in member_names -%}
        {{ name }}
    {% endfor %}
{% endif -%}
{% endmacro -%}
{#- ====================================================== -#}

{{ fullname }}
{{  '=' * fullname|length }}

.. role:: h2-no-toc
    :class: h2-no-toc

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
    {% if subpackages or submodules %}
    .. rubric:: :h2-no-toc:`Substructure`

    .. toctree::
        :maxdepth: 2

        {% for package in subpackages %}
        {{ fullname + '.' + package }}
        {% endfor -%}
        {% for module in submodules -%}
        {{ fullname + '.' + module  }}
        {% endfor -%}
    {%- endif -%}

    {{ summary_section("class", "Classes") | indent }}
    {{ summary_section("function", "Functions") | indent }}
    {{ summary_section("exception", "Exceptions") | indent }}
    {{ summary_section("data", "Data") | indent }}

    {% if members %}
    .. rubric:: :h2-no-toc:`Reference`
    {%- endif %}

