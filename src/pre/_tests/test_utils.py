import pytest


demo_sections = ['m1a', 'm2a', 'm3a', 'm1b', 'm2b', 'm3b']

def test_parse_sections(exp_config_path):

    from pre.utils import parse_sections
    sections = parse_sections(exp_config_path)

    for s in demo_sections:
        assert s in sections
