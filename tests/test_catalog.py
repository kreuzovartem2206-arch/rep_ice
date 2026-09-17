from unittest.mock import Mock, patch

import pytest

from ice_monitor.catalog import s2_search


def test_stac_pagination_and_duplicate_ids():
    def scene(name, when):
        return {'id': name, 'properties': {'datetime':when}}
    a, b = scene('a', '2025-06-01T00:00:00Z'), scene('b', '2025-06-02T00:00:00Z')
    pages = [{'features':[a], 'links':[{'rel':'next','href':'https://example.test/page2'}]},
             {'features':[a,b], 'links':[]}]
    client = Mock()
    client.get.side_effect = [Mock(json=Mock(return_value=p)) for p in pages]
    with patch('ice_monitor.catalog.session') as factory:
        factory.return_value.__enter__.return_value = client
        result = s2_search([72,73,73,74], 'start', 'end', 5)
    assert [i['id'] for i in result] == ['a','b']
    assert client.get.call_count == 2


def test_catalogue_limit_fails_instead_of_silent_truncation():
    page = {'features':[{'id':'a','properties':{'datetime':'x'}}],
            'links':[{'rel':'next','href':'https://example.test/page2'}]}
    with patch('ice_monitor.catalog.session') as factory:
        factory.return_value.__enter__.return_value.get.return_value.json.return_value = page
        with pytest.raises(RuntimeError, match='limit'):
            s2_search([72,73,73,74], 'start', 'end', 1)
