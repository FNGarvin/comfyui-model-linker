"""
Tests for core/workflow_updater.py — run with: python tests/test_workflow_updater.py

Resolved paths are passed relative, so no ComfyUI folder_paths is needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.workflow_updater import update_workflow_nodes  # noqa: E402


def test_plain_widget_value_replaced():
    wf = {'nodes': [{'id': 1, 'widgets_values': ['old.safetensors']}]}
    update_workflow_nodes(wf, [{'node_id': 1, 'widget_index': 0, 'resolved_path': 'new.safetensors'}])
    assert wf['nodes'][0]['widgets_values'] == ['new.safetensors']


def test_rgthree_dict_slot_updated_in_place():
    slot = {'on': False, 'lora': 'old.safetensors', 'strength': 0.7}
    wf = {'nodes': [{'id': 1, 'widgets_values': [{}, slot, '']}]}
    update_workflow_nodes(wf, [{'node_id': 1, 'widget_index': 1, 'nested_key': 'lora',
                                'resolved_path': 'sub/new.safetensors'}])
    assert wf['nodes'][0]['widgets_values'][1] == {'on': False, 'lora': 'sub/new.safetensors', 'strength': 0.7}


def test_lora_manager_entry_and_text_updated():
    wf = {'nodes': [{'id': 1, 'widgets_values': [
        '', '<lora:old:1.0> <lora:keep:0.5>',
        [{'name': 'old', 'strength': 1.0, 'active': True}, {'name': 'keep', 'strength': 0.5}],
    ]}]}
    update_workflow_nodes(wf, [{'node_id': 1, 'widget_index': 2, 'list_index': 0, 'nested_key': 'name',
                                'text_widget_index': 1, 'resolved_path': 'styles\\new_lora.safetensors'}])
    values = wf['nodes'][0]['widgets_values']
    assert values[2][0] == {'name': 'new_lora', 'strength': 1.0, 'active': True}
    assert values[2][1]['name'] == 'keep'
    assert values[1] == '<lora:new_lora:1.0> <lora:keep:0.5>'


def test_lora_manager_bad_list_index_is_rejected():
    wf = {'nodes': [{'id': 1, 'widgets_values': ['', '', [{'name': 'a'}]]}]}
    update_workflow_nodes(wf, [{'node_id': 1, 'widget_index': 2, 'list_index': 5, 'nested_key': 'name',
                                'resolved_path': 'b.safetensors'}])
    assert wf['nodes'][0]['widgets_values'][2] == [{'name': 'a'}]


def test_subgraph_definition_node_updated():
    wf = {
        'nodes': [{'id': 1, 'widgets_values': ['top.safetensors']}],
        'definitions': {'subgraphs': [{'id': 'sg', 'nodes': [
            {'id': 1, 'widgets_values': [{'lora': 'inner.safetensors'}]}]}]},
    }
    update_workflow_nodes(wf, [{'node_id': 1, 'widget_index': 0, 'nested_key': 'lora', 'subgraph_id': 'sg',
                                'is_top_level': False, 'resolved_path': 'fixed.safetensors'}])
    assert wf['nodes'][0]['widgets_values'] == ['top.safetensors']
    assert wf['definitions']['subgraphs'][0]['nodes'][0]['widgets_values'][0]['lora'] == 'fixed.safetensors'


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except AssertionError as e:
                failures += 1
                print(f'FAIL {name}: {e}')
    sys.exit(1 if failures else 0)
