# -*- coding: utf-8 -*


class RuGraphInspector:

    # обязательные поля для каждого типа узла
    ATTRS_NODE_MAP = {
        'acc'  :   {
                    'type': 'acc',
                    'has_predict_edges': False, # исходит ли из него хоть одно ребро типа predict (нет)
                    'acc_obj': 'UN'             # объект-аккумулятор (см. класс DataAccumulator)
                    },

        'plain': {
                    'type': 'plain',
                    'has_predict_edges': 'UN',
                    'input': 'UN',
                    'bias': 'UN',
                    'waiting_inputs':'UN',
                    'activation': 'UN',
                    'activation_change': 'UN',
                    'acc_node_id': 'UN'
                 },

        'input': {
                  'type':'input',
                  'has_predict_edges': False,
                  'activation': 'UN',
                  'activation_change': 'UN',
                  'acc_node_id': 'UN'
                }
    }

    def __init__(self):
        self.is_ok = True
        self.err_msg = ""

    def inspect(self, G):
        dispatcher = {
            'acc': self.inspect_acc,
            'plain': self.inspect_plain,
            'input': self.inspect_input
        }
        for node_id in G.nodes():
            try:
                node_attrs = G.node[node_id]
                node_type = node_attrs['type']
                handler = dispatcher[node_type]
                is_ok = handler(node_attrs)
                if not is_ok:
                    return False
            except KeyError as e:
                self.err_msg = 'unknown node type' + str(e)
                return False
        return True

    def inspect_acc(self, real_attrs):
        must_have_attrs = self.ATTRS_NODE_MAP['acc']
        diff = must_have_attrs.viewkeys() - real_attrs.viewkeys()
        if len(diff) != 0:
            self.err_msg = "acc's parameters mismatch: " + str(diff)
            return False
        return True

    def inspect_plain(self, real_attrs):
        must_have_attrs = self.ATTRS_NODE_MAP['plain']
        diff = must_have_attrs.viewkeys() - real_attrs.viewkeys()
        if len(diff) != 0:
            self.err_msg = "plane's parameters mismatch: " + str(diff)
            return False
        return True

    def inspect_input(self, real_attrs):
        must_have_attrs = self.ATTRS_NODE_MAP['input']
        diff = must_have_attrs.viewkeys() - real_attrs.viewkeys()
        if len(diff) != 0:
            self.err_msg = "input's parameters mismatch: " + str(diff)
            return False
        return True