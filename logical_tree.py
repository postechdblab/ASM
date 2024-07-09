import json
import numpy as np
import re
import pickle5 as pickle
import time
from Join_scheme.data_prepare import get_imdb_schema

ops = ['!=', '>=', '<=', '>', '<', '=']

def timestamp_transform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "'%Y-%m-%d %H:%M:%S'")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def split_string_space(string):
    pattern = r'[^\s\'"]+|\'[^\']*\'|"[^"]*"'
    matches = re.findall(pattern, string)

    return matches

def split_string_space_comma(s):
    pattern = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    pattern += r'|[^,\s]+(?:[\s,]+[^,\s]+)*'

    return re.findall(pattern, s)

def is_nested_string(s):
    assert len(s) > 0, f's = {s}'
    if s[0] == "'" and s[-1] == "'":
        return True
    if s[0] == '"' and s[-1] == '"':
        return True
    return False

def try_convert_float(v):
    try:
        return float(v), True
    except TypeError:
        return v, False
    except ValueError:
        return v, False
    except Exception:
        return v, False

def try_convert_int(v):
    try:
        return int(v), True
    except TypeError:
        return v, False
    except ValueError:
        return v, False
    except Exception:
        return v, False

def try_convert_numeric(v):
    v, converted = try_convert_int(v)
    if converted:
        return v
    else:
        v, _ = try_convert_float(v)
        return v

class Node:
    def __init__(self, op, children=None, value=None, cols=None, col=None):
        self.op = op
        self.children = children or []
        self.col = col

        if cols and value:
            self.value = ""
            self.op = ""
            tokens = split_string_space(value)
            found_col = False
            long_op = False
            for i, token in enumerate(tokens):
                if found_col or long_op:
                    found_col = False
                    token = token.lower()
                    if token in ["is", "not"]:
                        long_op = True
                    elif token in ["null", "like", "in"]:
                        long_op = False
                    if len(self.op) > 0:
                        self.op += ' '
                    self.op += token
                else:
                    for col in cols:
                        if token == col:
                            self.col = col
                            found_col = True
                            break
                    if not found_col:
                        if len(self.value) > 0:
                            self.value += ' '
                        self.value += token
        else:
            self.value = value

        if isinstance(self.value, str):
            if len(self.value) > 1 and is_nested_string(self.value):
                self.value = self.value[1:-1]

            elif len(self.value) > 0:
                self.value = try_convert_numeric(self.value)

        if (self.op.upper() in ["IN", "NOT IN"]) and isinstance(self.value, str):
            if (self.value[0] == "(" and self.value[-1] == ")") or (self.value[0] == "[" and self.value[-1] == "]"):
                tokens = split_string_space_comma(self.value[1:-1])
                tokens = [ token[1:-1] if is_nested_string(token) else token for token in tokens ]
                self.value = list(tokens)

    def str(self, indent=0):
        space = '  ' * indent
        if self.children:
            return space + f"{self.op}:\n" + space + "(\n" + f"{' '.join(child.str(indent+1) for child in self.children)}\n" + space + ")"
        elif self.value is not None:
            return space + str(self.value)
        else:
            return space + str(self.op)

    def to_dict(self):
        d = {"col": self.col, "op": self.op, "value": self.value.tolist() if isinstance(self.value, np.ndarray) else self.value}
        if self.children:
            d["children"] = [child.to_dict() for child in self.children]
        return d

def parse_logic_tree(expression, alias, table, schema):
    tokens, cols = tokenize(expression, alias, table, schema)
    root, _ = parse_expr(tokens, cols)
    return root


def tokenize(expression, alias, table, schema):
    delimiters = ['AND', 'OR', 'or', 'and', '(', ')']

    cols = [alias + '.' + col for col in schema.table_dictionary[table].attributes]

    new_expression = ""
    in_string_value = False
    for i in range(len(expression)):
        if expression[i] in ['"', "'"]:
            in_string_value = not in_string_value
            new_expression += expression[i]
        elif expression[i] in ['(', ')']:
            if not in_string_value:
                new_expression += ' '
                new_expression += expression[i]
                new_expression += ' '
            else:
                new_expression += expression[i]
        else:
            new_expression += expression[i]

    rep = new_expression
    for col in cols:
        for op in ops:
            rep = rep.replace(col + op, col + ' ' + op + ' ')
    tokens = split_string_space(rep)

    found_col = False
    new_tokens = []
    found_in = False
    acc = ""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if found_in:
            acc += token
            if token[0] not in ['(', "'", '"']:
                acc += ' '
            if token == ')':
                new_tokens.append(acc)
                acc = ""
                found_in = False
        elif found_col:
            found_col = False
            found_op = False
            rep = token.lower()
            for op in ops:
                if rep.startswith(op):
                    found_op = True
                    if len(op) < len(rep):
                        new_tokens.append(op)
                        new_tokens.append(token[len(op):])
                    else:
                        new_tokens.append(op)
                    break
            if not found_op:
                new_tokens.append(token)
                if rep == 'in' or (rep == 'not' and tokens[i+1].lower() == 'in'):
                    found_in = True
        else:
            for col in cols:
                if col == token:
                    found_col = True
                    new_tokens.append(token)
                    break
            if not found_col:
                new_tokens.append(token)
        i += 1

    ret = []
    acc = ""
    for token in new_tokens:
        if token in delimiters:
            if len(acc) > 0:
                ret.append(acc)
                acc = ""
            ret.append(token)
        else:
            if len(acc) > 0:
                acc += " " + token
            else:
                acc += token
    if len(acc) > 0:
        ret.append(acc)

    return ret, cols


def parse_expr(tokens, cols):
    stack = []
    while tokens:
        token = tokens.pop(0)
        if token == '(':
            node, tokens = parse_expr(tokens, cols)
            stack.append(node)
        elif token == ')':
            break
        elif token in ('AND', 'OR', 'and', 'or'):
            stack.append(Node(token.upper()))
        else:
            stack.append(Node(None, value=token, cols=cols))
    return build_tree(stack), tokens


def build_tree(stack):
    while len(stack) > 1:
        right = stack.pop()
        operator = stack.pop()
        left = stack.pop()
        operator.children.append(left)
        operator.children.append(right)
        stack.append(operator)
    return stack[0]

def fillcol(tree):
    if tree.col is None:
        ret = set()
        for child in tree.children:
            cols = fillcol(child)
            for col in cols:
                ret.add(col)
        if len(ret) == 1:
            tree.col = list(ret)[0]
        return ret
    else:
        return {tree.col}

def group(tree, ret, context=None):
    return

def get_subtree(tree, col):
    if tree.col == col:
        return tree

    for child in tree.children:
        subtree = get_subtree(child, col)
        if subtree is not None:
            return subtree

    return None


def to_neurocard_ops(tree):
    if tree.op == "is null":
        tree.op = "IS_NULL"
    elif tree.op == "is not null":
        tree.op = "IS_NOT_NULL"
    elif tree.op == "in":
        tree.op = "IN"
    elif tree.op == "not in":
        tree.op = "NOT_IN"
    elif tree.op == "like":
        tree.op = "LIKE"
    elif tree.op == "not like":
        tree.op = "NOT_LIKE"
    else:
        assert tree.op in ["AND", "OR", ">", "<", ">=", "<=", "=", "!="]

    if isinstance(tree.col, str) and "Date" in tree.col:
        print('(to_neurocard_ops) col', tree.col, 'val', tree.value)
        if tree.value is not None:
            assert "::timestamp" in tree.value
            tree.value = timestamp_transform(tree.value.strip().split("::timestamp")[0].strip())

    for child in tree.children:
        to_neurocard_ops(child)
