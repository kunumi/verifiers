import random
import math
import operator as op
from copy import deepcopy
import numpy as np

import time
import re 

import matplotlib.pyplot as plt
import networkx as nx
from io import StringIO


def tokenize(chars):
    "Convert a string of characters into a list of tokens."
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()

def parse(program):
    "Read a Scheme expression from a string."
    return read_from_tokens(tokenize(program))

def read_from_tokens(tokens):
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF')
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0) # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token):
    "Numbers become numbers; every other token is a symbol."
    try: return int(token)
    except ValueError:
        try: return float(token)
        except ValueError:
            return str(token)
        
def unparse(exp):
    """
    Convert a parsed Scheme expression back into its string representation.
    
    Args:
        exp (Exp): A parsed Scheme expression, either an Atom or a nested list.
    
    Returns:
        str: The string representation of the Scheme expression.
    """
    if isinstance(exp, list):
        return '(' + ' '.join(map(unparse, exp)) + ')'
    else:
        return str(exp)

def standard_env():
    env = Env()
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
        'cos':np.cos, 'sin':np.sin, 'log':np.log,
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
        'append':  op.add,  
        'apply':   lambda proc, args: proc(*args),
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'expt':    pow,
        'equal?':  op.eq, 
        'length':  len, 
        'list':    lambda *x: list(x), 
        'list?':   lambda x: isinstance(x, list), 
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, (int, float)),  
		'print':   print,
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, str),
        'and': lambda *x: all(x),
        'or': lambda *x: any(x)
    })
    return env
    
class Env(dict):
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        return self if (var in self) else self.outer.find(var)

class Procedure(object):

    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        return eval_lisp(self.body, Env(self.parms, args, self.env))

global_env = standard_env()


def eval_lisp(x, env=global_env):
    
    if isinstance(x, str):    # variable reference
        return env.find(x)[x]
    elif not isinstance(x, list):# constant 
        return x   
    op, *args = x       
    if op == 'quote':            # quotation
        return args[0]
    elif op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval_lisp(test, env) else alt)
        return eval_lisp(exp, env)
    elif op == 'define':         # definition
        (symbol, exp) = args
        env[symbol] = eval_lisp(exp, env)
    elif op == 'set!':           # assignment
        (symbol, exp) = args
        env.find(symbol)[symbol] = eval_lisp(exp, env)
    elif op == 'lambda':         # procedure
        (parms, body) = args
        return Procedure(parms, body, env)
    elif op == 'defun':          # define function
        (symbol, parms, *body) = args
        body = body[0]  # Extract the body from the list
        proc = Procedure(parms, body, env)
        env[symbol] = proc
    else:                        # procedure call
        proc = eval_lisp(op, env)
        vals = [eval_lisp(arg, env) for arg in args]
        return proc(*vals)

class GeneticProgramming:
    def __init__(self, function_set, terminal_set, digits=True, seed=None):
        """
        Initialize with the function set and terminal set.
        """
        self.function_set = function_set  # e.g., ['+', '-', '*', '/']
        self.terminal_set = terminal_set  # e.g., [1, 2, 3, 'x']
        self.digits = digits


        if (seed is not None):
            np.random.seed(seed)
            random.seed(seed)

    def generate_random_constant(self):
        return random.uniform(-10, 10)

    def generate_random_tree(self, max_depth=3, current_depth=0):
        """
        Gera uma árvore aleatória em notação prefixada.
        
        Args:
            max_depth (int): Profundidade máxima da árvore.
            current_depth (int): Profundidade atual da recursão.
            
        Returns:
            list or int/str: Árvore gerada, representada como lista em notação prefixada.
        """
        # Terminal condition
        if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.5):
            if random.random() > 0.5 and self.digits:
                return self.generate_random_constant()
            else:
                return random.choice(self.terminal_set)

        
        # Choose a random function and get its arity
        func = random.choice(list(self.function_set.keys()))
        arity = self.function_set[func]
        
        # Generate subtrees based on arity
        subtrees = [self.generate_random_tree(max_depth, current_depth + 1) for _ in range(arity)]
        
        return [func] + subtrees


    def mutate(self, tree, max_depth=3):
        """
        Mutates a tree by replacing a random subtree.
        """
        def random_subtree(depth=0):
            """Generate a random subtree."""
            if depth >= max_depth or (depth > 0 and random.random() < 0.5):
                return random.choice(self.terminal_set)
            func = random.choice(list(self.function_set.keys()))
            arity = self.function_set[func]
            return [func] + [random_subtree(depth + 1) for _ in range(arity)]
        
        # Randomly choose a mutation point in the tree
        def mutate_node(subtree):
            if not isinstance(subtree, list) or random.random() < 0.5:
                return random_subtree()  # Replace with new subtree
            func = subtree[0]
            arity = self.function_set[func]
            return [func] + [mutate_node(child) for child in subtree[1:1+arity]]
        
        return mutate_node(tree)

    def crossover(self, tree1, tree2):
        """
        Perform crossover between two trees by exchanging random subtrees.
        """
        def get_random_subtree(tree):
            """Get a random subtree from a tree."""
            if not isinstance(tree, list) or random.random() < 0.5:
                return tree, None  # Return terminal or operator without subtree
            idx = random.randint(1, len(tree) - 1)  # Choose a random child
            return tree[idx], idx

        # Select random subtrees from both parents
        subtree1, idx1 = get_random_subtree(tree1)
        subtree2, idx2 = get_random_subtree(tree2)

        # Swap subtrees
        if idx1 is not None:
            tree1 = deepcopy(tree1)
            tree1[idx1] = subtree2
        if idx2 is not None:
            tree2 = deepcopy(tree2)
            tree2[idx2] = subtree1

        return tree1, tree2