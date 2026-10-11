"""
================================================================================
Advanced Non-Neural Network Code Generation AI
================================================================================
"""

import ast
import random
import numpy as np
from collections import defaultdict, deque, OrderedDict
import math
import json
import time
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import sys
import io
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import heapq
import copy
from pathlib import Path
import pickle
import threading
from abc import ABC, abstractmethod
import re


# ============================================================================
# PART 1: CONFIGURATION SYSTEM
# ============================================================================

@dataclass
class AIConfig:
    """Configuration for the AI system with all tunable parameters"""
    
    # Genetic Programming Parameters
    population_size: int = 100
    generations: int = 200
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_ratio: float = 0.1
    tournament_size: int = 5
    
    # MCTS Parameters
    mcts_iterations: int = 1000
    mcts_exploration: float = 1.414  # sqrt(2) - optimal for UCB1
    mcts_rollback_depth: int = 3
    mcts_max_tree_depth: int = 15
    
    # A* Search Parameters
    astar_max_iterations: int = 1000
    astar_heuristic_weight: float = 1.0
    
    # Fuzzy Logic Parameters
    fuzzy_resolution: int = 100
    defuzzification_method: str = "centroid"
    
    # Multi-objective Optimization
    use_pareto: bool = True
    pareto_size: int = 10
    pareto_objectives: List[str] = field(default_factory=lambda: ['correctness', 'simplicity', 'size'])
    
    # Performance Parameters
    use_parallel: bool = True
    max_workers: int = 4
    cache_size: int = 10000
    batch_evaluation_size: int = 50
    
    # Feature Flags
    auto_generate_tests: bool = True
    auto_refactor: bool = True
    interactive_mode: bool = False
    export_projects: bool = True
    grammar_induction: bool = True
    type_aware_operations: bool = True
    
    # Logging Parameters
    log_level: str = "INFO"
    save_evolution_data: bool = True
    verbose_output: bool = True
    
    # Fitness Weights
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'correctness': 0.7,
        'simplicity': 0.2,
        'size': 0.1
    })
    
    # Adaptive Parameters
    stagnation_threshold: int = 50
    max_mutation_rate: float = 0.8
    min_mutation_rate: float = 0.1
    diversity_threshold: float = 0.3
    
    # Code Generation Limits
    max_function_depth: int = 10
    max_expression_depth: int = 6
    max_function_size: int = 100  # nodes
    max_execution_time: float = 5.0
    
    # Refactoring Parameters
    refactoring_passes: int = 3
    max_refactor_time: float = 2.0


class Logger:
    """Thread-safe logging system"""
    
    LEVELS = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
    
    def __init__(self, level: str = "INFO", output: Optional[io.TextIOBase] = None):
        self.level = self.LEVELS.get(level, 1)
        self.output = output or sys.stdout
        self.lock = threading.Lock()
        self.log_history: List[Dict] = []
    
    def log(self, level: str, message: str, **kwargs):
        """Log a message with specified level"""
        msg_level = self.LEVELS.get(level, 1)
        if msg_level >= self.level:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            formatted = f"[{timestamp}] [{level}] {message}"
            
            with self.lock:
                print(formatted, file=self.output, flush=True)
                self.log_history.append({
                    'timestamp': timestamp,
                    'level': level,
                    'message': message,
                    **kwargs
                })
    
    def debug(self, message: str, **kwargs):
        self.log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)
    
    def get_history(self) -> List[Dict]:
        """Get log history"""
        with self.lock:
            return self.log_history.copy()


# Global logger instance
logger = Logger()


# ============================================================================
# PART 2: ENHANCED FUZZY LOGIC SYSTEM
# ============================================================================

class MembershipFunction(Enum):
    """Types of membership functions"""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    SIGMOID = "sigmoid"
    BELL = "bell"
    PIECEWISE = "piecewise"


@dataclass
class FuzzySet:
    """Represents a fuzzy set with a membership function"""
    name: str
    function_type: MembershipFunction
    params: Dict[str, float]
    points: List[Tuple[float, float]] = field(default_factory=list)
    
    def membership(self, x: float) -> float:
        """Calculate membership degree for crisp value x"""
        try:
            if self.function_type == MembershipFunction.TRIANGULAR:
                return self._triangular(x)
            elif self.function_type == MembershipFunction.TRAPEZOIDAL:
                return self._trapezoidal(x)
            elif self.function_type == MembershipFunction.GAUSSIAN:
                return self._gaussian(x)
            elif self.function_type == MembershipFunction.SIGMOID:
                return self._sigmoid(x)
            elif self.function_type == MembershipFunction.BELL:
                return self._bell(x)
            elif self.function_type == MembershipFunction.PIECEWISE:
                return self._piecewise(x)
        except (ZeroDivisionError, ValueError, OverflowError):
            return 0.0
        return 0.0
    
    def _triangular(self, x: float) -> float:
        a, b, c = self.params['a'], self.params['b'], self.params['c']
        if b == a or c == b:
            return 1.0 if x == b else 0.0
        if x <= a or x >= c:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    def _trapezoidal(self, x: float) -> float:
        a, b, c, d = self.params['a'], self.params['b'], self.params['c'], self.params['d']
        if b == a or d == c:
            return 0.0
        if x <= a or x >= d:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a)
        elif x <= c:
            return 1.0
        else:
            return (d - x) / (d - c)
    
    def _gaussian(self, x: float) -> float:
        mean, sigma = self.params['mean'], self.params['sigma']
        if sigma <= 0:
            return 0.0
        return math.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    
    def _sigmoid(self, x: float) -> float:
        center, slope = self.params['center'], self.params['slope']
        try:
            exp_val = -slope * (x - center)
            if exp_val > 500:
                return 0.0
            elif exp_val < -500:
                return 1.0
            return 1.0 / (1.0 + math.exp(exp_val))
        except OverflowError:
            return 0.0
    
    def _bell(self, x: float) -> float:
        a, b, c = self.params['a'], self.params['b'], self.params['c']
        if a <= 0 or b <= 0:
            return 0.0
        return 1.0 / (1.0 + abs((x - c) / a) ** (2 * b))
    
    def _piecewise(self, x: float) -> float:
        if not self.points or len(self.points) < 2:
            return 0.0
        
        sorted_points = sorted(self.points, key=lambda p: p[0])
        
        if x <= sorted_points[0][0]:
            return sorted_points[0][1]
        if x >= sorted_points[-1][0]:
            return sorted_points[-1][1]
        
        for i in range(len(sorted_points) - 1):
            x1, y1 = sorted_points[i]
            x2, y2 = sorted_points[i + 1]
            if x1 <= x <= x2:
                if x2 == x1:
                    return (y1 + y2) / 2
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        
        return 0.0


class FuzzyVariable:
    """Fuzzy variable with multiple membership functions"""
    
    def __init__(self, name: str, min_val: float, max_val: float, resolution: int = 100):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.resolution = resolution
        self.sets: Dict[str, FuzzySet] = {}
    
    def add_set(self, name: str, function_type: MembershipFunction,
                params: Optional[Dict[str, float]] = None,
                points: Optional[List[Tuple[float, float]]] = None):
        """Add a fuzzy set to this variable"""
        params = params or {}
        self.sets[name] = FuzzySet(name, function_type, params, points or [])
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """Convert crisp value to fuzzy membership degrees"""
        value = max(self.min_val, min(self.max_val, value))
        return {name: fset.membership(value) for name, fset in self.sets.items()}
    
    def defuzzify_centroid(self, fuzzy_dict: Dict[str, float],
                          resolution: Optional[int] = None) -> float:
        """Defuzzify using centroid method with numpy optimization"""
        res = resolution or self.resolution
        x_values = np.linspace(self.min_val, self.max_val, res)
        memberships = np.zeros(res)
        
        for name, degree in fuzzy_dict.items():
            if degree > 0 and name in self.sets:
                for i, x in enumerate(x_values):
                    memberships[i] = max(memberships[i], self.sets[name].membership(x) * degree)
        
        denominator = np.sum(memberships)
        if denominator > 0:
            return float(np.sum(x_values * memberships) / denominator)
        return (self.min_val + self.max_val) / 2
    
    def defuzzify_mean_max(self, fuzzy_dict: Dict[str, float]) -> float:
        """Defuzzify using mean of maxima method"""
        max_values = []
        max_membership = 0.0
        
        for name, degree in fuzzy_dict.items():
            if degree > 0 and name in self.sets:
                fset = self.sets[name]
                test_values = np.linspace(self.min_val, self.max_val, self.resolution)
                memberships = [fset.membership(x) * degree for x in test_values]
                local_max = max(memberships)
                
                if local_max > max_membership:
                    max_membership = local_max
                    max_values = [test_values[i] for i, m in enumerate(memberships) if abs(m - local_max) < 1e-9]
                elif abs(local_max - max_membership) < 1e-9:
                    max_values.extend([test_values[i] for i, m in enumerate(memberships) if abs(m - local_max) < 1e-9])
        
        return sum(max_values) / len(max_values) if max_values else (self.min_val + self.max_val) / 2
    
    def defuzzify_bisector(self, fuzzy_dict: Dict[str, float]) -> float:
        """Defuzzify using bisector method"""
        res = self.resolution
        x_values = np.linspace(self.min_val, self.max_val, res)
        memberships = np.zeros(res)
        
        for name, degree in fuzzy_dict.items():
            if degree > 0 and name in self.sets:
                for i, x in enumerate(x_values):
                    memberships[i] = max(memberships[i], self.sets[name].membership(x) * degree)
        
        total_area = np.sum(memberships)
        if total_area == 0:
            return (self.min_val + self.max_val) / 2
        
        half_area = total_area / 2
        cumulative = 0.0
        
        for i, m in enumerate(memberships):
            cumulative += m
            if cumulative >= half_area:
                return float(x_values[i])
        
        return float(x_values[-1])


class FuzzyRule:
    """Fuzzy if-then rule with support for different t-norms"""
    
    def __init__(self, antecedents: Dict[str, str], consequents: Dict[str, str],
                 weight: float = 1.0, tnorm: str = "min"):
        self.antecedents = antecedents
        self.consequents = consequents
        self.weight = weight
        self.tnorm = tnorm
    
    def evaluate(self, fuzzy_inputs: Dict[str, Dict[str, float]]) -> float:
        """Evaluate rule strength using specified t-norm"""
        strengths = []
        for var_name, set_name in self.antecedents.items():
            if var_name in fuzzy_inputs and set_name in fuzzy_inputs[var_name]:
                strengths.append(fuzzy_inputs[var_name][set_name])
            else:
                return 0.0
        
        if not strengths:
            return 0.0
        
        if self.tnorm == "min":
            result = min(strengths)
        elif self.tnorm == "product":
            result = math.prod(strengths)
        elif self.tnorm == "lukasiewicz":
            result = max(0, sum(strengths) - (len(strengths) - 1))
        else:
            result = min(strengths)
        
        return result * self.weight


class EnhancedFuzzySystem:
    """Complete fuzzy inference system with multiple defuzzification methods"""
    
    def __init__(self, defuzzification_method: str = "centroid"):
        self.variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.defuzzification_method = defuzzification_method
    
    def add_variable(self, name: str, min_val: float, max_val: float,
                    resolution: int = 100) -> FuzzyVariable:
        """Add a fuzzy variable"""
        var = FuzzyVariable(name, min_val, max_val, resolution)
        self.variables[name] = var
        return var
    
    def add_rule(self, antecedents: Dict[str, str], consequents: Dict[str, str],
                 weight: float = 1.0, tnorm: str = "min"):
        """Add a fuzzy rule"""
        self.rules.append(FuzzyRule(antecedents, consequents, weight, tnorm))
    
    def infer(self, crisp_inputs: Dict[str, float]) -> Dict[str, float]:
        """Perform complete fuzzy inference"""
        # Step 1: Fuzzify inputs
        fuzzy_inputs = {}
        for var_name, value in crisp_inputs.items():
            if var_name in self.variables:
                fuzzy_inputs[var_name] = self.variables[var_name].fuzzify(value)
        
        # Step 2: Evaluate rules and aggregate
        output_fuzzy: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        for rule in self.rules:
            strength = rule.evaluate(fuzzy_inputs)
            if strength > 0:
                for var_name, set_name in rule.consequents.items():
                    output_fuzzy[var_name][set_name] = max(
                        output_fuzzy[var_name][set_name],
                        strength
                    )
        
        # Step 3: Defuzzify outputs
        crisp_outputs = {}
        for var_name, fuzzy_dict in output_fuzzy.items():
            if var_name in self.variables:
                if self.defuzzification_method == "centroid":
                    crisp_outputs[var_name] = self.variables[var_name].defuzzify_centroid(fuzzy_dict)
                elif self.defuzzification_method == "mean_max":
                    crisp_outputs[var_name] = self.variables[var_name].defuzzify_mean_max(fuzzy_dict)
                elif self.defuzzification_method == "bisector":
                    crisp_outputs[var_name] = self.variables[var_name].defuzzify_bisector(fuzzy_dict)
                else:
                    crisp_outputs[var_name] = self.variables[var_name].defuzzify_centroid(fuzzy_dict)
        
        return crisp_outputs
    
    def create_standard_linguistic_variable(self, name: str, min_val: float,
                                            max_val: float, num_terms: int = 5) -> FuzzyVariable:
        """Create standard linguistic variable with evenly spaced terms"""
        var = self.add_variable(name, min_val, max_val)
        
        if num_terms == 3:
            terms = ['low', 'medium', 'high']
        elif num_terms == 5:
            terms = ['very_low', 'low', 'medium', 'high', 'very_high']
        elif num_terms == 7:
            terms = ['extremely_low', 'very_low', 'low', 'medium', 'high', 'very_high', 'extremely_high']
        else:
            terms = [f'term_{i}' for i in range(num_terms)]
        
        step = (max_val - min_val) / (num_terms - 1)
        
        for i, term in enumerate(terms):
            center = min_val + i * step
            width = step * 0.8
            
            if i == 0:
                var.add_set(term, MembershipFunction.TRAPEZOIDAL,
                           {'a': min_val, 'b': min_val, 'c': center, 'd': center + width})
            elif i == num_terms - 1:
                var.add_set(term, MembershipFunction.TRAPEZOIDAL,
                           {'a': center - width, 'b': center, 'c': max_val, 'd': max_val})
            else:
                var.add_set(term, MembershipFunction.TRIANGULAR,
                           {'a': center - width, 'b': center, 'c': center + width})
        
        return var


# ============================================================================
# PART 3: AST REPRESENTATION AND TYPE SYSTEM
# ============================================================================

@dataclass
class ASTNode:
    """Represents a node in the code AST with type information"""
    node_type: str
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    return_type: Optional[str] = None
    
    def __hash__(self):
        content = (self.node_type, str(self.value),
                  tuple(hash(c) for c in self.children))
        return hash(content)
    
    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        return (self.node_type == other.node_type and
                self.value == other.value and
                len(self.children) == len(other.children) and
                all(c1 == c2 for c1, c2 in zip(self.children, other.children)))
    
    def deep_copy(self) -> 'ASTNode':
        """Create a deep copy of this node"""
        new_node = ASTNode(self.node_type, self.value)
        new_node.children = [child.deep_copy() for child in self.children]
        new_node.metadata = self.metadata.copy()
        new_node.return_type = self.return_type
        return new_node
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'node_type': self.node_type,
            'value': self.value,
            'children': [c.to_dict() for c in self.children],
            'return_type': self.return_type,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ASTNode':
        """Deserialize from dictionary"""
        node = cls(data['node_type'], data.get('value'))
        node.children = [cls.from_dict(c) for c in data.get('children', [])]
        node.return_type = data.get('return_type')
        node.metadata = data.get('metadata', {})
        return node
    
    def get_all_nodes(self) -> List['ASTNode']:
        """Get all nodes in the subtree"""
        nodes = []
        stack = [self]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes
    
    def get_nodes_by_type(self, node_type: str) -> List['ASTNode']:
        """Get all nodes of a specific type"""
        return [n for n in self.get_all_nodes() if n.node_type == node_type]


class TypeSystem:
    """Enhanced type system for AST nodes"""
    
    OPERATOR_TYPES = {
        ('int', '+', 'int'): 'int',
        ('int', '-', 'int'): 'int',
        ('int', '*', 'int'): 'int',
        ('int', '/', 'int'): 'float',
        ('int', '%', 'int'): 'int',
        ('int', '//', 'int'): 'int',
        ('int', '**', 'int'): 'int',
        ('float', '+', 'float'): 'float',
        ('float', '-', 'float'): 'float',
        ('float', '*', 'float'): 'float',
        ('float', '/', 'float'): 'float',
        ('list', '+', 'list'): 'list',
        ('str', '+', 'str'): 'str',
        ('str', '*', 'int'): 'str',
        ('int', '*', 'str'): 'str',
    }
    
    BUILTIN_RETURN_TYPES = {
        'len': ('list', 'int'),
        'sum': ('list', 'int'),
        'max': ('list', 'int'),
        'min': ('list', 'int'),
        'abs': ('int', 'int'),
        'sorted': ('list', 'list'),
        'range': ('int', 'list'),
        'int': ('any', 'int'),
        'float': ('any', 'float'),
        'str': ('any', 'str'),
        'list': ('any', 'list'),
        'enumerate': ('list', 'list'),
        'zip': ('list', 'list'),
    }
    
    @classmethod
    def infer_type(cls, node: ASTNode, context: Optional[Dict[str, str]] = None) -> str:
        """Infer the type of an AST node"""
        context = context or {}
        
        if node.node_type == 'literal':
            if isinstance(node.value, bool):
                return 'bool'
            elif isinstance(node.value, int):
                return 'int'
            elif isinstance(node.value, float):
                return 'float'
            elif isinstance(node.value, str):
                return 'str'
            elif isinstance(node.value, list):
                return 'list'
            return 'any'
        
        elif node.node_type == 'variable':
            if node.children and node.children[0].value:
                return context.get(node.children[0].value, 'any')
            return 'any'
        
        elif node.node_type == 'binary_op':
            if len(node.children) >= 3:
                left_type = cls.infer_type(node.children[0], context)
                right_type = cls.infer_type(node.children[2], context)
                op = node.children[1].value if node.children[1].value else '+'
                return cls.OPERATOR_TYPES.get((left_type, op, right_type), 'any')
            return 'any'
        
        elif node.node_type in ['comparison', 'logical', 'not_condition', 'membership_test', 'none_check']:
            return 'bool'
        
        elif node.node_type == 'func_call':
            if node.children and node.children[0].value:
                func_name = node.children[0].value
                if func_name in cls.BUILTIN_RETURN_TYPES:
                    return cls.BUILTIN_RETURN_TYPES[func_name][1]
            return 'any'
        
        elif node.node_type == 'list_literal':
            if node.children:
                elem_types = [cls.infer_type(c, context) for c in node.children]
                return f'list[{elem_types[0]}]' if elem_types else 'list'
            return 'list'
        
        elif node.node_type == 'index_access':
            if node.children:
                obj_type = cls.infer_type(node.children[0], context)
                if obj_type.startswith('list['):
                    return obj_type[5:-1]
                elif obj_type == 'list':
                    return 'any'
                elif obj_type == 'str':
                    return 'str'
            return 'any'
        
        elif node.node_type == 'ternary':
            if node.children:
                return cls.infer_type(node.children[0], context)
            return 'any'
        
        elif node.node_type == 'list_comp':
            if node.children:
                return f'list[{cls.infer_type(node.children[0], context)}]'
            return 'list'
        
        return 'any'
    
    @classmethod
    def is_compatible(cls, type1: str, type2: str) -> bool:
        """Check if two types are compatible"""
        if type1 == 'any' or type2 == 'any':
            return True
        if type1 == type2:
            return True
        if {type1, type2} == {'int', 'float'}:
            return True
        if type1.startswith('list[') and type2 == 'list':
            return True
        if type2.startswith('list[') and type1 == 'list':
            return True
        return False
    
    @classmethod
    def get_compatible_nodes(cls, target_type: str, nodes: List[ASTNode],
                            context: Optional[Dict[str, str]] = None) -> List[ASTNode]:
        """Get nodes that are type-compatible with target"""
        return [n for n in nodes if cls.is_compatible(cls.infer_type(n, context), target_type)]


# ============================================================================
# PART 4: AST GRAMMAR
# ============================================================================

class ASTGrammar:
    """Advanced grammar for AST-based code generation"""
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.rules = self._build_grammar()
        self.node_weights = self._build_node_weights()
        self.max_depths = self._build_max_depths()
        self.type_system = TypeSystem()
    
    def _build_grammar(self) -> Dict[str, List[Dict]]:
        """Build comprehensive grammar rules"""
        return {
            'program': [
                {'type': 'func_def', 'children': ['identifier', 'params', 'block']}
            ],
            'func_def': [
                {'type': 'function', 'template': 'def {name}({params}):\n    {body}'}
            ],
            'params': [
                {'type': 'param_list', 'options': [['x'], ['n'], ['x', 'y'], ['data'], ['items'], ['arr'], ['lst'], ['s'], ['text']]},
                {'type': 'param_list', 'options': [['n', 'x'], ['arr', 'target'], ['lst', 'start'], ['data', 'key']]},
            ],
            'block': [
                {'type': 'return_stmt', 'children': ['expression']},
                {'type': 'if_stmt', 'children': ['condition', 'block', 'block']},
                {'type': 'for_stmt', 'children': ['identifier', 'expression', 'block']},
                {'type': 'while_stmt', 'children': ['condition', 'block']},
                {'type': 'assignment', 'children': ['identifier', 'expression']},
                {'type': 'multi_stmt', 'children': ['statement', 'statement']},
                {'type': 'var_init_stmt', 'children': ['identifier', 'expression', 'block']},
            ],
            'statement': [
                {'type': 'return_stmt', 'children': ['expression']},
                {'type': 'assignment', 'children': ['identifier', 'expression']},
                {'type': 'if_stmt', 'children': ['condition', 'block', 'block']},
                {'type': 'for_stmt', 'children': ['identifier', 'expression', 'block']},
                {'type': 'while_stmt', 'children': ['condition', 'block']},
                {'type': 'augmented_assignment', 'children': ['identifier', 'operator', 'expression']},
                {'type': 'break_stmt'},
                {'type': 'continue_stmt'},
            ],
            'expression': [
                {'type': 'binary_op', 'children': ['expression', 'operator', 'expression']},
                {'type': 'unary_op', 'children': ['operator', 'expression']},
                {'type': 'func_call', 'children': ['identifier', 'arg_list']},
                {'type': 'literal', 'options': ['number', 'string', 'boolean']},
                {'type': 'variable', 'children': ['identifier']},
                {'type': 'list_comp', 'children': ['expression', 'identifier', 'expression', 'condition']},
                {'type': 'ternary', 'children': ['expression', 'condition', 'expression']},
                {'type': 'list_literal', 'children': ['expression_list']},
                {'type': 'index_access', 'children': ['expression', 'expression']},
                {'type': 'slice_access', 'children': ['expression', 'expression', 'expression']},
                {'type': 'method_call', 'children': ['expression', 'identifier', 'arg_list']},
            ],
            'expression_list': [
                {'type': 'expr_list', 'children': ['expression']},
                {'type': 'expr_list', 'children': ['expression', 'expression_list']},
            ],
            'condition': [
                {'type': 'comparison', 'children': ['expression', 'comparison_op', 'expression']},
                {'type': 'logical', 'children': ['condition', 'logical_op', 'condition']},
                {'type': 'not_condition', 'children': ['condition']},
                {'type': 'bool_literal', 'options': [True, False]},
                {'type': 'membership_test', 'children': ['expression', 'expression']},
                {'type': 'none_check', 'children': ['expression']},
            ],
            'operator': [
                {'type': 'arithmetic', 'options': ['+', '-', '*', '/', '%', '**', '//']}
            ],
            'comparison_op': [
                {'type': 'comparison', 'options': ['==', '!=', '<', '>', '<=', '>=']}
            ],
            'logical_op': [
                {'type': 'logical', 'options': ['and', 'or']}
            ],
            'identifier': [
                {'type': 'var', 'options': ['x', 'y', 'n', 'result', 'total', 'count', 'i', 'j', 'k',
                                           'item', 'data', 'arr', 'lst', 'temp', 'max_val', 'min_val',
                                           'sum_val', 'avg', 'idx', 'elem', 'val', 'acc', 'found', 'pos']}
            ],
            'number': [
                {'type': 'literal', 'options': [0, 1, 2, 3, 5, 10, 100, -1, 0.5, 2.0, 10.0, 1e6, -100]}
            ],
            'arg_list': [
                {'type': 'args', 'children': ['expression']},
                {'type': 'multi_args', 'children': ['expression', 'expression']},
                {'type': 'no_args'},
            ],
        }
    
    def _build_node_weights(self) -> Dict[str, float]:
        """Build weights for node selection (bias toward simpler constructs)"""
        return {
            'literal': 1.2,
            'variable': 1.1,
            'binary_op': 0.8,
            'unary_op': 0.6,
            'func_call': 0.7,
            'method_call': 0.5,
            'if_stmt': 0.5,
            'for_stmt': 0.5,
            'while_stmt': 0.25,
            'list_comp': 0.4,
            'ternary': 0.35,
            'index_access': 0.5,
            'slice_access': 0.3,
            'assignment': 0.7,
            'augmented_assignment': 0.6,
            'return_stmt': 0.9,
            'multi_stmt': 0.4,
            'var_init_stmt': 0.5,
        }
    
    def _build_max_depths(self) -> Dict[str, int]:
        """Build maximum depth limits per non-terminal"""
        return {
            'program': 10,
            'block': 8,
            'statement': 7,
            'expression': 6,
            'condition': 4,
            'arg_list': 4,
            'expression_list': 5,
        }
    
    def get_valid_expansions(self, nonterminal: str) -> List[Dict]:
        """Get valid grammar expansions for a non-terminal"""
        return self.rules.get(nonterminal, [])
    
    def get_weighted_expansion(self, nonterminal: str) -> Optional[Dict]:
        """Get expansion weighted by node weights"""
        expansions = self.get_valid_expansions(nonterminal)
        if not expansions:
            return None
        
        weights = [self.node_weights.get(exp['type'], 0.5) for exp in expansions]
        total = sum(weights)
        if total == 0:
            return random.choice(expansions)
        weights = [w / total for w in weights]
        
        return random.choices(expansions, weights=weights, k=1)[0]
    
    def get_max_depth(self, nonterminal: str, default: int = 6) -> int:
        """Get maximum depth for a non-terminal"""
        return self.max_depths.get(nonterminal, default)


# ============================================================================
# PART 5: FITNESS CACHE
# ============================================================================

class FitnessCache:
    """Thread-safe LRU fitness cache"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, code: str) -> Optional[Dict]:
        """Get cached fitness if exists"""
        key = hashlib.md5(code.encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, code: str, fitness_data: Dict):
        """Store fitness in cache"""
        key = hashlib.md5(code.encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = fitness_data
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


# ============================================================================
# PART 6: AST INDIVIDUAL
# ============================================================================

class ASTIndividual:
    """Individual in GP population represented as AST"""
    
    _id_counter = 0
    _id_lock = threading.Lock()
    
    def __init__(self, ast_root: Optional[ASTNode] = None):
        with ASTIndividual._id_lock:
            ASTIndividual._id_counter += 1
            self.id = ASTIndividual._id_counter
        
        self.ast_root = ast_root
        self.fitness = 0.0
        self.fitness_components: Dict[str, float] = {}
        self.code_string = ""
        self.execution_result = None
        self.error = None
        self.complexity = 0
        self.size = 0
        self.depth = 0
        self.test_results: List[bool] = []
        self.is_dominated = False
        self.rank = 0
        self.crowding_distance = 0.0
        self.age = 0
        self.parent_ids: Set[int] = set()
    
    def calculate_metrics(self) -> Dict[str, int]:
        """Calculate all code metrics"""
        return {
            'complexity': self.calculate_complexity(),
            'size': self.calculate_size(),
            'depth': self.calculate_depth()
        }
    
    def calculate_complexity(self) -> int:
        """Calculate cyclomatic complexity"""
        if not self.ast_root:
            return 0
        
        complexity = 1
        for node in self.ast_root.get_all_nodes():
            if node.node_type in ['if_stmt', 'for_stmt', 'while_stmt', 'list_comp', 'ternary']:
                complexity += 1
            elif node.node_type == 'logical' and node.children:
                op = node.children[1].value if len(node.children) > 1 else None
                if op in ['and', 'or']:
                    complexity += 1
        
        self.complexity = complexity
        return complexity
    
    def calculate_size(self) -> int:
        """Calculate AST size (number of nodes)"""
        if not self.ast_root:
            return 0
        
        self.size = len(self.ast_root.get_all_nodes())
        return self.size
    
    def calculate_depth(self) -> int:
        """Calculate AST depth iteratively"""
        if not self.ast_root:
            return 0
        
        max_depth = 0
        stack = [(self.ast_root, 1)]
        
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            for child in node.children:
                stack.append((child, depth + 1))
        
        self.depth = max_depth
        return max_depth
    
    def to_code(self) -> str:
        """Convert AST to code string"""
        if not self.ast_root:
            return ""
        return self._node_to_code(self.ast_root)
    
    def _node_to_code(self, node: ASTNode, indent: int = 0) -> str:
        """Recursively convert AST node to code"""
        indent_str = "    " * indent
        
        if node.node_type == 'function':
            name = node.children[0].value if node.children else 'func'
            params_list = node.children[1].children if len(node.children) > 1 else []
            params = ', '.join([c.value for c in params_list if c.value])
            body = '\n'.join([self._node_to_code(c, indent + 1) for c in node.children[2:]])
            return f"{indent_str}def {name}({params}):\n{body}"
        
        elif node.node_type == 'return_stmt':
            expr = self._node_to_code(node.children[0]) if node.children else 'None'
            return f"{indent_str}return {expr}"
        
        elif node.node_type == 'if_stmt':
            cond = self._node_to_code(node.children[0]) if node.children else 'True'
            then_body = '\n'.join([self._node_to_code(c, indent + 1) for c in node.children[1:2]])
            result = f"{indent_str}if {cond}:\n{then_body}"
            if len(node.children) > 2:
                else_body = '\n'.join([self._node_to_code(c, indent + 1) for c in node.children[2:]])
                result += f"\n{indent_str}else:\n{else_body}"
            return result
        
        elif node.node_type == 'for_stmt':
            var = node.children[0].value if node.children else 'i'
            iterable = self._node_to_code(node.children[1]) if len(node.children) > 1 else 'range(10)'
            body = '\n'.join([self._node_to_code(c, indent + 1) for c in node.children[2:]])
            return f"{indent_str}for {var} in {iterable}:\n{body}"
        
        elif node.node_type == 'while_stmt':
            cond = self._node_to_code(node.children[0]) if node.children else 'True'
            body = '\n'.join([self._node_to_code(c, indent + 1) for c in node.children[1:]])
            return f"{indent_str}while {cond}:\n{body}"
        
        elif node.node_type == 'assignment':
            var = node.children[0].value if node.children else 'x'
            expr = self._node_to_code(node.children[1]) if len(node.children) > 1 else '0'
            return f"{indent_str}{var} = {expr}"
        
        elif node.node_type == 'augmented_assignment':
            var = node.children[0].value if node.children else 'x'
            op = node.children[1].value if len(node.children) > 1 else '+'
            expr = self._node_to_code(node.children[2]) if len(node.children) > 2 else '1'
            return f"{indent_str}{var} {op}= {expr}"
        
        elif node.node_type == 'var_init_stmt':
            var = node.children[0].value if node.children else 'result'
            init = self._node_to_code(node.children[1]) if len(node.children) > 1 else '0'
            body = '\n'.join([self._node_to_code(c, indent + 1) for c in node.children[2:]])
            return f"{indent_str}{var} = {init}\n{body}"
        
        elif node.node_type == 'break_stmt':
            return f"{indent_str}break"
        
        elif node.node_type == 'continue_stmt':
            return f"{indent_str}continue"
        
        elif node.node_type == 'binary_op':
            left = self._node_to_code(node.children[0]) if node.children else '0'
            op = node.children[1].value if len(node.children) > 1 else '+'
            right = self._node_to_code(node.children[2]) if len(node.children) > 2 else '0'
            return f"({left} {op} {right})"
        
        elif node.node_type == 'unary_op':
            op = node.children[0].value if node.children else '-'
            expr = self._node_to_code(node.children[1]) if len(node.children) > 1 else '0'
            return f"({op}{expr})"
        
        elif node.node_type == 'comparison':
            left = self._node_to_code(node.children[0]) if node.children else '0'
            op = node.children[1].value if len(node.children) > 1 else '=='
            right = self._node_to_code(node.children[2]) if len(node.children) > 2 else '0'
            return f"({left} {op} {right})"
        
        elif node.node_type == 'logical':
            left = self._node_to_code(node.children[0]) if node.children else 'True'
            op = node.children[1].value if len(node.children) > 1 else 'and'
            right = self._node_to_code(node.children[2]) if len(node.children) > 2 else 'True'
            return f"({left} {op} {right})"
        
        elif node.node_type == 'not_condition':
            cond = self._node_to_code(node.children[0]) if node.children else 'True'
            return f"(not {cond})"
        
        elif node.node_type == 'membership_test':
            elem = self._node_to_code(node.children[0]) if node.children else 'x'
            container = self._node_to_code(node.children[1]) if len(node.children) > 1 else '[]'
            return f"({elem} in {container})"
        
        elif node.node_type == 'none_check':
            expr = self._node_to_code(node.children[0]) if node.children else 'x'
            return f"({expr} is not None)"
        
        elif node.node_type == 'variable':
            return node.children[0].value if node.children else 'x'
        
        elif node.node_type == 'literal':
            if isinstance(node.value, str):
                return f"'{node.value}'"
            elif node.value is None:
                return 'None'
            elif isinstance(node.value, bool):
                return 'True' if node.value else 'False'
            return str(node.value)
        
        elif node.node_type == 'func_call':
            func_name = node.children[0].value if node.children else 'func'
            if len(node.children) > 1 and node.children[1].node_type != 'no_args':
                args = ', '.join([self._node_to_code(c) for c in node.children[1:]])
                return f"{func_name}({args})"
            return f"{func_name}()"
        
        elif node.node_type == 'method_call':
            obj = self._node_to_code(node.children[0]) if node.children else 'x'
            method = node.children[1].value if len(node.children) > 1 else 'method'
            if len(node.children) > 2 and node.children[2].node_type != 'no_args':
                args = ', '.join([self._node_to_code(c) for c in node.children[2:]])
                return f"{obj}.{method}({args})"
            return f"{obj}.{method}()"
        
        elif node.node_type == 'ternary':
            then_expr = self._node_to_code(node.children[0]) if node.children else '0'
            cond = self._node_to_code(node.children[1]) if len(node.children) > 1 else 'True'
            else_expr = self._node_to_code(node.children[2]) if len(node.children) > 2 else '0'
            return f"({then_expr} if {cond} else {else_expr})"
        
        elif node.node_type == 'list_literal':
            elements = [self._node_to_code(c) for c in node.children]
            return f"[{', '.join(elements)}]"
        
        elif node.node_type == 'list_comp':
            expr = self._node_to_code(node.children[0]) if node.children else 'x'
            var = node.children[1].value if len(node.children) > 1 else 'x'
            iterable = self._node_to_code(node.children[2]) if len(node.children) > 2 else '[]'
            if len(node.children) > 3:
                cond = self._node_to_code(node.children[3])
                return f"[{expr} for {var} in {iterable} if {cond}]"
            return f"[{expr} for {var} in {iterable}]"
        
        elif node.node_type == 'index_access':
            obj = self._node_to_code(node.children[0]) if node.children else 'arr'
            idx = self._node_to_code(node.children[1]) if len(node.children) > 1 else '0'
            return f"{obj}[{idx}]"
        
        elif node.node_type == 'slice_access':
            obj = self._node_to_code(node.children[0]) if node.children else 'arr'
            start = self._node_to_code(node.children[1]) if len(node.children) > 1 else ''
            end = self._node_to_code(node.children[2]) if len(node.children) > 2 else ''
            return f"{obj}[{start}:{end}]"
        
        elif node.node_type == 'param_list':
            return ', '.join([c.value for c in node.children if c.value])
        
        elif node.node_type == 'no_args':
            return ''
        
        elif node.node_type == 'multi_stmt':
            return '\n'.join([self._node_to_code(c, indent) for c in node.children])
        
        elif node.node_type == 'expr_list':
            return ', '.join([self._node_to_code(c) for c in node.children])
        
        return str(node.value) if node.value is not None else ''


# ============================================================================
# PART 7: SANDBOXED EXECUTOR
# ============================================================================

class SandboxedExecutor:
    """Safely executes generated code in a sandboxed environment"""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.allowed_builtins = {
            'abs': abs, 'min': min, 'max': max, 'len': len, 'range': range,
            'sum': sum, 'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'tuple': tuple, 'dict': dict, 'set': set,
            'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'round': round,
            'pow': pow, 'any': any, 'all': all, 'isinstance': isinstance,
            'type': type, 'Exception': Exception, 'ValueError': ValueError,
            'TypeError': TypeError, 'KeyError': KeyError, 'IndexError': IndexError,
        }
    
    def execute(self, code: str, test_input: Any = None,
                timeout: Optional[float] = None) -> Tuple[Any, Optional[str]]:
        """Execute code and return result or error"""
        timeout = timeout or self.timeout
        start_time = time.time()
        
        try:
            # Check for potentially dangerous code
            if self._is_dangerous(code):
                return None, "Potentially dangerous code detected"
            
            sandbox = {'__builtins__': self.allowed_builtins}
            
            # Compile first to catch syntax errors
            try:
                compiled = compile(code, '<generated>', 'exec')
            except SyntaxError as e:
                return None, f"Syntax error: {e}"
            
            exec(compiled, sandbox)
            
            # Find the function
            func_name = None
            for name, obj in sandbox.items():
                if callable(obj) and not name.startswith('_') and name not in self.allowed_builtins:
                    func_name = name
                    break
            
            if func_name is None:
                return None, "No function found"
            
            if test_input is not None:
                result = sandbox[func_name](test_input)
            else:
                result = sandbox[func_name]()
            
            # Check timeout
            if time.time() - start_time > timeout:
                return None, "Execution timeout"
            
            return result, None
        
        except RecursionError:
            return None, "Maximum recursion depth exceeded"
        except MemoryError:
            return None, "Memory limit exceeded"
        except TimeoutError:
            return None, "Execution timeout"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    
    def _is_dangerous(self, code: str) -> bool:
        """Check for potentially dangerous code patterns"""
        dangerous_patterns = [
            r'import\s+', r'__import__', r'exec\s*\(', r'eval\s*\(',
            r'open\s*\(', r'os\.', r'sys\.', r'subprocess', r'socket',
            r'globals\s*\(', r'locals\s*\(', r'getattr\s*\(', r'setattr\s*\(',
            r'delattr\s*\(', r'compile\s*\(', r'__\w+__',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return True
        return False


# ============================================================================
# PART 8: TEST GENERATOR
# ============================================================================

class TestGenerator:
    """Automatically generates test cases for functions"""
    
    def __init__(self, specs: Optional[Dict[str, Any]] = None):
        self.specs = specs or {}
    
    def generate_tests(self, num_tests: int = 10,
                      input_type: Optional[str] = None,
                      output_type: Optional[str] = None) -> List[Tuple[Any, Any]]:
        """Generate test cases based on specifications"""
        tests = []
        
        in_type = input_type or self.specs.get('input_type', 'int')
        out_type = output_type or self.specs.get('output_type', 'int')
        
        for _ in range(num_tests):
            test_input = self._generate_input(in_type)
            expected = self._generate_output(out_type)
            tests.append((test_input, expected))
        
        return tests
    
    def generate_edge_cases(self, input_type: str = 'int') -> List[Tuple[Any, Any]]:
        """Generate edge case tests"""
        if input_type == 'int':
            return [
                (0, None), (1, None), (-1, None),
                (2**31 - 1, None), (-2**31, None),
                (100, None), (-100, None),
            ]
        elif input_type == 'list':
            return [
                ([], None), ([1], None), ([1, 2, 3], None),
                ([1] * 100, None), (list(range(1000)), None),
                ([-1, 0, 1], None),
            ]
        elif input_type == 'str':
            return [
                ("", None), ("a", None), ("abc", None),
                ("a" * 1000, None), ("hello world", None),
            ]
        return [(None, None)]
    
    def _generate_input(self, input_type: str) -> Any:
        """Generate a random input of specified type"""
        if input_type == 'int':
            return random.randint(-100, 100)
        elif input_type == 'float':
            return random.uniform(-100.0, 100.0)
        elif input_type == 'list':
            length = random.randint(0, 10)
            return [random.randint(-50, 50) for _ in range(length)]
        elif input_type == 'str':
            length = random.randint(1, 15)
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        elif input_type == 'bool':
            return random.choice([True, False])
        return None
    
    def _generate_output(self, output_type: str) -> Any:
        """Generate a random expected output (placeholder)"""
        if output_type == 'int':
            return random.randint(-100, 100)
        elif output_type == 'float':
            return random.uniform(-100.0, 100.0)
        elif output_type == 'list':
            length = random.randint(0, 10)
            return [random.randint(-50, 50) for _ in range(length)]
        elif output_type == 'str':
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 10)))
        elif output_type == 'bool':
            return random.choice([True, False])
        return None
    
    def infer_types_from_examples(self, examples: List[Tuple[Any, Any]]) -> Dict[str, str]:
        """Infer input/output types from examples"""
        if not examples:
            return {'input_type': 'int', 'output_type': 'int'}
        
        input_types = set()
        output_types = set()
        
        for inp, out in examples:
            if isinstance(inp, int):
                input_types.add('int')
            elif isinstance(inp, float):
                input_types.add('float')
            elif isinstance(inp, list):
                input_types.add('list')
            elif isinstance(inp, str):
                input_types.add('str')
            elif isinstance(inp, bool):
                input_types.add('bool')
            
            if isinstance(out, int):
                output_types.add('int')
            elif isinstance(out, float):
                output_types.add('float')
            elif isinstance(out, list):
                output_types.add('list')
            elif isinstance(out, str):
                output_types.add('str')
            elif isinstance(out, bool):
                output_types.add('bool')
        
        return {
            'input_type': input_types.pop() if len(input_types) == 1 else 'any',
            'output_type': output_types.pop() if len(output_types) == 1 else 'any'
        }


# ============================================================================
# PART 9: PARETO FRONT
# ============================================================================

class ParetoFront:
    """Manages Pareto front for multi-objective optimization"""
    
    def __init__(self, objectives: List[str], max_size: int = 10):
        self.objectives = objectives
        self.max_size = max_size
        self.front: List[ASTIndividual] = []
    
    def update(self, individual: ASTIndividual) -> bool:
        """Add individual to Pareto front if non-dominated. Returns True if added."""
        # Check if dominated by any existing individual
        for existing in self.front:
            if self._dominates(existing, individual):
                individual.is_dominated = True
                return False
        
        # Remove individuals dominated by new one
        self.front = [ind for ind in self.front if not self._dominates(individual, ind)]
        
        # Add new individual
        self.front.append(individual)
        individual.is_dominated = False
        
        # Maintain front size using crowding distance
        if len(self.front) > self.max_size:
            self._calculate_crowding_distance()
            self.front.sort(key=lambda x: x.crowding_distance, reverse=True)
            self.front = self.front[:self.max_size]
        
        return True
    
    def _dominates(self, ind1: ASTIndividual, ind2: ASTIndividual) -> bool:
        """Check if ind1 dominates ind2 (all objectives >=, at least one >)"""
        dominates = False
        for obj in self.objectives:
            val1 = ind1.fitness_components.get(obj, 0)
            val2 = ind2.fitness_components.get(obj, 0)
            
            if val1 < val2:
                return False
            elif val1 > val2:
                dominates = True
        
        return dominates
    
    def _calculate_crowding_distance(self):
        """Calculate crowding distance for diversity preservation"""
        for ind in self.front:
            ind.crowding_distance = 0.0
        
        if len(self.front) <= 2:
            for ind in self.front:
                ind.crowding_distance = float('inf')
            return
        
        for obj in self.objectives:
            sorted_front = sorted(self.front, key=lambda x: x.fitness_components.get(obj, 0))
            
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')
            
            obj_range = sorted_front[-1].fitness_components.get(obj, 0) - sorted_front[0].fitness_components.get(obj, 0)
            
            if obj_range > 0:
                for i in range(1, len(sorted_front) - 1):
                    prev_val = sorted_front[i-1].fitness_components.get(obj, 0)
                    next_val = sorted_front[i+1].fitness_components.get(obj, 0)
                    sorted_front[i].crowding_distance += (next_val - prev_val) / obj_range
    
    def get_best(self) -> Optional[ASTIndividual]:
        """Get the best individual from front (highest crowding distance)"""
        if not self.front:
            return None
        self._calculate_crowding_distance()
        return max(self.front, key=lambda x: x.crowding_distance)
    
    def get_diverse_set(self, n: int) -> List[ASTIndividual]:
        """Get a diverse set of n individuals from front"""
        if len(self.front) <= n:
            return self.front.copy()
        
        self._calculate_crowding_distance()
        sorted_front = sorted(self.front, key=lambda x: x.crowding_distance, reverse=True)
        return sorted_front[:n]
    
    def clear(self):
        """Clear the front"""
        self.front.clear()


# ============================================================================
# PART 10: MCTS NODE AND INTEGRATION
# ============================================================================

@dataclass
class MCTSNode:
    """Node in Monte Carlo Tree Search"""
    state: str  # Non-terminal symbol
    expansion: Optional[Dict] = None
    visits: int = 0
    total_value: float = 0.0
    children: List['MCTSNode'] = field(default_factory=list)
    parent: Optional['MCTSNode'] = None
    untried_expansions: List[Dict] = field(default_factory=list)
    
    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0
    
    def ucb1(self, exploration: float = 1.414) -> float:
        """Calculate UCB1 score"""
        if self.visits == 0:
            return float('inf')
        
        parent_visits = self.parent.visits if self.parent else 1
        exploit = self.value
        explore = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploit + explore
    
    def best_child(self, exploration: float = 1.414) -> Optional['MCTSNode']:
        """Get child with highest UCB1 score"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1(exploration))
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_expansions) == 0
    
    def is_terminal(self) -> bool:
        return self.state in ['literal', 'identifier', 'number', 'bool_literal', 'no_args', 'break_stmt', 'continue_stmt']


class MCTSIntegrator:
    """Integrates MCTS with Genetic Programming for guided generation"""
    
    def __init__(self, grammar: ASTGrammar, config: AIConfig):
        self.grammar = grammar
        self.config = config
        self.root: Optional[MCTSNode] = None
        self.stats: Dict[str, Dict[str, float]] = {}
        self.lock = threading.Lock()
    
    def initialize(self, start_symbol: str = 'program'):
        """Initialize MCTS tree"""
        expansions = self.grammar.get_valid_expansions(start_symbol)
        self.root = MCTSNode(
            state=start_symbol,
            untried_expansions=expansions.copy()
        )
    
    def select_expansion(self, symbol: str, depth: int = 0) -> Dict:
        """Select expansion using MCTS statistics"""
        expansions = self.grammar.get_valid_expansions(symbol)
        if not expansions:
            return {'type': 'literal', 'options': [0]}
        
        # Use accumulated stats for UCB1
        best_expansion = None
        best_ucb = -float('inf')
        total_visits = sum(self.stats.get(str(exp), {}).get('visits', 0) for exp in expansions)
        
        for expansion in expansions:
            exp_key = str(expansion)
            stats = self.stats.get(exp_key, {'visits': 0, 'value': 0.0})
            
            visits = stats['visits']
            value = stats['value'] / visits if visits > 0 else 0.5
            
            if visits == 0:
                ucb = float('inf')
            else:
                exploration = self.config.mcts_exploration
                ucb = value + exploration * math.sqrt(math.log(max(total_visits, 1)) / visits)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_expansion = expansion
        
        return best_expansion if best_expansion else expansions[0]
    
    def update_stats(self, expansion: Dict, reward: float):
        """Update statistics after evaluation"""
        with self.lock:
            exp_key = str(expansion)
            if exp_key not in self.stats:
                self.stats[exp_key] = {'visits': 0, 'value': 0.0}
            
            self.stats[exp_key]['visits'] += 1
            # Running average
            old_value = self.stats[exp_key]['value']
            visits = self.stats[exp_key]['visits']
            self.stats[exp_key]['value'] = old_value + (reward - old_value) / visits
    
    def get_stats_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of MCTS statistics"""
        with self.lock:
            return self.stats.copy()
    
    def clear_stats(self):
        """Clear all statistics"""
        with self.lock:
            self.stats.clear()


# ============================================================================
# PART 11: CODE REFACTORING ENGINE
# ============================================================================

class CodeRefactorer:
    """Refactors generated code for better quality"""
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.refactoring_rules = [
            self._remove_redundant_parens,
            self._simplify_arithmetic,
            self._simplify_comparisons,
            self._remove_dead_code,
            self._simplify_boolean_expressions,
        ]
    
    def refactor(self, code: str) -> str:
        """Apply all refactoring rules"""
        refactored = code
        for _ in range(self.config.refactoring_passes):
            new_code = refactored
            for rule in self.refactoring_rules:
                new_code = rule(new_code)
            if new_code == refactored:
                break
            refactored = new_code
        return refactored
    
    def _remove_redundant_parens(self, code: str) -> str:
        """Remove unnecessary parentheses"""
        # Simple pattern-based removal
        patterns = [
            (r'\(\(([^\(\)]+)\)\)', r'(\1)'),  # ((x)) -> (x)
            (r'\((\d+)\)', r'\1'),  # (5) -> 5
            (r'\((\'[^\']*\')\)', r'\1'),  # ('str') -> 'str'
        ]
        
        result = code
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        return result
    
    def _simplify_arithmetic(self, code: str) -> str:
        """Simplify arithmetic expressions"""
        patterns = [
            (r'\(x \+ 0\)', 'x'),
            (r'\(0 \+ x\)', 'x'),
            (r'\(x \* 1\)', 'x'),
            (r'\(1 \* x\)', 'x'),
            (r'\(x \* 0\)', '0'),
            (r'\(0 \* x\)', '0'),
            (r'\(x \- 0\)', 'x'),
            (r'\(x \/ 1\)', 'x'),
            (r'\(x \*\* 1\)', 'x'),
            (r'\(x \*\* 0\)', '1'),
        ]
        
        result = code
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        return result
    
    def _simplify_comparisons(self, code: str) -> str:
        """Simplify comparison expressions"""
        patterns = [
            (r'\(x == x\)', 'True'),
            (r'\(x != x\)', 'False'),
            (r'\(x <= x\)', 'True'),
            (r'\(x >= x\)', 'True'),
        ]
        
        result = code
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        return result
    
    def _remove_dead_code(self, code: str) -> str:
        """Remove obviously dead code"""
        # Remove unreachable return after return
        lines = code.split('\n')
        result = []
        after_return = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('return '):
                after_return = True
                result.append(line)
            elif after_return and stripped and not stripped.startswith('#'):
                continue
            else:
                after_return = False
                result.append(line)
        
        return '\n'.join(result)
    
    def _simplify_boolean_expressions(self, code: str) -> str:
        """Simplify boolean expressions"""
        patterns = [
            (r'\(True and x\)', 'x'),
            (r'\(x and True\)', 'x'),
            (r'\(False and x\)', 'False'),
            (r'\(x and False\)', 'False'),
            (r'\(True or x\)', 'True'),
            (r'\(x or True\)', 'True'),
            (r'\(False or x\)', 'x'),
            (r'\(x or False\)', 'x'),
            (r'\(not True\)', 'False'),
            (r'\(not False\)', 'True'),
        ]
        
        result = code
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        return result


# ============================================================================
# PART 12: CONSTRAINT SOLVER
# ============================================================================

class ConstraintSolver:
    """Constraint solver for CSS generation using backtracking with arc consistency"""
    
    def __init__(self):
        self.constraints: List[Dict] = []
        self.variables: Dict[str, List[Any]] = {}  # variable -> domain
        self.solutions: List[Dict[str, Any]] = []
    
    def add_variable(self, name: str, domain: List[Any]):
        """Add a variable with its domain"""
        self.variables[name] = domain.copy()
    
    def add_constraint(self, var1: str, op: str, var2_or_value: Any):
        """Add a constraint"""
        self.constraints.append({
            'var1': var1,
            'op': op,
            'var2': var2_or_value,
            'is_literal': var2_or_value not in self.variables
        })
    
    def solve(self, max_solutions: int = 1) -> List[Dict[str, Any]]:
        """Solve constraints using backtracking with forward checking"""
        self.solutions = []
        assignment: Dict[str, Any] = {}
        domains = {k: v.copy() for k, v in self.variables.items()}
        
        self._backtrack(assignment, domains, list(self.variables.keys()), 0, max_solutions)
        return self.solutions
    
    def _backtrack(self, assignment: Dict[str, Any], domains: Dict[str, List[Any]],
                   var_order: List[str], idx: int, max_solutions: int) -> bool:
        """Backtracking search with forward checking"""
        if len(self.solutions) >= max_solutions:
            return True
        
        if idx >= len(var_order):
            self.solutions.append(assignment.copy())
            return True
        
        var = var_order[idx]
        if var in assignment:
            return self._backtrack(assignment, domains, var_order, idx + 1, max_solutions)
        
        for value in domains[var]:
            if self._is_consistent(var, value, assignment):
                assignment[var] = value
                
                # Forward checking: reduce domains of future variables
                saved_domains = {k: v.copy() for k, v in domains.items()}
                if self._forward_check(var, value, domains, var_order[idx+1:]):
                    if self._backtrack(assignment, domains, var_order, idx + 1, max_solutions):
                        if len(self.solutions) >= max_solutions:
                            return True
                
                # Restore domains
                domains.update(saved_domains)
                del assignment[var]
        
        return False
    
    def _is_consistent(self, var: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """Check if assigning value to var is consistent with current assignment"""
        for constraint in self.constraints:
            if constraint['var1'] == var:
                other = constraint['var2']
                op = constraint['op']
                
                if constraint['is_literal']:
                    if not self._satisfies(value, op, other):
                        return False
                elif other in assignment:
                    if not self._satisfies(value, op, assignment[other]):
                        return False
            
            elif not constraint['is_literal'] and constraint['var2'] == var:
                other = constraint['var1']
                op = constraint['op']
                
                if other in assignment:
                    # Reverse the comparison
                    reverse_op = self._reverse_op(op)
                    if not self._satisfies(assignment[other], reverse_op, value):
                        return False
        
        return True
    
    def _forward_check(self, var: str, value: Any, domains: Dict[str, List[Any]],
                       future_vars: List[str]) -> bool:
        """Reduce domains of future variables based on current assignment"""
        for future_var in future_vars:
            if future_var in domains:
                new_domain = []
                for future_val in domains[future_var]:
                    # Check all constraints involving both var and future_var
                    consistent = True
                    for constraint in self.constraints:
                        if (constraint['var1'] == var and constraint['var2'] == future_var) or \
                           (constraint['var1'] == future_var and constraint['var2'] == var):
                            op = constraint['op']
                            if constraint['var1'] == var:
                                if not self._satisfies(value, op, future_val):
                                    consistent = False
                                    break
                            else:
                                reverse_op = self._reverse_op(op)
                                if not self._satisfies(future_val, reverse_op, value):
                                    consistent = False
                                    break
                    
                    if consistent:
                        new_domain.append(future_val)
                
                domains[future_var] = new_domain
                if not new_domain:
                    return False
        
        return True
    
    def _satisfies(self, value1: Any, op: str, value2: Any) -> bool:
        """Check if value1 op value2 is satisfied"""
        cmp = self._compare(value1, value2)
        
        if op == '==':
            return cmp == 0
        elif op == '!=':
            return cmp != 0
        elif op == '<':
            return cmp < 0
        elif op == '>':
            return cmp > 0
        elif op == '<=':
            return cmp <= 0
        elif op == '>=':
            return cmp >= 0
        return False
    
    def _compare(self, v1: Any, v2: Any) -> int:
        """Compare two values, extracting numbers from CSS values"""
        num1 = self._extract_number(v1)
        num2 = self._extract_number(v2)
        
        if num1 is not None and num2 is not None:
            return (num1 > num2) - (num1 < num2)
        
        return (str(v1) > str(v2)) - (str(v1) < str(v2))
    
    def _extract_number(self, value: Any) -> Optional[float]:
        """Extract numeric value from CSS value"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r'[\d.]+', value)
            if match:
                return float(match.group())
        return None
    
    def _reverse_op(self, op: str) -> str:
        """Reverse comparison operator"""
        reversals = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '==': '==', '!=': '!='}
        return reversals.get(op, op)


# ============================================================================
# PART 13: CSS GENERATOR
# ============================================================================

class CSSGenerator:
    """Generates CSS using constraint solving and fuzzy specifications"""
    
    COLOR_PALETTE = {
        'primary': ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#1abc9c', '#f39c12'],
        'secondary': ['#2980b9', '#27ae60', '#c0392b', '#8e44ad', '#16a085', '#e67e22'],
        'neutral': ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7', '#ecf0f1'],
        'background': ['#ffffff', '#f5f5f5', '#f8f9fa', '#e9ecef', '#dee2e6'],
        'text': ['#212529', '#495057', '#6c757d', '#343a40']
    }
    
    SIZE_SCALE = {
        'xs': ['2px', '4px', '8px'],
        'sm': ['8px', '12px', '16px'],
        'md': ['16px', '20px', '24px'],
        'lg': ['24px', '32px', '48px'],
        'xl': ['48px', '64px', '96px']
    }
    
    def __init__(self, fuzzy_specs: Optional[Dict[str, Any]] = None):
        self.specs = fuzzy_specs or {}
        self.solver = ConstraintSolver()
    
    def generate(self, style_type: str = "modern") -> str:
        """Generate CSS based on specifications"""
        complexity = self.specs.get('complexity', 5.0)
        formality = self.specs.get('formality', 5.0)
        color_scheme = self.specs.get('color_scheme', 'blue')
        
        # Set up constraint solver
        self._setup_constraints(complexity, formality)
        
        # Solve constraints
        solutions = self.solver.solve(max_solutions=1)
        solution = solutions[0] if solutions else {}
        
        # Generate CSS
        if style_type == "modern":
            return self._generate_modern_css(solution, complexity)
        elif style_type == "minimal":
            return self._generate_minimal_css(solution)
        elif style_type == "corporate":
            return self._generate_corporate_css(solution, formality)
        else:
            return self._generate_modern_css(solution, complexity)
    
    def _setup_constraints(self, complexity: float, formality: float):
        """Set up CSS constraints"""
        # Colors
        primary_color = self._get_primary_color()
        self.solver.add_variable('primary_color', self.COLOR_PALETTE['primary'])
        self.solver.add_variable('text_color', self.COLOR_PALETTE['text'])
        self.solver.add_variable('bg_color', self.COLOR_PALETTE['background'])
        
        # Sizes based on complexity
        size_key = 'lg' if complexity > 7 else ('md' if complexity > 4 else 'sm')
        radius_sizes = self.SIZE_SCALE['sm'] if formality > 6 else self.SIZE_SCALE['md']
        padding_sizes = self.SIZE_SCALE['md'] if complexity > 6 else self.SIZE_SCALE['sm']
        
        self.solver.add_variable('border_radius', radius_sizes)
        self.solver.add_variable('padding', padding_sizes)
        self.solver.add_variable('margin', self.SIZE_SCALE['sm'])
        self.solver.add_variable('font_size', ['14px', '16px', '18px', '20px'])
        
        # Constraints
        if complexity > 5:
            self.solver.add_constraint('border_radius', '>=', '8px')
            self.solver.add_constraint('padding', '>=', '16px')
    
    def _get_primary_color(self) -> str:
        """Get primary color based on specifications"""
        scheme = self.specs.get('color_scheme', 'blue')
        color_map = {
            'blue': '#3498db', 'green': '#2ecc71', 'red': '#e74c3c',
            'purple': '#9b59b6', 'teal': '#1abc9c', 'orange': '#f39c12'
        }
        return color_map.get(scheme, '#3498db')
    
    def _generate_modern_css(self, solution: Dict[str, Any], complexity: float) -> str:
        """Generate modern CSS"""
        primary = solution.get('primary_color', '#3498db')
        radius = solution.get('border_radius', '8px')
        padding = solution.get('padding', '16px')
        text_color = solution.get('text_color', '#212529')
        bg_color = solution.get('bg_color', '#ffffff')
        
        css = f"""/* Generated Modern CSS */
:root {{
    --primary-color: {primary};
    --text-color: {text_color};
    --bg-color: {bg_color};
    --border-radius: {radius};
    --padding: {padding};
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background-color: var(--bg-color);
    color: var(--text-color);
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--padding);
}}

.card {{
    background: white;
    padding: var(--padding);
    border-radius: var(--border-radius);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    transition: transform 0.2s, box-shadow 0.2s;
}}

.card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}}

.btn {{
    display: inline-block;
    background-color: var(--primary-color);
    color: white;
    padding: 12px 24px;
    border: none;
    border-radius: var(--border-radius);
    cursor: pointer;
    font-size: 16px;
    transition: all 0.3s ease;
}}

.btn:hover {{
    opacity: 0.9;
    transform: translateY(-1px);
}}

.btn:active {{
    transform: translateY(1px);
}}
"""
        if complexity > 6:
            css += f"""
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--padding);
}}

.flex {{
    display: flex;
    gap: var(--padding);
    align-items: center;
}}
"""
        return css
    
    def _generate_minimal_css(self, solution: Dict[str, Any]) -> str:
        """Generate minimal CSS"""
        text_color = solution.get('text_color', '#212529')
        bg_color = solution.get('bg_color', '#ffffff')
        
        return f"""/* Generated Minimal CSS */
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: system-ui, sans-serif;
    background: {bg_color};
    color: {text_color};
    line-height: 1.5;
}}

.container {{
    max-width: 960px;
    margin: 0 auto;
    padding: 20px;
}}
"""
    
    def _generate_corporate_css(self, solution: Dict[str, Any], formality: float) -> str:
        """Generate corporate/formal CSS"""
        primary = solution.get('primary_color', '#3498db')
        text_color = solution.get('text_color', '#212529')
        radius = '4px' if formality > 6 else '8px'
        
        return f"""/* Generated Corporate CSS */
:root {{
    --primary: {primary};
    --primary-dark: {self._darken_color(primary)};
    --text: {text_color};
    --border-radius: {radius};
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #ffffff;
    color: var(--text);
    line-height: 1.6;
}}

.header {{
    background: var(--primary);
    color: white;
    padding: 20px 40px;
}}

.btn {{
    background: var(--primary);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: var(--border-radius);
    cursor: pointer;
    font-weight: 500;
}}

.btn:hover {{
    background: var(--primary-dark);
}}

.card {{
    border: 1px solid #e0e0e0;
    border-radius: var(--border-radius);
    padding: 24px;
    margin-bottom: 16px;
}}
"""
    
    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color by 20%"""
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            
            r = int(r * 0.8)
            g = int(g * 0.8)
            b = int(b * 0.8)
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return hex_color


# ============================================================================
# PART 14: GRAMMAR INDUCTION
# ============================================================================

class GrammarInducer:
    """Induces grammar rules from code examples"""
    
    def __init__(self, grammar: ASTGrammar):
        self.grammar = grammar
        self.pattern_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.sequence_patterns: Dict[str, List[List[str]]] = defaultdict(list)
    
    def learn_from_examples(self, code_examples: List[str]):
        """Learn patterns from code examples"""
        for code in code_examples:
            try:
                tree = ast.parse(code)
                self._extract_patterns(tree)
            except SyntaxError:
                continue
    
    def _extract_patterns(self, tree: ast.AST):
        """Extract patterns from AST"""
        for node in ast.walk(tree):
            self._analyze_node(node)
    
    def _analyze_node(self, node: ast.AST):
        """Analyze a single AST node"""
        node_type = type(node).__name__
        
        # Count child types
        child_types = []
        for child in ast.iter_child_nodes(node):
            child_type = type(child).__name__
            child_types.append(child_type)
            self.pattern_counts[node_type][child_type] += 1
        
        # Record sequences
        if child_types:
            self.sequence_patterns[node_type].append(child_types)
    
    def get_common_patterns(self, min_count: int = 2) -> Dict[str, List[Tuple[str, int]]]:
        """Get commonly observed patterns"""
        common = {}
        for node_type, child_counts in self.pattern_counts.items():
            patterns = [(child, count) for child, count in child_counts.items() if count >= min_count]
            patterns.sort(key=lambda x: x[1], reverse=True)
            if patterns:
                common[node_type] = patterns[:5]  # Top 5
        return common
    
    def suggest_grammar_enhancements(self) -> List[Dict]:
        """Suggest grammar rule enhancements based on learned patterns"""
        suggestions = []
        patterns = self.get_common_patterns()
        
        for node_type, child_patterns in patterns.items():
            for child_type, count in child_patterns:
                if child_type not in [str(exp.get('type', '')) for exp in self.grammar.get_valid_expansions(node_type, []) if isinstance(exp, dict)]:
                    suggestions.append({
                        'parent': node_type,
                        'child': child_type,
                        'count': count,
                        'suggestion': f"Consider adding {child_type} as child of {node_type}"
                    })
        
        return suggestions


# ============================================================================
# PART 15: MAIN GENETIC PROGRAMMING ENGINE
# ============================================================================

class MCTSIntegratedGP:
    """Genetic Programming with deep MCTS integration"""
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.grammar = ASTGrammar()
        self.executor = SandboxedExecutor(self.config.max_execution_time)
        self.refactorer = CodeRefactorer(self.config)
        self.mcts = MCTSIntegrator(self.grammar, self.config)
        self.pareto_front = ParetoFront(self.config.pareto_objectives, self.config.pareto_size)
        self.fitness_cache = FitnessCache(self.config.cache_size)
        self.test_generator = TestGenerator()
        self.grammar_inducer = GrammarInducer(self.grammar)
        
        self.population: List[ASTIndividual] = []
        self.best_individual: Optional[ASTIndividual] = None
        self.generation_stats: List[Dict] = []
        self.evolution_history: List[List[ASTIndividual]] = []
        
        # Initialize MCTS
        self.mcts.initialize('program')
    
    def initialize_population(self):
        """Create initial population"""
        self.population = []
        for _ in range(self.config.population_size):
            try:
                individual = ASTIndividual()
                individual.ast_root = self._generate_ast_simple('program', depth=0, max_depth=self.config.max_function_depth)
                individual.code_string = individual.to_code()
                individual.calculate_metrics()
                self.population.append(individual)
            except Exception as e:
                logger.debug(f"Failed to generate individual: {e}")
                individual = ASTIndividual()
                individual.ast_root = self._create_fallback_ast()
                individual.code_string = individual.to_code()
                individual.calculate_metrics()
                self.population.append(individual)
    
    def _generate_ast_simple(self, symbol: str, depth: int, max_depth: int) -> ASTNode:
        """Simple AST generation without MCTS (for initial population)"""
        if depth >= max_depth:
            return self._force_terminal(symbol)
        
        expansions = self.grammar.get_valid_expansions(symbol)
        if not expansions:
            return self._force_terminal(symbol)
        
        # Use weighted selection
        expansion = self.grammar.get_weighted_expansion(symbol)
        if not expansion:
            expansion = expansions[0]
        
        node = ASTNode(expansion['type'])
        
        if 'options' in expansion:
            if isinstance(expansion['options'][0], list):
                # Options is a list of lists (e.g., param_list)
                for item in random.choice(expansion['options']):
                    child = ASTNode('var', item)
                    node.children.append(child)
            else:
                node.value = random.choice(expansion['options'])
            return node
        
        if 'children' in expansion:
            for child_symbol in expansion['children']:
                try:
                    child_max_depth = self.grammar.get_max_depth(child_symbol, max_depth - depth)
                    child_node = self._generate_ast_simple(child_symbol, depth + 1, depth + child_max_depth)
                    node.children.append(child_node)
                except Exception:
                    node.children.append(self._force_terminal(child_symbol))
        
        return node
    
    def _generate_ast_with_mcts(self, symbol: str, depth: int, max_depth: int) -> ASTNode:
        """Generate AST with MCTS guidance"""
        if depth >= max_depth:
            return self._force_terminal(symbol)
        
        expansions = self.grammar.get_valid_expansions(symbol)
        if not expansions:
            return self._force_terminal(symbol)
        
        # Use MCTS to select expansion
        expansion = self.mcts.select_expansion(symbol, depth)
        if not expansion:
            expansion = self.grammar.get_weighted_expansion(symbol) or expansions[0]
        
        node = ASTNode(expansion['type'])
        
        if 'options' in expansion:
            if isinstance(expansion['options'][0], list):
                for item in random.choice(expansion['options']):
                    child = ASTNode('var', item)
                    node.children.append(child)
            else:
                node.value = random.choice(expansion['options'])
            return node
        
        if 'children' in expansion:
            for child_symbol in expansion['children']:
                try:
                    child_max_depth = self.grammar.get_max_depth(child_symbol, max_depth - depth)
                    child_node = self._generate_ast_with_mcts(child_symbol, depth + 1, depth + child_max_depth)
                    node.children.append(child_node)
                except Exception:
                    node.children.append(self._force_terminal(child_symbol))
        
        return node
    
    def _create_fallback_ast(self) -> ASTNode:
        """Create a simple fallback AST: def func(x): return x"""
        root = ASTNode('function')
        name_node = ASTNode('var', 'func')
        
        params_node = ASTNode('param_list')
        params_node.children.append(ASTNode('var', 'x'))
        
        return_stmt = ASTNode('return_stmt')
        var_node = ASTNode('variable')
        var_node.children.append(ASTNode('var', 'x'))
        return_stmt.children.append(var_node)
        
        root.children = [name_node, params_node, return_stmt]
        return root
    
    def _force_terminal(self, symbol: str) -> ASTNode:
        """Force generation of terminal node"""
        if symbol == 'number':
            return ASTNode('literal', random.choice([0, 1, 2, 5, 10, -1]))
        elif symbol == 'identifier':
            return ASTNode('var', random.choice(['x', 'n', 'i', 'result', 'total']))
        elif symbol == 'operator':
            return ASTNode('operator', random.choice(['+', '-', '*', '/']))
        elif symbol == 'comparison_op':
            return ASTNode('comparison_op', random.choice(['==', '!=', '<', '>']))
        elif symbol == 'logical_op':
            return ASTNode('logical_op', 'and')
        elif symbol == 'bool_literal':
            return ASTNode('literal', random.choice([True, False]))
        elif symbol == 'no_args':
            return ASTNode('no_args')
        elif symbol == 'break_stmt':
            return ASTNode('break_stmt')
        elif symbol == 'continue_stmt':
            return ASTNode('continue_stmt')
        return ASTNode('literal', 0)
    
    def evaluate_fitness(self, individual: ASTIndividual, test_cases: List[Tuple[Any, Any]]) -> float:
        """Evaluate individual with multi-objective fitness"""
        if not individual.code_string:
            individual.fitness = 0.0
            return 0.0
        
        # Check cache
        cached = self.fitness_cache.get(individual.code_string)
        if cached:
            individual.fitness = cached['total']
            individual.fitness_components = cached['components'].copy()
            individual.test_results = cached['test_results'].copy()
            return individual.fitness
        
        try:
            # Test correctness
            correctness_score = 0.0
            test_results = []
            
            for test_input, expected in test_cases:
                try:
                    result, error = self.executor.execute(individual.code_string, test_input)
                    
                    if error:
                        test_results.append(False)
                        continue
                    
                    if result == expected:
                        correctness_score += 1.0
                        test_results.append(True)
                    elif isinstance(result, (int, float)) and isinstance(expected, (int, float)):
                        diff = abs(result - expected)
                        max_val = max(abs(expected), 1)
                        score = max(0, 1.0 - diff / max_val)
                        correctness_score += score
                        test_results.append(score > 0.5)
                    else:
                        test_results.append(False)
                except Exception:
                    test_results.append(False)
            
            correctness_score /= len(test_cases) if test_cases else 1
            
            # Calculate other metrics
            metrics = individual.calculate_metrics()
            complexity = metrics['complexity']
            size = metrics['size']
            
            # Normalize scores
            simplicity_score = max(0, 1.0 - complexity / 20.0)
            size_score = max(0, 1.0 - size / self.config.max_function_size)
            
            # Store components
            individual.fitness_components = {
                'correctness': correctness_score,
                'simplicity': simplicity_score,
                'size': size_score
            }
            individual.test_results = test_results
            
            # Update Pareto front
            if self.config.use_pareto:
                self.pareto_front.update(individual)
            
            # Calculate total fitness
            weights = self.config.fitness_weights
            fitness = (
                weights.get('correctness', 0.7) * correctness_score +
                weights.get('simplicity', 0.2) * simplicity_score +
                weights.get('size', 0.1) * size_score
            )
            
            individual.fitness = fitness
            
            # Cache result
            self.fitness_cache.put(individual.code_string, {
                'total': fitness,
                'components': individual.fitness_components,
                'test_results': test_results
            })
            
            # Update MCTS stats for successful expansions
            if fitness > 0.5:
                self._update_mcts_from_individual(individual, fitness)
            
            return fitness
        
        except Exception as e:
            individual.fitness = 0.0
            individual.error = str(e)
            return 0.0
    
    def _update_mcts_from_individual(self, individual: ASTIndividual, reward: float):
        """Update MCTS statistics based on individual's performance"""
        if not individual.ast_root:
            return
        
        for node in individual.ast_root.get_all_nodes():
            # Create a pseudo-expansion from the node
            expansion = {'type': node.node_type}
            if node.value is not None:
                expansion['value'] = node.value
            self.mcts.update_stats(expansion, reward)
    
    def crossover(self, parent1: ASTIndividual, parent2: ASTIndividual) -> ASTIndividual:
        """AST-aware crossover with type checking"""
        child = ASTIndividual()
        child.parent_ids = {parent1.id, parent2.id}
        
        if not parent1.ast_root or not parent2.ast_root:
            child.ast_root = parent1.ast_root.deep_copy() if parent1.ast_root else self._create_fallback_ast()
            child.code_string = child.to_code()
            return child
        
        try:
            nodes1 = parent1.ast_root.get_all_nodes()
            nodes2 = parent2.ast_root.get_all_nodes()
            
            if not nodes1 or not nodes2:
                child.ast_root = parent1.ast_root.deep_copy()
                child.code_string = child.to_code()
                return child
            
            # Try type-compatible crossover
            for _ in range(10):  # Max attempts
                node1 = random.choice(nodes1)
                node1_type = TypeSystem.infer_type(node1)
                
                compatible_nodes = TypeSystem.get_compatible_nodes(node1_type, nodes2)
                
                if compatible_nodes:
                    node2 = random.choice(compatible_nodes)
                    child.ast_root = parent1.ast_root.deep_copy()
                    self._replace_subtree(child.ast_root, node1, node2.deep_copy())
                    break
            else:
                # Fallback to random crossover
                node1 = random.choice(nodes1)
                node2 = random.choice(nodes2)
                child.ast_root = parent1.ast_root.deep_copy()
                self._replace_subtree(child.ast_root, node1, node2.deep_copy())
            
            child.code_string = child.to_code()
        except Exception as e:
            logger.debug(f"Crossover failed: {e}")
            child.ast_root = parent1.ast_root.deep_copy()
            child.code_string = child.to_code()
        
        return child
    
    def mutate(self, individual: ASTIndividual) -> ASTIndividual:
        """AST-aware mutation with semantic preservation"""
        if random.random() > self.config.mutation_rate or not individual.ast_root:
            return individual
        
        try:
            mutation_type = random.choices(
                ['point', 'subtree', 'hoist', 'swap'],
                weights=[0.4, 0.3, 0.2, 0.1],
                k=1
            )[0]
            
            if mutation_type == 'point':
                self._point_mutation(individual)
            elif mutation_type == 'subtree':
                self._subtree_mutation(individual)
            elif mutation_type == 'hoist':
                self._hoist_mutation(individual)
            elif mutation_type == 'swap':
                self._swap_mutation(individual)
            
            individual.code_string = individual.to_code()
        except Exception as e:
            logger.debug(f"Mutation failed: {e}")
        
        return individual
    
    def _point_mutation(self, individual: ASTIndividual):
        """Mutate a single node's value"""
        nodes = individual.ast_root.get_all_nodes()
        if not nodes:
            return
        
        node = random.choice(nodes)
        
        if node.node_type == 'literal' and isinstance(node.value, (int, float)):
            # Small perturbation
            delta = random.choice([-1, 1, -2, 2, -5, 5, -10, 10])
            node.value = node.value + delta if isinstance(node.value, int) else node.value + delta * 0.1
        elif node.node_type == 'operator':
            node.value = random.choice(['+', '-', '*', '/', '%', '//', '**'])
        elif node.node_type == 'comparison_op':
            node.value = random.choice(['==', '!=', '<', '>', '<=', '>='])
        elif node.node_type == 'logical_op':
            node.value = random.choice(['and', 'or'])
    
    def _subtree_mutation(self, individual: ASTIndividual):
        """Replace a subtree with a newly generated one"""
        nodes = individual.ast_root.get_all_nodes()
        if not nodes:
            return
        
        # Prefer expression nodes for subtree mutation
        expr_nodes = [n for n in nodes if n.node_type in ['expression', 'binary_op', 'func_call', 'variable', 'literal']]
        if not expr_nodes:
            expr_nodes = nodes
        
        node = random.choice(expr_nodes)
        new_subtree = self._generate_ast_with_mcts('expression', 0, 3)
        self._replace_subtree(individual.ast_root, node, new_subtree)
    
    def _hoist_mutation(self, individual: ASTIndividual):
        """Replace a node with one of its children (simplification)"""
        nodes = individual.ast_root.get_all_nodes()
        if not nodes:
            return
        
        # Only apply to nodes with children
        parent_nodes = [n for n in nodes if n.children]
        if not parent_nodes:
            return
        
        node = random.choice(parent_nodes)
        child = random.choice(node.children)
        self._replace_subtree(individual.ast_root, node, child.deep_copy())
    
    def _swap_mutation(self, individual: ASTIndividual):
        """Swap two sibling nodes"""
        nodes = individual.ast_root.get_all_nodes()
        if not nodes:
            return
        
        # Find nodes with multiple children
        multi_child_nodes = [n for n in nodes if len(n.children) >= 2]
        if not multi_child_nodes:
            return
        
        node = random.choice(multi_child_nodes)
        if len(node.children) >= 2:
            i, j = random.sample(range(len(node.children)), 2)
            node.children[i], node.children[j] = node.children[j], node.children[i]
    
    def _replace_subtree(self, root: ASTNode, old_node: ASTNode, new_node: ASTNode):
        """Replace subtree in AST"""
        if root is old_node:
            root.node_type = new_node.node_type
            root.value = new_node.value
            root.children = new_node.children
            return
        
        for i, child in enumerate(root.children):
            if child is old_node:
                root.children[i] = new_node
                return
            self._replace_subtree(child, old_node, new_node)
    
    def _tournament_select(self, tournament_size: Optional[int] = None) -> ASTIndividual:
        """Tournament selection"""
        size = tournament_size or self.config.tournament_size
        tournament = random.sample(self.population, min(size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity using code hash diversity"""
        if len(self.population) < 2:
            return 1.0
        
        unique_hashes = set()
        for ind in self.population:
            if ind.code_string:
                unique_hashes.add(hashlib.md5(ind.code_string.encode()).hexdigest())
        
        return len(unique_hashes) / len(self.population)
    
    def _adaptive_mutation_rate(self, generation: int) -> float:
        """Adaptively adjust mutation rate based on progress"""
        if len(self.generation_stats) < self.config.stagnation_threshold:
            return self.config.mutation_rate
        
        # Check for stagnation
        recent_stats = self.generation_stats[-self.config.stagnation_threshold:]
        improvement = recent_stats[-1]['best_fitness'] - recent_stats[0]['best_fitness']
        
        if improvement < 0.01:
            # Stagnating - increase mutation
            new_rate = min(self.config.max_mutation_rate, self.config.mutation_rate * 1.2)
        else:
            # Improving - decrease mutation
            new_rate = max(self.config.min_mutation_rate, self.config.mutation_rate * 0.95)
        
        self.config.mutation_rate = new_rate
        return new_rate
    
    def evolve(self, test_cases: List[Tuple[Any, Any]], verbose: Optional[bool] = None) -> ASTIndividual:
        """Main evolution loop"""
        verbose = verbose if verbose is not None else self.config.verbose_output
        
        logger.info(f"Starting evolution: {self.config.population_size} individuals, {self.config.generations} generations")
        
        self.initialize_population()
        start_time = time.time()
        
        for generation in range(self.config.generations):
            gen_start = time.time()
            
            # Evaluate fitness (with optional parallelization)
            if self.config.use_parallel and len(self.population) > 10:
                self._parallel_evaluate(test_cases)
            else:
                for individual in self.population:
                    self.evaluate_fitness(individual, test_cases)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = self.population[0].deep_copy() if hasattr(self.population[0], 'deep_copy') else self._copy_individual(self.population[0])
                if self.config.auto_refactor:
                    self.best_individual.code_string = self.refactorer.refactor(self.best_individual.code_string)
            
            # Calculate statistics
            avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
            diversity = self._calculate_diversity()
            
            stats = {
                'generation': generation,
                'best_fitness': self.population[0].fitness,
                'avg_fitness': avg_fitness,
                'diversity': diversity,
                'pareto_size': len(self.pareto_front.front),
                'mutation_rate': self.config.mutation_rate,
                'gen_time': time.time() - gen_start
            }
            self.generation_stats.append(stats)
            
            # Adaptive mutation
            self._adaptive_mutation_rate(generation)
            
            # Logging
            if verbose and (generation % 20 == 0 or generation == self.config.generations - 1):
                logger.info(
                    f"Gen {generation}: Best={self.population[0].fitness:.3f}, "
                    f"Avg={avg_fitness:.3f}, Diversity={diversity:.2f}, "
                    f"Pareto={len(self.pareto_front.front)}, "
                    f"Mutation={self.config.mutation_rate:.2f}"
                )
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_size = max(1, int(self.config.population_size * self.config.elite_ratio))
            new_population.extend(self.population[:elite_size])
            
            # Fill with offspring
            while len(new_population) < self.config.population_size:
                try:
                    if random.random() < self.config.crossover_rate:
                        parent1 = self._tournament_select()
                        parent2 = self._tournament_select()
                        child = self.crossover(parent1, parent2)
                    else:
                        parent = self._tournament_select()
                        child = self._copy_individual(parent)
                    
                    child = self.mutate(child)
                    child.age = 0
                    new_population.append(child)
                except Exception as e:
                    logger.debug(f"Failed to create offspring: {e}")
                    fallback = ASTIndividual()
                    fallback.ast_root = self._create_fallback_ast()
                    fallback.code_string = fallback.to_code()
                    new_population.append(fallback)
            
            # Age all individuals
            for ind in new_population:
                ind.age += 1
            
            self.population = new_population
            
            # Save evolution history
            if self.config.save_evolution_data:
                self.evolution_history.append([self._copy_individual(ind) for ind in self.population[:5]])
        
        total_time = time.time() - start_time
        logger.info(f"Evolution completed in {total_time:.2f}s")
        logger.info(f"Best fitness: {self.best_individual.fitness:.4f}")
        logger.info(f"Cache stats: {self.fitness_cache.get_stats()}")
        
        return self.best_individual
    
    def _parallel_evaluate(self, test_cases: List[Tuple[Any, Any]]):
        """Evaluate population in parallel"""
        def eval_individual(individual):
            self.evaluate_fitness(individual, test_cases)
            return individual
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(eval_individual, ind): ind for ind in self.population}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Parallel evaluation error: {e}")
    
    def _copy_individual(self, individual: ASTIndividual) -> ASTIndividual:
        """Create a copy of an individual"""
        new_ind = ASTIndividual(individual.ast_root.deep_copy() if individual.ast_root else None)
        new_ind.fitness = individual.fitness
        new_ind.fitness_components = individual.fitness_components.copy()
        new_ind.code_string = individual.code_string
        new_ind.complexity = individual.complexity
        new_ind.size = individual.size
        new_ind.depth = individual.depth
        new_ind.test_results = individual.test_results.copy()
        new_ind.age = individual.age
        new_ind.parent_ids = individual.parent_ids.copy()
        return new_ind
    
    def generate_with_examples(self, examples: List[Tuple[Any, Any]], 
                               input_type: Optional[str] = None,
                               output_type: Optional[str] = None) -> Optional[ASTIndividual]:
        """Generate code from input/output examples"""
        # Infer types if not provided
        type_info = self.test_generator.infer_types_from_examples(examples)
        input_type = input_type or type_info['input_type']
        output_type = output_type or type_info['output_type']
        
        # Generate additional test cases
        test_cases = examples.copy()
        additional_tests = self.test_generator.generate_tests(10, input_type, output_type)
        test_cases.extend(additional_tests)
        
        # Run evolution
        return self.evolve(test_cases)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution"""
        if not self.generation_stats:
            return {}
        
        best_fitnesses = [s['best_fitness'] for s in self.generation_stats]
        avg_fitnesses = [s['avg_fitness'] for s in self.generation_stats]
        
        return {
            'total_generations': len(self.generation_stats),
            'final_best_fitness': best_fitnesses[-1] if best_fitnesses else 0,
            'final_avg_fitness': avg_fitnesses[-1] if avg_fitnesses else 0,
            'best_ever_fitness': max(best_fitnesses) if best_fitnesses else 0,
            'best_generation': best_fitnesses.index(max(best_fitnesses)) if best_fitnesses else 0,
            'pareto_front_size': len(self.pareto_front.front),
            'cache_stats': self.fitness_cache.get_stats(),
            'best_code': self.best_individual.code_string if self.best_individual else "",
            'best_metrics': {
                'complexity': self.best_individual.complexity if self.best_individual else 0,
                'size': self.best_individual.size if self.best_individual else 0,
                'depth': self.best_individual.depth if self.best_individual else 0,
            } if self.best_individual else {},
            'fitness_history': {
                'best': best_fitnesses,
                'avg': avg_fitnesses
            }
        }


# ============================================================================
# PART 16: INTERACTIVE EVOLUTION MODE
# ============================================================================

class InteractiveEvolution:
    """Interactive mode for human-guided evolution"""
    
    def __init__(self, gp_engine: MCTSIntegratedGP):
        self.engine = gp_engine
        self.current_population: List[ASTIndividual] = []
        self.selected_indices: Set[int] = set()
        self.generation = 0
    
    def initialize(self, population_size: int = 8):
        """Initialize interactive session"""
        self.engine.config.population_size = population_size
        self.engine.initialize_population()
        self.current_population = self.engine.population[:population_size]
        self.generation = 0
    
    def get_current_options(self) -> List[Dict[str, Any]]:
        """Get current population for user selection"""
        options = []
        for i, ind in enumerate(self.current_population):
            options.append({
                'index': i,
                'code': ind.code_string,
                'fitness': ind.fitness,
                'complexity': ind.complexity,
                'size': ind.size
            })
        return options
    
    def select_individuals(self, indices: List[int]):
        """User selects preferred individuals"""
        self.selected_indices = set(indices)
    
    def evolve_generation(self) -> List[Dict[str, Any]]:
        """Evolve one generation based on user selection"""
        if not self.selected_indices:
            # Random selection if none chosen
            self.selected_indices = set(random.sample(range(len(self.current_population)), 
                                                      max(1, len(self.current_population) // 2)))
        
        # Create new population from selected
        selected = [self.current_population[i] for i in self.selected_indices if i < len(self.current_population)]
        
        new_population = []
        
        # Keep selected
        new_population.extend(selected)
        
        # Generate offspring from selected
        while len(new_population) < len(self.current_population):
            if len(selected) >= 2 and random.random() < 0.7:
                p1, p2 = random.sample(selected, 2)
                child = self.engine.crossover(p1, p2)
            else:
                parent = random.choice(selected)
                child = self.engine._copy_individual(parent)
            
            child = self.engine.mutate(child)
            new_population.append(child)
        
        self.current_population = new_population
        self.generation += 1
        self.selected_indices.clear()
        
        return self.get_current_options()
    
    def get_best_so_far(self) -> Optional[ASTIndividual]:
        """Get the best individual seen so far"""
        if self.current_population:
            return max(self.current_population, key=lambda x: x.fitness)
        return None


# ============================================================================
# PART 17: PROJECT EXPORTER
# ============================================================================

class ProjectExporter:
    """Exports generated code as a project"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_python_project(self, individual: ASTIndividual, 
                             project_name: str = "generated_project",
                             include_tests: bool = True,
                             include_readme: bool = True) -> Path:
        """Export as a Python project"""
        project_dir = self.output_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Main module
        main_file = project_dir / "main.py"
        with open(main_file, 'w') as f:
            f.write(individual.code_string)
            f.write("\n\nif __name__ == '__main__':\n")
            f.write("    # Example usage\n")
            f.write("    print('Generated function ready to use')\n")
        
        # Tests
        if include_tests and individual.test_results:
            test_file = project_dir / "test_main.py"
            with open(test_file, 'w') as f:
                f.write("import pytest\n")
                f.write("from main import *\n\n")
                f.write("# Auto-generated tests\n")
                f.write("# TODO: Add specific test cases\n")
        
        # README
        if include_readme:
            readme_file = project_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(f"# {project_name}\n\n")
                f.write("Auto-generated Python project.\n\n")
                f.write("## Generated Code\n\n")
                f.write("```python\n")
                f.write(individual.code_string)
                f.write("\n```\n\n")
                f.write("## Metrics\n\n")
                f.write(f"- Complexity: {individual.complexity}\n")
                f.write(f"- Size: {individual.size} nodes\n")
                f.write(f"- Depth: {individual.depth}\n")
                f.write(f"- Fitness: {individual.fitness:.4f}\n")
        
        return project_dir
    
    def export_css_project(self, css: str, 
                          project_name: str = "generated_styles",
                          include_html: bool = True) -> Path:
        """Export as a CSS project"""
        project_dir = self.output_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # CSS file
        css_file = project_dir / "styles.css"
        with open(css_file, 'w') as f:
            f.write(css)
        
        # HTML template
        if include_html:
            html_file = project_dir / "index.html"
            with open(html_file, 'w') as f:
                f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Page</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Generated Page</h1>
            <p>This page uses auto-generated CSS.</p>
            <button class="btn">Click Me</button>
        </div>
    </div>
</body>
</html>
""")
        
        return project_dir
    
    def export_evolution_data(self, gp_engine: MCTSIntegratedGP,
                             filename: str = "evolution_data.json") -> Path:
        """Export evolution data as JSON"""
        output_file = self.output_dir / filename
        
        data = gp_engine.get_evolution_summary()
        data['config'] = {
            'population_size': gp_engine.config.population_size,
            'generations': gp_engine.config.generations,
            'mutation_rate': gp_engine.config.mutation_rate,
            'crossover_rate': gp_engine.config.crossover_rate,
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_file


# ============================================================================
# PART 18: MAIN INTERFACE
# ============================================================================

class CodeGeneratorAI:
    """Main interface for the Code Generation AI"""
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.gp_engine = MCTSIntegratedGP(self.config)
        self.css_generator = CSSGenerator()
        self.interactive = InteractiveEvolution(self.gp_engine)
        self.exporter = ProjectExporter()
        
        logger.info("Code Generator AI v3.1 initialized")
    
    def generate_function(self, 
                         description: Optional[str] = None,
                         examples: Optional[List[Tuple[Any, Any]]] = None,
                         input_type: Optional[str] = None,
                         output_type: Optional[str] = None,
                         num_tests: int = 15) -> Optional[ASTIndividual]:
        """Generate a function from description or examples"""
        logger.info("Starting function generation")
        
        # Generate test cases
        if examples:
            test_cases = examples.copy()
            type_info = self.test_generator.infer_types_from_examples(examples)
            input_type = input_type or type_info['input_type']
            output_type = output_type or type_info['output_type']
        else:
            input_type = input_type or 'int'
            output_type = output_type or 'int'
            test_cases = self.gp_engine.test_generator.generate_tests(num_tests, input_type, output_type)
        
        # Add edge cases
        edge_cases = self.gp_engine.test_generator.generate_edge_cases(input_type)
        test_cases.extend(edge_cases)
        
        # Run evolution
        result = self.gp_engine.evolve(test_cases)
        
        if result and self.config.auto_refactor:
            result.code_string = self.gp_engine.refactorer.refactor(result.code_string)
        
        return result
    
    def generate_css(self, 
                    style_type: str = "modern",
                    complexity: float = 5.0,
                    formality: float = 5.0,
                    color_scheme: str = "blue") -> str:
        """Generate CSS based on specifications"""
        specs = {
            'complexity': complexity,
            'formality': formality,
            'color_scheme': color_scheme
        }
        
        self.css_generator.specs = specs
        return self.css_generator.generate(style_type)
    
    def start_interactive(self, population_size: int = 8):
        """Start interactive evolution mode"""
        self.interactive.initialize(population_size)
        return self.interactive.get_current_options()
    
    def interactive_evolve(self, selected_indices: List[int]) -> List[Dict]:
        """Evolve one step in interactive mode"""
        self.interactive.select_individuals(selected_indices)
        return self.interactive.evolve_generation()
    
    def export_project(self, individual: ASTIndividual, project_name: str = "generated") -> Path:
        """Export generated code as a project"""
        return self.exporter.export_python_project(individual, project_name)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        return self.gp_engine.get_evolution_summary()


# ============================================================================
# PART 19: DEMO AND EXAMPLES
# ============================================================================

def demo_function_generation():
    """Demonstrate function generation"""
    print("=" * 70)
    print("DEMO: Function Generation from Examples")
    print("=" * 70)
    
    # Create AI instance
    config = AIConfig(
        population_size=50,
        generations=100,
        verbose_output=True,
        log_level="INFO"
    )
    ai = CodeGeneratorAI(config)
    
    # Example: Generate a function that doubles a number
    examples = [
        (5, 10),
        (3, 6),
        (-2, -4),
        (0, 0),
        (7, 14),
    ]
    
    print("\nExamples (input -> expected output):")
    for inp, out in examples:
        print(f"  {inp} -> {out}")
    
    print("\nEvolving...")
    result = ai.generate_function(examples=examples, input_type='int', output_type='int')
    
    if result:
        print("\n" + "-" * 70)
        print("BEST GENERATED CODE:")
        print("-" * 70)
        print(result.code_string)
        print("-" * 70)
        print(f"Fitness: {result.fitness:.4f}")
        print(f"Complexity: {result.complexity}")
        print(f"Size: {result.size} nodes")
        
        # Test the generated function
        print("\nTesting generated function:")
        for inp, expected in examples:
            exec_result, error = ai.gp_engine.executor.execute(result.code_string, inp)
            status = "✓" if exec_result == expected else "✗"
            print(f"  {status} {inp} -> {exec_result} (expected {expected})")
    
    return result


def demo_css_generation():
    """Demonstrate CSS generation"""
    print("\n" + "=" * 70)
    print("DEMO: CSS Generation")
    print("=" * 70)
    
    ai = CodeGeneratorAI()
    
    # Generate modern CSS
    css = ai.generate_css(
        style_type="modern",
        complexity=7.0,
        formality=4.0,
        color_scheme="blue"
    )
    
    print("\nGenerated Modern CSS:")
    print("-" * 70)
    print(css)
    print("-" * 70)
    
    # Generate corporate CSS
    css_corporate = ai.generate_css(
        style_type="corporate",
        complexity=5.0,
        formality=8.0,
        color_scheme="purple"
    )
    
    print("\nGenerated Corporate CSS:")
    print("-" * 70)
    print(css_corporate)
    print("-" * 70)
    
    return css


def demo_fuzzy_logic():
    """Demonstrate fuzzy logic system"""
    print("\n" + "=" * 70)
    print("DEMO: Fuzzy Logic System")
    print("=" * 70)
    
    # Create fuzzy system
    fuzzy = EnhancedFuzzySystem()
    
    # Create linguistic variables
    complexity = fuzzy.create_standard_linguistic_variable("complexity", 0, 10, 5)
    formality = fuzzy.create_standard_linguistic_variable("formality", 0, 10, 5)
    style = fuzzy.add_variable("style", 0, 10)
    
    # Add output sets
    style.add_set("minimal", MembershipFunction.GAUSSIAN, {'mean': 2, 'sigma': 1})
    style.add_set("modern", MembershipFunction.GAUSSIAN, {'mean': 5, 'sigma': 1.5})
    style.add_set("complex", MembershipFunction.GAUSSIAN, {'mean': 8, 'sigma': 1})
    
    # Add rules
    fuzzy.add_rule({"complexity": "very_low", "formality": "low"}, {"style": "minimal"}, 1.0)
    fuzzy.add_rule({"complexity": "medium", "formality": "medium"}, {"style": "modern"}, 1.0)
    fuzzy.add_rule({"complexity": "very_high", "formality": "high"}, {"style": "complex"}, 1.0)
    fuzzy.add_rule({"complexity": "low"}, {"style": "minimal"}, 0.7)
    fuzzy.add_rule({"complexity": "high"}, {"style": "complex"}, 0.7)
    
    # Test with different inputs
    test_inputs = [
        {"complexity": 2.0, "formality": 3.0},
        {"complexity": 5.0, "formality": 5.0},
        {"complexity": 8.5, "formality": 7.0},
    ]
    
    for inputs in test_inputs:
        # Show fuzzification
        print(f"\nInput: complexity={inputs['complexity']}, formality={inputs['formality']}")
        
        fuzz_complexity = complexity.fuzzify(inputs['complexity'])
        print(f"  Fuzzified complexity: {fuzz_complexity}")
        
        # Infer
        output = fuzzy.infer(inputs)
        print(f"  Output style: {output.get('style', 'N/A'):.2f}")
        
        # Interpret output
        style_val = output.get('style', 0)
        if style_val < 3.5:
            interpretation = "Minimal"
        elif style_val < 6.5:
            interpretation = "Modern"
        else:
            interpretation = "Complex"
        print(f"  Interpretation: {interpretation}")


def demo_pareto_optimization():
    """Demonstrate Pareto front optimization"""
    print("\n" + "=" * 70)
    print("DEMO: Pareto Front Multi-Objective Optimization")
    print("=" * 70)
    
    pareto = ParetoFront(['correctness', 'simplicity', 'size'])
    
    # Add some individuals
    individuals = [
        ("Perfect but complex", 1.0, 0.2, 0.1),
        ("Good and simple", 0.8, 0.9, 0.8),
        ("Medium everything", 0.5, 0.5, 0.5),
        ("Simple but wrong", 0.2, 1.0, 1.0),
        ("Perfect and simple", 1.0, 0.9, 0.8),
        ("Good and medium", 0.8, 0.6, 0.5),
    ]
    
    print("\nAdding individuals to Pareto front:")
    print("-" * 70)
    
    for name, corr, simp, size in individuals:
        ind = ASTIndividual()
        ind.fitness_components = {
            'correctness': corr,
            'simplicity': simp,
            'size': size
        }
        
        added = pareto.update(ind)
        status = "Added to front" if added else "Dominated"
        print(f"  {name:25s} -> {status}")
        print(f"    Correctness: {corr:.1f}, Simplicity: {simp:.1f}, Size: {size:.1f}")
    
    print(f"\nPareto Front Size: {len(pareto.front)}")
    print("\nPareto Optimal Solutions:")
    print("-" * 70)
    for i, ind in enumerate(pareto.front):
        print(f"  Solution {i+1}:")
        print(f"    Correctness: {ind.fitness_components['correctness']:.1f}")
        print(f"    Simplicity: {ind.fitness_components['simplicity']:.1f}")
        print(f"    Size: {ind.fitness_components['size']:.1f}")
        print(f"    Crowding Distance: {ind.crowding_distance:.2f}")


def demo_full_pipeline():
    """Demonstrate the full pipeline"""
    print("\n" + "=" * 70)
    print("DEMO: Full Pipeline")
    print("=" * 70)
    
    # Initialize
    config = AIConfig(
        population_size=30,
        generations=50,
        verbose_output=True,
        auto_refactor=True,
        export_projects=True
    )
    ai = CodeGeneratorAI(config)
    
    # Generate a function to sum a list
    print("\nTask: Generate a function that sums a list of numbers")
    
    examples = [
        ([1, 2, 3], 6),
        ([5, 5, 5], 15),
        ([], 0),
        ([-1, 1], 0),
        ([10], 10),
    ]
    
    result = ai.generate_function(
        examples=examples,
        input_type='list',
        output_type='int'
    )
    
    if result:
        print("\n" + "=" * 70)
        print("FINAL RESULT:")
        print("=" * 70)
        print(result.code_string)
        print("=" * 70)
        
        # Export project
        if config.export_projects:
            project_path = ai.export_project(result, "sum_function")
            print(f"\nProject exported to: {project_path}")
    
    # Get summary
    summary = ai.get_summary()
    print(f"\nEvolution Summary:")
    print(f"  Generations: {summary.get('total_generations', 0)}")
    print(f"  Best Fitness: {summary.get('final_best_fitness', 0):.4f}")
    print(f"  Cache Hit Rate: {summary.get('cache_stats', {}).get('hit_rate', 0):.2%}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    # Run demos
    demo_fuzzy_logic()
    demo_pareto_optimization()
    demo_css_generation()
    demo_function_generation()
    demo_full_pipeline()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)