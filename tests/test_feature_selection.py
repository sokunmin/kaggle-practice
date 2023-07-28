from typing import List, Any, Set
from unittest import TestCase

from selection.feature_selection import Model, backward_elimination


class Test(TestCase):
    def test_backward_elimination(self) -> None:
        variables = ['a', 'b', 'c', 'd', 'e']
        maps = {'a': 4, 'b': 2, 'c': -2, 'd': 3, 'e': -1}

        def train_model(variables: Set[str]) -> Any:
            return f'Model-{"".join(variables)}'

        def score_model(_model: Model, variables: Set[str]) -> float:
            return -sum(maps[v] for v in variables)

        result: Any = backward_elimination(variables, train_model, score_model, verbose=True)
        assert result[1] == {'a', 'b', 'd'}
