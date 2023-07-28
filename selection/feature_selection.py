"""
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""

import itertools
from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Tuple, TypedDict, TypeVar, Set

Model = TypeVar('Model')
TrainModel = Callable[[Set[str]], Model]
ScoreModel = Callable[[Model, Set[str]], float]

class ExhaustiveSearchResult(TypedDict):
    n: int
    variables: List[str]
    score: float
    model: Any

class Step(NamedTuple):
    score: float
    variable: Optional[str]
    model: Any


def exhaustive_search(variables: List[str], train_model: TrainModel,
                      score_model: ScoreModel) -> List[ExhaustiveSearchResult]:
    """ Variable selection using backward elimination

    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores

    Returns:
        List of the best subset models for increasing number of variables
    """
    # create models of increasing size and determine the best models in each case
    result = []
    for nvariables in range(1, len(variables) + 1):
        best: Optional[ExhaustiveSearchResult] = None
        for varcombo in itertools.combinations(variables, nvariables):
            subset = list(varcombo)
            subset_model = train_model(subset)
            subset_score = score_model(subset_model, subset)
            if best is None or best['score'] > subset_score:
                best = ExhaustiveSearchResult(
                    n=nvariables,
                    variables=subset,
                    score=subset_score,
                    model=subset_model
                )
        assert best is not None
        result.append(best)
    return result


def forward_selection(variables: Iterable[str], train_model: TrainModel, score_model: ScoreModel, *,
                      verbose: bool = False):
    """ Variable selection using forward selection

    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores

    Returns:
        (best_model, best_variables)
    """
    best_variables = set(variables)
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print('Variables: ' + ', '.join(variables))
        print(f'Start: score={best_score: .2f}')

    while len(best_variables) > 1:
        steps = [Step(best_score, None, best_model)]
        for remove_var in best_variables:
            step_vars = set(best_variables)
            step_vars.remove(remove_var)
            step_model = train_model(step_vars)
            step_score = score_model(step_model, step_vars)
            steps.append(Step(step_score, remove_var, step_model))

        steps.sort(key=lambda x: x[0])

        best_score, removed_step, best_model = steps[0]
        if verbose:
            print(f'Step: score={best_score:.2f}, remove {removed_step}')
        if removed_step is None:
            break
        best_variables.remove(removed_step)
    return best_model, best_variables


def backward_elimination(variables: Iterable[str], train_model: TrainModel, score_model: ScoreModel, *,
                         verbose: bool = False) -> Tuple[Model, List[str]]:
    """ Variable selection using backward elimination

    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores

    Returns:
        (best_model, best_variables)
    """
    # we start with a model that contains all variables
    best_variables = set(variables)
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print('Variables: ' + ', '.join(variables))
        print(f'Start: score={best_score: .2f}')

    while len(best_variables) > 1:
        steps = [Step(best_score, None, best_model)]
        for remove_var in best_variables:
            step_vars = set(best_variables)
            step_vars.remove(remove_var)
            step_model = train_model(step_vars)
            step_score = score_model(step_model, step_vars)
            steps.append(Step(step_score, remove_var, step_model))

        # sort by ascending score
        steps.sort(key=lambda x: x[0])

        # the first entry is the model with the lowest score
        best_score, removed_step, best_model = steps[0]
        if verbose:
            print(f'Step: score={best_score:.2f}, remove {removed_step}')
        if removed_step is None:
            # step here, as removing more variables is detrimental to performance
            break
        best_variables.remove(removed_step)

    return best_model, best_variables