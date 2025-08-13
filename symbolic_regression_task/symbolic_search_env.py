from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple
import numpy as np

from openai import OpenAI


from verifiers.rubrics import Rubric
import re

from verifiers.parsers import XMLParser
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.envs.singleturn_env import SingleTurnEnv

from symbolic_regression_task.lisp import eval_lisp, parse, global_env



def get_next_iteration(state, prompt, completion):
    llm_parser = XMLParser(fields=["think", "answer"])
    answer = state['answer']
    base = prompt[-1]['content']
    # if 'last tries' not in base:
    #     base = base + '\n\nYour last tries and the feedback:\n'
    # base = base + 'Try:\n' + completion[-1]["content"]

    # parsed = llm_parser.parse(completion[-1]["content"])
    # if hasattr(parsed, 'answer') and parsed.answer is not None:
    #     try: 
    #         r = test_solution(parsed.answer, answer)
    #         return {"role": "user", "content": base + f'\nThe performance of your last try was: {r}\n'}, state
    #     except Exception:
    #         return {"role": "user", "content": base + '\nYour last try was Invalid\n'}, state
    

    return {"role": "user", "content": base + '\nYour last try was Invalid'}, state


def test_solution(func_str, answer_str):
    base_str = f'(define original (lambda (x) {answer_str} ))'
    eval_lisp(parse(base_str))

    base_str = f'(define pred (lambda (x) {func_str} ))'
    eval_lisp(parse(base_str))

    xs = np.linspace(-100, 100, 201)
    print(global_env['original'](xs))
    rel_mse = np.mean(((global_env['original'](xs) - global_env['pred'](xs)) / (np.abs(global_env['original'](xs)) + 1e-12))**2)

    return rel_mse

class SymbolicSearchRubric(Rubric):
    def __init__(
        self,
        parser: XMLParser = XMLParser(fields=["think", "answer"]),
        env_parser: XMLParser = XMLParser(fields=["output"]),
        **kwargs
    ):
        super().__init__( **kwargs)
        self.parser = parser
        self.env_parser = env_parser
        # self.reward_funcs = [
        #     # self.exact_answer_reward_func,
        #     # self.int_answer_reward_func,
        #     self.binary_search_guess_reward_func,
        #     # self.search_len_reward_func,
        #     #self.optimal_guess_reward_func,
        #     # self.parser.get_xml_reward_func(),
        #     # self.parser.get_format_reward_func()
        # ]
        self.add_reward_func(self.parser.get_format_reward_func())
        # self.add_reward_func(self.distance_reward_func)

    def distance_reward_func(self, *args, **kwargs):
        prompt = kwargs.get("prompt", None)
        answer = kwargs.get("answer", None)
        completion = kwargs.get("completion", None)

        parsed = self.parser.parse(completion['content'])

        if hasattr(parsed,"answer") and parsed.answer is not None:
            try: 
                r = test_solution(parsed.answer, answer)
                return r
            except Exception as e:
                return 0
        return -5



class SymbolicSearchEnv(SingleTurnEnv):
    def __init__(self, dataset, max_turns: int = 10, **kwargs):

        super().__init__(dataset=dataset["train"], eval_dataset=dataset['eval'], message_type='chat', **kwargs)
        self.max_turns = max_turns

        self.llm_parser = XMLParser(fields=["think", "answer"])
        self.env_parser = XMLParser(fields=["output"])
        self.rubric = SymbolicSearchRubric(parser=self.llm_parser, env_parser=self.env_parser)
