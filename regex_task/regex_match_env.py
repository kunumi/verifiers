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



def regex_check(completion, answer):
    answer = eval(answer)

    pos = [i for i, j in answer.items() if j]
    neg = [i for i, j in answer.items() if not j]

    try:
        pred_pos = [bool(re.fullmatch(completion, s)) for s in pos]
        pred_neg = [bool(re.fullmatch(completion, s)) for s in neg]
    except re.error as e:
        return 0.0, f"Invalid regex: {e}", [], []

    test = pos + neg
    gold = [True]*len(pos) + [False]*len(neg)
    pred = pred_pos + pred_neg
    accuracy = sum(p == g for p, g in zip(pred, gold)) / (len(test) or 1)

    # Listas pedidas:
    falsos_negativos = [s for s, ok in zip(pos, pred_pos) if not ok]  # positivos que NÃO passaram
    falsos_positivos = [s for s, ok in zip(neg, pred_neg) if ok]      # negativos que passaram

    return accuracy, "OK", falsos_negativos, falsos_positivos

def get_next_step(state, prompt, completion):
    #print('C: ', completion)
    llm_parser = XMLParser(fields=["think", "answer", "summary"])
    answer = state['answer']
    base = prompt[-1]['content']
    if 'last tries' not in base:
        base = base + '\n\nYour last tries and the feedback:\n'
    
    base = base + '<try>\n'

    parsed = llm_parser.parse(completion[-1]["content"])

    if hasattr(parsed, 'summary') and parsed.summary is not None:
        base = base + f'<summary>{parsed.summary}</summary>\n'
    else:
        base = base + f'<summary>Invalid step summary passed</summary>\n'

    if hasattr(parsed, 'answer') and parsed.answer is not None:
        print('A: ', parsed.answer)
        base = base + f'<past_answer>{parsed.answer}</past_answer>'
        try: 
            r = regex_check(parsed.answer, answer)
            feedback = f'\nThe performance of your last try was: {r[0]}'
            feedback += f'\nThe positive cases that didn\'t pass where: {r[2]}'
            feedback += f'\nThe negative cases that passed where: {r[3]}'
            return {"role": "user", "content": base + feedback + '\n</try>'}, state
        except Exception as e:
            print(e)
            return {"role": "user", "content": base + '\nYour last try was Invalid\n</try>'}, state
    
    return {"role": "user", "content": base + '\nYour last try was Invalid\n</try>'}, state
        


class RegexMatchRubric(Rubric):
    def __init__(
        self,
        parser: XMLParser = XMLParser(fields=["think", "answer", "summary"]),
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
        self.add_reward_func(self.regex_reward_func)

    def regex_reward_func(self, *args, **kwargs):
        prompt = kwargs.get("prompt", None)
        answer = kwargs.get("answer", None)
        completion = kwargs.get("completion", None)

        parsed = self.parser.parse(completion[-1]['content'])

        if hasattr(parsed,"answer") and parsed.answer is not None:
            try: 
                r = regex_check(parsed.answer, answer)
                return r[0]
            except Exception as e:
                return 0
        return 0



class RegexMatchEnv(SingleTurnEnv):
    def __init__(self, dataset, max_turns: int = 10, **kwargs):

        super().__init__(dataset=dataset["train"], eval_dataset=dataset['eval'], message_type='chat', **kwargs)
        self.max_turns = max_turns

        self.llm_parser = XMLParser(fields=["think", "answer", "summary"])
        self.env_parser = XMLParser(fields=["output"])
        self.rubric = RegexMatchRubric(parser=self.llm_parser, env_parser=self.env_parser)
