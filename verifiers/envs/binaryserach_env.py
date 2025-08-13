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


DEFAULT_SEARCH_PROMPT_TEMPLATE = """
Respond in the following format, using careful step-by-step reasoning.

<think>
...
</think>
<answer>
...
</answer>
"""



def extract_integers(input_string):
    """
    Extracts all integers from a string and returns them as a list of integers.
    
    Args:
        input_string (str): The string to extract integers from
        
    Returns:
        list: A list of integers found in the string
    """
    # Find all matches of integers (including negative numbers)
    numbers = re.findall(r'-?\d+', input_string)
    
    # Convert each found string to integer
    return [int(num) for num in numbers]


class BinarySearchRubric(Rubric):
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
        self.add_reward_func(self.binary_search_guess_reward_func)
        self.add_reward_func(self.parser.get_format_reward_func())
        self.add_reward_func(self.optimal_guess_reward_func)
        self.add_reward_func(self.search_len_reward_func)
        # self.add_reward_func(self.int_answer_reward_func)
        # self.add_reward_func(self.exact_answer_reward_func)

    def search_len_reward_func(
        self,
        *args,
        **kwargs
    ) -> List[float]:
        prompt = kwargs.get("prompt", None)
        answer = kwargs.get("answer", None)
        completion = kwargs.get("completion", None)
        if answer is None:
            raise ValueError("Missing required keyword argument: 'answers'")

        def len_score(trajectory, expected_answer):
            total_guess_steps = 0
            is_completed = False
            for i, msg in enumerate(trajectory):
                if msg['role'] == "user":
                    if "Congratulations" in msg["content"]:
                        is_completed = True
            if not is_completed:
                return -5
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    #if hasattr(parsed,"answer") and parsed.answer is not None:
                    total_guess_steps += 1
            optimal_steps = np.math.log2(1000)
            if total_guess_steps <= optimal_steps:
                score_steps = 1.0  # Full reward if the model guesses in fewer or exactly log2(n) steps
            else:
                score_steps = np.exp(-((total_guess_steps - optimal_steps) ** 2) / (2 * (optimal_steps / 2) ** 2))
            return score_steps
        return float(len_score(completion, answer))

    
    def binary_search_guess_reward_func(
        self,
        *args,
        **kwargs
    ) -> List[float]:
        prompt = kwargs.get("prompt", None)
        answer = kwargs.get("answer", None)
        completion = kwargs.get("completion", None)
        if answer is None:
            raise ValueError("Missing required keyword argument: 'answers'")

        for i, msg in enumerate(prompt):
            if msg['role'] == 'user':
                max_range = extract_integers(msg["content"])[-1]
                break

        def reasonable_guess_score(trajectory, expected_answer, max_range):

            reasonable_range = (0,max_range)
            total_guess_steps = 0

            scores = []
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    total_guess_steps += 1
                    if hasattr(parsed,"answer") and parsed.answer is not None:
                        #total_guess_steps += 1
                        ans_score = -1.5
                        if str(parsed.answer).isdigit() and parsed.answer is not None:
                            guess = int(parsed.answer)
                            if guess == expected_answer:
                                ans_score = 5.
                            elif (guess < expected_answer) & (guess in range(*reasonable_range)):
                                reasonable_range = (guess+1,reasonable_range[1])
                                ans_score = 0.5
                            elif (guess > expected_answer) & (guess in range(*reasonable_range)):
                                reasonable_range = (reasonable_range[0],guess)
                                ans_score = 0.5
                            elif (guess < expected_answer) & (guess not in range(*reasonable_range)): # palpite fora do reasonable
                                ans_score = -0.5
                            elif (guess > expected_answer) & (guess not in range(*reasonable_range)): # palpite fora do reasonable
                                ans_score = -0.5
                            else:
                                ans_score = -1.5
                        else: # resposta não é um inteiro
                            ans_score = -1.5
                        scores.append(ans_score)
                    else:
                        ans_score = -5
                        scores.append(ans_score)
            score_reasonable = np.sum(scores)/total_guess_steps
            return score_reasonable
        return reasonable_guess_score(completion,answer,max_range)

    def optimal_guess_reward_func(
        self,
        **kwargs
    ) -> List[float]:
        prompt = kwargs.get("prompt", None)
        answer = kwargs.get("answer", None)
        completion = kwargs.get("completion", None)
        if answer is None:
            raise ValueError("Missing required keyword argument: 'answers'")

        for i, msg in enumerate(prompt):
            if msg['role'] == 'user':
                max_range = extract_integers(msg["content"])[-1]
                break

        def optimal_guess_score(trajectory, expected_answer):

            reasonable_range = (0,max_range)
            total_guess_steps = 0
            scores = []
            for i, msg in enumerate(trajectory):
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg['content'])
                    total_guess_steps += 1
                    if hasattr(parsed,"answer") and parsed.answer is not None:
                        ans_score = -0.5
                        if str(parsed.answer).isdigit() and parsed.answer is not None:
                            guess = int(parsed.answer)
                            optimal = int((reasonable_range[0] + reasonable_range[1])/2)
                            ans_score = 1 - (np.abs(optimal-guess)/optimal)
                            if guess == expected_answer:
                                ans_score = 5.
                            elif (guess < expected_answer) & (guess in range(*reasonable_range)):
                                reasonable_range = (guess+1,reasonable_range[1])
                            elif (guess > expected_answer) & (guess in range(*reasonable_range)):
                                reasonable_range = (reasonable_range[0],guess)
                        else:
                            ans_score = -1.5
                        scores.append(ans_score)
                    else:
                        ans_score = -5
                        scores.append(ans_score)
            return np.mean(scores)
        return optimal_guess_score(completion,answer)
            




class BinarySearchEnv(SingleTurnEnv):
    def __init__(self, dataset, max_turns: int = 10, **kwargs):

        super().__init__(dataset=dataset["train"], eval_dataset=dataset['eval'], message_type='chat', **kwargs)
        self.max_turns = max_turns

        self.llm_parser = XMLParser(fields=["think", "answer"])
        self.env_parser = XMLParser(fields=["output"])
        self.rubric = BinarySearchRubric(parser=self.llm_parser, env_parser=self.env_parser)

    def is_completed(self,
                     messages: str,
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        try:
            for i, msg in enumerate(messages):
                if msg['role'] == 'user':
                    if "Congratulations" in msg["content"]:
                        return True
            return False
        except Exception:
            return False

    def env_response(self,
                     messages: str,
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid code field (not just None from failed parsing)
            answer = state['answer']
            if hasattr(parsed, 'answer') and parsed.answer is not None:
                if str(parsed.answer).isdigit() and parsed.answer is not None:
                    if int(parsed.answer) == answer:
                        return {"role": "user", "content": "Congratulations!! Right answer!"}, state
                    elif int(parsed.answer) > answer:
                        return {"role": "user", "content": f"n is lower than {str(parsed.answer)}"}, state
                    elif int(parsed.answer) < answer:
                        return {"role": "user", "content": f"n is higher than {str(parsed.answer)}"}, state
                    else:
                        return {"role": "user", "content": "Error: invalid answer, you must pass an integer number only inside the tags <answer> </answer>."}, state
                else:
                    return {"role": "user", "content": "Error: answer is not an integer, you must pass an integer number only inside the tags <answer> </answer>."}, state
        except Exception:
            pass
        return {"role": "user", "content": "Error: Answer not found or invalid XML format. Please ensure correct formatting. You must pass an integer number only inside the tags <answer> </answer>."}, state
