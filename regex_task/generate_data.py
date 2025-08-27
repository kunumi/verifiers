import random, re, json, string
import pandas as pd

# --- 1. Building-blocks -----------------------------------------------------

ALPHAS  = string.ascii_lowercase
DIGITS  = string.digits
CHARS   = ALPHAS + DIGITS

CHAR_CLASSES = [
    ("[a-z]",  lambda n: ''.join(random.choice(ALPHAS) for _ in range(n))),
    ("[0-9]",  lambda n: ''.join(random.choice(DIGITS) for _ in range(n))),
    ("[A-F]",  lambda n: ''.join(random.choice("ABCDEF") for _ in range(n))),
    ("[a-z0-9]",lambda n: ''.join(random.choice(CHARS)  for _ in range(n))),
]

QUANTIFIERS = [
    ("+",  lambda: random.randint(1, 3)),          # ≥1, choose 1-3
    ("*",  lambda: random.randint(0, 3)),          # ≥0, choose 0-3
    ("?",  lambda: random.randint(0, 1)),          # 0-1
    ("{n}",lambda: lambda: random.randint(1, 4)),  # exactly n (filled later)
]

def random_segment():
    """Return (regex_piece, generator_fn) for one segment."""
    char_pat, gen_chars = random.choice(CHAR_CLASSES)
    q_pat, q_gen_len    = random.choice(QUANTIFIERS)
    if q_pat == "{n}":
        n      = q_gen_len()           # choose exact repetition
        q_pat  = f"{{{n}}}"
        maker  = lambda: gen_chars(n)
    else:
        maker  = lambda: gen_chars(q_gen_len())
    return char_pat + q_pat, maker

# --- 2. Complete regex builder ---------------------------------------------

def make_random_regex(min_segments=2, max_segments=4):
    segs, gens = zip(*(random_segment() for _ in range(random.randint(min_segments, max_segments))))
    pattern = "^" + "".join(segs) + "$"
    return pattern, gens

# --- 3. Example generation --------------------------------------------------

def sample_strings(regex, gens, n_pos=10, n_neg=10, max_attempts=5000):
    # positives: directly synthesize from gens
    positives = [''.join(g() for g in gens) for _ in range(n_pos)]

    # negatives: brute-force random strings until we have n_neg non-matches
    negatives, attempts = [], 0
    clen = max(len(s) for s in positives) + 2                # reasonable length bound
    while len(negatives) < n_neg and attempts < max_attempts:
        candidate = ''.join(random.choice(CHARS) for _ in range(random.randint(1, clen)))
        if not re.fullmatch(regex, candidate):
            negatives.append(candidate)
        attempts += 1
    if len(negatives) < n_neg:
        raise RuntimeError("Could not generate enough negative samples.")
    return positives, negatives

# --- 4. One-shot task -------------------------------------------------------

def generate_reverse_regex_task():
    regex, gens         = make_random_regex()
    pos, neg            = sample_strings(regex, gens)

    prompt = "Below are strings labelled 'positive' (they match an unknown pattern) " \
                "and 'negative' (they do not). Infer **one** regular expression that " \
                "matches _all_ positives and _none_ of the negatives." \
                "\nPositives: " + str(pos) + '\n' \
                "Negatives: " + str(neg) \
                
    answer = {p: True for p in pos}
    answer |= {n: False for n in neg}

    print(regex)

    return prompt, answer

# --- 5. (Optional) Evaluation helper ---------------------------------------

def get_next_step(state, prompt, completion):
    llm_parser = XMLParser(fields=["think", "answer"])
    answer = state['answer']
    base = prompt[-1]['content']
    if 'last tries' not in base:
        base = base + '\n\nYour last tries and the feedback:\n'
    base = base + 'Try:\n' + completion[-1]["content"]

    parsed = llm_parser.parse(completion[-1]["content"])
    if hasattr(parsed, 'answer') and parsed.answer is not None:
        try: 
            r = reward_fn(parsed.answer, answer)
            feedback = f'\nThe performance of your last try was: {r[0]}'
            feedback += f'\nThe positive cases that didn\'t pass where: {r[2]}'
            feedback += f'\nThe negative cases that passed where: {r[3]}'
            return {"role": "user", "content": base + feedback}, state
        except Exception:
            return {"role": "user", "content": base + '\nYour last try was Invalid\n'}, state
        
    return {"role": "user", "content": base + '\nYour last try was Invalid'}, state

def reward_fn(completion, answer):
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


# --- 6. Demo ----------------------------------------------------------------
if __name__ == "__main__":
    # prompts = []
    # answers = []

    # while len(prompts) < 100:
    #     try:
    #         prompt, answer = generate_reverse_regex_task()
    #         prompts.append(prompt)
    #         answers.append(answer)
    #     except Exception as e:
    #         pass


    # d = {'prompt': prompts, 'answer': answers}
    # df = pd.DataFrame(d)
    # df.to_csv('data.csv', index=False)

    print(reward_fn('^(?=.*[A-Z])(?=.*\d)[A-Za-z0-9]{3,12}$', "{'33B22': True, 'vh394DCC249': True, 'va5BD': True, '54FAD286': True, 'c76B6': True, '813FFC880': True, 'c682EE': True, 'n1C': True, 'd158DBD5': True, 'k70CAF226': True, '3qaxyo': False, '5qi': False, 'x5ioxp80id': False, '121c9dqf': False, '3nmivr4vtxz': False, '98': False, 'jwygtmy7avtuj': False, 'z': False, 'y': False, '014srsfikk': False}"))
