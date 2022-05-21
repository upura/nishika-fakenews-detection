import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, T5Tokenizer


# https://github.com/HendrikStrobelt/detecting-fake-text
class AbstractLanguageChecker:
    """
    Abstract Class that defines the Backend API of GLTR.

    To extend the GLTR interface, you need to inherit this and
    fill in the defined functions.
    """

    def __init__(self):
        """
        In the subclass, you need to load all necessary components
        for the other functions.
        Typically, this will comprise a tokenizer and a model.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text, topk=40):
        """
        Function that GLTR interacts with to check the probabilities of words

        Params:
        - in_text: str -- The text that you want to check
        - topk: int -- Your desired truncation of the head of the distribution

        Output:
        - payload: dict -- The wrapper for results in this function, described below

        Payload values
        ==============
        bpe_strings: list of str -- Each individual token in the text
        real_topk: list of tuples -- (ranking, prob) of each token
        pred_topk: list of list of tuple -- (word, prob) for all topk
        """
        raise NotImplementedError

    def postprocess(self, token):
        """
        clean up the tokens from any special chars and encode
        leading space by UTF-8 code '\u0120', linebreak with UTF-8 code 266 '\u010A'
        :param token:  str -- raw token text
        :return: str -- cleaned and re-encoded token text
        """
        raise NotImplementedError


class LM(AbstractLanguageChecker):
    def __init__(self, model, tokenizer):
        super(LM, self).__init__()
        self.enc = tokenizer
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.start_token = self.enc.encode_plus(
            self.enc.bos_token,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).data["input_ids"][0]

    def check_probabilities(self, in_text, topk=40):
        # Process input
        token_ids = self.enc.encode_plus(
            in_text, padding="max_length", max_length=512, return_tensors="pt"
        ).data["input_ids"][0]
        token_ids = torch.cat([self.start_token, token_ids])
        # Forward through the model
        output = self.model(token_ids.to(self.device))
        all_logits = output.logits[:-1].detach().squeeze()
        # construct target and pred
        # yhat = torch.softmax(logits[0, :-1], dim=-1)
        all_probs = torch.softmax(all_logits, dim=1)

        y = token_ids[1:]
        # Sort the predictions for each timestep
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        # [(pos, prob), ...]
        real_topk_pos = list(
            [
                int(np.where(sorted_preds[i] == y[i].item())[0][0])
                for i in range(y.shape[0])
            ]
        )
        real_topk_probs = (
            all_probs[np.arange(0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        )
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

        real_topk = list(zip(real_topk_pos, real_topk_probs))

        # pred_topk = []
        payload = {
            "real_topk": real_topk,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return payload


if __name__ == "__main__":

    train = pd.read_csv("../input/nishika-fakenews/train.csv")
    test = pd.read_csv("../input/nishika-fakenews/test.csv")

    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")

    results_fake = []
    for raw_text in tqdm(train.query("isFake==1")["text"]):
        lm = LM(model=model, tokenizer=tokenizer)
        payload = lm.check_probabilities(raw_text, topk=5)
        results_fake.append(
            np.percentile([prob[1] for prob in payload["real_topk"]], [25, 50, 75])
        )
        del payload
        gc.collect()
    np.save("results_fake", np.array(results_fake))

    results_real = []
    for raw_text in tqdm(train.query("isFake==0")["text"]):
        lm = LM(model=model, tokenizer=tokenizer)
        payload = lm.check_probabilities(raw_text, topk=5)
        results_real.append(
            np.percentile([prob[1] for prob in payload["real_topk"]], [25, 50, 75])
        )
        del payload
        gc.collect()
    np.save("results_real", np.array(results_real))

    results_test = []
    for raw_text in tqdm(test["text"]):
        lm = LM(model=model, tokenizer=tokenizer)
        payload = lm.check_probabilities(raw_text, topk=5)
        results_test.append(
            np.percentile([prob[1] for prob in payload["real_topk"]], [25, 50, 75])
        )
        del payload
        gc.collect()
    np.save("results_test", np.array(results_test))
