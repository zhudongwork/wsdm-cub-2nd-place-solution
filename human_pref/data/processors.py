import json

import torch
# who is proficient in multiple languages 

class ProcessorPAB:
    PROMPT_PREFIX = """Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better."""

    PROMPT_SUFFIX = "verdict is: [["

    LABEL_COLS = ["winner_model_a", "winner_model_b"]

    def __init__(self, tokenizer, max_length, support_system_role):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.support_system_role = support_system_role

    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Conversation between a User and two Assistants|>"
        tail = "<|The End of Conversation between a User and two Assistants|>\n"
        parts = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            parts.append(
                f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n\n### Assistant B:\n{response_b}\n"
            )
        text = "".join(parts)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return head + truncated_text + tail

    def build_input(self, data):
        conversation = self.build_conversation(
            # json.loads(data["prompt"]),
            # json.loads(data["response_a"]),
            # json.loads(data["response_b"]),
            [data["prompt"]],
            [data["response_a"]],
            [data["response_b"]],
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        label = torch.tensor([data[col] for col in self.LABEL_COLS]).float()
        return dict(
            input_ids=input_ids,
            input_text=input_text,
            label=label,
        )

class LMSYSProcessorPAB:
    PROMPT_PREFIX = """Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie."""

    PROMPT_SUFFIX = "verdict is: [["

    LABEL_COLS = ["winner_model_a", "winner_model_b"]

    def __init__(self, tokenizer, max_length, support_system_role):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.support_system_role = support_system_role

    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Conversation between a User and two Assistants|>"
        tail = "<|The End of Conversation between a User and two Assistants|>\n"
        
        rounds = [
            f"\n### User:\n{prompts[i]}\n\n### Assistant A:\n{responses_a[i]}\n\n### Assistant B:\n{responses_b[i]}\n"
            for i in range(len(prompts))
        ]

        for k in range(len(rounds)):
            text = "".join(rounds[k:])
            if len(self.tokenizer(text)["input_ids"] )<4096: break
        # print(text)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return head + truncated_text + tail

    def build_input(self, data):
        conversation = self.build_conversation(
            json.loads(data["prompt"]),
            json.loads(data["response_a"]),
            json.loads(data["response_b"]),
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        label = torch.tensor([data[col] for col in self.LABEL_COLS]).float()
        return dict(
            input_ids=input_ids,
            input_text=input_text,
            label=label,
        )


class MIDProcessorPAB(ProcessorPAB):
    def tructure_text(self, text, type="prompt"):
        if type == 'prompt':
            max_no = 1362
            s_no = 680
            e_no = -681
        else:
            max_no = 1342
            s_no = 770
            e_no = -771
      
        encoded = self.tokenizer(text, return_offsets_mapping=True)
        if len(encoded['input_ids']) > max_no:
            start_idx, end_idx = encoded['offset_mapping'][s_no]
            new_text = text[:end_idx]
        
            start_idx, end_idx = encoded['offset_mapping'][e_no]
            new_text = new_text + "\n...\n...\n" + text[start_idx:]
           
            text = new_text
   
        return text

    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Conversation between a User and two Assistants|>"
        tail = "<|The End of Conversation between a User and two Assistants|>\n"
        parts = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            prompt = self.tructure_text(prompt)
            response_a = self.tructure_text(response_a, "response")
            response_b = self.tructure_text(response_b, "response")
            parts.append(
                f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n\n### Assistant B:\n{response_b}\n"
            )
        text = "".join(parts)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return head + truncated_text + tail

    def build_input(self, data):
        conversation = self.build_conversation(
            # json.loads(data["prompt"]),
            # json.loads(data["response_a"]),
            # json.loads(data["response_b"]),
            [data["prompt"]],
            [data["response_a"]],
            [data["response_b"]],
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        label = torch.tensor([data[col] for col in self.LABEL_COLS]).float()
        return dict(
            input_ids=input_ids,
            input_text=input_text,
            label=label,
        )


class MIDV2ProcessorPAB(ProcessorPAB):
    def tructure_text(self, text, encoded, max_no):
        if len(encoded['input_ids']) > max_no:
            s_no = (max_no - 5) // 2
            e_no = -s_no

            start_idx, end_idx = encoded['offset_mapping'][s_no]
            new_text = text[:end_idx]
        
            start_idx, end_idx = encoded['offset_mapping'][e_no]
            new_text = new_text + "\n...\n...\n" + text[start_idx:]
            
            text = new_text

        return text


    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Conversation between a User and two Assistants|>"
        tail = "<|The End of Conversation between a User and two Assistants|>\n"
        parts = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            
            prompt_encoded = self.tokenizer(prompt, return_offsets_mapping=True)
            response_a_encoded = self.tokenizer(response_a, return_offsets_mapping=True)
            response_b_encoded = self.tokenizer(response_b, return_offsets_mapping=True)

            if len(prompt_encoded['input_ids']) + len(response_a_encoded['input_ids']) + len(response_b_encoded['input_ids']) > self.max_length - 13:
                max_prompt_length = 1365
                prompt_len = min(len(prompt_encoded['input_ids']), max_prompt_length)
                # 剩余长度
                remain_length = self.max_length - 13 - prompt_len
                ratio = len(response_a_encoded['input_ids']) / (len(response_b_encoded['input_ids']) + len(response_a_encoded['input_ids']))
                remain_length_a = int(remain_length * ratio)
                remain_length_b = remain_length - remain_length_a

                prompt = self.tructure_text(prompt, prompt_encoded, prompt_len)
                response_a = self.tructure_text(response_a, response_a_encoded, remain_length_a)
                response_b = self.tructure_text(response_b, response_b_encoded, remain_length_b)
            parts.append(
                f"### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n\n### Assistant B:\n{response_b}"
            )
        text = "".join(parts)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return head + truncated_text + tail

    def build_input(self, data):
        conversation = self.build_conversation(
            [data["prompt"]],
            [data["response_a"]],
            [data["response_b"]],
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        label = torch.tensor([data[col] for col in self.LABEL_COLS]).float()
        return dict(
            input_ids=input_ids,
            input_text=input_text,
            label=label,
        )



class DISTMIDProcessorPAB(MIDProcessorPAB):
    def build_input(self, data):
        conversation = self.build_conversation(
            # json.loads(data["prompt"]),
            # json.loads(data["response_a"]),
            # json.loads(data["response_b"]),
            [data["prompt"]],
            [data["response_a"]],
            [data["response_b"]],
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        label = torch.tensor([data[col] for col in self.LABEL_COLS]).float()
        grad1 = torch.tensor(data["grad1"]).float()
        grad2 = torch.tensor(data["grad2"]).float()
        return dict(
            input_ids=input_ids,
            input_text=input_text,
            label=label,
            grad1=grad1,
            grad2=grad2,
        )


class ZDProcessorPAB(ProcessorPAB):
    PROMPT_PREFIX = """Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Be as objective as possible. Output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better."""
    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Conversation between a User and two Assistants|>"
        tail = "<|The End of Conversation between a User and two Assistants|>\n"
        parts = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            parts.append(
                f"""<PROMPT>{prompt}\n<assistant_A>\n{response_a}\n</assistant_A>\n<assistant_B>\n{response_b}\n</assistant_B>"""
            )
        text = "".join(parts)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return truncated_text

    def build_input(self, data):
        conversation = self.build_conversation(
            [data["prompt"]],
            [data["response_a"]],
            [data["response_b"]],
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        label = torch.tensor([data[col] for col in self.LABEL_COLS]).float()
        return dict(
            input_ids=input_ids,
            input_text=input_text,
            label=label,
        )    


class ProcessorPAPB(ProcessorPAB):
    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Assistant A’s Conversation with User|>"
        sep = "<|The End of Assistant A’s Conversation with User|>\n\n<|The Start of Assistant B’s Conversation with User|>"
        tail = "<|The End of Assistant B’s Conversation with User|>\n"
        parts_a = []
        parts_b = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            parts_a.append(f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n")
            parts_b.append(f"\n### User:\n{prompt}\n\n### Assistant B:\n{response_b}\n")
        text_a = "".join(parts_a)
        text_b = "".join(parts_b)
        input_ids_a = self.tokenizer(
            text_a,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids
        input_ids_b = self.tokenizer(
            text_b,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text_a = self.tokenizer.decode(input_ids_a)
        truncated_text_b = self.tokenizer.decode(input_ids_b)
        return head + truncated_text_a + sep + truncated_text_b + tail


class ProcessorPABChinese(ProcessorPAB):
    PROMPT_PREFIX = """请扮演一位公正的评判者,评估两位AI助手对下面显示的用户问题所提供的回答质量。你应选择更好地遵循用户指示并回答用户问题的助手。你的评估应考虑以下因素:回答的有用性、相关性、准确性、深度、创造性和详细程度。开始你的评估时,请比较这两个回答并提供简短解释。避免任何立场偏见,确保回答呈现的顺序不会影响你的决定。不要让回答的长度影响你的评估。不要偏爱某些助手的名字。尽可能保持客观。请严格按照以下格式输出你的最终判决，然后提供解释:"[[A]]"如果助手A更好,"[[B]]"如果助手B更好,以及"[[C]]"表示平局"""

    PROMPT_SUFFIX = "最终判决: [["

    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|对话开始|>"
        tail = "<|对话结束|>\n"
        parts = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            parts.append(
                f"\n### 用户:\n{prompt}\n\n### 助手 A:\n{response_a}\n\n### 助手 B:\n{response_b}\n"
            )
        text = "".join(parts)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return head + truncated_text + tail
