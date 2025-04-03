IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} \n{input} [/INST]"
    ),
    "peersum_prompt": (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. You are given a paper abstract and a set of peer review messages. "
        "Your goal is to produce a coherent meta-review that summarizes the key points and assessments from the provided reviews.\n"
        "<</SYS>>\n\n"
        "### Paper Abstract:\n{paper_abstract}\n\n"
        "### Reviews:\n{review_contents}\n\n"
        "### Meta Review: [/INST]"
    ),
    "metagen_prompt": (
        "[INST] <<SYS>>\n"
        "You are an AC for {year} {conference} {subject}. "
        "Provide a meta-review for the paper titled \"{title}\".\n"
        "<</SYS>>\n\n"
        "Abstract of the paper:\n{abstract}\n\n"
        "{review_threads}\n"
        "{generate_instruction}\n[/INST]"
    ),
}