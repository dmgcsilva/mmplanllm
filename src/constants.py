DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


OA_PROMPT_DICT = (
    "<|prompter|> I will give you a recipe and I want you to help me do it step by step. Please use a {system_tone} tone of voice. Recipe: {recipe} This is the current step: {current_step}. <|endoftext|> <|assistant|> ok! <|endoftext|> {dialog} <|endoftext|> <|assistant|>"
)

VICUNA_PROMPT_DICT = (
    "{user_token} I will give you a recipe and I want you to help me do it step by step. Please use a {system_tone} "
    "tone of voice. Recipe: {recipe} {current_step} {sep_token} {sys_token} ok! {sep_token} {dialog} {sys_token} "
)

CURRENT_STEP_TEMP = "We are on Step {step_num}: {step_text}"
NO_STEP_TEMP = "We are just starting the recipe"



CONTEXT_WINDOW = 3

# ======== DEFAULT ARGUMENTS ========
DEFAULT_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_STOP_CRITERIA = -1.0
DEFAULT_SEED = 11731
DEFAULT_LR_STEP_SIZE = 250
DEFAULT_LR_NUM_CYCLES = 10
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_LEN = 512
DEFAULT_GRAD_CLIP = 0.5
