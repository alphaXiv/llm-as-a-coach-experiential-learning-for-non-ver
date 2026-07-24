# Experiment configuration. Arms/seeds vary ONLY via child-branch edits here.
ARM = "el"
SEED = 0

POLICY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
COACH_MODEL = "Qwen/Qwen2.5-7B-Instruct"       # frozen train-time judge+coach (all arms)
EVAL_JUDGE_MODEL = "Qwen/Qwen3-8B"             # held-out eval judge

# Data sizes (split is fixed with seed 0 for every arm/seed)
N_TRAIN_PROMPTS = 512
N_HELDOUT = 128        # in-domain held-out (no_robots)
N_ALPACA = 100         # OOD: AlpacaEval subset
N_DOLLY = 64           # OOD: Dolly creative_writing subset

# Training (matched across all arms)
PROMPTS_PER_STEP = 16
SAMPLES_PER_PROMPT = 4
TRAIN_STEPS = 64       # = 2 epochs over 512 prompts
LR = 2e-5
LORA_R = 32
LORA_ALPHA = 64
MICRO_BATCH = 8
GRAD_CLIP = 1.0
TOPK_KL = 256          # reverse-KL over student's top-256 tokens (paper's choice)
TRAIN_TEMP = 0.8
MAX_PROMPT_TOKENS = 512
MAX_NEW_TOKENS = 384
MAX_TRAIN_SECONDS = 8100   # hard wall guard: stop training, still run eval

# Eval
EVAL_TEMP = 0.7
EVAL_MAX_NEW_TOKENS = 384

COACH_PORT = 8000
COACH_GPU_FRAC = 0.35
JUDGE_GPU_FRAC = 0.4
