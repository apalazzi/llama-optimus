# src/llama_optimus/search_space.py
import os
from .override_patterns import OVERRIDE_PATTERNS  # if needed

# count number of available cpu cores
max_threads = os.cpu_count()

# Cache type combinations: (cache_type_k, cache_type_v) tuples
# Only test specific combinations to avoid redundant trials
CACHE_COMBINATIONS = {
    'f16_f16': ('f16', 'f16'),
    'q8_0_q8_0': ('q8_0', 'q8_0'),
    'q8_0_q5_1': ('q8_0', 'q5_1'),
    'q4_0_q4_0': ('q4_0', 'q4_0'),
}

SEARCH_SPACE = {
    'batch_size'     : {'low': 8, 'high': 16384},   # 
    'ubatch_size'    : {'low': 4, 'high': 8192},    #  
    'threads':    {'low': 1, 'high': max_threads},  # Adjust range to your hardware
    'gpu_layers': {'low': 0, 'high': 149},          # (-ngl) Set max according to model and VRAM; The max value must be determined for each setup
    'flash_attn': 1,                            # Always enabled; use --no-flash-attn CLI flag to disable       
    'override_spc'   : list(OVERRIDE_PATTERNS.keys()), # Read list from src/llama_optimus/override_patterns.py
    'cache_type'     : list(CACHE_COMBINATIONS.keys())  # Cache type combinations
    #'flash_attn_type': [0, 1, 2], # Not yet merged to main llama.cpp
}
