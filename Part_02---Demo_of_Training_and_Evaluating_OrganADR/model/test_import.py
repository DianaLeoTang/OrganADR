print("=== Start ===")

try:
    print("1. Testing torch import...")
    import torch
    print(f"   torch version: {torch.__version__}")
    
    print("2. Testing torchdrug import...")
    from torchdrug.layers.functional import spmm as _spmm
    print("   torchdrug OK")
    
    print("3. Testing other imports...")
    import pickle
    import numpy as np
    print("   All imports OK")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=== End ===")