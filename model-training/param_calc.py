# Parameter search script for NHAI face detection model

def bn_params(c):
    return 4 * c

def conv2d_params(in_c, out_c, k=3, use_bias=False):
    params = k * k * in_c * out_c
    if use_bias:
        params += out_c
    return params

def dw_conv2d_params(c, k=3, use_bias=False):
    params = k * k * c
    if use_bias:
        params += c
    return params

def dense_params(in_c, out_c, use_bias=True):
    params = in_c * out_c
    if use_bias:
        params += out_c
    return params

def mbconv_params(in_c, out_c, k=3, stride=1, expansion=4, use_se=True):
    total = 0
    hidden_dim = in_c * expansion
    
    # 1. Expansion (1x1 conv)
    if expansion != 1:
        total += conv2d_params(in_c, hidden_dim, k=1, use_bias=False)
        total += bn_params(hidden_dim) # BN
        
    # 2. Depthwise Conv
    total += dw_conv2d_params(hidden_dim, k=k, use_bias=False)
    total += bn_params(hidden_dim) # BN
    
    # 3. SE Block
    if use_se:
        se_channels = max(1, hidden_dim // 4)
        total += dense_params(hidden_dim, se_channels, use_bias=True)
        total += dense_params(se_channels, hidden_dim, use_bias=True)
        
    # 4. Projection (1x1 conv)
    total += conv2d_params(hidden_dim, out_c, k=1, use_bias=False)
    total += bn_params(out_c) # BN
    
    return total

def calculate_params(cfg):
    total = 0
    # Stem: Conv2D(3, stem_c, k=3, stride=2)
    stem_c = cfg['stem_c']
    total += conv2d_params(3, stem_c, k=3)
    total += bn_params(stem_c)
    
    current_c = stem_c
    
    for block in cfg['blocks']:
        b_type = block.get('type', 'mbconv')
        out_c = block['c']
        k = block.get('k', 3)
        stride = block.get('stride', 1)
        
        if b_type == 'mbconv':
            expansion = block.get('expansion', 3)
            use_se = block.get('se', True)
            total += mbconv_params(current_c, out_c, k=k, stride=stride, expansion=expansion, use_se=use_se)
        elif b_type == 'ds':
            # depthwise separable
            total += dw_conv2d_params(current_c, k=k)
            total += bn_params(current_c)
            total += conv2d_params(current_c, out_c, k=1)
            total += bn_params(out_c)
            
        current_c = out_c
        
    # Dense head
    dense_h = cfg['dense_h']
    total += dense_params(current_c, dense_h)
    
    # Outputs
    total += dense_params(dense_h, 4)
    total += dense_params(dense_h, 10)
    total += dense_params(dense_h, 1)
    
    return total

# Define candidate architectures
candidates = []

# Candidate A: MBConv with low expansion and moderate channels
candidates.append({
    'name': 'Candidate A (Light MBConv + SE)',
    'stem_c': 16,
    'blocks': [
        {'type': 'mbconv', 'c': 24, 'k': 3, 'stride': 1, 'expansion': 3, 'se': True},
        {'type': 'mbconv', 'c': 32, 'k': 3, 'stride': 2, 'expansion': 3, 'se': True}, # 32x32
        {'type': 'mbconv', 'c': 48, 'k': 3, 'stride': 1, 'expansion': 3, 'se': True},
        {'type': 'mbconv', 'c': 64, 'k': 3, 'stride': 2, 'expansion': 3, 'se': True}, # 16x16
        {'type': 'mbconv', 'c': 96, 'k': 3, 'stride': 1, 'expansion': 3, 'se': True},
        {'type': 'mbconv', 'c': 128, 'k': 5, 'stride': 2, 'expansion': 3, 'se': True}, # 8x8
        {'type': 'mbconv', 'c': 128, 'k': 5, 'stride': 1, 'expansion': 3, 'se': True},
    ],
    'dense_h': 128
})

# Candidate B: MBConv with slightly more blocks, no SE in early blocks to save params
candidates.append({
    'name': 'Candidate B (Mixed SE + optimized MBConv)',
    'stem_c': 16,
    'blocks': [
        {'type': 'mbconv', 'c': 24, 'k': 3, 'stride': 1, 'expansion': 2, 'se': False},
        {'type': 'mbconv', 'c': 32, 'k': 3, 'stride': 2, 'expansion': 3, 'se': False}, # 32x32
        {'type': 'mbconv', 'c': 48, 'k': 3, 'stride': 1, 'expansion': 3, 'se': True},
        {'type': 'mbconv', 'c': 64, 'k': 3, 'stride': 2, 'expansion': 4, 'se': True},  # 16x16
        {'type': 'mbconv', 'c': 96, 'k': 3, 'stride': 1, 'expansion': 4, 'se': True},
        {'type': 'mbconv', 'c': 128, 'k': 5, 'stride': 2, 'expansion': 4, 'se': True}, # 8x8
        {'type': 'mbconv', 'c': 160, 'k': 5, 'stride': 1, 'expansion': 4, 'se': True},
    ],
    'dense_h': 160
})

# Candidate C: An even wider candidate targeting exactly ~400k-450k
candidates.append({
    'name': 'Candidate C (Robust MBConv V3 - Optimized)',
    'stem_c': 16,
    'blocks': [
        {'type': 'mbconv', 'c': 24, 'k': 3, 'stride': 1, 'expansion': 2, 'se': False}, # 64x64
        {'type': 'mbconv', 'c': 32, 'k': 3, 'stride': 2, 'expansion': 3, 'se': False}, # 32x32
        {'type': 'mbconv', 'c': 48, 'k': 3, 'stride': 1, 'expansion': 3, 'se': True},  # 32x32
        {'type': 'mbconv', 'c': 64, 'k': 3, 'stride': 2, 'expansion': 4, 'se': True},  # 16x16
        {'type': 'mbconv', 'c': 96, 'k': 3, 'stride': 1, 'expansion': 4, 'se': True},  # 16x16
        {'type': 'mbconv', 'c': 128, 'k': 5, 'stride': 2, 'expansion': 4, 'se': True}, # 8x8
        {'type': 'mbconv', 'c': 144, 'k': 5, 'stride': 1, 'expansion': 4, 'se': True}, # 8x8
    ],
    'dense_h': 128
})

# Let's write a loop to search channel sizes for a Candidate D
# which has 7 blocks:
# Block 1: 16 -> 24 (s=1)
# Block 2: 24 -> 32 (s=2)
# Block 3: 32 -> 48 (s=1)
# Block 4: 48 -> 64 (s=2)
# Block 5: 64 -> 96 (s=1)
# Block 6: 96 -> 128 (s=2)
# Block 7: 128 -> C7 (s=1)
# We can search C7 in [128, 144, 160], and dense_h in [128, 160, 192, 256]

for c7 in [128, 144, 160]:
    for dense_h in [128, 160, 256]:
        for exp in [3, 4]:
            cfg = {
                'name': f'Search MBConv (c7={c7}, dense_h={dense_h}, exp={exp})',
                'stem_c': 16,
                'blocks': [
                    {'type': 'mbconv', 'c': 24, 'k': 3, 'stride': 1, 'expansion': 2, 'se': False},
                    {'type': 'mbconv', 'c': 32, 'k': 3, 'stride': 2, 'expansion': exp, 'se': False},
                    {'type': 'mbconv', 'c': 48, 'k': 3, 'stride': 1, 'expansion': exp, 'se': True},
                    {'type': 'mbconv', 'c': 64, 'k': 3, 'stride': 2, 'expansion': exp, 'se': True},
                    {'type': 'mbconv', 'c': 96, 'k': 3, 'stride': 1, 'expansion': exp, 'se': True},
                    {'type': 'mbconv', 'c': 128, 'k': 5, 'stride': 2, 'expansion': exp, 'se': True},
                    {'type': 'mbconv', 'c': c7, 'k': 5, 'stride': 1, 'expansion': exp, 'se': True},
                ],
                'dense_h': dense_h
            }
            p = calculate_params(cfg)
            if 350000 <= p <= 490000:
                print(f"{cfg['name']}: {p:,} params (~{p/1024/1024:.3f} MB INT8)")

print("\n--- Main Candidates ---")
for cand in candidates:
    p = calculate_params(cand)
    print(f"{cand['name']}: {p:,} params (~{p/1024/1024:.3f} MB INT8)")
