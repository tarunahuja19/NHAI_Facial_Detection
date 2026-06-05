# Param calculator incorporating new guidelines:
# 1. No GAP bottleneck (use Conv and Dense).
# 2. Residual skip connections in blocks.
# 3. Final feature map size: 16x16x192.
# 4. Separate heads earlier (separate branch for Box, Landmarks, Confidence).

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

def ds_conv_params(in_c, out_c, k=3):
    # Depthwise separable convolution parameters
    total = dw_conv2d_params(in_c, k=k) + bn_params(in_c)
    total += conv2d_params(in_c, out_c, k=1) + bn_params(out_c)
    return total

def mbconv_params(in_c, out_c, k=3, expansion=3, use_se=True):
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

def calculate_full_model(cfg):
    total = 0
    # Stem: Conv2D(3, stem_c, k=3, stride=2)
    stem_c = cfg['stem_c']
    total += conv2d_params(3, stem_c, k=3)
    total += bn_params(stem_c)
    
    current_c = stem_c
    
    # Backbone blocks
    for b in cfg['backbone_blocks']:
        total += mbconv_params(current_c, b['c'], k=b.get('k', 3), expansion=b.get('exp', 3), use_se=b.get('se', True))
        current_c = b['c']
        
    assert current_c == 192, f"Expected final backbone channels to be 192, got {current_c}"
    
    # Heads branches split early from 16x16x192
    
    # 1. Box Branch
    # Conv to reduce channels and spatial size from 16x16x192 -> 8x8x(box_c)
    box_c = cfg['box_branch']['c']
    total += ds_conv_params(192, box_c, k=3)
    box_flat_size = 8 * 8 * box_c
    total += dense_params(box_flat_size, cfg['box_branch']['dense_h'])
    total += dense_params(cfg['box_branch']['dense_h'], 4)
    
    # 2. Landmark Branch
    # Conv to reduce channels and spatial size from 16x16x192 -> 8x8x(lm_c)
    lm_c = cfg['lm_branch']['c']
    total += ds_conv_params(192, lm_c, k=3)
    lm_flat_size = 8 * 8 * lm_c
    total += dense_params(lm_flat_size, cfg['lm_branch']['dense_h'])
    total += dense_params(cfg['lm_branch']['dense_h'], 10)
    
    # 3. Confidence Branch
    # Conv to reduce channels and spatial size from 16x16x192 -> 8x8x(conf_c)
    conf_c = cfg['conf_branch']['c']
    total += ds_conv_params(192, conf_c, k=3)
    conf_flat_size = 8 * 8 * conf_c
    total += dense_params(conf_flat_size, cfg['conf_branch']['dense_h'])
    total += dense_params(cfg['conf_branch']['dense_h'], 1)
    
    # 4. Optional Multi-face Head (since user requested multi-face detection flag)
    # We can share confidence features or landmark features, or give it its own branch.
    # Let's give it its own branch (or predict from confidence branch).
    # If it has its own branch:
    multi_c = cfg['multi_branch']['c']
    total += ds_conv_params(192, multi_c, k=3)
    multi_flat_size = 8 * 8 * multi_c
    total += dense_params(multi_flat_size, cfg['multi_branch']['dense_h'])
    total += dense_params(cfg['multi_branch']['dense_h'], 1)
    
    return total

# Let's search for configurations that fit the parameter budget (under 500k parameters)
# Backbone Blocks:
# 1. 16 -> 24 (stride 1)
# 2. 24 -> 32 (stride 2)
# 3. 32 -> 48 (stride 1)
# 4. 48 -> 96 (stride 2)  -> 16x16
# 5. 96 -> 144 (stride 1) -> 16x16
# 6. 144 -> 192 (stride 1) -> 16x16

backbone_cfg = [
    {'c': 24, 'k': 3, 'exp': 2, 'se': False}, # 64x64
    {'c': 32, 'k': 3, 'exp': 3, 'se': False}, # 32x32
    {'c': 48, 'k': 3, 'exp': 3, 'se': True},  # 32x32
    {'c': 96, 'k': 3, 'exp': 3, 'se': True},  # 16x16
    {'c': 144, 'k': 3, 'exp': 3, 'se': True}, # 16x16
    {'c': 192, 'k': 5, 'exp': 2, 'se': True}, # 16x16 (uses k=5 and exp=2 to save params!)
]

# Calculate backbone params
backbone_params = conv2d_params(3, 16, k=3) + bn_params(16)
c = 16
for b in backbone_cfg:
    backbone_params += mbconv_params(c, b['c'], k=b['k'], expansion=b['exp'], use_se=b['se'])
    c = b['c']
print(f"Backbone params: {backbone_params:,}")

# Let's test different configurations of heads
for box_c in [16, 24, 32]:
    for lm_c in [16, 24, 32]:
        for conf_c in [8, 16]:
            for multi_c in [8, 16]:
                for h_dense in [16, 32, 64]:
                    cfg = {
                        'stem_c': 16,
                        'backbone_blocks': backbone_cfg,
                        'box_branch': {'c': box_c, 'dense_h': h_dense},
                        'lm_branch': {'c': lm_c, 'dense_h': h_dense},
                        'conf_branch': {'c': conf_c, 'dense_h': h_dense // 2 if h_dense > 16 else 16},
                        'multi_branch': {'c': multi_c, 'dense_h': h_dense // 2 if h_dense > 16 else 16}
                    }
                    p = calculate_full_model(cfg)
                    if 350000 <= p <= 490000:
                        print(f"Fit: box_c={box_c}, lm_c={lm_c}, conf_c={conf_c}, multi_c={multi_c}, h_dense={h_dense} -> {p:,} params (~{p/1024/1024:.3f} MB INT8)")
