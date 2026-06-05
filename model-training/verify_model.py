# Mock script to verify model(1).py shape propagation and parameter count

import sys

# Mocking tensorflow and keras
class MockTensor:
    def __init__(self, shape, name=""):
        self.shape = shape
        self.name = name

class MockLayer:
    def __init__(self, name=""):
        self.name = name
    def __call__(self, *args, **kwargs):
        return args[0]

# Keep track of global parameters
total_params = 0

def add_params(count, desc=""):
    global total_params
    total_params += count
    # print(f"  [Param Log] {desc}: +{count:,} (Total: {total_params:,})")

class Input:
    def __init__(self, shape, name=None):
        self.shape = (None,) + shape
        self.name = name
    def __call__(self, *args):
        return MockTensor(self.shape, self.name)

class Conv2D:
    def __init__(self, filters, kernel_size, strides=1, padding='same', use_bias=True, name=None):
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.use_bias = use_bias
        self.name = name
    def __call__(self, x):
        in_channels = x.shape[-1]
        out_channels = self.filters
        kh, kw = self.kernel_size
        sh, sw = self.strides
        
        # calculate parameters
        p_conv = kh * kw * in_channels * out_channels
        if self.use_bias:
            p_conv += out_channels
        add_params(p_conv, f"Conv2D ({kh}x{kw}, {in_channels}->{out_channels})")
        
        # calculate output shape
        batch, h, w, c = x.shape
        if h is not None:
            new_h = (h + 2*(kh//2) - kh) // sh + 1 if h is not None else None
            new_w = (w + 2*(kw//2) - kw) // sw + 1 if w is not None else None
        else:
            new_h, new_w = None, None
        
        out_shape = (batch, new_h, new_w, out_channels)
        # print(f"Conv2D output shape: {out_shape}")
        return MockTensor(out_shape)

class BatchNormalization:
    def __init__(self):
        pass
    def __call__(self, x):
        channels = x.shape[-1]
        # scale, bias, mean, variance -> 4 parameters per channel
        add_params(4 * channels, f"BatchNormalization ({channels} channels)")
        return x

class ReLU:
    def __init__(self, max_value=None):
        pass
    def __call__(self, x):
        return x

class DepthwiseConv2D:
    def __init__(self, kernel_size, strides=1, padding='same', use_bias=True):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.use_bias = use_bias
    def __call__(self, x):
        in_channels = x.shape[-1]
        kh, kw = self.kernel_size
        sh, sw = self.strides
        
        # calculate parameters
        p_conv = kh * kw * in_channels
        if self.use_bias:
            p_conv += in_channels
        add_params(p_conv, f"DepthwiseConv2D ({kh}x{kw}, channels={in_channels})")
        
        # calculate output shape
        batch, h, w, c = x.shape
        if h is not None:
            new_h = (h + 2*(kh//2) - kh) // sh + 1
            new_w = (w + 2*(kw//2) - kw) // sw + 1
        else:
            new_h, new_w = None, None
            
        out_shape = (batch, new_h, new_w, in_channels)
        return MockTensor(out_shape)

class GlobalAveragePooling2D:
    def __init__(self):
        pass
    def __call__(self, x):
        batch, h, w, c = x.shape
        return MockTensor((batch, c))

class Reshape:
    def __init__(self, target_shape):
        self.target_shape = target_shape
    def __call__(self, x):
        batch = x.shape[0]
        return MockTensor((batch,) + self.target_shape)

class Multiply:
    def __init__(self):
        pass
    def __call__(self, inputs):
        # inputs: [x, se] where x is (B, H, W, C) and se is (B, 1, 1, C)
        x, se = inputs
        # verify shape compatibility
        assert x.shape[-1] == se.shape[-1], f"Channels must match, got {x.shape} and {se.shape}"
        return x

class Dense:
    def __init__(self, units, activation=None, use_bias=True, name=None):
        self.units = units
        self.use_bias = use_bias
        self.name = name
    def __call__(self, x):
        in_features = x.shape[-1]
        p_dense = in_features * self.units
        if self.use_bias:
            p_dense += self.units
        add_params(p_dense, f"Dense ({in_features}->{self.units})")
        
        batch = x.shape[0]
        # output shape preserves all other dimensions except the last one which is units
        return MockTensor(x.shape[:-1] + (self.units,))

class Add:
    def __init__(self):
        pass
    def __call__(self, inputs):
        x1, x2 = inputs
        assert x1.shape == x2.shape, f"Shapes must be identical for residual Add, got {x1.shape} and {x2.shape}"
        return x1

class Dropout:
    def __init__(self, rate):
        pass
    def __call__(self, x):
        return x

class Model:
    def __init__(self, inputs, outputs, name=None):
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
    def count_params(self):
        return total_params

# Inject mock classes into sys.modules to simulate tensorflow/keras import
import types
keras_layers = types.ModuleType("keras.layers")
keras_layers.Input = Input
keras_layers.Conv2D = Conv2D
keras_layers.BatchNormalization = BatchNormalization
keras_layers.ReLU = ReLU
keras_layers.DepthwiseConv2D = DepthwiseConv2D
keras_layers.GlobalAveragePooling2D = GlobalAveragePooling2D
keras_layers.Reshape = Reshape
keras_layers.Multiply = Multiply
keras_layers.Dense = Dense
keras_layers.Add = Add
keras_layers.Dropout = Dropout

keras = types.ModuleType("keras")
keras.layers = keras_layers
keras.Model = Model

tensorflow = types.ModuleType("tensorflow")

sys.modules["tensorflow"] = tensorflow
sys.modules["keras"] = keras
sys.modules["keras.layers"] = keras_layers

# Now import build_face_detection_model from model(1).py
model = __import__("model(1)")

print("="*60)
print("  Running Face Detector Model Verification (Mock Keras)")
print("="*60)

try:
    # Build model using input size 128
    inputs = Input((128, 128, 3))()
    
    # We will build the model by executing the function
    model_obj = model.build_face_detection_model(input_size=128)
    
    print("\nModel Built Successfully!")
    print(f"Total calculated parameters: {model_obj.count_params():,}")
    print(f"Quantized INT8 model size:   {model_obj.count_params() / 1024 / 1024:.3f} MB")
    
    # Print shapes
    print("\nOutputs:")
    for out in model_obj.outputs:
        print(f"  {out.name if hasattr(out, 'name') else 'output'}: shape {out.shape}")
        
except Exception as e:
    print(f"Verification failed with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
