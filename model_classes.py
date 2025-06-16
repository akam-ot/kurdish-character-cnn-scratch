import numpy as np
import pickle

# ================================================================================
# FAST, VECTORIZED HELPER FUNCTIONS 
# ================================================================================

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

# ================================================================================
# CNN LAYERS
# ================================================================================

class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=1):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        fan_in = input_channels * kernel_size * kernel_size
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros(output_channels)
        
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

    def forward(self, X):
        self.X_shape = X.shape
        N, C, H, W = X.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        X_col = im2col_indices(X, self.kernel_size, self.kernel_size, self.padding, self.stride)
        W_col = self.weights.reshape(self.output_channels, -1)
        
        out = W_col @ X_col + self.bias.reshape(-1, 1)
        out = out.reshape(self.output_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        
        self.X_col = X_col
        return out

    def backward(self, dout):
        N, C, H, W = self.X_shape
        
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.output_channels, -1)
        
        self.dW = (dout_reshaped @ self.X_col.T).reshape(self.weights.shape)
        self.db = np.sum(dout_reshaped, axis=1)
        
        W_reshape = self.weights.reshape(self.output_channels, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col, self.X_shape, self.kernel_size, self.kernel_size, self.padding, self.stride)
        
        return dX

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        out_h = H // self.stride
        out_w = W // self.stride
        
        X_reshaped = X.reshape(N * C, 1, H, W)
        X_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride)
        
        self.max_idx = np.argmax(X_col, axis=0)
        out = X_col[self.max_idx, range(self.max_idx.size)]
        
        out = out.reshape(out_h, out_w, N, C)
        out = out.transpose(2, 3, 0, 1)
        
        return out

    def backward(self, dout):
        N, C, H, W = self.X.shape
        
        dX_col = np.zeros((self.pool_size * self.pool_size, dout.size))
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        
        dX_col[self.max_idx, range(self.max_idx.size)] = dout_flat
        
        dX = col2im_indices(dX_col, (N * C, 1, H, W), self.pool_size, self.pool_size, padding=0, stride=self.stride)
        dX = dX.reshape(self.X.shape)
        
        return dX

class ReLULayer:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout):
        return dout * (self.X > 0)

class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

    def forward(self, X):
        self.X = X
        return X @ self.weights + self.bias

    def backward(self, dout):
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.weights.T

# ================================================================================
# COMPLETE CNN MODEL
# ================================================================================

class CNNModel:
    def __init__(self, num_classes=35):
        self.conv1 = ConvLayer(1, 32, 3, stride=1, padding=1)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolLayer(2, 2)
        
        self.conv2 = ConvLayer(32, 64, 3, stride=1, padding=1)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolLayer(2, 2)
        
        self.fc1 = FCLayer(64 * 8 * 8, 128)
        self.relu3 = ReLULayer()
        self.fc2 = FCLayer(128, num_classes)
        
        self.num_classes = num_classes

    def forward(self, X):
        out = self.pool1.forward(self.relu1.forward(self.conv1.forward(X)))
        out = self.pool2.forward(self.relu2.forward(self.conv2.forward(out)))
        
        self.flatten_input_shape = out.shape
        out = out.reshape(out.shape[0], -1)
        
        out = self.relu3.forward(self.fc1.forward(out))
        out = self.fc2.forward(out)
        
        return self.softmax(out)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss(self, probs, y):
        m = probs.shape[0]
        log_likelihood = -np.log(probs[range(m), y] + 1e-8)
        return np.sum(log_likelihood) / m

    def backward(self, probs, y):
        m = probs.shape[0]
        
        # Output layer gradient
        grad = probs.copy()
        grad[range(m), y] -= 1
        grad /= m
        
        # Fully connected layers
        grad = self.fc2.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.fc1.backward(grad)
        
        # Reshape back to conv output shape
        grad = grad.reshape(self.flatten_input_shape)
        
        # Convolutional layers
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

    def save_model(self, filepath):
        """Save the trained model weights to a file."""
        model_data = {
            'conv1_weights': self.conv1.weights,
            'conv1_bias': self.conv1.bias,
            'conv2_weights': self.conv2.weights,
            'conv2_bias': self.conv2.bias,
            'fc1_weights': self.fc1.weights,
            'fc1_bias': self.fc1.bias,
            'fc2_weights': self.fc2.weights,
            'fc2_bias': self.fc2.bias,
            'num_classes': self.num_classes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to: {filepath}")

    def load_model(self, filepath):
        """Load model weights from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.conv1.weights = model_data['conv1_weights']
        self.conv1.bias = model_data['conv1_bias']
        self.conv2.weights = model_data['conv2_weights']
        self.conv2.bias = model_data['conv2_bias']
        self.fc1.weights = model_data['fc1_weights']
        self.fc1.bias = model_data['fc1_bias']
        self.fc2.weights = model_data['fc2_weights']
        self.fc2.bias = model_data['fc2_bias']
        self.num_classes = model_data['num_classes']
        print(f"✅ Model loaded from: {filepath}")
