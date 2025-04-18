# ################################
# Simple Operations
# ################################
class Negative:
    def forward(self, x):
        self.cache = x
        return -x

    def backward(self, d_out):
        d_x = -d_out
        return d_x

class Add:
    def forward(self, x, y):
        return x + y
    def backward(self, d_out):
        d_x = d_out
        d_y = d_out
        return d_x, d_y
    
class Multiply:
    def forward(self, x, y):
        self.cache = (x, y)
        return x * y
    def backward(self, d_out):
        x, y = self.cache
        d_x = d_out * y
        d_y = d_out * x
        return d_x, d_y
        
class Divide:
    def forward(self, x, y):
        self.cache = (x, y)
        return x / y
    
    def backward(self, d_out):
        x, y = self.cache
        d_x = d_out * (1.0 / y)
        d_y = d_out * (-x / (y**2))
        return d_x, d_y


class Sum:
    def forward(self, x):
        self.cache = x
        return np.sum(x)
    def backward(self, d_out):
        x = self.cache
        d_x = d_out * np.ones_like(x) # !
        return d_x
        
class Subtract:
    def forward(self, x, y):
        return x - y
    
    def backward(self, d_out):
        d_x = d_out        # derivative of x - y w.r.t. x is +1
        d_y = -d_out       # derivative of x - y w.r.t. y is -1
        return d_x, d_y
        
class DotProduct:
    def forward(self, x, y):
        self.cache = x, y
        return np.dot(x, y)
    def backward(self, d_out):
        x, y = self.cache
        d_x = d_out * y
        d_y = d_out * x
        return d_x, d_y
        
class Exp:
    def forward(self, x):
        self.cache = x
        out = np.exp(x)
        self.out = out
        return out
    
    def backward(self, d_out):
        # d/dx exp(x) = exp(x)
        return d_out * self.out

class Log:
    def forward(self, x):
        self.cache = x
        return np.log(x)
    
    def backward(self, d_out):
        x = self.cache
        return d_out * (1.0 / x)
        
class Power:
    def __init__(self, p):
        self.p = p

    def forward(self, x):
        self.cache = x
        return x ** self.p

    def backward(self, d_out):
        x = self.cache
        p = self.p
        # derivative: p * x^(p-1)
        return d_out * (p * (x ** (p - 1)))

class Max:
    def forward(self, x, y):
        # store which is bigger
        self.x = x
        self.y = y
        if x >= y:
            self.output_is_x = True
            return x
        else:
            self.output_is_x = False
            return y
    
    def backward(self, d_out):
        if self.output_is_x:
            d_x = d_out
            d_y = 0.0
        else:
            d_x = 0.0
            d_y = d_out
        return d_x, d_y

class Min:
    def forward(self, x, y):
        self.x = x
        self.y = y
        if x <= y:
            self.output_is_x = True
            return x
        else:
            self.output_is_x = False
            return y
    
    def backward(self, d_out):
        if self.output_is_x:
            d_x = d_out
            d_y = 0.0
        else:
            d_x = 0.0
            d_y = d_out
        return d_x, d_y

class Clip:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        self.cache = x
        out = np.clip(x, self.min_val, self.max_val)
        self.out = out
        return out
    
    def backward(self, d_out):
        x = self.cache
        min_mask = (x < self.min_val)
        max_mask = (x > self.max_val)
        pass_mask = ~(min_mask | max_mask)  # region where gradient passes

        d_x = d_out * pass_mask  # zero out gradient for clipped values
        return d_x

# ################################
# Activation Functions:
# ################################
class ReLU:
    def forward(self, inputs: np.ndarray):
        self.cache = inputs
        out = np.maximum(0, inputs)
        return out
    def backward(self, d_out: np.ndarray):

        inputs = self.cache
        d_inputs = d_out * (inputs > 0).astype(int)
        return d_inputs

class Sigmoid:
    def forward(self, x):
        self.cache = x
        out = 1.0 / (1.0 + np.exp(-x))
        self.output = out
        return out

    def backward(self, d_out):
        out = self.output
        d_x = d_out * out * (1.0 - out)  
        return d_x

class Tanh:
    def forward(self, x):
        self.cache = x
        out = np.tanh(x)
        self.output = out
        return out

    def backward(self, d_out):
        out = self.output
        d_x = d_out * (1.0 - out**2)
        return d_x

class Softmax:
    def forward(self, x):
        # x shape: (N, D)
        # subtract max for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.output = out
        return out

    def backward(self, d_out):
        """
        d_out: gradient wrt the Softmax output, shape (N, D)
        Returns the gradient wrt x, shape (N, D)
        
        In detail, for each row n:
          d_x[n] = (Jacobian of softmax at x[n]) * d_out[n]
        
        Where the Jacobian is:
          J[i, j] = out[i] * (δ[i, j] - out[j])
        """
        out = self.output
        N, D = out.shape
        d_x = np.zeros_like(d_out)
        
        # For each sample in the batch, apply the Jacobian
        for n in range(N):
            y = out[n]           # shape (D,)
            grad_out_n = d_out[n]  # shape (D,)
            # Jacobian-vector product with the softmax Jacobian
            # J*v = y * (v - (y dot v))
            # (Equivalent to the well-known formula for the derivative of softmax)
            dot = np.sum(y * grad_out_n)
            d_x[n] = y * (grad_out_n - dot)
        
        return d_x
    
# ################################
# Loss Functions:
# ################################
class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        """
        y_pred: shape (N, D), predictions (e.g. from softmax)
        y_true: shape (N, D), one-hot or distribution
        Returns the average loss scalar
        """
        self.cache = (y_pred, y_true)
        # To avoid log(0), clip y_pred
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]
        return loss

    def backward(self):
        """
        Returns gradient wrt y_pred
        """
        y_pred, y_true = self.cache
        N = y_pred.shape[0]
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        
        d_y_pred = - (y_true / y_pred_clipped) / N
        return d_y_pred

class MSELoss:
    def forward(self, y_pred, y_true):
        """
        y_pred: shape (N, D)
        y_true: shape (N, D)
        Returns scalar
        """
        self.cache = (y_pred, y_true)
        diff = y_pred - y_true
        loss = np.mean(diff**2)
        return loss
    
    def backward(self):
        """
        Returns gradient wrt y_pred, shape (N, D)
        """
        y_pred, y_true = self.cache
        N = y_pred.shape[0]
        d_y_pred = (2.0 / N) * (y_pred - y_true)
        return d_y_pred
    
# ################################
# Transform Functions:
# ################################
class Flatten:
    def forward(self, x):
        """
        x shape: (N, C, H, W)
        returns shape: (N, C*H*W)
        """
        self.cache = x.shape
        out = x.reshape(x.shape[0], -1)
        return out

    def backward(self, d_out):
        """
        d_out shape: (N, C*H*W)
        returns shape: (N, C, H, W)
        """
        original_shape = self.cache
        d_x = d_out.reshape(original_shape)
        return d_x
# ################################
# Regularizer:
# ################################
class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def forward(self, input):
        if self.training:
            mask = input.new_empty(input.shape)
            mask.bernoulli_(1 - self.p)
            scaling = 1 / (1 - self.p)
            return scaling * mask * input
        else:
            return input


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x):
        if self.training:
            # sample a mask from a Bernoulli(1-p)
            self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
            # scale outputs by 1/(1-p) to keep expected values consistent
            out = x * self.mask / (1.0 - self.p)
            return out
        else:
            # no dropout at test time
            return x

    def backward(self, d_out):
        if self.training:
            d_x = d_out * self.mask / (1.0 - self.p)
            return d_x
        else:
            # pass the gradient as is during inference
            return d_out


# ################################
# Architectures:
# ################################

class Affine:
    def forward(self, inputs, weight, bias):
        self.cache = (inputs, weight, bias) # (N, D), (D, H), (H)
        
        out = inputs @ weight + bias # out = inputs.dot(weight) + bias
        
        return out # (N, H)
    def backward(self, d_out):
        inputs, weight, bias = self.cache # (N, H)
        
        d_inputs = d_out @ weight.T # (N, D)
        d_weight = inputs.T @ d_out # (D, H)
        d_bias = d_out.sum(axis=0) # (H)
        
        return d_inputs, d_weight, d_bias

class Conv1D:
    """
    A simple 1D convolution with 'valid' padding, stride=1, single input channel, single output channel
    """
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.weight = np.random.randn(kernel_size) # Random initialization of kernel
        self.bias = 0.0

    def forward(self, x):
        """
        x: shape (N, L)  # N: batch size, L: length
        out: shape (N, L - kernel_size + 1)
        """
        N, L = x.shape
        K = self.kernel_size
        self.cache = x
        out_length = L - K + 1

        out = np.zeros((N, out_length))
        for n in range(N):
            for i in range(out_length):
                out[n, i] = np.sum(x[n, i:i+K] * self.weight) + self.bias

        self.out = out
        return out

    def backward(self, d_out):
        """
        d_out: shape (N, L - K + 1)
        returns:
          d_x: gradient w.r.t. input x, shape (N, L)
          d_w: gradient w.r.t. weight, shape (K,)
          d_b: gradient w.r.t. bias, scalar
        """
        x = self.cache
        N, L = x.shape
        K = self.kernel_size
        out_length = L - K + 1

        d_x = np.zeros_like(x)
        d_w = np.zeros_like(self.weight)
        d_b = 0.0

        for n in range(N):
            for i in range(out_length):
                # Accumulate gradient w.r.t. bias
                d_b += d_out[n, i]
                # Accumulate gradient w.r.t. weight
                for k in range(K):
                    d_w[k] += x[n, i+k] * d_out[n, i]
                    d_x[n, i+k] += self.weight[k] * d_out[n, i]

        return d_x, d_w, d_b