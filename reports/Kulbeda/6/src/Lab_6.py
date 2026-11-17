import numpy as np
import matplotlib.pyplot as plt

# Параметры
a, b, c, d = 0.4, 0.4, 0.08, 0.4
n_in = 6
n_hid = 2
epochs = 2000

# Данные
x = np.linspace(0, 20, 200)
y = a * np.cos(b * x) + c * np.sin(d * x)

X = []
Y = []
for i in range(n_in, len(y)):
    X.append(y[i-n_in:i])
    Y.append(y[i])
X = np.array(X)
Y = np.array(Y)

n_train = int(0.7 * len(X))
Xtr = X[:n_train]
Ytr = Y[:n_train]
Xte = X[n_train:]
Yte = Y[n_train:]

mx = Xtr.mean(axis=0)
sx = Xtr.std(axis=0) + 1e-8
my = Ytr.mean()
sy = Ytr.std() + 1e-8

Xtr = (Xtr - mx) / sx
Xte = (Xte - mx) / sx
Ytr = (Ytr - my) / sy
Yte = (Yte - my) / sy

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(alpha):
    np.random.seed(1)
    W_in = np.random.randn(1, n_hid) * 0.5
    W_ctx = np.random.randn(n_hid, n_hid) * 0.5
    b_h = np.zeros(n_hid)
    W_out = np.random.randn(n_hid, 1) * 0.5
    b_out = np.zeros(1)
    
    losses = []
    
    for ep in range(epochs):
        h_all = np.zeros((len(Xtr), n_in, n_hid))
        ctx = np.zeros((len(Xtr), n_hid))
        
        for t in range(n_in):
            x_t = Xtr[:, t].reshape(-1, 1)
            if t == 0:
                c_t = ctx
            else:
                c_t = h_all[:, t-1, :]
            h_t = sigmoid(x_t @ W_in + c_t @ W_ctx + b_h)
            h_all[:, t, :] = h_t
            
        h_last = h_all[:, -1, :]
        pred = (h_last @ W_out + b_out).flatten()
        
        loss = np.mean((pred - Ytr) ** 2)
        losses.append(loss)
        
        d_out = 2 * (pred - Ytr) / len(Ytr)
        dW_out = h_last.T @ d_out.reshape(-1, 1)
        db_out = d_out.sum()
        
        dH = d_out.reshape(-1, 1) @ W_out.T
        
        dW_in = np.zeros_like(W_in)
        dW_ctx = np.zeros_like(W_ctx)
        db_h = np.zeros_like(b_h)
        
        for t in reversed(range(n_in)):
            h_t = h_all[:, t, :]
            dZ_h = dH * h_t * (1 - h_t)
            
            x_t = Xtr[:, t].reshape(-1, 1)
            if t == 0:
                c_t = ctx
            else:
                c_t = h_all[:, t-1, :]
                
            dW_in += x_t.T @ dZ_h
            dW_ctx += c_t.T @ dZ_h
            db_h += dZ_h.sum(axis=0)
            
            if t > 0:
                dH = dZ_h @ W_ctx.T
                
        W_in -= alpha * dW_in
        W_ctx -= alpha * dW_ctx
        b_h -= alpha * db_h
        W_out -= alpha * dW_out
        b_out -= alpha * db_out
        
    return losses, (W_in, W_ctx, b_h, W_out, b_out)

def predict(X_data, weights):
    W_in, W_ctx, b_h, W_out, b_out = weights
    h_all = np.zeros((len(X_data), n_in, n_hid))
    ctx = np.zeros((len(X_data), n_hid))
    
    for t in range(n_in):
        x_t = X_data[:, t].reshape(-1, 1)
        if t == 0:
            c_t = ctx
        else:
            c_t = h_all[:, t-1, :]
        h_t = sigmoid(x_t @ W_in + c_t @ W_ctx + b_h)
        h_all[:, t, :] = h_t
        
    h_last = h_all[:, -1, :]
    return (h_last @ W_out + b_out).flatten()

alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
best_alpha = 0
best_err = 999
best_w = None
best_l = None

for al in alphas:
    l, w = train(al)
    p = predict(Xte, w)
    err = np.mean((p - Yte) ** 2)
    print("alpha =", al, " | test err =", round(err, 6))
    if err < best_err:
        best_err = err
        best_alpha = al
        best_w = w
        best_l = l

print("best alpha =", best_alpha)

ptr = predict(Xtr, best_w) * sy + my
pte = predict(Xte, best_w) * sy + my
Ytr0 = Ytr * sy + my
Yte0 = Yte * sy + my

# Красивый вывод таблиц
print("\nОбучение:")
print(f"{'i':>4} | {'эталон':>10} | {'прогноз':>10} | {'отклонение':>12}")
print("-" * 45)
for i in range(len(Ytr0)):
    print(f"{i:>4} | {Ytr0[i]:>10.4f} | {ptr[i]:>10.4f} | {ptr[i] - Ytr0[i]:>12.4f}")

print("\nТест:")
print(f"{'i':>4} | {'эталон':>10} | {'прогноз':>10} | {'отклонение':>12}")
print("-" * 45)
for i in range(len(Yte0)):
    print(f"{i:>4} | {Yte0[i]:>10.4f} | {pte[i]:>10.4f} | {pte[i] - Yte0[i]:>12.4f}")

plt.figure()
plt.plot(x, y, 'k-', label='функция')
plt.plot(x[n_in:n_in+len(ptr)], ptr, 'b.', label='train')
plt.plot(x[n_in+len(ptr):n_in+len(ptr)+len(pte)], pte, 'r.', label='test')
plt.legend()
plt.grid()
plt.title('прогноз')
plt.show()

plt.figure()
plt.plot(best_l)
plt.title('ошибка по итерациям')
plt.xlabel('эпоха')
plt.ylabel('MSE')
plt.grid()
plt.show()