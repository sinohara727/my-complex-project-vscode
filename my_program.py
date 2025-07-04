#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5回演習問題
"""
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist


n_train = 500
n_test = 500

plot_misslabeled = False # True

##### データの取得
#クラス数を定義
m = 3

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train[y_train < m, :, :]

x_test = x_test.astype('float32') / 255.
x_test = x_test[y_test < m, :, :]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

## プログラム作成中は訓練データを小さくして，
## 実行時間が短くなるようにしておく
x_train = x_train[range(n_train),:,:]
y_train = y_train[range(n_train)]
x_test = x_test[range(n_test),:,:]
y_test = y_test[range(n_test)]

n, T, d = x_train.shape
n_test, _, _ = x_test.shape

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播
def softmax(x):
    u = x.T
    e = np.exp(u-np.max(u, axis=0))
    return (e/np.sum(e, axis = 0)).T

def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp, tmp*(1-tmp)

def CrossEntoropy(x, y):
    # returnの後にクロスエントロピーを返すプログラムを書く
    return -np.sum(y*np.log(x))

def forward(x, z_prev, W_in, W, actfunc):
    ### 課題1. 順伝播のプログラムを書く
    # 注意: xは呼び出し元で定数項も含む形で渡されている
    # 注意: z_prevはz_{t-1}に対応
    # 注意: actfuncは一つ目の返り値として活性化関数fの値を，
    #       二つ目の返り値として活性化関数を微分したnabla fの値を返す
    #       関数型引数（これまで通り）
    a = np.dot(W_in,x)+np.dot(W,z_prev)
    f,nabla_f = actfunc(a)
    return f,nabla_f

def backward(W, W_out, delta, delta_out, derivative):
    ### 課題２
    # 逆伝播のプログラムを書く
    # (転置の存在に注意)
    return np.dot(W_out.T,delta_out)*derivative + np.dot(W.T,delta)*derivative

def adam(W, m, v, dEdW, t, 
         alpha = 0.001, beta1 = 0.9, beta2 = 0.999, tol = 10**(-8)):
    ### 課題４
    # adamを作成（前回と同じでよい）
    m_t = beta1*m + (1-beta1)*dEdW
    v_t = beta2*v + (1-beta2)*dEdW*dEdW
    
    m_hat = m_t/(1-beta1**t)
    v_hat = v_t/(1-beta2**t)
    
    W_t = W - alpha*m_hat/(np.sqrt(v_hat)+tol)

    return W_t, m_t, v_t

##### 中間層のユニット数とパラメータの初期値
q = 128

W_in = np.random.normal(0, 0.2, size=(q, d+1))
W = np.random.normal(0, 0.2, size=(q, q))
W_out = np.random.normal(0, 0.2, size=(m, q+1))

########## 確率的勾配降下法によるパラメータ推定
num_epoch = 50
#num_epoch = 10

error = []
error_test = []

prob = np.zeros((n_test,m))

##### adamのパラメータの初期値
m_in = np.zeros(shape=W_in.shape)
v_in = np.zeros(shape=W_in.shape)
m_hidden = np.zeros(shape=W.shape)
v_hidden = np.zeros(shape=W.shape)
m_out = np.zeros(shape=W_out.shape)
v_out = np.zeros(shape=W_out.shape)

eta = 0.01

n_update = 0

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    print("epoch =",epoch)

    e = np.full(n,np.nan)        
    for i in index:
        xi = x_train[i, :, :]  
        yi = y_train[i, :]
        
        ##### 順伝播
        # 課題1. Z_prime, nabla_fを作成する
        Z_prime = np.zeros((q,T+1))
        nabla_f = np.zeros((q,T))

        # 小さいtから順番に計算する
        for t in range(T):
            # Z_primeの「t+1列目」, nablra_fの「t列目」を作成する
            # 注: 指定しべき列を間違えないよう注意
            # 注: 今回はxiが「T x d 行列」になっている
            #     (元が28x28ピクセルなので実際にはT=28,d=29)
            Z_prime[:,t+1], nabla_f[:,t] = forward(np.append(1, xi[t,:]), Z_prime[:,t], W_in, W, sigmoid)
        
        Z_T = np.append(1, Z_prime[:,T])

        z_out = softmax(np.dot(W_out, Z_T))        

        ##### 誤差評価
        e[i] = CrossEntoropy(z_out, yi)

        if epoch == 0:
            # 誤差推移観察のepoch=0はパラメタ更新しない
            # (実際には最初から更新しても構わない)
            continue
        
        ##### 課題2. 逆伝播

        # delta_outを定義する
        
        delta_out = z_out - yi

        # 以下の行列の各列にdelta_1, ..., delta_Tを作成
        # backward関数の内部を作成
        delta = np.zeros((q,T)) 
        for t in reversed(range(T)):
            if t == T-1:
                delta[:,t] = backward(W, W_out[:,1:], np.zeros(q), delta_out, nabla_f[:,t]) 
            else:        
                delta[:,t] = backward(W, W_out[:,1:], delta[:,t+1], np.zeros(m), nabla_f[:,t]) 
        
        ### 課題3. 勾配の計算        

        ## dEdW_outの作成
        # ヒント: np.dotかnp.outerのどちらを使うべきか適切に判断すること
        #         また，上で作成したZ_Tを利用できる
        dEdW_out = np.outer(delta_out,Z_T.T)

        ## dEdE_inの作成
        # ヒント: 以下のXが定数項含んだTx(d+1)行列 
        # (np.c_は横方向の結合. Xをコンソールで見てみると
        #  何が行われいてるかわかってよい)
        X = np.c_[np.ones(d), xi] 
        dEdW_in = np.dot(delta,X)

        ## dEdWの作成
        # ヒント: Z_primeの0列目からT-1列目(つまり最後の列以外)は"Z_prime[:,:T]"で指定できる
        #         また，転置の存在に注意せよ
        dEdW =np.dot(delta,Z_prime[:,:T].T)
        
        ##### パラメータの更新
        # W_out -= eta*dEdW_out/epoch
        # W -= eta*dEdW/epoch
        # W_in -= eta*dEdW_in/epoch

        ### 課題4 adamを作成して更新方法を以下に変更（上の確率勾配降下の更新は消す）
        n_update += 1
        W_out, m_out, v_out = adam(W_out, m_out, v_out, dEdW_out, n_update)
        W, m_hidden, v_hidden = adam(W, m_hidden, v_hidden, dEdW, n_update) 
        W_in, m_in, v_in = adam(W_in, m_in, v_in, dEdW_in, n_update)  

    ##### training error
    error.append(sum(e)/n)

    e_test = np.full(n_test,np.nan)            
    ##### test error
    for i in range(0, n_test):
        xi = x_test[i, :, :]
        yi = y_test[i, :]
        
        ##### 順伝播
        Z_prime = np.zeros((q,T+1))
        for t in range(T):
        # 訓練の時と同じ手順でZ_primeを作成
        # (こちらではnabla_fは使用しないので, 最後に"[0]"を
        #  つけることで返り値を一つだけ受け取っている)
            Z_prime[:,t+1] = forward(np.append(1, xi[t,:]), Z_prime[:,t], W_in, W, sigmoid)[0]
        
        z_out = softmax(np.dot(W_out, np.append(1, Z_prime[:,T])))        
        prob[i,:] = z_out

        e_test[i] = CrossEntoropy(z_out, yi)
    
    error_test.append(sum(e_test)/n_test)

########## 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Cross-entropy",fontsize=18)
plt.grid()
plt.legend(fontsize = 16)
plt.savefig("./error_adam.pdf", bbox_inches='tight', transparent=True)

predict = np.argmax(prob, 1)

if plot_misslabeled:
    n_maxplot = 20
    n_plot = 0

    ##### 誤分類結果のプロット

    for i in range(m):
        idx_true = (y_test[:, i]==1)
        for j in range(m):
            idx_predict = (predict==j)
            # ConfMat[i, j] = sum(idx_true*idx_predict)
            if j != i:
                for l in np.where(idx_true*idx_predict == True)[0]:
                    plt.clf()
                    D = x_test[l, :, :]
                    sns.heatmap(D, cbar =False, cmap="Blues", square=True)
                    plt.axis("off")
                    plt.title('{} to {}'.format(i, j))
                    plt.savefig("./misslabeled{}.pdf".format(l), bbox_inches='tight', transparent=True)
                    n_plot += 1
                    if n_plot >= n_maxplot:
                        break
            if n_plot >= n_maxplot:
                break
        if n_plot >= n_maxplot:
            break

predict_label = np.argmax(prob, axis=1)
true_label = np.argmax(y_test, axis=1)

ConfMat = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        ConfMat[i, j] = np.sum((true_label == i) & (predict_label == j))

plt.clf()
fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="1", cbar =False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion_adam.pdf", bbox_inches="tight", transparent=True)
plt.close()
