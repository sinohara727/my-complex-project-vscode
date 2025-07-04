#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5回演習問題
再帰型ニューラルネットワーク (RNN) の基本実装による手書き数字分類
このプログラムは、MNISTデータセットを用いて、数字の画像分類を行います。
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist


# --- ハイパーパラメータ設定 ---
n_train = 500   # 訓練データ数を1000から500に戻す
n_test = 500    # テストデータ数を1000から500に戻す
num_epoch = 50  # エポック数を100から50に戻す
q = 128         # 中間層のユニット数
eta = 0.01      # 学習率 (Adamを使っているため、Adamのalphaが優先されるが、コードに存在するため残す)

plot_misslabeled = True # 誤分類結果の画像をプロットするかどうか

##### データの取得 - MNISTデータセットのロードと前処理
# MNISTデータセットの読み込みと、特定のクラス（0, 1, 2）への限定処理を行います。
# 画像データは0-1の範囲に正規化され、ラベルはOne-Hotエンコーディング形式に変換されます。
# クラス数を定義 (0, 1, 2の3クラス分類)
m = 3

# MNISTデータセットをロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 画像データを0-1の範囲に正規化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 指定したクラス数(m)未満の数字のみをフィルタリング
x_train = x_train[y_train < m, :, :]
x_test = x_test[y_test < m, :, :]

y_train = y_train[y_train < m]
y_test = y_test[y_test < m]

# ラベルをOne-Hotエンコーディングに変換 (例: 0 -> [1,0,0], 1 -> [0,1,0])
y_train = to_categorical(y_train, m)
y_test = to_categorical(y_test, m)

## プログラム作成中は訓練データを小さくして，実行時間が短くなるようにしておく
# 設定したn_train, n_testに合わせてデータをトリミング
x_train = x_train[range(n_train),:,:]
y_train = y_train[range(n_train)]
x_test = x_test[range(n_test),:,:]
y_test = y_test[range(n_test)]

n, T, d = x_train.shape # n: データ数, T: 時系列長(画像高さ), d: 特徴量次元(画像幅)
n_test, _, _ = x_test.shape

np.random.seed(123) # 乱数シードの固定で再現性を確保

##### 活性化関数, 誤差関数, 順伝播, 逆伝播

def softmax(x):
    """
    ソフトマックス関数：多クラス分類の出力層で確率分布を生成。
    """
    u = x.T
    e = np.exp(u - np.max(u, axis=0)) # オーバーフロー対策
    return (e / np.sum(e, axis = 0)).T

def sigmoid(x):
    """
    シグモイド関数とその微分：中間層の活性化関数として使用。
    """
    tmp = 1 / (1 + np.exp(-x))
    return tmp, tmp * (1 - tmp) # 値と微分の両方を返す

def CrossEntoropy(x, y):
    """
    クロスエントロピー誤差関数：分類問題の誤差評価に利用。
    x: 予測確率 (softmax出力)
    y: 正解ラベル (One-Hotエンコーディング)
    """
    # 数値安定性のためnp.logの引数に微小な値を加える
    return -np.sum(y * np.log(x + 1e-10))

def forward(x, z_prev, W_in, W, actfunc):
    """
    順伝播の計算：入力層から中間層、中間層から中間層への伝播を処理。
    x: 入力 (定数項を含む)
    z_prev: 前の時刻の中間層出力 (z_{t-1})
    W_in: 入力層-中間層間の重み
    W: 中間層-中間層間のリカレント重み
    actfunc: 活性化関数（sigmoidなど、値と微分の両方を返す関数）
    """
    # 課題1. 順伝播のプログラムを書く
    # 注意: xは呼び出し元で定数項も含む形で渡されている
    # 注意: z_prevはz_{t-1}に対応
    # 注意: actfuncは一つ目の返り値として活性化関数fの値を，
    #       二つ目の返り値として活性化関数を微分したnabla fの値を返す
    #       関数型引数（これまで通り）
    a = np.dot(W_in,x) + np.dot(W,z_prev)
    f, nabla_f = actfunc(a)
    return f, nabla_f

def backward(W, W_out, delta, delta_out, derivative):
    """
    逆伝播の計算：誤差を逆方向に伝播させ、勾配を計算するための準備。
    W: リカレント重み
    W_out: 中間層-出力層間の重み
    delta: 次の時刻の誤差（リカレント伝播用）
    delta_out: 出力層からの誤差（出力層への逆伝播用）
    derivative: 活性化関数の微分値 (nabla f)
    """
    # 課題２
    # 逆伝播のプログラムを書く
    # (転置の存在に注意)
    # delta_outは出力層からの誤差、deltaは次の時刻の中間層からの誤差
    return (np.dot(W_out.T, delta_out) + np.dot(W.T, delta)) * derivative

def adam(W, m_val, v_val, dEdW, t, 
          alpha = 0.001, beta1 = 0.9, beta2 = 0.999, tol = 10**(-8)):
    """
    Adamオプティマイザの実装。
    W: 更新対象の重み
    m_val: 1次モーメント推定値 (Adam内部状態)
    v_val: 2次モーメント推定値 (Adam内部状態)
    dEdW: 重みWに関する勾配
    t: 更新回数 (エポックとは異なるグローバルな更新ステップ数)
    alpha: 学習率
    beta1, beta2: モーメントの減衰率
    tol: 分母のゼロ除算を防ぐための微小値
    """
    # 課題４
    # adamを作成（前回と同じでよい）
    m_t = beta1 * m_val + (1 - beta1) * dEdW
    v_t = beta2 * v_val + (1 - beta2) * dEdW * dEdW
    
    m_hat = m_t / (1 - beta1**t) # バイアス補正
    v_hat = v_t / (1 - beta2**t) # バイアス補正
    
    W_t = W - alpha * m_hat / (np.sqrt(v_hat) + tol) # 重み更新

    return W_t, m_t, v_t

##### パラメータの初期値
# 中間層のユニット数qに応じて重みを初期化
# W_in: 入力層から中間層への重み (q x (d+1)) - dは入力特徴量、+1はバイアス項
# W: 中間層間のリカレント重み (q x q)
# W_out: 中間層から出力層への重み (m x (q+1)) - mは出力クラス数、+1はバイアス項
W_in = np.random.normal(0, 0.2, size=(q, d+1))
W = np.random.normal(0, 0.2, size=(q, q))
W_out = np.random.normal(0, 0.2, size=(m, q+1))

########## 確率的勾配降下法によるパラメータ推定 (SGD with Adam)
error = []      # 訓練誤差の記録リスト
error_test = [] # テスト誤差の記録リスト

prob = np.zeros((n_test,m)) # テストデータの予測確率を格納

##### Adamのパラメータの初期値 (各重み行列に対応)
m_in = np.zeros(shape=W_in.shape)
v_in = np.zeros(shape=W_in.shape)
m_hidden = np.zeros(shape=W.shape)
v_hidden = np.zeros(shape=W.shape)
m_out = np.zeros(shape=W_out.shape)
v_out = np.zeros(shape=W_out.shape)

n_update = 0 # Adamの更新ステップ数をカウント

for epoch in range(0, num_epoch):
    index = np.random.permutation(n) # 訓練データのインデックスをシャッフル
    print(f"epoch = {epoch+1}/{num_epoch}") # エポックの進捗を表示

    e = np.full(n,np.nan) # 各訓練サンプルの誤差を一時的に格納
    for i in index: # 各訓練サンプルについてループ
        xi = x_train[i, :, :]   # 現在の訓練入力 (画像データ)
        yi = y_train[i, :]      # 現在の訓練正解ラベル

        ##### 順伝播
        # Z_prime: 時刻t+1時点の中間層の活性化関数の出力（隠れ状態）を格納 (q x T+1)
        # nabla_f: 時刻t時点の活性化関数の微分値を格納 (q x T)
        Z_prime = np.zeros((q,T+1))
        nabla_f = np.zeros((q,T))

        # 小さいtから順番に計算 (RNNの順方向パス)
        for t in range(T):
            # Z_primeの「t+1列目」, nablra_fの「t列目」を作成する
            # 注: 指定しべき列を間違えないよう注意
            # 注: 今回はxiが「T x d 行列」になっている
            #     (元が28x28ピクセルなので実際にはT=28,d=29)
            # np.append(1, xi[t,:]) で入力にバイアス項1を追加
            Z_prime[:,t+1], nabla_f[:,t] = forward(np.append(1, xi[t,:]), Z_prime[:,t], W_in, W, sigmoid)
        
        Z_T = np.append(1, Z_prime[:,T]) # 最終時刻Tの中間層出力にバイアス項1を追加

        z_out = softmax(np.dot(W_out, Z_T)) # 出力層の計算とソフトマックス活性化

        ##### 誤差評価
        e[i] = CrossEntoropy(z_out, yi) # 現在のサンプルのクロスエントロピー誤差を計算

        if epoch == 0:
            # 誤差推移観察のepoch=0はパラメタ更新しない
            # (実際には最初から更新しても構わない)
            continue
        
        ##### 課題2. 逆伝播

        # delta_outを出力層の誤差として定義
        # 出力誤差は (予測確率 - 正解ラベル)
        delta_out = z_out - yi

        # 以下の行列の各列にdelta_1, ..., delta_Tを作成
        # backward関数の内部を作成
        delta = np.zeros((q,T)) # 時刻tの中間層の誤差を格納
        for t in reversed(range(T)): # 時刻Tから逆順に計算 (RNNの逆方向パス)
            if t == T-1: # 最終時刻の場合
                # 出力層からの誤差delta_outをW_out[:,1:]で逆伝播
                # Recurrentな誤差はゼロ (最後なので次の時刻はない)
                delta[:,t] = backward(W, W_out[:,1:], np.zeros(q), delta_out, nabla_f[:,t]) 
            else: # 最終時刻ではない場合
                # 出力層からの誤差はゼロ、次の時刻の中間層からの誤差delta[:,t+1]を逆伝播
                delta[:,t] = backward(W, W_out[:,1:], delta[:,t+1], np.zeros(m), nabla_f[:,t]) 
        
        ### 課題3. 勾配の計算 (dEdW_out, dEdW_in, dEdW の計算)

        ## dEdW_outの作成 (出力層の重みW_outの勾配)
        # ヒント: np.dotかnp.outerのどちらを使うべきか適切に判断すること
        #       また，上で作成したZ_Tを利用できる
        # delta_outと最終中間層出力Z_Tの外積で勾配を計算
        dEdW_out = np.outer(delta_out, Z_T.T)

        ## dEdW_inの作成 (入力層-中間層間の重みW_inの勾配)
        # ヒント: 以下のXが定数項含んだTx(d+1)行列 
        # (np.c_は横方向の結合. Xをコンソールで見てみると
        #  何が行われいてるかわかってよい)
        X = np.c_[np.ones(T), xi.reshape(T, d)] # 時刻tにおける入力(1と画像ピクセル値)を行方向に並べた行列
        dEdW_in = np.dot(delta, X) # deltaとXの内積で勾配を計算

        ## dEdWの作成 (リカレント重みWの勾配)
        # ヒント: Z_primeの0列目からT-1列目(つまり最後の列以外)は"Z_prime[:,:T]"で指定できる
        #       また，転置の存在に注意せよ
        dEdW = np.dot(delta, Z_prime[:, :T].T) # deltaと前時刻の中間層出力の内積で勾配を計算
        
        ##### パラメータの更新 (Adamオプティマイザを使用)
        # W_out -= eta*dEdW_out/epoch
        # W -= eta*dEdW/epoch
        # W_in -= eta*dEdW_in/epoch

        ### 課題4 adamを作成して更新方法を以下に変更（上の確率勾配降下の更新は消す）
        n_update += 1 # グローバルな更新ステップ数をインクリメント
        W_out, m_out, v_out = adam(W_out, m_out, v_out, dEdW_out, n_update)
        W, m_hidden, v_hidden = adam(W, m_hidden, v_hidden, dEdW, n_update) 
        W_in, m_in, v_in = adam(W_in, m_in, v_in, dEdW_in, n_update) 

    ##### training error (エポックごとの訓練誤差を計算し記録)
    error.append(sum(e)/n)

    e_test = np.full(n_test,np.nan) # 各テストサンプルの誤差を一時的に格納
    ##### test error (エポックごとのテスト誤差を計算し記録)
    for i in range(0, n_test):
        xi = x_test[i, :, :]
        yi = y_test[i, :]
        
        ##### 順伝播 (テストデータに対する予測)
        Z_prime = np.zeros((q,T+1))
        for t in range(T):
        # 訓練の時と同じ手順でZ_primeを作成
        # (こちらではnabla_fは使用しないので, 最後に"[0]"を
        #  つけることで返り値を一つだけ受け取っている)
            Z_prime[:,t+1] = forward(np.append(1, xi[t,:]), Z_prime[:,t], W_in, W, sigmoid)[0]
        
        z_out = softmax(np.dot(W_out, np.append(1, Z_prime[:,T]))) 
        prob[i,:] = z_out # 予測確率を格納

        e_test[i] = CrossEntoropy(z_out, yi) # テスト誤差を計算
    
    error_test.append(sum(e_test)/n_test)

########## 誤差関数のプロット
plt.clf() # 現在のプロットをクリア
plt.plot(error, label="training", lw=3)    # 訓練誤差を青線でプロット
plt.plot(error_test, label="test", lw=3)    # テスト誤差をオレンジ線でプロット
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Cross-entropy",fontsize=18)
plt.grid() # グリッド線を表示
plt.legend(fontsize = 16) # 凡例を表示
plt.savefig("./error_adam_results.pdf", bbox_inches='tight', transparent=True) # プロットをPDFとして保存
plt.close() # プロットウィンドウを閉じる (Aggバックエンド使用時など)

predict = np.argmax(prob, 1) # 最も確率の高いクラスを予測結果とする

if plot_misslabeled:
    n_maxplot = 20 # 誤分類をプロットする最大数
    n_plot = 0

    ##### 誤分類結果のプロット
    # 混同行列のように、真のラベルと予測ラベルの組み合わせで誤分類画像を抽出
    for i in range(m): # 真のラベル (i)
        idx_true = (y_test[:, i]==1) # 真のラベルがiであるデータ
        for j in range(m): # 予測されたラベル (j)
            idx_predict = (predict==j) # 予測されたラベルがjであるデータ
            # ConfMat[i, j] = sum(idx_true*idx_predict) # これは混同行列の計算部分、プロットでは使用せず
            if j != i: # 予測が間違っている場合 (真のラベルと予測が異なる場合)
                for l in np.where(idx_true*idx_predict == True)[0]: # 誤分類されたデータのインデックスを取得
                    plt.clf() # 新しいプロットを準備
                    D = x_test[l, :, :] # 誤分類された画像データを取得
                    sns.heatmap(D, cbar =False, cmap="Blues", square=True) # ヒートマップとして表示
                    plt.axis("off") # 軸を非表示に
                    plt.title('True: {}, Predicted: {}'.format(i, j)) # タイトルに真のラベルと予測ラベルを表示
                    plt.savefig("./misslabeled_img_{}.pdf".format(l), bbox_inches='tight', transparent=True) # 画像をPDFとして保存
                    n_plot += 1
                    if n_plot >= n_maxplot: # 設定した最大プロット数に達したらループを抜ける
                        break
            if n_plot >= n_maxplot:
                break
        if n_plot >= n_maxplot:
            break

# --- 混同行列の計算とプロット ---
predict_label = np.argmax(prob, axis=1) # 予測ラベル
true_label = np.argmax(y_test, axis=1) # 真のラベル

ConfMat = np.zeros((m, m)) # m x m のゼロ行列で混同行列を初期化
for i in range(m): # 真のラベルの各クラスについて
    for j in range(m): # 予測ラベルの各クラスについて
        # 真のラベルがiかつ予測ラベルがjであるサンプルの数をカウント
        ConfMat[i, j] = np.sum((true_label == i) & (predict_label == j))

plt.clf() # プロットをクリア
fig, ax = plt.subplots(figsize=(6,6),tight_layout=True) # 図と軸を作成
fig.show() # プロットウィンドウを表示（VS Codeなどで実行時に表示される）
sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="d", cbar =False, cmap="Blues",
            xticklabels=[str(k) for k in range(m)], yticklabels=[str(k) for k in range(m)]) # 混同行列をヒートマップで表示
ax.set_xlabel(xlabel="Predicted Label", fontsize=18) # 予測ラベルの軸ラベル (フォントサイズを18に戻す)
ax.set_ylabel(ylabel="True Label", fontsize=18)       # 真のラベルの軸ラベル (フォントサイズを18に戻す)
plt.savefig("./confusion_matrix_adam_results.pdf", bbox_inches="tight", transparent=True) # 混同行列をPDFとして保存
plt.close() # プロットウィンドウを閉じる