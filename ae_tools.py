import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sklearn.preprocessing as prep


# 把数据压缩到0-1之间
def min_max_scale(x_train, x_test):
    preprocessor = prep.MinMaxScaler().fit(x_train)
    return preprocessor.transform(x_train), preprocessor.transform(x_test)


def get_random_block_from_data(data, batch_size, fixed=False):
    start_index = 0 if fixed else np.random.randint(0, len(data) - batch_size)
    return data[start_index: start_index + batch_size]


# 打印中间结果
def save_result(images, encodes, encode_decode, n_show=10, save_path="result/result.jpg"):

    # 创建文件夹
    path, _ = os.path.split(save_path)
    if not os.path.exists(path):
        os.makedirs(path)

    # 对比原始图片重建图片
    plt.figure(figsize=(n_show, 3))
    gs = gridspec.GridSpec(3, n_show)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(n_show):
        # 原始图片
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(images[i], (28, 28)))

        # 编码后的图
        ax = plt.subplot(gs[i + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(encodes[i], (encodes[i].shape[0]//8, 8)))

        # 解码后的图
        ax = plt.subplot(gs[i + n_show + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    pass


# 打印中间结果
def gaussian_save_result(images, gaussian_images, encode_decode, n_show=10, save_path="result/result.jpg"):

    # 创建文件夹
    path, _ = os.path.split(save_path)
    if not os.path.exists(path):
        os.makedirs(path)

    # 对比原始图片重建图片
    plt.figure(figsize=(n_show, 3))
    gs = gridspec.GridSpec(3, n_show)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(n_show):
        # 原始图片
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(images[i], (28, 28)))

        # 编码后的图
        ax = plt.subplot(gs[i + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(gaussian_images[i], (28, 28)))

        # 解码后的图
        ax = plt.subplot(gs[i + n_show + n_show])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    pass
