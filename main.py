import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def emg_muap(x):
    N = len(x)  # длина файла
    wnd_size = 600  # ширина окна
    muap_left = 500  # прибавить к окну слева
    muap_right = 500  # прибавить к окну справа
    muaps = np.zeros((muap_left + muap_right, 500)) #размер окна на 500

    numMuaps = 0
    for k in range(0, N - muap_right - wnd_size):

        # берем кусочек сигнала в текущей позиции окна
        xWnd = np.abs(x[k:k + wnd_size])
        xWnd = xWnd - np.mean(xWnd)  # вычитаем среднее
        maxValue = np.max(xWnd)  # ищем максимальное значение
        maxIndices = np.argmax(xWnd)  # сохраняем индекс максимального значения
        if (maxIndices == wnd_size // 2) and (maxValue > 5000):
            muaps[:, numMuaps] = x[k + wnd_size // 2 - muap_left:k + wnd_size // 2 + muap_right]
            numMuaps = numMuaps + 1  # ещё один муап

    muaps = muaps[:, :numMuaps]
    return muaps


if __name__ == '__main__':
    x1_1 = np.loadtxt('001_001_cln.txt')  # Загрузили первое движение в x1
    x1_2 = np.loadtxt('001_002_cln.txt')  # Загрузили первое движение в x2

    x2_1 = np.loadtxt('002_001.txt')
    x2_2 = np.loadtxt('002_002.txt')

    x3_1 = np.loadtxt('003_001.txt')
    x3_2 = np.loadtxt('003_002_cln.txt')

    x4_1 = np.loadtxt('004_001.txt')
    x4_2 = np.loadtxt('004_002.txt')

    x6_1 = np.loadtxt('006_001.txt')
    x6_2 = np.loadtxt('006_002.txt')

    # x7_1 = np.loadtxt('007_001.txt')
    # x7_2 = np.loadtxt('007_002_mod.txt')

    x8_1 = np.loadtxt('008_001.txt')
    x8_2 = np.loadtxt('008_002_cln.txt')

    x1_1 -= np.mean(x1_1)  # Вычесть матожидание
    x1_2 -= np.mean(x1_2)
    x2_1 -= np.mean(x2_1)
    x2_2 -= np.mean(x2_2)
    x3_1 -= np.mean(x3_1)
    x3_2 -= np.mean(x3_2)
    x4_1 -= np.mean(x4_1)
    x4_2 -= np.mean(x4_2)
    x6_1 -= np.mean(x1_1)
    x6_2 -= np.mean(x1_1)
    # x7_1 -= np.mean(x1_1)
    # x7_2 -= np.mean(x1_1)
    x8_1 -= np.mean(x1_1)
    x8_2 -= np.mean(x1_1)

    # Очистка от пиков
    # plt.figure(figsize=(10, 6))
    # x2[21126] = x1[21125]
    #
    # plt.plot(x7_2[:])

    # np.savetxt('008_002_cln.txt',x2)
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(x1[:], label='Движения первого типа') #Вывести
    # plt.plot(x2[:], label='Движения второго типа')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Вычистить частоты от 0 до 5 Гц
    SPS = 1000.0  # 1 кГц
    hflt = signal.firls(513, [0., 5., 7., SPS / 2], [0., 0., 1.0, 1.0], fs=SPS)
    # plt.figure(figsize=(10, 6))
    # plt.clf()
    w, h = signal.freqz(hflt, fs=SPS)
    # plt.plot(w, 20 * np.log10(abs(h)), 'b')
    # plt.grid(True)
    # plt.xlabel('Частота, Гц')
    # plt.ylabel('Частотный отклик, дБ')
    # _ = plt.title('АЧХ фильтра')

    # plt.show()

    y1_1 = np.convolve(hflt, x1_1,
                       'same')  # возвращает дискретную линейную свертку двух одномерных последовательностей.
    y1_2 = np.convolve(hflt, x1_2, 'same')  # пропуск через фильтр

    y2_1 = np.convolve(hflt, x2_1, 'same')
    y2_2 = np.convolve(hflt, x2_2, 'same')

    y3_1 = np.convolve(hflt, x3_1, 'same')
    y3_2 = np.convolve(hflt, x3_2, 'same')

    y4_1 = np.convolve(hflt, x4_1, 'same')
    y4_2 = np.convolve(hflt, x4_2, 'same')

    y6_1 = np.convolve(hflt, x6_1, 'same')
    y6_2 = np.convolve(hflt, x6_2, 'same')

    # y7_1 = np.convolve(hflt, x7_1, 'same')
    # y7_2 = np.convolve(hflt, x7_2, 'same')

    y8_1 = np.convolve(hflt, x8_1, 'same')
    y8_2 = np.convolve(hflt, x8_2, 'same')

    # plt.figure(figsize=(10, 6))
    # plt.plot(x7_1[:], linewidth=0.7, label='x1 до фильтрации')
    # plt.plot(y7_1[:], 'r-', linewidth=0.7, label='x1 после фильтрации')
    # plt.legend()
    # _ = plt.xlabel('Номер отсчета')
    # _ = plt.ylabel('Амплитуда сигнала')
    # plt.grid(True) #для x1 показать резницу
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(x7_1[:], linewidth=0.7, label='x2 до фильтрации')
    # plt.plot(y7_2[:], 'r-', linewidth=0.7, label='x2 после фильтрации')
    # plt.legend()
    # _ = plt.xlabel('Номер отсчета')
    # _ = plt.ylabel('Амплитуда сигнала')
    # plt.grid(True) #для x1 показать резницу

    # plt.show()

    muaps1_1 = emg_muap(y1_1)
    muaps1_2 = emg_muap(y1_2)

    muaps2_1 = emg_muap(y2_1)
    muaps2_2 = emg_muap(y2_2)
    muaps3_1 = emg_muap(y3_1)
    muaps3_2 = emg_muap(y3_2)
    muaps4_1 = emg_muap(y4_1)
    muaps4_2 = emg_muap(y4_2)
    muaps6_1 = emg_muap(y6_1)
    muaps6_2 = emg_muap(y6_2)
    # muaps7_1 = emg_muap(y7_1)
    # muaps7_2 = emg_muap(y7_2)
    muaps8_1 = emg_muap(y8_1)
    muaps8_2 = emg_muap(y8_2)

    plt.figure(figsize=(12, 8))
    plt.plot(muaps1_1[:, :])
    plt.plot(muaps2_1[:, :])
    plt.plot(muaps3_1[:, :])
    plt.plot(muaps4_1[:, :])
    plt.plot(muaps6_1[:, :])
    # plt.plot(muaps7_1[:, :])
    plt.plot(muaps8_1[:, :])
    plt.grid(True)
    # plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(muaps1_2[:, :])
    plt.plot(muaps2_2[:, :])
    plt.plot(muaps3_2[:, :])
    plt.plot(muaps4_2[:, :])
    plt.plot(muaps6_2[:, :])
    # plt.plot(muaps7_2[:, :])
    plt.plot(muaps8_2[:, :])
    plt.grid(True)
    # plt.show()

    # Формирование обучающей выборки

    Xlern = np.column_stack((muaps1_1[:,1], muaps1_2[:,1],
                            muaps2_1[:,1], muaps2_2[:,1],
                            muaps3_1[:,1], muaps3_2[:,1],
                            muaps4_1[:,1], muaps4_2[:,1],
                            muaps6_1[:,1], muaps6_2[:,1],
                            muaps8_1[:,1], muaps8_2[:,1],
                            muaps1_1[:,2], muaps1_2[:,2],
                            muaps2_1[:,2], muaps2_2[:,2],
                            muaps3_1[:,2], muaps3_2[:,2],
                            muaps4_1[:,2], muaps4_2[:,2],
                            muaps6_1[:,2], muaps6_2[:,2],
                            muaps8_1[:,2], muaps8_2[:,2],
                            muaps1_1[:,3], muaps1_2[:,3],
                            muaps2_1[:,3], muaps2_2[:,3],
                            muaps3_1[:,3], muaps3_2[:,3],
                            muaps4_1[:,3], muaps4_2[:,3],
                            muaps6_1[:,3], muaps6_2[:,3],
                            muaps8_1[:,3], muaps8_2[:,3])).T

    print(Xlern.shape)

    # Правильные ответы для обучающей выборки
    Ylearn = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
         1, -1, 1, -1]

    # Метод наименьших квадратов

    C = np.linalg.inv(Xlern.T @ Xlern) @ Xlern.T @ Ylearn

    print(C)

    X = np.array([])

    for i in range(3,16):

             np.concatenate((X,np.column_stack(
                muaps1_1[:, i], muaps1_2[:, i],
                muaps2_1[:, i], muaps2_2[:, i],
                muaps3_1[:, i], muaps3_2[:, i],
                muaps4_1[:, i], muaps4_2[:, i],
                muaps6_1[:, i], muaps6_2[:, i],
                muaps8_1[:, i], muaps8_2[:, i],
            )

