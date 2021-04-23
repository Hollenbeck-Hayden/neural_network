import matplotlib.pyplot as plt

def read_dataset(infile):
    data = [[], []]
    data_size = int(infile.readline())
    for i in range(data_size):
        items = infile.readline().split(", ")
        data[0].append(float(items[0]))
        data[1].append(float(items[1]))
    return data

def plot_data(filename):

    with open(filename) as infile:
        data = read_dataset(infile)
        pred = read_dataset(infile)

    plt.plot(data[0], data[1], "o", label="outputs")
    plt.plot(pred[0], pred[1], "-", label="results")
    plt.legend()
    plt.savefig("results.png")
    plt.clf()

def plot_xvalid(filename):
    with open(filename) as infile:
        data = read_dataset(infile)

    indices = [i for i in range(len(data[0]))]

    mindex = 0
    for i in range(1, len(data[1])):
        if (data[1][i] < data[1][mindex]):
            mindex = i

    plt.plot(indices, data[0], label="training")
    plt.plot(indices, data[1], label="validation")
    plt.plot([mindex, mindex], [data[0][mindex], data[1][mindex]], 'bo', label="minimum")
    plt.legend()
    plt.show()
    plt.clf()

count = 4

#for i in range(count):
#    plot_xvalid("../data/xvalid_" + str(i) + ".data")

#for i in range(count):
#    plot_data("../data/results_" + str(i) + ".data")

plot_data("../data/results_average.data")
