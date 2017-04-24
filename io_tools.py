import scipy.io

def to_matlab(data_list, file_name="numpy"):
    data = dict()
    for i in range(len(data_list)):
        data[file_name + str(i)] = data_list[i]

    scipy.io.savemat(file_name, data)

def from_matlab(file_name="numpy"):

    return scipy.io.loadmat(file_name)
