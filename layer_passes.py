import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from global_config import home

input_time_length, input_channels = 1200, 85
import numpy as np


def get_max_v(input_dict):
    return list(input_dict.keys())[-1][1]


def get_max_h(input_dict):
    return list(input_dict.keys())[-1][0]


def get_output_shape(input_shape, kernel_size, stride, dilation):
    h_shape = input_shape[0] - (kernel_size[0] - 1) * dilation[0]
    v_shape = input_shape[1] - (kernel_size[1] - 1) * dilation[1]
    return h_shape, v_shape


def max_pool_pass(input_dict, kernel_size, stride, dilation):
    pass


def get_initial_dict(input_time_length, num_of_channels):
    complete_dict = {}
    for x in range(input_time_length):
        for y in range(num_of_channels):
            xy_dict = np.zeros([input_time_length, input_channels])
            xy_dict[x, y] = 1
            complete_dict[x, y] = xy_dict
        print(x)
    return complete_dict


def update_dict(input_dict, output_dict, maxpool=False, kernel_size=None):
    # print(input_dict.shape)
    # print(output_dict.shape)

    if maxpool:
        output_dict += input_dict / kernel_size
    else:
        output_dict += input_dict
    return output_dict


def initialize_layer_dict(max_k, max_l):
    layer_dict = {}
    for k in range(max_k):
        for l in range(max_l):
            kl_dict = np.zeros([input_time_length, input_channels])
            layer_dict[k, l] = kl_dict
    return layer_dict


def convolution_pass(input_dict, kernel_size, stride, dilation, input_shape, maxpool):
    max_k, max_l = get_output_shape(input_shape, kernel_size, stride, dilation)
    layer_dict = initialize_layer_dict(max_k, max_l)
    print(max_k, max_l)
    i = 0
    j = 0
    for l in range(0, max_l):
        for k in range(0, max_k):
            h_indices = [index + (index - i) * (dilation[0] - 1) for index in range(i, i + kernel_size[0])]
            v_indices = [index + (index - j) * (dilation[1] - 1) for index in range(j, j + kernel_size[1])]
            # print(h_indices, v_indices)
            for h_index in h_indices:
                for v_index in v_indices:
                    update_dict(input_dict[h_index, v_index], layer_dict[k, l], kernel_size=kernel_size[0],
                                maxpool=maxpool)
            i += stride[0]
        print(j)
        j += stride[1]
        i = 0
    return layer_dict, max_k, max_l


def visualize_heatmap(arr, file, max_k):
    arr = np.transpose(arr)
    print(arr.shape)
    arr = arr[:10, :850]
    sns.heatmap(arr)
    plt.plot([max_k, max_k], [0, 85], color='red', label='not_shifted')
    plt.plot([int(max_k/2), int(max_k/2)], [0, 85], color='green', label='shifted')
    plt.legend()
    plt.xlabel('')
    plt.title(file)
    plt.tight_layout()
    plt.savefig(f'{home}/results/graphs/{file}.png')
    plt.show()


def visualize_multiple_heat_maps(kernel, dilation):
    file = get_name_from_kd(kernel, dilation)
    max_k, max_l = get_num_of_predictions(kernel, dilation)
    smaller_window = input_time_length - max_k + 1
    arr = np.load(f'{home}/outputs/receptive_fields/{file}')
    visualize_heatmap(arr, file, smaller_window)


def get_num_of_predictions(kernels, dilations, layer=None):
    print('made for sbp0')
    if layer == 'conv_spat':
        return 1200 - 10 + 1, 0

    max_k, max_l = get_output_shape((input_time_length, input_channels), (10, 1), (1, 1), (1, 1),)
    max_k, max_l = get_output_shape((max_k, max_l), (1, input_channels), (1, 1), (1, 1),)
    if True:
        max_k, max_l = get_output_shape((max_k, max_l), (kernels[0], 1), (1, 1), (dilations[0], 1))
        i = 1
    else:
        max_k, max_l = get_output_shape((max_k, max_l), (kernels[0], 1), (1, 1), (1, 1))
        i = 0

    max_k, max_l = get_output_shape((max_k, max_l), (10, 1), (1, 1), (3, 1))
    if layer == 'conv_2':
        return max_k, max_l
    # max_k, max_l = get_output_shape((max_k, max_l), (kernels[1], 1), (1, 1), (dilations[1], 1))
    max_k, max_l = get_output_shape((max_k, max_l), (kernels[1], 1), (1, 1), (dilations[i], 1))
    i += 1

    max_k, max_l = get_output_shape((max_k, max_l), (10, 1), (1, 1), (9, 1))
    if layer == 'conv_3':
        return max_k, max_l
    # max_k, max_l = get_output_shape((max_k, max_l), (kernels[2], 1), (1, 1), (dilations[2], 1))
    max_k, max_l = get_output_shape((max_k, max_l), (kernels[2], 1), (1, 1), (dilations[i], 1))
    i += 1

    max_k, max_l = get_output_shape((max_k, max_l), (10, 1), (1, 1), (27, 1))
    if layer == 'conv_4':
        return max_k, max_l
    # max_k, max_l = get_output_shape((max_k, max_l), (kernels[3], 1), (1, 1), (dilations[3], 1))
    max_k, max_l = get_output_shape((max_k, max_l), (kernels[3], 1), (1, 1), (dilations[i], 1))

    max_k, max_l = get_output_shape((max_k, max_l), (2, 1), (1, 1), (81, 1))
    return max_k, max_l


def get_name_from_kd(kernels, dilations):
    d_name = ''.join([str(x) for x in dilations])
    k_name = ''.join([str(x) for x in kernels])
    return f'rf_k_{k_name}_d_{d_name}_sbb_False_0.npy'


def create_receptive_fields():
    initial_dict = get_initial_dict(input_time_length, num_of_channels=input_channels)
    for kernels in [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]:
    # for kernels in [[3, 3, 3, 3]]:
        for dilations in [[1, 1, 1, 1], [2, 4, 8, 16], [3, 9, 27, 81]]:
        # for dilations in [[3, 9, 27, 81]]:
            d_name = ''.join([str(x) for x in dilations])
            k_name = ''.join([str(x) for x in kernels])

            first_layer_dict, max_k, max_l = convolution_pass(initial_dict, (10, 1), (1, 1), (1, 1),
                                                              (input_time_length, input_channels), False)
            print(max_k, max_l)
            second_layer_dict, max_k, max_l = convolution_pass(first_layer_dict, (1, input_channels), (1, 1), (1, 1),
                                                               (max_k, max_l), False)
            print(max_k, max_l)
            second_layer_dict, max_k, max_l = convolution_pass(second_layer_dict, (kernels[0], 1), (1, 1),
                                                               (1, 1), (max_k, max_l), True)
            print(max_k, max_l)
            second_layer_dict, max_k, max_l = convolution_pass(second_layer_dict, (10, 1), (1, 1), (3, 1),
                                                               (max_k, max_l), False)
            print(max_k, max_l)

            second_layer_dict, max_k, max_l = convolution_pass(second_layer_dict, (kernels[1], 1), (1, 1),
                                                               (dilations[0], 1), (max_k, max_l), True)
            print(max_k, max_l)

            second_layer_dict, max_k, max_l = convolution_pass(second_layer_dict, (10, 1), (1, 1), (9, 1),
                                                               (max_k, max_l), False)
            print(max_k, max_l)

            second_layer_dict, max_k, max_l = convolution_pass(second_layer_dict, (kernels[2], 1), (1, 1),
                                                               (dilations[1], 1), (max_k, max_l), True)
            print(max_k, max_l)

            second_layer_dict, max_k, max_l = convolution_pass(second_layer_dict, (10, 1), (1, 1), (27, 1),
                                                               (max_k, max_l), False)
            print(max_k, max_l)

            second_layer_dict, max_k, max_l = convolution_pass(second_layer_dict, (kernels[3], 1), (1, 1),
                                                               (dilations[2], 1), (max_k, max_l), True)
            print(max_k, max_l)

            second_layer_dict, max_k, max_l = convolution_pass(second_layer_dict, (2, 1), (1, 1), (81, 1),
                                                               (max_k, max_l), False)
            print(max_k, max_l)

            np.save(f'{home}/outputs/receptive_fields/rf_k_{k_name}_d_{d_name}_sbb_False_0.npy', second_layer_dict[0, 0])
            np.save(f'{home}/outputs/receptive_fields/rf_k_{k_name}_d_{d_name}_sbb_False_600.npy',
                    second_layer_dict[max_k - 1, 0])

            print('done', k_name, d_name)


if __name__ == '__main__':
    # for kernel in [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]:
    # for kernel in [[3, 3, 3, 3]]:
    #     for dilation in [[3, 9, 27, 81]]:
            #     for dilation in [[1, 1, 1, 1], [2, 4, 8, 16], [3, 9, 27, 81]]:
            # visualize_multiple_heat_maps(kernel, dilation)
    create_receptive_fields()