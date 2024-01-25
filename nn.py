import random
import math
import torch

num_layers = 5
neurons_per_layer = 5


layers = []
layer_biases = []
for layer_index in range(num_layers):
    nerves = []
    biases = []
    for nerve_index in range(neurons_per_layer):
        nerve_connections = []
        for i in range(neurons_per_layer):
            nerve_connections.append(random.uniform(-1, 1))
        nerves.append(nerve_connections)
        biases.append(random.uniform(-1, 1))
    layers.append(nerves)
    layer_biases.append(biases)


def l1loss(a, b):
    sum = 0
    for i in range(len(a)):
        sum += abs(a[i] - b[i])
    return sum

all_original_inputs = [
    [1, 2, 3, 4, 5],
    [5, 6, 7, 8, 9]
]
all_targets = [
    [11, 12, 13, 14, 15],
    [-1, -2, -3, -4, -5]
]

for train_step in range(2000000):
        
    for i in range(2):
        original_input = all_original_inputs[i]
        target = all_targets[i]
        # print(original_input)
        # print(target)

        # forward pass
        data = original_input.copy()
        inputs = []
        outputs = []
        for layer_index in range(len(layers)):
            layer = layers[layer_index]
            inputs.append(data.copy())
            layer_output = [None] * len(layer)
            for nerve_index in range(len(layer)):
                nerve = layer[nerve_index]
                nerve_value = 0
                for connection_index in range(len(nerve)):
                    nerve_value += nerve[connection_index] * data[connection_index]
                if layer_index < len(layers) - 1:
                    layer_output[nerve_index] = max(0, nerve_value)
                else:
                    layer_output[nerve_index] = nerve_value
                layer_output[nerve_index] += layer_biases[layer_index][nerve_index]
            outputs.append(layer_output.copy())
            data = layer_output.copy()
        
        # print("outputs", outputs)

        loss = l1loss(data, target)
        
        gradients = []
        # compute loss
        loss_gradient = []
        for i in range(len(data)):
            # derivative of l1 loss wrt output is 1
            # the gradient attempts to bring it to zero
            # so the gradient must negative if the output is positive, and positive if the output is negative
            loss_gradient.append(1 if target[i] < data[i] else -1)
        gradients.append(loss_gradient)
        # print(loss_gradient)
        # exit()

        for layer_index in range(len(layers)-1, -1, -1):
            layer = layers[layer_index]
            input = inputs[layer_index]
            output = outputs[layer_index]
            layer_gradients = []
            for nerve_index in range(len(layer)):
                nerve = layer[nerve_index]
                nerve_weight_gradients = []
                if output[nerve_index] > 0 or layer_index == len(layers) - 1:
                    for connection_index in range(len(nerve)):
                        # compute how changing the weight will affect the output
                        # derivative of y = mxwrt m is x
                        partial_derivative_wrt_m = input[connection_index] * gradients[-1][nerve_index]
                        # update weight
                        nerve[connection_index] -= partial_derivative_wrt_m * 0.00001
                        layer_biases[layer_index][nerve_index] -= gradients[-1][nerve_index] * 0.00001

                        
                        # compute how changing the input will affect the output
                        # derivative of y = mx wrt x is m
                        partial_derivative_wrt_x = nerve[connection_index] * gradients[-1][nerve_index]
                        nerve_weight_gradients.append(partial_derivative_wrt_x)
                else:
                    for connection_index in range(len(nerve)):
                        nerve_weight_gradients.append(0)
                layer_gradients.append(nerve_weight_gradients)


            layer_gradient_sums = []
            for nerve_index in range(len(layer)):
                layer_gradient_sums.append(0)

            for nerve_index in range(len(layer)):
                for connection_index in range(len(nerve)):
                    layer_gradient_sums[connection_index] += layer_gradients[nerve_index][connection_index]
            gradients.append(layer_gradient_sums)

        # reverse
        # gradients = gradients[::-1]
        # print(gradients)
        # update weights
        # for layer_index in range(len(layers)):
        #     layer = layers[layer_index]
        #     layer_gradients = gradients[layer_index]
        #     # print("layer_gradients", layer_gradients)

        #     for nerve_index in range(len(layer)):
        #         nerve = layer[nerve_index]
        #         for connection_index in range(len(nerve)):
        #             nerve[connection_index] -= layer_gradients[nerve_index][connection_index] * 0.001

        # print("original input", original_input)
        
        if train_step % 1000 == 0:
            print("train step", train_step)
            print("output", data)
            print("target", target)
            print(loss)
        # print("gradients", gradients)
        # print("weights", layers[0])
        for val in data:
            if math.isnan(val):
                print("contains nan")
                exit()
        # exit()








