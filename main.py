import random
import math


layers = []
for layer_index in range(5):
    nerves = []
    for nerve_index in range(20):
        nerve_connections = []
        for i in range(20):
            nerve_connections.append(random.uniform(-1, 1))
        nerves.append(nerve_connections)
    layers.append(nerves)


original_input_1 = []
original_input_2 = []
target_output_1 = []
target_output_2 = []
for i in range(20):
    original_input_1.append(i)
    target_output_1.append(20+i)
    original_input_2.append(100-i)
    target_output_2.append(100+i)

for iteration in range(500000):
    for i in range (2):
        if i == 0:
            original_input = original_input_1
            target_output = target_output_1
        else:
            original_input = original_input_2
            target_output = target_output_2
        input = original_input
        inputs = []
        outputs = []
        for layer_index in range(len(layers)):
            inputs.append(input)
            layer = layers[layer_index]
            layer_output = []
            for nerve_index in range(len(layer)):
                nerve = layer[nerve_index]
                nerve_output = []
                sum = 0
                for connection_index in range(len(nerve)):
                    nerve_output.append(nerve[connection_index] * input[connection_index])
                    sum += nerve_output[connection_index]
                if layer_index < len(layers) - 1:
                    layer_output.append(max(0, sum)) # if it's less than 0, just output 0
                else:
                    layer_output.append(sum)
            input = layer_output
            outputs.append(layer_output)
        final_output = input
        loss = []
        for i in range(len(input)):
            loss.append(target_output[i] - input[i])

        target_info = loss
        for layer_index in range(len(layers)-1, -1, -1):
            layer = layers[layer_index]
            input = inputs[layer_index]
            nerve_directions = []
            for nerve_index in range(len(layer)):
                nerve = layer[nerve_index]
                directions = []
                for connection_index in range(len(nerve)):
                    # if loss[nerve_index] > 0: # if it needs to go up to be correct, increase the weight
                    direction = 0
                    if target_info[nerve_index] > 0:
                        direction = 1
                    elif target_info[nerve_index] < 0:
                        direction = -1
                    nerve[connection_index] += abs(nerve[connection_index]) * 0.01 * direction
                    # print("udpate", nerve[connection_index] * 0.0001 * direction)
                    directions.append(direction)
                nerve_directions.append(directions)
            target_info = []
            for connection_index in range(len(input)):
                sum = 0
                for nerve_index in range(len(layer)):
                    nerve = layer[nerve_index]
                    sum += nerve_directions[nerve_index][connection_index]
                target_info.append(sum)
            # print("target info", target_info)
            
        if iteration % 500 == 0:
            print("input         ", original_input)
            print("output        ", final_output)
            print("target_output ", target_output)
            print("loss          ", loss)
            loss_sum = 0
            for i in range(len(loss)):
                loss_sum += abs(loss[i])
            print("loss_sum", loss_sum)