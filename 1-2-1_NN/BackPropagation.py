import numpy as np

input_data = 0.8
weight = [[0.3,-0.3],[-0.1,0.1]]
bias = [[0,0],[0]]
output_data = 0.0
teach_data = 0.72
middle_layer = [0,0]

learning_rate = 0.3
diff_teach_data = 0.0

def tahn_function(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def diff_tahn(x):
    return 4/((np.exp(x) + np.exp(-x))*(np.exp(x) + np.exp(-x)))

def update_weight():
    return

def forward_propagation(weight,input,bias):
    init_middle_layer = [0,0]
    init_output = 0.0
    for l_idx,layer in enumerate(weight):
        for e_idx,edge in enumerate(layer):
            if(l_idx == 0):
                init_middle_layer[e_idx] = tahn_function(init_middle_layer[e_idx] + edge * input + bias[l_idx][e_idx])
            else:
                init_output = init_output + weight[l_idx][e_idx] * init_middle_layer[e_idx]
    init_output = init_output + bias[1][0]
    return init_output,init_middle_layer

def back_propagation(diff,weight,bias,input):
    np_weight = np.array(weight)
    output_weight = [[0.3,-0.3],[-0.1,0.1]]
    output_bias = [[0,0],[0]]
    gradient_list = [[0,0,0,0],[0,0,0]]
    # 勾配を求める
    for l_idx in reversed(range(np_weight.shape[0])):
        for e_idx in range(np_weight.shape[0]):
            if(l_idx == 1):
                gradient = diff*middle_layer[e_idx]
                gradient_list[l_idx][e_idx] = gradient
                output_weight[l_idx][e_idx] = weight[l_idx][e_idx] - learning_rate*gradient
            else:
                gradient = diff * weight[(np_weight.shape[0]-1)-l_idx][e_idx]*diff_tahn(weight[l_idx][e_idx]*input)*input
                gradient_list[l_idx][e_idx] = gradient
                output_weight[l_idx][e_idx] = weight[l_idx][e_idx] - learning_rate*gradient
    for l_idx in reversed(range(len(output_bias))):
        for e_idx in range(len(output_bias[l_idx])):
            if(l_idx==1):
                output_bias[l_idx][e_idx] = bias[l_idx][e_idx]-diff*learning_rate
                gradient_list[l_idx][e_idx] = diff*learning_rate
            else:
                gradient = diff * weight[(np_weight.shape[0]-1)-l_idx][e_idx]*diff_tahn(weight[l_idx][e_idx]*input)
                gradient_list[l_idx][e_idx] = gradient
                output_bias[l_idx][e_idx] = bias[l_idx][e_idx] - learning_rate*gradient
    return output_weight,output_bias,gradient_list

if __name__ == "__main__":
    for i in range(300):
        output_data,middle_layer = forward_propagation(weight=weight,input=input_data,bias=bias)
        diff_teach_data = -(teach_data - output_data)
        if(i<=3 or i==299):
            print("重みは{0}".format(weight))
            print("バイアスは{0}".format(bias))
            print("中間層Z：{0}".format(middle_layer))
            print("出力データ：{0}".format(output_data))
            print("差分データ：{0}".format(diff_teach_data))
        weight,bias,gradients = back_propagation(diff=diff_teach_data,weight=weight,bias=bias,input=input_data)
        if(i<=3 or i==299):
            print("δ1：{0}".format(gradients[0]))
            print("δ2：{0}".format(gradients[1]))
            print("ΔW1,Δb1：{0},{1}".format([learning_rate * gradients[0][0],learning_rate * gradients[0][1]],[learning_rate * gradients[0][2],learning_rate * gradients[0][3]]))
            print("ΔW2,Δb2：{0},{1}".format([learning_rate * gradients[1][0],learning_rate * gradients[1][1]],[learning_rate * gradients[1][2]]))