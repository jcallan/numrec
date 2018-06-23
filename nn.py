import sys
import mnist
import cProfile
from random import randrange
from copy import deepcopy


class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs  = num_inputs
        self.num_neurons = num_neurons
        self.weights = []
        self.inputs  = [0 for _ in range(self.num_inputs)]
        self.neurons = [0 for _ in range(self.num_neurons)]
        for _i in range(self.num_neurons):
            self.weights.append([randrange(512) - 256 for _ in range(self.num_inputs)])
        #print('Layer has {:d} inputs and {:d} output neurons'.format(len(self.inputs),
        #                                                             len(self.neurons)))
    
    #def calc_output(self, i):
    #    result = 0
    #    for j in range(self.num_inputs):
    #        result += self.weights[i][j] * self.inputs[j]
    #    result = int(result / (self.num_inputs * self.num_neurons))
    #    return result

    def calc_outputs(self):
        for i in range(self.num_neurons):
            result = 0
            for j in range(self.num_inputs):
                result += self.weights[i][j] * self.inputs[j]
            self.neurons[i] = int(result / (self.num_inputs * self.num_neurons))

class Net:
    def __init__(self, num_neurons):
        '''Initialise a Net with a certain number of neurons per layer'''
        self.num_layers = len(num_neurons) - 1
        self.num_neurons = num_neurons
        self.layers = []
        for i in range(1, self.num_layers + 1):
            self.layers.append(Layer(self.num_neurons[i - 1], self.num_neurons[i]))
    
    def error_score(self, desired_outputs):
        outputs = self.layers[self.num_layers - 1].neurons
        error_total = 0
        for i in range(len(desired_outputs)):
            error_total += (outputs[i] - desired_outputs[i]) ** 2
        return error_total
    
    def tweak(self, inputs, desired_outputs, as_is=None):
        if as_is is None:
            self.run(inputs, 0)
            as_is = self.error_score(desired_outputs)
        for i in range(self.num_layers):
            #print('Starting score is {:d}, tweaking layer {:d}'.format(as_is, i))
            old_weights = deepcopy(self.layers[i].weights)
            old_neurons = deepcopy(self.layers[i].neurons)
            for _ in range(self.layers[i].num_neurons * self.layers[i].num_inputs >> 2):
                j = randrange(self.layers[i].num_neurons)
                k = randrange(self.layers[i].num_inputs)
                delta = randrange(64) - 32
                self.layers[i].weights[j][k] += delta
            self.run(inputs, i)             # We only affect this and subsequent layers
            tweaked = self.error_score(desired_outputs)
            #print('Tweaked score is {:d}'.format(tweaked))
            if tweaked > as_is:
                self.layers[i].weights = old_weights
                self.layers[i].neurons = old_neurons

        if tweaked > as_is:
            return as_is
        else:
            return tweaked
            
    def run(self, inputs, start_layer=0):
        for i in range(start_layer, self.num_layers):
            self.layers[i].inputs = inputs if i == 0 else self.layers[i - 1].neurons
            #print('Running layer {:d}, {:d} inputs'.format(i, len(self.layers[i].inputs)))
            self.layers[i].calc_outputs()

def main():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    train_images_flat = train_images.reshape((train_images.shape[0],
                                              train_images.shape[1] * train_images.shape[2]))
    
    #print('train_images[0][0] is ' + repr(train_images[0][0]))
    #test_images = mnist.test_images()
    #test_labels = mnist.test_labels()

    my_net = Net((784, 32, 32, 10))

    for n in range(len(train_images)):
        input_data = train_images_flat[n]
        desired_outputs = [0 if i != train_labels[n] else 100 for i in range(10)]
        #print(input_data)
        print('{:5d} {:1d} '.format(n, train_labels[n]), end='')
        #print(desired_outputs)
        t = None
        for _ in range(1):
            t = my_net.tweak(input_data, desired_outputs, t)
        print(' {:6d}  '.format(t), end='')
        print(my_net.layers[2].neurons)
    print()
    print(my_net.layers[2].neurons)
    print(my_net.error_score(desired_outputs))
    
if __name__ == '__main__':
    main()
    #cProfile.run('main()')

