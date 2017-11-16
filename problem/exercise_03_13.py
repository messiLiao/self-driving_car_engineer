import numpy as np

def gradient_decent(x_list, y_list):
	w = [0.1, 0.2]
	b = -0.5
	eita = 0.1
	while True:
		end_flag = True
		for xi, yi in zip(x_list, y_list):
			if yi*(w[0]*xi[0] + w[1]*xi[1] + b) <= 0:
				end_flag = False
				w[0] = w[0] + yi * xi[0] * eita
				w[1] = w[1] + yi * xi[1] * eita
				b = b + yi * eita
			print w, b
		if end_flag:
			break
	print w, b


def sigmoid(x):
    # TODO: Implement sigmoid function
    pass
    return (1.0 / (1.0 + np.exp(-x)))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# TODO: Calculate the output
output = sigmoid((inputs.dot(weights) + bias))

print('Output:')
print(output)

if __name__ == '__main__':
	x_list = [(0, 1), (1, 0), (0, 0), (1, 1)]
	y_list = [-1, -1, -1, 1]
	gradient_decent(x_list, y_list)