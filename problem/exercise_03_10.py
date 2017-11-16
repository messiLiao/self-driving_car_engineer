import pandas as pd

# TODO: Set weight1, weight2, and bias
weight1 = 1.0
weight2 = 1.0
bias = -1.5


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))


def sign(x):
	if x >= 0:
		return 1
	else:
		return 0

def grad(x_list, y_list):
	w = [0.0, 0.0]
	b = 0.0
	eita = 1.0
	count = 100
	while count > 0:
		count -= 1
		all_pass = True
		for (x0, x1), y in zip(x_list, y_list):
			w[0] = w[0] + y * x0 * eita
			w[1] = w[1] + y * x1 * eita
			b = b - y * eita
			print w, b, w[0] * x0 + w[1] * x1 + b
			if sign((w[0] * x0 + w[1] * x1 + b)) != y:
				all_pass = False
		if all_pass:
			break
		# print w, b

	print w, b

def main():
	x_list = [(1, 0), (0, 1), (0, 0), (1, 1)]
	y_list = [0, 0, 0, 1]
	grad(x_list, y_list)

if __name__ == '__main__':
	main()