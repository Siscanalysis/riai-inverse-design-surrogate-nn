''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Model exported to Python. Use Network().calculate_outputs([...]).

Input Names: 
	0) cutting_speed
	1) feed
	2) depth_of_cut
	3) tool_wear_VB

For batch prediction (input must be np.ndarray):
	nn.calculate_batch_output(np.array([[1, 2], [4, 5]]))
''' 
import math
import numpy as np
import pandas as pd

class Network:

	def __init__(self):
		self.inputs_number = 4
		self.input_names = ['cutting_speed', 'feed', 'depth_of_cut', 'tool_wear_VB']

	@staticmethod
	def Identity(x):
		return x

	@staticmethod
	def Tanh(x):
		return np.tanh(x)

	def calculate_outputs(self, inputs):
		cutting_speed = inputs[0]
		feed = inputs[1]
		depth_of_cut = inputs[2]
		tool_wear_VB = inputs[3]

		scaled_cutting_speed = cutting_speed*0.04404795915-15.32154655
		scaled_feed = feed*41.62880325-4.095374107
		scaled_depth_of_cut = depth_of_cut*4.971045494-2.263840914
		scaled_tool_wear_VB = tool_wear_VB*9.705727577-0.524633944
		dense_layer_1_output_0 = self.Tanh( -0.09185280651 + (-0.2047008425*scaled_cutting_speed) + (-0.787545085*scaled_feed) + (-0.3412374258*scaled_depth_of_cut) + (-0.3127225041*scaled_tool_wear_VB) )
		dense_layer_1_output_1 = self.Tanh( 0.007145399693 + (0.9110519886*scaled_cutting_speed) + (-0.4117322564*scaled_feed) + (-0.278711617*scaled_depth_of_cut) + (0.111624375*scaled_tool_wear_VB) )
		dense_layer_1_output_2 = self.Tanh( 0.03981614858 + (0.2353466749*scaled_cutting_speed) + (-0.2134883106*scaled_feed) + (-0.9455154538*scaled_depth_of_cut) + (-0.4937115014*scaled_tool_wear_VB) )
		dense_layer_1_output_3 = self.Tanh( 0.03238311782 + (0.6575165987*scaled_cutting_speed) + (0.9507523775*scaled_feed) + (-0.09933747351*scaled_depth_of_cut) + (0.2490080297*scaled_tool_wear_VB) )
		approximation_layer_output_0 = self.Identity( -0.002671034075 + (-0.3361181915*dense_layer_1_output_0) + (-0.3194186091*dense_layer_1_output_1) + (-0.2485544533*dense_layer_1_output_2) + (0.4346860647*dense_layer_1_output_3) )
		approximation_layer_output_1 = self.Identity( -0.0006784461439 + (-0.08778019994*dense_layer_1_output_0) + (0.04579597712*dense_layer_1_output_1) + (-1.188303471*dense_layer_1_output_2) + (0.3138833344*dense_layer_1_output_3) )
		unscaling_layer_output_0=approximation_layer_output_0*0.3177354038+0.7007522583
		unscaling_layer_output_1=approximation_layer_output_1*93.18291473+187.4463806
		Ra = max(0.1420000046, unscaling_layer_output_0)
		Ra = min(1.812000036, Ra)
		F = max(69.33789825, unscaling_layer_output_1)
		F = min(384.2600098, F)
		outputs = [Ra, F]
		return outputs

	def calculate_batch_output(self, input_batch):
		output_batch = np.zeros((len(input_batch), 2))
		for i in range(len(input_batch)):
			inputs = list(input_batch[i])
			output = self.calculate_outputs(inputs)
			output_batch[i] = output
		return output_batch

def main():

	# Introduce your input values here
	cutting_speed = 0  # cutting_speed
	feed = 0  # feed
	depth_of_cut = 0  # depth_of_cut
	tool_wear_VB = 0  # tool_wear_VB

	# --- Data conversion (DO NOT modify) ---
	inputs = []

	inputs.append(cutting_speed)
	inputs.append(feed)
	inputs.append(depth_of_cut)
	inputs.append(tool_wear_VB)

	nn = Network()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
