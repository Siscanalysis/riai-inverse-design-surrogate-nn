''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Model exported to Python. Use Network().calculate_outputs([...]).

Input Names: 
	0) Cement_kg_m3
	1) FineAgg_kg_m3
	2) CoarseAgg_kg_m3
	3) Water_kg_m3
	4) WaterReducer_kg_m3
	5) FlyAsh_kg_m3
	6) Accelerator_kg_m3
	7) SilicaFume_kg_m3
	8) TestAge_days

For batch prediction (input must be np.ndarray):
	nn.calculate_batch_output(np.array([[1, 2], [4, 5]]))
''' 
import math
import numpy as np
import pandas as pd

class Network:

	def __init__(self):
		self.inputs_number = 9
		self.input_names = ['Cement_kg_m3', 'FineAgg_kg_m3', 'CoarseAgg_kg_m3', 'Water_kg_m3', 'WaterReducer_kg_m3', 'FlyAsh_kg_m3', 'Accelerator_kg_m3', 'SilicaFume_kg_m3', 'TestAge_days']

	@staticmethod
	def Identity(x):
		return x

	@staticmethod
	def Tanh(x):
		return np.tanh(x)

	def calculate_outputs(self, inputs):
		Cement_kg_m3 = inputs[0]
		FineAgg_kg_m3 = inputs[1]
		CoarseAgg_kg_m3 = inputs[2]
		Water_kg_m3 = inputs[3]
		WaterReducer_kg_m3 = inputs[4]
		FlyAsh_kg_m3 = inputs[5]
		Accelerator_kg_m3 = inputs[6]
		SilicaFume_kg_m3 = inputs[7]
		TestAge_days = inputs[8]

		scaled_Cement_kg_m3 = Cement_kg_m3*0.01008542255-3.642921925
		scaled_FineAgg_kg_m3 = FineAgg_kg_m3*0.005547297187-4.753080368
		scaled_CoarseAgg_kg_m3 = CoarseAgg_kg_m3*0.003701786976-3.586598396
		scaled_Water_kg_m3 = Water_kg_m3*0.02708028629-4.783146381
		scaled_WaterReducer_kg_m3 = WaterReducer_kg_m3*0.2169221789-1.193800926
		scaled_FlyAsh_kg_m3 = FlyAsh_kg_m3*0.02764091827-0.7177001238
		scaled_Accelerator_kg_m3 = Accelerator_kg_m3*0.1807288378-0.1600123793
		scaled_SilicaFume_kg_m3 = SilicaFume_kg_m3*0.4673493803-0.1103060618
		scaled_TestAge_days = TestAge_days*0.09520531446-1.684720159
		dense_layer_1_output_0 = self.Tanh( 1.473006248 + (0.04146245494*scaled_Cement_kg_m3) + (-0.3355035484*scaled_FineAgg_kg_m3) + (-0.9354643822*scaled_CoarseAgg_kg_m3) + (1.310935378*scaled_Water_kg_m3) + (-0.09412388504*scaled_WaterReducer_kg_m3) + (0.1230836287*scaled_FlyAsh_kg_m3) + (-0.1829253137*scaled_Accelerator_kg_m3) + (-0.3798045218*scaled_SilicaFume_kg_m3) + (0.02488416992*scaled_TestAge_days) )
		dense_layer_1_output_1 = self.Tanh( -0.3904629648 + (0.3743785918*scaled_Cement_kg_m3) + (-0.5373387337*scaled_FineAgg_kg_m3) + (-0.3074010015*scaled_CoarseAgg_kg_m3) + (0.09318710119*scaled_Water_kg_m3) + (-0.026028255*scaled_WaterReducer_kg_m3) + (0.8646964431*scaled_FlyAsh_kg_m3) + (-0.04981450737*scaled_Accelerator_kg_m3) + (-0.1012014374*scaled_SilicaFume_kg_m3) + (-0.01827514544*scaled_TestAge_days) )
		dense_layer_1_output_2 = self.Tanh( 0.4746104777 + (-0.4060034454*scaled_Cement_kg_m3) + (-0.4298837185*scaled_FineAgg_kg_m3) + (-0.3157092333*scaled_CoarseAgg_kg_m3) + (0.08643170446*scaled_Water_kg_m3) + (-0.2115524262*scaled_WaterReducer_kg_m3) + (-0.3007702231*scaled_FlyAsh_kg_m3) + (-0.1318445504*scaled_Accelerator_kg_m3) + (-0.09294509888*scaled_SilicaFume_kg_m3) + (-0.1940863878*scaled_TestAge_days) )
		dense_layer_1_output_3 = self.Tanh( -0.1523843259 + (-0.4256442189*scaled_Cement_kg_m3) + (0.7013630867*scaled_FineAgg_kg_m3) + (1.156660199*scaled_CoarseAgg_kg_m3) + (-0.9089245796*scaled_Water_kg_m3) + (-0.1456351876*scaled_WaterReducer_kg_m3) + (-1.084731102*scaled_FlyAsh_kg_m3) + (-0.3804892302*scaled_Accelerator_kg_m3) + (0.0334841907*scaled_SilicaFume_kg_m3) + (0.02789361402*scaled_TestAge_days) )
		dense_layer_1_output_4 = self.Tanh( -0.345426321 + (0.3690585494*scaled_Cement_kg_m3) + (-0.6301394701*scaled_FineAgg_kg_m3) + (-0.0003828287881*scaled_CoarseAgg_kg_m3) + (-1.154114842*scaled_Water_kg_m3) + (-0.2871964872*scaled_WaterReducer_kg_m3) + (0.3317716718*scaled_FlyAsh_kg_m3) + (-0.225922361*scaled_Accelerator_kg_m3) + (-0.1461304873*scaled_SilicaFume_kg_m3) + (0.01794848591*scaled_TestAge_days) )
		dense_layer_1_output_5 = self.Tanh( 0.419678688 + (-0.01295754593*scaled_Cement_kg_m3) + (-0.1562808752*scaled_FineAgg_kg_m3) + (-0.8460399508*scaled_CoarseAgg_kg_m3) + (1.352445364*scaled_Water_kg_m3) + (-0.3051399887*scaled_WaterReducer_kg_m3) + (0.4768893719*scaled_FlyAsh_kg_m3) + (-0.001824146369*scaled_Accelerator_kg_m3) + (-0.1037592068*scaled_SilicaFume_kg_m3) + (-0.05439930037*scaled_TestAge_days) )
		dense_layer_1_output_6 = self.Tanh( -0.8007000685 + (1.221448541*scaled_Cement_kg_m3) + (-0.5773018599*scaled_FineAgg_kg_m3) + (0.2227224857*scaled_CoarseAgg_kg_m3) + (-0.4748704731*scaled_Water_kg_m3) + (0.3932888806*scaled_WaterReducer_kg_m3) + (0.3341915905*scaled_FlyAsh_kg_m3) + (-0.4562169611*scaled_Accelerator_kg_m3) + (-0.2811943889*scaled_SilicaFume_kg_m3) + (0.02144267783*scaled_TestAge_days) )
		dense_layer_1_output_7 = self.Tanh( -0.2289734483 + (0.411950022*scaled_Cement_kg_m3) + (-0.06890229136*scaled_FineAgg_kg_m3) + (-1.128086567*scaled_CoarseAgg_kg_m3) + (0.8244467974*scaled_Water_kg_m3) + (0.3814511299*scaled_WaterReducer_kg_m3) + (0.5010895133*scaled_FlyAsh_kg_m3) + (0.3146092892*scaled_Accelerator_kg_m3) + (-0.2143384665*scaled_SilicaFume_kg_m3) + (0.03262827545*scaled_TestAge_days) )
		dense_layer_1_output_8 = self.Tanh( 0.5145612359 + (0.9029397368*scaled_Cement_kg_m3) + (-0.2623242438*scaled_FineAgg_kg_m3) + (-1.139389038*scaled_CoarseAgg_kg_m3) + (0.7409703135*scaled_Water_kg_m3) + (0.4489476383*scaled_WaterReducer_kg_m3) + (0.2301191986*scaled_FlyAsh_kg_m3) + (-0.03791128099*scaled_Accelerator_kg_m3) + (0.1296496093*scaled_SilicaFume_kg_m3) + (0.002748032566*scaled_TestAge_days) )
		dense_layer_1_output_9 = self.Tanh( -0.06807979196 + (-0.07018794864*scaled_Cement_kg_m3) + (-0.1393634975*scaled_FineAgg_kg_m3) + (-0.0428179279*scaled_CoarseAgg_kg_m3) + (0.3336191475*scaled_Water_kg_m3) + (-0.1899397671*scaled_WaterReducer_kg_m3) + (-0.4399750233*scaled_FlyAsh_kg_m3) + (-0.06981174648*scaled_Accelerator_kg_m3) + (0.02602636069*scaled_SilicaFume_kg_m3) + (0.02163969167*scaled_TestAge_days) )
		dense_layer_1_output_10 = self.Tanh( -0.2298167944 + (-0.1954885423*scaled_Cement_kg_m3) + (-0.1710293591*scaled_FineAgg_kg_m3) + (0.1415832043*scaled_CoarseAgg_kg_m3) + (0.05560301989*scaled_Water_kg_m3) + (-0.2261928022*scaled_WaterReducer_kg_m3) + (0.1282065213*scaled_FlyAsh_kg_m3) + (-0.2424024791*scaled_Accelerator_kg_m3) + (-0.002990685171*scaled_SilicaFume_kg_m3) + (-0.05886955932*scaled_TestAge_days) )
		dense_layer_1_output_11 = self.Tanh( -9.410219354e-05 + (0.09896141291*scaled_Cement_kg_m3) + (-0.02534750849*scaled_FineAgg_kg_m3) + (0.03545098379*scaled_CoarseAgg_kg_m3) + (0.06488309801*scaled_Water_kg_m3) + (0.08125602454*scaled_WaterReducer_kg_m3) + (-0.08456422389*scaled_FlyAsh_kg_m3) + (-0.08393768221*scaled_Accelerator_kg_m3) + (-0.01339187473*scaled_SilicaFume_kg_m3) + (0.0113838641*scaled_TestAge_days) )
		approximation_layer_output_0 = self.Identity( -0.2328972518 + (0.5009697676*dense_layer_1_output_0) + (0.8347575665*dense_layer_1_output_1) + (-1.638963699*dense_layer_1_output_2) + (0.3931051791*dense_layer_1_output_3) + (0.8627032042*dense_layer_1_output_4) + (-0.109924145*dense_layer_1_output_5) + (-0.390335083*dense_layer_1_output_6) + (-1.154671192*dense_layer_1_output_7) + (0.8783288598*dense_layer_1_output_8) + (0.543643713*dense_layer_1_output_9) + (-0.7480915785*dense_layer_1_output_10) + (0.01936690137*dense_layer_1_output_11) )
		approximation_layer_output_1 = self.Identity( -0.9164974689 + (1.470229983*dense_layer_1_output_0) + (1.69467032*dense_layer_1_output_1) + (-0.2729686201*dense_layer_1_output_2) + (1.491572738*dense_layer_1_output_3) + (-0.8032118082*dense_layer_1_output_4) + (-1.28017199*dense_layer_1_output_5) + (-0.8167824745*dense_layer_1_output_6) + (-0.2099075764*dense_layer_1_output_7) + (1.342028499*dense_layer_1_output_8) + (-1.409093142*dense_layer_1_output_9) + (1.090621233*dense_layer_1_output_10) + (0.07777994126*dense_layer_1_output_11) )
		unscaling_layer_output_0=approximation_layer_output_0*12.52991581+38.26176834
		unscaling_layer_output_1=approximation_layer_output_1*36.27813339+178.5
		CompressiveStrength_MPa = unscaling_layer_output_0
		Slump_mm = unscaling_layer_output_1
		outputs = [CompressiveStrength_MPa, Slump_mm]
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
	Cement_kg_m3 = 0  # Cement_kg_m3
	FineAgg_kg_m3 = 0  # FineAgg_kg_m3
	CoarseAgg_kg_m3 = 0  # CoarseAgg_kg_m3
	Water_kg_m3 = 0  # Water_kg_m3
	WaterReducer_kg_m3 = 0  # WaterReducer_kg_m3
	FlyAsh_kg_m3 = 0  # FlyAsh_kg_m3
	Accelerator_kg_m3 = 0  # Accelerator_kg_m3
	SilicaFume_kg_m3 = 0  # SilicaFume_kg_m3
	TestAge_days = 0  # TestAge_days

	# --- Data conversion (DO NOT modify) ---
	inputs = []

	inputs.append(Cement_kg_m3)
	inputs.append(FineAgg_kg_m3)
	inputs.append(CoarseAgg_kg_m3)
	inputs.append(Water_kg_m3)
	inputs.append(WaterReducer_kg_m3)
	inputs.append(FlyAsh_kg_m3)
	inputs.append(Accelerator_kg_m3)
	inputs.append(SilicaFume_kg_m3)
	inputs.append(TestAge_days)

	nn = Network()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
