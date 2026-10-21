''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Model exported to Python. Use Network().calculate_outputs([...]).

Input Names: 
	0) Al
	1) Co
	2) Cr
	3) Cu
	4) Fe
	5) Mn
	6) Mo
	7) Ni
	8) Ti
	9) V

For batch prediction (input must be np.ndarray):
	nn.calculate_batch_output(np.array([[1, 2], [4, 5]]))
''' 
import math
import numpy as np
import pandas as pd

class Network:

	def __init__(self):
		self.inputs_number = 10
		self.input_names = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'V']

	@staticmethod
	def Identity(x):
		return x

	@staticmethod
	def Tanh(x):
		return np.tanh(x)

	def calculate_outputs(self, inputs):
		Al = inputs[0]
		Co = inputs[1]
		Cr = inputs[2]
		Cu = inputs[3]
		Fe = inputs[4]
		Mn = inputs[5]
		Mo = inputs[6]
		Ni = inputs[7]
		Ti = inputs[8]
		V = inputs[9]

		scaled_Al = Al*7.211092472-1.184740305
		scaled_Co = Co*11.30341434-1.678230166
		scaled_Cr = Cr*14.56493378-2.623915434
		scaled_Cu = Cu*11.4944315-0.9189026952
		scaled_Fe = Fe*11.9285841-2.208494902
		scaled_Mn = Mn*15.61688709-0.3256144822
		scaled_Mo = Mo*25.84593964-0.2515687048
		scaled_Ni = Ni*11.18345547-1.889505744
		scaled_Ti = Ti*15.01351643-0.4049523473
		scaled_V = V*18.97544098-0.2938098609
		dense_layer_1_output_0 = self.Tanh( 0.02809223905 + (0.02343267947*scaled_Al) + (-0.09709390998*scaled_Co) + (-0.03622171283*scaled_Cr) + (-0.001049169572*scaled_Cu) + (-0.06795194745*scaled_Fe) + (-0.01106783655*scaled_Mn) + (0.05503439903*scaled_Mo) + (0.07625805587*scaled_Ni) + (0.06510430574*scaled_Ti) + (-0.004277005326*scaled_V) )
		dense_layer_1_output_1 = self.Tanh( -0.4676351845 + (-1.586142778*scaled_Al) + (0.5971301198*scaled_Co) + (0.1175711751*scaled_Cr) + (0.4977466762*scaled_Cu) + (0.1999198794*scaled_Fe) + (0.1491964906*scaled_Mn) + (-0.08807309717*scaled_Mo) + (0.5550606251*scaled_Ni) + (0.8933131695*scaled_Ti) + (-0.1947733164*scaled_V) )
		dense_layer_1_output_2 = self.Tanh( -0.3683581352 + (-0.08869209886*scaled_Al) + (-0.008301691152*scaled_Co) + (-0.2609931827*scaled_Cr) + (-0.2946729064*scaled_Cu) + (0.4782681763*scaled_Fe) + (0.1588550955*scaled_Mn) + (0.2866176963*scaled_Mo) + (0.2576204836*scaled_Ni) + (-0.2503438592*scaled_Ti) + (-0.2273883075*scaled_V) )
		dense_layer_1_output_3 = self.Tanh( 0.2345270514 + (0.1952441633*scaled_Al) + (-0.2811706066*scaled_Co) + (0.1736843884*scaled_Cr) + (0.6810194254*scaled_Cu) + (-0.02914050967*scaled_Fe) + (-0.07225435972*scaled_Mn) + (0.4222991168*scaled_Mo) + (-0.2268377542*scaled_Ni) + (0.02561718598*scaled_Ti) + (-0.8285344243*scaled_V) )
		dense_layer_1_output_4 = self.Tanh( 0.1359609962 + (-0.07564923912*scaled_Al) + (-0.07513976097*scaled_Co) + (0.3380773962*scaled_Cr) + (-0.06634981185*scaled_Cu) + (-0.4652349651*scaled_Fe) + (0.02984481864*scaled_Mn) + (-0.04069691896*scaled_Mo) + (-0.07059732825*scaled_Ni) + (0.3717722893*scaled_Ti) + (0.1660165489*scaled_V) )
		dense_layer_1_output_5 = self.Tanh( 0.1000568196 + (0.6072679758*scaled_Al) + (-0.06205450371*scaled_Co) + (-0.2573488355*scaled_Cr) + (-0.001658528578*scaled_Cu) + (-0.1027315184*scaled_Fe) + (-0.2310498655*scaled_Mn) + (-0.218843624*scaled_Mo) + (-0.3290075064*scaled_Ni) + (0.1145270616*scaled_Ti) + (-0.05283529684*scaled_V) )
		dense_layer_1_output_6 = self.Tanh( 0.3288666606 + (0.01465681568*scaled_Al) + (-0.1748888791*scaled_Co) + (0.1212085783*scaled_Cr) + (0.3753463328*scaled_Cu) + (0.06732535362*scaled_Fe) + (0.01871088333*scaled_Mn) + (-0.2218973488*scaled_Mo) + (0.4487543404*scaled_Ni) + (-0.635661602*scaled_Ti) + (-0.06426362693*scaled_V) )
		dense_layer_1_output_7 = self.Tanh( -0.0820325166 + (0.1404224783*scaled_Al) + (0.1770175993*scaled_Co) + (0.00250738184*scaled_Cr) + (0.01433213241*scaled_Cu) + (-0.122755155*scaled_Fe) + (-0.1757336557*scaled_Mn) + (-0.08982230723*scaled_Mo) + (-0.04636159167*scaled_Ni) + (0.1284455806*scaled_Ti) + (-0.08713120222*scaled_V) )
		dense_layer_1_output_8 = self.Tanh( 0.1808908135 + (-0.2261574864*scaled_Al) + (-0.3190716803*scaled_Co) + (0.08092620969*scaled_Cr) + (0.3948693275*scaled_Cu) + (-0.1448153406*scaled_Fe) + (-0.1568977088*scaled_Mn) + (-0.338707149*scaled_Mo) + (0.4116799533*scaled_Ni) + (-0.007935966365*scaled_Ti) + (0.2650396526*scaled_V) )
		dense_layer_1_output_9 = self.Tanh( -0.2831938267 + (-0.2282671034*scaled_Al) + (0.2742633522*scaled_Co) + (0.07848446071*scaled_Cr) + (0.1259907782*scaled_Cu) + (0.3426280022*scaled_Fe) + (0.00430084765*scaled_Mn) + (0.1269476265*scaled_Mo) + (-0.2435948253*scaled_Ni) + (-0.5890340805*scaled_Ti) + (0.1390488595*scaled_V) )
		dense_layer_1_output_10 = self.Tanh( 0.02885005064 + (-0.01944759488*scaled_Al) + (-0.02549361624*scaled_Co) + (-0.05685807392*scaled_Cr) + (-0.003349139821*scaled_Cu) + (0.045631852*scaled_Fe) + (-0.05909034237*scaled_Mn) + (-0.08720047772*scaled_Mo) + (0.0534292236*scaled_Ni) + (0.06621672213*scaled_Ti) + (-0.02115561441*scaled_V) )
		dense_layer_1_output_11 = self.Tanh( -0.1333458573 + (0.1833798438*scaled_Al) + (0.1225366294*scaled_Co) + (-0.04305982962*scaled_Cr) + (0.4417238235*scaled_Cu) + (-0.0258568693*scaled_Fe) + (0.002846211893*scaled_Mn) + (-0.3093580604*scaled_Mo) + (-0.4239532053*scaled_Ni) + (-0.2043813169*scaled_Ti) + (-0.154967472*scaled_V) )
		dense_layer_1_output_12 = self.Tanh( 0.01788052917 + (0.05230760202*scaled_Al) + (0.2648800015*scaled_Co) + (-0.1812473089*scaled_Cr) + (-0.4064041674*scaled_Cu) + (0.2531735301*scaled_Fe) + (0.285516113*scaled_Mn) + (-0.5186365843*scaled_Mo) + (0.06267241389*scaled_Ni) + (0.02644746751*scaled_Ti) + (-0.01707387343*scaled_V) )
		dense_layer_1_output_13 = self.Tanh( -0.5083307028 + (-0.4487892687*scaled_Al) + (0.2358887941*scaled_Co) + (-0.5314557552*scaled_Cr) + (-0.1181559339*scaled_Cu) + (0.5995140076*scaled_Fe) + (-0.4608083963*scaled_Mn) + (0.4593996704*scaled_Mo) + (-0.02103679441*scaled_Ni) + (0.2252072394*scaled_Ti) + (0.5024388433*scaled_V) )
		dense_layer_1_output_14 = self.Tanh( 0.3527347147 + (0.03895663098*scaled_Al) + (-0.2400137782*scaled_Co) + (0.0152586028*scaled_Cr) + (-0.2722527683*scaled_Cu) + (0.3045892715*scaled_Fe) + (0.1915781796*scaled_Mn) + (0.0801307112*scaled_Mo) + (0.1122627184*scaled_Ni) + (-0.2233522087*scaled_Ti) + (0.05750334635*scaled_V) )
		dense_layer_1_output_15 = self.Tanh( 0.1384024024 + (-0.4060468376*scaled_Al) + (-0.2662906647*scaled_Co) + (0.2115296572*scaled_Cr) + (0.3744380772*scaled_Cu) + (-0.1311639547*scaled_Fe) + (-0.01876971312*scaled_Mn) + (-0.129178077*scaled_Mo) + (-0.07335652411*scaled_Ni) + (0.6511639953*scaled_Ti) + (0.2444981188*scaled_V) )
		approximation_layer_output_0 = self.Identity( -0.2953850925 + (0.0598429814*dense_layer_1_output_0) + (-0.9342574477*dense_layer_1_output_1) + (-0.8191846013*dense_layer_1_output_2) + (0.7028244734*dense_layer_1_output_3) + (0.6028003693*dense_layer_1_output_4) + (-0.6742514968*dense_layer_1_output_5) + (-0.9625467658*dense_layer_1_output_6) + (-0.2801096141*dense_layer_1_output_7) + (0.4474008977*dense_layer_1_output_8) + (-0.7321328521*dense_layer_1_output_9) + (-0.104505077*dense_layer_1_output_10) + (0.4480010867*dense_layer_1_output_11) + (0.5901174545*dense_layer_1_output_12) + (0.8247690797*dense_layer_1_output_13) + (0.5610242486*dense_layer_1_output_14) + (-0.6490028501*dense_layer_1_output_15) )
		unscaling_layer_output_0=approximation_layer_output_0*226.4383545+490.4120789
		HV = max(109, unscaling_layer_output_0)
		HV = min(920.2000122, HV)
		outputs = [HV]
		return outputs

	def calculate_batch_output(self, input_batch):
		output_batch = np.zeros((len(input_batch), 1))
		for i in range(len(input_batch)):
			inputs = list(input_batch[i])
			output = self.calculate_outputs(inputs)
			output_batch[i] = output
		return output_batch

def main():

	# Introduce your input values here
	Al = 0  # Al
	Co = 0  # Co
	Cr = 0  # Cr
	Cu = 0  # Cu
	Fe = 0  # Fe
	Mn = 0  # Mn
	Mo = 0  # Mo
	Ni = 0  # Ni
	Ti = 0  # Ti
	V = 0  # V

	# --- Data conversion (DO NOT modify) ---
	inputs = []

	inputs.append(Al)
	inputs.append(Co)
	inputs.append(Cr)
	inputs.append(Cu)
	inputs.append(Fe)
	inputs.append(Mn)
	inputs.append(Mo)
	inputs.append(Ni)
	inputs.append(Ti)
	inputs.append(V)

	nn = Network()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
