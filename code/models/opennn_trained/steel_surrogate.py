''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Model exported to Python. Use Network().calculate_outputs([...]).

Input Names: 
	0) C
	1) Si
	2) Mn
	3) P
	4) S
	5) Ni
	6) Cr
	7) Mo
	8) Cu
	9) V
	10) Al
	11) N
	12) Nb_Ta
	13) temperature_C

For batch prediction (input must be np.ndarray):
	nn.calculate_batch_output(np.array([[1, 2], [4, 5]]))
''' 
import math
import numpy as np
import pandas as pd

class Network:

	def __init__(self):
		self.inputs_number = 14
		self.input_names = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Al', 'N', 'Nb_Ta', 'temperature_C']

	@staticmethod
	def Identity(x):
		return x

	@staticmethod
	def Tanh(x):
		return np.tanh(x)

	def calculate_outputs(self, inputs):
		C = inputs[0]
		Si = inputs[1]
		Mn = inputs[2]
		P = inputs[3]
		S = inputs[4]
		Ni = inputs[5]
		Cr = inputs[6]
		Mo = inputs[7]
		Cu = inputs[8]
		V = inputs[9]
		Al = inputs[10]
		N = inputs[11]
		Nb_Ta = inputs[12]
		temperature_C = inputs[13]

		scaled_C = C*16.59432411-2.923998356
		scaled_Si = Si*11.87180233-3.721389294
		scaled_Mn = Mn*3.008248091-2.386227608
		scaled_P = P*205.0680695-2.916810751
		scaled_S = S*247.3213501-2.624332666
		scaled_Ni = Ni*5.764391899-0.8084672689
		scaled_Cr = Cr*2.182897091-0.964520812
		scaled_Mo = Mo*2.48534441-1.065022826
		scaled_Cu = Cu*17.43036842-1.397723317
		scaled_V = V*10.81885719-0.6187875271
		scaled_Al = Al*79.1653595-0.9418807626
		scaled_N = N*442.8123169-3.364746094
		scaled_Nb_Ta = Nb_Ta*4504.991211-0.1326666772
		scaled_temperature_C = temperature_C*0.005270169582-1.85312438
		dense_layer_1_output_0 = self.Tanh( 0.5130391121 + (-0.09208875895*scaled_C) + (0.4131321311*scaled_Si) + (0.1520143449*scaled_Mn) + (0.08302076906*scaled_P) + (-0.3469391465*scaled_S) + (-0.6992649436*scaled_Ni) + (-1.468432546*scaled_Cr) + (0.4480624497*scaled_Mo) + (-0.05767414346*scaled_Cu) + (-0.6285356283*scaled_V) + (0.8909363747*scaled_Al) + (-0.0621743463*scaled_N) + (-0.5604695082*scaled_Nb_Ta) + (-0.7170539498*scaled_temperature_C) )
		dense_layer_1_output_1 = self.Tanh( 1.10481894 + (-0.4262143373*scaled_C) + (0.3310537338*scaled_Si) + (0.6922773719*scaled_Mn) + (-0.1435060948*scaled_P) + (-0.05965100601*scaled_S) + (0.7380302548*scaled_Ni) + (0.3651030064*scaled_Cr) + (0.2255795598*scaled_Mo) + (0.003212108742*scaled_Cu) + (0.3908185065*scaled_V) + (0.2802695632*scaled_Al) + (0.04415139183*scaled_N) + (0.07548178732*scaled_Nb_Ta) + (0.5449055433*scaled_temperature_C) )
		dense_layer_1_output_2 = self.Tanh( 2.348524332 + (-0.1489579827*scaled_C) + (-0.04402268678*scaled_Si) + (0.1054534465*scaled_Mn) + (-0.04520896822*scaled_P) + (0.02056040429*scaled_S) + (-0.001939409063*scaled_Ni) + (-0.05797601864*scaled_Cr) + (0.4198080599*scaled_Mo) + (-0.03787151352*scaled_Cu) + (0.2633003592*scaled_V) + (0.1335187554*scaled_Al) + (0.001240798854*scaled_N) + (-0.04680431262*scaled_Nb_Ta) + (-1.690889597*scaled_temperature_C) )
		dense_layer_1_output_3 = self.Tanh( -0.851498425 + (-0.5758953691*scaled_C) + (-0.6070384979*scaled_Si) + (-0.3827618957*scaled_Mn) + (-0.1212551147*scaled_P) + (0.293084532*scaled_S) + (-0.9739306569*scaled_Ni) + (-0.1359946281*scaled_Cr) + (-0.1398642212*scaled_Mo) + (0.07112583518*scaled_Cu) + (-0.481688261*scaled_V) + (-1.022218823*scaled_Al) + (-0.1863705218*scaled_N) + (-0.04285567254*scaled_Nb_Ta) + (0.2578317523*scaled_temperature_C) )
		dense_layer_1_output_4 = self.Tanh( -0.3865213692 + (-0.007319081109*scaled_C) + (-0.1511392593*scaled_Si) + (-0.4786857963*scaled_Mn) + (-0.1039743572*scaled_P) + (-0.03255828843*scaled_S) + (-0.2947009504*scaled_Ni) + (-0.1746439636*scaled_Cr) + (-0.3654738963*scaled_Mo) + (-0.1660837233*scaled_Cu) + (-0.6487380266*scaled_V) + (-0.2657811642*scaled_Al) + (-0.09940835088*scaled_N) + (0.2721574306*scaled_Nb_Ta) + (1.245955825*scaled_temperature_C) )
		dense_layer_1_output_5 = self.Tanh( 0.07714353502 + (0.702279985*scaled_C) + (-0.3317526877*scaled_Si) + (-1.428197265*scaled_Mn) + (-0.1316980273*scaled_P) + (-0.1360477507*scaled_S) + (-0.4423360229*scaled_Ni) + (0.8327448964*scaled_Cr) + (-1.312199354*scaled_Mo) + (-0.03740774468*scaled_Cu) + (-0.1494701505*scaled_V) + (0.4371091127*scaled_Al) + (0.1569782495*scaled_N) + (0.3796010911*scaled_Nb_Ta) + (-0.3754453361*scaled_temperature_C) )
		dense_layer_1_output_6 = self.Tanh( 0.1837951988 + (0.4518071711*scaled_C) + (0.04334201291*scaled_Si) + (-0.2693068683*scaled_Mn) + (0.0233292263*scaled_P) + (-0.04084209353*scaled_S) + (0.03679736704*scaled_Ni) + (-0.2916628122*scaled_Cr) + (-0.3893809319*scaled_Mo) + (-0.1421336383*scaled_Cu) + (0.03865187243*scaled_V) + (0.0499897413*scaled_Al) + (-0.05721538886*scaled_N) + (-0.08215717971*scaled_Nb_Ta) + (-0.4525436461*scaled_temperature_C) )
		dense_layer_1_output_7 = self.Tanh( 0.3844534755 + (-0.1741505712*scaled_C) + (0.03714190796*scaled_Si) + (0.2240240425*scaled_Mn) + (0.08417666703*scaled_P) + (0.05257050321*scaled_S) + (0.05051473528*scaled_Ni) + (-0.1423578858*scaled_Cr) + (0.3758843839*scaled_Mo) + (0.1506692171*scaled_Cu) + (0.4404783845*scaled_V) + (0.1606497169*scaled_Al) + (0.1588264406*scaled_N) + (-0.7033883929*scaled_Nb_Ta) + (-1.920283437*scaled_temperature_C) )
		approximation_layer_output_0 = self.Identity( -0.7192615867 + (-0.6995344758*dense_layer_1_output_0) + (0.8293580413*dense_layer_1_output_1) + (0.5450696349*dense_layer_1_output_2) + (0.269823432*dense_layer_1_output_3) + (-0.1830903739*dense_layer_1_output_4) + (-0.4945137501*dense_layer_1_output_5) + (1.394534349*dense_layer_1_output_6) + (0.2588124871*dense_layer_1_output_7) )
		approximation_layer_output_1 = self.Identity( -1.435056925 + (-0.7973309755*dense_layer_1_output_0) + (0.6531379819*dense_layer_1_output_1) + (1.602581143*dense_layer_1_output_2) + (-0.06332557648*dense_layer_1_output_3) + (0.6988498569*dense_layer_1_output_4) + (-0.3088439107*dense_layer_1_output_5) + (1.320355415*dense_layer_1_output_6) + (0.6858456731*dense_layer_1_output_7) )
		approximation_layer_output_2 = self.Identity( 1.468973875 + (0.5466590524*dense_layer_1_output_0) + (0.01978817955*dense_layer_1_output_1) + (-2.025735617*dense_layer_1_output_2) + (0.1923063397*dense_layer_1_output_3) + (-0.5462281704*dense_layer_1_output_4) + (0.4352910817*dense_layer_1_output_5) + (-0.09918309003*dense_layer_1_output_6) + (-0.5139402151*dense_layer_1_output_7) )
		approximation_layer_output_3 = self.Identity( 0.08113814145 + (0.678840518*dense_layer_1_output_0) + (1.263955116*dense_layer_1_output_1) + (-0.9290831685*dense_layer_1_output_2) + (0.8554539084*dense_layer_1_output_3) + (-1.925898314*dense_layer_1_output_4) + (0.693700552*dense_layer_1_output_5) + (0.2885358334*dense_layer_1_output_6) + (-1.831340671*dense_layer_1_output_7) )
		unscaling_layer_output_0=approximation_layer_output_0*135.5055847+325.2110291
		unscaling_layer_output_1=approximation_layer_output_1*127.563858+488.9700928
		unscaling_layer_output_2=approximation_layer_output_2*8.893019676+26.80315018
		unscaling_layer_output_3=approximation_layer_output_3*11.69675541+70.35905457
		proof_stress_MPa = max(27, unscaling_layer_output_0)
		proof_stress_MPa = min(690, proof_stress_MPa)
		tensile_strength_MPa = max(162, unscaling_layer_output_1)
		tensile_strength_MPa = min(830, tensile_strength_MPa)
		elongation_pct = max(10, unscaling_layer_output_2)
		elongation_pct = min(78, elongation_pct)
		reduction_area_pct = max(18, unscaling_layer_output_3)
		reduction_area_pct = min(94, reduction_area_pct)
		outputs = [proof_stress_MPa, tensile_strength_MPa, elongation_pct, reduction_area_pct]
		return outputs

	def calculate_batch_output(self, input_batch):
		output_batch = np.zeros((len(input_batch), 4))
		for i in range(len(input_batch)):
			inputs = list(input_batch[i])
			output = self.calculate_outputs(inputs)
			output_batch[i] = output
		return output_batch

def main():

	# Introduce your input values here
	C = 0  # C
	Si = 0  # Si
	Mn = 0  # Mn
	P = 0  # P
	S = 0  # S
	Ni = 0  # Ni
	Cr = 0  # Cr
	Mo = 0  # Mo
	Cu = 0  # Cu
	V = 0  # V
	Al = 0  # Al
	N = 0  # N
	Nb_Ta = 0  # Nb_Ta
	temperature_C = 0  # temperature_C

	# --- Data conversion (DO NOT modify) ---
	inputs = []

	inputs.append(C)
	inputs.append(Si)
	inputs.append(Mn)
	inputs.append(P)
	inputs.append(S)
	inputs.append(Ni)
	inputs.append(Cr)
	inputs.append(Mo)
	inputs.append(Cu)
	inputs.append(V)
	inputs.append(Al)
	inputs.append(N)
	inputs.append(Nb_Ta)
	inputs.append(temperature_C)

	nn = Network()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
