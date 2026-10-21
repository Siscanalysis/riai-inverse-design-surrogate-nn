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
		dense_layer_1_output_0 = self.Tanh( -1.161480784 + (0.3740318418*scaled_C) + (-0.2557557523*scaled_Si) + (-0.5726395249*scaled_Mn) + (0.2088427395*scaled_P) + (-0.1175428703*scaled_S) + (-0.6622128487*scaled_Ni) + (-0.32267946*scaled_Cr) + (-0.3022291064*scaled_Mo) + (0.06434592605*scaled_Cu) + (-0.2637598217*scaled_V) + (-0.1153189763*scaled_Al) + (-0.08274962008*scaled_N) + (-0.1765606552*scaled_Nb_Ta) + (-0.5314637423*scaled_temperature_C) )
		dense_layer_1_output_1 = self.Tanh( -2.034921408 + (0.1226990893*scaled_C) + (-0.08717302978*scaled_Si) + (0.06335435063*scaled_Mn) + (0.08653251827*scaled_P) + (-0.08064337075*scaled_S) + (-0.215036869*scaled_Ni) + (-0.1977530718*scaled_Cr) + (-0.1911408752*scaled_Mo) + (-0.05559284613*scaled_Cu) + (-0.08448185027*scaled_V) + (-0.1069873273*scaled_Al) + (0.002141205128*scaled_N) + (-0.07971496135*scaled_Nb_Ta) + (1.738316774*scaled_temperature_C) )
		dense_layer_1_output_2 = self.Tanh( -0.05591130257 + (0.456881851*scaled_C) + (-0.01376823708*scaled_Si) + (-0.330250442*scaled_Mn) + (0.1643577665*scaled_P) + (-0.01563246734*scaled_S) + (0.2181115746*scaled_Ni) + (-0.3399033844*scaled_Cr) + (-0.5740648508*scaled_Mo) + (-0.0611651279*scaled_Cu) + (0.2945836782*scaled_V) + (0.09194043279*scaled_Al) + (-0.0673282519*scaled_N) + (-0.1829673201*scaled_Nb_Ta) + (-0.5217530727*scaled_temperature_C) )
		dense_layer_1_output_3 = self.Tanh( -0.8180921674 + (-0.9870576859*scaled_C) + (-0.2523259223*scaled_Si) + (-0.2209137976*scaled_Mn) + (0.006626455113*scaled_P) + (-0.3332146108*scaled_S) + (0.4778753817*scaled_Ni) + (0.3801203966*scaled_Cr) + (0.5322515368*scaled_Mo) + (0.3102928698*scaled_Cu) + (-0.8833789825*scaled_V) + (1.064866185*scaled_Al) + (0.8123215437*scaled_N) + (0.3791162372*scaled_Nb_Ta) + (0.08344525099*scaled_temperature_C) )
		dense_layer_1_output_4 = self.Tanh( 0.4127160013 + (0.06869277358*scaled_C) + (0.2382621169*scaled_Si) + (0.6120679379*scaled_Mn) + (-0.001003703568*scaled_P) + (0.09234984964*scaled_S) + (0.4462459683*scaled_Ni) + (0.4667675197*scaled_Cr) + (0.1594310999*scaled_Mo) + (0.2056678683*scaled_Cu) + (0.7758242488*scaled_V) + (0.222914651*scaled_Al) + (0.08326587826*scaled_N) + (-0.1819650829*scaled_Nb_Ta) + (-1.657134891*scaled_temperature_C) )
		dense_layer_1_output_5 = self.Tanh( -0.2675434947 + (0.1491933316*scaled_C) + (-0.1078658625*scaled_Si) + (-0.1725456715*scaled_Mn) + (-0.09744393826*scaled_P) + (-0.06115720049*scaled_S) + (-0.2022479624*scaled_Ni) + (0.04965783656*scaled_Cr) + (-0.2473443151*scaled_Mo) + (-0.1349639297*scaled_Cu) + (-0.3510020673*scaled_V) + (-0.06263197958*scaled_Al) + (-0.1151699945*scaled_N) + (0.0731338039*scaled_Nb_Ta) + (1.850916862*scaled_temperature_C) )
		dense_layer_1_output_6 = self.Tanh( -0.01574512757 + (-0.5198339224*scaled_C) + (0.3716980815*scaled_Si) + (0.03232409433*scaled_Mn) + (0.2270718068*scaled_P) + (-0.2439461648*scaled_S) + (-0.6145108938*scaled_Ni) + (0.02161668055*scaled_Cr) + (-0.04434253275*scaled_Mo) + (0.1890614331*scaled_Cu) + (-0.5328919291*scaled_V) + (-0.3370089233*scaled_Al) + (-0.161643371*scaled_N) + (0.09323316813*scaled_Nb_Ta) + (-0.2302142233*scaled_temperature_C) )
		dense_layer_1_output_7 = self.Tanh( -0.01013606228 + (0.06665205956*scaled_C) + (0.01381828636*scaled_Si) + (-0.02686700411*scaled_Mn) + (-0.2069128454*scaled_P) + (0.03866370395*scaled_S) + (0.06462214887*scaled_Ni) + (-0.1330147088*scaled_Cr) + (0.0697927624*scaled_Mo) + (-0.1686196923*scaled_Cu) + (0.05039419234*scaled_V) + (-0.08431230485*scaled_Al) + (0.03857034817*scaled_N) + (-0.1406414807*scaled_Nb_Ta) + (-0.03499327973*scaled_temperature_C) )
		dense_layer_1_output_8 = self.Tanh( -0.03962660208 + (0.1172500029*scaled_C) + (-0.1103431508*scaled_Si) + (-0.03301499411*scaled_Mn) + (0.1148785949*scaled_P) + (-0.1947112083*scaled_S) + (-0.03053651191*scaled_Ni) + (0.139259249*scaled_Cr) + (0.03173192963*scaled_Mo) + (-0.2126434445*scaled_Cu) + (0.09306742251*scaled_V) + (-0.02419470064*scaled_Al) + (0.1362891048*scaled_N) + (0.1353123635*scaled_Nb_Ta) + (-0.1010863036*scaled_temperature_C) )
		dense_layer_1_output_9 = self.Tanh( 0.02793316729 + (0.231079638*scaled_C) + (-0.1143582985*scaled_Si) + (-0.09180625528*scaled_Mn) + (-0.1373263448*scaled_P) + (0.02277498133*scaled_S) + (-0.09128941596*scaled_Ni) + (0.09414017946*scaled_Cr) + (0.0002964481246*scaled_Mo) + (-0.1656939089*scaled_Cu) + (0.02857892029*scaled_V) + (-0.09758032113*scaled_Al) + (-0.2021335363*scaled_N) + (0.1064664498*scaled_Nb_Ta) + (-0.1444587857*scaled_temperature_C) )
		dense_layer_1_output_10 = self.Tanh( 0.05296573788 + (0.06112219766*scaled_C) + (-0.1057036072*scaled_Si) + (-0.03097977862*scaled_Mn) + (-0.1450002939*scaled_P) + (0.06697399169*scaled_S) + (0.04366975278*scaled_Ni) + (-0.03491616994*scaled_Cr) + (-0.0388694331*scaled_Mo) + (-0.05673503131*scaled_Cu) + (-0.0950634405*scaled_V) + (-0.09554789215*scaled_Al) + (-0.1779413521*scaled_N) + (-0.07735604793*scaled_Nb_Ta) + (-0.0466665104*scaled_temperature_C) )
		dense_layer_1_output_11 = self.Tanh( -0.0109266052 + (-0.0684280619*scaled_C) + (0.06861609221*scaled_Si) + (-0.07132488489*scaled_Mn) + (-0.1032130197*scaled_P) + (0.07441786677*scaled_S) + (-0.04745917395*scaled_Ni) + (-0.1346525103*scaled_Cr) + (0.1176491231*scaled_Mo) + (0.1483105272*scaled_Cu) + (0.06330654025*scaled_V) + (-0.1558110118*scaled_Al) + (-0.01876311935*scaled_N) + (-0.1069170982*scaled_Nb_Ta) + (-0.03151833266*scaled_temperature_C) )
		dense_layer_1_output_12 = self.Tanh( -0.007744711824 + (0.09638801962*scaled_C) + (-0.0366496034*scaled_Si) + (-0.1265713722*scaled_Mn) + (0.03179970011*scaled_P) + (-0.000554571161*scaled_S) + (0.0812036097*scaled_Ni) + (0.02696220763*scaled_Cr) + (-0.02500193007*scaled_Mo) + (-0.1805461645*scaled_Cu) + (-0.1240977421*scaled_V) + (0.08056422323*scaled_Al) + (-0.07159414142*scaled_N) + (-0.09662490338*scaled_Nb_Ta) + (-0.1186454594*scaled_temperature_C) )
		dense_layer_1_output_13 = self.Tanh( -0.006080493797 + (0.07616066933*scaled_C) + (-0.05286377668*scaled_Si) + (-0.00645689154*scaled_Mn) + (0.1003187075*scaled_P) + (-0.1287950873*scaled_S) + (0.07490437478*scaled_Ni) + (0.1452593505*scaled_Cr) + (-0.07701642811*scaled_Mo) + (-0.05954482406*scaled_Cu) + (-0.03570563719*scaled_V) + (0.07503866404*scaled_Al) + (-0.07756552845*scaled_N) + (0.07741576433*scaled_Nb_Ta) + (-0.06153474003*scaled_temperature_C) )
		dense_layer_1_output_14 = self.Tanh( -0.04852996022 + (0.04174369574*scaled_C) + (0.006457835436*scaled_Si) + (0.1154824048*scaled_Mn) + (0.2516764998*scaled_P) + (-0.1905648559*scaled_S) + (0.01647541486*scaled_Ni) + (0.03037625179*scaled_Cr) + (-0.2869175673*scaled_Mo) + (0.01214546338*scaled_Cu) + (-0.1693342924*scaled_V) + (0.2326917499*scaled_Al) + (-0.08334563673*scaled_N) + (-0.100801684*scaled_Nb_Ta) + (-0.2606469095*scaled_temperature_C) )
		dense_layer_1_output_15 = self.Tanh( -0.001812488539 + (-0.008255399764*scaled_C) + (0.04938488081*scaled_Si) + (0.07288919389*scaled_Mn) + (0.0653456077*scaled_P) + (-0.08837308735*scaled_S) + (-0.04972367361*scaled_Ni) + (-0.08775549382*scaled_Cr) + (-0.02577110194*scaled_Mo) + (0.04611988738*scaled_Cu) + (0.05475749075*scaled_V) + (0.03528321907*scaled_Al) + (0.03600575402*scaled_N) + (0.005764121655*scaled_Nb_Ta) + (0.08419967443*scaled_temperature_C) )
		approximation_layer_output_0 = self.Identity( -1.021904469 + (-1.316334844*dense_layer_1_output_0) + (-0.4859859347*dense_layer_1_output_1) + (0.974963665*dense_layer_1_output_2) + (-0.3052110374*dense_layer_1_output_3) + (0.2131916136*dense_layer_1_output_4) + (-0.1991094202*dense_layer_1_output_5) + (0.1782791317*dense_layer_1_output_6) + (0.0809527263*dense_layer_1_output_7) + (0.1030784398*dense_layer_1_output_8) + (0.0142163299*dense_layer_1_output_9) + (-0.02107269876*dense_layer_1_output_10) + (-0.02937435172*dense_layer_1_output_11) + (0.06560982764*dense_layer_1_output_12) + (-0.09040181339*dense_layer_1_output_13) + (-0.3890604079*dense_layer_1_output_14) + (0.050644923*dense_layer_1_output_15) )
		approximation_layer_output_1 = self.Identity( -1.470160365 + (-1.15292275*dense_layer_1_output_0) + (-1.349272966*dense_layer_1_output_1) + (0.6737723351*dense_layer_1_output_2) + (-0.3934406638*dense_layer_1_output_3) + (-0.5443322062*dense_layer_1_output_4) + (-0.5774301887*dense_layer_1_output_5) + (-0.05647490919*dense_layer_1_output_6) + (-0.06626135856*dense_layer_1_output_7) + (0.2967788577*dense_layer_1_output_8) + (0.1766030788*dense_layer_1_output_9) + (-0.08909342438*dense_layer_1_output_10) + (0.004068193957*dense_layer_1_output_11) + (-0.01488373615*dense_layer_1_output_12) + (-0.02182267793*dense_layer_1_output_13) + (-0.090520069*dense_layer_1_output_14) + (0.09155433625*dense_layer_1_output_15) )
		approximation_layer_output_2 = self.Identity( 1.396640658 + (0.4607876837*dense_layer_1_output_0) + (1.547886372*dense_layer_1_output_1) + (0.7933767438*dense_layer_1_output_2) + (0.7182337046*dense_layer_1_output_3) + (0.4007571638*dense_layer_1_output_4) + (0.5950031281*dense_layer_1_output_5) + (0.4436528385*dense_layer_1_output_6) + (0.03145841509*dense_layer_1_output_7) + (-0.07045896351*dense_layer_1_output_8) + (0.1659406722*dense_layer_1_output_9) + (0.1300492138*dense_layer_1_output_10) + (0.09818574786*dense_layer_1_output_11) + (-0.2014873326*dense_layer_1_output_12) + (-0.1084876657*dense_layer_1_output_13) + (-0.3396234512*dense_layer_1_output_14) + (0.01638429798*dense_layer_1_output_15) )
		approximation_layer_output_3 = self.Identity( 0.5223364234 + (-0.450568825*dense_layer_1_output_0) + (1.01577127*dense_layer_1_output_1) + (0.8541978598*dense_layer_1_output_2) + (0.7550722957*dense_layer_1_output_3) + (1.410441518*dense_layer_1_output_4) + (1.797981262*dense_layer_1_output_5) + (0.840151608*dense_layer_1_output_6) + (0.2357468605*dense_layer_1_output_7) + (0.0861755237*dense_layer_1_output_8) + (0.3833936155*dense_layer_1_output_9) + (0.2519363761*dense_layer_1_output_10) + (0.2216024995*dense_layer_1_output_11) + (0.1646610647*dense_layer_1_output_12) + (-0.1431590915*dense_layer_1_output_13) + (-0.04970697314*dense_layer_1_output_14) + (-0.09089981765*dense_layer_1_output_15) )
		unscaling_layer_output_0=approximation_layer_output_0*135.5055847+325.2110291
		unscaling_layer_output_1=approximation_layer_output_1*127.563858+488.9700928
		unscaling_layer_output_2=approximation_layer_output_2*8.893019676+26.80315018
		unscaling_layer_output_3=approximation_layer_output_3*11.69675541+70.35905457
		proof_stress_MPa = unscaling_layer_output_0
		tensile_strength_MPa = unscaling_layer_output_1
		elongation_pct = unscaling_layer_output_2
		reduction_area_pct = unscaling_layer_output_3
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
