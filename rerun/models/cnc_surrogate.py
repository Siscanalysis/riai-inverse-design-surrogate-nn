''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Model exported to Python. Use NeuralNetwork().calculate_outputs([...]).

Input Names: 
	0) cutting_speed
	1) feed
	2) depth_of_cut
	3) tool_wear_VB

For batch prediction (input must be np.ndarray):
	nn.calculate_batch_output(np.array([[1, 2], [4, 5]]))
''' 
import numpy as np
import pandas as pd

class NeuralNetwork:

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

		scaled_cutting_speed = (cutting_speed-350)/23.78300095
		scaled_feed = (feed-0.1000000015)/0.02353389934
		scaled_depth_of_cut = (depth_of_cut-0.4499999881)/0.1979440004
		scaled_tool_wear_VB = (tool_wear_VB-0.06274510175)/0.1084830016
		dense2d_layer_1_output_0 = self.Tanh( 0.3033355176 + (-0.0173370894*scaled_cutting_speed) + (0.01149973553*scaled_feed) + (0.9297085404*scaled_depth_of_cut) + (-0.1473789066*scaled_tool_wear_VB) )
		dense2d_layer_1_output_1 = self.Tanh( 0.7648762465 + (-0.1661525965*scaled_cutting_speed) + (-0.5675094724*scaled_feed) + (0.1502665579*scaled_depth_of_cut) + (-0.01048111822*scaled_tool_wear_VB) )
		dense2d_layer_1_output_2 = self.Tanh( 0.04441745207 + (0.03758119047*scaled_cutting_speed) + (0.1566088349*scaled_feed) + (0.04430253804*scaled_depth_of_cut) + (0.2030033618*scaled_tool_wear_VB) )
		dense2d_layer_1_output_3 = self.Tanh( 0.1157783493 + (0.08861973882*scaled_cutting_speed) + (0.2005952895*scaled_feed) + (0.01944502257*scaled_depth_of_cut) + (0.1017409191*scaled_tool_wear_VB) )
		dense2d_layer_1_output_4 = self.Tanh( -0.2371274382 + (0.08616152406*scaled_cutting_speed) + (0.313914448*scaled_feed) + (-0.3337987363*scaled_depth_of_cut) + (-0.08031181991*scaled_tool_wear_VB) )
		dense2d_layer_1_output_5 = self.Tanh( -0.7768033743 + (0.01526417583*scaled_cutting_speed) + (0.478061974*scaled_feed) + (0.6555826664*scaled_depth_of_cut) + (0.2456617057*scaled_tool_wear_VB) )
		dense2d_layer_1_output_6 = self.Tanh( -0.1329317391 + (0.1744114906*scaled_cutting_speed) + (-0.5367841721*scaled_feed) + (0.7561575174*scaled_depth_of_cut) + (-0.125806421*scaled_tool_wear_VB) )
		dense2d_layer_1_output_7 = self.Tanh( 0.5921817422 + (0.06325713545*scaled_cutting_speed) + (0.02537619323*scaled_feed) + (-0.4888656437*scaled_depth_of_cut) + (-0.5232517123*scaled_tool_wear_VB) )
		dense2d_layer_1_output_8 = self.Tanh( -0.04980302975 + (-0.009526878595*scaled_cutting_speed) + (0.2019443661*scaled_feed) + (0.08724156767*scaled_depth_of_cut) + (-0.04238595068*scaled_tool_wear_VB) )
		dense2d_layer_1_output_9 = self.Tanh( 0.1974852979 + (-0.0223314669*scaled_cutting_speed) + (-0.3111961782*scaled_feed) + (-0.0626007989*scaled_depth_of_cut) + (0.2260141522*scaled_tool_wear_VB) )
		dense2d_layer_1_output_10 = self.Tanh( -0.3492477834 + (0.01835773699*scaled_cutting_speed) + (0.3956112266*scaled_feed) + (0.4998908043*scaled_depth_of_cut) + (0.006312849931*scaled_tool_wear_VB) )
		dense2d_layer_1_output_11 = self.Tanh( 0.08871808648 + (0.02666033991*scaled_cutting_speed) + (-0.3455956876*scaled_feed) + (0.06124634296*scaled_depth_of_cut) + (0.106659621*scaled_tool_wear_VB) )
		dense2d_layer_1_output_12 = self.Tanh( 0.8165928721 + (-0.4708617032*scaled_cutting_speed) + (-0.3785893321*scaled_feed) + (0.3209649324*scaled_depth_of_cut) + (-1.104278088*scaled_tool_wear_VB) )
		dense2d_layer_1_output_13 = self.Tanh( -0.8090728521 + (0.4932955205*scaled_cutting_speed) + (0.7612145543*scaled_feed) + (-0.0008486547158*scaled_depth_of_cut) + (-0.0928523168*scaled_tool_wear_VB) )
		dense2d_layer_1_output_14 = self.Tanh( 0.2145554274 + (-0.1253493875*scaled_cutting_speed) + (-0.1966998577*scaled_feed) + (-0.08560871333*scaled_depth_of_cut) + (0.3727247119*scaled_tool_wear_VB) )
		dense2d_layer_1_output_15 = self.Tanh( 1.12628293 + (0.7084359527*scaled_cutting_speed) + (-0.7113797665*scaled_feed) + (-0.2786874473*scaled_depth_of_cut) + (1.208105803*scaled_tool_wear_VB) )
		dense2d_layer_1_output_16 = self.Tanh( -0.5397161841 + (0.5237358809*scaled_cutting_speed) + (0.1860455573*scaled_feed) + (0.9606871605*scaled_depth_of_cut) + (0.0336888507*scaled_tool_wear_VB) )
		dense2d_layer_1_output_17 = self.Tanh( -0.3583438694 + (0.1764932871*scaled_cutting_speed) + (-0.3042505085*scaled_feed) + (0.09191834182*scaled_depth_of_cut) + (-0.118097879*scaled_tool_wear_VB) )
		dense2d_layer_1_output_18 = self.Tanh( 0.2061236054 + (-0.08051465452*scaled_cutting_speed) + (-0.2420433909*scaled_feed) + (0.3560473323*scaled_depth_of_cut) + (0.04980149865*scaled_tool_wear_VB) )
		dense2d_layer_1_output_19 = self.Tanh( -0.5028137565 + (-0.1632918417*scaled_cutting_speed) + (-0.3794927895*scaled_feed) + (0.5480651259*scaled_depth_of_cut) + (1.201931477*scaled_tool_wear_VB) )
		dense2d_layer_1_output_20 = self.Tanh( 0.2679788768 + (-0.6341581941*scaled_cutting_speed) + (-0.2587671578*scaled_feed) + (0.4112055898*scaled_depth_of_cut) + (0.2812964618*scaled_tool_wear_VB) )
		dense2d_layer_1_output_21 = self.Tanh( 0.5181258321 + (-0.1011962518*scaled_cutting_speed) + (0.1393183023*scaled_feed) + (0.5934101343*scaled_depth_of_cut) + (-0.5740301013*scaled_tool_wear_VB) )
		dense2d_layer_1_output_22 = self.Tanh( 0.5973853469 + (0.5839691162*scaled_cutting_speed) + (0.04347306862*scaled_feed) + (0.1699102819*scaled_depth_of_cut) + (0.04630977288*scaled_tool_wear_VB) )
		dense2d_layer_1_output_23 = self.Tanh( -0.1871795952 + (0.1217667982*scaled_cutting_speed) + (0.1480626613*scaled_feed) + (-0.3371584117*scaled_depth_of_cut) + (-0.07368452102*scaled_tool_wear_VB) )
		dense2d_layer_1_output_24 = self.Tanh( -0.2519772351 + (0.1005204394*scaled_cutting_speed) + (0.2502130568*scaled_feed) + (0.04134719074*scaled_depth_of_cut) + (-0.3711030483*scaled_tool_wear_VB) )
		dense2d_layer_1_output_25 = self.Tanh( -0.2654901445 + (-0.1828550547*scaled_cutting_speed) + (-0.2240920216*scaled_feed) + (-0.04449424893*scaled_depth_of_cut) + (-0.1001420394*scaled_tool_wear_VB) )
		dense2d_layer_1_output_26 = self.Tanh( -0.04231489822 + (-0.1030542329*scaled_cutting_speed) + (-0.1822793335*scaled_feed) + (-0.008906045929*scaled_depth_of_cut) + (0.1291626841*scaled_tool_wear_VB) )
		dense2d_layer_1_output_27 = self.Tanh( -0.6315580606 + (0.6971297264*scaled_cutting_speed) + (0.6133471727*scaled_feed) + (-0.8959864378*scaled_depth_of_cut) + (-0.1138594523*scaled_tool_wear_VB) )
		dense2d_layer_1_output_28 = self.Tanh( 0.07468057424 + (-0.4687700868*scaled_cutting_speed) + (-0.06904111058*scaled_feed) + (-0.1090326458*scaled_depth_of_cut) + (0.2202237695*scaled_tool_wear_VB) )
		dense2d_layer_1_output_29 = self.Tanh( -0.02986673266 + (-0.7505504489*scaled_cutting_speed) + (-0.5414429307*scaled_feed) + (-1.011263013*scaled_depth_of_cut) + (0.4156668186*scaled_tool_wear_VB) )
		dense2d_layer_1_output_30 = self.Tanh( -0.1569084227 + (-0.07280413061*scaled_cutting_speed) + (-0.2108327299*scaled_feed) + (-0.01604205742*scaled_depth_of_cut) + (-0.1025677174*scaled_tool_wear_VB) )
		dense2d_layer_1_output_31 = self.Tanh( 0.4316868484 + (-0.3793715537*scaled_cutting_speed) + (-0.02084636688*scaled_feed) + (-0.8863213062*scaled_depth_of_cut) + (0.2829802334*scaled_tool_wear_VB) )
		approximation_layer_output_0 = self.Identity( 0.2589830458 + (-0.6857063174*dense2d_layer_1_output_0) + (-0.5550584197*dense2d_layer_1_output_1) + (0.08223634213*dense2d_layer_1_output_2) + (0.1547351182*dense2d_layer_1_output_3) + (0.3891340792*dense2d_layer_1_output_4) + (-0.06239243597*dense2d_layer_1_output_5) + (0.3013315797*dense2d_layer_1_output_6) + (-0.4664597213*dense2d_layer_1_output_7) + (-0.01494812965*dense2d_layer_1_output_8) + (-0.1742371619*dense2d_layer_1_output_9) + (-0.3876539767*dense2d_layer_1_output_10) + (-0.353201896*dense2d_layer_1_output_11) + (1.035614371*dense2d_layer_1_output_12) + (0.7821089625*dense2d_layer_1_output_13) + (-0.1500612795*dense2d_layer_1_output_14) + (-0.9712728858*dense2d_layer_1_output_15) + (-0.7620967627*dense2d_layer_1_output_16) + (-0.1238158345*dense2d_layer_1_output_17) + (-0.2005101591*dense2d_layer_1_output_18) + (0.9354122877*dense2d_layer_1_output_19) + (0.2582707703*dense2d_layer_1_output_20) + (-0.3637490869*dense2d_layer_1_output_21) + (0.8148829937*dense2d_layer_1_output_22) + (0.04591169208*dense2d_layer_1_output_23) + (0.2572370768*dense2d_layer_1_output_24) + (-0.4698248804*dense2d_layer_1_output_25) + (-0.1966513842*dense2d_layer_1_output_26) + (-0.8973715305*dense2d_layer_1_output_27) + (-0.2951875031*dense2d_layer_1_output_28) + (-0.8879622817*dense2d_layer_1_output_29) + (-0.2696020305*dense2d_layer_1_output_30) + (0.6246395707*dense2d_layer_1_output_31) )
		approximation_layer_output_1 = self.Identity( 0.2384479493 + (0.09850319475*dense2d_layer_1_output_0) + (0.6870729327*dense2d_layer_1_output_1) + (0.194917053*dense2d_layer_1_output_2) + (0.1957682967*dense2d_layer_1_output_3) + (-0.3025941551*dense2d_layer_1_output_4) + (0.5817201138*dense2d_layer_1_output_5) + (0.3068341911*dense2d_layer_1_output_6) + (-0.3955861032*dense2d_layer_1_output_7) + (0.1890077591*dense2d_layer_1_output_8) + (-0.3531556726*dense2d_layer_1_output_9) + (0.06816719472*dense2d_layer_1_output_10) + (-0.181242153*dense2d_layer_1_output_11) + (-0.5541375875*dense2d_layer_1_output_12) + (0.08070086688*dense2d_layer_1_output_13) + (-0.2482068837*dense2d_layer_1_output_14) + (-0.1125778705*dense2d_layer_1_output_15) + (0.09923790395*dense2d_layer_1_output_16) + (-0.4510307312*dense2d_layer_1_output_17) + (0.3373349607*dense2d_layer_1_output_18) + (-0.09941076487*dense2d_layer_1_output_19) + (-0.3243768215*dense2d_layer_1_output_20) + (-0.4339235723*dense2d_layer_1_output_21) + (0.1965159476*dense2d_layer_1_output_22) + (-0.4311050177*dense2d_layer_1_output_23) + (0.2604085207*dense2d_layer_1_output_24) + (-0.201776877*dense2d_layer_1_output_25) + (-0.2482727021*dense2d_layer_1_output_26) + (-0.2005489171*dense2d_layer_1_output_27) + (0.440536499*dense2d_layer_1_output_28) + (0.2024220973*dense2d_layer_1_output_29) + (-0.2480544001*dense2d_layer_1_output_30) + (-0.128080368*dense2d_layer_1_output_31) )
		unscaling_layer_output_0=approximation_layer_output_0*0.3043832779+0.6826797128
		unscaling_layer_output_1=approximation_layer_output_1*92.5605011+193.3598785
		Ra = unscaling_layer_output_0
		F = unscaling_layer_output_1
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

	nn = NeuralNetwork()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
