''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Model exported to Python. Use Network().calculate_outputs([...]).

Input Names: 
	0) TBX
	1) TBY
	2) LOCAL
	3) WPTX
	4) WPTY
	5) VECTOR
	6) UNROLL_FACTOR
	7) PADDING

For batch prediction (input must be np.ndarray):
	nn.calculate_batch_output(np.array([[1, 2], [4, 5]]))
''' 
import math
import numpy as np
import pandas as pd

class Network:

	def __init__(self):
		self.inputs_number = 8
		self.input_names = ['TBX', 'TBY', 'LOCAL', 'WPTX', 'WPTY', 'VECTOR', 'UNROLL_FACTOR', 'PADDING']

	@staticmethod
	def Identity(x):
		return x

	@staticmethod
	def Tanh(x):
		return np.tanh(x)

	def calculate_outputs(self, inputs):
		TBX = inputs[0]
		TBY = inputs[1]
		LOCAL = inputs[2]
		WPTX = inputs[3]
		WPTY = inputs[4]
		VECTOR = inputs[5]
		UNROLL_FACTOR = inputs[6]
		PADDING = inputs[7]

		scaled_TBX = TBX*0.05753935874-1.245709777
		scaled_TBY = TBY*0.05681433901-1.252562642
		scaled_LOCAL = LOCAL*1.347999692-1.403245568
		scaled_WPTX = WPTX*0.3928792775-1.601008415
		scaled_WPTY = WPTY*0.3989917636-1.328652859
		scaled_VECTOR = VECTOR*0.9784658551-1.758959055
		scaled_UNROLL_FACTOR = UNROLL_FACTOR*0.333285749-1.321065307
		scaled_PADDING = PADDING*2.0-1.0
		dense_layer_1_output_0 = self.Tanh( -1.388568282 + (-0.07528177649*scaled_TBX) + (0.02090460621*scaled_TBY) + (0.02695137821*scaled_LOCAL) + (-1.299762607*scaled_WPTX) + (-0.1340833157*scaled_WPTY) + (-0.0376460962*scaled_VECTOR) + (0.06432881951*scaled_UNROLL_FACTOR) + (-0.02873419039*scaled_PADDING) )
		dense_layer_1_output_1 = self.Tanh( 0.4470171034 + (0.5219060183*scaled_TBX) + (-0.3132762015*scaled_TBY) + (-0.2270425409*scaled_LOCAL) + (-0.3138727844*scaled_WPTX) + (0.1010498703*scaled_WPTY) + (-0.07928972691*scaled_VECTOR) + (-0.001549645793*scaled_UNROLL_FACTOR) + (-0.05503067374*scaled_PADDING) )
		dense_layer_1_output_2 = self.Tanh( 0.7239971161 + (0.00694407057*scaled_TBX) + (-0.01238186192*scaled_TBY) + (-0.1318278462*scaled_LOCAL) + (-0.386054039*scaled_WPTX) + (-0.5225765705*scaled_WPTY) + (-0.04826911166*scaled_VECTOR) + (-0.7904697061*scaled_UNROLL_FACTOR) + (0.02213766426*scaled_PADDING) )
		dense_layer_1_output_3 = self.Tanh( -0.07744214684 + (0.06075279787*scaled_TBX) + (0.02463284694*scaled_TBY) + (0.2705583572*scaled_LOCAL) + (-1.402402163*scaled_WPTX) + (-0.2747608125*scaled_WPTY) + (-0.1676437259*scaled_VECTOR) + (0.4326362312*scaled_UNROLL_FACTOR) + (0.1885114461*scaled_PADDING) )
		dense_layer_1_output_4 = self.Tanh( -0.6026614904 + (-0.05312112719*scaled_TBX) + (-0.007945034653*scaled_TBY) + (0.479439646*scaled_LOCAL) + (0.7380466461*scaled_WPTX) + (0.6418535709*scaled_WPTY) + (0.07783623785*scaled_VECTOR) + (0.1766673476*scaled_UNROLL_FACTOR) + (-0.139509052*scaled_PADDING) )
		dense_layer_1_output_5 = self.Tanh( -0.1597072929 + (-0.09527163953*scaled_TBX) + (-0.01386060379*scaled_TBY) + (-0.4303832054*scaled_LOCAL) + (-0.138485685*scaled_WPTX) + (-0.697846055*scaled_WPTY) + (0.02787197381*scaled_VECTOR) + (0.7683655024*scaled_UNROLL_FACTOR) + (0.1368943155*scaled_PADDING) )
		dense_layer_1_output_6 = self.Tanh( -1.09226048 + (0.03342719749*scaled_TBX) + (-0.06803967059*scaled_TBY) + (0.2077962011*scaled_LOCAL) + (-0.5091417432*scaled_WPTX) + (-0.8787878156*scaled_WPTY) + (0.01535707992*scaled_VECTOR) + (-0.3636495769*scaled_UNROLL_FACTOR) + (0.1477393657*scaled_PADDING) )
		dense_layer_1_output_7 = self.Tanh( -0.3164347708 + (-0.3504271507*scaled_TBX) + (-0.1637402475*scaled_TBY) + (-0.9165766835*scaled_LOCAL) + (-0.1483747661*scaled_WPTX) + (0.037892472*scaled_WPTY) + (0.03020384163*scaled_VECTOR) + (-0.3319775462*scaled_UNROLL_FACTOR) + (-0.04078171402*scaled_PADDING) )
		dense_layer_1_output_8 = self.Tanh( -0.1908854097 + (0.3918347657*scaled_TBX) + (0.4940882027*scaled_TBY) + (0.4636234045*scaled_LOCAL) + (0.1790803373*scaled_WPTX) + (-0.1220042855*scaled_WPTY) + (0.04859754443*scaled_VECTOR) + (0.002576870378*scaled_UNROLL_FACTOR) + (0.01196305268*scaled_PADDING) )
		dense_layer_1_output_9 = self.Tanh( -1.076374531 + (-0.02706408501*scaled_TBX) + (-0.02777860872*scaled_TBY) + (-0.5861406326*scaled_LOCAL) + (-0.759619832*scaled_WPTX) + (0.4906414747*scaled_WPTY) + (0.005777086597*scaled_VECTOR) + (-0.02712644637*scaled_UNROLL_FACTOR) + (-0.1517820358*scaled_PADDING) )
		dense_layer_1_output_10 = self.Tanh( -0.4625326991 + (0.05698444694*scaled_TBX) + (0.0198175814*scaled_TBY) + (0.3194606304*scaled_LOCAL) + (-1.14946723*scaled_WPTX) + (-0.742844522*scaled_WPTY) + (-0.0765138343*scaled_VECTOR) + (0.1912934482*scaled_UNROLL_FACTOR) + (-0.2846179605*scaled_PADDING) )
		dense_layer_1_output_11 = self.Tanh( -0.5306948423 + (-0.1024988368*scaled_TBX) + (-0.07673048973*scaled_TBY) + (0.9082841873*scaled_LOCAL) + (0.02463842183*scaled_WPTX) + (0.07782986015*scaled_WPTY) + (-0.16238603*scaled_VECTOR) + (-0.1426076889*scaled_UNROLL_FACTOR) + (0.1970381886*scaled_PADDING) )
		dense_layer_1_output_12 = self.Tanh( -0.8536499739 + (-0.005055477377*scaled_TBX) + (-0.03148557618*scaled_TBY) + (-0.2052892596*scaled_LOCAL) + (-0.235414058*scaled_WPTX) + (-0.7261021733*scaled_WPTY) + (-0.1385519803*scaled_VECTOR) + (0.4339343011*scaled_UNROLL_FACTOR) + (-0.577994585*scaled_PADDING) )
		dense_layer_1_output_13 = self.Tanh( -0.7053870559 + (0.07933333516*scaled_TBX) + (0.1162928939*scaled_TBY) + (0.124628149*scaled_LOCAL) + (0.1274839342*scaled_WPTX) + (-1.494929194*scaled_WPTY) + (0.01035570074*scaled_VECTOR) + (-0.2381380349*scaled_UNROLL_FACTOR) + (0.4634000659*scaled_PADDING) )
		dense_layer_1_output_14 = self.Tanh( 0.09722025692 + (0.004228427541*scaled_TBX) + (0.03770535812*scaled_TBY) + (0.3138877749*scaled_LOCAL) + (0.04217642918*scaled_WPTX) + (-0.5474033952*scaled_WPTY) + (0.005066235084*scaled_VECTOR) + (-0.1972848177*scaled_UNROLL_FACTOR) + (0.2201324552*scaled_PADDING) )
		dense_layer_1_output_15 = self.Tanh( 0.2799501419 + (0.04665109143*scaled_TBX) + (0.0600137189*scaled_TBY) + (1.041526794*scaled_LOCAL) + (-0.3897428811*scaled_WPTX) + (-0.5351923108*scaled_WPTY) + (-0.0782237947*scaled_VECTOR) + (0.2292479575*scaled_UNROLL_FACTOR) + (-0.05764552206*scaled_PADDING) )
		dense_layer_1_output_16 = self.Tanh( 0.0566333048 + (0.07817213237*scaled_TBX) + (0.06121718511*scaled_TBY) + (0.247902602*scaled_LOCAL) + (-0.1376733631*scaled_WPTX) + (1.603002429*scaled_WPTY) + (-0.1081402451*scaled_VECTOR) + (-0.4216457307*scaled_UNROLL_FACTOR) + (0.3527908921*scaled_PADDING) )
		dense_layer_1_output_17 = self.Tanh( -0.7805877328 + (-0.1305706948*scaled_TBX) + (-0.1573668122*scaled_TBY) + (-0.02308102138*scaled_LOCAL) + (-0.8505660892*scaled_WPTX) + (0.03773605078*scaled_WPTY) + (-0.03504526988*scaled_VECTOR) + (0.4600960612*scaled_UNROLL_FACTOR) + (0.3245888054*scaled_PADDING) )
		dense_layer_1_output_18 = self.Tanh( 0.1841230094 + (0.01159921195*scaled_TBX) + (-0.01892541349*scaled_TBY) + (-0.6310620308*scaled_LOCAL) + (0.5894619226*scaled_WPTX) + (-0.3309607804*scaled_WPTY) + (0.1516084075*scaled_VECTOR) + (0.3836813569*scaled_UNROLL_FACTOR) + (-0.2262297571*scaled_PADDING) )
		dense_layer_1_output_19 = self.Tanh( -0.3402654827 + (0.03927583247*scaled_TBX) + (0.006082615349*scaled_TBY) + (-0.5618293881*scaled_LOCAL) + (0.2983860672*scaled_WPTX) + (-0.5444009304*scaled_WPTY) + (-0.0566467084*scaled_VECTOR) + (-0.06327690184*scaled_UNROLL_FACTOR) + (0.353256911*scaled_PADDING) )
		dense_layer_1_output_20 = self.Tanh( 0.2173919231 + (0.2888033688*scaled_TBX) + (0.2147934139*scaled_TBY) + (-0.439494282*scaled_LOCAL) + (0.04624922574*scaled_WPTX) + (0.4627944231*scaled_WPTY) + (-0.07727704942*scaled_VECTOR) + (-0.06663668901*scaled_UNROLL_FACTOR) + (0.04878140241*scaled_PADDING) )
		dense_layer_1_output_21 = self.Tanh( -0.1829657406 + (0.07258897275*scaled_TBX) + (0.06812459975*scaled_TBY) + (-0.09730245173*scaled_LOCAL) + (-1.08468163*scaled_WPTX) + (0.2743156254*scaled_WPTY) + (0.03908291459*scaled_VECTOR) + (-0.08363473415*scaled_UNROLL_FACTOR) + (-0.02839875035*scaled_PADDING) )
		dense_layer_1_output_22 = self.Tanh( -1.11819756 + (0.00709409453*scaled_TBX) + (-0.005409200676*scaled_TBY) + (-0.1257621646*scaled_LOCAL) + (-0.1220232397*scaled_WPTX) + (-1.525193214*scaled_WPTY) + (0.005103789736*scaled_VECTOR) + (-0.1346445084*scaled_UNROLL_FACTOR) + (-0.1210150495*scaled_PADDING) )
		dense_layer_1_output_23 = self.Tanh( 0.4189358354 + (-0.08995425701*scaled_TBX) + (-0.01729570329*scaled_TBY) + (-0.9499117136*scaled_LOCAL) + (0.4353945255*scaled_WPTX) + (0.6285613775*scaled_WPTY) + (-0.06082859635*scaled_VECTOR) + (0.476780206*scaled_UNROLL_FACTOR) + (-0.1730777174*scaled_PADDING) )
		approximation_layer_output_0 = self.Identity( 0.8618177176 + (1.166745543*dense_layer_1_output_0) + (-0.2190010846*dense_layer_1_output_1) + (1.196768284*dense_layer_1_output_2) + (-0.3403141201*dense_layer_1_output_3) + (0.00120105152*dense_layer_1_output_4) + (-0.1791896373*dense_layer_1_output_5) + (0.415558517*dense_layer_1_output_6) + (-0.2801341712*dense_layer_1_output_7) + (-0.2084418386*dense_layer_1_output_8) + (-0.129031226*dense_layer_1_output_9) + (0.07113457471*dense_layer_1_output_10) + (0.2805917561*dense_layer_1_output_11) + (-0.1102340221*dense_layer_1_output_12) + (0.0394889228*dense_layer_1_output_13) + (-0.6403995752*dense_layer_1_output_14) + (-0.3351328075*dense_layer_1_output_15) + (0.1788119227*dense_layer_1_output_16) + (0.04648523778*dense_layer_1_output_17) + (0.2331049293*dense_layer_1_output_18) + (-0.05370378494*dense_layer_1_output_19) + (0.3029567301*dense_layer_1_output_20) + (-0.4427498877*dense_layer_1_output_21) + (0.7809380293*dense_layer_1_output_22) + (-0.4033088684*dense_layer_1_output_23) )
		approximation_layer_output_1 = self.Identity( 0.2125849724 + (0.7071948051*dense_layer_1_output_0) + (-0.5134590268*dense_layer_1_output_1) + (-0.0234811902*dense_layer_1_output_2) + (-0.9761220813*dense_layer_1_output_3) + (-0.8395872712*dense_layer_1_output_4) + (-0.8712235093*dense_layer_1_output_5) + (-0.90586555*dense_layer_1_output_6) + (-0.6911933422*dense_layer_1_output_7) + (-0.8317807913*dense_layer_1_output_8) + (1.141492963*dense_layer_1_output_9) + (0.9171490073*dense_layer_1_output_10) + (-0.9074171782*dense_layer_1_output_11) + (-1.166028738*dense_layer_1_output_12) + (-0.9299675226*dense_layer_1_output_13) + (-0.1736187041*dense_layer_1_output_14) + (0.8415257335*dense_layer_1_output_15) + (-0.9573896527*dense_layer_1_output_16) + (0.9152655005*dense_layer_1_output_17) + (-0.5097623467*dense_layer_1_output_18) + (0.898809433*dense_layer_1_output_19) + (0.5249337554*dense_layer_1_output_20) + (-0.706240952*dense_layer_1_output_21) + (1.030458927*dense_layer_1_output_22) + (-0.9055255651*dense_layer_1_output_23) )
		unscaling_layer_output_0=approximation_layer_output_0*0.389977783+3.827429771
		unscaling_layer_output_1=approximation_layer_output_1*0.2399235964+0.5503542423
		log10_duration = unscaling_layer_output_0
		achieved_occupancy = unscaling_layer_output_1
		outputs = [log10_duration, achieved_occupancy]
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
	TBX = 0  # TBX
	TBY = 0  # TBY
	LOCAL = 0  # LOCAL
	WPTX = 0  # WPTX
	WPTY = 0  # WPTY
	VECTOR = 0  # VECTOR
	UNROLL_FACTOR = 0  # UNROLL_FACTOR
	PADDING = 0  # PADDING

	# --- Data conversion (DO NOT modify) ---
	inputs = []

	inputs.append(TBX)
	inputs.append(TBY)
	inputs.append(LOCAL)
	inputs.append(WPTX)
	inputs.append(WPTY)
	inputs.append(VECTOR)
	inputs.append(UNROLL_FACTOR)
	inputs.append(PADDING)

	nn = Network()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
