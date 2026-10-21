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
		dense_layer_1_output_0 = self.Tanh( -0.08723794669 + (0.06472938508*scaled_Cement_kg_m3) + (0.05330406502*scaled_FineAgg_kg_m3) + (0.06573780626*scaled_CoarseAgg_kg_m3) + (0.09408868104*scaled_Water_kg_m3) + (-0.1130617633*scaled_WaterReducer_kg_m3) + (-0.1489917785*scaled_FlyAsh_kg_m3) + (-0.00437515229*scaled_Accelerator_kg_m3) + (-0.05757721514*scaled_SilicaFume_kg_m3) + (0.04491255432*scaled_TestAge_days) )
		dense_layer_1_output_1 = self.Tanh( -0.333368063 + (0.2759580016*scaled_Cement_kg_m3) + (0.01018443331*scaled_FineAgg_kg_m3) + (0.08892595768*scaled_CoarseAgg_kg_m3) + (-0.1132860109*scaled_Water_kg_m3) + (0.1396854073*scaled_WaterReducer_kg_m3) + (0.6570603848*scaled_FlyAsh_kg_m3) + (0.1149676442*scaled_Accelerator_kg_m3) + (0.1897604316*scaled_SilicaFume_kg_m3) + (-0.05962859839*scaled_TestAge_days) )
		dense_layer_1_output_2 = self.Tanh( 0.3820424974 + (-0.5483861566*scaled_Cement_kg_m3) + (-0.7400714755*scaled_FineAgg_kg_m3) + (0.1101111919*scaled_CoarseAgg_kg_m3) + (0.5647526979*scaled_Water_kg_m3) + (-0.1614068151*scaled_WaterReducer_kg_m3) + (-0.1644394696*scaled_FlyAsh_kg_m3) + (-0.01793326065*scaled_Accelerator_kg_m3) + (0.1827520132*scaled_SilicaFume_kg_m3) + (-0.01850437*scaled_TestAge_days) )
		dense_layer_1_output_3 = self.Tanh( 0.02800095454 + (0.02265222371*scaled_Cement_kg_m3) + (-0.4908289611*scaled_FineAgg_kg_m3) + (0.1276650131*scaled_CoarseAgg_kg_m3) + (0.3637919128*scaled_Water_kg_m3) + (-0.3431560695*scaled_WaterReducer_kg_m3) + (-0.2386975586*scaled_FlyAsh_kg_m3) + (0.2460563034*scaled_Accelerator_kg_m3) + (0.04323417321*scaled_SilicaFume_kg_m3) + (-0.1004959196*scaled_TestAge_days) )
		dense_layer_1_output_4 = self.Tanh( 0.2743210196 + (-0.155155912*scaled_Cement_kg_m3) + (0.02838599682*scaled_FineAgg_kg_m3) + (0.00692708604*scaled_CoarseAgg_kg_m3) + (0.04174593091*scaled_Water_kg_m3) + (-0.1287900358*scaled_WaterReducer_kg_m3) + (-0.1047660112*scaled_FlyAsh_kg_m3) + (-0.1428195387*scaled_Accelerator_kg_m3) + (-0.03580995649*scaled_SilicaFume_kg_m3) + (-0.2708119154*scaled_TestAge_days) )
		dense_layer_1_output_5 = self.Tanh( -0.5624060631 + (0.03981223702*scaled_Cement_kg_m3) + (-0.2434609979*scaled_FineAgg_kg_m3) + (-0.0608936213*scaled_CoarseAgg_kg_m3) + (0.6096356511*scaled_Water_kg_m3) + (-0.2039957494*scaled_WaterReducer_kg_m3) + (-0.3647164404*scaled_FlyAsh_kg_m3) + (-0.05986984819*scaled_Accelerator_kg_m3) + (-0.07154803723*scaled_SilicaFume_kg_m3) + (-0.01432820782*scaled_TestAge_days) )
		dense_layer_1_output_6 = self.Tanh( -0.1716462225 + (0.0170575697*scaled_Cement_kg_m3) + (0.1390826106*scaled_FineAgg_kg_m3) + (0.1157152504*scaled_CoarseAgg_kg_m3) + (0.004950590432*scaled_Water_kg_m3) + (-0.4186978638*scaled_WaterReducer_kg_m3) + (-0.3239260018*scaled_FlyAsh_kg_m3) + (0.06857269257*scaled_Accelerator_kg_m3) + (0.08001565188*scaled_SilicaFume_kg_m3) + (-0.01136751939*scaled_TestAge_days) )
		dense_layer_1_output_7 = self.Tanh( 0.05584417656 + (0.1643588841*scaled_Cement_kg_m3) + (0.8483743072*scaled_FineAgg_kg_m3) + (-0.3906590939*scaled_CoarseAgg_kg_m3) + (0.407220751*scaled_Water_kg_m3) + (0.1243137866*scaled_WaterReducer_kg_m3) + (0.003898628056*scaled_FlyAsh_kg_m3) + (0.3128461838*scaled_Accelerator_kg_m3) + (0.08716902882*scaled_SilicaFume_kg_m3) + (-0.0386348702*scaled_TestAge_days) )
		dense_layer_1_output_8 = self.Tanh( 0.001591041801 + (-0.003415165935*scaled_Cement_kg_m3) + (-0.01418278366*scaled_FineAgg_kg_m3) + (0.1317058802*scaled_CoarseAgg_kg_m3) + (-0.1245983914*scaled_Water_kg_m3) + (0.1317370385*scaled_WaterReducer_kg_m3) + (0.04901699722*scaled_FlyAsh_kg_m3) + (-0.01245748438*scaled_Accelerator_kg_m3) + (-0.07062202692*scaled_SilicaFume_kg_m3) + (-0.2127188742*scaled_TestAge_days) )
		dense_layer_1_output_9 = self.Tanh( -0.3572106361 + (-0.3889627457*scaled_Cement_kg_m3) + (-0.2832016945*scaled_FineAgg_kg_m3) + (-0.01163787767*scaled_CoarseAgg_kg_m3) + (0.2189905792*scaled_Water_kg_m3) + (-0.1770361066*scaled_WaterReducer_kg_m3) + (0.5208283663*scaled_FlyAsh_kg_m3) + (0.1118829548*scaled_Accelerator_kg_m3) + (0.2330232263*scaled_SilicaFume_kg_m3) + (-0.06598640978*scaled_TestAge_days) )
		dense_layer_1_output_10 = self.Tanh( 0.05132380873 + (-0.2574320436*scaled_Cement_kg_m3) + (-0.209422946*scaled_FineAgg_kg_m3) + (0.04058921337*scaled_CoarseAgg_kg_m3) + (0.2681785226*scaled_Water_kg_m3) + (-0.2855088413*scaled_WaterReducer_kg_m3) + (0.1454906017*scaled_FlyAsh_kg_m3) + (-0.2344871759*scaled_Accelerator_kg_m3) + (-0.04560475424*scaled_SilicaFume_kg_m3) + (0.003450990887*scaled_TestAge_days) )
		dense_layer_1_output_11 = self.Tanh( -0.5853554606 + (0.5021941066*scaled_Cement_kg_m3) + (-0.07825641334*scaled_FineAgg_kg_m3) + (-1.179256678*scaled_CoarseAgg_kg_m3) + (0.6306108832*scaled_Water_kg_m3) + (0.26031214*scaled_WaterReducer_kg_m3) + (0.1670004129*scaled_FlyAsh_kg_m3) + (-0.01339058578*scaled_Accelerator_kg_m3) + (0.2325423062*scaled_SilicaFume_kg_m3) + (-0.02576963231*scaled_TestAge_days) )
		dense_layer_1_output_12 = self.Tanh( 0.1771866232 + (-0.3412840962*scaled_Cement_kg_m3) + (-0.003430662211*scaled_FineAgg_kg_m3) + (0.1229169294*scaled_CoarseAgg_kg_m3) + (0.03875649348*scaled_Water_kg_m3) + (-0.01577629708*scaled_WaterReducer_kg_m3) + (-0.3182755411*scaled_FlyAsh_kg_m3) + (-0.001051033265*scaled_Accelerator_kg_m3) + (-0.09190413356*scaled_SilicaFume_kg_m3) + (0.00999817811*scaled_TestAge_days) )
		dense_layer_1_output_13 = self.Tanh( 0.03307745606 + (0.06674446166*scaled_Cement_kg_m3) + (-0.03773789108*scaled_FineAgg_kg_m3) + (-0.1226761267*scaled_CoarseAgg_kg_m3) + (-4.14251117e-05*scaled_Water_kg_m3) + (-0.07279842347*scaled_WaterReducer_kg_m3) + (-0.1606741548*scaled_FlyAsh_kg_m3) + (0.0537321046*scaled_Accelerator_kg_m3) + (0.1435086131*scaled_SilicaFume_kg_m3) + (-0.02173534408*scaled_TestAge_days) )
		dense_layer_1_output_14 = self.Tanh( -0.1068344563 + (0.3656748533*scaled_Cement_kg_m3) + (-0.367618233*scaled_FineAgg_kg_m3) + (0.1120629162*scaled_CoarseAgg_kg_m3) + (-0.2174916267*scaled_Water_kg_m3) + (-0.03502526134*scaled_WaterReducer_kg_m3) + (-0.2154489905*scaled_FlyAsh_kg_m3) + (0.09833405912*scaled_Accelerator_kg_m3) + (0.02474603802*scaled_SilicaFume_kg_m3) + (0.02421198227*scaled_TestAge_days) )
		dense_layer_1_output_15 = self.Tanh( -0.5078556538 + (-0.2358243465*scaled_Cement_kg_m3) + (0.1491222829*scaled_FineAgg_kg_m3) + (0.8642971516*scaled_CoarseAgg_kg_m3) + (-0.4275924563*scaled_Water_kg_m3) + (0.66998595*scaled_WaterReducer_kg_m3) + (0.5117475986*scaled_FlyAsh_kg_m3) + (0.1475552619*scaled_Accelerator_kg_m3) + (-0.10762804*scaled_SilicaFume_kg_m3) + (-0.05030074716*scaled_TestAge_days) )
		dense_layer_1_output_16 = self.Tanh( -0.4273526669 + (-0.9984916449*scaled_Cement_kg_m3) + (0.1470871568*scaled_FineAgg_kg_m3) + (1.139835238*scaled_CoarseAgg_kg_m3) + (-0.8119212985*scaled_Water_kg_m3) + (-0.3533891439*scaled_WaterReducer_kg_m3) + (-0.08556368202*scaled_FlyAsh_kg_m3) + (0.3899533153*scaled_Accelerator_kg_m3) + (0.3976908028*scaled_SilicaFume_kg_m3) + (0.008049334399*scaled_TestAge_days) )
		dense_layer_1_output_17 = self.Tanh( -0.008655048907 + (0.1279246807*scaled_Cement_kg_m3) + (0.08027345687*scaled_FineAgg_kg_m3) + (-0.006950419396*scaled_CoarseAgg_kg_m3) + (-0.02591645531*scaled_Water_kg_m3) + (0.1880277991*scaled_WaterReducer_kg_m3) + (-0.1652563214*scaled_FlyAsh_kg_m3) + (-0.2931528091*scaled_Accelerator_kg_m3) + (-0.06960444897*scaled_SilicaFume_kg_m3) + (-0.0907690376*scaled_TestAge_days) )
		dense_layer_1_output_18 = self.Tanh( 0.02488574572 + (-0.05378703028*scaled_Cement_kg_m3) + (0.04499039426*scaled_FineAgg_kg_m3) + (0.009924427606*scaled_CoarseAgg_kg_m3) + (-0.1152629033*scaled_Water_kg_m3) + (-0.1517524123*scaled_WaterReducer_kg_m3) + (0.1295244247*scaled_FlyAsh_kg_m3) + (-0.05472797155*scaled_Accelerator_kg_m3) + (-0.03863733262*scaled_SilicaFume_kg_m3) + (-0.01676692814*scaled_TestAge_days) )
		dense_layer_1_output_19 = self.Tanh( 0.3985144198 + (-0.2285667062*scaled_Cement_kg_m3) + (-0.04337909818*scaled_FineAgg_kg_m3) + (0.1111536846*scaled_CoarseAgg_kg_m3) + (0.1820272505*scaled_Water_kg_m3) + (-0.11565727*scaled_WaterReducer_kg_m3) + (-0.1163747683*scaled_FlyAsh_kg_m3) + (-0.05834658444*scaled_Accelerator_kg_m3) + (-0.06708905846*scaled_SilicaFume_kg_m3) + (-0.1570183635*scaled_TestAge_days) )
		dense_layer_1_output_20 = self.Tanh( 0.01672912762 + (0.164469853*scaled_Cement_kg_m3) + (-0.6476704478*scaled_FineAgg_kg_m3) + (-0.03243572637*scaled_CoarseAgg_kg_m3) + (0.5812932849*scaled_Water_kg_m3) + (0.8834937811*scaled_WaterReducer_kg_m3) + (-0.2538973987*scaled_FlyAsh_kg_m3) + (0.1767922938*scaled_Accelerator_kg_m3) + (-0.2857473195*scaled_SilicaFume_kg_m3) + (0.0280841291*scaled_TestAge_days) )
		dense_layer_1_output_21 = self.Tanh( 0.2525551915 + (-0.03190105408*scaled_Cement_kg_m3) + (-0.2108766884*scaled_FineAgg_kg_m3) + (-0.06079361588*scaled_CoarseAgg_kg_m3) + (0.2306459546*scaled_Water_kg_m3) + (0.1201436892*scaled_WaterReducer_kg_m3) + (0.2956635952*scaled_FlyAsh_kg_m3) + (0.0129687516*scaled_Accelerator_kg_m3) + (0.05047076195*scaled_SilicaFume_kg_m3) + (-0.0615554899*scaled_TestAge_days) )
		dense_layer_1_output_22 = self.Tanh( 0.1821741164 + (0.0488935113*scaled_Cement_kg_m3) + (0.04430603236*scaled_FineAgg_kg_m3) + (-0.01250810921*scaled_CoarseAgg_kg_m3) + (0.00668912055*scaled_Water_kg_m3) + (-0.1097758338*scaled_WaterReducer_kg_m3) + (-0.2114249766*scaled_FlyAsh_kg_m3) + (0.02610028535*scaled_Accelerator_kg_m3) + (-0.08720383793*scaled_SilicaFume_kg_m3) + (0.02061000094*scaled_TestAge_days) )
		dense_layer_1_output_23 = self.Tanh( 0.6534997821 + (-0.7773479819*scaled_Cement_kg_m3) + (0.6237139106*scaled_FineAgg_kg_m3) + (-0.282473594*scaled_CoarseAgg_kg_m3) + (0.1307152063*scaled_Water_kg_m3) + (-0.8162408471*scaled_WaterReducer_kg_m3) + (-0.07061674446*scaled_FlyAsh_kg_m3) + (0.3068894744*scaled_Accelerator_kg_m3) + (0.1990415752*scaled_SilicaFume_kg_m3) + (-0.05486782268*scaled_TestAge_days) )
		dense_layer_1_output_24 = self.Tanh( 0.09283985198 + (-0.1969963759*scaled_Cement_kg_m3) + (-0.1543070823*scaled_FineAgg_kg_m3) + (0.05683472008*scaled_CoarseAgg_kg_m3) + (0.1763855666*scaled_Water_kg_m3) + (-0.2316711545*scaled_WaterReducer_kg_m3) + (0.1604102254*scaled_FlyAsh_kg_m3) + (0.345630914*scaled_Accelerator_kg_m3) + (-0.1759797931*scaled_SilicaFume_kg_m3) + (0.04631236941*scaled_TestAge_days) )
		dense_layer_1_output_25 = self.Tanh( -0.03957412392 + (0.04633349553*scaled_Cement_kg_m3) + (-0.1297551394*scaled_FineAgg_kg_m3) + (-0.04001391307*scaled_CoarseAgg_kg_m3) + (0.026706459*scaled_Water_kg_m3) + (0.1298819631*scaled_WaterReducer_kg_m3) + (-0.04979088157*scaled_FlyAsh_kg_m3) + (0.08071548492*scaled_Accelerator_kg_m3) + (0.04488057271*scaled_SilicaFume_kg_m3) + (0.04078688473*scaled_TestAge_days) )
		dense_layer_1_output_26 = self.Tanh( 0.07774826884 + (-0.02900469676*scaled_Cement_kg_m3) + (0.05597172305*scaled_FineAgg_kg_m3) + (0.02515978552*scaled_CoarseAgg_kg_m3) + (-0.1365154088*scaled_Water_kg_m3) + (-0.2007153034*scaled_WaterReducer_kg_m3) + (0.08763903379*scaled_FlyAsh_kg_m3) + (-0.08045479655*scaled_Accelerator_kg_m3) + (-0.04636723548*scaled_SilicaFume_kg_m3) + (-0.1538091302*scaled_TestAge_days) )
		dense_layer_1_output_27 = self.Tanh( -0.08577816933 + (0.131357342*scaled_Cement_kg_m3) + (-0.2470122725*scaled_FineAgg_kg_m3) + (-0.00127025519*scaled_CoarseAgg_kg_m3) + (0.1417453289*scaled_Water_kg_m3) + (0.003739940701*scaled_WaterReducer_kg_m3) + (0.1527250707*scaled_FlyAsh_kg_m3) + (0.03663068265*scaled_Accelerator_kg_m3) + (-0.152135849*scaled_SilicaFume_kg_m3) + (-0.03754350916*scaled_TestAge_days) )
		dense_layer_1_output_28 = self.Tanh( -0.4420473278 + (-0.6439831853*scaled_Cement_kg_m3) + (0.07598149031*scaled_FineAgg_kg_m3) + (0.4786743522*scaled_CoarseAgg_kg_m3) + (-0.488640517*scaled_Water_kg_m3) + (0.4310205877*scaled_WaterReducer_kg_m3) + (-0.47209692*scaled_FlyAsh_kg_m3) + (-0.05366284028*scaled_Accelerator_kg_m3) + (-0.009794942103*scaled_SilicaFume_kg_m3) + (0.08404330909*scaled_TestAge_days) )
		dense_layer_1_output_29 = self.Tanh( 0.467535466 + (0.06026721001*scaled_Cement_kg_m3) + (-0.002559760585*scaled_FineAgg_kg_m3) + (-0.2045107335*scaled_CoarseAgg_kg_m3) + (-0.1425183862*scaled_Water_kg_m3) + (-0.1372641176*scaled_WaterReducer_kg_m3) + (-0.5620085001*scaled_FlyAsh_kg_m3) + (0.04023681208*scaled_Accelerator_kg_m3) + (-0.1824468225*scaled_SilicaFume_kg_m3) + (-0.0136186583*scaled_TestAge_days) )
		dense_layer_1_output_30 = self.Tanh( -0.1926144212 + (0.6997698545*scaled_Cement_kg_m3) + (-0.04730115831*scaled_FineAgg_kg_m3) + (-0.07028707862*scaled_CoarseAgg_kg_m3) + (-1.053072214*scaled_Water_kg_m3) + (-0.2981987596*scaled_WaterReducer_kg_m3) + (0.1934344172*scaled_FlyAsh_kg_m3) + (0.4304212928*scaled_Accelerator_kg_m3) + (0.1759150922*scaled_SilicaFume_kg_m3) + (0.009283599444*scaled_TestAge_days) )
		dense_layer_1_output_31 = self.Tanh( 0.2875235975 + (0.0008820551448*scaled_Cement_kg_m3) + (-0.3999630213*scaled_FineAgg_kg_m3) + (-1.009498596*scaled_CoarseAgg_kg_m3) + (1.518513322*scaled_Water_kg_m3) + (0.02041286789*scaled_WaterReducer_kg_m3) + (0.6312955618*scaled_FlyAsh_kg_m3) + (0.2571983337*scaled_Accelerator_kg_m3) + (-0.2022142261*scaled_SilicaFume_kg_m3) + (-0.0359697789*scaled_TestAge_days) )
		approximation_layer_output_0 = self.Identity( -0.02249860764 + (0.2554397285*dense_layer_1_output_0) + (0.4239289463*dense_layer_1_output_1) + (-0.5588248968*dense_layer_1_output_2) + (-0.1504955143*dense_layer_1_output_3) + (-0.6377533078*dense_layer_1_output_4) + (0.4582490623*dense_layer_1_output_5) + (-0.3879502714*dense_layer_1_output_6) + (-0.8226630688*dense_layer_1_output_7) + (-0.4188239872*dense_layer_1_output_8) + (0.5470277071*dense_layer_1_output_9) + (-0.6187444925*dense_layer_1_output_10) + (-0.9396119118*dense_layer_1_output_11) + (-0.06493265927*dense_layer_1_output_12) + (0.1769427061*dense_layer_1_output_13) + (0.5102612972*dense_layer_1_output_14) + (-0.06451737136*dense_layer_1_output_15) + (-0.5216575861*dense_layer_1_output_16) + (0.3088863194*dense_layer_1_output_17) + (-0.3018718064*dense_layer_1_output_18) + (-0.556820631*dense_layer_1_output_19) + (-0.1935706735*dense_layer_1_output_20) + (0.424341917*dense_layer_1_output_21) + (0.03380134702*dense_layer_1_output_22) + (0.5460373163*dense_layer_1_output_23) + (-0.5103510618*dense_layer_1_output_24) + (0.3196653426*dense_layer_1_output_25) + (-0.4762378931*dense_layer_1_output_26) + (0.161559552*dense_layer_1_output_27) + (-0.8018148541*dense_layer_1_output_28) + (-0.1641395688*dense_layer_1_output_29) + (-0.1860208958*dense_layer_1_output_30) + (-0.83005023*dense_layer_1_output_31) )
		approximation_layer_output_1 = self.Identity( -0.4157185555 + (-0.3994989395*dense_layer_1_output_0) + (0.7213566303*dense_layer_1_output_1) + (-0.7814577222*dense_layer_1_output_2) + (0.5014930964*dense_layer_1_output_3) + (-0.120055683*dense_layer_1_output_4) + (-0.7645418644*dense_layer_1_output_5) + (-0.3819645643*dense_layer_1_output_6) + (0.2914896309*dense_layer_1_output_7) + (0.2292420268*dense_layer_1_output_8) + (0.6102060676*dense_layer_1_output_9) + (0.2479524612*dense_layer_1_output_10) + (-0.9697709084*dense_layer_1_output_11) + (-0.4334883988*dense_layer_1_output_12) + (-0.2274998575*dense_layer_1_output_13) + (-0.3728552163*dense_layer_1_output_14) + (-1.027633786*dense_layer_1_output_15) + (-1.201468229*dense_layer_1_output_16) + (-0.3504603207*dense_layer_1_output_17) + (0.1025084928*dense_layer_1_output_18) + (-0.3742225766*dense_layer_1_output_19) + (1.19113338*dense_layer_1_output_20) + (0.04309827834*dense_layer_1_output_21) + (-0.4455103874*dense_layer_1_output_22) + (0.9780939817*dense_layer_1_output_23) + (0.2530332506*dense_layer_1_output_24) + (0.1090655178*dense_layer_1_output_25) + (0.1471760869*dense_layer_1_output_26) + (0.2830915749*dense_layer_1_output_27) + (0.1656210721*dense_layer_1_output_28) + (-0.7404814959*dense_layer_1_output_29) + (-1.041450739*dense_layer_1_output_30) + (-1.707428575*dense_layer_1_output_31) )
		unscaling_layer_output_0=approximation_layer_output_0*12.52991581+38.26176834
		unscaling_layer_output_1=approximation_layer_output_1*36.27813339+178.5
		CompressiveStrength_MPa = max(4.300000191, unscaling_layer_output_0)
		CompressiveStrength_MPa = min(76.30000305, CompressiveStrength_MPa)
		Slump_mm = max(50, unscaling_layer_output_1)
		Slump_mm = min(230, Slump_mm)
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
