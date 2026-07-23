''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Model exported to Python. Use NeuralNetwork().calculate_outputs([...]).

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
import numpy as np
import pandas as pd

class NeuralNetwork:

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

		scaled_Al = (Al-0.1696780026)/0.1383219957
		scaled_Co = (Co-0.1518770009)/0.08867190033
		scaled_Cr = (Cr-0.1776400059)/0.06926059723
		scaled_Cu = (Cu-0.07900989801)/0.08697970212
		scaled_Fe = (Fe-0.1784799993)/0.08002910018
		scaled_Mn = (Mn-0.01920880005)/0.06212360039
		scaled_Mo = (Mo-0.01037380006)/0.03896119818
		scaled_Ni = (Ni-0.1734489948)/0.09208159894
		scaled_Ti = (Ti-0.02496539988)/0.06535550207
		scaled_V = (V-0.01531790011)/0.05251679942
		dense2d_layer_1_output_0 = self.Tanh( 0.3878651261 + (0.1621605158*scaled_Al) + (-0.4920036495*scaled_Co) + (0.01992530376*scaled_Cr) + (0.3659285307*scaled_Cu) + (0.3255742788*scaled_Fe) + (-0.3424162567*scaled_Mn) + (0.07351847738*scaled_Mo) + (-0.4628992975*scaled_Ni) + (0.3200575113*scaled_Ti) + (0.05222685263*scaled_V) )
		dense2d_layer_1_output_1 = self.Tanh( 0.3316882849 + (0.4069102705*scaled_Al) + (-0.1325802356*scaled_Co) + (0.3164407611*scaled_Cr) + (-0.06022797525*scaled_Cu) + (-0.3021041155*scaled_Fe) + (-0.1422887892*scaled_Mn) + (-0.1096348912*scaled_Mo) + (-0.05969056115*scaled_Ni) + (-0.1100168228*scaled_Ti) + (-0.007053526118*scaled_V) )
		dense2d_layer_1_output_2 = self.Tanh( 0.02996760607 + (-0.2541643679*scaled_Al) + (-0.5517036319*scaled_Co) + (0.1227377802*scaled_Cr) + (0.2872110307*scaled_Cu) + (0.6016375422*scaled_Fe) + (-0.3398681283*scaled_Mn) + (0.01224347111*scaled_Mo) + (0.1041984037*scaled_Ni) + (0.1147731468*scaled_Ti) + (-0.01856601425*scaled_V) )
		dense2d_layer_1_output_3 = self.Tanh( -0.3698215783 + (0.1667025238*scaled_Al) + (-0.2602851689*scaled_Co) + (0.5269675851*scaled_Cr) + (0.04100244865*scaled_Cu) + (-0.2787780464*scaled_Fe) + (0.1755996943*scaled_Mn) + (-0.07574687153*scaled_Mo) + (-0.218848899*scaled_Ni) + (-0.05187483504*scaled_Ti) + (-0.01004085597*scaled_V) )
		dense2d_layer_1_output_4 = self.Tanh( 0.007852022536 + (0.01809479482*scaled_Al) + (-0.03115740232*scaled_Co) + (-0.01322855707*scaled_Cr) + (0.006821345072*scaled_Cu) + (0.008897897787*scaled_Fe) + (0.002909094561*scaled_Mn) + (-0.003506485838*scaled_Mo) + (-0.004572852049*scaled_Ni) + (0.0009404972079*scaled_Ti) + (0.01028314885*scaled_V) )
		dense2d_layer_1_output_5 = self.Tanh( 0.3744817376 + (0.4582899809*scaled_Al) + (-0.2068382502*scaled_Co) + (-0.1943041831*scaled_Cr) + (-0.0186719466*scaled_Cu) + (-0.2468150854*scaled_Fe) + (0.1346561164*scaled_Mn) + (-0.3041423261*scaled_Mo) + (-0.01350205112*scaled_Ni) + (-0.0432658568*scaled_Ti) + (-0.05692682788*scaled_V) )
		dense2d_layer_1_output_6 = self.Tanh( 0.1587396264 + (0.1260746866*scaled_Al) + (-0.1904564798*scaled_Co) + (0.03230298683*scaled_Cr) + (-0.1132320017*scaled_Cu) + (-0.1310048997*scaled_Fe) + (-0.1161472201*scaled_Mn) + (0.08839806914*scaled_Mo) + (0.132638067*scaled_Ni) + (0.2822566926*scaled_Ti) + (-0.0301565621*scaled_V) )
		dense2d_layer_1_output_7 = self.Tanh( -0.007179861423 + (-0.0163285993*scaled_Al) + (0.02985983901*scaled_Co) + (0.01258150302*scaled_Cr) + (-0.00737933768*scaled_Cu) + (-0.008290533908*scaled_Fe) + (-0.002611806151*scaled_Mn) + (0.004236098379*scaled_Mo) + (0.004361963831*scaled_Ni) + (-0.0008739188197*scaled_Ti) + (-0.009942410514*scaled_V) )
		dense2d_layer_1_output_8 = self.Tanh( 0.2260310799 + (0.04381626099*scaled_Al) + (0.004493208136*scaled_Co) + (0.04565637559*scaled_Cr) + (-0.4385714829*scaled_Cu) + (0.2613659799*scaled_Fe) + (0.001773652155*scaled_Mn) + (-0.00868678093*scaled_Mo) + (0.07055204362*scaled_Ni) + (0.1328757107*scaled_Ti) + (-0.1029145271*scaled_V) )
		dense2d_layer_1_output_9 = self.Tanh( -0.007348405197 + (-0.01707189344*scaled_Al) + (0.02894885279*scaled_Co) + (0.01228167675*scaled_Cr) + (-0.006649578456*scaled_Cu) + (-0.008240068331*scaled_Fe) + (-0.002631224925*scaled_Mn) + (0.004140864126*scaled_Mo) + (0.004305799957*scaled_Ni) + (-0.0006665087421*scaled_Ti) + (-0.01027841307*scaled_V) )
		dense2d_layer_1_output_10 = self.Tanh( 0.0008485103026 + (0.001203989028*scaled_Al) + (-0.002603802131*scaled_Co) + (-0.001016670954*scaled_Cr) + (0.0008850563318*scaled_Cu) + (0.0006303476403*scaled_Fe) + (0.0002077247482*scaled_Mn) + (-0.0003622496151*scaled_Mo) + (-0.0005460966495*scaled_Ni) + (8.401347441e-05*scaled_Ti) + (0.0008627797361*scaled_V) )
		dense2d_layer_1_output_11 = self.Tanh( -0.1077488139 + (-0.1300847828*scaled_Al) + (-0.3214483559*scaled_Co) + (0.2861437798*scaled_Cr) + (-0.5346714854*scaled_Cu) + (0.2867568135*scaled_Fe) + (-0.2185038328*scaled_Mn) + (0.1435013711*scaled_Mo) + (0.6202355623*scaled_Ni) + (0.1911845356*scaled_Ti) + (-0.1999885738*scaled_V) )
		dense2d_layer_1_output_12 = self.Tanh( 0.01173677668 + (0.0351013951*scaled_Al) + (-0.0522463657*scaled_Co) + (-0.02090610936*scaled_Cr) + (0.00462902477*scaled_Cu) + (0.004773125518*scaled_Fe) + (0.00505394116*scaled_Mn) + (-0.007712092716*scaled_Mo) + (-0.006633888464*scaled_Ni) + (0.002763756085*scaled_Ti) + (0.0161753837*scaled_V) )
		dense2d_layer_1_output_13 = self.Tanh( -0.003277780488 + (-0.006977066398*scaled_Al) + (0.0129779689*scaled_Co) + (0.004944673739*scaled_Cr) + (-0.003753554542*scaled_Cu) + (-0.003307077335*scaled_Fe) + (-0.0009172111168*scaled_Mn) + (0.001855051145*scaled_Mo) + (0.002713594818*scaled_Ni) + (-0.0002334042219*scaled_Ti) + (-0.004117077682*scaled_V) )
		dense2d_layer_1_output_14 = self.Tanh( -0.001843563747 + (0.04040529206*scaled_Al) + (-0.05983236805*scaled_Co) + (-0.03042389266*scaled_Cr) + (-0.01965934597*scaled_Cu) + (0.01885864139*scaled_Fe) + (0.005597983487*scaled_Mn) + (-0.006734196097*scaled_Mo) + (-0.0008917137748*scaled_Ni) + (-0.004446398467*scaled_Ti) + (0.02223705314*scaled_V) )
		dense2d_layer_1_output_15 = self.Tanh( 0.07689311355 + (0.4380305409*scaled_Al) + (0.1178115979*scaled_Co) + (-0.2351607829*scaled_Cr) + (-0.07100028545*scaled_Cu) + (-0.2324041128*scaled_Fe) + (-0.06472810358*scaled_Mn) + (0.04943297431*scaled_Mo) + (0.1514892131*scaled_Ni) + (-0.4102466702*scaled_Ti) + (0.1072650254*scaled_V) )
		dense2d_layer_1_output_16 = self.Tanh( 0.2628047466 + (-0.1578732133*scaled_Al) + (0.1549059153*scaled_Co) + (0.09003156424*scaled_Cr) + (0.3755118847*scaled_Cu) + (-0.1447025985*scaled_Fe) + (-0.06582796574*scaled_Mn) + (-0.1074895263*scaled_Mo) + (-0.1724904627*scaled_Ni) + (0.03629185632*scaled_Ti) + (0.05547951907*scaled_V) )
		dense2d_layer_1_output_17 = self.Tanh( -0.0008483950514 + (-0.001377080334*scaled_Al) + (0.002580762608*scaled_Co) + (0.0009936949937*scaled_Cr) + (-0.0008671708056*scaled_Cu) + (-0.000741965312*scaled_Fe) + (-0.0001925952674*scaled_Mn) + (0.0003879945143*scaled_Mo) + (0.0004223054566*scaled_Ni) + (-9.077462164e-05*scaled_Ti) + (-0.0008281236514*scaled_V) )
		dense2d_layer_1_output_18 = self.Tanh( -0.06418500096 + (0.2966057658*scaled_Al) + (0.1175930277*scaled_Co) + (-0.2226748765*scaled_Cr) + (0.06137938052*scaled_Cu) + (-0.4064258337*scaled_Fe) + (0.09555973113*scaled_Mn) + (0.0852227658*scaled_Mo) + (0.1245423555*scaled_Ni) + (-0.3569446802*scaled_Ti) + (-0.08667689562*scaled_V) )
		dense2d_layer_1_output_19 = self.Tanh( -0.2572826147 + (0.01474555861*scaled_Al) + (0.2490850687*scaled_Co) + (0.2456140518*scaled_Cr) + (-0.3020154834*scaled_Cu) + (-0.04407073557*scaled_Fe) + (0.04671292379*scaled_Mn) + (0.007299117278*scaled_Mo) + (-0.103427425*scaled_Ni) + (0.0002408411383*scaled_Ti) + (-0.02056313306*scaled_V) )
		dense2d_layer_1_output_20 = self.Tanh( 7.583091246e-06 + (1.491208423e-05*scaled_Al) + (-3.281548925e-05*scaled_Co) + (-1.161662476e-05*scaled_Cr) + (1.278536092e-05*scaled_Cu) + (7.43495184e-06*scaled_Fe) + (2.044279199e-06*scaled_Mn) + (-5.010922905e-06*scaled_Mo) + (-4.360788353e-06*scaled_Ni) + (-8.851851021e-07*scaled_Ti) + (1.129387874e-05*scaled_V) )
		dense2d_layer_1_output_21 = self.Tanh( -0.07550127804 + (0.1658837348*scaled_Al) + (0.1565961689*scaled_Co) + (-0.5913273692*scaled_Cr) + (-0.469607234*scaled_Cu) + (0.008620226756*scaled_Fe) + (0.09137286246*scaled_Mn) + (0.2818015516*scaled_Mo) + (-0.3713565171*scaled_Ni) + (0.6545851231*scaled_Ti) + (0.3692517579*scaled_V) )
		dense2d_layer_1_output_22 = self.Tanh( 0.0005145688774 + (0.00103940058*scaled_Al) + (-0.002130745212*scaled_Co) + (-0.0008042828413*scaled_Cr) + (0.0007747457712*scaled_Cu) + (0.000493532687*scaled_Fe) + (0.0001435079466*scaled_Mn) + (-0.0003242394596*scaled_Mo) + (-0.0003092762781*scaled_Ni) + (-4.762072058e-05*scaled_Ti) + (0.0007076095208*scaled_V) )
		dense2d_layer_1_output_23 = self.Tanh( -0.3782241642 + (0.4027432203*scaled_Al) + (-0.1401186883*scaled_Co) + (-0.09441242367*scaled_Cr) + (-0.6216428876*scaled_Cu) + (0.1206600443*scaled_Fe) + (0.1447297633*scaled_Mn) + (-0.1400744468*scaled_Mo) + (0.1632168889*scaled_Ni) + (-0.01640156284*scaled_Ti) + (-0.0946450457*scaled_V) )
		dense2d_layer_1_output_24 = self.Tanh( 0.1269336194 + (0.2171382308*scaled_Al) + (-0.06581705064*scaled_Co) + (-0.1145437732*scaled_Cr) + (-0.2404763401*scaled_Cu) + (-0.6313973069*scaled_Fe) + (0.04434210807*scaled_Mn) + (0.01623395085*scaled_Mo) + (0.2206588835*scaled_Ni) + (0.5445839763*scaled_Ti) + (-0.02215737663*scaled_V) )
		dense2d_layer_1_output_25 = self.Tanh( 0.4191363156 + (1.255684018*scaled_Al) + (-0.5161767602*scaled_Co) + (-0.169546783*scaled_Cr) + (-0.6027715206*scaled_Cu) + (-0.2221502811*scaled_Fe) + (-0.2202419341*scaled_Mn) + (0.1018591896*scaled_Mo) + (-0.5994356871*scaled_Ni) + (-0.7148737907*scaled_Ti) + (0.07730763406*scaled_V) )
		dense2d_layer_1_output_26 = self.Tanh( 0.006202648394 + (0.01406531874*scaled_Al) + (-0.02443137579*scaled_Co) + (-0.01015937701*scaled_Cr) + (0.006274026353*scaled_Cu) + (0.007067259401*scaled_Fe) + (0.002147881314*scaled_Mn) + (-0.003288394306*scaled_Mo) + (-0.003817948746*scaled_Ni) + (0.0006604160299*scaled_Ti) + (0.008069242351*scaled_V) )
		dense2d_layer_1_output_27 = self.Tanh( -0.01572709903 + (-0.04219190404*scaled_Al) + (0.05864175782*scaled_Co) + (0.03163434938*scaled_Cr) + (0.01927654259*scaled_Cu) + (-0.01557017118*scaled_Fe) + (-0.004850139376*scaled_Mn) + (0.003711867845*scaled_Mo) + (0.006710448768*scaled_Ni) + (0.001611115295*scaled_Ti) + (-0.02477952093*scaled_V) )
		dense2d_layer_1_output_28 = self.Tanh( -0.290259093 + (-0.03048423119*scaled_Al) + (-0.1843004525*scaled_Co) + (0.01030449849*scaled_Cr) + (0.4701489806*scaled_Cu) + (-0.1890655011*scaled_Fe) + (0.0112244999*scaled_Mn) + (0.07310029119*scaled_Mo) + (-0.1447940022*scaled_Ni) + (-0.1332326084*scaled_Ti) + (0.2261630595*scaled_V) )
		dense2d_layer_1_output_29 = self.Tanh( 0.04950351268 + (0.001019928255*scaled_Al) + (0.09593442082*scaled_Co) + (0.04053302854*scaled_Cr) + (-0.2965899408*scaled_Cu) + (0.144592464*scaled_Fe) + (0.006346073002*scaled_Mn) + (-0.0103473831*scaled_Mo) + (-0.0178324338*scaled_Ni) + (0.07024610043*scaled_Ti) + (-0.05362676084*scaled_V) )
		dense2d_layer_1_output_30 = self.Tanh( 0.1133718044 + (-0.2375555038*scaled_Al) + (-0.008043738082*scaled_Co) + (0.4700775445*scaled_Cr) + (-0.1605581045*scaled_Cu) + (0.3088047206*scaled_Fe) + (0.06902322918*scaled_Mn) + (-0.1191449836*scaled_Mo) + (0.2014041543*scaled_Ni) + (-0.1543602496*scaled_Ti) + (-0.08266063035*scaled_V) )
		dense2d_layer_1_output_31 = self.Tanh( 0.2969947159 + (0.3886719346*scaled_Al) + (-0.08424461633*scaled_Co) + (-0.3322500587*scaled_Cr) + (-0.08561665565*scaled_Cu) + (0.1338734478*scaled_Fe) + (0.1131445318*scaled_Mn) + (-0.01639025658*scaled_Mo) + (-0.3758682013*scaled_Ni) + (-0.05737932399*scaled_Ti) + (0.07751529664*scaled_V) )
		dense2d_layer_1_output_32 = self.Tanh( 0.1501682252 + (0.02893552929*scaled_Al) + (-0.194751814*scaled_Co) + (-0.1739973873*scaled_Cr) + (0.2281575352*scaled_Cu) + (0.02382531203*scaled_Fe) + (-0.0207164418*scaled_Mn) + (0.07037581503*scaled_Mo) + (0.06801029295*scaled_Ni) + (-0.0002568012278*scaled_Ti) + (-0.07513900846*scaled_V) )
		dense2d_layer_1_output_33 = self.Tanh( 0.01163236704 + (0.0273007527*scaled_Al) + (-0.04454119131*scaled_Co) + (-0.01996746473*scaled_Cr) + (-0.006608109456*scaled_Cu) + (0.01131139044*scaled_Fe) + (0.003586592153*scaled_Mn) + (-0.005898816045*scaled_Mo) + (-0.006594479084*scaled_Ni) + (-0.0004092184245*scaled_Ti) + (0.01609716937*scaled_V) )
		dense2d_layer_1_output_34 = self.Tanh( 0.278605938 + (-0.3288426399*scaled_Al) + (0.3192980886*scaled_Co) + (0.248452425*scaled_Cr) + (-0.1262803674*scaled_Cu) + (-0.00554084871*scaled_Fe) + (-0.3104839325*scaled_Mn) + (0.4839759767*scaled_Mo) + (0.06879725307*scaled_Ni) + (0.152889505*scaled_Ti) + (-0.1377291828*scaled_V) )
		dense2d_layer_1_output_35 = self.Tanh( 0.2736386061 + (0.06035529822*scaled_Al) + (-0.2619147003*scaled_Co) + (-0.07240798324*scaled_Cr) + (-0.1643889248*scaled_Cu) + (-0.2318316698*scaled_Fe) + (-0.09451916069*scaled_Mn) + (-0.008928991854*scaled_Mo) + (0.4244590402*scaled_Ni) + (0.3483681083*scaled_Ti) + (-0.1016807854*scaled_V) )
		dense2d_layer_1_output_36 = self.Tanh( -0.01002022251 + (-0.03115268983*scaled_Al) + (0.04294921085*scaled_Co) + (0.02277458087*scaled_Cr) + (-0.009267128073*scaled_Cu) + (-0.02379643545*scaled_Fe) + (-0.004182762932*scaled_Mn) + (0.01171453483*scaled_Mo) + (0.01928188652*scaled_Ni) + (0.008573344909*scaled_Ti) + (-0.01707874238*scaled_V) )
		approximation_layer_output_0 = self.Identity( -0.1751651466 + (0.669455409*dense2d_layer_1_output_0) + (-0.5043676496*dense2d_layer_1_output_1) + (0.585667491*dense2d_layer_1_output_2) + (0.5071127415*dense2d_layer_1_output_3) + (-0.0508273989*dense2d_layer_1_output_4) + (-0.6343079209*dense2d_layer_1_output_5) + (0.3529197276*dense2d_layer_1_output_6) + (0.04716737196*dense2d_layer_1_output_7) + (0.412156105*dense2d_layer_1_output_8) + (0.04874852672*dense2d_layer_1_output_9) + (-0.004446819425*dense2d_layer_1_output_10) + (-0.6658608317*dense2d_layer_1_output_11) + (-0.0739794746*dense2d_layer_1_output_12) + (0.02148642205*dense2d_layer_1_output_13) + (-0.1140867472*dense2d_layer_1_output_14) + (0.4646408856*dense2d_layer_1_output_15) + (0.441097796*dense2d_layer_1_output_16) + (0.003804411506*dense2d_layer_1_output_17) + (0.4847997725*dense2d_layer_1_output_18) + (0.4708241522*dense2d_layer_1_output_19) + (-5.360382784e-05*dense2d_layer_1_output_20) + (0.7921639085*dense2d_layer_1_output_21) + (-0.003713573562*dense2d_layer_1_output_22) + (-0.6606520414*dense2d_layer_1_output_23) + (0.6516354084*dense2d_layer_1_output_24) + (1.014901042*dense2d_layer_1_output_25) + (-0.04032618552*dense2d_layer_1_output_26) + (0.1084036008*dense2d_layer_1_output_27) + (-0.4833393097*dense2d_layer_1_output_28) + (0.3029858768*dense2d_layer_1_output_29) + (0.5539913774*dense2d_layer_1_output_30) + (-0.5861546993*dense2d_layer_1_output_31) + (-0.3319468498*dense2d_layer_1_output_32) + (-0.0807583034*dense2d_layer_1_output_33) + (-0.5819454193*dense2d_layer_1_output_34) + (0.5399955511*dense2d_layer_1_output_35) + (0.09040411562*dense2d_layer_1_output_36) )
		unscaling_layer_output_0=approximation_layer_output_0*228.0934296+492.0870667
		HV = unscaling_layer_output_0
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

	nn = NeuralNetwork()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
