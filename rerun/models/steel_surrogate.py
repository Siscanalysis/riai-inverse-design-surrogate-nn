''' 
Artificial Intelligence Techniques SL
artelnics@artelnics.com

Model exported to Python. Use NeuralNetwork().calculate_outputs([...]).

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
import numpy as np
import pandas as pd

class NeuralNetwork:

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

		scaled_C = (C-0.1743749976)/0.05990219861
		scaled_Si = (Si-0.3079600036)/0.08513739705
		scaled_Mn = (Mn-0.8011839986)/0.3368929923
		scaled_P = (P-0.01450090017)/0.005301710218
		scaled_S = (S-0.01044990029)/0.003994040191
		scaled_Ni = (Ni-0.1362220049)/0.1674840003
		scaled_Cr = (Cr-0.4270710051)/0.4528209865
		scaled_Mo = (Mo-0.4447030127)/0.3862459958
		scaled_Cu = (Cu-0.07832419872)/0.05778539926
		scaled_V = (V-0.05642259866)/0.09426630288
		scaled_Al = (Al-0.01185250003)/0.01239969954
		scaled_N = (N-0.007640980184)/0.002279639943
		scaled_Nb_Ta = (Nb_Ta-4.644810178e-05)/0.0002773889864
		scaled_temperature_C = (temperature_C-341.1040039)/190.3710022
		dense_layer_1_output_0 = self.Tanh( 0.2903479636 + (-0.7759026289*scaled_C) + (0.1186971813*scaled_Si) + (-0.2590221465*scaled_Mn) + (-0.1141418517*scaled_P) + (-0.03479405493*scaled_S) + (-0.406234473*scaled_Ni) + (-0.3365697265*scaled_Cr) + (0.1236733198*scaled_Mo) + (0.1959812194*scaled_Cu) + (-0.5632587671*scaled_V) + (0.007306851447*scaled_Al) + (-0.2907035649*scaled_N) + (0.09537111968*scaled_Nb_Ta) + (-0.1879571527*scaled_temperature_C) )
		dense_layer_1_output_1 = self.Tanh( 0.8063901663 + (0.1621275544*scaled_C) + (0.3044210076*scaled_Si) + (-0.2408645153*scaled_Mn) + (-0.2384805381*scaled_P) + (0.2320878953*scaled_S) + (0.5396498442*scaled_Ni) + (-0.1941964179*scaled_Cr) + (-0.1140181124*scaled_Mo) + (-0.1226888299*scaled_Cu) + (0.8993384242*scaled_V) + (-0.2786861658*scaled_Al) + (-0.01312712021*scaled_N) + (-0.7853944302*scaled_Nb_Ta) + (0.8935969472*scaled_temperature_C) )
		dense_layer_1_output_2 = self.Tanh( 0.1231464744 + (-0.07071972638*scaled_C) + (-0.2359359115*scaled_Si) + (0.2020110786*scaled_Mn) + (-0.05863793939*scaled_P) + (-0.04738568515*scaled_S) + (-0.1867277324*scaled_Ni) + (0.06132340804*scaled_Cr) + (0.3333692551*scaled_Mo) + (-0.08946347982*scaled_Cu) + (-0.2257372886*scaled_V) + (0.09826363623*scaled_Al) + (-0.3478634655*scaled_N) + (-0.2362961769*scaled_Nb_Ta) + (-0.2497420013*scaled_temperature_C) )
		dense_layer_1_output_3 = self.Tanh( -0.529327631 + (0.1400546134*scaled_C) + (0.8831394315*scaled_Si) + (0.1275663674*scaled_Mn) + (0.1875772178*scaled_P) + (-0.5088077784*scaled_S) + (-0.2087751925*scaled_Ni) + (-0.6866105199*scaled_Cr) + (0.5314921737*scaled_Mo) + (-0.3996553719*scaled_Cu) + (0.4411357641*scaled_V) + (-0.2788105905*scaled_Al) + (-0.02846094221*scaled_N) + (0.379786253*scaled_Nb_Ta) + (-0.106662035*scaled_temperature_C) )
		dense_layer_1_output_4 = self.Tanh( -0.1738300025 + (-0.2046704292*scaled_C) + (0.3055343628*scaled_Si) + (0.5364173651*scaled_Mn) + (-0.01476522814*scaled_P) + (-0.3429907858*scaled_S) + (-0.05863195658*scaled_Ni) + (-0.4465985596*scaled_Cr) + (-0.2491532117*scaled_Mo) + (0.3229385018*scaled_Cu) + (-0.1576578319*scaled_V) + (0.1908324659*scaled_Al) + (-0.0321694836*scaled_N) + (0.0942453444*scaled_Nb_Ta) + (-0.06024966016*scaled_temperature_C) )
		dense_layer_1_output_5 = self.Tanh( -0.592084527 + (0.04391765222*scaled_C) + (0.06859843433*scaled_Si) + (0.03020574152*scaled_Mn) + (-0.02799818479*scaled_P) + (-0.07764515281*scaled_S) + (0.03866615146*scaled_Ni) + (-0.1675457805*scaled_Cr) + (-0.0678980574*scaled_Mo) + (-0.3056977987*scaled_Cu) + (0.09901578724*scaled_V) + (-0.08294627815*scaled_Al) + (-0.2364364117*scaled_N) + (0.001593718072*scaled_Nb_Ta) + (1.372437239*scaled_temperature_C) )
		dense_layer_1_output_6 = self.Tanh( 0.7013918757 + (-0.4300955832*scaled_C) + (0.3660916984*scaled_Si) + (0.07291522622*scaled_Mn) + (-0.1084553227*scaled_P) + (-0.02521723881*scaled_S) + (0.1997826248*scaled_Ni) + (-0.06261076778*scaled_Cr) + (0.175541997*scaled_Mo) + (0.02400524728*scaled_Cu) + (0.3822082281*scaled_V) + (-0.1789150536*scaled_Al) + (-0.08749922365*scaled_N) + (0.07893844694*scaled_Nb_Ta) + (-0.5264090896*scaled_temperature_C) )
		dense_layer_1_output_7 = self.Tanh( 1.017593026 + (-0.3712976277*scaled_C) + (-0.6740014553*scaled_Si) + (-0.2117962986*scaled_Mn) + (0.2204250693*scaled_P) + (-0.007607746869*scaled_S) + (0.1953513175*scaled_Ni) + (-0.06344857067*scaled_Cr) + (-0.1890853047*scaled_Mo) + (0.1910230666*scaled_Cu) + (-0.1164136231*scaled_V) + (-0.002808740595*scaled_Al) + (-0.04082108289*scaled_N) + (-0.211997658*scaled_Nb_Ta) + (0.04805984721*scaled_temperature_C) )
		dense_layer_1_output_8 = self.Tanh( -0.03020787984 + (0.2178038955*scaled_C) + (0.09748654813*scaled_Si) + (-0.06748662144*scaled_Mn) + (-0.07979051769*scaled_P) + (0.6987696886*scaled_S) + (0.1622708589*scaled_Ni) + (0.3292900026*scaled_Cr) + (-0.1678819656*scaled_Mo) + (0.1234304681*scaled_Cu) + (0.3458790481*scaled_V) + (0.02967847139*scaled_Al) + (0.6712613702*scaled_N) + (0.1472823769*scaled_Nb_Ta) + (-0.06894820929*scaled_temperature_C) )
		dense_layer_1_output_9 = self.Tanh( 0.8143090606 + (-0.02983169071*scaled_C) + (-0.4825070202*scaled_Si) + (-0.1323833168*scaled_Mn) + (-0.05634329095*scaled_P) + (0.2709905505*scaled_S) + (-0.2305996567*scaled_Ni) + (-0.08383703232*scaled_Cr) + (-0.0601336956*scaled_Mo) + (0.1836319417*scaled_Cu) + (-0.2403299063*scaled_V) + (0.08846573532*scaled_Al) + (0.3194133341*scaled_N) + (-0.1526713818*scaled_Nb_Ta) + (0.190555349*scaled_temperature_C) )
		dense_layer_1_output_10 = self.Tanh( -0.1877285391 + (-0.4775273204*scaled_C) + (0.3168842494*scaled_Si) + (-0.2290712744*scaled_Mn) + (-0.6471377015*scaled_P) + (0.03481223807*scaled_S) + (0.3254522085*scaled_Ni) + (0.1817956716*scaled_Cr) + (0.1538256258*scaled_Mo) + (0.06817628443*scaled_Cu) + (-0.1992505938*scaled_V) + (-0.2100823671*scaled_Al) + (-0.1277630329*scaled_N) + (-0.1154777855*scaled_Nb_Ta) + (0.1879612207*scaled_temperature_C) )
		dense_layer_1_output_11 = self.Tanh( 0.2931038141 + (0.1057397053*scaled_C) + (-0.03488435224*scaled_Si) + (0.4428013265*scaled_Mn) + (0.5261195302*scaled_P) + (-0.0428180173*scaled_S) + (0.8557581306*scaled_Ni) + (0.1120225191*scaled_Cr) + (-0.4345372915*scaled_Mo) + (0.4441814721*scaled_Cu) + (0.4066739678*scaled_V) + (-0.2907600105*scaled_Al) + (-0.03489396721*scaled_N) + (0.2407575846*scaled_Nb_Ta) + (0.2048585713*scaled_temperature_C) )
		dense_layer_1_output_12 = self.Tanh( 0.4085376263 + (0.03207242116*scaled_C) + (0.06346848607*scaled_Si) + (0.08155623823*scaled_Mn) + (0.1163944453*scaled_P) + (0.08181671053*scaled_S) + (0.1602697074*scaled_Ni) + (-0.4208669066*scaled_Cr) + (0.164501816*scaled_Mo) + (-0.1602384746*scaled_Cu) + (0.2847942412*scaled_V) + (-0.2877572477*scaled_Al) + (0.003378954716*scaled_N) + (0.04697814211*scaled_Nb_Ta) + (-1.003171444*scaled_temperature_C) )
		dense_layer_1_output_13 = self.Tanh( 1.458530188 + (-0.02793982998*scaled_C) + (-0.1384381056*scaled_Si) + (-0.2579353452*scaled_Mn) + (0.01013824064*scaled_P) + (0.07074706256*scaled_S) + (0.08317553252*scaled_Ni) + (-0.1136746705*scaled_Cr) + (0.165716961*scaled_Mo) + (-0.07105018944*scaled_Cu) + (-0.002828967292*scaled_V) + (0.08967281133*scaled_Al) + (-0.08042708784*scaled_N) + (-1.09579277*scaled_Nb_Ta) + (-1.211223006*scaled_temperature_C) )
		dense_layer_1_output_14 = self.Tanh( -0.2417686731 + (-0.3265863955*scaled_C) + (-0.7141656876*scaled_Si) + (0.009983102791*scaled_Mn) + (0.07148148865*scaled_P) + (0.8452844024*scaled_S) + (0.2033946663*scaled_Ni) + (0.2640436888*scaled_Cr) + (0.1446265429*scaled_Mo) + (-0.2536258399*scaled_Cu) + (0.3966023922*scaled_V) + (0.2675050497*scaled_Al) + (-0.01060906705*scaled_N) + (0.2668833137*scaled_Nb_Ta) + (0.289819628*scaled_temperature_C) )
		dense_layer_1_output_15 = self.Tanh( 0.3207675815 + (0.1604773253*scaled_C) + (-0.5823477507*scaled_Si) + (0.1314150542*scaled_Mn) + (0.204633683*scaled_P) + (0.09606301785*scaled_S) + (0.0356714502*scaled_Ni) + (0.1599715501*scaled_Cr) + (0.4611086249*scaled_Mo) + (-0.287165761*scaled_Cu) + (0.1167048067*scaled_V) + (0.2185897976*scaled_Al) + (-0.1313405484*scaled_N) + (-0.3743704855*scaled_Nb_Ta) + (0.2837339342*scaled_temperature_C) )
		dense_layer_1_output_16 = self.Tanh( 0.01578653976 + (-0.4554884136*scaled_C) + (0.2284595966*scaled_Si) + (-0.1034557968*scaled_Mn) + (0.006320982706*scaled_P) + (-0.01539796218*scaled_S) + (-0.3910220563*scaled_Ni) + (-0.1870200187*scaled_Cr) + (-0.6163324118*scaled_Mo) + (0.2961213887*scaled_Cu) + (-0.1244992912*scaled_V) + (0.3270022571*scaled_Al) + (-0.01016048528*scaled_N) + (-0.7135426402*scaled_Nb_Ta) + (0.4149518907*scaled_temperature_C) )
		dense_layer_1_output_17 = self.Tanh( -0.00396157708 + (-0.2374584079*scaled_C) + (0.7732799053*scaled_Si) + (0.5637906194*scaled_Mn) + (0.3149324954*scaled_P) + (-0.4104206264*scaled_S) + (-0.2923178375*scaled_Ni) + (-0.02852664329*scaled_Cr) + (0.04867651686*scaled_Mo) + (-0.433277458*scaled_Cu) + (0.6169899106*scaled_V) + (-0.01061698236*scaled_Al) + (-0.3736593723*scaled_N) + (-0.3994211853*scaled_Nb_Ta) + (-0.1727948189*scaled_temperature_C) )
		dense_layer_1_output_18 = self.Tanh( -0.9738147855 + (0.1477820277*scaled_C) + (-0.3211866617*scaled_Si) + (0.05781411007*scaled_Mn) + (0.05346563831*scaled_P) + (-0.1272085905*scaled_S) + (-0.2973185182*scaled_Ni) + (-0.5494785905*scaled_Cr) + (0.04586329684*scaled_Mo) + (-0.07335454226*scaled_Cu) + (0.2106003463*scaled_V) + (-0.2697505355*scaled_Al) + (0.05939610675*scaled_N) + (-0.182656467*scaled_Nb_Ta) + (1.160827518*scaled_temperature_C) )
		dense_layer_1_output_19 = self.Tanh( -1.581903934 + (-0.08502906561*scaled_C) + (-0.09533640742*scaled_Si) + (-0.09985870868*scaled_Mn) + (0.01997177489*scaled_P) + (-0.07100672275*scaled_S) + (-0.08700557053*scaled_Ni) + (-0.1090719551*scaled_Cr) + (-0.4753011465*scaled_Mo) + (0.07815407962*scaled_Cu) + (0.04710644111*scaled_V) + (-0.07218699902*scaled_Al) + (0.005103911739*scaled_N) + (0.006860672496*scaled_Nb_Ta) + (1.186089039*scaled_temperature_C) )
		dense_layer_1_output_20 = self.Tanh( -0.7195888758 + (-0.08438404649*scaled_C) + (-0.3292106092*scaled_Si) + (0.5901693106*scaled_Mn) + (0.1166970134*scaled_P) + (0.1832756549*scaled_S) + (0.3320582807*scaled_Ni) + (0.0663819313*scaled_Cr) + (-0.2096412182*scaled_Mo) + (-0.005133908708*scaled_Cu) + (0.1577165574*scaled_V) + (0.5586751103*scaled_Al) + (-0.08072827011*scaled_N) + (0.4352063835*scaled_Nb_Ta) + (0.09179561585*scaled_temperature_C) )
		dense_layer_1_output_21 = self.Tanh( 1.132216096 + (-0.1901865751*scaled_C) + (-0.2990601063*scaled_Si) + (-0.05471440032*scaled_Mn) + (0.02955098823*scaled_P) + (-0.2405104637*scaled_S) + (-0.16574727*scaled_Ni) + (0.07239221781*scaled_Cr) + (-0.06482500583*scaled_Mo) + (0.2360198498*scaled_Cu) + (-0.0478107892*scaled_V) + (-0.02831585705*scaled_Al) + (0.02261425927*scaled_N) + (-0.0521036312*scaled_Nb_Ta) + (-0.8429602385*scaled_temperature_C) )
		dense_layer_1_output_22 = self.Tanh( 0.564678669 + (0.007176242303*scaled_C) + (0.002670686226*scaled_Si) + (-0.3322938085*scaled_Mn) + (-0.248347789*scaled_P) + (0.1850614846*scaled_S) + (0.3280165792*scaled_Ni) + (0.08596344292*scaled_Cr) + (-0.01675280556*scaled_Mo) + (-0.04283683002*scaled_Cu) + (-0.1701218039*scaled_V) + (-0.4496810138*scaled_Al) + (0.2430002987*scaled_N) + (-0.7647450566*scaled_Nb_Ta) + (-0.01218150929*scaled_temperature_C) )
		dense_layer_1_output_23 = self.Tanh( 0.1160718352 + (-0.1142610684*scaled_C) + (-0.4897217453*scaled_Si) + (-0.04811726138*scaled_Mn) + (0.2197088748*scaled_P) + (-0.1129765809*scaled_S) + (-0.1077561453*scaled_Ni) + (-0.1889332235*scaled_Cr) + (-0.07385797799*scaled_Mo) + (0.8772854209*scaled_Cu) + (0.5517250299*scaled_V) + (-0.5312336683*scaled_Al) + (-0.1798357517*scaled_N) + (-0.133066237*scaled_Nb_Ta) + (0.006082569715*scaled_temperature_C) )
		dense_layer_1_output_24 = self.Tanh( -0.3928181529 + (0.6671644449*scaled_C) + (0.03245198727*scaled_Si) + (0.1823533028*scaled_Mn) + (-0.04998356104*scaled_P) + (-0.2991215289*scaled_S) + (-0.2902913392*scaled_Ni) + (-0.05179268122*scaled_Cr) + (0.01162529644*scaled_Mo) + (0.2309036702*scaled_Cu) + (-0.2891707718*scaled_V) + (-0.4033490121*scaled_Al) + (0.2013847828*scaled_N) + (0.8754588962*scaled_Nb_Ta) + (-0.002727363026*scaled_temperature_C) )
		dense_layer_1_output_25 = self.Tanh( -0.2394129634 + (-0.1645864546*scaled_C) + (0.1433573961*scaled_Si) + (0.3071546555*scaled_Mn) + (0.01340284944*scaled_P) + (0.03096238524*scaled_S) + (0.1989038438*scaled_Ni) + (0.08817415684*scaled_Cr) + (0.01093769539*scaled_Mo) + (0.0869814679*scaled_Cu) + (-0.06359198689*scaled_V) + (0.1696690321*scaled_Al) + (0.04397606105*scaled_N) + (-0.08783245087*scaled_Nb_Ta) + (-0.8152320981*scaled_temperature_C) )
		dense_layer_1_output_26 = self.Tanh( -0.7623574734 + (0.01326617412*scaled_C) + (-0.1436086744*scaled_Si) + (0.6755585074*scaled_Mn) + (0.01708830148*scaled_P) + (0.2083333135*scaled_S) + (0.06279316545*scaled_Ni) + (-0.03538418561*scaled_Cr) + (0.0359320417*scaled_Mo) + (-0.2111729532*scaled_Cu) + (0.01498729456*scaled_V) + (0.1667629033*scaled_Al) + (0.04478362203*scaled_N) + (0.184100315*scaled_Nb_Ta) + (-0.8449469805*scaled_temperature_C) )
		dense_layer_1_output_27 = self.Tanh( 0.03053359315 + (0.06506919116*scaled_C) + (0.288004458*scaled_Si) + (0.4328534305*scaled_Mn) + (0.08979201317*scaled_P) + (0.2392562181*scaled_S) + (-0.2484163642*scaled_Ni) + (-0.2270828187*scaled_Cr) + (-0.1170417666*scaled_Mo) + (-0.008418391459*scaled_Cu) + (-0.1206780821*scaled_V) + (0.1657773256*scaled_Al) + (-0.01946533285*scaled_N) + (-0.08693916351*scaled_Nb_Ta) + (0.01661451347*scaled_temperature_C) )
		dense_layer_1_output_28 = self.Tanh( 0.06121472642 + (-0.4814828634*scaled_C) + (0.4910341799*scaled_Si) + (0.5226294994*scaled_Mn) + (0.372379899*scaled_P) + (-0.03171874955*scaled_S) + (-0.4015616477*scaled_Ni) + (-0.07302696258*scaled_Cr) + (-0.322755754*scaled_Mo) + (-0.8147550821*scaled_Cu) + (-0.3707171679*scaled_V) + (0.3441733122*scaled_Al) + (0.1987504959*scaled_N) + (0.01887235418*scaled_Nb_Ta) + (-0.1145639941*scaled_temperature_C) )
		dense_layer_1_output_29 = self.Tanh( -0.05810979381 + (-0.2048922777*scaled_C) + (-0.0697131753*scaled_Si) + (0.4077632427*scaled_Mn) + (-0.2283878922*scaled_P) + (-0.02588312887*scaled_S) + (0.1321917772*scaled_Ni) + (-0.002436395036*scaled_Cr) + (0.4660448432*scaled_Mo) + (0.1536857039*scaled_Cu) + (0.1231680661*scaled_V) + (0.08708261698*scaled_Al) + (-0.09227061272*scaled_N) + (0.06984693557*scaled_Nb_Ta) + (-0.6019698381*scaled_temperature_C) )
		dense_layer_1_output_30 = self.Tanh( -0.01912927814 + (-0.05542119965*scaled_C) + (0.2232363671*scaled_Si) + (-0.4224032462*scaled_Mn) + (0.01347155217*scaled_P) + (-0.01181936357*scaled_S) + (-0.6677039266*scaled_Ni) + (0.2455130666*scaled_Cr) + (-0.2450113595*scaled_Mo) + (0.06585906446*scaled_Cu) + (-0.1827856004*scaled_V) + (0.3252337277*scaled_Al) + (-0.1378991604*scaled_N) + (-0.313313067*scaled_Nb_Ta) + (-0.008073040284*scaled_temperature_C) )
		dense_layer_1_output_31 = self.Tanh( 0.4242067933 + (-0.06173950061*scaled_C) + (-0.1665043384*scaled_Si) + (0.1722435057*scaled_Mn) + (0.3737237155*scaled_P) + (0.1989171654*scaled_S) + (0.2316417992*scaled_Ni) + (-0.270965308*scaled_Cr) + (-0.1482577175*scaled_Mo) + (-0.2170191109*scaled_Cu) + (-0.1346898526*scaled_V) + (0.4547498822*scaled_Al) + (-0.0953419134*scaled_N) + (0.1683038771*scaled_Nb_Ta) + (-0.3664205074*scaled_temperature_C) )
		approximation_layer_output_0 = self.Identity( 0.005235382821 + (0.02260867506*dense_layer_1_output_0) + (0.008305655792*dense_layer_1_output_1) + (0.2545650303*dense_layer_1_output_2) + (0.3885550201*dense_layer_1_output_3) + (0.05669589713*dense_layer_1_output_4) + (0.3436820209*dense_layer_1_output_5) + (0.4098629653*dense_layer_1_output_6) + (-0.3188871145*dense_layer_1_output_7) + (0.2777798772*dense_layer_1_output_8) + (-0.5540415049*dense_layer_1_output_9) + (-0.5586510897*dense_layer_1_output_10) + (-0.08739925921*dense_layer_1_output_11) + (0.0763328746*dense_layer_1_output_12) + (0.4939677417*dense_layer_1_output_13) + (0.3343503177*dense_layer_1_output_14) + (0.2665079832*dense_layer_1_output_15) + (0.475366503*dense_layer_1_output_16) + (-0.1501054019*dense_layer_1_output_17) + (-0.4026854634*dense_layer_1_output_18) + (0.05513259396*dense_layer_1_output_19) + (0.4257436991*dense_layer_1_output_20) + (0.3519163132*dense_layer_1_output_21) + (0.3140011132*dense_layer_1_output_22) + (0.2957801521*dense_layer_1_output_23) + (0.1343214661*dense_layer_1_output_24) + (0.1495090276*dense_layer_1_output_25) + (0.2037200332*dense_layer_1_output_26) + (0.04555259272*dense_layer_1_output_27) + (0.1061294526*dense_layer_1_output_28) + (0.2027731538*dense_layer_1_output_29) + (-0.3144707084*dense_layer_1_output_30) + (-0.7356576324*dense_layer_1_output_31) )
		approximation_layer_output_1 = self.Identity( -0.7675365806 + (-0.03012658842*dense_layer_1_output_0) + (0.6230843663*dense_layer_1_output_1) + (0.1966904998*dense_layer_1_output_2) + (0.3917534351*dense_layer_1_output_3) + (0.05773769692*dense_layer_1_output_4) + (-0.1774916798*dense_layer_1_output_5) + (0.06757829338*dense_layer_1_output_6) + (-0.1903043091*dense_layer_1_output_7) + (0.2187059075*dense_layer_1_output_8) + (-0.1965088844*dense_layer_1_output_9) + (0.02596984431*dense_layer_1_output_10) + (-0.07269894332*dense_layer_1_output_11) + (0.5081684589*dense_layer_1_output_12) + (0.5274014473*dense_layer_1_output_13) + (0.2469926029*dense_layer_1_output_14) + (0.2058500648*dense_layer_1_output_15) + (0.3539320827*dense_layer_1_output_16) + (-0.09633576125*dense_layer_1_output_17) + (-0.4216586649*dense_layer_1_output_18) + (-0.5920502543*dense_layer_1_output_19) + (0.6571763754*dense_layer_1_output_20) + (0.4954208136*dense_layer_1_output_21) + (-0.3544791043*dense_layer_1_output_22) + (0.3758362532*dense_layer_1_output_23) + (0.3040527403*dense_layer_1_output_24) + (-0.3195098639*dense_layer_1_output_25) + (0.3455695212*dense_layer_1_output_26) + (-0.6477609277*dense_layer_1_output_27) + (0.2069787979*dense_layer_1_output_28) + (-0.2587555051*dense_layer_1_output_29) + (0.07603421807*dense_layer_1_output_30) + (-0.2584955096*dense_layer_1_output_31) )
		approximation_layer_output_2 = self.Identity( 0.754755199 + (0.2937133908*dense_layer_1_output_0) + (0.2681605816*dense_layer_1_output_1) + (-0.3382074833*dense_layer_1_output_2) + (-0.258459121*dense_layer_1_output_3) + (-0.2026133388*dense_layer_1_output_4) + (0.356490016*dense_layer_1_output_5) + (-0.456569314*dense_layer_1_output_6) + (0.7469007969*dense_layer_1_output_7) + (-0.2046237439*dense_layer_1_output_8) + (0.1276035011*dense_layer_1_output_9) + (-0.1682025939*dense_layer_1_output_10) + (0.557682395*dense_layer_1_output_11) + (0.179317534*dense_layer_1_output_12) + (-0.4486702085*dense_layer_1_output_13) + (-0.2722783983*dense_layer_1_output_14) + (-0.4019297659*dense_layer_1_output_15) + (-0.1594079584*dense_layer_1_output_16) + (-0.114875406*dense_layer_1_output_17) + (0.2189338058*dense_layer_1_output_18) + (1.117713213*dense_layer_1_output_19) + (-0.6119130254*dense_layer_1_output_20) + (-0.4913381934*dense_layer_1_output_21) + (0.06121841446*dense_layer_1_output_22) + (-0.6739599705*dense_layer_1_output_23) + (0.1205920503*dense_layer_1_output_24) + (0.1512648761*dense_layer_1_output_25) + (0.3224317133*dense_layer_1_output_26) + (-0.1344655156*dense_layer_1_output_27) + (-0.07897105068*dense_layer_1_output_28) + (0.9625977874*dense_layer_1_output_29) + (0.8495989442*dense_layer_1_output_30) + (0.2836882472*dense_layer_1_output_31) )
		approximation_layer_output_3 = self.Identity( 1.073782086 + (0.2523213327*dense_layer_1_output_0) + (-0.2483553439*dense_layer_1_output_1) + (0.1745485961*dense_layer_1_output_2) + (-0.3996938169*dense_layer_1_output_3) + (-0.4203903377*dense_layer_1_output_4) + (0.7855053544*dense_layer_1_output_5) + (0.199578777*dense_layer_1_output_6) + (0.3095079362*dense_layer_1_output_7) + (-0.0314187482*dense_layer_1_output_8) + (-0.8390903473*dense_layer_1_output_9) + (-0.3127911091*dense_layer_1_output_10) + (0.4888546467*dense_layer_1_output_11) + (-0.9433879256*dense_layer_1_output_12) + (0.007015484385*dense_layer_1_output_13) + (-0.25237903*dense_layer_1_output_14) + (-0.6648055911*dense_layer_1_output_15) + (-0.1178416088*dense_layer_1_output_16) + (-0.5141721368*dense_layer_1_output_17) + (0.7234899402*dense_layer_1_output_18) + (0.07845172286*dense_layer_1_output_19) + (-0.3698863387*dense_layer_1_output_20) + (-0.5914531946*dense_layer_1_output_21) + (0.1502689868*dense_layer_1_output_22) + (-0.1728538722*dense_layer_1_output_23) + (-0.2260028869*dense_layer_1_output_24) + (0.6612746716*dense_layer_1_output_25) + (0.4961248934*dense_layer_1_output_26) + (-0.136240527*dense_layer_1_output_27) + (0.5064796209*dense_layer_1_output_28) + (0.3206586838*dense_layer_1_output_29) + (0.04654295743*dense_layer_1_output_30) + (-0.1584683657*dense_layer_1_output_31) )
		unscaling_layer_output_0=approximation_layer_output_0*132.3099518+325.8287659
		unscaling_layer_output_1=approximation_layer_output_1*129.0365906+491.4043579
		unscaling_layer_output_2=approximation_layer_output_2*9.219474792+26.8670311
		unscaling_layer_output_3=approximation_layer_output_3*12.6859293+69.51184082
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

	nn = NeuralNetwork()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

if __name__ == "__main__":
	main()
