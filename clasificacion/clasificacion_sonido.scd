

// Multi Layer Perceptron Classification

// Predicción, categorización, clasificación

// Flujo de Trabajo
// Audio analisis -> NN -> Label



// Abrimos los dos archivo de sonido que contiene las muestras de un Trobón y un Oboe. Los cargamos en dos Buffers

~trombone = Buffer.read(s,FluidFilesPath("Olencki-TenTromboneLongTones-M.wav"));
~oboe = Buffer.read(s,FluidFilesPath("Harker-DS-TenOboeMultiphonics-M.wav"));

FluidWaveform(~trombone, imageColorScheme:1,imageColorScaling:1, waveformColor: Color.red);

// Podemos reproducir los Buffers usando el método play;

~trombone.play;
~oboe.play;

// Los Buffer tienen un canal.

// Vamos a dividir los sonido en dos, para tener un Buffer para el entrenemiento y otro para testear la red neuronal. Lo dividiremos en el sample 1783040

// Desde el frame 0 a 1783040 guardaremos un Buffer pata entrenar

~trombone_train = Buffer.read(s,FluidFilesPath("Olencki-TenTromboneLongTones-M.wav"),0,1783040);

FluidWaveform(~trombone_train, imageColorScheme: 2);

~oboe_train = Buffer.read(s,FluidFilesPath("Harker-DS-TenOboeMultiphonics-M.wav"),0,1783040);

// Desde el cuadro 1783040 hasta el final guardaremos un Buffer para testear.

~trombone_test = Buffer.read(s,FluidFilesPath("Olencki-TenTromboneLongTones-M.wav"),1783040);
~oboe_test = Buffer.read(s,FluidFilesPath("Harker-DS-TenOboeMultiphonics-M.wav"),1783040);


(

// Declaramos una variable para contener la cantidad de coeficientes que queramos obtener del analisis MFCC

~nmfccs = 13;

// Declaramos un Buffer de 13 elementos para contener los valores de los coeficientes

~mfccbuf = Buffer.alloc(s,13);

// Declaramos un FluidDataSet para los datos del timbre y un FluidDataSet para los labels que vamos a asociar a los datos de timbre

~timbredata = FluidDataSet(s);
~labels = FluidLabelSet(s);

//Declaramos una variable para contar los puntos que agregaremos a la red neuronal

~counter = 0;

// Declaramos una función audio análisis usando el análisis  Mel-Frequency Cepstral Coefficients (cepstral) para la descripción del timbre del objeto sonoro.

~realtime_analysis = {

	// Declaramos el argumento de la función para el buffer que se analizará y reproducirá en la función
	
	arg buf;

	{
		// Dentro de la función declaramos un instrumento
		// que reproduzca el Buffer con el Ugen PlayBuf
		// y analice el timbre con FluidMFCC
		
		var sig = PlayBuf.ar(
			1,
			buf,
			BufRateScale.ir(buf),
			doneAction:2
		);

		var mfccs = FluidMFCC.kr(
			sig,

			// Cantidad de coeficientes de salida. Este está limitado
			// por el número máximo de coeficientes
		
			~nmfccs,
			// Comenzamos del coeficiente 1 y no 0, debido
			// a que este último representa la energía,
			// la que no no queremos usar en la NN 

			startCoeff: 1,

			// Número máximo de coeficientes

			maxNumCoeffs: ~nmfccs
		);
		
		// Guardamos el análisis timbral en un Buffer
		// Usando el método FluidKrToBuf que nos
		// ayuda a escribir un flujo de datos de control
		// en un Buffer

		FluidKrToBuf.kr(mfccs,~mfccbuf);
		
		// Repetimos la señal para escuchar
		// en los dos canales de audio
		
		sig.dup;
	}.play;
};


// Definimos una función ~add_point para agregar la data del timbre y la data del label a sus respectivos FluidDataSet junto con el identificador id que servira para asociar ambos cuenpos de datos, es decir el timbre con su respectivo label, con los cuales sera entrenada la red neuronal.

~add_point = {
	arg label;
	var id = "example-%".format(~counter);
	~timbredata.addPoint(id,~mfccbuf);
	~labels.addLabel(id,label);
	~counter = ~counter + 1;
};
)

// Función reproducción + análisis

// Para ingresar los datos que serán usados en la NN
// que van a la clasificación, reproduciremos el sonido con
// la función ~realtime_analisis al mismo tiempo que agregamos
// puntos a nuestros FluidDataSet con el argumento del label que
// corresponde al sonido

// Reproducimos y analizamos el buffer de entrenamiento
// que contiene el sonido del trombone

~realtime_analysis.(~trombone_train);

// Mientras el sonido del trombone suena
// agregamos un punto con el label del trombone
// como argumento de la función.

~add_point.("trombone");

// Lo mismo para el oboe.

~realtime_analysis.(~oboe_train);
~add_point.("oboe");

// Podemos ver las componentes de
// lod FluidDataSet en que guaradmos
// el análisis timbral y sus respectivos
// labels.

~timbredata.print;
~labels.print;

// Para limpiar el contenido de los
// FluidDataSet

~timbredata.clear;
~labels.clear;


// Creamos el clasificador usando el UGen FluidMLPClassifier. Este UGen realiza la clasificacion entre un FluidDataSet y un FluidDataSetLabel usando una red neuronal perceptron multi-capa.
(
~nn = FluidMLPClassifier(
	// servidor
	s,
	
	// Un Array que define la estructura interna de la red neronal.
	// Cada valor de la lista representa una capa de la red neuronal,
	// y el valor es el número de neuronas en esa capa.

	hidden:[5],

	// La activacion es un entero que indica la funcion
	// que cada neurona de la(las) capas internas usarán.
	// 0 identity, 1 sigmoid, 2	relu, 3	tanh.

	activation:1,

	// El número de épocas de entrenamiento
	// En una época consiste en el entrenamiento usando
	// todos los data points una vez.
	
	maxIter: 1000,

	// Un escalar que indica cuanto la red neuronal
	// debe ajustar sus parametros internos durane el entrenamiento.
	// Este es un parametro importante para ajustar la NN. 0.0 - 1.0
	
	learnRate: 0.1,

	// Un escalar que aplica una porcion
	// de los ajustes de la epoca anterior a la actual epoca
	// durante el entrenamiento. 0.0 - 0.99
	
	momentum: 0.9,

	// El número de data points usados entre ajustes durante el entrenamiento.

	batchSize: 5,

	// Un porcentaje, representado como número decimal,
	// de los data points no usados, dejados fuera,
	// y no usados para el entrenamiento, y que es
	// re-seleccionado en cada epoca, y asi calcular
	// como el entrenamiento de la red neuronal esta evolucionando
	// 0 - 0.9
	
	validation: 0
);
)


// Función para realizar una epoca manualmente, que nos entrega
//  

(

// La función fit entrena la red neuronal
// usando el FluidDataSet fuente y el
// FluidDataSet de salída.

~nn.fit(~timbredata, ~labels, {
	arg loss;
	loss.postln;
});
)

// Definimos una funcion para definir querying function on the synth

(

// Creamos un Buffer de un cuadro

~predictions = Buffer.alloc(s, 1);

// Definir una función llamado ~server_predictions

~server_predictions = {

	// El argumento de la función
	// para pasar el buffer del sonido
	// que queramos clasificar
	
	arg buf;

	// Definimos una funcion OSCdef con key \predictions
	// que responde a la dirección /prediction
	
	OSCdef(\predictions,{
		arg msg;
		msg[3].postln;
	},"/prediction");

	// Definimos el synth  
	
	{
		// Declaramos una variable sig
		// para la señal ue contendra la reproducción
		// del Buffer que contienen el sonido

		var sig = PlayBuf.ar(
			1,
			buf,
			BufRateScale.ir(buf),
			doneAction: 2
		);

		// Declaramos en la variable mfcss
		// el FluidMFCC. Este analisis timbral es el que 
		// pasaremos a la entrada de la red neuronal
		
		var mfccs = FluidMFCC.kr(
			sig,
			~nmfccs,
			startCoeff: 1,
			maxNumCoeffs: ~nmfccs
		);

		// Declaramos una variable para guardar
		// el resultado es el valor del loudness
		// en dBFS 
		
		var loudness = FluidLoudness.kr(sig)[0];

		// Guardamos en una variable Threshold
		// y la incializamos a -40 dB.

		var thresh = -40;

		// Declaramos una variable que contiene
		// un valor True o False, dependiendo
		// si el Loudness del sonido de entada
		// es mayor al threshold definido
		
		var isPredicting = (loudness >= thresh);

		// Definimos una variable trig que contiene
		// una señal Impulse que gatilla 30 impulsos
		// por segundos. 
		
		var trig = Impulse.kr(30);

		// Escribimos el flujo de datos de control
		// en un Buffer contebido en la variable ~mfccbuf

		
		FluidKrToBuf.kr(mfccs, ~mfccbuf);

		// Ingresamos el análisis timbral del sonido
		// en la red neuronal si el loudness es mayor
		// a -40dB. Los resultado es guardado
		// en el Buffer ~predictions 
		
		~nn.kr(
			trig * isPredicting,
			~mfccbuf, ~predictions
		);

		// Con el UGen SendReplay enviamos el array de datos
		// del Buffer de predicciones al mensaje /prediction
		// si ispredicting es true
		
		SendReply.kr(
			trig * isPredicting,
			"/prediction", FluidBufToKr.kr(~predictions)
		);

		// Con el UGen SendReplay enviamos el mensaje -1
		// cuando no estamos prediciendo
		
		SendReply.kr(
			trig * (1 - isPredicting),
			"/prediction", -1);

		// Duplicamos la señal de audio
		// para reproducir el sonido en
		// dos canales
		
		sig.dup;
		
	}.play;
};
)

// Hacemos correr la función para predecir con el
// Buffer de test del trombone
~server_predictions.(~trombone_test);


// Hacemos correr la función para predecir con el
// Buffer de test del oboe
~server_predictions.(~oboe_test);


~realtime_analysis.(~trombone_test);
~realtime_analysis.(~oboe_test);

// Ejecutando el código
(
~nn.predictPoint(~mfccbuf,{
arg label;
label.postln;
});
)


OSCdef.trace(true);
OSCdef.trace(false);


