
// PARTE 01. SUPERCOLLIDER

//BLOQUES DE CÓDIGO

// En Supercollider los bloque de código se definen con parentesis "()". Dentro de los parentesis podemoes definir variables que serán validas solo dentro del bloque.

(
var a, b;
var c = 0;

//declaracion de un array;

var milista = [1, 4, 6];
milista.postln;

// insertamos un nuevo elemento
milista.insert(milista.size, 100);

milista.postln;

)

// Las variables globales se definen con una Virgulilla antes del nombre. 

~variableGlobal = 0;

//Hay que tener en cuenta que cuando la variable no es un numero, no es global y se encuentra dentro de un bloque de código esta necesita ser deckarada como variable. 


// Las funciones se declaran de la siguiente forma.

(
var miFuncion;
miFuncion = { arg x;
	("mi número es: " + x).postln;
}
)

// donde arg es el argumento de la funcion. 
// En su forma global:

(
~miFuncion = { arg x;
	("mi número es: " + x).postln;
}
)

// Las funciones son llamadas de la siguiente forma:

~miFuncion.value(2);

// Un diccionario es una colleccion asociativa que crea un mapa entre keys y valores.

(
var d = Dictionary();
d.put(\a, 440);
d.keys; // Set[\a]
d.put(\a, nil); // removes the value 44
d.keys; // Set[]

// creacion de un diccionario desde una coleccion
d = Dictionary.newFrom([\a, 1, \b, 2, \c, 4]);

// desde un array de asociaciones
d = Dictionary.with(*[\a->1, \b->2, \c->3])

)


// PARTE 02. FLUCOMA Y CORPUS DE SONIDO

// 0. INICIAMOS EL SERVIDOR DE AUDIO

(
// Incrementamos el numero de Buses 
s.options.numBuffers = 1024 * 16;
//Incrememtamos la cantidad de memoria del servidor
s.options.memSize = 8192 * 64;
//Iniciamos el servidor
 s.boot;
)

// podemos iniciar el meter y el plot Tree
// si estamos reproduciendo el sonido
s.meter
s.plotTree;

// desde ahora definiremos variables locales que potencialmente e idealmente seria adecuado hacer todo de forma local en un solo bloque de código para hacer correr todo el código de una vez. Sin embargo, por motivos de desarrollo es conveniente comenzar por secciones separadas usando variable globales, por si existen errores. 

// 1. DEFINI|MOS EL DIRECTORIO DE TRABAJO

// Declaramos la variable global que contiene la ruta al directorio
// en que existen los archivos de sonido que usaremos.
//En este caso, es el directorio de sonido con que viene FluCoMa
//para ser usados en sus ejemplos.

~folder = FluidFilesPath();

// 2. CARGAMOS EL DIRECTORIO EN UN BUFFER

// Con una ruta dada, la clase "FluidLoadFolder" carga cada archivo de audio en la ruta, y los asigna a un Buffer cargando los archivos desde la cabeza a la cola en el buffer. La información sobre los puntos de inicio y fin, el número de canales y las tasas de muestreo se almacenan luego en el indice  de "FluidLoadFolder" como un "IdentityDictionary".

~loader = FluidLoadFolder(~folder).play(s,{"done loading folder".postln;});

// 3. STEREO A MONO

// comprobamos si los archivos son estereos
// si son estereo sumamos los dos canales y los
// convertimos a mono

(
// si el Buffer ~loader tiene mas de un canal 
if(~loader.buffer.numChannels > 1){
	// declaramos un Buffer de salída
	~src = Buffer(s);
	//Para cada mienbro del Buffer ~loader haga (do):
	~loader.buffer.numChannels.do{
		// argumento para contener el número de canales
		arg chan_i;

		// "FluidBufCompose" es una utilidad para manipular el contenido de buffers.
		// Este objeto es una herramienta de bajo nivel para manipular Buffers y su contenido.
		// Proporcionando el rango de muestras y canales para ser copiados,
		// asi como tambien, el Buffer destinatario, la copia de canales y la ganacia,

		//FluidBufCompose.processBlocking(server, source, startFrame: 0, numFrames: -1, startChan: 0, numChans: -1, gain: 1, destination, destStartFrame: 0, destStartChan: 0, destGain: 0, freeWhenDone: true, action)
		
		FluidBufCompose.processBlocking(s,
			~loader.buffer,
			startChan:chan_i,
			numChans:1,
			gain:~loader.buffer.numChannels.reciprocal,
			destination: ~src,
			destGain:1,
			action:{"copied channel: %".format(chan_i).postln}
		);
	};
}{
	"loader buffer is already mono".postln;
	~src = ~loader.buffer;
};
)

// 4. SLICE EL BUFFER EN TIEMPO NO REAL
(
// creamos un Buffer para contener los indices
// en los cuales se haran los slices
~indices = Buffer(s);

// "FluidBufOnsetSlice" implementa una selección de  detectores de evntos sonoros con los cuales se segmenta un archivo de audio. Esta clase segmenta un sonido en base a la diferencia spectral entre cuadros. En sonido digital nos referimos a un cuadro, al cada sample del sonido independiente de la cantidad de canales del archivo de audio. La metrica para calcular la diferencia espectral puede ser elejida desde una seleccion ya curada, de tal manera que se puede ejecutar la segmentacion de material sonoro con amplia gama de caracteristitas diferentes. El argumento con el que se define la metrica  puede ser pasado como número entero o simbolo: \power, \hfc, \flux, \mkl, \is, \cosine, \phase, \wphase, \complex, or \rcomplex. En nuestro ejemplo usamos el número entero 9 que corresponde a rcomplex.

//OJO ACA QUE HAY UNS SERIE DE METODOS PARA HACER LOS SLICES. VERLOS PORQUE DEPENDE DEL SOURCE Y LO QUE SE QUIERE CAPTURAR

FluidBufOnsetSlice.processBlocking(
	s,
	~src,
	metric: 9,
	threshold: 0.05,
	indices: ~indices,
	action: {
	"found % slice points".format(~indices.numFrames).postln;
	"average duration per slice: %".format(~src.duration / (~indices.numFrames+1)).postln;
});
)

// 5. ANALISIS DE LOS SEGMENTOS


(

// creamos una variable que contenga un "FluidDataSet"

~analyses = FluidDataSet(s);

// Pasamos los elementos del Buffer ~indices como un Array de números de punto flotante y procedemos a hacer la siguiente accion para cada uno de sus mienbres definiendo una funcion.

// NOTA: Es importante notar que en Supercollider los Buffer funcionan a nivel de servidor de sonido, mientras que los Array funcionan a nivel de lenguaje. De esta forma, en SC existen muchos métodos que nos permiten pasar datos desde el servidor de sonido al lenguaje. 

~indices.loadToFloatArray( action: {
	
	//creamos un argumento llamado fa, que contendra el Array de punto flotante.
	
	arg fa;

	//creamos tres Buffer mfccs para guardar los features de nuestro analisis, stats para guardar las estadisticas de nuestro analisis, y finalmente flat, para guradar el resltado del proceso de aplanar las estadisticas a un Buffer unidimensional. Esto último es comun particularmente cuando usamos contenedores de datos multidimensionales con machine learning y AI.
	
	var mfccs = Buffer(s);
	var stats = Buffer(s);
	var flat = Buffer(s);

	// Para cada elemento del Array (o Buffer). el método "doAdjcentpairs" llama a una funcion para cada adyacente par de elementos de una coleccion. El método nos entraga los dos elemnetos adyacentes de la coleccion "fa" son pasados a traves de los argumentos "start" y "end", los que representan el corte de cada segmento o onset encontrado. Debiso a esto un segmento tiene un largo en muestras o samples que corresponde al proximo punto de corte menos el actual punto de corte. Finalmente, "doAdjacentPairs" nos entrega el indice. 
	
	fa.doAdjacentPairs{
		arg start, end, i;

		// obtenemos el largo de cada segmento 
		var num = end - start;

		// "FluidBufMFCC" es un descriptor de timbral espectros, llamado Mel-Frequency Cepstral Coefficients (MFCCs). MFCC se refiere a "Mel-Frequency Cepstral Coefficients". Este analisis es generalmente usado para la descripcion y comparacion de timbres. El algoritmo comprime el espectro de un sonido en un pequeño numero de coeficientes que, cuando son manipulados juntos, describen un contorno general del espectro.

		// NOTA: Las  clases "Fluid" procesan el Buffer de entrada en el servidor de sonido. En estos casos es usado "processBlocking"  en vez de "process" debido a que ejecutará un trabajo que toma una gran cantidad de tiempo, por lo que es necesario capturar el servidor hasta cuando el proceso se termine. En cambio "process" usa otro hilo "thread" del procesador para que los otros procesos continuen. El problema aparece cuando códigoi posterior a la salida del proceso mientras este todavia no termina de ejecutarlo, lo que aroojaria un error. 

		FluidBufMFCC.processBlocking(
			s, 
			~src,
			start,
			num,
			features: mfccs, //Buffer de salida
			numCoeffs: 13, 
			startCoeff:1
		);

		// "FluidBufStats" sumariza estadisticamente una serie de momentos ( o cualquie tipo de valor) que es en un Buffer, devolviendo 7 estadisticas para cada canal: el promedio del Buffer por canal, la desviacion estandard, el skewness o asimetría, la kurtosis o la forma en que los datos están distribuidos, ademas de los valores bajos, los medios y altos. Todo estas dimensiones son guardadas en el Buffer "stats" 
	
		FluidBufStats.processBlocking(
			s,
			mfccs,
			stats: stats,
			select: [\mean]
		);

		// "FluidBufFlatten" aplana un Buffer multicanal en una dimension o canal. Esto se usa para reestructurar el Buffer de tal forma que puede ser agregado a un "FluidDataSet"
		
		FluidBufFlatten.processBlocking(
			s,
			stats,
			destination: flat //el Buffer de salida
		);

		//finalmente, para cada elemento de flat usando el indice i, agregamos un punto al Buffer analysis		
		~analyses.addPoint(i,flat);

		"analyzing slice % / %".format(i+1,fa.size-1).postln;

		// utilizamos este código para que el servidor se sincronice con el cliente del lenguaje cada 100 iteraciones
		//	if((i%100) == 99){s.sync;}
		//if(i % 100 == 1,{s.sync});
	};

	s.sync;

	~analyses.print;
});
)

// 6. REDUCCION DIMENSIONAL DEL FLUIDDATASET

// FluidUMAP reduce las dimensones de un FluidDataSet usando un algoritmo llamado UMAP (Uniform Manifold Approximation and Projection). Una vez hecho la reduccion lo guardamos en un FluidDataSet llamado ~umapped.

// La razon para hacer esta reduccionse debe a que nuestro cuerpo de datos proveniente del analisis t guardado en el FluidDataSet ~analysis tiene 1473 puntos y 13 dimensiones o columnas. Sin embargo, nosotros necesitamos que el cuerpo de datos sea contenido en dos dimenciones o columnas porque lo vamos a representar en 2 dimensiones, con coordenadas x, y. 

(
~umapped = FluidDataSet(s);


FluidUMAP(
	s,
	numNeighbours: 15,
	minDist: 0.9
).fitTransform(~analyses, ~umapped, action:{"umap done".postln});

)

~umapped.print;


// 6. NORMALIZACION DEL DATASET

// Normalizamos el data set y lo guardamos en un FluidDataSet llamado ~normed. El proceso de normalización refiere a hacer que todos los valores del data set tengan un rango de valores que va de entre 0 a 1. Esto es importante porque pordemos manipular mas facil los valores mantenuendo la relacion de los valores contenidos en una misma dimension.

(
~normed = FluidDataSet(s);
FluidNormalize(s).fitTransform(~umapped,~normed);
)

~normed.print;

// 7. FIT A KDtree

//Un árbol de k-dimensiones para búsquedas de vecindad eficientes de datos multi-dimensionales. FluidKDTree facilita búsquedas de vecinos más cercanos de datos multi-dimensionales almacenados en un FluidDataSet de manera eficiente.

// Creamos un KDtree con el dataset normalizado
~tree = FluidKDTree(s).fit(~normed);



// 8. CRAMOS UNA FUNCION PARA REPRODUCIR LOS SEGMENTOS DE SONIDO

(
~play_slice = {

	// el argumento es el indice

	arg index;
	{
		//Index into a table with a signal. The input signal value is truncated to an integer value and used as an index into the table. Out-of-range index values are clipped to the valid range.
		var startsamp = Index.kr(~indices, index);
		var stopsamp = Index.kr(~indices, index + 1);
		var phs = Phasor.ar(
			0,
			BufRateScale.ir(~src),
			startsamp,
			stopsamp);
		
		var sig = BufRd.ar(1,~src,phs);
		var dursecs = (stopsamp - startsamp) / BufSampleRate.ir(~src);
		var env;

		dursecs = min(dursecs,1);

		env = EnvGen.kr(
			Env([0,1,1,0], [0.03,dursecs-0.06,0.03]),
			doneAction:2
		);
		sig.dup * env;
	}.play;
};
)

// 9.A. DIBUJAMOS LOS DATOS

// Usando el metodo dump en el FluidDataSet normalizado. El metodo "dump" almacena los datos del FluidDataSet como un diccionario en un objeto "Dictionary". Tiene como argumento una funcion que se ejecuta cuando el servidor ha completado el traslado de los datos al diccionario.

(
~normed.dump({

	// declaramos el argumento dict que contendra el dictionario que almacenara los datos
	
	arg dict;

	//creamos una variable point que contendra un Buffer de dos canales y una variable previous inicializada con nil. 
	
	var point = Buffer.alloc(s,2);
	var previous = nil;

	//finalmente visualizamos el dictionario en la consola.
	
	dict.postln;

	// Creamos una función "defer" para crear la componente gráfica. El termino "defer" retrasa la evalución de la función por un delta en segundos. Esto es necesario cuando se usa QT o una función gráfica sino arroja un error.
	
	defer{

		// "FluidPlotter", es una clase de FluCoMa para visualizador de datos. Esta clase tiene un argumento "mouseMoveAction" con el cual podemos navegar por los datos usando el ratón. Manteniendo la tecla ALt y arrastrar el raton sobre una seccion del plotter producira un zoom en ese espacio.

		FluidPlotter(
			dict: dict,
			mouseMoveAction: {

				// Declaración argumento view,
				// x e y para las coordenadas del ratón

				arg view, x, y;
				[x,y].postln;
				
				// Agregamos las coordenadas x e y
				// del ratón al Buffer point
				// de dos canales canales
				// en el índice 0
				
				point.setn(0, [x, y]);
				
				// Usamos el método "kNearest" de
				// la instancia ~tree de la clase KPTree.
				// la que nos entrega el punto mas
				// cercano dado un punto x, y.
				
				~tree.kNearest(
					point, // Punto x, y al que buscaremos el elemento mas cercano
					1, // El número de vecinos que buscará
					{  // La acción a realizar (una función)

						// La función recibe como argumento
						// el elemento mas cercanos como argumento
						
						arg nearest;

						// Si el elemento "nearest" es distinto
						// al elemento "nearest" anterior encontrado? 
						if(nearest != previous){
							
							nearest.postln;

							// Highlight el elemento mas cercano
							view.highlight_(nearest);
							
							// Llamamos a la función "~play_slice" para
							// reproducir el segmento de audio
							// más cercano
							~play_slice.(nearest.asInteger);

							// Copiamos el índice del elemento
							// más cercano, a la variable "previous" 
							previous = nearest;
						}
					});
			});
	}
});
)

// Hasta acá hemos controlado la clasificación de sonidos con el ratón, pero si queremos controlar externamente el punto en donde navegaremos por el corpus de datos necesitamos cambiar el código, debido a que la clase "FluidPlotter" solo proporciona la función del parametro "action:" para cuando el ratón es activado.

// Para esto crearemos una clase gráfica llamada "UserView" que usaremos con fondo transparente en la cual dibujaremos un circulo.

// 9.B. DIBUJAMOS LOS DATOS CONTROL EXTERNO

// Creamos 

(

// Creamos variables para la función circle, para la ventana, para el UserView, para el Buffer point, para el punto previo y para el "FluidPlotter". 

var w, uv, fp;

var ypos = 0;
var xpos = 0;

// Reservamos el espacio de un Buffer de dos canales y
// lo asignamos a la variable point
var point = Buffer.alloc(s, 2);

// Inicializamos la variable previous con un valor nil o nulo. 
var previous = nil;

// Definimos la función que dibujará un círculo
// con tres argumentos, la coordenadas x e y
// y el radio del círculo

var circle = {
	|x, y, r|
	Pen.fillColor = Color.new255(255, 0, 0, 80);
	Pen.strokeColor = Color.red;
	Pen.addArc(x@y + ((r@r)), r, 0, 2pi);
	Pen.fillStroke;
};

// Creamos la ventana y la asignamos a la variable w
// de nombre "Monitor" y tamaño de 800 x 800 pixeles,

w = Window("Monitor", bounds: Rect(0, 0, 800, 800)).front;

// Definimos el color del fondo con total transparencia
w.view.background_(Color.hsv(1, 1, 1, alpha: 0));

// Con el FluidDataSet como diccionario, dibujamos los elementos
// con "FluidPlotter" sin la función action. Es decir sin la función
// que activa el ratón.

// Si no damos cuenta el primer parametro "FluidPlotter"
// es la ventana emparentada.

~normed.dump({
	arg dict;
	~normed_dict = dict;
	dict.postln;
	defer{	
		fp = FluidPlotter(
			w,
			dict: dict,
			xmin: 0,
			xmax: 1,
			ymin: 0,
			ymax: 1,
			bounds: Rect(0, 0, w.bounds.width, w.bounds.height));

		// Inicializamos la clase UserView y la asignamos a la variable uv
		// En la funcion "drawFunc" dibujamos el circulo llamado a label
		// función circle. Como están los valores del FluisDataSet estan
		// normalizados, los multiplicamos por el ancho y el alto de la ventana.
		// El radio es 10.
		
		uv = UserView(w, Rect(0, 0, w.bounds.width, w.bounds.height))
		.drawFunc = { |uvview|
			circle.value(
				xpos * w.bounds.width,
				ypos * w.bounds.height,
				10
			);
		};
	}
});

// La funcion OSCdef 

OSCdef(\xy, {
	arg msg;

	msg.postln;


	// Descomentar para usar sin snap
	// el punto al elemento más cercano
	
	/*
	xpos = msg[1];
	ypos = msg[2];
	
	fork{
		point.setn(0, [xpos, ypos]);
		~tree.kNearest(point, 1, {
			arg nearest;
			if(nearest != previous){
				nearest.postln;
				~play_slice.(nearest.asInteger);
				previous = nearest;
			};
		};)
	};
	
	defer{uv.refresh};
	*/

	// Descomentar para usar snap
	// el punto al emeneto más cercano

	
	point.setn(0, [msg[1], msg[2]]);	
	~tree.kNearest(point, 1, {
		arg nearest;
		if(nearest != previous){
			//nearest.asString.postln;
			# xpos, ypos = ~normed_dict["data"][nearest.asString];
			nearest.postln;
			~play_slice.(nearest.asInteger);
			defer{ uv.refresh };
			previous = nearest;
		};
	};
	);

}, '/3/xy');
)


// TEST PARA GENERAR OSC MESSAGES CON NUMEROS AL AZAR

(
m = NetAddr.new("localhost", 57120);
~rndmsg = Routine{loop {
	m.sendMsg("/3/xy", 1.0.rand, 1.0.rand);
	1.wait;
}};

~rndmsg.play;
)

~rndmsg.stop;


// FUNCION USANDO PYTHON MEDIA PIPE

// Vamos a navegar a traves de gestos. Le asignameros a cada gesto de cada mano un número que sume o reste a las variables xpos e ypos 


// 9.C. DIBUJAMOS LOS DATOS CONTROL EXTERNO USANDO GESTOS



(
var uv, fp, getNearest;
var txpos = 0.5;
var typos = 0.5;
var handedness0 = nil;
var handedness1 = nil;
var ypos = 0;
var xpos = 0;
var point = Buffer.alloc(s, 2);
var previous = nil;

var circle = {
	|x, y, r|
	Pen.fillColor = Color.new255(255, 0, 0, 80);
	Pen.strokeColor = Color.red;
	Pen.addArc(x@y + ((r@r)), r, 0, 2pi);
	Pen.fillStroke;
};

var w = Window("Monitor", bounds: Rect(0, 0, 800, 800)).front;
//var uv = UserView(w, Rect(0, 0, w.bounds.width, w.bounds.height));

w.view.background_(Color.hsv(1, 1, 1, alpha: 0));

~normed.dump({
	arg dict;
	~normed_dict = dict;
	dict.postln;
	defer{	
		fp = FluidPlotter(
			w,
			dict: dict,
			xmin: 0,
			xmax: 1,
			ymin: 0,
			ymax: 1,
			bounds: Rect(0, 0, w.bounds.width, w.bounds.height));
		uv = UserView(w, Rect(0, 0, w.bounds.width, w.bounds.height))
		.drawFunc = { |uvview|
			circle.value(
				xpos * w.bounds.width,
				(1 - ypos) * w.bounds.height,
				10
			);
		};
	}
});

/*
getNearest = { arg xx, yy;
	point.setn(0, [xx, yy]);	
	~tree.kNearest(point, 1, {
		arg nearest;
		if(nearest != previous){
			//nearest.asString.postln;
			# xpos, ypos = ~normed_dict["data"][nearest.asString];
			nearest.postln;
			~play_slice.(nearest.asInteger);
			defer{ uv.refresh };
			previous = nearest;
		};
	};
	);	
};
	*/		

// La funcion OSCdef 

OSCdef(\gesture0, {
	arg msg;
	//	msg.postln;
	if(handedness0 == 'Right'){
		if(msg[1] == 'Closed_Fist', {
			txpos = txpos + 0.001;
			("X temporal Value: " + txpos).postln});
		if(msg[1] == 'Open_Palm', {
			txpos = txpos - 0.001;
			("X temporal Value: " + txpos).postln});
		if(msg[1] == 'Pointing_Up', {
			txpos = txpos + 0.01;
			("X temporal Value: " + txpos).postln});
		if(msg[1] == 'Victory', {
			txpos = txpos - 0.01;
			("X temporal Value: " + txpos).postln});
		if(msg[1] == 'Thumb_up', {
			txpos = rrand(1.0);
			("X temporal Value: " + txpos).postln});
	}{
		if(msg[1] == 'Closed_Fist', {
			typos = typos + 0.001;
			("Y temporal Value: " + typos).postln});
		if(msg[1] == 'Open_Palm', {
			typos = typos - 0.001;
			("Y temporal Value: " + typos).postln});
		if(msg[1] == 'Pointing_Up', {
			typos = typos + 0.01;
			("Y temporal Value: " + typos).postln});
		if(msg[1] == 'Victory', {
			typos = typos - 0.01;
			("Y temporal Value: " + typos).postln});
		if(msg[1] == 'Thumb_up', {
			typos = rrand(1.0);
			("Y temporal Value: " + typos).postln});
	};
	
	point.setn(0, [txpos, typos]);	
	~tree.kNearest(point, 1, {
		arg nearest;
		if(nearest != previous){
			//nearest.asString.postln;
			# xpos, ypos = ~normed_dict["data"][nearest.asString];
			nearest.postln;
			~play_slice.(nearest.asInteger);
			defer{ uv.refresh };
			previous = nearest;
		};
	};
	);
	
	
	/*
	{
	getNearest.value(txpos, typos);
	};
	*/


	
}, '/gesture/0');


OSCdef(\gesture1, {
	arg msg;
	if(handedness1 == 'Right'){
		if(msg[1] == 'Closed_Fist', {
			txpos = txpos + 0.001;
			("X temporal Value: " + txpos).postln});
		if(msg[1] == 'Open_Palm', {
			txpos = txpos - 0.001;
			("X temporal Value: " + txpos).postln});
		if(msg[1] == 'Pointing_Up', {
			txpos = txpos + 0.01;
			("X temporal Value: " + txpos).postln});
		if(msg[1] == 'Victory', {
			txpos = txpos - 0.01;
			("X temporal Value: " + txpos).postln});
		if(msg[1] == 'Thumb_up', {
			txpos = rrand(1.0);
			("X temporal Value: " + txpos).postln});
	}{
		if(msg[1] == 'Closed_Fist', {
			typos = typos + 0.001;
			("Y temporal Value: " + typos).postln});
		if(msg[1] == 'Open_Palm', {
			typos = typos - 0.001;
			("Y temporal Value: " + typos).postln});
		if(msg[1] == 'Pointing_Up', {
			typos = typos + 0.01;
			("Y temporal Value: " + typos).postln});
		if(msg[1] == 'Victory', {
			typos = typos - 0.01;
			("Y temporal Value: " + typos).postln});
		if(msg[1] == 'Thumb_up', {
			typos = rrand(1.0);
			("Y temporal Value: " + typos).postln});
		};
	
	point.setn(0, [txpos, typos]);	
	~tree.kNearest(point, 1, {
		arg nearest;
		if(nearest != previous){
			//nearest.asString.postln;
			# xpos, ypos = ~normed_dict["data"][nearest.asString];
			nearest.postln;
			~play_slice.(nearest.asInteger);
			defer{ uv.refresh };
			previous = nearest;
		};
	};
	);
	
	/// EL VALUE!!!!
	
	/*
	{
		getNearest.value(txpos, typos);
	}
	*/
	
	//	("GESTURE 1: " + msg[1]).postln;
}, '/gesture/1');



OSCdef(\handedness0, {
	arg msg;

	handedness0 = msg[1];

	//	("HANDEDNESS 0: " + msg[1]).postln;
}, '/handedness/0');

OSCdef(\handedness1, {
	arg msg;

	handedness0 = msg[1];
	
	//	("HANDEDNESS 1: " + msg[1]).postln;
}, '/handedness/1');



)