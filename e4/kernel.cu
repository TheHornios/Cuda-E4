/**
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* Entrega 4
*
* Alumno: Rodrigo Pascual Arnaiz y Villar Solla, Alejandro
* Fecha: 11/11/2022
*
*/
///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
///////////////////////////////////////////////////////////////////////////
// defines
///////////////////////////////////////////////////////////////////////////

// declaracion de funciones
// HOST: funciones llamadas desde el host y ejecutada en el host

/**
* Funcion: propiedadesDispositivo
* Objetivo: Mustra las propiedades del dispositvo, esta funcion
* es ejecutada llamada y ejecutada desde el host
*
* Param: INT id_dispositivo -> ID del dispotivo
* Return: cudaDeviceProp -> retorna el onjeto que tiene todas las
* propiedades del dispositivo CUDA
*/
__host__ cudaDeviceProp propiedadesDispositivo(int id_dispositivo)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, id_dispositivo);
	// calculo del numero de cores (SP)
	int cuda_cores = 0;
	int multi_processor_count = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	char* arquitectura = (char*)"";
	switch (major)
	{
	case 1:
		//TESLA
		cuda_cores = 8;
		arquitectura = (char*)"TESLA";
		break;
	case 2:
		//FERMI
		arquitectura = (char*)"FERMI";
		if (minor == 0)
			cuda_cores = 32;
		else
			cuda_cores = 48;
		break;
	case 3:
		//KEPLER
		arquitectura = (char*)"KEPLER";
		cuda_cores = 192;
		break;
	case 5:
		//MAXWELL
		arquitectura = (char*)"MAXWELL";
		cuda_cores = 128;
		break;
	case 6:
		//PASCAL
		arquitectura = (char*)"PASCAL";
		cuda_cores = 64;
		break;
	case 7:
		//VOLTA
		arquitectura = (char*)"VOLTA";
		cuda_cores = 64;
		break;
	case 8:
		//AMPERE
		arquitectura = (char*)"AMPERE";
		cuda_cores = 128;
		break;
	default:
		arquitectura = (char*)"DESCONOCIDA";
		//DESCONOCIDA
		cuda_cores = 0;
		printf("!!!!!dispositivo desconocido!!!!!\n");
	}
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", id_dispositivo, deviceProp.name);
	printf("***************************************************\n");
	printf("> Capacidad de Computo \t\t\t: %d.%d\n", major, minor);
	printf("> Arquitectura CUDA \t\t\t: %s \n", arquitectura);
	printf("> No. de MultiProcesadores \t\t: %d \n",
		multi_processor_count);
	printf("> No. de CUDA Cores (%dx%d) \t\t: %d \n", cuda_cores,
		multi_processor_count, cuda_cores*
		multi_processor_count);
	printf("> No. max. de Hilos (por bloque) \t: %d \n",
		deviceProp.maxThreadsPerBlock);
	printf("***************************************************\n");

	return deviceProp;
}

/**
* Funcion: mostrarArray
* Mostrar el valor que tiene un array de una dirección
*
* @param int *array -> Array a mostrar
* @param int tam -> Tamaño del array
*/
__host__ void mostrarArray(int* array, int tam)
{
	for (int i = 0; i < tam; i++)
	{
		printf("%i ", array[i]);
	}
	
}


/**
* Funcion: pedirUnNumero
* Pedir un numero entre un rango
*
* @param char* texto -> Texto a mostrar
* @param int minimo -> Número minimo
* @param int maximo -> Número maximo
* @return int -> Número instroducido por el usuario
*/
__host__ int pedirUnNumero(char* texto, int minimo, int maximo)
{
	// Comenzamos en con el valor minimo menos uno pra que siempre se muestre una vez
	int resultado = minimo -1 ;

	// Preguntamos hasta obtener un valor correcto
	while (resultado < minimo || resultado > maximo)
	{
		// Mostramos el texto por pantalla
		printf("%s: ", texto);

		// Leemos el valor introducido por un usuario
		scanf("%i", &resultado);

		// En el caso de que el numero del usuario supere al numero minimo 
		if (resultado < minimo)
		{
			printf("\nEl valor introducido no puede ser menor de %i", minimo);
		}

		//  En el caso de que el numero del usuario supere al numero maximo
		if (resultado > maximo)
		{
			printf("\nEl valor introducido no puede ser mayor de %i", maximo);
		}

	}
	return resultado;
}

/**
* Funcion: rellenarArrayAleatorio
* Llenar el array de forma alezatoria
*
* @param int *arr  -> Array que hay que rellenar 
* @param int tam -> Tamaño del array
*/
__host__ void rellenarArrayAleatorio(int* arr, int tam)
{
	for (int i = 0; i < tam; i++)
	{
		// El numero tiene que estar entre 1 y 31 
		arr[i] = (int)(rand() % 30 + 1 );
	}
}


// declaracion de funciones
// DEVICE

/**
* Funcion: ordenacionPorRango
* Función que ordena un array utilizando el algoritmo de ordenacion por rango
*
* @param int* original -> array original que hay que ordenar 
* @param int* final -> Array ordenado que devuelve el valor
* @param int size -> Tamaño del array 
*/
__global__ void ordenacionPorRango(int* original, int* final, int tam)
{
	int rango = 0;
	for (int i = 0; i < tam; i++)
	{
		if (original[threadIdx.x] > original[i])
		{
			rango++;
		}
		if (original[threadIdx.x] == original[i] && threadIdx.x > i)
		{
			rango++;
		}
		final[rango] = original[threadIdx.x];
	}
}

///////////////////////////////////////////////////////////////////////////
// MAIN: Inicio del programa
int main(int argc, char** argv)
{
	// Para que cada ejecucion sea aleatoria 
	srand(time(NULL));

	// Declaración de variables
	int* hst_original, * hst_final; // Array en el hots 
	int* dev_original, * dev_final; // Array en el device 
	int tamanyo, numero_dispositivos; // Tamño del los arrays y numero de dispositivos de ejecucion CUDA 
	float elapsedTime; // Elipsis de tiempo 
	cudaDeviceProp props; // Guardar propiedades
	// Declaración de eventos
	cudaEvent_t inicio;
	cudaEvent_t fin;

	// buscando dispositivos
	cudaGetDeviceCount(&numero_dispositivos);

	if (numero_dispositivos != 0)
	{
		for (int i = 0; i < numero_dispositivos; i++)
		{
			props = propiedadesDispositivo(i);
		}
	}
	else
	{
		printf("!!!!!ERROR!!!!!\n");
		printf("Este ordenador no tiene dispositivo de ejecucion CUDA\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	}

	
	// Creación de eventos
	cudaEventCreate(&inicio);
	cudaEventCreate(&fin);

	// Preguntar por el numero de elementos a ordenar 
	tamanyo = pedirUnNumero("\nElige una cantidad de elementos para el vector", 0, props.maxThreadsPerBlock);

	// Asignación de espacio a las variables en el host
	hst_original = (int*)malloc(tamanyo * sizeof(int));
	hst_final = (int*)malloc(tamanyo * sizeof(int));

	// Asignación de espacio a las variables en el device
	cudaMalloc((void**)&dev_original, tamanyo * sizeof(int));
	cudaMalloc((void**)&dev_final, tamanyo * sizeof(int));

	// Llenar array con números aleatorios entre 1 y 31
	rellenarArrayAleatorio(hst_original, tamanyo);

	// Copiar datos al dispositivo
	cudaMemcpy(dev_original, hst_original, sizeof(int) * tamanyo,cudaMemcpyHostToDevice);

	
	cudaEventRecord(inicio, 0); // Iniciamos el evento de inicio a 0
	ordenacionPorRango <<<1, tamanyo >>> (dev_original, dev_final, tamanyo); 	// Ordenar por rango
	cudaEventRecord(fin, 0); // Iniciamos el evento de fin a 0

	// Sincronizar Eventos
	cudaEventSynchronize(fin);

	// Traer datos del device
	cudaMemcpy(hst_final, dev_final, sizeof(int) * tamanyo,cudaMemcpyDeviceToHost);
	// Calcular tiempo de ejecucion
	cudaEventElapsedTime(&elapsedTime, inicio, fin);

	// Mostrar tiempo de ejecución por pantalla
	printf("> Kernel de %i bloque con %i hilos (%i hilos)\n", 1, tamanyo, 1* tamanyo);
	printf("> Tiempo Ejecucion:\t%f ms\n", elapsedTime);


	// Mostrar arrays
	printf("> VECTOR INCIAL:\n");
	mostrarArray(hst_original, tamanyo);
	printf("\n> VECTOR ORDENADO:\n");
	mostrarArray(hst_final, tamanyo);
	printf("\n");

	// Destruimos los eventos
	cudaEventDestroy(inicio);
	cudaEventDestroy(fin);

	printf("***************************************************\n");
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;



} 
