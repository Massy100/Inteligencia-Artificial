using System;

namespace Perceptron
{ // compuertas logicas AND, OR y XOR
    class Program{
        static void Main(string[] args)
        {
            Random aleatorio = new Random();
            int salidaInt;
            
            // COMPUERTA AND
            Console.WriteLine("=== COMPUERTA AND ===");
            int[,] datosAND = {{1,1,1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
            double[] pesosAND = {aleatorio.NextDouble(), aleatorio.NextDouble(), aleatorio.NextDouble()};
            bool aprendizaje = true;
            int epocasAND = 0;
            
            while (aprendizaje)
            {
                aprendizaje = false;
                for (int i = 0; i < 4; i++)
                {
                    double salidaDoub = datosAND[i,0] * pesosAND[0] + datosAND[i,1] * pesosAND[1] + pesosAND[2];
                    if (salidaDoub > 0) salidaInt = 1; else salidaInt = 0;
                    if (salidaInt != datosAND[i, 2])
                    {
                        pesosAND[0] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        pesosAND[1] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        pesosAND[2] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        aprendizaje = true;
                    }
                }
                epocasAND++;
            }
            
            // Pruebas AND
            for (int i = 0; i < 4; i++)
            {
                double salidaDoub = datosAND[i,0] * pesosAND[0] + datosAND[i,1] * pesosAND[1] + pesosAND[2];
                if (salidaDoub > 0) salidaInt = 1; else salidaInt = 0;
                Console.WriteLine($"ENTRADAS: {datosAND[i,0]} AND {datosAND[i,1]} = {datosAND[i,2]} | PERCEPTRON = {salidaInt}");
            }
            Console.WriteLine("EPOCAS: " + epocasAND.ToString());
            Console.WriteLine($"PESOS UTILES: w0 = {pesosAND[0]} w1 = {pesosAND[1]} bias = {pesosAND[2]}");
            
            Console.WriteLine("\n=== COMPUERTA OR ===");
            
            // COMPUERTA OR
            int[,] datosOR = {{1,1,1}, {1, 0, 1}, {0, 1, 1}, {0, 0, 0}};
            double[] pesosOR = {aleatorio.NextDouble(), aleatorio.NextDouble(), aleatorio.NextDouble()};
            aprendizaje = true;
            int epocasOR = 0;
            
            while (aprendizaje)
            {
                aprendizaje = false;
                for (int i = 0; i < 4; i++)
                {
                    double salidaDoub = datosOR[i,0] * pesosOR[0] + datosOR[i,1] * pesosOR[1] + pesosOR[2];
                    if (salidaDoub > 0) salidaInt = 1; else salidaInt = 0;
                    if (salidaInt != datosOR[i, 2])
                    {
                        pesosOR[0] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        pesosOR[1] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        pesosOR[2] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        aprendizaje = true;
                    }
                }
                epocasOR++;
            }
            
            // Pruebas OR
            for (int i = 0; i < 4; i++)
            {
                double salidaDoub = datosOR[i,0] * pesosOR[0] + datosOR[i,1] * pesosOR[1] + pesosOR[2];
                if (salidaDoub > 0) salidaInt = 1; else salidaInt = 0;
                Console.WriteLine($"ENTRADAS: {datosOR[i,0]} OR {datosOR[i,1]} = {datosOR[i,2]} | PERCEPTRON = {salidaInt}");
            }
            Console.WriteLine("EPOCAS: " + epocasOR.ToString());
            Console.WriteLine($"PESOS UTILES: w0 = {pesosOR[0]} w1 = {pesosOR[1]} bias = {pesosOR[2]}");
            
            Console.WriteLine("\n=== COMPUERTA XOR ===");
            Console.WriteLine("NOTA: XOR no es linealmente separable, un solo perceptron no puede resolverlo");
            
            // COMPUERTA XOR (NO ES LINEALMENTE SEPARABLE)
            int[,] datosXOR = {{1,1,0}, {1, 0, 1}, {0, 1, 1}, {0, 0, 0}};
            double[] pesosXOR = {aleatorio.NextDouble(), aleatorio.NextDouble(), aleatorio.NextDouble()};
            aprendizaje = true;
            int epocasXOR = 0;
            int maxEpocas = 10000; // Límite para evitar bucle infinito
            
            while (aprendizaje && epocasXOR < maxEpocas)
            {
                aprendizaje = false;
                for (int i = 0; i < 4; i++)
                {
                    double salidaDoub = datosXOR[i,0] * pesosXOR[0] + datosXOR[i,1] * pesosXOR[1] + pesosXOR[2];
                    if (salidaDoub > 0) salidaInt = 1; else salidaInt = 0;
                    if (salidaInt != datosXOR[i, 2])
                    {
                        pesosXOR[0] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        pesosXOR[1] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        pesosXOR[2] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        aprendizaje = true;
                    }
                }
                epocasXOR++;
            }
            
            // Pruebas XOR
            for (int i = 0; i < 4; i++)
            {
                double salidaDoub = datosXOR[i,0] * pesosXOR[0] + datosXOR[i,1] * pesosXOR[1] + pesosXOR[2];
                if (salidaDoub > 0) salidaInt = 1; else salidaInt = 0;
                Console.WriteLine($"ENTRADAS: {datosXOR[i,0]} XOR {datosXOR[i,1]} = {datosXOR[i,2]} | PERCEPTRON = {salidaInt}");
            }
            Console.WriteLine("EPOCAS: " + epocasXOR.ToString());
            Console.WriteLine($"PESOS UTILES: w0 = {pesosXOR[0]} w1 = {pesosXOR[1]} bias = {pesosXOR[2]}");
            
            if (epocasXOR >= maxEpocas)
            {
                Console.WriteLine("El perceptron no pudo aprender XOR (no es linealmente separable)");
            }
            
            Console.ReadLine();
        }
    }
}