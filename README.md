# Proyecto-simulacion
Bienvenido a Climas Locos, su herramienta digital para simular y predecir el comportamiento futuro de las temperaturas basándose en datos históricos. Esta guía le explicará paso a paso cómo operar el programa.
#Si quiere ver las imagenes tendra que sacar del rar las imagenes y  modificar las lineas 39 a la 42 en el code con su propia ruta
NOMBRE_IMAGEN = r'C:\Users\LENOVO\Desktop\RECETAS PROMODEL\imagen1.png'
RUTA_EMOJI_CALIENTE = r'C:\Users\LENOVO\Desktop\RECETAS PROMODEL\EMOJICALIENTE.jpg'
RUTA_EMOJI_FRIO = r'C:\Users\LENOVO\Desktop\RECETAS PROMODEL\EMOJIFRIO.jpg'
RUTA_EMOJI_MEDIA = r'C:\Users\LENOVO\Desktop\RECETAS PROMODEL\EMOJIMEDIA.jpg'

1. Pantalla de Inicio y Carga de Datos
Al abrir el programa, verá la pantalla de bienvenida con un fondo y el título del proyecto. Antes de poder realizar cualquier predicción, el sistema necesita cargar una base de datos.

Cargar Base de Datos:

Localice el botón "CARGAR BASE" en la parte derecha de la pantalla.

Haga clic en él. Se abrirá una ventana para buscar archivos.

Seleccione el archivo de registros (formato .csv) que contiene la historia climática.


Ingresar al Sistema:

Una vez cargada la base, haga clic en el botón "INICIO" (a la izquierda) para entrar al panel de control principal.

2. Panel de Predicción
Esta es la zona de trabajo principal. Aquí podrá configurar qué fecha desea consultar. El panel se divide en Controles (arriba) y Resultados (abajo).

¿Cómo hacer una consulta?
Selecciona el año que desea predecir, si quiere predecir mensualmente aparte del año el mes, si quiere predecir un dia especifico seleccione el dia

A. Predicción Detallada (Por Día)
Ideal para saber el clima exacto de una fecha específica.

Seleccionar Año: Despliegue la lista "Año" y elija uno (ej. 2025).

Seleccionar Mes: Elija el mes deseado.

Seleccionar Día: Elija el día específico.

Ejecutar: Presione el botón grande "Predecir".

B. Proyección General (Promedio Mensual)
Ideal para ver la tendencia de todo un año completo (ej. "¿Cómo se comportará el clima durante todo el 2030?").

Seleccionar Año: Elija el año que le interesa.

Dejar Mes y Día en "Seleccionar": No elija ningún mes ni día específico.

Botón Especial: Notará que aparece un botón nuevo a la izquierda llamado "PROMEDIO MENSUAL". Haga clic ahí.

C. Predicción Mensual (Por Mes)

Seleccionar Año: Despliegue la lista "Año" y elija uno (ej. 2025).

Seleccionar Mes: Elija el mes deseado.

Ejecutar: Presione el botón grande "Predecir".

D. Predicción Anual (Por año)

Seleccionar Año: Despliegue la lista "Año" y elija uno (ej. 2025).

Ejecutar: Presione el botón grande "Predecir".

3. Interpretación de Resultados
Dependiendo de su consulta, el sistema le mostrará diferente información:

En la Vista Diaria (Gráfica)
Gráfica Central: Verá una curva de temperaturas.

Línea Roja: Representa la Temperatura Máxima (calor).

Línea Azul: Representa la Temperatura Mínima (frío).

En el centro aparecerá una imagen indicando la sensación térmica del día:

❄️ Frio: Si el promedio es menor a 15°C.

🌿 Templado: Si está entre 15°C y 25°C.

🔥 Cálido: Si supera los 26°C.

Panel de Texto (Derecha): Muestra los valores numéricos exactos y la explicación del modelo matemático utilizado (SARIMA/Gumbel) y los riesgos de eventos extremos.

En la Vista Mensual (Tabla)
Verá una Tabla de Datos que lista los 12 meses del año seleccionado.

Cada mes muestra un valor proyectado calculado mediante tendencias polinomiales (Método CANSA).

Podrá ver los coeficientes de la ecuación matemática usada para ese cálculo.

4. Navegación y Salida
Regresar: Si está en la vista mensual, use el botón "Regresar a Gráfica" para volver al modo normal.

Volver al Inicio: El botón "Regresar al Inicio" en la parte superior derecha le llevará a la portada (útil si desea cargar una base de datos diferente).

Salir: En la pantalla de bienvenida, use el botón "Salir" para cerrar el programa definitivamente.
Proyecto climas locos
