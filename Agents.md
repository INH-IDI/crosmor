# CROSMOR 2026 — AGENTS.md: Registro de Arquitectura, Tareas y Guía para Agentes

Este documento describe la transformación completa del modelo morfodinámico costero **CROSMOR** (originalmente programado en Fortran 77 por el Prof. Leo C. van Rijn, versiones 1998–2010) a una arquitectura moderna en **Python 3.12 / PyQt6 / Numba / NumPy**, detallando los hitos ejecutados, las decisiones de diseño y las instrucciones para futuros agentes y desarrolladores.

---

## 1. Visión General del Proyecto

- **Repositorio**: `https://github.com/INH-IDI/crosmor`
- **Ubicación local**: `crosmor_app/` (dentro de `/home/model2/Documents/Hulp/MODELLING PACKAGE 2013/CROSMOR2010/`)
- **Objetivo**: Convertir el código legacy monolítico en una suite profesional de modelación morfodinámica con:
  - Motor de cálculo numérico acelerado por vectorización y compilación JIT (`@njit`).
  - Interfaz gráfica de escritorio (GUI) moderna, reactiva y desacoplada mediante hilos (`QThread`).
  - Visualización sincronizada de perfiles, hidrodinámica, transporte y concentraciones en tiempo real.
  - Compatibilidad bidireccional total con archivos de entrada (`.inp`) y salida (`.out`) legacy, más formatos modernos (JSON, Excel `.xlsx`, CSV).

---

## 2. Arquitectura del Código y Equivalencia Fortran → Python

El código original en Fortran 77 dependía de arrays estáticos globales (`common /room/ a(220000)`), archivos scratch de disco en binario sin formato (`units 21, 22, 32, 43, 44`) y subrutinas acopladas. En `crosmor_app/src/crosmor/` se estructuró en capas desacopladas:

```
crosmor_app/
├── src/crosmor/
│   ├── models/              # Dataclasses tipadas (Pydantic / dataclass)
│   │   ├── config.py        # Configuración completa del caso de simulación
│   │   └── results.py       # Estructuras de resultados espacio-temporales
│   ├── numerics/            # Solucionadores matemáticos fundamentales
│   │   ├── ode_rk.py        # Runge-Kutta orden 1 a 4 (RK.F, RK1ST.F, COMP.F, FKT.F)
│   │   ├── root_finding.py  # Algoritmo de Brent/Zeroin (ZEROIN.F, TOLX.F, TOLV.F)
│   │   ├── smoothing.py     # Filtros espaciales y de fondo (SMOOTH.F, SMOOP.F)
│   │   └── tridiagonal.py   # Algoritmo de Thomas tridiagonal acelerado (TRIDIAR.F)
│   ├── hydrodynamics/       # Hidrodinámica costera
│   │   ├── wave_solver.py   # Balance de acción de oleaje y roller (golf.f)
│   │   ├── dispersion.py    # Relación de dispersión y longitudes de onda (FWL.F, DISP.F)
│   │   ├── rayleigh.py      # Espectro Rayleigh y alturas características (RAYL.F)
│   │   ├── longshore.py     # Corrientes longitudinales y undertow (SNELH.F, FVV.F)
│   │   └── statistics.py    # Momentos estadísticos del oleaje (STAT.F, ORDER.F)
│   ├── morphodynamics/      # Transporte y morfología
│   │   ├── bedload.py       # Transporte de fondo con sub-pasos intra-ola (BODEM.F, @njit)
│   │   ├── suspended.py     # Concentración de referencia y transporte en suspensión (SUSPEN.F)
│   │   ├── beach.py         # Zona de swash, runup y acople de playa seca (BEACH.F)
│   │   ├── morpho_solver.py # Conservación de masa de sedimento y evolución del perfil (TRANSP.F)
│   │   └── sediment_props.py# Densidad, viscosidad, sedimentación y curvas de arrastre (CALC.F, FDSTER.F)
│   ├── io/                  # Lectura y exportación
│   │   ├── inp_reader.py    # Lector robusto de archivos .inp legacy (bloques 001 a 011)
│   │   ├── inp_writer.py    # Escritor de archivos .inp con formato exacto
│   │   ├── out_reader.py    # Lector de archivos .out tabulados legacy
│   │   └── exporters.py     # Exportadores a Excel (.xlsx), CSV y JSON
│   ├── engine/              # Orquestación de la simulación
│   │   ├── runner.py        # Ciclo temporal, paso adaptativo/fijo y emisión de señales
│   │   └── cli.py           # Interfaz de línea de comandos para simulación en lote
│   └── gui/                 # Interfaz gráfica de usuario en PyQt6
│       ├── main_window.py   # Ventana principal, menús, integración y responsividad
│       ├── worker.py        # QThread asíncrono para ejecutar el runner sin congelar la GUI
│       ├── tabs/            # Editores con scroll para configuración
│       │   ├── tab_grid.py     # Pestaña 1: Malla y Batimetría
│       │   ├── tab_forcing.py  # Pestaña 2: Oleaje y Marea
│       │   ├── tab_hydro.py    # Pestaña 3: Switches hidrodinámicos y coeficientes
│       │   └── tab_sediment.py # Pestaña 4: Fracciones de sedimento, factime y dt fijo
│       └── plots/           # Dashboard gráfico en tiempo real
│           ├── dashboard.py    # 5 paneles interactivos, presets (3 de 5) y sincronización
│           └── player_bar.py   # Barra de ejecución (Play/Pause/Stop/Slider de tiempo)
└── tests/                   # Suite de pruebas automatizadas (20/20 pasando)
    ├── regression/          # Golden Master tests comparando con Dgoot1.out y Atest1.out
    ├── test_io.py           # Validación de lectura/escritura .inp/.out/JSON/Excel
    ├── test_numerics.py     # Validación de algoritmos numéricos
    ├── test_physics.py      # Validación física de fórmulas hidrodinámicas y de transporte
    └── test_runner_and_gui.py# Integración de la simulación, runner y componentes GUI
```

---

## 3. Registro Cronológico de Tareas y Mejoras Ejecutadas

### Hito 1: Creación del Proyecto y Subida Inicial a GitHub
- **Commit**: `b5ea5ed`
- **Descripción**: Estructuración completa del paquete `crosmor_app` con pyproject.toml, dependencias (PyQt6, pyqtgraph, numba, numpy, scipy, openpyxl), tests unitarios e inicialización del repositorio remoto en `https://github.com/INH-IDI/crosmor`.

---

### Hito 2: Corrección de Estaciones de Sedimento y Error de Broadcasting
- **Commit**: `c77877e`
- **Problema**: El usuario reportó el error `could not broadcast input array from shape (3,) into shape (0,)` al editar las estaciones en la pestaña 4.
- **Causa**:
  1. Si la tabla de estaciones de sedimento estaba vacía o tenía un solo punto, la interpolación espacial de fracciones devolvía arrays vacíos `(0,)`.
  2. Había una discrepancia entre el número de fracciones de sedimento del caso global y las columnas interpoladas.
- **Solución**:
  - Implementación de un fallback homogéneo cuando la tabla está vacía o tiene un único punto.
  - Validación de dimensiones en `crosmor.hydrodynamics.wave_solver.build_grid` y en `TabSediment`.
  - Agregado de validación automática en la interfaz antes de lanzar el cálculo.

---

### Hito 3: Paso de Tiempo Numérico (Criterio CFL vs. Modo dt Fijo)
- **Commit**: `d524f9e`
- **Consulta del usuario**: ¿Cómo estima el dt? ¿Usa un criterio CFL? ¿Por qué `factime` altera el resultado y se puede fijar un `dt`?
- **Análisis físico/numérico**:
  - En el modelo original de Leo van Rijn no se usaba un CFL estricto para morfología, sino una **tabla empírica heurística** dependiente del transporte neto y pendiente, escalada por el parámetro multiplicador `factime` (por defecto `dt_tab = [10, 30, 60, 180, 300, 600, 1200] * factime`). Al cambiar `factime`, cambiaba el paso temporal y el truncamiento numérico acumulativo.
- **Solución implementada**:
  - Se añadió en el motor (`runner.py`) y en la configuración (`SimulationConfig.morpho`) la opción de elegir entre:
    1. **Adaptive (`factime`)**: Paso variable original escalado por `factime`.
    2. **Fixed `dt`**: Paso temporal explícito y constante en segundos (`dt_fixed`, ej. 60 s, 300 s, etc.).
  - En la pestaña **4. Sediment & Morpho**, se agregó un selector dinámico para alternar entre ambos modos con inputs interactivos.

---

### Hito 4: Preservación de la Playa Seca y Duna (Subaerial Beach)
- **Commit**: `3f39de3`
- **Problema**: En la salida del modelo y en los gráficos, la parte seca de la playa por encima del nivel del agua se borraba o se ponía en cero.
- **Causa**:
  - El solucionador hidrodinámico solo calcula celdas mojadas ($x \le x_{wl}$).
  - En el módulo morfológico, al actualizar el lecho, las celdas secas ($x > x_{wl}$) no recibían transporte y el array de resultados quedaba truncado o reseteado, perdiendo la berma, duna y playa alta.
- **Solución**:
  - Se preservó el dominio espacial completo ($x \in [0, x_{max}]$) a lo largo de todo el ciclo de cálculo.
  - Las cotas del lecho $Z_b(x, t)$ para las estaciones subaéreas se mantienen estables respetando la geometría inicial más el arrastre por swash/runup (`BEACH.F`).
  - Los gráficos en pantalla y los archivos de exportación (Excel, CSV) reflejan el perfil topobatimétrico continuo de principio a fin.

---

### Hito 5: Gráfico Independiente de Concentración y Selector Dinámico (3 de 5)
- **Commit**: `e7c718e`
- **Requerimiento del usuario**:
  - Añadir un gráfico dedicado para la concentración de sedimentos ($m^3_{\text{sed}}/m^3_{\text{agua}}$) sin usar un eje secundario en el gráfico de transporte.
  - Dar la opción de elegir **3 gráficos de 4 (o 3 de 5)** para que las curvas no se vean comprimidas verticalmente.
- **Solución implementada**:
  - **Panel 4 dedicado (`p_conc`)**: Muestra la concentración de fondo $C_a$ y la concentración promediada en la vertical $\bar{C}$.
  - **Panel 3 (`p3`)**: Restaurado exclusivamente para tasas de transporte ($S_t, S_b, S_s, S_{t,long}$) en $\text{kg}/(\text{s}\cdot\text{m})$.
  - **Selector de Vistas Rápidas (Presets)**: Dropdown con 8 combinaciones preestablecidas de 3 paneles (ej. *Perfil + Hidrodinámica + Concentración*, *Perfil + Transporte + Concentración*, etc.) o 4 paneles.
  - **Casillas individuales de activación**: Permiten al usuario alternar libremente cualquiera de los 5 gráficos.
  - **Reconstrucción dinámica (`_rebuild_plot_layout`)**: Oculta los paneles no seleccionados y expande los elegidos ocupando cada uno 1/3 de la ventana vertical, manteniendo los ejes $X$ perfectamente vinculados.

---

### Hito 6: Geometría de Ventana Responsiva y Barra Compacta
- **Commit**: `1026f18`
- **Problema reportado**: *"la ventana es más grande que la pantalla... ¿Eso está bien?"*
- **Diagnóstico**:
  - La barra superior del dashboard tenía 12 controles alineados horizontalmente, requiriendo un ancho mínimo de **1293 px**.
  - Las 4 pestañas de parámetros tenían formularios sin barra de scroll que exigían **917 px** mínimos.
  - El divisor horizontal (`QSplitter`) sumaba ambos requerimientos, forzando un ancho mínimo rígido de **2222 px** (mayor que una pantalla 1080p estándar de 1920 px y que cualquier pantalla de laptop de 1366 px o con DPI scaling al 125%/150%).
- **Solución**:
  - **Barra de gráficos en 2 filas**:
    - Fila 1: Presets y selección de paneles visibles.
    - Fila 2: Capas activas (Initial Bed, Water Level, Transport Parts) y botón `⟲ Auto Range`.
  - **Scroll horizontal y vertical en pestañas**: Cada pestaña se envolvió en un `QScrollArea`, permitiendo que el panel izquierdo se encoja tanto como sea necesario.
  - **Adaptación automática a la pantalla**: `MainWindow` consulta `screen.availableGeometry()` y se ajusta automáticamente al **90% del ancho y 88% del alto útil**, centrada y respetando la barra de tareas.
  - **Resultado**: El ancho mínimo se redujo de **2222 px a solo 842 px**, funcionando de forma óptima en monitores desde 1024×768 hasta 4K.

---

## 4. Estado de Verificación y Pruebas

Toda la suite de pruebas automatizadas está pasando al 100%:
- **Total de pruebas**: **20 pruebas pasando** en **11.5 segundos**.
- **Golden Master Tests**:
  - `test_dgoot1_hydrodynamics_golden_master`: Coincidencia milimétrica con la salida histórica de Fortran `dgoot1.out`.
  - `test_atest1_wave_height_golden_master`: Coincidencia con `atest1.out`.
- **Comando para ejecutar pruebas**:
  ```bash
  cd crosmor_app
  python3 -m pytest tests/ -v
  ```

---

## 5. Instrucciones y Directrices para Futuros Agentes / Desarrolladores

1. **Modificación de Código**:
   - Todo código nuevo debe residir dentro de `crosmor_app/src/crosmor/`.
   - No reintroducir variables globales ni acoplamientos rígidos. Toda configuración se gestiona mediante `SimulationConfig` y los resultados mediante `SimulationResults`.
   - Las rutinas numéricas intensivas deben decorarse con `@njit(fastmath=True, cache=True)` de Numba o vectorizarse con NumPy.
2. **Interfaz Gráfica (PyQt6)**:
   - Los cálculos pesados **nunca** deben correr en el hilo de la GUI; deben pasar a través de `SimulationWorker(QThread)` emitiendo señales `step_finished` y `progress_updated`.
   - Mantener el diseño responsivo: evitar anchos fijos mayores a 300 px y asegurar que nuevos controles utilicen layouts verticales u horizontales adaptables.
3. **Flujo de Trabajo Git**:
   - Trabajar siempre en el repositorio `crosmor_app/`.
   - Ejecutar la suite completa de pruebas (`pytest tests/ -v`) antes de cualquier commit.
   - Utilizar mensajes de commit con convención semántica (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).
   - Realizar push a `origin main` tras verificar que las pruebas pasaron.
