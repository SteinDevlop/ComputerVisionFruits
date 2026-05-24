# FruitVision AI

FruitVision AI es un sistema avanzado de visión por computadora que permite detectar y clasificar frutas en tiempo real. Está diseñado con una arquitectura moderna de microservicios, optimizada para velocidad y precisión mediante el uso de inteligencia artificial de vanguardia.

## 🚀 Arquitectura

El sistema utiliza un **Pipeline de 1 Etapa (One-Stage Detection)**:
- **Modelo Base:** `YOLOv8` (You Only Look Once), entrenado de manera personalizada con un dataset de alta calidad.
- **Detección Unificada:** YOLOv8 se encarga simultáneamente de trazar la caja delimitadora (bounding box) alrededor de la fruta y clasificar exactamente qué fruta es entre 15 clases diferentes.
- **Backend (Gateway):** Construido en **FastAPI**, sirve el modelo de IA y expone Endpoints REST y WebSockets para comunicación en tiempo real.
- **Frontend:** Construido en **React + Vite**, provee una interfaz de usuario interactiva y fluida para subir videos, usar la cámara en tiempo real o explorar el historial de detecciones.

### Clases Soportadas
El modelo es capaz de reconocer las siguientes frutas:
`cucumber, apple, kiwi, banana, orange, coconut, peach, cherry, pear, pomegranate, pineapple, watermelon, melon, grape, strawberry`

## 🛠️ Tecnologías

- **IA / Visión:** Python 3.11, PyTorch, Ultralytics (YOLOv8), OpenCV.
- **Backend:** FastAPI, Uvicorn, asyncio (procesamiento asíncrono).
- **Frontend:** React, Vite, TailwindCSS (opcional/estilos globales).
- **Despliegue:** Docker, Docker Compose, Nginx.

## 📦 Instalación y Despliegue

La forma más fácil y recomendada de desplegar todo el sistema es usando **Docker**.

### 1. Clonar el repositorio
```bash
git clone <tu-repositorio>
cd ComputerVisionFruits
```

### 2. Levantar los contenedores (Docker Compose)
El proyecto incluye un entorno Docker preconfigurado que levantará tanto el frontend como el backend de manera orquestada.

```bash
cd deployment
docker compose up -d --build
```

- La **Interfaz Web** estará disponible en: `http://localhost:3000`
- El **Backend API** estará corriendo en: `http://localhost:8080`

## 💻 Desarrollo Local (Sin Docker)

Si deseas trabajar en el código o re-entrenar el modelo, necesitas configurar el entorno local.

### 1. Entorno de Python
```bash
# Crear entorno virtual
python -m venv frutas_venv

# Activar entorno (Windows)
frutas_venv\Scripts\activate

# Instalar dependencias del backend
pip install -r src/frontend/backend/requirements.txt
```

### 2. Iniciar el Backend Manualmente
```bash
cd src
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Iniciar el Frontend Manualmente
```bash
cd src/frontend
npm install
npm run dev
```

## 🧠 Re-Entrenamiento del Modelo

Si deseas agregar más frutas o mejorar la precisión con un nuevo dataset:
1. Coloca tu dataset estructurado en YOLO en la carpeta `Fruits/` en la raíz del proyecto.
2. Abre y ejecuta el Jupyter Notebook provisto: `entrenamiento_yolov8_nuevo.ipynb`.
3. El notebook generará tu archivo `data.yaml` y entrenará el modelo.
4. Mueve el archivo resultante `best.pt` a la ruta final: `src/backend/models/finetuned/best.pt`.
5. Reinicia el contenedor del backend para cargar el nuevo modelo.

## 📄 Estructura de Directorios Clave

```text
📦ComputerVisionFruits
 ┣ 📂deployment            # Infraestructura y despliegue
 ┃ ┣ 📜docker-compose.yml  # Orquestador de contenedores
 ┃ ┣ 📜Dockerfile.gateway  # Imagen del backend
 ┃ ┗ 📜...
 ┣ 📂src                   # Código fuente principal
 ┃ ┣ 📂backend
 ┃ ┃ ┣ 📂database          # Base de datos (precios.json)
 ┃ ┃ ┣ 📂detector_agent    # Core de IA (detector.py, tracker.py)
 ┃ ┃ ┣ 📂models            # Pesos del modelo entrenado (finetuned/best.pt)
 ┃ ┃ ┗ 📂shared            # Configuraciones globales (config.py)
 ┃ ┗ 📂frontend
 ┃ ┃ ┣ 📂backend           # Gateway en FastAPI (main.py, processor.py)
 ┃ ┃ ┣ 📂src               # Código de React (App.jsx, componentes)
 ┃ ┃ ┣ 📜Dockerfile        # Imagen del frontend
 ┃ ┃ ┗ 📜package.json      
 ┣ 📜entrenamiento_yolov8_nuevo.ipynb  # Notebook para re-entrenar IA
 ┣ 📜.gitignore
 ┗ 📜README.md
```

---
*Este proyecto es parte de la asignatura Inteligencia Artificial - 7mo Semestre.*
