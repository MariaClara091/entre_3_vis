FROM python:3.9-slim

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos
COPY . .

# Exponer puerto
EXPOSE 8050

# Comando para ejecutar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app_proyect_1:server"]