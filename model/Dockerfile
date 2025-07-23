FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV JUPYTER_ENABLE_LAB=yes

COPY src/ ./src/

# Expose the port Jupyter will run on
EXPOSE 8888

# Command to run Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "src/model.ipynb", "--port=8888", "--no-browser", "--allow-root", "--ip=0.0.0.0", "start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]