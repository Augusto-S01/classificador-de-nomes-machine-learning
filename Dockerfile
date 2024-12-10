# Usar a imagem base do Miniconda
FROM continuumio/miniconda3:latest

# Definir o diretório de trabalho
WORKDIR /app

# Copiar o arquivo environment.yml
COPY environment.yml /app/

# Criar o ambiente Conda e ativá-lo
RUN conda env create -f environment.yml && \
    echo "conda activate te2" >> ~/.bashrc

# Tornar o ambiente Conda o padrão
ENV PATH /opt/conda/envs/te2/bin:$PATH

# Copiar o restante do código para o container
COPY . /app/

# Rodar collectstatic para coletar os arquivos estáticos
RUN python manage.py collectstatic --noinput

# Expor a porta padrão do Django
EXPOSE 8000

# Executar o servidor Django usando o ambiente Conda
CMD ["python", "manage.py", "runserver", "0.0.0.0:9992"]

