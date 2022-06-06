FROM python:3.8-buster

ENV PORT=8080
ENV GOOGLE_APPLICATION_CREDENTIALS=/service_accounts/mackathon-team-c-8ee597408a92.json



RUN python -m pip install --upgrade pip

RUN pip install cmake
RUN pip install face-recognition
RUN pip install poetry

COPY ./pyproject.toml .

# Install packages and dependencies on main python
RUN poetry config virtualenvs.create false --local \
    && poetry lock \
    && poetry install

EXPOSE $PORT

# Run a WSGI server to serve the application. gunicorn must be declared as
# a dependency in requirements.txt.
ADD . /
ENTRYPOINT gunicorn -b 0.0.0.0:$PORT main:app -t 3000000 -c gunicorn.conf.py
