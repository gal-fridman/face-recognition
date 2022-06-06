
# Create cloud run service
```shell
docker build -t gcr.io/mackathon-team-c/face .
docker push gcr.io/mackathon-team-c/face:latest 
gcloud run deploy face-recognition --image gcr.io/mackathon-team-c/face
```


# Setup
Open permissions for gcloud
```shell
gcloud auth application-default login
gcloud auth configure-docker
```

#Etc
## Build locally & deploy using yaml script (fails)
1. Build locally (debug)
```shell
docker build -t face:latest .
```
2. build on google cloud (build --> push --> deploy) (docker-compose.yml)
```shell
gcloud builds submit --region=us-central1 --timeout=9999 
```