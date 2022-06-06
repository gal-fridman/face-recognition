
# Create cloud run service
```shell
docker build -t gcr.io/mackathon-team-c/face .
docker push gcr.io/mackathon-team-c/face:latest 
gcloud run deploy face-recognition --image gcr.io/mackathon-team-c/face
```


# Setup
1. Get data files and place them in root folder (google cloud storage): 
[data files](https://console.cloud.google.com/storage/browser/mackathon-team-c.appspot.com/face-recognition-python-data)
2. Setup dervice account:\
Service account: `mackathon-team-c@appspot.gserviceaccount.com`\
[Credentials](https://console.cloud.google.com/iam-admin/serviceaccounts/details/106320937501857696450/keys?orgonly=true&project=mackathon-team-c&supportedpurview=organizationId)
3. Open permissions for gcloud\
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
