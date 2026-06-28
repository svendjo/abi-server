# Balut Eye — AWS deployment guide

How to deploy **abi-server** (backend) and **abi-www** (frontend) to AWS, mirroring
the `c6-server` / `c6-www` (Count Chocolate) setup on the same account. AWS resources
use the project name **`balut-*`** (just like c6 uses `chocolate-*`).

| | Service | Where |
|---|---|---|
| Backend (`abi-server`) | **ECR** → **App Runner** | region `us-west-2` |
| Frontend (`abi-www`) | **S3** (static website) → **CloudFront** | bucket in `us-west-2`, CloudFront global |

Constants for this account:
- AWS account: `021891586863`
- ECR registry: `021891586863.dkr.ecr.us-west-2.amazonaws.com`
- Names: ECR repo `balut-repository`, App Runner service `balut-backend`,
  S3 bucket `balut-frontend`, local image `balut-docker`.

> **Order matters:** deploy the **backend first**, get its App Runner URL, then put
> that URL into the frontend before building and uploading it.

(A custom domain via Route 53 + ACM is deliberately left out for now — the site
lives at the `*.cloudfront.net` URL, same as c6.)

---

## Part A — Backend → ECR + App Runner

All commands run from `abi-server/`.

### A1. Build the image (and test locally)
```sh
docker build -t balut-docker:latest .
docker run -p 8080:8080 balut-docker:latest        # in another shell:
curl -X POST http://localhost:8080/read -F "file=@../abi-dataset/images/1.jpg"
```
> The Dockerfile installs `requirements.txt` (the light **tflite-runtime** path, not
> full TensorFlow) and copies `server.py`, `scorecard.py`, `trocr_reader.py`,
> `ctc_reader.py`, and `digit-model0622.tflite`. If you change `MODEL_PATH`, update
> the Dockerfile's `COPY` to match.
>
> You're on an Intel Mac, so the image is already `linux/amd64` (what App Runner
> needs). On Apple Silicon you'd have to add `--platform linux/amd64` to the build.

### A2. Create the ECR repository (one time)
```sh
aws ecr create-repository --repository-name balut-repository --region us-west-2
```

### A3. Tag & push
```sh
aws ecr get-login-password --region us-west-2 \
  | docker login --username AWS --password-stdin 021891586863.dkr.ecr.us-west-2.amazonaws.com

docker tag balut-docker:latest 021891586863.dkr.ecr.us-west-2.amazonaws.com/balut-repository:latest
docker push 021891586863.dkr.ecr.us-west-2.amazonaws.com/balut-repository:latest
```

### A4. Create the App Runner service (console, one time)
AWS Console → **App Runner** → **Create service** (region us-west-2):
- **Source**: Container registry → Amazon ECR → image `balut-repository:latest`.
- **Deployment trigger**: Manual (matches chocolate; flip to Automatic if you want
  every push to redeploy). App Runner offers to **create the ECR access role** —
  accept it (chocolate's is `ChocolateAppRunnerECRAccessRole`; this one can be
  `BalutAppRunnerECRAccessRole`).
- **Port**: `8080`.
- **CPU / memory**: **1 vCPU / 2 GB** recommended. (chocolate runs 0.25/0.5, but
  abi does OpenCV image processing + a TFLite model per request, so give it more.)
- **Health check**: TCP (default) is fine — abi-server has no `GET /` route.
- Create, then copy the **Default domain** → `https://<id>.us-west-2.awsapprunner.com`.
  That is your backend URL; you'll need it for the frontend.

### A5. Redeploy later
Rebuild → tag → push (A1, A3), then (Manual trigger) press **Deploy** in the console,
or:
```sh
aws apprunner start-deployment --region us-west-2 \
  --service-arn <balut-backend-service-arn>
```

### A6. Environments & results storage (S3)

abi-server selects an environment from `config/<APP_ENV>.yaml` (`config.py`); the YAML
carries the run flags (`use_ctc`, `debug_crops`, `reload`) and the results backend, so
the run command stays just `python server.py`.

| `APP_ENV`              | results backend                  | set by |
|------------------------|----------------------------------|--------|
| `local-dev` (default)  | `results/<id>/` on local disk    | nothing — it's the default |
| `aws-prod`             | `s3://balut-results/<id>/`        | `ENV APP_ENV=aws-prod` in the Dockerfile |

S3 is required in prod: App Runner's local disk is **wiped on every restart/redeploy**
and **not shared across instances**, so `/accept`, `/decline` and `/feedback` (which
arrive as separate requests) would otherwise not find the read. Writing to S3 makes the
input images, verdicts and corrected-scorecard ground truth durable and shared.

**One-time AWS setup (already provisioned):**
- S3 bucket **`balut-results`** (us-west-2, all public access blocked).
- IAM role **`BalutAppRunnerInstanceRole`** — inline policy `BalutResultsS3`:
  `s3:PutObject`/`s3:GetObject` on `arn:aws:s3:::balut-results/*` and `s3:ListBucket`
  on the bucket.

**Attach the instance role to the service** (one time; this is a *different* role from
the ECR access role in A4). Console → App Runner → `balut-backend` → **Configuration →
Edit → Security → Instance role** → `BalutAppRunnerInstanceRole` → Save (redeploys).
The console is recommended over `aws apprunner update-service` so you don't have to
re-specify CPU/memory. **Until the instance role is attached, prod reads will 500 on the
S3 write** (the container has no AWS credentials).

---

## Part B — Frontend → S3 + CloudFront

All commands run from `abi-www/`.

### B1. Point the frontend at the backend (do this first!)
Edit `src/App.js` and change the four `*_URL` constants from `http://localhost:8080`
to your App Runner URL from A4:
```js
const READ_URL    = 'https://<id>.us-west-2.awsapprunner.com/read';
const ACCEPT_URL  = 'https://<id>.us-west-2.awsapprunner.com/accept';
const DECLINE_URL = 'https://<id>.us-west-2.awsapprunner.com/decline';
const SUBMIT_URL  = 'https://<id>.us-west-2.awsapprunner.com/feedback';
```
(CORS is already open for POST on the server, so no extra config is needed.)

### B2. Build
```sh
npm install        # first time only
npm run build
```

### B3. Create + configure the S3 bucket (one time)
```sh
aws s3 mb s3://balut-frontend --region us-west-2
aws s3 website s3://balut-frontend --index-document index.html --error-document index.html
# allow a public-read bucket policy
aws s3api put-public-access-block --bucket balut-frontend \
  --public-access-block-configuration \
  BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false
aws s3api put-bucket-policy --bucket balut-frontend --policy '{
  "Version":"2012-10-17",
  "Statement":[{"Sid":"PublicReadGetObject","Effect":"Allow","Principal":"*",
    "Action":"s3:GetObject","Resource":"arn:aws:s3:::balut-frontend/*"}]
}'
```

### B4. Upload the build
```sh
aws s3 sync ./build s3://balut-frontend
```

### B5. Create the CloudFront distribution (console, one time)
AWS Console → **CloudFront** → **Create distribution**:
- **Origin domain**: the bucket's **website endpoint** —
  `balut-frontend.s3-website-us-west-2.amazonaws.com` (type it in; do **not** pick the
  REST `balut-frontend.s3.amazonaws.com` suggestion). CloudFront treats it as a custom
  origin, **HTTP only** — that's correct for an S3 website endpoint.
- **Viewer protocol policy**: Redirect HTTP to HTTPS.
- **Default root object**: leave blank (the S3 website serves `index.html`).
- Create, then note the **Distribution domain name** `https://<dist>.cloudfront.net`
  — the site is live there.

### B6. Update later
```sh
npm run build
aws s3 sync ./build s3://balut-frontend --delete
aws cloudfront create-invalidation --distribution-id <DISTRIBUTION_ID> --paths "/*"
```

---

## Quick recap
1. **Backend**: `docker build` → push to `balut-repository` (ECR) → App Runner `balut-backend` → copy its URL.
2. **Frontend**: set the four `*_URL`s in `App.js` to that URL → `npm run build` → `s3 sync` to `balut-frontend` → CloudFront in front of the S3 *website* endpoint.
