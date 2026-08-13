# Deploying

Two services from one repository: the Python API and the built frontend. The
frontend forwards `/api/*` to the API, so the browser only ever talks to one
origin and CORS never enters the picture.

## Before the first deploy

**1. Pin the dependencies on the interpreter you actually deploy on.**

```
source .venv/bin/activate
python scripts/check_env.py
```

Paste the pins it prints into `api/requirements.txt`, replacing the unpinned
names. Pins resolved against a different Python version than the host runs is
the single most reliable way to lose an afternoon to a missing wheel.

**2. Check the Python version matches.**

`.python-version` and the `PYTHON_VERSION` in `render.yaml` must agree, and
both must match what you tested on. `check_env.py` fails loudly if they do not.

**3. Confirm the dataset is committed.**

```
git ls-files data/public/monthly_returns.csv
```

The API loads it once at startup and refuses to serve without it. It is
committed deliberately: the deployed app never fetches market data, which
removes the largest source of deployment fragility -- the price source
rate-limits cloud hosts, so a fetch that works on a laptop can fail once
deployed.

**4. Run the tests.**

```
python -m pytest
```

## Deploying

Push to GitHub, then in Render: **New → Blueprint**, point it at the
repository. It reads `render.yaml` and creates both services.

The API name in `render.yaml` and the rewrite destination must match. If you
rename the service, update the `destination` under the static site's routes to
the same hostname.

## After it is live

Check in this order, because each one rules out a different failure:

```
curl https://saa-lab-api.onrender.com/api/health
```
The API alone. A failure here means the Python service did not start -- read
the logs, which are unbuffered.

```
curl https://saa-lab-web.onrender.com/api/health
```
The rewrite. If this returns HTML instead of JSON, the `/api/*` route is
being caught by the single-page fallback: the API rule must come first.

Then open the site itself.

## Known characteristics of the free plan

**The service sleeps after inactivity.** The first request after idling takes
30 to 60 seconds while the container starts. Nothing is wrong; it is the plan.
Worth mentioning next to any link you share, so a reviewer does not read a slow
first load as a broken site.

**Memory is limited.** The dataset is small and held once, but a solve at
20,000 samples allocates a matrix of every sampled portfolio's return series.
The interface defaults to 8,000, which is comfortable.
