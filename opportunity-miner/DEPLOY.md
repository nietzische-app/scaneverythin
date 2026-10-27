# Deploy on a Hetzner VPS

The SQLite file is `data/radar.sqlite` (plus the `-wal` and `-shm` files beside it). Docker Compose bind-mounts that directory to `/app/data`.

## Docker Compose

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# log out and back in so the docker group applies

git clone https://github.com/nietzische-app/scaneverythin.git
cd scaneverythin/opportunity-miner
cp .env.example .env
# set GEMINI_API_KEY, DEEPSEEK_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

docker compose up -d --build
```

The dashboard is `http://SERVER:8000/`. Allow TCP 8000 in the Hetzner firewall. Compose forces `DATABASE_PATH=/app/data/radar.sqlite`, so the file stays on the host at `opportunity-miner/data/radar.sqlite`.

## systemd

```bash
sudo apt update && sudo apt install -y git python3.11 python3.11-venv
sudo git clone https://github.com/nietzische-app/scaneverythin.git /opt/scaneverythin
cd /opt/scaneverythin/opportunity-miner
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# set the same keys as above; DATABASE_PATH=data/radar.sqlite is already in the example

sudo tee /etc/systemd/system/opportunity-radar.service >/dev/null <<'EOF'
[Unit]
Description=Opportunity Radar
After=network-online.target

[Service]
WorkingDirectory=/opt/scaneverythin/opportunity-miner
EnvironmentFile=/opt/scaneverythin/opportunity-miner/.env
ExecStart=/opt/scaneverythin/opportunity-miner/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now opportunity-radar
```
