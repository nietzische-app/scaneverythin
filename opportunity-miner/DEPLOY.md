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

Compose binds the app to `127.0.0.1:8000` only and forces `DATABASE_PATH=/app/data/radar.sqlite`, so the file stays on the host at `opportunity-miner/data/radar.sqlite`. Do not open port 8000 in the Hetzner firewall.

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
ExecStart=/opt/scaneverythin/opportunity-miner/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now opportunity-radar
```

## HTTPS

Point an A record at the VPS, then publish only 22, 80, and 443 in the Hetzner firewall. Caddy fetches the certificate. Replace `radar.senindomainin.com` with that name.

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy

sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
radar.senindomainin.com {
    reverse_proxy localhost:8000
}
EOF
sudo systemctl reload caddy
```

The dashboard is `https://radar.senindomainin.com/`.

Nginx does the same job if you already run it. Certbot writes the TLS server block.

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
sudo tee /etc/nginx/sites-available/radar >/dev/null <<'EOF'
server {
    listen 80;
    server_name radar.senindomainin.com;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/radar /etc/nginx/sites-enabled/radar
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d radar.senindomainin.com
```
