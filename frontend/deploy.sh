#!/usr/bin/env bash
# graphq-browser/deploy.sh — One-command GraphQ deployment
set -e

# ── Config ──────────────────────────────────────────────────────────────────
GRAPHQ_DIR="${GRAPHQ_DIR:-$HOME/.hermes/skills/graphq-browser}"
VENV="${VENV:-$HOME/.hermes/venvs/docgraph/bin/python}"
PIPELINE_DIR="${PIPELINE_DIR:-$HOME/.hermes/skills/npmi-gee-lider}"
CREDS="${FIREBASE_CREDS:-$HOME/.config/firebase/service-account.json}"
GRAPHQ_PORT="${GRAPHQ_PORT:-8766}"

CF_TOKEN="${CLOUDFLARE_API_TOKEN:-YOUR_CLOUDFLARE_API_TOKEN}"
CF_ACCT="${CLOUDFLARE_ACCOUNT_ID:-YOUR_CLOUDFLARE_ACCOUNT_ID}"
WRANGLER="${WRANGLER:-/tmp/node_modules/.bin/wrangler}"

# ── Helpers ─────────────────────────────────────────────────────────────────
info()  { echo "ℹ️  $*" ; }
warn()  { echo "⚠️  $*" ; }
die()   { echo "❌ $*" ; exit 1 ; }

need()  { command -v "$1" >/dev/null 2>&1 || die "Need '$1' but not found" ; }

# ── Checks ───────────────────────────────────────────────────────────────────
need curl
need lsof
[[ -d "$GRAPHQ_DIR" ]] || die "GraphQ skill dir not found: $GRAPHQ_DIR"
[[ -f "$CREDS" ]]     || die "Service account not found: $CREDS"
[[ -x "$WRANGLER" ]]  || die "Wrangler not found: $WRANGLER"

# ── Step 1: Kill old server ───────────────────────────────────────────────────
info "Stopping any existing GraphQ server on port $GRAPHQ_PORT..."
kill $(lsof -ti:$GRAPHQ_PORT) 2>/dev/null && info "Killed old server" || info "No server running"

# ── Step 2: Start backend ────────────────────────────────────────────────────
info "Starting GraphQ FastAPI server on port $GRAPHQ_PORT..."
cd "$PIPELINE_DIR"
GRAPHQ_PORT=$GRAPHQ_PORT nohup $VENV app.py > /tmp/graphq_server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /tmp/graphq_server.pid
sleep 2

# Verify health
HEALTH=$(curl -sf http://localhost:$GRAPHQ_PORT/health 2>/dev/null) || {
    die "Server failed to start. Log: $(tail -20 /tmp/graphq_server.log)"
}
info "Backend ready: $HEALTH"

# ── Step 3: Start cloudflared tunnel ─────────────────────────────────────────
info "Starting cloudflared quick tunnel..."
pkill -f "cloudflared.*8766" 2>/dev/null || true

# Start cloudflared in background, capture URL from log
LOG="/tmp/cloudflared-graphq.log"
nohup /tmp/cloudflared tunnel --url http://localhost:$GRAPHQ_PORT > "$LOG" 2>&1 &
CLOUDFLARED_PID=$!
echo $CLOUDFLARED_PID > /tmp/cloudflared_graphq.pid

# Poll for URL (up to 20s)
TUNNEL_URL=""
for i in $(seq 1 20); do
    sleep 1
    TUNNEL_URL=$(grep -o 'https://[^ ]*\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1)
    [[ -n "$TUNNEL_URL" ]] && break
done

if [[ -z "$TUNNEL_URL" ]]; then
    die "Cloudflared tunnel failed to start. Log: $(tail -10 $LOG)"
fi
info "Tunnel ready: $TUNNEL_URL"

# ── Step 4: Update frontend API_BASE ───────────────────────────────────────
info "Updating API_BASE in app.js..."
TUNNEL_HOST=$(echo "$TUNNEL_URL" | sed 's/https:\/\///')
sed -i "s|const API_BASE = 'https://[^']*';|const API_BASE = 'https://$TUNNEL_HOST';|" \
    "$GRAPHQ_DIR/assets/app.js"

# ── Step 5: Build public/ ───────────────────────────────────────────────────
info "Building public/ directory..."
mkdir -p "$GRAPHQ_DIR/public"
cp "$GRAPHQ_DIR/assets/index.html" \
   "$GRAPHQ_DIR/assets/style.css"   \
   "$GRAPHQ_DIR/assets/app.js"     \
   "$GRAPHQ_DIR/public/"

# ── Step 6: Deploy to Cloudflare Pages ──────────────────────────────────────
info "Deploying to Cloudflare Pages..."
DEPLOY_URL=$(CLOUDFLARE_API_TOKEN="$CF_TOKEN" \
             CLOUDFLARE_ACCOUNT_ID="$CF_ACCT" \
             $WRANGLER pages deploy "$GRAPHQ_DIR/public" \
                 --project-name=graphq \
                 --commit-message="GraphQ deploy $(date -u +%Y%m%d-%H%M%S)" 2>&1 \
             | grep -o 'https://[^ ]*\.pages\.dev' | tail -1)

if [[ -z "$DEPLOY_URL" ]]; then
    die "Cloudflare Pages deploy failed."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅  GraphQ PDF Q&A — Deployed!"
echo "═══════════════════════════════════════════════════════"
echo "  Frontend:   $DEPLOY_URL"
echo "  API (tunnel): $TUNNEL_URL"
echo "  Backend:    localhost:$GRAPHQ_PORT (PID $SERVER_PID)"
echo ""
echo "  Tunnel PID: $CLOUDFLARED_PID (kills on server restart)"
echo "  Server PID: $SERVER_PID (kills on: kill \$(lsof -ti:$GRAPHQ_PORT))"
echo ""
echo "⚠️  Quick tunnel URL changes on restart."
echo "    For permanent URL: set up a named Cloudflare Tunnel."
echo "═══════════════════════════════════════════════════════"
