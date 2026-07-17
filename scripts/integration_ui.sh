#!/bin/bash
# Integration UI stack: gym-retro-integration (stable-retro Qt tool) on a
# virtual display, streamed to the dashboard's Integration tab via noVNC.
#
#   Xvfb :99 (1600x900) -> openbox -> gym-retro-integration
#   x11vnc :99 -> localhost:5901
#   websockify localhost:6080 (serves /usr/share/novnc + WS to 5901)
#   backend server.py proxies /integration/* -> 6080 (single-origin, tunnel-safe)
#
# Managed by user unit retro-integration-ui.service.

set -u
DISP=:99
GEOM=1600x900x24
UI_BIN="$HOME/stable-retro/gym-retro-integration"
VNC_PORT=5901
WEB_PORT=6080

cleanup() { pkill -P $$ 2>/dev/null; }
trap cleanup EXIT

echo "[integration-ui] starting stack on $DISP ($GEOM)"
Xvfb "$DISP" -screen 0 "$GEOM" -nolisten tcp &
sleep 1
DISPLAY=$DISP openbox &
x11vnc -display "$DISP" -localhost -forever -shared -nopw \
    -rfbport "$VNC_PORT" -quiet -noxdamage &
websockify --web /usr/share/novnc "127.0.0.1:$WEB_PORT" "127.0.0.1:$VNC_PORT" &

# cwd = the studio's game workspaces so Open dialogs land there and saved
# states/json flow straight into games/<id>/.
cd "$HOME/retro-dreamer/games"

# Respawn the UI if it's closed — the tab should never show an empty desktop.
while true; do
    echo "[integration-ui] launching $UI_BIN"
    (sleep 3; DISPLAY=$DISP wmctrl -r gym-retro-integration \
        -b add,maximized_vert,maximized_horz) &
    DISPLAY=$DISP "$UI_BIN"
    echo "[integration-ui] UI exited ($?), respawning in 2s"
    sleep 2
done
