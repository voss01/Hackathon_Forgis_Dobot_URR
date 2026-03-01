#!/usr/bin/env bash
# Run the YOLO test inside the backend Docker container.
# Usage:
#   ./scripts/run_yolo_test.sh
#   CLASS_FILTER="cardboard box" CONF_THRESHOLD=0.5 ./scripts/run_yolo_test.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy latest version of the script into the container
docker cp "$SCRIPT_DIR/test_yolo.py" forgis-backend:/tmp/test_yolo.py

# Run it, forwarding any env vars the user set
docker exec \
  -e CLASS_FILTER="${CLASS_FILTER:-}" \
  -e CONF_THRESHOLD="${CONF_THRESHOLD:-0.3}" \
  -e OUTPUT_PATH="${OUTPUT_PATH:-/tmp/yolo_result.jpg}" \
  forgis-backend bash -c \
  "source /app/.venv/bin/activate && python3 /tmp/test_yolo.py"

# Copy result image back to host
docker cp forgis-backend:/tmp/yolo_result.jpg /tmp/yolo_result.jpg
echo ""
echo "Annotated image saved to /tmp/yolo_result.jpg"
xdg-open /tmp/yolo_result.jpg 2>/dev/null || true
