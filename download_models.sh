#!/bin/bash
# Download MediaPipe models (Face Landmarker + Hand Landmarker)

MODEL_DIR="models"
FACE_MODEL="face_landmarker.task"
HAND_MODEL="hand_landmarker.task"
FACE_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

echo "============================================"
echo "MediaPipe Models Downloader"
echo "============================================"

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Function to download model
download_model() {
    local MODEL_FILE=$1
    local MODEL_URL=$2
    local MODEL_NAME=$3

    echo ""
    echo "[$MODEL_NAME]"

    if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
        echo "  ✓ Already exists: $MODEL_DIR/$MODEL_FILE"
        echo "  Size: $(du -h "$MODEL_DIR/$MODEL_FILE" | cut -f1)"
        return 0
    fi

    echo "  Downloading $MODEL_NAME..."
    wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL" -q --show-progress

    if [ $? -eq 0 ]; then
        echo "  ✓ Downloaded successfully"
        echo "  Size: $(du -h "$MODEL_DIR/$MODEL_FILE" | cut -f1)"
        return 0
    else
        echo "  ✗ Failed to download"
        return 1
    fi
}

# Download models
download_model "$FACE_MODEL" "$FACE_URL" "Face Landmarker"
FACE_STATUS=$?

download_model "$HAND_MODEL" "$HAND_URL" "Hand Landmarker"
HAND_STATUS=$?

# Summary
echo ""
echo "============================================"
echo "Download Summary:"
echo "============================================"

if [ $FACE_STATUS -eq 0 ]; then
    echo "  ✓ Face Landmarker: OK"
else
    echo "  ✗ Face Landmarker: FAILED"
fi

if [ $HAND_STATUS -eq 0 ]; then
    echo "  ✓ Hand Landmarker: OK"
else
    echo "  ✗ Hand Landmarker: FAILED"
fi

echo ""

if [ $FACE_STATUS -eq 0 ] && [ $HAND_STATUS -eq 0 ]; then
    echo "✓ All models downloaded successfully!"
    exit 0
else
    echo "✗ Some models failed to download"
    exit 1
fi
