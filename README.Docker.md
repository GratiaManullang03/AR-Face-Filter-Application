# Docker Setup Guide

This guide helps you run the AR Face Filter application using Docker, making it easy to test without manual dependency installation.

## Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (usually comes with Docker Desktop)
- Webcam connected
- Linux with X11 (for GUI display)

## Quick Start

### 1. Allow X11 Display Access

On Linux, allow Docker to access your X11 display:

```bash
xhost +local:docker
```

### 2. Build and Run

```bash
# Build the Docker image
docker-compose build

# Run the application
docker-compose up
```

That's it! The application will:
- Automatically download MediaPipe models during build (~11MB total)
- Access your webcam via `/dev/video0`
- Display the GUI window on your screen
- Save screenshots to `./captures/` directory

### 3. Stop the Application

Press `Q` in the application window, or:

```bash
# Stop the container
docker-compose down
```

## What Docker Does

The Dockerfile automatically:
1. Installs all system dependencies (OpenCV, GUI libraries, etc.)
2. Installs Python dependencies from `requirements.txt`
3. Downloads Face Landmarker model (~3.6MB)
4. Downloads Hand Landmarker model (~7.6MB)
5. Sets up directories for captures and MediaPipe cache

## Volume Mounts

The `docker-compose.yml` mounts these directories:

- `./assets` → Container's `/app/assets` (filters and textures)
- `./captures` → Container's `/app/captures` (screenshots persist on host)
- `./src` → Container's `/app/src` (hot reload for development)

## Troubleshooting

### No Display Window

**Problem**: Application runs but no window appears.

**Solution**:
```bash
# Allow X11 access
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY

# If empty, set it
export DISPLAY=:0
```

### Camera Not Found

**Problem**: Error message "Camera not available".

**Solution**:
```bash
# Check if webcam is /dev/video0
ls -l /dev/video*

# If using different device (e.g., /dev/video1), edit docker-compose.yml:
devices:
  - /dev/video1:/dev/video0
```

### Permission Denied

**Problem**: Camera access denied.

**Solution**: The `docker-compose.yml` already includes `privileged: true` for camera access. If still failing:

```bash
# Add your user to video group
sudo usermod -aG video $USER

# Logout and login again
```

### Build Fails - Model Download

**Problem**: Model download fails during build.

**Solution**:
```bash
# Check internet connection
ping google.com

# Rebuild without cache
docker-compose build --no-cache

# Or download models manually on host, then build
./download_models.sh
```

## Windows Support

Docker GUI display on Windows requires additional setup:

1. Install VcXsrv or Xming (X11 server for Windows)
2. Start X server with "Disable access control" option
3. Set DISPLAY environment variable:
   ```bash
   export DISPLAY=host.docker.internal:0
   ```
4. Update `docker-compose.yml`:
   ```yaml
   environment:
     DISPLAY: host.docker.internal:0
   ```

## macOS Support

Docker GUI on macOS requires XQuartz:

1. Install XQuartz: `brew install --cask xquartz`
2. Start XQuartz
3. In XQuartz preferences, enable "Allow connections from network clients"
4. Allow connections:
   ```bash
   xhost + 127.0.0.1
   ```
5. Update `docker-compose.yml`:
   ```yaml
   environment:
     DISPLAY: host.docker.internal:0
   ```

## Development Mode

The `docker-compose.yml` mounts `./src` for hot reload:

1. Edit code in `./src/` on your host machine
2. Restart container to see changes:
   ```bash
   docker-compose restart
   ```

## Production Build

For production (without source code mounting):

```bash
# Build production image
docker build -t ar-face-filter:prod .

# Run without volume mounts
docker run --rm -it \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v $(pwd)/captures:/app/captures \
  --privileged \
  ar-face-filter:prod
```

## Clean Up

Remove Docker images and containers:

```bash
# Stop and remove containers
docker-compose down

# Remove image
docker rmi ar-face-filter:latest

# Remove all unused Docker data
docker system prune -a
```

## Notes

- First build takes ~5-10 minutes (downloading dependencies + models)
- Subsequent builds are faster due to Docker layer caching
- Models are downloaded during build, not at runtime
- Screenshots are saved to host's `./captures/` directory
- Container runs with `privileged: true` for webcam access (security consideration)

## Support

For issues specific to:
- **Application features**: See main [README.md](README.md)
- **Docker setup**: Check this guide's Troubleshooting section
- **MediaPipe models**: See [download_models.sh](download_models.sh)
