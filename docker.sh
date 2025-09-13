#!/bin/bash

# AI Health & Fitness Agent Docker Management Script

set -e

case "$1" in
    build)
        echo "Building Docker image..."
        docker build -t ai-health-fitness-agent .
        echo "✅ Docker image built successfully!"
        ;;
    run)
        echo "Running AI Health & Fitness Agent..."
        docker run -e -d \
            --name ai-health-fitness-agent \
            -p 8501:8501 \
            ai-health-fitness-agent
        echo "✅ Application is running at http://localhost:8501"
        ;;
    compose-up)
        echo "Starting services with Docker Compose..."
        docker-compose up -d
        echo "✅ Services started! Application available at http://localhost:8501"
        ;;
    compose-down)
        echo "Stopping services..."
        docker-compose down
        echo "✅ Services stopped!"
        ;;
    stop)
        echo "Stopping container..."
        docker stop ai-health-fitness-agent
        docker rm ai-health-fitness-agent
        echo "✅ Container stopped and removed!"
        ;;
    logs)
        echo "Showing application logs..."
        docker logs -f ai-health-fitness-agent
        ;;
    shell)
        echo "Opening shell in container..."
        docker exec -it ai-health-fitness-agent /bin/bash
        ;;
    clean)
        echo "Cleaning up Docker resources..."
        docker stop ai-health-fitness-agent 2>/dev/null || true
        docker rm ai-health-fitness-agent 2>/dev/null || true
        docker rmi ai-health-fitness-agent 2>/dev/null || true
        echo "✅ Cleanup completed!"
        ;;
    *)
        echo "AI Health & Fitness Agent Docker Management"
        echo ""
        echo "Usage: $0 {build|run|compose-up|compose-down|stop|logs|shell|clean}"
        echo ""
        echo "Commands:"
        echo "  build        - Build the Docker image"
        echo "  run          - Run the application container"
        echo "  compose-up   - Start services using Docker Compose"
        echo "  compose-down - Stop services using Docker Compose"
        echo "  stop         - Stop and remove the container"
        echo "  logs         - Show application logs"
        echo "  shell        - Open shell in running container"
        echo "  clean        - Clean up all Docker resources"
        echo ""
        exit 1
        ;;
esac
